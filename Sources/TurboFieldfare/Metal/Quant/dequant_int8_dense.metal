#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int8_dense — MLX `affine` 8-bit dense decode kernels: embedding
// lookup, fused Q/K/V projection and the greedy LM head.
//
// 8-bit twins of `embed_lookup_int4`, `dequant_int4_qkv_gemv_simd` and
// `lm_head_greedy_int4_rows_chunk_raw`: same buffer bindings, dispatch
// geometry and output contracts. The greedy head's second pass reuses
// `lm_head_greedy_int4_rows_reduce` from `logit.metal`, which only reads the
// per-threadgroup (value, index) summaries and does not depend on the weight
// width.
//
// Owner: S2 of docs/plans/8bit-model-variant-plan.md. This module is
// concatenated after the 4-bit modules and reuses `kInt8GroupSize`
// (`dequant_int8.metal`) and the `kLMHead*` / `kLogitMaxSimdGroups` constants
// (`logit.metal`). New function constants use indices 100-109.
//
// Layout per weight row of length N (see `dequant_int8.metal`):
//   q       : N unsigned bytes; byte k is column k, range 0..255.
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[k] = float(q[k]) * scale[k/64] + bias[k/64].
//
// Alignment: the kernels assume only a 4-byte-aligned weight base (the
// installer guarantees 16 bytes for `affine8`; the margin is deliberate).
// Every row is N bytes with N % 64 == 0, so each row base keeps the 4-byte
// alignment and the row walk may read weights through `uint` loads. Scales and
// biases are read as BF16 and need only their natural 2-byte alignment.
// ============================================================================

constant uint FC_INT8_HEAD_D      [[function_constant(100)]];
constant uint FC_INT8_HEAD_V      [[function_constant(101)]];
constant bool FC_INT8_HEAD_USE_FC [[function_constant(102)]];
constant uint FC_INT8_QKV_MQ      [[function_constant(103)]];
constant uint FC_INT8_QKV_MKV     [[function_constant(104)]];
constant uint FC_INT8_QKV_N       [[function_constant(105)]];
constant bool FC_INT8_QKV_USE_FC  [[function_constant(106)]];

static inline uint int8_head_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_INT8_HEAD_USE_FC) &&
            FC_INT8_HEAD_USE_FC &&
            is_function_constant_defined(FC_INT8_HEAD_D)) ? FC_INT8_HEAD_D : D;
}

static inline uint int8_head_fc_v(constant uint& V) {
    return (is_function_constant_defined(FC_INT8_HEAD_USE_FC) &&
            FC_INT8_HEAD_USE_FC &&
            is_function_constant_defined(FC_INT8_HEAD_V)) ? FC_INT8_HEAD_V : V;
}

static inline uint int8_qkv_fc_mq(constant uint& Mq) {
    return (is_function_constant_defined(FC_INT8_QKV_USE_FC) &&
            FC_INT8_QKV_USE_FC &&
            is_function_constant_defined(FC_INT8_QKV_MQ)) ? FC_INT8_QKV_MQ : Mq;
}

static inline uint int8_qkv_fc_mkv(constant uint& Mkv) {
    return (is_function_constant_defined(FC_INT8_QKV_USE_FC) &&
            FC_INT8_QKV_USE_FC &&
            is_function_constant_defined(FC_INT8_QKV_MKV)) ? FC_INT8_QKV_MKV : Mkv;
}

static inline uint int8_qkv_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT8_QKV_USE_FC) &&
            FC_INT8_QKV_USE_FC &&
            is_function_constant_defined(FC_INT8_QKV_N)) ? FC_INT8_QKV_N : N;
}


kernel void embed_lookup_int8(
    device const uint8_t* table     [[buffer(0)]],   // [V, D] bytes
    device const bfloat*  scales    [[buffer(1)]],   // [V, D/64] BF16
    device const bfloat*  biases    [[buffer(2)]],   // [V, D/64] BF16
    device half*          out       [[buffer(3)]],   // [D] FP16
    constant uint&        token_id  [[buffer(4)]],
    constant uint&        D         [[buffer(5)]],
    constant float&       out_scale [[buffer(6)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups_per_row = D / kInt8GroupSize;
    device const uint8_t* row_q = table  + uint(token_id) * D;
    device const bfloat*  row_s = scales + uint(token_id) * groups_per_row;
    device const bfloat*  row_b = biases + uint(token_id) * groups_per_row;
    const float q = float(uint(row_q[gid]));
    const float s = float(row_s[gid / kInt8GroupSize]);
    const float b = float(row_b[gid / kInt8GroupSize]);
    out[gid] = half((q * s + b) * out_scale);
}

// One SIMD group computes W_row · x; every lane returns the full sum.
//
// Full 4-group (256-byte) blocks: the 32 lanes split 8 per group, each lane
// reading 8 contiguous weight bytes as two `uint` loads and the matching 8
// inputs as two `half4` loads. Scale and bias are constant inside a group, so
// the affine factoring s·Σqx + b·Σx holds per lane and `simd_sum` aggregates.
// The x reads land on 16-byte boundaries (lane*8 elements). N=2816/4096/8192
// give 44/64/128 groups, all exact 4-blocks; the remainder loop (for example
// N=2112, 33 groups) covers one group per iteration with 2 bytes per lane.
static inline float int8_dense_row_dot_simd(
    device const uint8_t* W_row,
    device const bfloat*  s_row,
    device const bfloat*  b_row,
    device const half*    x,
    uint                  N,
    uint                  lane
) {
    const uint n_groups = N / kInt8GroupSize;
    const uint full_blocks = n_groups / 4u;
    float acc = 0.0f;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint base = blk * 256u + lane * 8u;
        device const uint* wp = (device const uint*)(W_row + base);
        const float4 qa = float4(as_type<uchar4>(wp[0]));
        const float4 qb = float4(as_type<uchar4>(wp[1]));
        const float4 xa = float4(*((device const half4*)(x + base)));
        const float4 xb = float4(*((device const half4*)(x + base + 4u)));
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const float qx = dot(qa, xa) + dot(qb, xb);
        const float sx = (xa.x + xa.y + xa.z + xa.w) + (xb.x + xb.y + xb.z + xb.w);
        acc = fma(s, qx, acc);
        acc = fma(b, sx, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const uint i0 = g * kInt8GroupSize + lane * 2u;
        const float x0 = float(x[i0]);
        const float x1 = float(x[i0 + 1u]);
        const float qx = fma(float(uint(W_row[i0])), x0,
                             float(uint(W_row[i0 + 1u])) * x1);
        acc = fma(float(s_row[g]), qx, acc);
        acc = fma(float(b_row[g]), x0 + x1, acc);
    }
    return simd_sum(acc);
}

// y[row] = W[row, :] · x for one row per SIMD group. Rows past M are skipped
// by the whole SIMD group, so the reduction never runs with missing lanes.
static inline void int8_dense_gemv_row_simd(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    uint                  row,
    uint                  M,
    uint                  N,
    uint                  lane
) {
    if (row >= M) return;
    const uint n_groups = N / kInt8GroupSize;
    const float acc = int8_dense_row_dot_simd(W      + row * N,
                                              scales + row * n_groups,
                                              biases + row * n_groups,
                                              x, N, lane);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// Fused Q/K/V decode projection. Same bindings and geometry as
// `dequant_int4_qkv_gemv_simd`: eight rows per 256-thread threadgroup over
// Mq + 2*Mkv rows, Q rows first, then K, then V. Full-attention layers have no
// v_proj; the caller binds the K tensor again as V (as for the 4-bit kernel).
kernel void dequant_int8_qkv_gemv_simd(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int8_qkv_fc_mq(Mq);
    const uint KK = int8_qkv_fc_mkv(Mkv);
    const uint NN = int8_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row;
        M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ;
        M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK;
        M = KK;
    }
    int8_dense_gemv_row_simd(W, scales, biases, x, y, local_row, M, NN, lane);
}

// Greedy LM head, first pass. Eight SIMD groups each evaluate one INT8 row of
// the tied embedding table against the normed hidden state and write one
// (max value, row index) summary per threadgroup; non-finite logits never win.
// `lm_head_greedy_int4_rows_reduce` turns the summaries into the token ID,
// with ties resolved to the lowest index and index 0 when nothing is finite.
[[kernel, max_total_threads_per_threadgroup(256)]]
void lm_head_greedy_int8_rows_chunk_raw(
    device const half*    x_normed     [[buffer(0)]],
    device const uint8_t* W            [[buffer(1)]],
    device const bfloat*  scales       [[buffer(2)]],
    device const bfloat*  biases       [[buffer(3)]],
    device       float*   summaries    [[buffer(4)]],
    constant     uint&    D            [[buffer(5)]],
    constant     uint&    V            [[buffer(6)]],
    uint  tg_idx         [[threadgroup_position_in_grid]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]],
    uint  simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint  simdgroups     [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial_v[kLogitMaxSimdGroups];
    threadgroup uint  partial_i[kLogitMaxSimdGroups];
    const uint DD = int8_head_fc_d(D);
    const uint VV = int8_head_fc_v(V);

    const uint row = tg_idx * kLMHeadRowsPerTG + simd_group_id;
    float best_v = -INFINITY;
    uint best_i = 0xFFFFFFFFu;

    if (row < VV) {
        const uint n_groups = DD / kInt8GroupSize;
        const float z = int8_dense_row_dot_simd(W      + row * DD,
                                                scales + row * n_groups,
                                                biases + row * n_groups,
                                                x_normed, DD, simd_lane_id);
        if (simd_lane_id == 0 && isfinite(z)) {
            best_v = z;
            best_i = row;
        }
    }

    if (simd_lane_id == 0) {
        partial_v[simd_group_id] = best_v;
        partial_i[simd_group_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        const bool active = simd_lane_id < simdgroups;
        const float v = active ? partial_v[simd_lane_id] : -INFINITY;
        const uint idx = active ? partial_i[simd_lane_id] : 0xFFFFFFFFu;
        const float v_all = simd_max(v);
        uint i_all = (v == v_all) ? idx : 0xFFFFFFFFu;
        i_all = simd_min(i_all);
        if (simd_lane_id == 0) {
            device float* slot = summaries + tg_idx * kLMHeadRowSummaryStride;
            slot[0] = v_all;
            slot[1] = as_type<float>(i_all);
        }
    }
}
