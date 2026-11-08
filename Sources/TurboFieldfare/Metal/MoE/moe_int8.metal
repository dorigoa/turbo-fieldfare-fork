#include <metal_stdlib>
using namespace metal;

// ============================================================================
// moe_int8 — MLX `affine` 8-bit routed-expert decode kernels.
//
// 8-bit twins of `moe_phase1_gate_up_act_u16load`,
// `moe_phase1_gate_up_act_subset_u16load` and `moe_phase2_down_reduce_k8`
// (`moe.metal`): same buffer bindings, same `RoutedBlobs` argument buffer,
// same `ExpertOffsets` struct and the same dispatch geometry, so `MoE` keeps
// one argument-buffer layout and one encoding path for both widths.
//
// Owner: S3 of docs/plans/8bit-model-variant-plan.md. This module is
// concatenated after `moe.metal` and reuses its `kMoEGroupSize`,
// `ExpertOffsets`, `RoutedBlobs`, `gelu_pytorch_tanh` and the `moe_fc_*`
// helpers with function constants 0-3 (D, F, top-k, enable). It declares no
// function constants of its own; S3's reserved range is 110-119.
//
// Layout per weight row of length N (MLX affine 8-bit):
//   q       : N unsigned bytes; byte k is column k, range 0..255.
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[k] = float(q[k]) * scale[k/64] + bias[k/64].
//
// Alignment: the kernels assume only a 4-byte-aligned weight sub-tensor
// offset inside the expert blob (the installer places routed weights on
// 16-byte boundaries; the margin is deliberate). Every row is N bytes with
// N % 64 == 0, so each row base keeps the 4-byte alignment and the row walk
// reads weights through `uint` loads. Scales and biases are read as BF16 and
// need only their natural 2-byte alignment.
// ============================================================================

// Dot product of 8 INT8 codes, packed little-endian in two 32-bit words,
// with 8 activations.
static inline float moe_int8_dot8(uint w0, uint w1, float4 xa, float4 xb) {
    const float4 qa = float4(as_type<uchar4>(w0));
    const float4 qb = float4(as_type<uchar4>(w1));
    return dot(qa, xa) + dot(qb, xb);
}

// One SIMD group computes W_row · x for a single affine INT8 row.
//
// Full 4-group (256-byte) blocks: the 32 lanes split 8 per group, each lane
// reading 8 contiguous weight bytes as two `uint` loads and the matching 8
// activations as two `half4` loads. Scale and bias are constant inside a
// group, so the affine factoring s·Σqx + b·Σx holds per lane and `simd_sum`
// aggregates. Remaining groups (F = 704 has 11: two blocks plus three) cover
// one group per iteration with 2 bytes per lane.
static inline float moe_int8_gemv_row_simd(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    device const uint8_t* W_row = W + uint(row) * N;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4u;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint base = blk * 256u + lane * 8u;
        device const uint* wp = (device const uint*)(W_row + base);
        const float4 xa = float4(*((device const half4*)(x + base)));
        const float4 xb = float4(*((device const half4*)(x + base + 4u)));
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const float dot_qx = moe_int8_dot8(wp[0], wp[1], xa, xb);
        const float sum_x = (xa.x + xa.y + xa.z + xa.w) + (xb.x + xb.y + xb.z + xb.w);
        acc = fma(s, dot_qx, acc);
        acc = fma(b, sum_x, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const uint idx = g * kMoEGroupSize + lane * 2u;
        const uint w2 = uint(*((device const ushort*)(W_row + idx)));
        const float2 xv = float2(*((device const half2*)(x + idx)));
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        float dot_qx = float(w2 & 0xFFu) * xv.x;
        dot_qx = fma(float(w2 >> 8), xv.y, dot_qx);
        acc = fma(s, dot_qx, acc);
        acc = fma(b, xv.x + xv.y, acc);
    }
    return simd_sum(acc);
}

// Gate and up rows share the activation loads; same walk as
// `moe_int8_gemv_row_simd`, two rows at a time.
static inline float2 moe_int8_gate_up_rows_simd(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    device const uint8_t* gW_row = gateW + uint(row) * N;
    device const uint8_t* uW_row = upW + uint(row) * N;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4u;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint base = blk * 256u + lane * 8u;
        device const uint* gp = (device const uint*)(gW_row + base);
        device const uint* up = (device const uint*)(uW_row + base);
        const float4 xa = float4(*((device const half4*)(x + base)));
        const float4 xb = float4(*((device const half4*)(x + base + 4u)));
        const float sum_x = (xa.x + xa.y + xa.z + xa.w) + (xb.x + xb.y + xb.z + xb.w);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const float g_dot = moe_int8_dot8(gp[0], gp[1], xa, xb);
        const float u_dot = moe_int8_dot8(up[0], up[1], xa, xb);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum_x, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum_x, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const uint idx = g * kMoEGroupSize + lane * 2u;
        const uint gw2 = uint(*((device const ushort*)(gW_row + idx)));
        const uint uw2 = uint(*((device const ushort*)(uW_row + idx)));
        const float2 xv = float2(*((device const half2*)(x + idx)));
        const float sum_x = xv.x + xv.y;
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        float g_dot = float(gw2 & 0xFFu) * xv.x;
        g_dot = fma(float(gw2 >> 8), xv.y, g_dot);
        float u_dot = float(uw2 & 0xFFu) * xv.x;
        u_dot = fma(float(uw2 >> 8), xv.y, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum_x, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum_x, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

// One SIMD group per (slot, f) pair: acts[slot * F + f] = gelu(gate) * up.
static inline void moe_int8_phase1_row(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    uint slot,
    uint f,
    uint D,
    uint F,
    uint lane
) {
    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int8_gate_up_rows_simd(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(gelu_pytorch_tanh(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_int8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint K = moe_fc_top_k(top_k);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= K * FF) return;
    moe_int8_phase1_row(routed, routed_offsets, x, acts,
                        rowg / FF, rowg % FF, DD, FF, lane);
}

kernel void moe_phase1_gate_up_act_subset_int8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    const uint K = moe_fc_top_k(top_k);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * FF) return;
    const uint slot = active_slots[rowg / FF];
    if (slot >= K) return;
    moe_int8_phase1_row(routed, routed_offsets, x, acts,
                        slot, rowg % FF, DD, FF, lane);
}

// One threadgroup per output element d; SIMD group k computes slot k's down
// row and the first lane adds the eight weighted partials to the residual in
// slot order, as the 4-bit kernel does.
kernel void moe_phase2_down_reduce_k8_int8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_int8[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int8_gemv_row_simd(dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial_int8[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial_int8[0]; acc += partial_int8[1];
        acc += partial_int8[2]; acc += partial_int8[3];
        acc += partial_int8[4]; acc += partial_int8[5];
        acc += partial_int8[6]; acc += partial_int8[7];
        y[d] = half(acc);
    }
}
