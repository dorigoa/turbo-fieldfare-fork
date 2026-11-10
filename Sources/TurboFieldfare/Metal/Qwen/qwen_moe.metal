#include <metal_stdlib>
using namespace metal;

// Qwen 3.5 MoE kernels: routed expert SiLU-multiply and shared expert.
// Owned by plan section S17; math from plan 3.5 (Blocco MoE), expert blob
// layout from contract C5: one blob per expert holding gate, up, down x
// weight, scales, biases, in that order.
//
// The INT4 coding is the repo-wide affine one (see Quant/dequant_int4.metal):
// low nibble of byte k = even component 2k, high nibble = odd 2k+1; BF16
// scale/bias per group of 64; value = float(nibble) * scale + bias. The GEMV
// factorization per group matches dequant_int4_gemv_simd:
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
//
// Per token the block is (3.5):
//   routed_k = W_down_k * ( silu(W_gate_k * x) * (W_up_k * x) )   (per slot k)
//   shared   = W_down_s * ( silu(W_gate_s * x) * (W_up_s * x) )
//   y        = residual + sum_k w_k * routed_k + sg * shared
// The three kernels below implement exactly these three stages. The scalar
// gate `sg` is the FP16 [1] value produced by the S13 router chain
// (DequantInt8GEMV m=1 + qwen_sigmoid_f16).
//
// No function constants: per-variant shapes are fixed and modest, so plain
// runtime constants keep a single pipeline (the C3 range 240-249 stays
// unused; precedent S13, S15, S16).

constant constexpr uint qwenMoeGroupSize = 64;
constant constexpr uint qwenMoeMaxSlots = 8;

struct QwenExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

struct QwenRoutedBlobs {
    device const uint8_t* blob[qwenMoeMaxSlots];
};

static inline float qwen_moe_silu(float v) {
    return v / (1.0f + exp(-v));
}

// One INT4 affine row dot product, one SIMD group per row: each lane reads
// one byte (two components) per group of 64, accumulates in FP32 and the
// group is reduced with simd_sum. Mirrors dequant_int4_gemv_simd_body's
// scalar tail loop exactly (ascending groups, s-dot then b-sum FMA order).
static inline float qwen_int4_row_dot(
    device const uint8_t* W,       // row start, N/2 bytes
    device const bfloat*  scales,  // row start, N/64
    device const bfloat*  biases,  // row start, N/64
    device const half*    x,       // [N]
    uint                      N,
    uint                      lane)
{
    const uint n_groups = N / qwenMoeGroupSize;
    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(scales[g]);
        const float b = float(biases[g]);
        const uint8_t byte = W[g * (qwenMoeGroupSize / 2u) + lane];
        const float x0 = float(x[g * qwenMoeGroupSize + lane * 2u]);
        const float x1 = float(x[g * qwenMoeGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    return simd_sum(acc);
}

// ----------------------------------------------------------------------------
// Stage 1a: act[slot][i] = silu(gate[slot]_i . x) * (up[slot]_i . x).
// Rows of gate/up are `inter` x `hidden`; blobs[slot] + gate/up offsets per
// contract C5. `slots` is 8 for the routed phase (one blob per streamed
// expert slot) or 1 for the shared expert. One threadgroup of 8 SIMD groups,
// one row per group; lane 0 writes the FP16 activation.
// ----------------------------------------------------------------------------
kernel void qwen_moe_gate_up_silu(
    device const QwenRoutedBlobs& blobs [[buffer(0)]],
    constant QwenExpertOffsets& off [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& inter [[buffer(5)]],
    constant uint& slots [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    const uint slot = row / inter;
    const uint i = row % inter;
    if (slot >= slots) return;

    const uint row_bytes = hidden / 2u;
    const uint row_groups = hidden / qwenMoeGroupSize;

    device const uint8_t* base = blobs.blob[slot];
    const float g = qwen_int4_row_dot(
        base + off.gate_W_off + uint(i) * row_bytes,
        (device const bfloat*)(base + off.gate_s_off) + uint(i) * row_groups,
        (device const bfloat*)(base + off.gate_b_off) + uint(i) * row_groups,
        x, hidden, lane);
    const float u = qwen_int4_row_dot(
        base + off.up_W_off + uint(i) * row_bytes,
        (device const bfloat*)(base + off.up_s_off) + uint(i) * row_groups,
        (device const bfloat*)(base + off.up_b_off) + uint(i) * row_groups,
        x, hidden, lane);
    if (lane == 0) {
        acts[slot * inter + i] = half(qwen_moe_silu(g) * u);
    }
}

// ----------------------------------------------------------------------------
// Stage 1b: routedOut[slot][d] = down[slot]_d . act[slot].
// Rows of down are `hidden` x `inter`; act was written by stage 1a.
// Same threadgroup shape: one row per SIMD group, lane 0 writes FP16.
// ----------------------------------------------------------------------------
kernel void qwen_moe_down(
    device const QwenRoutedBlobs& blobs [[buffer(0)]],
    constant QwenExpertOffsets& off [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device half* routedOut [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& inter [[buffer(5)]],
    constant uint& slots [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint row = tg_idx * rows_per_tg + sg_idx;
    const uint slot = row / hidden;
    const uint d = row % hidden;
    if (slot >= slots) return;

    const uint row_bytes = inter / 2u;
    const uint row_groups = inter / qwenMoeGroupSize;

    device const uint8_t* base = blobs.blob[slot];
    const float v = qwen_int4_row_dot(
        base + off.down_W_off + uint(d) * row_bytes,
        (device const bfloat*)(base + off.down_s_off) + uint(d) * row_groups,
        (device const bfloat*)(base + off.down_b_off) + uint(d) * row_groups,
        acts + slot * inter, inter, lane);
    if (lane == 0) {
        routedOut[slot * hidden + d] = half(v);
    }
}

// ----------------------------------------------------------------------------
// Stage 2: y[d] = residual[d] + sum_k w_k * routedOut[k][d] + sg * shared[d].
// `w` is the FP32 [slots] renormalized router weights from qwen_router_topk
// (S13), `sg` the FP16 [1] shared-expert scalar gate (S13). One thread per
// output component; FP32 accumulation, FP16 write.
// ----------------------------------------------------------------------------
kernel void qwen_moe_residual(
    device const half* residual [[buffer(0)]],
    device const half* routedOut [[buffer(1)]],
    device const half* sharedOut [[buffer(2)]],
    device const float* routingW [[buffer(3)]],
    device const half* sg [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant uint& slots [[buffer(7)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= hidden) return;
    float acc = float(residual[d]);
    const float gate = float(sg[0]);
    for (uint k = 0; k < slots; ++k) {
        acc += routingW[k] * float(routedOut[k * hidden + d]);
    }
    acc += gate * float(sharedOut[d]);
    y[d] = half(acc);
}
