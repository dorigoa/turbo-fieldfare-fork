#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Qwen 3.5 MoE router, embedding lookup and final head kernels.
// Owned by plan section S13.
//
// The embedding lookup (EmbedLookupInt4) and the logit head (DequantInt4GEMV)
// reuse the shared Quant kernels unchanged; this file holds only the two
// pieces the Quant family does not provide: the MoE router softmax/top-k and
// the shared-expert scalar gate sigmoid.
//
// No function constants here: shapes are tiny and uniform (256 experts,
// top-8, one token), so runtime constants are enough. The C3 range reserved
// for S13 (205-209) stays unused.
// ============================================================================

// ----------------------------------------------------------------------------
// MoE router: softmax over E expert logits (FP32), greedy top-K selection,
// renormalized weights (plan 3.5: p = softmax(W_gate*x) FP32, top-8, w_e
// renormalized). One threadgroup of 256 threads; E <= 256, K <= 8.
//
//   p[i] = exp(logit[i] - max) / sum(exp(logit[j] - max))   (FP32, all E)
//   repeat K times: pick the largest surviving p; ties -> LOWEST index
//                   (strict > comparison scanning i ascending)
//   weights[k] = p[i_k] / sum of the K selected p's          (FP32 out)
//
// Outputs: indices[k] = uint expert id, weights[k] = FP32 renormalized
// weight, both in selection order (descending probability, ties by ascending
// index). Invalid expert lanes (i >= E) carry a -1e30 sentinel so they never
// win the max or add to the sum; a selected expert is consumed with -2.0,
// below every valid softmax probability (all >= 0).
//
// Grid height is the row count: prefill packs [rows][E] compact logits and
// [rows][K] outputs, and one threadgroup per row covers the chunk (S26 step
// 2). Row 0 reproduces the old single-token call, so decode is unchanged.
// ----------------------------------------------------------------------------
constant constexpr uint kQwenRouterMaxExperts = 256;
constant constexpr uint kQwenRouterMaxTopK    = 8;
constant constexpr uint kQwenRouterThreads    = 256;

[[kernel, max_total_threads_per_threadgroup(kQwenRouterThreads)]]
void qwen_router_topk(
    device const half* logits  [[buffer(0)]],   // rows * E FP16 (GEMV output)
    device       uint* indices [[buffer(1)]],   // rows * K selected expert ids
    device     float*  weights [[buffer(2)]],   // rows * K renormalized FP32
    constant     uint& experts [[buffer(3)]],   // E <= 256
    constant     uint& topK    [[buffer(4)]],   // K <= 8, K <= E
    uint2 tg_position     [[threadgroup_position_in_grid]],
    uint2 tid2            [[thread_position_in_threadgroup]],
    uint  simd_lane_id    [[thread_index_in_simdgroup]],
    uint  simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint  simdgroups      [[simdgroups_per_threadgroup]]
) {
    // Position builtins must all be scalar or all the same vector width
    // (qwen_l2norm_perhead precedent): the row count needs a vector grid.
    const uint lid = tid2.x;
    threadgroup float p[kQwenRouterMaxExperts];
    threadgroup float partial[kQwenRouterThreads / 32];
    const uint E = min(experts, kQwenRouterMaxExperts);

    logits  += size_t(tg_position.y) * experts;
    indices += size_t(tg_position.y) * topK;
    weights += size_t(tg_position.y) * topK;

    const float v = (lid < E) ? float(logits[lid]) : -1.0e30f;
    p[lid] = v;

    // Two-stage block max (same shape as qwen_block_sum_sq in qwen_norms.metal).
    float mx = simd_max(v);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : -1.0e30f;
        t = simd_max(t);
        if (simd_lane_id == 0) {
            partial[0] = t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = partial[0];

    // Softmax: exp relative to the max, block sum, divide.
    const float e = (lid < E) ? exp(p[lid] - mx) : 0.0f;
    float sum = simd_sum(e);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) {
            partial[0] = t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p[lid] = e / partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Greedy selection on one thread: scan i ascending with strict `>`, so
    // equal probabilities keep the LOWEST index (documented tie rule, S13).
    if (lid == 0) {
        const uint K = min(topK, kQwenRouterMaxTopK);
        uint  isel[kQwenRouterMaxTopK];
        float wsel[kQwenRouterMaxTopK];
        float acc = 0.0f;
        for (uint k = 0u; k < K; k++) {
            uint best = 0u;
            for (uint i = 1u; i < E; i++) {
                if (p[i] > p[best]) {
                    best = i;
                }
            }
            isel[k] = best;
            wsel[k] = p[best];
            acc += p[best];
            p[best] = -2.0f;   // consumed: below every valid probability
        }
        // A softmax over at least one finite logit always has acc > 0.
        const float inv = (acc > 0.0f) ? 1.0f / acc : 0.0f;
        for (uint k = 0u; k < K; k++) {
            indices[k] = isel[k];
            weights[k] = wsel[k] * inv;
        }
    }
}

// ----------------------------------------------------------------------------
// Shared-expert scalar gate: y = sigmoid(x), elementwise FP16 -> FP16.
// The router's shared gate is a single value sg = sigmoid(W_sg * x) (plan
// 3.5); the same elementwise kernel also covers any future vector gate.
// ----------------------------------------------------------------------------
[[kernel]]
void qwen_sigmoid_f16(
    device const half* x    [[buffer(0)]],
    device       half* out  [[buffer(1)]],
    constant     uint& n    [[buffer(2)]],
    uint  gid [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }
    const float v = float(x[gid]);
    out[gid] = half(1.0f / (1.0f + exp(-v)));
}
