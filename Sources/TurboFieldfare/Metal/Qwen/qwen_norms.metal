#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Qwen 3.5 MoE normalization kernels (plan section S12).
//
// All kernels: FP16 storage in and out, FP32 accumulation. Norm weights are
// BF16 in the .gturbo pack (verified by S6: repack keeps them verbatim).
//
// Weight mode for the RMSNorm kernels is a function constant (C3, index 200):
//   FC_QWEN_RMS_ONE_PLUS_W false/undefined -> y = x_hat * w
//   FC_QWEN_RMS_ONE_PLUS_W true           -> y = x_hat * (1 + w)
// The checkpoint ships shifted weights (normWeightsShifted = true, S1/S8),
// so production uses the plain variant; the 1+w variant covers the HF layout.
//
// Dispatch shapes: one threadgroup per row (row RMSNorm) or per (head, row)
// pair (per-head kernels). Grid width = heads (or rows for the row kernel),
// grid height = rows; T = 1 is decode, T > 1 is prefill. 256 threads per
// group — a multiple of the 32-lane SIMD width, so no partial SIMD-group.
// ============================================================================

constant bool FC_QWEN_RMS_ONE_PLUS_W [[function_constant(200)]];

static inline bool qwen_rms_one_plus_w() {
    return is_function_constant_defined(FC_QWEN_RMS_ONE_PLUS_W) &&
           FC_QWEN_RMS_ONE_PLUS_W;
}

// At most 256/32 = 8 partial sums per threadgroup. Slot 0 is reused after
// the merge to broadcast the reciprocal.
constant constexpr uint kQwenMaxSimdGroups = 8;

// Block reduction of the sum of squares over D FP16 elements (FP32
// accumulator). Returns the sum broadcast to every thread via threadgroup
// memory slot 0. Same two-stage structure as rms_block_inv in
// rmsnorm.metal, but without the mean division (L2 norm needs the raw sum).
static inline float qwen_block_sum_sq(
    device const half* x,
    uint  D,
    uint  lid,
    uint  lsize,
    uint  simd_lane_id,
    uint  simd_group_id,
    uint  simdgroups,
    threadgroup float* partial
) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(x[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            partial[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

static inline float qwen_silu(float v) {
    return v / (1.0f + exp(-v));
}

// ----------------------------------------------------------------------------
// Row RMSNorm over the hidden dimension (input_layernorm,
// post_attention_layernorm, model.norm). One threadgroup per row; grid.x = T
// rows (T = 1 decode).
//
//   inv    = rsqrt(mean(x[i]^2) + eps)
//   y[i]   = x[i] * inv * w'[i],  w' = w or 1 + w  (FC 200)
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_rmsnorm_row(
    device const half*   x        [[buffer(0)]],   // [rows * D] FP16
    device const bfloat* weight   [[buffer(1)]],   // [D] BF16
    device       half*   out      [[buffer(2)]],   // [rows * D] FP16
    constant     uint&   D        [[buffer(3)]],
    constant     float&  eps      [[buffer(4)]],
    uint  row              [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kQwenMaxSimdGroups];
    device const half* xr = x + size_t(row) * D;
    device       half* yr = out + size_t(row) * D;

    const float sumSq = qwen_block_sum_sq(xr, D, lid, lsize,
                                          simd_lane_id, simd_group_id,
                                          simdgroups, partial);
    const float inv = rsqrt(sumSq / float(D) + eps);
    const bool onePlusW = qwen_rms_one_plus_w();

    for (uint i = lid; i < D; i += lsize) {
        float wv = float(weight[i]);
        if (onePlusW) {
            wv += 1.0f;
        }
        yr[i] = half(float(xr[i]) * inv * wv);
    }
}

// ----------------------------------------------------------------------------
// Per-head RMSNorm with a shared weight vector: attention q_norm / k_norm
// (headDim = 256, weight shared across heads). Grid.x = head, grid.y = row.
// x layout is [rows][numHeads * headDim] contiguous per token.
//
//   inv  = rsqrt(mean(x_i^2) + eps)   per head
//   y_i  = x_i * inv * w',  w' = w or 1 + w  (FC 200)
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_rmsnorm_perhead(
    device const half*   x        [[buffer(0)]],   // [rows * numHeads * headDim] FP16
    device const bfloat* weight   [[buffer(1)]],   // [headDim] BF16, shared
    device       half*   out      [[buffer(2)]],   // same layout as x
    constant     uint&   headDim  [[buffer(3)]],
    constant     uint&   numHeads [[buffer(4)]],
    constant     float&  eps      [[buffer(5)]],
    uint2 head_row         [[threadgroup_position_in_grid]],
    uint2 tid2             [[thread_position_in_threadgroup]],
    uint2 tsize2           [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kQwenMaxSimdGroups];
    const uint head  = head_row.x;
    const uint row   = head_row.y;
    const uint lid   = tid2.x;
    const uint lsize = tsize2.x;
    const size_t base = (size_t(row) * numHeads + head) * headDim;
    device const half* xh = x + base;
    device       half* yh = out + base;

    const float sumSq = qwen_block_sum_sq(xh, headDim, lid, lsize,
                                          simd_lane_id, simd_group_id,
                                          simdgroups, partial);
    const float inv = rsqrt(sumSq / float(headDim) + eps);
    const bool onePlusW = qwen_rms_one_plus_w();

    for (uint i = lid; i < headDim; i += lsize) {
        float wv = float(weight[i]);
        if (onePlusW) {
            wv += 1.0f;
        }
        yh[i] = half(float(xh[i]) * inv * wv);
    }
}

// ----------------------------------------------------------------------------
// Gated per-head RMSNorm: the DeltaNet output gate
// (linear_attn.norm, headDim = 128, weight always plain w — 3.3 punto 1).
// Grid.x = head, grid.y = row. Both o and z are [rows][numHeads * headDim].
//
//   inv = rsqrt(mean(o_i^2) + eps)
//   y_i = o_i * inv * w[i] * silu(z_i),  silu(v) = v / (1 + exp(-v))
//
// silu is evaluated in FP32 (matches the MLX RMSNormGated reference, S8).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_gated_rmsnorm_perhead(
    device const half*   o        [[buffer(0)]],   // [rows * numHeads * headDim] FP16
    device const half*   z        [[buffer(1)]],   // same layout
    device const bfloat* weight   [[buffer(2)]],   // [headDim] BF16, shared
    device       half*   out      [[buffer(3)]],   // same layout
    constant     uint&   headDim  [[buffer(4)]],
    constant     uint&   numHeads [[buffer(5)]],
    constant     float&  eps      [[buffer(6)]],
    uint2 head_row         [[threadgroup_position_in_grid]],
    uint2 tid2             [[thread_position_in_threadgroup]],
    uint2 tsize2           [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kQwenMaxSimdGroups];
    const uint head  = head_row.x;
    const uint row   = head_row.y;
    const uint lid   = tid2.x;
    const uint lsize = tsize2.x;
    const size_t base = (size_t(row) * numHeads + head) * headDim;
    device const half* oh = o + base;
    device const half* zh = z + base;
    device       half* yh = out + base;

    const float sumSq = qwen_block_sum_sq(oh, headDim, lid, lsize,
                                          simd_lane_id, simd_group_id,
                                          simdgroups, partial);
    const float inv = rsqrt(sumSq / float(headDim) + eps);

    for (uint i = lid; i < headDim; i += lsize) {
        float v = float(oh[i]) * inv * float(weight[i]);
        yh[i] = half(v * qwen_silu(float(zh[i])));
    }
}

// ----------------------------------------------------------------------------
// Per-head L2 norm: DeltaNet q/k renormalization (headDim = 128, no weight).
// Grid.x = head, grid.y = row.
//
//   inv = rsqrt(sum(x_i^2) + eps)     per head — SUM, not mean (3.5)
//   y_i = x_i * inv
//
// The extra 1/sqrt(128) scale that 3.5 applies to q afterwards is NOT part
// of this kernel; callers fold it in where they read the values.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_l2norm_perhead(
    device const half*  x          [[buffer(0)]],  // [rows * numHeads * headDim] FP16
    device       half*  out        [[buffer(1)]],  // same layout
    constant     uint&  headDim    [[buffer(2)]],
    constant     uint&  numHeads   [[buffer(3)]],
    constant     float& eps        [[buffer(4)]],
    uint2 head_row         [[threadgroup_position_in_grid]],
    uint2 tid2             [[thread_position_in_threadgroup]],
    uint2 tsize2           [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kQwenMaxSimdGroups];
    const uint head  = head_row.x;
    const uint row   = head_row.y;
    const uint lid   = tid2.x;
    const uint lsize = tsize2.x;
    const size_t base = (size_t(row) * numHeads + head) * headDim;
    device const half* xh = x + base;
    device       half* yh = out + base;

    const float sumSq = qwen_block_sum_sq(xh, headDim, lid, lsize,
                                          simd_lane_id, simd_group_id,
                                          simdgroups, partial);
    const float inv = rsqrt(sumSq + eps);

    for (uint i = lid; i < headDim; i += lsize) {
        yh[i] = half(float(xh[i]) * inv);
    }
}
