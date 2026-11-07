#include <metal_stdlib>
using namespace metal;

// ============================================================================
// prefill_int8 — INT8 affine dense prefill kernels: block embedding and QMM.
//
// Twins of `prefill_embed_lookup_int4_block` and
// `prefill_dequant_int4_qmm_f16_block` in prefill.metal for MLX affine 8-bit
// weights (group 64, BF16 scale and bias per group). The bindings and the
// dispatch geometry match the 4-bit kernels; only the weight row changes from
// K / 2 packed nibbles to K bytes, where byte k of a row is column k.
//
// Owner: S4 of docs/plans/8bit-model-variant-plan.md (function constants
// 120-129, none used). This module is concatenated after the 4-bit modules
// and reuses `kPrefillGroupSize` from prefill.metal. Every new symbol carries
// an int8 suffix or prefix.
// ============================================================================

constant constexpr uint kPrefillInt8WordsPerGroup = kPrefillGroupSize / 4u;

// One thread per output element, like the 4-bit twin. Reads one byte, so the
// table may sit at any offset.
kernel void prefill_embed_lookup_int8_block(
    device const uint8_t* table     [[buffer(0)]],
    device const bfloat*  scales    [[buffer(1)]],
    device const bfloat*  biases    [[buffer(2)]],
    device const uint*    tokens    [[buffer(3)]],
    device half*          out       [[buffer(4)]],
    constant uint&        T         [[buffer(5)]],
    constant uint&        D         [[buffer(6)]],
    constant float&       out_scale [[buffer(7)]],
    uint2                 gid       [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    const uint token = tokens[t];
    const uint groups_per_row = D / kPrefillGroupSize;
    device const uint8_t* row_q = table  + token * D;
    device const bfloat*  row_s = scales + token * groups_per_row;
    device const bfloat*  row_b = biases + token * groups_per_row;

    const uint q = uint(row_q[d]);
    const float s = float(row_s[d / kPrefillGroupSize]);
    const float b = float(row_b[d / kPrefillGroupSize]);
    out[t * D + d] = half((float(q) * s + b) * out_scale);
}

// Y[t, n] = X[t, :] · W[n, :], one thread per output element in 8x8
// threadgroups, like the 4-bit twin. Each group of 64 columns is read as 16
// little-endian words and folded as `s · Σ q·x + b · Σ x`, the pattern of
// `dequant_int8_gemv_simd`.
//
// Alignment: the weight base must be a multiple of 4 bytes (rows are K bytes
// and K is a multiple of 64, so every row and group stays aligned); scales,
// biases and X need 2 bytes. The Swift wrapper checks the weight offset.
kernel void prefill_dequant_int8_qmm_f16_block(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    X      [[buffer(3)]],
    device half*          Y      [[buffer(4)]],
    constant uint&        T      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        K      [[buffer(7)]],
    uint2                 tid    [[thread_position_in_threadgroup]],
    uint2                 tgid   [[threadgroup_position_in_grid]]
) {
    const uint n = tgid.x * 8u + tid.x;
    const uint t = tgid.y * 8u + tid.y;
    if (t >= T || n >= N) return;

    const uint groups = K / kPrefillGroupSize;
    device const uint* w_row = reinterpret_cast<device const uint*>(W + n * K);
    device const bfloat* s_row = scales + n * groups;
    device const bfloat* b_row = biases + n * groups;
    device const packed_half4* x_row =
        reinterpret_cast<device const packed_half4*>(X + t * K);

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        device const uint* w_group = w_row + g * kPrefillInt8WordsPerGroup;
        device const packed_half4* x_group = x_row + g * kPrefillInt8WordsPerGroup;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint i = 0; i < kPrefillInt8WordsPerGroup; ++i) {
            const uint word = w_group[i];
            const float4 q = float4(float(word & 0xffu),
                                    float((word >> 8) & 0xffu),
                                    float((word >> 16) & 0xffu),
                                    float(word >> 24));
            const float4 x = float4(half4(x_group[i]));
            dot_qx += dot(q, x);
            sum_x += (x.x + x.y) + (x.z + x.w);
        }
        acc = fma(float(s_row[g]), dot_qx, acc);
        acc = fma(float(b_row[g]), sum_x, acc);
    }
    Y[t * N + n] = half(acc);
}
