#include <metal_stdlib>
using namespace metal;

// ============================================================================
// prefill_moe_int8 — INT8 affine grouped routed-expert prefill kernels.
//
// Twins of `prefill_grouped_routed_moe_batched_phase1` and
// `prefill_grouped_routed_moe_batched_down` in prefill.metal for MLX affine
// 8-bit routed experts. The buffer indices, the `PrefillStreamedRoutedBlobsMSL`
// argument buffer, the `PrefillGroupedRoutedMoEStreamedParamsMSL` parameters,
// the scratch layout and the dispatch geometry (one thread per output element,
// 8x8 threadgroups) match the 4-bit kernels, so `PrefillGroupedRoutedMoE`
// keeps one encoding path for both widths. Only the weight rows change, from
// N / 2 packed nibbles to N bytes, where byte k of a row is column k.
//
// Owner: S5 of docs/plans/8bit-model-variant-plan.md (function constants
// 130-139, none used). This module is concatenated after prefill.metal and
// reuses its `kPrefillGroupSize`, `PrefillTokenExpertPairMSL`,
// `PrefillStreamedRoutedBlobsMSL`, `PrefillGroupedRoutedMoEStreamedParamsMSL`,
// `prefill_streamed_local_expert_id` and `prefill_gelu_pytorch_tanh`. Every new
// symbol carries an int8 suffix or prefix.
//
// Layout per weight row of length N (MLX affine 8-bit, plan C5):
//   q       : N unsigned bytes; byte k is column k, range 0..255.
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[k] = float(q[k]) * scale[k/64] + bias[k/64].
//
// Alignment: each group of 64 columns is read as 16 little-endian 32-bit
// words, so the weight sub-tensors (expert blob base plus `*_W_off`) must sit
// on 4-byte boundaries. Rows are N bytes with N % 64 == 0, so every row and
// group keeps that alignment. The installer places routed weights on 16-byte
// boundaries inside 2 MiB-aligned slots (plan §2.5); the 4-byte assumption is
// the deliberate margin of C5. Scales and biases are read as BF16 and need
// 2 bytes; activations are read as `packed_half4`, which also needs 2 bytes.
// `PrefillGroupedRoutedMoE` checks the weight and scale offsets.
// ============================================================================

constant constexpr uint kPrefillMoEInt8WordsPerGroup = kPrefillGroupSize / 4u;

static inline float4 prefill_moe_int8_codes(uint word) {
    return float4(float(word & 0xffu),
                  float((word >> 8) & 0xffu),
                  float((word >> 16) & 0xffu),
                  float(word >> 24));
}

// Same slot search as the 4-bit kernels: the first local slot whose expert
// matches, or `kPrefillMaxTileExperts` when the tile does not bind it.
static inline uint prefill_moe_int8_local_slot(
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p,
    uint expert
) {
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (prefill_streamed_local_expert_id(p, slot) == expert) {
            return slot;
        }
    }
    return kPrefillMaxTileExperts;
}

// W[row, :] · x for one affine INT8 row of N columns, folded per group as
// `s · Σ q·x + b · Σ x`, the pattern of `dequant_int8_gemv_simd`.
static inline float prefill_moe_int8_gemv_row_dev(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kPrefillGroupSize;
    device const uint* w_row = reinterpret_cast<device const uint*>(W + row * N);
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;
    device const packed_half4* x4 = reinterpret_cast<device const packed_half4*>(x);

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        device const uint* w_group = w_row + g * kPrefillMoEInt8WordsPerGroup;
        device const packed_half4* x_group = x4 + g * kPrefillMoEInt8WordsPerGroup;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint i = 0; i < kPrefillMoEInt8WordsPerGroup; ++i) {
            const float4 q = prefill_moe_int8_codes(w_group[i]);
            const float4 xv = float4(half4(x_group[i]));
            dot_qx += dot(q, xv);
            sum_x += (xv.x + xv.y) + (xv.z + xv.w);
        }
        acc = fma(float(s_row[g]), dot_qx, acc);
        acc = fma(float(b_row[g]), sum_x, acc);
    }
    return acc;
}

// Gate and up rows of the same expert share the activation loads. Each row
// sums in the same order as `prefill_moe_int8_gemv_row_dev`.
static inline float2 prefill_moe_int8_gate_up_rows_dev(
    device const uint8_t* gate_W,
    device const bfloat* gate_s,
    device const bfloat* gate_b,
    device const uint8_t* up_W,
    device const bfloat* up_s,
    device const bfloat* up_b,
    device const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kPrefillGroupSize;
    device const uint* g_row = reinterpret_cast<device const uint*>(gate_W + row * N);
    device const uint* u_row = reinterpret_cast<device const uint*>(up_W + row * N);
    device const bfloat* gs_row = gate_s + row * groups;
    device const bfloat* gb_row = gate_b + row * groups;
    device const bfloat* us_row = up_s + row * groups;
    device const bfloat* ub_row = up_b + row * groups;
    device const packed_half4* x4 = reinterpret_cast<device const packed_half4*>(x);

    float gate = 0.0f;
    float up = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        device const uint* g_group = g_row + g * kPrefillMoEInt8WordsPerGroup;
        device const uint* u_group = u_row + g * kPrefillMoEInt8WordsPerGroup;
        device const packed_half4* x_group = x4 + g * kPrefillMoEInt8WordsPerGroup;
        float gate_qx = 0.0f;
        float up_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint i = 0; i < kPrefillMoEInt8WordsPerGroup; ++i) {
            const float4 xv = float4(half4(x_group[i]));
            gate_qx += dot(prefill_moe_int8_codes(g_group[i]), xv);
            up_qx += dot(prefill_moe_int8_codes(u_group[i]), xv);
            sum_x += (xv.x + xv.y) + (xv.z + xv.w);
        }
        gate = fma(float(gs_row[g]), gate_qx, gate);
        gate = fma(float(gb_row[g]), sum_x, gate);
        up = fma(float(us_row[g]), up_qx, up);
        up = fma(float(ub_row[g]), sum_x, up);
    }
    return float2(gate, up);
}

// One thread per (intermediate row f, pair). Writes gate, up and
// gelu(gate) * up into the three planes of `gate_up_act_scratch`, like the
// 4-bit twin.
kernel void prefill_grouped_routed_moe_batched_phase1_int8(
    device const half*                                   hidden               [[buffer(0)]],
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         gate_up_act_scratch  [[buffer(7)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint f = gid.x;
    const uint pair_local = gid.y;
    if (f >= p.F || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    const uint local_slot = prefill_moe_int8_local_slot(p, pair.expert);
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const half* x = hidden + pair.token * p.hidden_stride_elements;
    const float2 gate_up = prefill_moe_int8_gate_up_rows_dev(
        expert + p.gate_W_off,
        reinterpret_cast<device const bfloat*>(expert + p.gate_s_off),
        reinterpret_cast<device const bfloat*>(expert + p.gate_b_off),
        expert + p.up_W_off,
        reinterpret_cast<device const bfloat*>(expert + p.up_s_off),
        reinterpret_cast<device const bfloat*>(expert + p.up_b_off),
        x,
        f,
        p.D);
    const float gate = gate_up.x;
    const float up = gate_up.y;
    const uint row_elements = p.pair_count * p.F;
    const uint index = pair_local * p.F + f;
    gate_up_act_scratch[index] = half(gate);
    gate_up_act_scratch[row_elements + index] = half(up);
    gate_up_act_scratch[2u * row_elements + index] =
        half(prefill_gelu_pytorch_tanh(gate) * up);
}

// One thread per (hidden element d, pair). Reads the activation plane written
// by phase 1 and writes the down projection to `down_scratch` and to the
// pair's (token, rank) row of `route_partials`, like the 4-bit twin.
kernel void prefill_grouped_routed_moe_batched_down_int8(
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         route_partials       [[buffer(5)]],
    device const half*                                   gate_up_act_scratch  [[buffer(7)]],
    device half*                                         down_scratch         [[buffer(8)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint pair_local = gid.y;
    if (d >= p.D || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    const uint local_slot = prefill_moe_int8_local_slot(p, pair.expert);
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const half* act = gate_up_act_scratch + 2u * p.pair_count * p.F + pair_local * p.F;
    const half value = half(prefill_moe_int8_gemv_row_dev(
        expert + p.down_W_off,
        reinterpret_cast<device const bfloat*>(expert + p.down_s_off),
        reinterpret_cast<device const bfloat*>(expert + p.down_b_off),
        act,
        d,
        p.F));
    down_scratch[pair_local * p.D + d] = value;
    route_partials[(pair.token * p.top_k + pair.rank) * p.D + d] = value;
}
