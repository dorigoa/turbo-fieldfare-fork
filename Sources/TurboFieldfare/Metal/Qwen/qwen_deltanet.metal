#include <metal_stdlib>
using namespace metal;

// Qwen 3.5 MoE Gated DeltaNet kernels: causal depthwise conv, gates and the
// recurrent state update. Owned by plan sections S14 and S15.
//
// S14 owns the three "front half" kernels: the causal depthwise conv with
// SiLU, the per-value-head beta/decay gates, and the FP16 scale used to fold
// 1/sqrt(128) into q after the S12 L2 norm. S15 appends the recurrence scan
// below.
//
// Shared conventions (plan 3.5, S1/S8 confirmations):
//   - qkv token slices are FP16 [tokens][conv_dim], conv_dim = 8192.
//   - conv1d.weight is BF16 [conv_dim][4][1] (MLX layout, taps along dim 1);
//     the conv is a cross-correlation: y_c[t] = sum_j w[c][j] * u_c[t-3+j].
//   - The conv state holds the last 3 PRE-activation inputs, FP32, D5 layout
//     [layer][3][conv_dim]; a single layer is [3][conv_dim].
//   - A_log and dt_bias are BF16 [32] (one entry per value head).
// `qwen_silu` comes from qwen_norms.metal, compiled earlier in the same
// library translation unit.
// ----------------------------------------------------------------------------

static inline float qwen_dn_softplus(float x) {
    // log(1 + exp(x)); MSL at this language level has no log1p. For x > 20
    // the 1 inside the log is below float32 precision, so return x directly
    // (this also keeps exp(x) finite).
    return x > 20.0f ? x : log(1.0f + exp(x));
}

// ----------------------------------------------------------------------------
// Causal depthwise conv (width 4) + SiLU over the conv_dim qkv channels.
// One thread per channel; the token loop carries the rolling 3-tap window in
// registers, so the SAME kernel serves decode (tokens = 1, one state update)
// and prefill (tokens > 1, state in and out). Channel-major thread mapping
// keeps reads of qkv[t][c] coalesced at every step of the loop.
//
//   u[t]  = qkv[t][c]                                (before activation)
//   y[t]  = silu(w[c][0]*u[t-3] + w[c][1]*u[t-2]
//                + w[c][2]*u[t-1] + w[c][3]*u[t])
//   state = [u[t-3], u[t-2], u[t-1]]                 (after the last token)
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(32)]]
void qwen_deltanet_conv_silu(
    device const half*   qkv       [[buffer(0)]],  // [tokens * channels] FP16
    device const bfloat* w         [[buffer(1)]],  // [channels * 4] BF16 (c*4 + j)
    device const float*  stateIn   [[buffer(2)]],  // [3 * channels] FP32 (slot*channels + c)
    device       half*   y         [[buffer(3)]],  // [tokens * channels] FP16
    device       float*  stateOut  [[buffer(4)]],  // [3 * channels] FP32
    constant     uint&   channels  [[buffer(5)]],
    constant     uint&   tokens    [[buffer(6)]],
    uint c               [[thread_position_in_grid]]
) {
    if (c >= channels) { return; }

    float u0 = stateIn[c];                    // u[t-3] seen at token 0
    float u1 = stateIn[channels + c];         // u[t-2]
    float u2 = stateIn[2 * channels + c];     // u[t-1]

    const device bfloat* wc = w + size_t(c) * 4;
    const float w0 = float(wc[0]);
    const float w1 = float(wc[1]);
    const float w2 = float(wc[2]);
    const float w3 = float(wc[3]);

    device const half* src = qkv + c;
    device       half* dst = y + c;
    const size_t stride = channels;
    for (uint t = 0; t < tokens; ++t) {
        const float u = float(src[t * stride]);
        const float acc = fma(w0, u0, fma(w1, u1, fma(w2, u2, w3 * u)));
        dst[t * stride] = half(qwen_silu(acc));
        u0 = u1; u1 = u2; u2 = u;
    }

    stateOut[c] = u0;
    stateOut[channels + c] = u1;
    stateOut[2 * channels + c] = u2;
}

// ----------------------------------------------------------------------------
// Per-value-head gates, one thread per (token, head):
//
//   beta_j[t]  = sigmoid(b[t][j])
//   g_j[t]     = -exp(A_log[j]) * softplus(a[t][j] + dt_bias[j])
//   decay_j[t] = exp(g_j[t])
//
// Inputs FP16 [tokens][heads] (W_b/W_a projections), BF16 [heads] for A_log
// and dt_bias; outputs FP32 [tokens][heads] (3.5: computed and consumed in
// FP32). heads = 32 (value heads).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(64)]]
void qwen_deltanet_gates(
    device const half*   a         [[buffer(0)]],  // [tokens * heads] FP16
    device const half*   b         [[buffer(1)]],  // [tokens * heads] FP16
    device const bfloat* aLog      [[buffer(2)]],  // [heads] BF16
    device const bfloat* dtBias    [[buffer(3)]],  // [heads] BF16
    device       float*  beta      [[buffer(4)]],  // [tokens * heads] FP32
    device       float*  decay     [[buffer(5)]],  // [tokens * heads] FP32
    constant     uint&   heads     [[buffer(6)]],
    constant     uint&   tokens    [[buffer(7)]],
    uint idx             [[thread_position_in_grid]]
) {
    const uint total = tokens * heads;
    if (idx >= total) { return; }
    const uint t = idx / heads;
    const uint h = idx - t * heads;

    const float av = float(a[idx]);
    const float bv = float(b[idx]);
    const float g = -exp(float(aLog[h])) * qwen_dn_softplus(av + float(dtBias[h]));

    beta[idx]  = 1.0f / (1.0f + exp(-bv));
    decay[idx] = exp(g);
}

// ----------------------------------------------------------------------------
// FP16 elementwise scale: out[i] = x[i] * scale. S14 uses it to fold the
// 1/sqrt(128) q scale (plan 3.5) into q after the S12 L2 norm; the S12
// l2norm kernel deliberately omits it. In-place safe (elementwise).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_scale_f16(
    device const half*  x      [[buffer(0)]],
    device       half*  out    [[buffer(1)]],
    constant     float& scale  [[buffer(2)]],
    constant     uint&  n      [[buffer(3)]],
    uint i             [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    out[i] = half(float(x[i]) * scale);
}
