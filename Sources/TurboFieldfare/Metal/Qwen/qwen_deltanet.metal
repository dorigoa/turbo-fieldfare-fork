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

// ----------------------------------------------------------------------------
// Recurrent Gated DeltaNet scan (S15, plan 3.5 "stato S_j").
//
// One SIMD group per (value head j, value column c): the 128 rows of the
// FP32 state column S_j[:, c] live in registers (32 lanes x keyDim/32
// rows, 4 at keyDim = 128) and the tokens are consumed in order, so the
// SAME kernel serves decode (tokens = 1) and prefill (tokens > 1), and
// decode continues a prefill exactly. Each (j, c) column is touched only
// by its own SIMD group, so the state buffer is read and written in place
// (D5 layout [value head][column c][row i], one column = keyDim floats).
//
// Per token t (plan 3.5, key head kh = j / (valueHeads / keyHeads)):
//   S[i][c]      *= decay_j[t]
//   kv_mem_c      = sum_i S[i][c] * k[t][kh][i]        (simd_sum)
//   delta_c       = (v[t][j*valueDim + c] - kv_mem_c) * beta_j[t]
//   S[i][c]      += k[t][kh][i] * delta_c
//   o[t][j][c]    = sum_i S[i][c] * q[t][kh][i]        (simd_sum)
//
// State arithmetic is FP32 (C8 tracks its drift over 256 steps); q, k, v
// arrive FP16 from the S12/S14 prep, beta/decay FP32 from the S14 gates.
// kDim must be a multiple of 32 and <= 256 (8 register rows per lane).
// ----------------------------------------------------------------------------
#define QWEN_DN_MAX_ROWS_PER_LANE 8u

[[kernel, max_total_threads_per_threadgroup(32)]]
void qwen_deltanet_scan(
    device const half*  q,           // [tokens][keyHeads*keyDim] FP16 (L2'd, scaled)
    device const half*  k,           // [tokens][keyHeads*keyDim] FP16 (L2'd)
    device const half*  v,           // [tokens][vRowStride] FP16, base at value head 0
    device const float* beta,        // [tokens][valueHeads] FP32
    device const float* decay,       // [tokens][valueHeads] FP32
    device       float* state,       // [valueHeads][valueDim][keyDim] FP32 (D5), in place
    device       half*  o,           // [tokens][valueHeads*valueDim] FP16
    constant     uint&  tokens       [[buffer(7)]],
    constant     uint&  valueHeads   [[buffer(8)]],
    constant     uint&  keyHeads     [[buffer(9)]],
    constant     uint&  keyDim       [[buffer(10)]],
    constant     uint&  valueDim     [[buffer(11)]],
    constant     uint&  vRowStride   [[buffer(12)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint j = tg.x;                     // value head
    const uint c = tg.y;                     // value column
    const uint kh = j / (valueHeads / keyHeads);
    const uint rowsPerLane = keyDim / 32u;

    float s[QWEN_DN_MAX_ROWS_PER_LANE];
    const size_t colBase = (size_t(j) * valueDim + c) * keyDim;
    #pragma unroll
    for (uint r = 0; r < QWEN_DN_MAX_ROWS_PER_LANE; ++r) { s[r] = 0.0f; }
    #pragma unroll
    for (uint r = 0; r < QWEN_DN_MAX_ROWS_PER_LANE; ++r) {
        if (r < rowsPerLane) { s[r] = state[colBase + lane * rowsPerLane + r]; }
    }

    const size_t qkRow = size_t(keyHeads) * keyDim;
    const size_t oRow  = size_t(valueHeads) * valueDim;
    const size_t vBase = size_t(j) * valueDim + c;
    const size_t qkLane = size_t(kh) * keyDim + lane * rowsPerLane;

    for (uint t = 0; t < tokens; ++t) {
        const float dj = decay[t * valueHeads + j];
        const float bj = beta[t * valueHeads + j];
        device const half* kt = k + size_t(t) * qkRow + qkLane;
        device const half* qt = q + size_t(t) * qkRow + qkLane;

        float kr[QWEN_DN_MAX_ROWS_PER_LANE];
        float qr[QWEN_DN_MAX_ROWS_PER_LANE];
        float dotK = 0.0f, dotQ = 0.0f;
        #pragma unroll
        for (uint r = 0; r < QWEN_DN_MAX_ROWS_PER_LANE; ++r) {
            if (r >= rowsPerLane) { kr[r] = 0.0f; qr[r] = 0.0f; continue; }
            kr[r] = float(kt[r]);
            qr[r] = float(qt[r]);
            s[r] *= dj;
            dotK = fma(s[r], kr[r], dotK);
        }
        const float kvMem = simd_sum(dotK);
        const float vv = float(v[size_t(t) * vRowStride + vBase]);
        const float delta = (vv - kvMem) * bj;

        #pragma unroll
        for (uint r = 0; r < QWEN_DN_MAX_ROWS_PER_LANE; ++r) {
            if (r >= rowsPerLane) { continue; }
            s[r] = fma(kr[r], delta, s[r]);
            dotQ = fma(s[r], qr[r], dotQ);
        }
        // simd_sum is collective: all 32 lanes must execute it.
        const float oVal = simd_sum(dotQ);
        if (lane == 0) {
            o[size_t(t) * oRow + vBase] = half(oVal);
        }
    }

    #pragma unroll
    for (uint r = 0; r < QWEN_DN_MAX_ROWS_PER_LANE; ++r) {
        if (r < rowsPerLane) { state[colBase + lane * rowsPerLane + r] = s[r]; }
    }
}

// ----------------------------------------------------------------------------
// FP16 elementwise add: out[i] = a[i] + b[i]. S15 uses it for the linear
// layer's residual sum h += W_out * o (plan 3.5); out == a or out == b is
// safe (elementwise).
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_add_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device       half* out [[buffer(2)]],
    constant     uint& n   [[buffer(3)]],
    uint i          [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    out[i] = half(float(a[i]) + float(b[i]));
}
