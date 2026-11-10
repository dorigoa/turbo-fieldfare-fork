import Foundation

/// FP32 CPU references for the Qwen 3.5 MoE DeltaNet front-half kernels
/// (plan S14): causal depthwise conv + SiLU, the per-head beta/decay gates
/// and the 1/sqrt(128) q scale.
///
/// Like the S12 references, these deliberately take a different path than
/// the Metal kernels: a scalar left-to-right accumulation chain (no `fma`)
/// and Foundation `exp`/`log1p`/`sqrt`, while the kernels use `fma` chains
/// and the hardware `exp`/`log1p`. Inputs are expected already rounded to
/// their storage precision (FP16 qkv/a/b, BF16 weights) — what the kernels
/// read from device memory.
public enum QwenDeltaNetPrepRef {

    /// Causal depthwise conv (width 4, cross-correlation, S8-confirmed tap
    /// order) + SiLU over one linear layer.
    ///
    /// - qkv:      FP16-rounded values, `[tokens][channels]` token slice.
    /// - weight:   BF16-rounded values, `[channels][4]` (channel-major, as
    ///             `conv1d.weight [8192][4][1]` flattens).
    /// - stateIn:  FP32 `[3][channels]` — u[t-3], u[t-2], u[t-1] before
    ///             token 0 (plan 3.5, D5 layout).
    /// - Returns:  `y` FP32 `[tokens][channels]` (compare after FP16
    ///             rounding of the GPU result) and `stateOut` FP32
    ///             `[3][channels]` — the last 3 pre-activation inputs.
    public static func convSilu(qkv: [Float], tokens: Int, channels: Int,
                                weight: [Float],
                                stateIn: [Float])
        -> (y: [Float], stateOut: [Float]) {
        precondition(qkv.count == tokens * channels, "qkv shape")
        precondition(weight.count == channels * 4, "weight must be [channels][4]")
        precondition(stateIn.count == 3 * channels, "stateIn must be [3][channels]")
        var y = [Float](repeating: 0, count: tokens * channels)
        var stateOut = [Float](repeating: 0, count: 3 * channels)
        for c in 0..<channels {
            var u0 = stateIn[c]
            var u1 = stateIn[channels + c]
            var u2 = stateIn[2 * channels + c]
            let w0 = weight[c * 4]
            let w1 = weight[c * 4 + 1]
            let w2 = weight[c * 4 + 2]
            let w3 = weight[c * 4 + 3]
            for t in 0..<tokens {
                let u = qkv[t * channels + c]
                // Plain mul/add chain: different rounding than the kernel's fma nest.
                let acc = w0 * u0 + w1 * u1 + w2 * u2 + w3 * u
                y[t * channels + c] = acc / (1.0 + exp(-acc))   // silu, FP32
                u0 = u1; u1 = u2; u2 = u
            }
            // stateOut = u[t-3], u[t-2], u[t-1] after the last token.
            stateOut[c] = u0
            stateOut[channels + c] = u1
            stateOut[2 * channels + c] = u2
        }
        return (y, stateOut)
    }

    /// Softplus with the same clamp point the kernel uses:
    /// log1p(exp(x)) below 20, plain x above (float32 log1p term vanishes).
    public static func softplus(_ x: Float) -> Float {
        x > 20.0 ? x : log1p(exp(x))
    }

    /// Per-value-head gates over `tokens` rows of `heads`:
    ///   beta[t][j]  = sigmoid(b[t][j])
    ///   g[t][j]     = -exp(A_log[j]) * softplus(a[t][j] + dt_bias[j])
    ///   decay[t][j] = exp(g[t][j])
    ///
    /// `a`, `b` FP16-rounded `[tokens][heads]`; `aLog`, `dtBias`
    /// BF16-rounded `[heads]`; outputs FP32 `[tokens][heads]`.
    public static func gates(a: [Float], b: [Float],
                             aLog: [Float], dtBias: [Float])
        -> (beta: [Float], decay: [Float]) {
        precondition(a.count == b.count, "a and b must match")
        precondition(a.count % aLog.count == 0, "a must be whole rows of heads")
        precondition(aLog.count == dtBias.count, "aLog and dtBias must match")
        let heads = aLog.count
        var beta = [Float](repeating: 0, count: a.count)
        var decay = [Float](repeating: 0, count: a.count)
        for idx in 0..<a.count {
            let h = idx % heads
            let bv = b[idx]
            let g = -exp(aLog[h]) * softplus(a[idx] + dtBias[h])
            beta[idx] = 1.0 / (1.0 + exp(-bv))
            decay[idx] = exp(g)
        }
        return (beta, decay)
    }
}
