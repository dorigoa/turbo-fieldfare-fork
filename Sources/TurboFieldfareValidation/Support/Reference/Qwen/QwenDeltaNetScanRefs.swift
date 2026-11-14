import Foundation

/// FP32 CPU reference for the Qwen 3.5 MoE DeltaNet recurrence scan (plan
/// S15, math from 3.5 "stato S_j").
///
/// Like the S12/S14 references this deliberately takes a different path than
/// the Metal kernel: a scalar per-row chain with plain mul/add (no `fma`)
/// and a plain sequential sum for `kv_mem_c`/`o_j[c]`, while the kernel
/// keeps the whole state column in registers across 32 lanes and reduces
/// with `simd_sum` trees. Inputs are expected already rounded to their
/// storage precision (FP16 q/k/v, FP32 beta/decay/state) — what the kernel
/// reads from device memory.
///
/// State layout is the D5 one: `[valueHeads][valueDim][keyDim]`, one value
/// column = `keyDim` contiguous floats (row i = key dimension, plan 3.5).
public enum QwenDeltaNetScanRef {

    /// One recurrent layer over `tokens` tokens.
    ///
    /// - q:    FP16-rounded `[tokens][keyHeads * keyDim]` (L2'd, scaled).
    /// - k:    FP16-rounded `[tokens][keyHeads * keyDim]` (L2'd).
    /// - v:    FP16-rounded `[tokens][vRowStride]` token slice (value head
    ///         j starts at column `j * valueDim`).
    /// - beta/decay: FP32 `[tokens][valueHeads]` from the S14 gates.
    /// - stateIn: FP32 `[valueHeads][valueDim][keyDim]`.
    /// - Returns: `o` FP32 `[tokens][valueHeads * valueDim]` (compare after
    ///   FP16 rounding of the GPU result) and `stateOut` FP32, same layout.
    public static func scan(q: [Float], k: [Float], v: [Float],
                            beta: [Float], decay: [Float],
                            stateIn: [Float],
                            tokens: Int, valueHeads: Int, keyHeads: Int,
                            keyDim: Int, valueDim: Int, vRowStride: Int)
        -> (o: [Float], stateOut: [Float]) {
        precondition(q.count == tokens * keyHeads * keyDim, "q shape")
        precondition(k.count == q.count, "k must match q")
        precondition(beta.count == tokens * valueHeads, "beta shape")
        precondition(decay.count == beta.count, "decay must match beta")
        precondition(stateIn.count == valueHeads * valueDim * keyDim, "state shape")
        precondition(valueHeads % keyHeads == 0, "valueHeads must be a multiple of keyHeads")
        let groups = valueHeads / keyHeads

        var state = stateIn
        var o = [Float](repeating: 0, count: tokens * valueHeads * valueDim)
        for t in 0..<tokens {
            for j in 0..<valueHeads {
                let kh = j / groups
                let bj = beta[t * valueHeads + j]
                let dj = decay[t * valueHeads + j]
                for c in 0..<valueDim {
                    let colBase = (j * valueDim + c) * keyDim
                    // Sequential per-column update, row by row.
                    for i in 0..<keyDim {
                        state[colBase + i] *= dj
                    }
                    var kvMem: Float = 0
                    for i in 0..<keyDim {
                        kvMem += state[colBase + i] * k[t * keyHeads * keyDim + kh * keyDim + i]
                    }
                    let vv = v[t * vRowStride + j * valueDim + c]
                    let delta = (vv - kvMem) * bj
                    for i in 0..<keyDim {
                        state[colBase + i] += k[t * keyHeads * keyDim + kh * keyDim + i] * delta
                    }
                    var oVal: Float = 0
                    for i in 0..<keyDim {
                        oVal += state[colBase + i] * q[t * keyHeads * keyDim + kh * keyDim + i]
                    }
                    o[t * valueHeads * valueDim + j * valueDim + c] = oVal
                }
            }
        }
        return (o, state)
    }

    /// Elementwise FP32 residual add: `out[i] = a[i] + b[i]` (one token row).
    public static func residualAdd(h: [Float], proj: [Float]) -> [Float] {
        precondition(h.count == proj.count, "h and proj must match length")
        return (0..<h.count).map { h[$0] + proj[$0] }
    }
}
