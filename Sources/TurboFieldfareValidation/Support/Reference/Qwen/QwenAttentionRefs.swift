import Foundation

/// FP32 CPU reference for the Qwen 3.5 MoE gated attention mixer (plan S16,
/// math from 3.5 "Layer di attention").
///
/// Like the S12-S15 references this deliberately takes a different path than
/// the Metal kernel: the softmax here is a plain two-pass (max, then sum,
/// then weighted V) while the kernel runs online softmax with FP32
/// accumulators in registers. Inputs are expected already rounded to their
/// storage precision (FP16 activations) — what the kernel reads from device
/// memory.
public enum QwenAttentionRef {

    /// Split `qg` [rows][queryHeads * 2 * headDim] into `q` and `gate`,
    /// each [rows][queryHeads * headDim] (plan 3.5: per head the query is
    /// qg[0:headDim], the gate qg[headDim:2*headDim]).
    public static func splitQG(qg: [Float], rows: Int, queryHeads: Int, headDim: Int)
        -> (q: [Float], gate: [Float]) {
        precondition(qg.count == rows * queryHeads * 2 * headDim, "qg shape")
        var q = [Float](repeating: 0, count: rows * queryHeads * headDim)
        var gate = [Float](repeating: 0, count: q.count)
        for r in 0..<rows {
            for h in 0..<queryHeads {
                let src = (r * queryHeads + h) * 2 * headDim
                let dst = (r * queryHeads + h) * headDim
                for d in 0..<headDim {
                    q[dst + d] = qg[src + d]
                    gate[dst + d] = qg[src + headDim + d]
                }
            }
        }
        return (q, gate)
    }

    /// Partial NeoX-style RoPE (plan 3.5): for each head, rotate only the
    /// first `2 * rotatedPairs` dims, pairing `i` with `i + rotatedPairs`,
    /// inv_freq_i = theta^(-2i / (2*rotatedPairs)); token row r sits at
    /// position `positionBase + r`. `x` is [rows][heads * headDim], returned
    /// rotated; dims outside the rotated block are untouched.
    public static func partialRoPE(x: [Float], rows: Int, heads: Int, headDim: Int,
                                   positionBase: Int, rotatedPairs: Int,
                                   theta: Float) -> [Float] {
        precondition(x.count == rows * heads * headDim, "x shape")
        precondition(2 * rotatedPairs <= headDim, "rotation exceeds headDim")
        var y = x
        for r in 0..<rows {
            let position = Float(positionBase + r)
            for h in 0..<heads {
                let base = (r * heads + h) * headDim
                for i in 0..<rotatedPairs {
                    let invFreq = pow(theta, -Float(2 * i) / Float(2 * rotatedPairs))
                    let angle = position * invFreq
                    let c = cos(angle), s = sin(angle)
                    let x0 = y[base + i], x1 = y[base + rotatedPairs + i]
                    y[base + i] = x0 * c - x1 * s
                    y[base + rotatedPairs + i] = x0 * s + x1 * c
                }
            }
        }
        return y
    }

    /// Append `k`/`v` ([rows][kvHeads * headDim]) to the KV caches at cache
    /// position `kvStart + r`, layout [position][kvHead][headDim] with the
    /// element strides `posStride`/`headStride` (D5: headStride = headDim,
    /// posStride = kvHeads * headDim, or larger for a capacity-strided cache).
    public static func writeKV(k: [Float], v: [Float],
                               cacheK: inout [Float], cacheV: inout [Float],
                               rows: Int, kvHeads: Int, headDim: Int,
                               kvStart: Int, posStride: Int, headStride: Int) {
        precondition(k.count == rows * kvHeads * headDim, "k shape")
        precondition(v.count == k.count, "v must match k")
        precondition(headStride >= headDim, "headStride must span a head")
        let needed = (kvStart + rows - 1) * posStride
            + (kvHeads - 1) * headStride + headDim
        precondition(cacheK.count >= needed && cacheV.count >= needed,
                     "cache too small")
        for r in 0..<rows {
            for h in 0..<kvHeads {
                let src = (r * kvHeads + h) * headDim
                let dst = (kvStart + r) * posStride + h * headStride
                for d in 0..<headDim {
                    cacheK[dst + d] = k[src + d]
                    cacheV[dst + d] = v[src + d]
                }
            }
        }
    }
}
