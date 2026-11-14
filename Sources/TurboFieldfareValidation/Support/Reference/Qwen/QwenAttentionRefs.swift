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

    /// Causal GQA attention over a KV cache (plan 3.5): query head h uses KV
    /// head `h / (queryHeads / kvHeads)`, query token t (absolute position
    /// `kvStart + t`) attends cache positions `[0, kvStart + t]`, scores are
    /// scaled by `scale` and the softmax is a plain FP32 two-pass (max, sum,
    /// weighted V) — the deliberate counterpart to the kernel's online
    /// softmax. `q` is [queryTokens][queryHeads * headDim], the output has
    /// the same shape; the caches use the writeKV layout and strides.
    public static func attention(q: [Float], cacheK: [Float], cacheV: [Float],
                                 posStride: Int, headStride: Int,
                                 queryTokens: Int, queryHeads: Int, kvHeads: Int,
                                 headDim: Int, kvStart: Int, scale: Float) -> [Float] {
        precondition(q.count == queryTokens * queryHeads * headDim, "q shape")
        precondition(queryHeads % kvHeads == 0, "GQA requires queryHeads % kvHeads == 0")
        precondition(headStride >= headDim && posStride >= headStride, "strides")
        let groups = queryHeads / kvHeads
        var out = [Float](repeating: 0, count: q.count)
        var scores = [Float](repeating: 0, count: kvStart + queryTokens)
        var probs = [Float](repeating: 0, count: scores.count)
        for t in 0..<queryTokens {
            let kvLen = kvStart + t + 1
            for h in 0..<queryHeads {
                let kvh = h / groups
                let qBase = (t * queryHeads + h) * headDim
                for p in 0..<kvLen {
                    let kBase = p * posStride + kvh * headStride
                    var dot: Float = 0
                    for d in 0..<headDim {
                        dot += q[qBase + d] * cacheK[kBase + d]
                    }
                    scores[p] = dot * scale
                }
                var maxS: Float = -1e30
                for p in 0..<kvLen { maxS = max(maxS, scores[p]) }
                var sum: Float = 0
                for p in 0..<kvLen {
                    let e = exp(scores[p] - maxS)
                    probs[p] = e
                    sum += e
                }
                let vBaseHead = kvh * headStride
                for d in 0..<headDim {
                    var o: Float = 0
                    for p in 0..<kvLen {
                        o += probs[p] * cacheV[p * posStride + vBaseHead + d]
                    }
                    out[qBase + d] = o / sum
                }
            }
        }
        return out
    }

    /// Output gate (plan 3.5): `out[i] = attn[i] * sigmoid(gate[i])`,
    /// elementwise with an FP32 sigmoid.
    public static func gate(attn: [Float], gate: [Float]) -> [Float] {
        precondition(attn.count == gate.count, "attn/gate must match")
        return (0..<attn.count).map { attn[$0] / (1 + exp(-gate[$0])) }
    }
}
