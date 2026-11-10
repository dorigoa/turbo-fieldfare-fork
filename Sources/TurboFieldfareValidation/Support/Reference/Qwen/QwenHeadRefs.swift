import Foundation

/// FP32 CPU references for the Qwen 3.5 MoE router and shared-expert gate
/// kernels (plan S13, math from 3.5).
///
/// `softmaxTopK` mirrors the kernel contract exactly: softmax with the max
/// subtracted, then K greedy selections where a strict `>` comparison over
/// ascending expert indices makes the LOWEST index win exact ties (documented
/// tie rule), the selected probability consumed with -2 so it cannot be
/// picked twice, and the K selected probabilities renormalized to sum 1.
/// The Metal kernel runs the identical algorithm; the two still disagree at
/// the ULP level because `exp` differs between CPU libm and Metal, which is
/// what makes the comparison meaningful.
public enum QwenHeadRef {

    /// Softmax over `logits`, top `topK` by descending probability (ties ->
    /// lowest index first), weights renormalized. Indices are expert ids in
    /// selection order.
    public static func softmaxTopK(logits: [Float], topK: Int)
        -> (indices: [Int], weights: [Float])
    {
        precondition(!logits.isEmpty, "logits must not be empty")
        precondition(topK >= 1 && topK <= logits.count,
                     "topK must be 1...logits.count")
        let mx = logits.max()!
        let exps = logits.map { exp($0 - mx) }
        var p = exps.map { $0 / exps.reduce(0, +) }

        var indices: [Int] = []
        var weights: [Float] = []
        indices.reserveCapacity(topK)
        weights.reserveCapacity(topK)
        var acc: Float = 0
        for _ in 0..<topK {
            var best = 0
            for i in 1..<p.count where p[i] > p[best] {
                best = i
            }
            indices.append(best)
            weights.append(p[best])
            acc += p[best]
            p[best] = -2   // consumed, below every valid probability
        }
        // A softmax over at least one finite logit always has acc > 0.
        let inv: Float = acc > 0 ? 1 / acc : 0
        return (indices, weights.map { $0 * inv })
    }

    /// Elementwise FP32 sigmoid: y = 1 / (1 + exp(-x)).
    public static func sigmoid(_ x: [Float]) -> [Float] {
        x.map { 1 / (1 + exp(-$0)) }
    }
}
