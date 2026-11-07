import Foundation
import TurboFieldfare

/// CPU side of the greedy LM-head contract shared by `LMHeadChainInt4` and
/// `LMHeadChainInt8`: the final RMSNorm feeds the head as FP16, and the token
/// is the index of the largest finite logit, the lowest index on ties, or 0
/// when no logit is finite.
public enum GreedyHeadRef {
    /// The head input: `RmsNormRef` of the FP16 hidden row with BF16 weights,
    /// rounded to FP16 as the GPU stores it.
    public static func normalizedInput(hidden: [Float16],
                                       normWeightBF16: [UInt16],
                                       eps: Float) -> [Float] {
        RmsNormRef.apply(x: hidden.map(Float.init),
                         weight: normWeightBF16.map(Quantization.bf16ToFloat),
                         eps: eps)
            .map { Float(Float16($0)) }
    }

    /// The greedy pick plus the best and runner-up finite values, so a test
    /// can report how decisive its fixture is.
    public static func argmax(_ logits: [Float])
        -> (index: Int, best: Float, runnerUp: Float) {
        var bestIndex = -1
        var best = -Float.infinity
        var runnerUp = -Float.infinity
        for (index, value) in logits.enumerated() where value.isFinite {
            if bestIndex < 0 || value > best {
                runnerUp = best
                best = value
                bestIndex = index
            } else if value > runnerUp {
                runnerUp = value
            }
        }
        return (max(bestIndex, 0), best, runnerUp)
    }
}
