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
}
