import Foundation
import Metal

/// Swift wrapper for the Qwen 3.5 MoE gated attention kernels (S16).
///
///   splitQG  per head, separate the query and the gate of the q_proj output
///            ([rows][queryHeads * 512] -> q and gate [rows][queryHeads * 256])
///   rope     partial NeoX-style RoPE on the FIRST `2 * rotatedPairs` dims of
///            each head, pairs (i, i + rotatedPairs), inv_freq_i =
///            theta^(-2i / (2*rotatedPairs)) — the 3.5 attention spec rotates
///            the first 64 dims with theta 1e7. Gemma's `rope_proportional_neox`
///            cannot be reused (it pairs (i, headDim/2 + i) and divides the
///            frequency exponent by headDim), so this is a qwen_ kernel per D4.
///   kvWrite  append rows of K and V to the D5 attention KV cache, laid out
///            [position][kvHead][headDim] inside the per-layer slice; the
///            position/head strides are parameters so the runner can pass a
///            capacity-strided buffer.
///   attn     causal GQA attention over the cache, FP32 online softmax,
///            decode (queryTokens = 1) and prefill (queryTokens = T) share
///            the code path: query token t attends cache positions
///            [0, kvStart + t].
///   gate     elementwise out = attn * sigmoid(gate).
///
/// `q_norm`/`k_norm` reuse `QwenNorms.encodePerHead` (S12, headDim 256,
/// `.plain` mode = `normWeightsShifted = true`). Buffer offsets are in bytes
/// throughout, like the other Qwen wrappers. No function constants (S16's C3
/// range 230-239 stays free).
final class QwenAttention {

    private let psoSplitQG: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoSplitQG = try context.pipeline("qwen_split_qg")
    }

    /// Separate query and gate of a `q_proj` output: `qg` FP16
    /// [rows][queryHeads * 2 * headDim] (per head: query then gate),
    /// `q`/`gate` FP16 [rows][queryHeads * headDim]. Elementwise copy.
    func encodeSplitQG(commandBuffer: MTLCommandBuffer,
                       qg: MTLBuffer, qgOffset: Int = 0,
                       q: MTLBuffer, qOffset: Int = 0,
                       gate: MTLBuffer, gateOffset: Int = 0,
                       queryHeads: Int,
                       headDim: Int,
                       rows: Int) {
        precondition(queryHeads > 0 && headDim > 0)
        guard rows > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSplitQG)
        enc.setBuffer(qg,   offset: qgOffset,   index: 0)
        enc.setBuffer(q,    offset: qOffset,    index: 1)
        enc.setBuffer(gate, offset: gateOffset, index: 2)
        var qh = UInt32(queryHeads)
        var hd = UInt32(headDim)
        var rw = UInt32(rows)
        enc.setBytes(&qh, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&rw, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreads(MTLSize(width: headDim, height: queryHeads, depth: rows),
                            threadsPerThreadgroup: MTLSize(width: 64,
                                                           height: 1, depth: 1))
        enc.endEncoding()
    }
}
