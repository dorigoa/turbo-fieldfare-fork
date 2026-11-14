import Foundation
import Metal

/// Swift wrapper for the Qwen 3.5 MoE DeltaNet recurrence kernels (S15):
///
///   scan  the recurrent state update of plan 3.5 ("stato S_j"): one SIMD
///         group per (value head, value column), the FP32 state column in
///         registers, tokens consumed in order; decode (tokens = 1) and
///         prefill (tokens > 1) share the code path, so decode continues a
///         prefill exactly
///   add   FP16 elementwise add, used for the residual sum h += W_out * o
///
/// The linear-layer output stage itself composes this wrapper with
/// `QwenNorms.encodeGatedPerHead` (S12, gated RMSNorm with silu(z)) and
/// `DequantInt4GEMV` (out_proj 4096 -> 2048, called unmodified per D4); the
/// scan only produces the per-token `o` activations, laid out compactly as
/// [tokens][valueHeads * valueDim] so the gated norm and the GEMV take them
/// directly.
///
/// Buffer offsets are in bytes throughout, like the other Qwen wrappers.
/// No function constants (S15's C3 range 220-229 stays free).
final class QwenDeltaNetScan {

    private let psoScan: MTLComputePipelineState
    private let psoAdd: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoScan = try context.pipeline("qwen_deltanet_scan")
        self.psoAdd  = try context.pipeline("qwen_add_f16")
    }

    /// Gated DeltaNet recurrence. `q`/`k` are FP16 `[tokens][keyHeads *
    /// keyDim]` (the S12-L2'd, S14-scaled q/k from the prep), `v` an FP16
    /// token slice with row stride `vRowStride` elements (8192 for the raw
    /// conv output, value heads starting at `vOffset` bytes); `beta`/`decay`
    /// FP32 `[tokens][valueHeads]` from the S14 gates. `state` is the FP32
    /// `[valueHeads][valueDim][keyDim]` D5 column-major state of one linear
    /// layer, read and written IN PLACE (safe: each SIMD group owns one
    /// column). `o` is FP16 `[tokens][valueHeads * valueDim]`.
    ///
    /// `valueHeads` must be a multiple of `keyHeads` (the 3.5 map: value
    /// head j uses key head `j / (valueHeads / keyHeads)`), and `keyDim` a
    /// multiple of 32 up to 256 (the kernel holds `keyDim / 32` register
    /// rows per lane, 4 at the real 128).
    func encodeScan(commandBuffer: MTLCommandBuffer,
                    q: MTLBuffer, qOffset: Int = 0,
                    k: MTLBuffer, kOffset: Int = 0,
                    v: MTLBuffer, vOffset: Int = 0,
                    beta: MTLBuffer, betaOffset: Int = 0,
                    decay: MTLBuffer, decayOffset: Int = 0,
                    state: MTLBuffer, stateOffset: Int = 0,
                    o: MTLBuffer, oOffset: Int = 0,
                    tokens: Int,
                    valueHeads: Int,
                    keyHeads: Int,
                    keyDim: Int,
                    valueDim: Int,
                    vRowStride: Int) {
        precondition(tokens > 0 && valueHeads > 0 && keyHeads > 0)
        precondition(valueHeads % keyHeads == 0,
                     "valueHeads must be a multiple of keyHeads")
        precondition(keyDim % 32 == 0 && keyDim <= 256,
                     "keyDim must be a multiple of 32 and <= 256, got \(keyDim)")
        precondition(vRowStride >= valueHeads * valueDim,
                     "vRowStride must cover all value heads")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoScan)
        enc.setBuffer(q,     offset: qOffset,     index: 0)
        enc.setBuffer(k,     offset: kOffset,     index: 1)
        enc.setBuffer(v,     offset: vOffset,     index: 2)
        enc.setBuffer(beta,  offset: betaOffset,  index: 3)
        enc.setBuffer(decay, offset: decayOffset, index: 4)
        enc.setBuffer(state, offset: stateOffset, index: 5)
        enc.setBuffer(o,     offset: oOffset,     index: 6)
        var tk = UInt32(tokens)
        var vh = UInt32(valueHeads)
        var kh = UInt32(keyHeads)
        var kd = UInt32(keyDim)
        var vd = UInt32(valueDim)
        var vs = UInt32(vRowStride)
        enc.setBytes(&tk, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&vh, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&kh, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&kd, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&vd, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&vs, length: MemoryLayout<UInt32>.size, index: 12)
        // One SIMD group (32 threads) per (value head, column).
        enc.dispatchThreadgroups(MTLSize(width: valueHeads, height: valueDim, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32,
                                                                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// FP16 elementwise add `out = a + b` over `n` elements (residual sum).
    /// In-place safe (elementwise).
    func encodeAdd(commandBuffer: MTLCommandBuffer,
                   a: MTLBuffer, aOffset: Int = 0,
                   b: MTLBuffer, bOffset: Int = 0,
                   out: MTLBuffer, outOffset: Int = 0,
                   n: Int) {
        guard n > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoAdd)
        enc.setBuffer(a,   offset: aOffset,   index: 0)
        enc.setBuffer(b,   offset: bOffset,   index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var count = UInt32(n)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let threads = min(Int(psoAdd.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: (n + threads - 1) / threads,
                                         height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads,
                                                                height: 1, depth: 1))
        enc.endEncoding()
    }
}
