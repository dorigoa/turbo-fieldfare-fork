import Foundation
import Metal

/// Swift wrapper for the Qwen 3.5 MoE DeltaNet "front half" kernels (S14):
///
///   convSilu  causal depthwise conv (width 4) + SiLU over the 8192 qkv
///             channels, with the FP32 3-tap state (D5 layout [3][channels]);
///             tokens = 1 is decode (one state update), tokens > 1 prefill
///   gates     per-value-head beta/decay in FP32
///             beta = sigmoid(b), decay = exp(-exp(A_log) * softplus(a + dt_bias))
///   scale     FP16 elementwise scale (folds the 1/sqrt(128) into q)
///
/// The q/k/v "split" is pure addressing on the conv output: with conv_dim =
/// 8192 the token slice holds q = [0, 2048) (16 heads x 128), k =
/// [2048, 4096) and v = [4096, 8192) i.e. 32 x 128; callers pass byte
/// offsets (2 B/element) into `QwenNorms.encodeL2PerHead` and `scale`.
///
/// No function constants (S14's C3 range 210-219 stays free). Buffer
/// offsets are in bytes throughout, like the other Qwen wrappers.
final class QwenDeltaNetPrep {

    private let psoConvSilu: MTLComputePipelineState
    private let psoGates: MTLComputePipelineState
    private let psoScale: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoConvSilu = try context.pipeline("qwen_deltanet_conv_silu")
        self.psoGates    = try context.pipeline("qwen_deltanet_gates")
        self.psoScale    = try context.pipeline("qwen_scale_f16")
    }

    /// Causal depthwise conv + SiLU. `qkv`/`y` are FP16 `[tokens][channels]`
    /// token slices; `stateIn`/`stateOut` are FP32 `[3][channels]` (the D5
    /// conv state of one linear layer, passed as offsets into a bigger
    /// buffer). Decode passes `tokens = 1`; the state update and the
    /// arithmetic are the same code path either way, so decode continues a
    /// prefill exactly.
    func encodeConvSilu(commandBuffer: MTLCommandBuffer,
                        qkv: MTLBuffer, qkvOffset: Int = 0,
                        convWeight: MTLBuffer, weightOffset: Int = 0,
                        stateIn: MTLBuffer, stateInOffset: Int = 0,
                        y: MTLBuffer, yOffset: Int = 0,
                        stateOut: MTLBuffer, stateOutOffset: Int = 0,
                        channels: UInt32,
                        tokens: Int) {
        guard tokens > 0, channels > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoConvSilu)
        enc.setBuffer(qkv,        offset: qkvOffset,      index: 0)
        enc.setBuffer(convWeight, offset: weightOffset,   index: 1)
        enc.setBuffer(stateIn,    offset: stateInOffset,  index: 2)
        enc.setBuffer(y,          offset: yOffset,        index: 3)
        enc.setBuffer(stateOut,   offset: stateOutOffset, index: 4)
        var ch = channels
        var tk = UInt32(tokens)
        enc.setBytes(&ch, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&tk, length: MemoryLayout<UInt32>.size, index: 6)
        // One thread per channel; 32 threads per group (SIMD width).
        let threads = min(Int(psoConvSilu.maxTotalThreadsPerThreadgroup), 32)
        enc.dispatchThreadgroups(MTLSize(width: (Int(channels) + threads - 1) / threads,
                                         height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads,
                                                                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Per-value-head gates over `tokens` rows of `heads` (32) FP16
    /// projections `a`, `b` (token slices) and the BF16 `[heads]` `aLog`,
    /// `dtBias`; writes FP32 `beta` and `decay` as `[tokens][heads]`.
    func encodeGates(commandBuffer: MTLCommandBuffer,
                     a: MTLBuffer, aOffset: Int = 0,
                     b: MTLBuffer, bOffset: Int = 0,
                     aLog: MTLBuffer, aLogOffset: Int = 0,
                     dtBias: MTLBuffer, dtBiasOffset: Int = 0,
                     betaOut: MTLBuffer, betaOutOffset: Int = 0,
                     decayOut: MTLBuffer, decayOutOffset: Int = 0,
                     heads: UInt32,
                     tokens: Int) {
        guard tokens > 0, heads > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGates)
        enc.setBuffer(a,        offset: aOffset,        index: 0)
        enc.setBuffer(b,        offset: bOffset,        index: 1)
        enc.setBuffer(aLog,     offset: aLogOffset,     index: 2)
        enc.setBuffer(dtBias,   offset: dtBiasOffset,   index: 3)
        enc.setBuffer(betaOut,  offset: betaOutOffset,  index: 4)
        enc.setBuffer(decayOut, offset: decayOutOffset, index: 5)
        var hd = heads
        var tk = UInt32(tokens)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&tk, length: MemoryLayout<UInt32>.size, index: 7)
        // One thread per (token, head).
        let threads = min(Int(psoGates.maxTotalThreadsPerThreadgroup), 64)
        let total = tokens * Int(heads)
        enc.dispatchThreadgroups(MTLSize(width: (total + threads - 1) / threads,
                                         height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads,
                                                                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// FP16 elementwise scale, one dispatch over `n` contiguous elements.
    /// In-place safe.
    func encodeScale(commandBuffer: MTLCommandBuffer,
                     x: MTLBuffer, xOffset: Int = 0,
                     out: MTLBuffer, outOffset: Int = 0,
                     scale: Float,
                     n: Int) {
        guard n > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoScale)
        enc.setBuffer(x,   offset: xOffset,  index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var s = scale
        var count = UInt32(n)
        enc.setBytes(&s,     length: MemoryLayout<Float>.size,  index: 2)
        enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let threads = min(Int(psoScale.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: (n + threads - 1) / threads,
                                         height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads,
                                                                height: 1, depth: 1))
        enc.endEncoding()
    }
}
