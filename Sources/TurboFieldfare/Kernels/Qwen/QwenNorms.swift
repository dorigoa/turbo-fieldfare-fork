import Foundation
import Metal

/// Swift wrapper for the Qwen 3.5 MoE normalization kernel family (S12).
///
///   rowRMS      y = x * rsqrt(mean(x^2) + eps) * w'   (w' = w or 1 + w)
///   perHeadRMS  same, over contiguous heads of `headDim` with a shared weight
///   gated       y = (o * rsqrt(mean(o^2) + eps) * w) * silu(z)   (always w)
///   l2          y = x * rsqrt(sum(x^2) + eps)          (sum, not mean)
///
/// All buffers are FP16; norm weights are BF16 as stored in the Qwen
/// `.gturbo` pack. The shifted-weight mode is a pipeline function constant
/// (C3, index 200): production weights are already shifted (S1/S8), so the
/// plain variant is what `normWeightsShifted = true` selects; the 1+w
/// variant exists for the raw HF layout.
///
/// Row and per-head kernels take a `rows` count: 1 row is decode, T > 1 rows
/// are prefill. Per-head x layout is `[rows][numHeads * headDim]`, the token
/// slice layout the projections write.
final class QwenNorms {

    /// Which weight expression the RMSNorm pipelines apply.
    enum WeightMode {
        case plain       // y = x_hat * w
        case onePlusW    // y = x_hat * (1 + w)
    }

    private static let functionConstantIndex: Int = 200

    private let psoRowPlain: MTLComputePipelineState
    private let psoRowOnePlusW: MTLComputePipelineState
    private let psoPerHeadPlain: MTLComputePipelineState
    private let psoPerHeadOnePlusW: MTLComputePipelineState
    private let psoGatedPerHead: MTLComputePipelineState
    private let psoL2PerHead: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoRowPlain        = try context.pipeline("qwen_rmsnorm_row")
        self.psoRowOnePlusW     = try Self.onePlusWPipeline(context, "qwen_rmsnorm_row")
        self.psoPerHeadPlain    = try context.pipeline("qwen_rmsnorm_perhead")
        self.psoPerHeadOnePlusW = try Self.onePlusWPipeline(context, "qwen_rmsnorm_perhead")
        self.psoGatedPerHead    = try context.pipeline("qwen_gated_rmsnorm_perhead")
        self.psoL2PerHead       = try context.pipeline("qwen_l2norm_perhead")
    }

    /// RMSNorm over `rows` rows of `d` values, one threadgroup per row.
    /// In-place safe (each row's threadgroup touches only its own range).
    func encodeRow(commandBuffer: MTLCommandBuffer,
                   x: MTLBuffer, xOffset: Int = 0,
                   weight: MTLBuffer, weightOffset: Int = 0,
                   out: MTLBuffer, outOffset: Int = 0,
                   d: UInt32,
                   rows: Int,
                   eps: Float,
                   mode: WeightMode = .plain) {
        guard rows > 0 else { return }
        let pso = mode == .onePlusW ? psoRowOnePlusW : psoRowPlain
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(x,      offset: xOffset,      index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out,    offset: outOffset,    index: 2)
        var dVar   = d
        var epsVar = eps
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 4)
        dispatch(enc: enc, pso: pso, width: rows, height: 1)
        enc.endEncoding()
    }

    /// RMSNorm over `rows` rows × `numHeads` contiguous heads of `headDim`,
    /// one dispatch, one threadgroup per (head, row). `weight` is the shared
    /// [headDim] per-head gain.
    func encodePerHead(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       weight: MTLBuffer, weightOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       headDim: UInt32,
                       numHeads: Int,
                       rows: Int,
                       eps: Float,
                       mode: WeightMode = .plain) {
        guard rows > 0, numHeads > 0 else { return }
        let pso = mode == .onePlusW ? psoPerHeadOnePlusW : psoPerHeadPlain
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(x,      offset: xOffset,      index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out,    offset: outOffset,    index: 2)
        var hd  = headDim
        var nh  = UInt32(numHeads)
        var epsVar = eps
        enc.setBytes(&hd,     length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&nh,     length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 5)
        dispatch(enc: enc, pso: pso, width: numHeads, height: rows)
        enc.endEncoding()
    }

    /// Gated per-head RMSNorm: (o normalized with the shared BF16 weight,
    /// always plain `w`) * silu(z), over `rows` rows × `numHeads` heads.
    /// `o`, `z` and `out` share the `[rows][numHeads * headDim]` layout.
    func encodeGatedPerHead(commandBuffer: MTLCommandBuffer,
                            o: MTLBuffer, oOffset: Int = 0,
                            z: MTLBuffer, zOffset: Int = 0,
                            weight: MTLBuffer, weightOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            headDim: UInt32,
                            numHeads: Int,
                            rows: Int,
                            eps: Float) {
        guard rows > 0, numHeads > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGatedPerHead)
        enc.setBuffer(o,      offset: oOffset,      index: 0)
        enc.setBuffer(z,      offset: zOffset,      index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        enc.setBuffer(out,    offset: outOffset,    index: 3)
        var hd  = headDim
        var nh  = UInt32(numHeads)
        var epsVar = eps
        enc.setBytes(&hd,     length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&nh,     length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 6)
        dispatch(enc: enc, pso: psoGatedPerHead, width: numHeads, height: rows)
        enc.endEncoding()
    }

    /// L2 norm (sum-of-squares, not mean) over `rows` rows × `numHeads`
    /// contiguous heads, one dispatch. No weight.
    func encodeL2PerHead(commandBuffer: MTLCommandBuffer,
                         x: MTLBuffer, xOffset: Int = 0,
                         out: MTLBuffer, outOffset: Int = 0,
                         headDim: UInt32,
                         numHeads: Int,
                         rows: Int,
                         eps: Float) {
        guard rows > 0, numHeads > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoL2PerHead)
        enc.setBuffer(x,   offset: xOffset,   index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var hd  = headDim
        var nh  = UInt32(numHeads)
        var epsVar = eps
        enc.setBytes(&hd,     length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&nh,     length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 4)
        dispatch(enc: enc, pso: psoL2PerHead, width: numHeads, height: rows)
        enc.endEncoding()
    }

    private func dispatch(enc: MTLComputeCommandEncoder,
                          pso: MTLComputePipelineState,
                          width: Int,
                          height: Int) {
        // One row (or head) per threadgroup, 256 threads: 256 is a multiple
        // of the 32-lane SIMD width on Apple silicon.
        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: width, height: height, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threadsPerGroup,
                                                                height: 1, depth: 1))
    }

    private static func onePlusWPipeline(_ context: MetalContext,
                                         _ name: String) throws -> MTLComputePipelineState {
        try context.pipeline(name,
                             constants: [MetalFunctionConstant(index: functionConstantIndex,
                                                               value: .bool(true))])
    }
}
