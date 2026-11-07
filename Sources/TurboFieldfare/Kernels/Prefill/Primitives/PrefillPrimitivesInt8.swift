import Foundation
import Metal

/// Prefill embedding lookup for a block of tokens over an MLX affine 8-bit
/// table: `out[t, d] = (q · scale + bias) · outScale`, with one byte per
/// column and a BF16 scale and bias per group of 64.
///
/// Twin of `PrefillEmbedLookupInt4`: same bindings, dispatch and output
/// scale. The kernel reads one byte per thread, so the table may sit at any
/// offset.
final class PrefillEmbedLookupInt8 {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_embed_lookup_int8_block")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                table: MTLBuffer, tableOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                tokens: MTLBuffer, tokensOffset: Int = 0,
                out: MTLBuffer, outOffset: Int = 0,
                t: UInt32,
                d: UInt32,
                outScale: Float) {
        precondition(d % UInt32(Quantization.groupSize) == 0,
                     "D must be a multiple of \(Quantization.groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(tokens, offset: tokensOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        var tVar = t
        var dVar = d
        var scaleVar = outScale
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 7)
        enc.dispatchThreads(MTLSize(width: Int(d), height: Int(t), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}

/// Prefill projection `Y[t, n] = X[t, :] · W[n, :]` over MLX affine 8-bit
/// weights, without matrix units.
///
/// Twin of `PrefillInt4QMM`: same bindings and 8 x 8 threadgroup geometry.
/// The kernel reads weights as 4-byte words, so the weight offset must be a
/// multiple of 4 (plan C5; `affine8` installs align resident weights to 16
/// bytes).
final class PrefillInt8QMM {
    /// Byte alignment the kernel needs for the weight base.
    static let weightOffsetAlignment = 4

    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_dequant_int8_qmm_f16_block")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                t: Int,
                n: Int,
                k: Int) {
        precondition(k % Quantization.groupSize == 0,
                     "K must be a multiple of \(Quantization.groupSize)")
        precondition(weightsOffset >= 0
                        && weightsOffset.isMultiple(of: Self.weightOffsetAlignment),
                     "INT8 weight offset \(weightsOffset) must be a multiple of \(Self.weightOffsetAlignment)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var tVar = UInt32(t)
        var nVar = UInt32(n)
        var kVar = UInt32(k)
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: (n + 7) / 8, height: (t + 7) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
    }
}

extension PrefillEmbedLookupInt8: AffinePrefillEmbedEncoding {}
extension PrefillInt8QMM: AffinePrefillQMMEncoding {}
