import Foundation
import Metal

/// Fused 8-bit affine Q/K/V decode projection.
///
/// The 8-bit twin of `FusedQKVGEMV`: one dispatch computes `qRows + 2 *
/// kvRows` output rows, one SIMD group per row and eight rows per threadgroup.
/// Full-attention layers have no `v_proj`; as with the 4-bit kernel the caller
/// binds the K tensor again as V.
///
/// The kernel reads weights through 4-byte loads, so every weight offset must
/// be 4-aligned. The `affine8` installer aligns resident weights to 16 bytes.
final class FusedQKVGEMVInt8 {
    private struct Shape: Hashable {
        var qRows: UInt32
        var kvRows: UInt32
        var n: UInt32
    }

    private let pso: MTLComputePipelineState
    private let specializedPSOs: [Shape: MTLComputePipelineState]

    private static let realDecodeShapes: [Shape] = [
        Shape(qRows: 4096, kvRows: 2048, n: 2816),   // sliding-window layers
        Shape(qRows: 8192, kvRows: 1024, n: 2816),   // full-attention layers
    ]

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_int8_qkv_gemv_simd",
                                        constants: [],
                                        maxTotalThreadsPerThreadgroup: 512)
        var variants: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try context.pipeline(
                "dequant_int8_qkv_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 103, value: .uint32(shape.qRows)),
                    MetalFunctionConstant(index: 104, value: .uint32(shape.kvRows)),
                    MetalFunctionConstant(index: 105, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 106, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPSOs = variants
    }

    func encode(commandBuffer: MTLCommandBuffer,
                qWeights: MTLBuffer, qWeightsOffset: Int = 0,
                qScales: MTLBuffer, qScalesOffset: Int = 0,
                qBiases: MTLBuffer, qBiasesOffset: Int = 0,
                kWeights: MTLBuffer, kWeightsOffset: Int = 0,
                kScales: MTLBuffer, kScalesOffset: Int = 0,
                kBiases: MTLBuffer, kBiasesOffset: Int = 0,
                vWeights: MTLBuffer, vWeightsOffset: Int = 0,
                vScales: MTLBuffer, vScalesOffset: Int = 0,
                vBiases: MTLBuffer, vBiasesOffset: Int = 0,
                x: MTLBuffer,
                qOut: MTLBuffer, qOutOffset: Int = 0,
                kOut: MTLBuffer, kOutOffset: Int = 0,
                vOut: MTLBuffer, vOutOffset: Int = 0,
                qRows: UInt32,
                kvRows: UInt32,
                n: UInt32) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(qWeightsOffset % 4 == 0 &&
                     kWeightsOffset % 4 == 0 &&
                     vWeightsOffset % 4 == 0,
                     "FusedQKVGEMVInt8 needs 4-aligned weights offsets")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        let shape = Shape(qRows: qRows, kvRows: kvRows, n: n)
        enc.setComputePipelineState(specializedPSOs[shape] ?? pso)
        enc.setBuffer(qWeights, offset: qWeightsOffset, index: 0)
        enc.setBuffer(qScales, offset: qScalesOffset, index: 1)
        enc.setBuffer(qBiases, offset: qBiasesOffset, index: 2)
        enc.setBuffer(kWeights, offset: kWeightsOffset, index: 3)
        enc.setBuffer(kScales, offset: kScalesOffset, index: 4)
        enc.setBuffer(kBiases, offset: kBiasesOffset, index: 5)
        enc.setBuffer(vWeights, offset: vWeightsOffset, index: 6)
        enc.setBuffer(vScales, offset: vScalesOffset, index: 7)
        enc.setBuffer(vBiases, offset: vBiasesOffset, index: 8)
        enc.setBuffer(x, offset: 0, index: 9)
        enc.setBuffer(qOut, offset: qOutOffset, index: 10)
        enc.setBuffer(kOut, offset: kOutOffset, index: 11)
        enc.setBuffer(vOut, offset: vOutOffset, index: 12)
        var qVar = qRows
        var kvVar = kvRows
        var nVar = n
        enc.setBytes(&qVar, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&kvVar, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 15)
        let totalRows = Int(qRows + 2 * kvRows)
        enc.dispatchThreadgroups(MTLSize(width: (totalRows + 7) / 8,
                                         height: 1,
                                         depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256,
                                                                 height: 1,
                                                                 depth: 1))
        enc.endEncoding()
    }
}

extension FusedQKVGEMVInt8: AffineFusedQKVEncoding {}
