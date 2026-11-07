import Foundation
import Metal

/// Prefill projection `Y[m, n] = X[m, :] · W[n, :]` on Metal Performance
/// Primitives over MLX affine 8-bit weights.
///
/// Twin of `MPPPrefillInt4QMM` on its `.control` path: the kernel
/// `mpp_prefill_affine8_threadgroup_f16` (private `tensorops_int8` library)
/// dequantizes a 32 x 64 weight tile to FP16 in threadgroup memory and runs
/// `matmul2d` with the same 64 x 32 x 64 tile. There is no Apple10 variant;
/// every device with tensor support takes this path.
///
/// `encode` reuses the 4-bit wrapper's `Path` and `PathMetadata`, so a caller
/// handles both widths the same way: `.affineThreadgroupF16` means the
/// projection was encoded, `.fallback` means nothing was encoded and the caller
/// must use another kernel. Besides an unsupported shape or an unavailable
/// library, the 8-bit path also falls back when the weight offset is not a
/// multiple of 4, because the kernel reads 4-byte words.
public final class MPPPrefillInt8QMM {
    public typealias Path = MPPPrefillInt4QMM.Path
    public typealias PathMetadata = MPPPrefillInt4QMM.PathMetadata

    public static let kernelName = "mpp_prefill_affine8_threadgroup_f16"
    public static let tileM = 64
    public static let tileN = 32
    public static let tileK = Quantization.groupSize
    public static let threadgroupBytes = tileN * tileK * MemoryLayout<Float16>.stride
    public static let cooperativeValueBytesEstimate = 2 * tileM * tileN * MemoryLayout<Float>.stride
    /// Byte alignment the kernel needs for the weight base.
    public static let weightBufferOffsetAlignment = 4

    private let pso: MTLComputePipelineState?
    public let unavailableReason: String?

    public init(context: MetalContext) {
        var pso: MTLComputePipelineState?
        var unavailableReason: String?
        do {
            let library = try MetalContext.privateLibrary(device: context.device,
                                                          module: "tensorops_int8")
            guard let function = library.makeFunction(name: Self.kernelName) else {
                throw NSError(
                    domain: "MPPPrefillInt8QMM",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "\(Self.kernelName) symbol unavailable"])
            }
            pso = try context.device.makeComputePipelineState(function: function)
        } catch {
            unavailableReason = String(describing: error)
        }
        self.pso = pso
        self.unavailableReason = unavailableReason
    }

    public var isAvailable: Bool {
        pso != nil
    }

    @discardableResult
    public func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       m: Int,
                       n: Int,
                       k: Int) -> PathMetadata {
        let supportedShape = m > 0 && n > 0 && k > 0
            && k.isMultiple(of: Self.tileK)
        guard supportedShape,
              weightsOffset >= 0,
              weightsOffset.isMultiple(of: Self.weightBufferOffsetAlignment),
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              yOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              let pso,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return metadata(path: .fallback,
                            weightsOffset: weightsOffset,
                            fallbackCount: 1)
        }

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pso.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return metadata(path: .affineThreadgroupF16,
                        weightsOffset: weightsOffset,
                        fallbackCount: 0)
    }

    private func metadata(path: Path,
                          weightsOffset: Int,
                          fallbackCount: Int) -> PathMetadata {
        PathMetadata(
            path: path,
            tileM: Self.tileM,
            tileN: Self.tileN,
            tileK: Self.tileK,
            modelDerivedCPUHeapScratchBytes: 0,
            expandedWeightBufferBytes: 0,
            threadgroupBytes: Self.threadgroupBytes,
            cooperativeValueBytesEstimate: Self.cooperativeValueBytesEstimate,
            simultaneousWeightTiles: 1,
            weightBufferOffsetAlignment: Self.weightBufferOffsetAlignment,
            weightBufferOffsetRemainder128: weightsOffset % 128,
            fallbackCount: fallbackCount)
    }
}

extension MPPPrefillInt8QMM: AffineMPPPrefillQMMEncoding {}
