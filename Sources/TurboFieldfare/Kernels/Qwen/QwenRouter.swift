import Foundation
import Metal

/// Swift wrapper for the Qwen 3.5 MoE router kernels in `qwen_head.metal`
/// (plan S13, math from 3.5):
///
///   topK     p = softmax(W_gate * x) in FP32 over E expert logits, top-K
///            greedy selection (ties -> lowest index), renormalized weights
///   sigmoid  the shared-expert scalar gate sg = sigmoid(W_sg * x)
///
/// `encodeTopK` runs one threadgroup of 256 threads, so E <= 256 (Qwen 3.5
/// MoE ships exactly 256 experts) and K <= 8 (exactly 8 routed experts). The
/// logits are the FP16 `y` buffer a DequantInt8GEMV just wrote; indices are
/// `UInt32[K]` and weights FP32[K] in selection order (descending probability,
/// ties by ascending expert index).
///
/// No function constants: the shapes are fixed and tiny, so plain runtime
/// constants keep a single pipeline (the C3 range 205-209 stays unused).
final class QwenRouter {

    /// Kernel limits: threadgroup width and the local selection arrays.
    static let maxExperts = 256
    static let maxTopK = 8

    private let psoTopK: MTLComputePipelineState
    private let psoSigmoid: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoTopK = try context.pipeline("qwen_router_topk")
        self.psoSigmoid = try context.pipeline("qwen_sigmoid_f16")
    }

    /// Softmax + top-K + renormalization over `experts` logits, `rows`
    /// compact rows in one dispatch (S26 step 2; rows = 1 is the old
    /// single-token call, so decode is unchanged).
    /// `logits`: FP16 [rows][experts]; `indices`: UInt32 [rows][topK];
    /// `weights`: Float [rows][topK].
    func encodeTopK(commandBuffer: MTLCommandBuffer,
                    logits: MTLBuffer, logitsOffset: Int = 0,
                    indices: MTLBuffer, indicesOffset: Int = 0,
                    weights: MTLBuffer, weightsOffset: Int = 0,
                    experts: Int,
                    topK: Int,
                    rows: Int = 1) {
        precondition((1...Self.maxExperts).contains(experts),
                     "experts must be 1...\(Self.maxExperts), got \(experts)")
        precondition(topK >= 1 && topK <= Self.maxTopK && topK <= experts,
                     "topK must be 1...min(\(Self.maxTopK), experts), got \(topK) with experts \(experts)")
        precondition(rows >= 1, "rows must be >= 1, got \(rows)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoTopK)
        enc.setBuffer(logits,  offset: logitsOffset,  index: 0)
        enc.setBuffer(indices, offset: indicesOffset, index: 1)
        enc.setBuffer(weights, offset: weightsOffset, index: 2)
        var eVar = UInt32(experts)
        var kVar = UInt32(topK)
        enc.setBytes(&eVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: Self.maxExperts,
                                                                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Elementwise FP16 -> FP16 sigmoid over `count` values (the shared-expert
    /// gate uses count = 1).
    func encodeSigmoid(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       count: Int) {
        guard count > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSigmoid)
        enc.setBuffer(x,   offset: xOffset,   index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var nVar = UInt32(count)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 2)
        let threadsPerGroup = min(Int(psoSigmoid.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threadsPerGroup,
                                                           height: 1, depth: 1))
        enc.endEncoding()
    }
}
