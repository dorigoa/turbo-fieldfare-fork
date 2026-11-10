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
    private let psoRope: MTLComputePipelineState
    private let psoKvWrite: MTLComputePipelineState
    private let psoAttention: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoSplitQG = try context.pipeline("qwen_split_qg")
        self.psoRope = try context.pipeline("qwen_rope_partial")
        self.psoKvWrite = try context.pipeline("qwen_kv_write")
        self.psoAttention = try context.pipeline("qwen_attention")
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

    /// Partial RoPE in place on `data` FP16 [rows][heads * headDim]: rotate
    /// the first `2 * rotatedPairs` dims of every head, pairs (i, i + pairs),
    /// inv_freq_i = theta^(-2i / (2*pairs)); token row r sits at position
    /// `positionBase + r`. Real call: q 16×256 / k 2×256, pairs 32, theta 1e7.
    func encodeRopePartial(commandBuffer: MTLCommandBuffer,
                           data: MTLBuffer, dataOffset: Int = 0,
                           positionBase: UInt32,
                           headDim: Int,
                           numHeads: Int,
                           rows: Int,
                           rotatedPairs: Int,
                           theta: Float) {
        precondition(rotatedPairs > 0 && 2 * rotatedPairs <= headDim,
                     "rotation block must fit in headDim")
        precondition(numHeads > 0)
        guard rows > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoRope)
        enc.setBuffer(data, offset: dataOffset, index: 0)
        var pos = positionBase
        var hd = UInt32(headDim)
        var nh = UInt32(numHeads)
        var rp = UInt32(rotatedPairs)
        var th = theta
        enc.setBytes(&pos, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&nh,  length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&rp,  length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&th,  length: MemoryLayout<Float>.size,  index: 5)
        enc.dispatchThreads(MTLSize(width: rotatedPairs, height: numHeads, depth: rows),
                            threadsPerThreadgroup: MTLSize(width: 32,
                                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Append `k`/`v` FP16 [rows][kvHeads * headDim] to the D5 attention KV
    /// cache: token row r goes to cache position `kvStart + r`, where a
    /// position spans `posStride` elements and a KV head inside it
    /// `headStride` (the plain D5 layout: headStride = headDim,
    /// posStride = kvHeads * headDim; larger strides allow a
    /// capacity-strided buffer). `kvK`/`kvV` are the per-layer cache
    /// slices (buffer offsets point at the layer). Elementwise copy.
    func encodeKVWrite(commandBuffer: MTLCommandBuffer,
                       k: MTLBuffer, kOffset: Int = 0,
                       v: MTLBuffer, vOffset: Int = 0,
                       kvK: MTLBuffer, kvKOffset: Int = 0,
                       kvV: MTLBuffer, kvVOffset: Int = 0,
                       rows: Int,
                       kvHeads: Int,
                       headDim: Int,
                       kvStart: UInt32,
                       posStride: Int,
                       headStride: Int) {
        precondition(kvHeads > 0 && headDim > 0)
        precondition(posStride >= headStride && headStride >= headDim,
                     "strides must not overlap between heads/positions")
        guard rows > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoKvWrite)
        enc.setBuffer(k,   offset: kOffset,   index: 0)
        enc.setBuffer(v,   offset: vOffset,   index: 1)
        enc.setBuffer(kvK, offset: kvKOffset, index: 2)
        enc.setBuffer(kvV, offset: kvVOffset, index: 3)
        var rw = UInt32(rows)
        var kh = UInt32(kvHeads)
        var hd = UInt32(headDim)
        var ks = kvStart
        var ps = UInt32(posStride)
        var hs = UInt32(headStride)
        enc.setBytes(&rw, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&kh, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&ks, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&ps, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&hs, length: MemoryLayout<UInt32>.size, index: 9)
        enc.dispatchThreads(MTLSize(width: headDim, height: kvHeads, depth: rows),
                            threadsPerThreadgroup: MTLSize(width: 64,
                                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Causal GQA attention over the D5 KV cache: `q` FP16 [T][queryHeads *
    /// headDim], `kvK`/`kvV` the cache slices with the encodeKVWrite strides,
    /// `out` FP16 like q. Query token t attends cache positions
    /// [0, kvStart + t]; decode is queryTokens = 1, prefill T (call after
    /// encodeKVWrite on the same command buffer). FP32 online softmax,
    /// GQA ratio queryHeads/kvHeads (real 16/2 = 8). headDim must be <= 256
    /// (the kernel gives one dimension to each of 256 threads).
    func encodeAttention(commandBuffer: MTLCommandBuffer,
                         q: MTLBuffer, qOffset: Int = 0,
                         kvK: MTLBuffer, kvKOffset: Int = 0,
                         kvV: MTLBuffer, kvVOffset: Int = 0,
                         out: MTLBuffer, outOffset: Int = 0,
                         queryHeads: Int,
                         kvHeads: Int,
                         headDim: Int,
                         queryTokens: Int,
                         kvStart: UInt32,
                         posStride: Int,
                         headStride: Int,
                         scale: Float) {
        precondition(queryHeads > 0 && kvHeads > 0)
        precondition(queryHeads % kvHeads == 0,
                     "queryHeads must be a multiple of kvHeads for GQA")
        precondition(headDim > 0 && headDim <= 256,
                     "headDim must be <= 256 (one dimension per thread)")
        precondition(posStride >= headStride && headStride >= headDim,
                     "strides must not overlap between heads/positions")
        guard queryTokens > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoAttention)
        enc.setBuffer(q,    offset: qOffset,    index: 0)
        enc.setBuffer(kvK,  offset: kvKOffset,  index: 1)
        enc.setBuffer(kvV,  offset: kvVOffset,  index: 2)
        enc.setBuffer(out,  offset: outOffset,  index: 3)
        var qh = UInt32(queryHeads)
        var kh = UInt32(kvHeads)
        var hd = UInt32(headDim)
        var qt = UInt32(queryTokens)
        var ks = kvStart
        var ps = UInt32(posStride)
        var hs = UInt32(headStride)
        var sc = scale
        enc.setBytes(&qh, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&kh, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&qt, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&ks, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&ps, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&hs, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&sc, length: MemoryLayout<Float>.size, index: 11)
        enc.dispatchThreadgroups(MTLSize(width: queryHeads, height: queryTokens, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256,
                                                                height: 1, depth: 1))
        enc.endEncoding()
    }
}
