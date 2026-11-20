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
///            [0, kvStart + t]. Shapes the grouped kernel covers (headDim a
///            multiple of 128 up to 256, at most 8 query heads per KV head —
///            the real model is 256 and 16/2) run `qwen_attention_grouped`,
///            split over position chunks in decode and merged by
///            `qwen_attention_combine`; any other shape runs the original
///            one-threadgroup-per-head `qwen_attention`.
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
    private let psoGrouped: MTLComputePipelineState
    private let psoCombine: MTLComputePipelineState
    private let psoGate: MTLComputePipelineState
    private let device: MTLDevice

    /// Per-chunk online-softmax state of a split decode: m and d are
    /// [rows][chunks] FP32, o is [rows][chunks][headDim] FP32. Allocated on the
    /// first split decode and grown only if a larger shape arrives; every layer
    /// of a token reuses them because the encoders of one command buffer run
    /// in order.
    private var mPartial: MTLBuffer?
    private var dPartial: MTLBuffer?
    private var oPartial: MTLBuffer?

    /// Most chunks a decode is split into, and the fewest positions a chunk
    /// holds. Chosen on an M4 Max at the real geometry (16/2 heads of 256, ten
    /// layers): 64 chunks was fastest from about 3K positions to 128K, and
    /// below that a chunk of about 16 positions was the sweet spot.
    static let maxDecodeChunks = 64
    static let minDecodeChunkLength = 16

    init(context: MetalContext) throws {
        self.psoSplitQG = try context.pipeline("qwen_split_qg")
        self.psoRope = try context.pipeline("qwen_rope_partial")
        self.psoKvWrite = try context.pipeline("qwen_kv_write")
        self.psoAttention = try context.pipeline("qwen_attention")
        self.psoGrouped = try context.pipeline("qwen_attention_grouped")
        self.psoCombine = try context.pipeline("qwen_attention_combine")
        self.psoGate = try context.pipeline("qwen_attn_gate")
        self.device = context.device
    }

    /// Whether `qwen_attention_grouped` covers this shape: one half4 column
    /// block per 128 dims (at most two), one SIMD group per query head of a
    /// GQA group (at most eight).
    static func groupedKernelCovers(queryHeads: Int, kvHeads: Int, headDim: Int) -> Bool {
        guard kvHeads > 0, queryHeads % kvHeads == 0 else { return false }
        let qPerKV = queryHeads / kvHeads
        return headDim % 128 == 0 && headDim <= 256 && qPerKV >= 1 && qPerKV <= 8
    }

    /// Chunks for a decode over `kvLen` positions: at least
    /// `minDecodeChunkLength` positions each, at most `maxDecodeChunks`, and
    /// no empty chunk (the count is recomputed from the rounded length).
    static func decodeSplit(kvLen: Int) -> (chunks: Int, chunkLength: Int) {
        let len = max(1, kvLen)
        let wanted = min(maxDecodeChunks, max(1, (len + minDecodeChunkLength - 1) / minDecodeChunkLength))
        let chunkLength = (len + wanted - 1) / wanted
        return ((len + chunkLength - 1) / chunkLength, chunkLength)
    }

    private func partialBuffers(rows: Int, chunks: Int, headDim: Int)
        -> (MTLBuffer, MTLBuffer, MTLBuffer)? {
        let states = rows * chunks * MemoryLayout<Float>.size
        let vectors = states * headDim
        if let m = mPartial, let d = dPartial, let o = oPartial,
           m.length >= states, o.length >= vectors {
            return (m, d, o)
        }
        guard let m = device.makeBuffer(length: states, options: .storageModePrivate),
              let d = device.makeBuffer(length: states, options: .storageModePrivate),
              let o = device.makeBuffer(length: vectors, options: .storageModePrivate)
        else { return nil }
        m.label = "qwen attention split m"
        d.label = "qwen attention split d"
        o.label = "qwen attention split o"
        mPartial = m; dPartial = d; oPartial = o
        return (m, d, o)
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
    /// GQA ratio queryHeads/kvHeads (real 16/2 = 8). headDim must be <= 256.
    /// Covered shapes take the grouped kernel (split over position chunks when
    /// queryTokens = 1); the rest take the original per-head kernel.
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
        if Self.groupedKernelCovers(queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim) {
            encodeGroupedAttention(commandBuffer: commandBuffer,
                                   q: q, qOffset: qOffset,
                                   kvK: kvK, kvKOffset: kvKOffset,
                                   kvV: kvV, kvVOffset: kvVOffset,
                                   out: out, outOffset: outOffset,
                                   queryHeads: queryHeads, kvHeads: kvHeads,
                                   headDim: headDim, queryTokens: queryTokens,
                                   kvStart: kvStart, posStride: posStride,
                                   headStride: headStride, scale: scale)
            return
        }
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

    private func encodeGroupedAttention(commandBuffer: MTLCommandBuffer,
                                        q: MTLBuffer, qOffset: Int,
                                        kvK: MTLBuffer, kvKOffset: Int,
                                        kvV: MTLBuffer, kvVOffset: Int,
                                        out: MTLBuffer, outOffset: Int,
                                        queryHeads: Int, kvHeads: Int,
                                        headDim: Int, queryTokens: Int,
                                        kvStart: UInt32, posStride: Int,
                                        headStride: Int, scale: Float) {
        // Prefill already has a threadgroup per (KV head, token); only a
        // single-token decode needs the cache split to fill the GPU.
        var chunks = 1
        var chunkLength = Int(kvStart) + queryTokens
        var partials: (MTLBuffer, MTLBuffer, MTLBuffer)?
        if queryTokens == 1 {
            let split = Self.decodeSplit(kvLen: Int(kvStart) + 1)
            if split.chunks > 1,
               let buffers = partialBuffers(rows: queryHeads, chunks: split.chunks,
                                            headDim: headDim) {
                chunks = split.chunks
                chunkLength = split.chunkLength
                partials = buffers
            }
        }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGrouped)
        enc.setBuffer(q,    offset: qOffset,    index: 0)
        enc.setBuffer(kvK,  offset: kvKOffset,  index: 1)
        enc.setBuffer(kvV,  offset: kvVOffset,  index: 2)
        enc.setBuffer(out,  offset: outOffset,  index: 3)
        // Unused when chunks == 1, but every declared buffer gets a binding.
        enc.setBuffer(partials?.0 ?? out, offset: 0, index: 4)
        enc.setBuffer(partials?.1 ?? out, offset: 0, index: 5)
        enc.setBuffer(partials?.2 ?? out, offset: 0, index: 6)
        var qh = UInt32(queryHeads)
        var kh = UInt32(kvHeads)
        var hd = UInt32(headDim)
        var qt = UInt32(queryTokens)
        var ks = kvStart
        var ps = UInt32(posStride)
        var hs = UInt32(headStride)
        var sc = scale
        var cl = UInt32(chunkLength)
        var nc = UInt32(chunks)
        enc.setBytes(&qh, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&kh, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&qt, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&ks, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&ps, length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&hs, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&sc, length: MemoryLayout<Float>.size,  index: 14)
        enc.setBytes(&cl, length: MemoryLayout<UInt32>.size, index: 15)
        enc.setBytes(&nc, length: MemoryLayout<UInt32>.size, index: 16)
        enc.dispatchThreadgroups(MTLSize(width: kvHeads, height: chunks, depth: queryTokens),
                                 threadsPerThreadgroup: MTLSize(width: 32 * (queryHeads / kvHeads),
                                                                height: 1, depth: 1))
        enc.endEncoding()

        guard let partials, chunks > 1,
              let combine = commandBuffer.makeComputeCommandEncoder() else { return }
        combine.setComputePipelineState(psoCombine)
        combine.setBuffer(partials.0, offset: 0, index: 0)
        combine.setBuffer(partials.1, offset: 0, index: 1)
        combine.setBuffer(partials.2, offset: 0, index: 2)
        combine.setBuffer(out, offset: outOffset, index: 3)
        var rows = UInt32(queryHeads)
        combine.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 4)
        combine.setBytes(&nc, length: MemoryLayout<UInt32>.size, index: 5)
        combine.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 6)
        combine.dispatchThreadgroups(MTLSize(width: queryHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: min(256, headDim),
                                                                    height: 1, depth: 1))
        combine.endEncoding()
    }

    /// Output gate (plan 3.5): `out = attn * sigmoid(gate)` elementwise over
    /// `count` FP16 elements; `gate` comes from encodeSplitQG, `attn` from
    /// encodeAttention. FP32 sigmoid, one FP16 rounding.
    func encodeGate(commandBuffer: MTLCommandBuffer,
                    attn: MTLBuffer, attnOffset: Int = 0,
                    gate: MTLBuffer, gateOffset: Int = 0,
                    out: MTLBuffer, outOffset: Int = 0,
                    count: Int) {
        precondition(count > 0)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGate)
        enc.setBuffer(attn, offset: attnOffset, index: 0)
        enc.setBuffer(gate, offset: gateOffset, index: 1)
        enc.setBuffer(out,  offset: outOffset,  index: 2)
        var n = UInt32(count)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64,
                                                           height: 1, depth: 1))
        enc.endEncoding()
    }
}
