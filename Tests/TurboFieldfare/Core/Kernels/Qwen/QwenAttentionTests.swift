import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the S16 gated attention kernels (`qwen_split_qg`,
/// `qwen_rope_partial`, `qwen_kv_write`, `qwen_attention`, `qwen_attn_gate`)
/// against the CPU references in `QwenAttentionRef`.
///
/// Real shapes per plan 3.2/3.5: 16 query heads of 256 (query+gate 512 in
/// the q_proj output), 2 KV heads of 256 (GQA ratio 8), partial RoPE on the
/// first 64 dims with theta 1e7, the KV cache in the D5 FP16 layout
/// [position][kvHead][256], causal GQA with scale 1/16 and FP32 softmax,
/// plus the output gate sigmoid(gate) * attn.
/// Tolerances per contract C8.
@Suite struct QwenAttentionTests {

    private static let queryHeads = 16
    private static let kvHeads = 2
    private static let headDim = 256            // 3.2: head_dim = 256
    private static let rotatedPairs = 32        // 3.5: first 64 dims
    private static let ropeTheta: Float = 1e7   // 3.2: partial theta = 1e7
    private static let eps: Float = 1e-6        // C4 rmsNormEps

    // MARK: - Helpers

    private struct BufferAllocFailure: Error {}

    /// BF16 norm-weight buffer + the FP32 values the CPU reference sees.
    private static func weightBuffer(_ ctx: MetalContext, _ values: [Float])
        throws -> (MTLBuffer, [Float]) {
        let bits = values.map { Quantization.bf16Bits($0) }
        guard let buf = ctx.device.makeBuffer(length: bits.count * 2,
                                              options: .storageModeShared) else {
            Issue.record("weight alloc failed")
            throw BufferAllocFailure()
        }
        let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: bits.count)
        for i in 0..<bits.count { ptr[i] = bits[i] }
        return (buf, bits.map { Quantization.bf16ToFloat($0) })
    }

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    // MARK: - Query/gate split

    /// The split is a pure element copy: GPU output must be bit-exact.
    private static func runSplit(rows: Int, queryHeads: Int, headDim: Int,
                                 seed: UInt64) throws {
        var rng = SeedTree(seed).key("qwen-split-r\(rows)-h\(queryHeads)-d\(headDim)")
        let n = rows * queryHeads * 2 * headDim
        let qgFp16 = (0..<n).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let qgRef = qgFp16.map { Float($0) }

        let ctx = try MetalContext()
        let attn = try QwenAttention(context: ctx)
        guard let qgBuf = Fp16Buffer.make(ctx.device, halves: qgFp16),
              let qBuf = Fp16Buffer.make(ctx.device, count: n / 2),
              let gateBuf = Fp16Buffer.make(ctx.device, count: n / 2) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        attn.encodeSplitQG(commandBuffer: cb, qg: qgBuf, q: qBuf, gate: gateBuf,
                           queryHeads: queryHeads, headDim: headDim, rows: rows)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenAttentionRef.splitQG(qg: qgRef, rows: rows,
                                           queryHeads: queryHeads, headDim: headDim)
        let qOut = Fp16Buffer.read(qBuf, count: n / 2)
        let gateOut = Fp16Buffer.read(gateBuf, count: n / 2)
        #expect(maxAbsDiff(qOut, ref.q) == 0, "q must be a bit-exact slice")
        #expect(maxAbsDiff(gateOut, ref.gate) == 0, "gate must be a bit-exact slice")
    }

    @Test func split_qg_small() throws {
        try Self.runSplit(rows: 3, queryHeads: 4, headDim: 128, seed: 0xB01)
    }

    @Test func split_qg_real() throws {
        // Real geometry: 16 heads of 512 (query + gate) per token.
        try Self.runSplit(rows: 3, queryHeads: Self.queryHeads, headDim: Self.headDim,
                          seed: 0xB02)
        try Self.runSplit(rows: 1, queryHeads: Self.queryHeads, headDim: Self.headDim,
                          seed: 0xB03)
    }

    // MARK: - q_norm / k_norm wiring (S12 kernel on the split slices)

    /// Split, then `QwenNorms.encodePerHead` on the query heads and on the
    /// raw k projection (the q_norm/k_norm the 3.5 spec demands), compared
    /// against `QwenNormRef.perHeadRMSNorm` with the plain (shifted) weight.
    private static func runNorms(rows: Int, queryHeads: Int, kvHeads: Int,
                                 headDim: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("qwen-attnnorm-r\(rows)-h\(queryHeads)-d\(headDim)")
        let qg = (0..<rows * queryHeads * 2 * headDim).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let kRaw = (0..<rows * kvHeads * headDim).map { _ in Float16(rng.uniform(-2.0, 2.0)) }

        let ctx = try MetalContext()
        let attn = try QwenAttention(context: ctx)
        let norms = try QwenNorms(context: ctx)
        let (wQ, wQRef) = try weightBuffer(ctx, (0..<headDim).map { _ in rng.uniform(0.5, 1.5) })
        let (wK, wKRef) = try weightBuffer(ctx, (0..<headDim).map { _ in rng.uniform(0.5, 1.5) })
        guard let qgBuf = Fp16Buffer.make(ctx.device, halves: qg),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kRaw),
              let qBuf = Fp16Buffer.make(ctx.device, count: rows * queryHeads * headDim),
              let gateBuf = Fp16Buffer.make(ctx.device, count: rows * queryHeads * headDim),
              let qNormBuf = Fp16Buffer.make(ctx.device, count: rows * queryHeads * headDim),
              let kNormBuf = Fp16Buffer.make(ctx.device, count: rows * kvHeads * headDim) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        attn.encodeSplitQG(commandBuffer: cb, qg: qgBuf, q: qBuf, gate: gateBuf,
                           queryHeads: queryHeads, headDim: headDim, rows: rows)
        norms.encodePerHead(commandBuffer: cb, x: qBuf, weight: wQ, out: qNormBuf,
                            headDim: UInt32(headDim), numHeads: queryHeads,
                            rows: rows, eps: eps, mode: .plain)
        norms.encodePerHead(commandBuffer: cb, x: kBuf, weight: wK, out: kNormBuf,
                            headDim: UInt32(headDim), numHeads: kvHeads,
                            rows: rows, eps: eps, mode: .plain)
        cb.commit(); cb.waitUntilCompleted()

        let split = QwenAttentionRef.splitQG(qg: qg.map { Float($0) }, rows: rows,
                                             queryHeads: queryHeads, headDim: headDim)
        let qRef = QwenNormRef.perHeadRMSNorm(x: split.q, weight: wQRef,
                                              headDim: headDim, eps: eps,
                                              onePlusW: false)
        let kRef = QwenNormRef.perHeadRMSNorm(x: kRaw.map { Float($0) }, weight: wKRef,
                                              headDim: headDim, eps: eps,
                                              onePlusW: false)
        let qOut = Fp16Buffer.read(qNormBuf, count: rows * queryHeads * headDim)
        let kOut = Fp16Buffer.read(kNormBuf, count: rows * kvHeads * headDim)
        let qRel = RelError.compute(actual: qOut, reference: qRef)
        let kRel = RelError.compute(actual: kOut, reference: kRef)
        #expect(qRel < Tolerance.fp16Reduction, "q_norm relErr=\(qRel)")
        #expect(kRel < Tolerance.fp16Reduction, "k_norm relErr=\(kRel)")
    }

    @Test func qkNorms_small() throws {
        try Self.runNorms(rows: 2, queryHeads: 4, kvHeads: 2, headDim: 128,
                          seed: 0xB10)
    }

    @Test func qkNorms_real() throws {
        try Self.runNorms(rows: 3, queryHeads: Self.queryHeads, kvHeads: Self.kvHeads,
                          headDim: Self.headDim, seed: 0xB11)
    }
}
