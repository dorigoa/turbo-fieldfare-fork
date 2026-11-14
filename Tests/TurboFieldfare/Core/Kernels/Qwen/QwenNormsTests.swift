import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the S12 Qwen normalization kernels against the CPU references in
/// `QwenNormRef`. The references use Accelerate sums-of-squares and a scalar
/// reciprocal; the kernels use per-thread `fma` chains with `simd_sum` tree
/// reductions and `rsqrt`. Bugs in one are unlikely to match the other.
///
/// Small shapes catch indexing and reduction bugs at a humanly readable
/// size; the real shapes (d = 2048 rows, 256/128-wide heads, 16/32 heads,
/// T = 1 decode and T > 1 prefill) are what the runner will dispatch (plan
/// 3.2/3.5). Tolerances per contract C8.
@Suite struct QwenNormsTests {

    private static let eps: Float = 1e-6

    private struct BufferAllocFailure: Error {}

    /// BF16 buffer for norm weights, plus the Float values the kernel sees.
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

    // MARK: - Row RMSNorm

    private static func runRow(d: Int, rows: Int, mode: QwenNorms.WeightMode,
                               seed: UInt64) throws {
        var rng = SeedTree(seed).key("qwen-row-d\(d)-rows\(rows)-\(mode)")
        let xFp16 = (0..<rows * d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let xRef  = xFp16.map { Float($0) }
        let wFp32 = (0..<d).map { _ in rng.uniform(0.5, 1.5) }

        let ctx = try MetalContext()
        let kernel = try QwenNorms(context: ctx)
        let (wBuf, wRef) = try weightBuffer(ctx, wFp32)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: rows * d) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeRow(commandBuffer: cb, x: xBuf, weight: wBuf, out: yBuf,
                         d: UInt32(d), rows: rows, eps: eps, mode: mode)
        cb.commit(); cb.waitUntilCompleted()

        var ref = [Float]()
        ref.reserveCapacity(rows * d)
        for r in 0..<rows {
            ref += QwenNormRef.rmsNorm(x: Array(xRef[r * d..<(r + 1) * d]),
                                       weight: wRef, eps: eps,
                                       onePlusW: mode == .onePlusW)
        }
        let actual = Fp16Buffer.read(yBuf, count: rows * d)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "row D=\(d) rows=\(rows) \(mode): relErr=\(relErr) maxAbs=\(maxAbs)")
    }

    @Test func qwenRmsNorm_row_small() throws {
        try Self.runRow(d: 64, rows: 1, mode: .plain, seed: 0xA01)
        try Self.runRow(d: 64, rows: 3, mode: .plain, seed: 0xA02)
    }

    @Test func qwenRmsNorm_row_real() throws {
        // decode (T = 1) and a small prefill (T = 4) at hidden width 2048.
        try Self.runRow(d: 2048, rows: 1, mode: .plain, seed: 0xA03)
        try Self.runRow(d: 2048, rows: 4, mode: .plain, seed: 0xA04)
    }

    @Test func qwenRmsNorm_row_onePlusW() throws {
        // The HF unshifted layout: y = x_hat * (1 + w).
        try Self.runRow(d: 64, rows: 2, mode: .onePlusW, seed: 0xA05)
        try Self.runRow(d: 2048, rows: 1, mode: .onePlusW, seed: 0xA06)
    }

    // MARK: - Per-head RMSNorm (q_norm / k_norm, headDim = 256)

    private static func runPerHead(headDim: Int, heads: Int, rows: Int,
                                   mode: QwenNorms.WeightMode,
                                   seed: UInt64) throws {
        let total = rows * heads * headDim
        var rng = SeedTree(seed).key("qwen-perhead-h\(headDim)-\(heads)-r\(rows)-\(mode)")
        let xFp16 = (0..<total).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let xRef  = xFp16.map { Float($0) }
        let wFp32 = (0..<headDim).map { _ in rng.uniform(0.5, 1.5) }

        let ctx = try MetalContext()
        let kernel = try QwenNorms(context: ctx)
        let (wBuf, wRef) = try weightBuffer(ctx, wFp32)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: total) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodePerHead(commandBuffer: cb, x: xBuf, weight: wBuf, out: yBuf,
                             headDim: UInt32(headDim), numHeads: heads,
                             rows: rows, eps: eps, mode: mode)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenNormRef.perHeadRMSNorm(x: xRef, weight: wRef,
                                             headDim: headDim, eps: eps,
                                             onePlusW: mode == .onePlusW)
        let actual = Fp16Buffer.read(yBuf, count: total)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "perHead h=\(headDim) heads=\(heads) rows=\(rows) \(mode): relErr=\(relErr) maxAbs=\(maxAbs)")
    }

    @Test func qwenRmsNorm_perHead_small() throws {
        try Self.runPerHead(headDim: 32, heads: 2, rows: 2,
                            mode: .plain, seed: 0xB01)
    }

    @Test func qwenRmsNorm_perHead_real() throws {
        // q: 16 heads × 256, decode and prefill; k: 2 heads × 256.
        try Self.runPerHead(headDim: 256, heads: 16, rows: 1,
                            mode: .plain, seed: 0xB02)
        try Self.runPerHead(headDim: 256, heads: 16, rows: 3,
                            mode: .plain, seed: 0xB03)
        try Self.runPerHead(headDim: 256, heads: 2, rows: 2,
                            mode: .plain, seed: 0xB04)
    }

    @Test func qwenRmsNorm_perHead_onePlusW() throws {
        try Self.runPerHead(headDim: 256, heads: 16, rows: 1,
                            mode: .onePlusW, seed: 0xB05)
    }

    // MARK: - Gated per-head RMSNorm (linear_attn.norm, headDim = 128)

    private static func runGated(headDim: Int, heads: Int, rows: Int,
                                 seed: UInt64) throws {
        let total = rows * heads * headDim
        var rng = SeedTree(seed).key("qwen-gated-h\(headDim)-\(heads)-r\(rows)")
        let oFp16 = (0..<total).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let zFp16 = (0..<total).map { _ in Float16(rng.uniform(-3.0, 3.0)) }
        let oRef  = oFp16.map { Float($0) }
        let zRef  = zFp16.map { Float($0) }
        let wFp32 = (0..<headDim).map { _ in rng.uniform(0.5, 1.5) }

        let ctx = try MetalContext()
        let kernel = try QwenNorms(context: ctx)
        let (wBuf, wRef) = try weightBuffer(ctx, wFp32)
        guard let oBuf = Fp16Buffer.make(ctx.device, halves: oFp16),
              let zBuf = Fp16Buffer.make(ctx.device, halves: zFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: total) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeGatedPerHead(commandBuffer: cb, o: oBuf, z: zBuf,
                                  weight: wBuf, out: yBuf,
                                  headDim: UInt32(headDim), numHeads: heads,
                                  rows: rows, eps: eps)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenNormRef.gatedPerHeadRMSNorm(o: oRef, z: zRef,
                                                  weight: wRef,
                                                  headDim: headDim, eps: eps)
        let actual = Fp16Buffer.read(yBuf, count: total)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "gated h=\(headDim) heads=\(heads) rows=\(rows): relErr=\(relErr) maxAbs=\(maxAbs)")
    }

    @Test func qwenGatedRmsNorm_perHead_small() throws {
        try Self.runGated(headDim: 32, heads: 2, rows: 2, seed: 0xC01)
    }

    @Test func qwenGatedRmsNorm_perHead_real() throws {
        // 32 value heads × 128, decode and a small prefill.
        try Self.runGated(headDim: 128, heads: 32, rows: 1, seed: 0xC02)
        try Self.runGated(headDim: 128, heads: 32, rows: 2, seed: 0xC03)
    }

    // MARK: - Per-head L2 norm (DeltaNet q/k renorm, headDim = 128)

    /// `rowStride` (FP16 elements) pads the x rows like the prefill conv
    /// slice (S26 step 2); `xOffsetElements` skips a leading prefix, like
    /// the k slice read after q. nil stride keeps the compact layout.
    private static func runL2(headDim: Int, heads: Int, rows: Int,
                              seed: UInt64,
                              rowStride: Int? = nil,
                              xOffsetElements: Int = 0) throws {
        let rowElems = heads * headDim
        let stride = rowStride ?? rowElems
        let total = rows * rowElems
        var rng = SeedTree(seed).key("qwen-l2-h\(headDim)-\(heads)-r\(rows)")
        let xCount = xOffsetElements + rows * stride
        let xFp16 = (0..<xCount).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        // The reference reads only the heads' elements of each padded row.
        let xRef = (0..<rows).flatMap { r in
            xFp16[(xOffsetElements + r * stride)..<(xOffsetElements + r * stride + rowElems)]
                .map { Float($0) }
        }

        let ctx = try MetalContext()
        let kernel = try QwenNorms(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: total) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeL2PerHead(commandBuffer: cb, x: xBuf,
                               xOffset: xOffsetElements * 2, out: yBuf,
                               headDim: UInt32(headDim), numHeads: heads,
                               rows: rows, eps: eps,
                               xRowStride: rowStride)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenNormRef.perHeadL2Norm(x: xRef, headDim: headDim, eps: eps)
        let actual = Fp16Buffer.read(yBuf, count: total)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "l2 h=\(headDim) heads=\(heads) rows=\(rows): relErr=\(relErr) maxAbs=\(maxAbs)")
    }

    @Test func qwenL2Norm_perHead_small() throws {
        try Self.runL2(headDim: 32, heads: 2, rows: 2, seed: 0xD01)
    }

    @Test func qwenL2Norm_perHead_real() throws {
        // q and k of DeltaNet: 16 heads × 128 each, decode and prefill.
        try Self.runL2(headDim: 128, heads: 16, rows: 1, seed: 0xD02)
        try Self.runL2(headDim: 128, heads: 16, rows: 3, seed: 0xD03)
    }

    @Test func qwenL2Norm_perHead_rowStride_small() throws {
        // Padded x rows and a leading prefix: only the heads' elements of
        // each row participate (S26 step 2, prefill conv slice).
        try Self.runL2(headDim: 32, heads: 2, rows: 3, seed: 0xD04,
                       rowStride: 4 * 32, xOffsetElements: 37)
    }

    @Test func qwenL2Norm_perHead_rowStride_real() throws {
        // The prefill conv row: 16 q heads of 128 read from a row of 4096
        // elements, then the k slice behind q (offset 16*128).
        try Self.runL2(headDim: 128, heads: 16, rows: 4, seed: 0xD05,
                       rowStride: 4096)
        try Self.runL2(headDim: 128, heads: 16, rows: 4, seed: 0xD06,
                       rowStride: 4096, xOffsetElements: 16 * 128)
    }
}
