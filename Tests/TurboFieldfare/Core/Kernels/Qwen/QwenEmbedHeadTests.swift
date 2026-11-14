import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// S13: the Qwen 3.5 MoE embedding lookup and final logit head reuse the
/// shared 4-bit Quant kernels unchanged. These tests verify the two claims
/// the card pins down:
///   * `EmbedLookupInt4` accepts the Qwen embedding table — vocabulary
///     V = 248,320, hidden width D = 2,048, `outScale = 1.0` (no
///     `sqrt(hidden)` multiplication, plan 3.5) — with no wrapper changes.
///   * `DequantInt4GEMV` on the generic (non-specialized) pipeline runs the
///     lm_head at m = 248,320, n = 2,048 and produces FP16 logits.
///
/// The kernel never takes a vocabulary size: rows are addressed as
/// `uint(token_id) * (D / 2)` bytes, which at V = 248,320 and D = 2,048 peaks
/// at 254.3 M — far below 2^32 — so the only real requirement is that the
/// caller supplies a full-size table. That is what the production-shaped tests
/// do: they allocate the whole V × D/2 table and query edge tokens.
@Suite struct QwenEmbedHeadTests {

    static let vocab = 248_320
    static let d = 2048
    static let groupsPerRow = d / Quantization.groupSize   // 32

    private struct BufferAllocFailure: Error {}

    private static func alloc(_ device: MTLDevice, _ bytes: Int) throws -> MTLBuffer {
        guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            Issue.record("alloc failed (\(bytes) bytes)")
            throw BufferAllocFailure()
        }
        return buf
    }

    // MARK: - Embedding

    /// Small full-table check at the real hidden width: 32 quantized rows,
    /// raw dequant (`outScale = 1.0`), compared against `EmbedLookupRef`.
    private static func runEmbedSmall(seed: UInt64) throws {
        let v = 32
        var rng = SeedTree(seed).key("qwen-embed-small-v\(v)-d\(d)")
        var packed = [UInt8](repeating: 0, count: v * (d / 2))
        var scales = [UInt16](repeating: 0, count: v * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: v * groupsPerRow)
        for row in 0..<v {
            let q = Quantization.quantizeInt4Affine(
                (0..<d).map { _ in rng.uniform(-1.0, 1.0) })
            for i in 0..<(d / 2) { packed[row * (d / 2) + i] = q.packed[i] }
            for g in 0..<groupsPerRow {
                scales[row * groupsPerRow + g] = q.scales[g]
                biases[row * groupsPerRow + g] = q.biases[g]
            }
        }

        let ctx = try MetalContext()
        let kernel = try EmbedLookupInt4(context: ctx)
        guard let tableBuf = ctx.device.makeBuffer(
                    bytes: packed, length: packed.count, options: .storageModeShared),
              let scalesBuf = ctx.device.makeBuffer(
                    bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                    options: .storageModeShared),
              let biasesBuf = ctx.device.makeBuffer(
                    bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                    options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: d) else {
            Issue.record("alloc failed"); return
        }
        let token: UInt32 = 31
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb,
                      table: tableBuf, scales: scalesBuf, biases: biasesBuf,
                      out: outBuf,
                      tokenId: token, d: UInt32(d), outScale: 1.0)
        cb.commit(); cb.waitUntilCompleted()

        let ref = EmbedLookupRef.applyInt4(
            tablePacked: packed, tableScales: scales, tableBiases: biases,
            tokenId: Int(token), d: d, outScale: 1.0)
        let actual = Fp16Buffer.read(outBuf, count: d)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.quantInt4, "embed small d=\(d): rel=\(rel)")
    }

    @Test func embedLookupInt4_smallTable_d2048_unitScale() throws {
        try Self.runEmbedSmall(seed: 0xE01)
    }

    /// The production claim: vocabulary 248,320 at D = 2,048, last-row token
    /// first (max row offset), first and an interior token as controls. Only
    /// the queried rows are written; the kernel touches exactly those rows.
    @Test func embedLookupInt4_fullVocabulary_edgeTokens() throws {
        let ctx = try MetalContext()
        let device = ctx.device
        let kernel = try EmbedLookupInt4(context: ctx)

        let tableBytes = Self.vocab * (Self.d / 2)
        let scaleBytes = Self.vocab * Self.groupsPerRow * MemoryLayout<UInt16>.size
        let tableBuf  = try Self.alloc(device, tableBytes)
        let scalesBuf = try Self.alloc(device, scaleBytes)
        let biasesBuf = try Self.alloc(device, scaleBytes)

        for token in [0, 123_457, Self.vocab - 1] {
            var rng = SeedTree(0xE02).key("qwen-embed-full-token\(token)")
            let row = Quantization.quantizeInt4Affine(
                (0..<Self.d).map { _ in rng.uniform(-1.0, 1.0) })
            // Shared storage: write the queried row straight through contents().
            let rowParamBytes = Self.groupsPerRow * MemoryLayout<UInt16>.size
            row.packed.withUnsafeBytes { src in
                tableBuf.contents().advanced(by: token * (Self.d / 2))
                    .copyMemory(from: src.baseAddress!, byteCount: src.count)
            }
            row.scales.withUnsafeBytes { src in
                scalesBuf.contents().advanced(by: token * rowParamBytes)
                    .copyMemory(from: src.baseAddress!, byteCount: src.count)
            }
            row.biases.withUnsafeBytes { src in
                biasesBuf.contents().advanced(by: token * rowParamBytes)
                    .copyMemory(from: src.baseAddress!, byteCount: src.count)
            }

            let outBuf = Fp16Buffer.make(device, count: Self.d)
            guard outBuf != nil else { Issue.record("alloc failed"); return }
            let cb = ctx.queue.makeCommandBuffer()!
            kernel.encode(commandBuffer: cb,
                          table: tableBuf, scales: scalesBuf, biases: biasesBuf,
                          out: outBuf!,
                          tokenId: UInt32(token), d: UInt32(Self.d), outScale: 1.0)
            cb.commit(); cb.waitUntilCompleted()

            // Row-only reference: the queried row becomes a one-row table.
            let ref = EmbedLookupRef.applyInt4(
                tablePacked: row.packed, tableScales: row.scales,
                tableBiases: row.biases,
                tokenId: 0, d: Self.d, outScale: 1.0)
            let actual = Fp16Buffer.read(outBuf!, count: Self.d)
            let rel = RelError.compute(actual: actual, reference: ref)
            #expect(rel < Tolerance.quantInt4,
                    "embed full V=\(Self.vocab) token=\(token): rel=\(rel)")
        }
    }

    // MARK: - Head

    /// Small full-matrix check at the real hidden width: 64 quantized lm_head
    /// rows, every logit compared against `DequantInt4GemvRef`. m = 64 keeps
    /// the kernel on the generic (non-specialized) pipeline, exactly like the
    /// production shape below.
    @Test func dequantInt4Gemv_head_small_d2048() throws {
        let m = 64
        let n = Self.d
        var rng = SeedTree(0xE11).key("qwen-head-small-m\(m)-n\(n)")
        let rows = (0..<m).map { _ in
            Quantization.quantizeInt4Affine(
                (0..<n).map { _ in rng.uniform(-0.5, 0.5) })
        }
        let packed = rows.flatMap(\.packed)
        let scales = rows.flatMap(\.scales)
        let biases = rows.flatMap(\.biases)

        let xFp16 = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let xRef  = xFp16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try DequantInt4GEMV(context: ctx)
        guard let wBuf = ctx.device.makeBuffer(
                    bytes: packed, length: packed.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                    bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                    options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(
                    bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                    options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: m) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb,
                      weights: wBuf, scales: sBuf, biases: bBuf,
                      x: xBuf, y: yBuf,
                      m: UInt32(m), n: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()

        let ref = DequantInt4GemvRef.apply(weightRows: rows, x: xRef, n: n)
        let actual = Fp16Buffer.read(yBuf, count: m)
        let rel = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(rel < Tolerance.fp16Reduction,
                "head small m=\(m) n=\(n): rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// The production head: lm_head as m = 248,320 rows × n = 2,048 on the
    /// generic `dequant_int4_gemv_simd` pipeline (the specialized pipelines
    /// cover only Gemma decode shapes), FP16 logits out. The whole table is
    /// allocated; four rows (first, second, middle, last) are reconstructed
    /// from the buffer bytes and checked on CPU — per the card, a CPU subset
    /// of rows is enough for the head.
    ///
    /// Weights are raw random nibble patterns (valid INT4 payloads); scales
    /// and biases are well-behaved BF16 uniforms so no row dequantizes to
    /// NaN or overflows FP16 in the dot product.
    @Test func dequantInt4Gemv_head_realShape_subsetRows() throws {
        let m = Self.vocab
        let n = Self.d
        let ctx = try MetalContext()
        let kernel = try DequantInt4GEMV(context: ctx)

        let tableBuf  = try Self.alloc(ctx.device, m * (n / 2))
        let scalesBuf = try Self.alloc(
            ctx.device, m * Self.groupsPerRow * MemoryLayout<UInt16>.size)
        let biasesBuf = try Self.alloc(
            ctx.device, m * Self.groupsPerRow * MemoryLayout<UInt16>.size)

        var rng = SeedTree(0xE12).key("qwen-head-full-m\(m)-n\(n)")
        let wordCount = m * (n / 2) / MemoryLayout<UInt64>.size
        let wPtr = tableBuf.contents().bindMemory(to: UInt64.self,
                                                  capacity: wordCount)
        for i in 0..<wordCount { wPtr[i] = rng.next() }
        let paramCount = m * Self.groupsPerRow
        let sPtr = scalesBuf.contents().bindMemory(to: UInt16.self,
                                                   capacity: paramCount)
        let bPtr = biasesBuf.contents().bindMemory(to: UInt16.self,
                                                   capacity: paramCount)
        for i in 0..<paramCount {
            sPtr[i] = Quantization.bf16Bits(rng.uniform(0.002, 0.05))
        }
        for i in 0..<paramCount {
            bPtr[i] = Quantization.bf16Bits(rng.uniform(-0.05, 0.05))
        }

        let xFp16 = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let xRef  = xFp16.map { Float($0) }
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: m) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb,
                      weights: tableBuf, scales: scalesBuf, biases: biasesBuf,
                      x: xBuf, y: yBuf,
                      m: UInt32(m), n: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        // Rebuild the probed rows from the exact bytes the kernel read.
        let wRows = tableBuf.contents().bindMemory(to: UInt8.self,
                                                   capacity: m * (n / 2))
        let probeRows = [0, 1, m / 2, m - 1]
        var ref: [Float] = []
        ref.reserveCapacity(probeRows.count)
        for row in probeRows {
            let packed = (0..<(n / 2)).map { wRows[row * (n / 2) + $0] }
            let scales = (0..<Self.groupsPerRow).map { sPtr[row * Self.groupsPerRow + $0] }
            let biases = (0..<Self.groupsPerRow).map { bPtr[row * Self.groupsPerRow + $0] }
            let q = Quantization.Int4AffineRow(packed: packed,
                                               scales: scales, biases: biases)
            ref += DequantInt4GemvRef.apply(weightRows: [q], x: xRef, n: n)
        }
        let logits = Fp16Buffer.read(yBuf, count: m)
        let actual = probeRows.map { logits[$0] }
        let rel = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(rel < Tolerance.fp16Reduction,
                "head m=\(m) n=\(n) rows \(probeRows): rel=\(rel) maxAbs=\(maxAbs)")
    }
}
