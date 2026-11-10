import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// S17: MoE block kernels for Qwen 3.5 — `qwen_moe_gate_up_silu` (SiLU
/// multiply from INT4 gate/up rows of a C5 expert blob), `qwen_moe_down`
/// (INT4 down projection) and `qwen_moe_residual` (weighted sum of the 8
/// routed experts, `sg * shared`, summed onto the residual).
///
/// Experts arrive as synthetic blob buffers (no streaming): each blob is
/// exactly the C5 layout — gate, up, down x weight, scales, biases — at the
/// offsets `QwenMoE.expertOffsets` computes, and the kernels get them through
/// an argument buffer like the real `PreadExpertStreamer` slot buffers.
/// Weights are quantized with `Quantization.quantizeInt4Affine`, so the CPU
/// reference (`QwenMoERef`) sees the very same INT4 payload. Tolerances per
/// contract C8: `fp16Reduction` per GEMV stage, `fp16ChainedReduction` for
/// the chained block.
@Suite struct QwenMoETests {

    private struct BufferAllocFailure: Error {}

    // MARK: - Synthetic C5 blobs

    private static func alloc(_ device: MTLDevice, _ bytes: Int) throws -> MTLBuffer {
        guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            Issue.record("alloc failed (\(bytes) bytes)")
            throw BufferAllocFailure()
        }
        return buf
    }

    /// `count` INT4 rows over `n` columns, quantized from uniform noise.
    private static func randomRows(_ count: Int, _ n: Int,
                                   rng: inout SplitMix64) -> [Quantization.Int4AffineRow]
    {
        (0..<count).map { _ in
            Quantization.quantizeInt4Affine((0..<n).map { _ in rng.uniform(-0.5, 0.5) })
        }
    }

    /// One expert blob in C5 order: gate, up, down x weight, scales, biases.
    private static func makeExpertBlob(
        gate: [Quantization.Int4AffineRow],
        up: [Quantization.Int4AffineRow],
        down: [Quantization.Int4AffineRow],
        hidden: Int, inter: Int,
        offsets: QwenMoE.ExpertOffsets
    ) -> [UInt8] {
        var blob = [UInt8](repeating: 0,
                           count: QwenMoE.expertBlobBytes(hidden: hidden, inter: inter))
        func putBytes(_ bytes: [UInt8], at off: UInt32) {
            for (i, b) in bytes.enumerated() { blob[Int(off) + i] = b }
        }
        func putHalves(_ values: [UInt16], at off: UInt32) {  // BF16, little-endian
            for (i, v) in values.enumerated() {
                blob[Int(off) + 2 * i] = UInt8(v & 0xFF)
                blob[Int(off) + 2 * i + 1] = UInt8(v >> 8)
            }
        }
        func putMatrix(_ rows: [Quantization.Int4AffineRow],
                       wOff: UInt32, sOff: UInt32, bOff: UInt32)
        {
            let rowBytes = rows[0].packed.count
            let rowGroups = rows[0].scales.count
            for (r, row) in rows.enumerated() {
                putBytes(row.packed, at: wOff + UInt32(r * rowBytes))
                putHalves(row.scales, at: sOff + UInt32(r * rowGroups * 2))
                putHalves(row.biases, at: bOff + UInt32(r * rowGroups * 2))
            }
        }
        putMatrix(gate, wOff: offsets.gateWOff, sOff: offsets.gateSOff, bOff: offsets.gateBOff)
        putMatrix(up,   wOff: offsets.upWOff,   sOff: offsets.upSOff,   bOff: offsets.upBOff)
        putMatrix(down, wOff: offsets.downWOff, sOff: offsets.downSOff, bOff: offsets.downBOff)
        return blob
    }

    private static func readFloat(_ buf: MTLBuffer, _ count: Int) -> [Float] {
        let p = buf.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { p[$0] }
    }

    // MARK: - Stage 1a: gate/up SiLU multiply from the blob

    private static func runGateUpSilu(hidden: Int, inter: Int, slots: Int,
                                      seed: UInt64, label: String) throws {
        var rng = SeedTree(seed).key(label)
        let offsets = QwenMoE.expertOffsets(hidden: hidden, inter: inter)

        var blobs: [MTLBuffer] = []
        var gateRows: [[Quantization.Int4AffineRow]] = []
        var upRows: [[Quantization.Int4AffineRow]] = []
        let ctx = try MetalContext()
        for _ in 0..<slots {
            let g = randomRows(inter, hidden, rng: &rng)
            let u = randomRows(inter, hidden, rng: &rng)
            gateRows.append(g)
            upRows.append(u)
            let blob = makeExpertBlob(gate: g, up: u,
                                      down: randomRows(hidden, inter, rng: &rng),
                                      hidden: hidden, inter: inter, offsets: offsets)
            blobs.append(try Self.alloc(ctx.device, blob.count))
            blobs.last!.contents().copyMemory(from: blob, byteCount: blob.count)
        }

        let xFp32 = (0..<hidden).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16) else {
            Issue.record("alloc failed"); throw BufferAllocFailure()
        }
        let actsBuf = try Self.alloc(ctx.device, slots * inter * 2)
        let moe = try QwenMoE(context: ctx)
        guard let arg = moe.makeBlobsArgumentBuffer(blobs: blobs, slots: slots) else {
            Issue.record("arg buffer alloc failed"); throw BufferAllocFailure()
        }
        let cb = ctx.queue.makeCommandBuffer()!
        moe.encodeGateUpSilu(commandBuffer: cb, blobsArgBuffer: arg, blobs: blobs,
                             offsets: offsets, x: xBuf, acts: actsBuf,
                             hidden: hidden, inter: inter, slots: slots)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(actsBuf, count: slots * inter)
        var ref: [Float] = []
        ref.reserveCapacity(slots * inter)
        for k in 0..<slots {
            ref += QwenMoERef.gateUpSilu(gate: gateRows[k], up: upRows[k],
                                         x: xRef, hidden: hidden).map { Float($0) }
        }
        let rel = RelError.compute(actual: actual, reference: ref)
        print("\(label): rel=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
        #expect(rel < Tolerance.fp16Reduction,
                "\(label): rel=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
    }

    @Test func gateUpSilu_small() throws {
        // 2 experts, hidden 128, inter 64.
        try Self.runGateUpSilu(hidden: 128, inter: 64, slots: 2,
                               seed: 0xA1, label: "gateup-small")
    }

    @Test func gateUpSilu_real() throws {
        // 8 routed experts at the real 3.4 shapes: hidden 2048, inter 512.
        try Self.runGateUpSilu(hidden: 2048, inter: 512, slots: 8,
                               seed: 0xA2, label: "gateup-real")
    }

    // MARK: - Stage 1b: down projection from synthetic acts

    private static func runDown(hidden: Int, inter: Int, slots: Int,
                                seed: UInt64, label: String) throws {
        var rng = SeedTree(seed).key(label)
        let offsets = QwenMoE.expertOffsets(hidden: hidden, inter: inter)

        var blobs: [MTLBuffer] = []
        var downRows: [[Quantization.Int4AffineRow]] = []
        let ctx = try MetalContext()
        for _ in 0..<slots {
            // gate/up unused here; down rows are the payload under test.
            let d = randomRows(hidden, inter, rng: &rng)
            downRows.append(d)
            let blob = makeExpertBlob(gate: randomRows(inter, hidden, rng: &rng),
                                      up: randomRows(inter, hidden, rng: &rng),
                                      down: d,
                                      hidden: hidden, inter: inter, offsets: offsets)
            blobs.append(try Self.alloc(ctx.device, blob.count))
            blobs.last!.contents().copyMemory(from: blob, byteCount: blob.count)
        }

        // Synthetic FP16 activations per slot (no streaming, no stage 1a).
        let actsFp16 = (0..<(slots * inter)).map { _ in
            Float16(rng.uniform(-1.0, 1.0))
        }
        guard let actsBuf = Fp16Buffer.make(ctx.device, halves: actsFp16) else {
            Issue.record("alloc failed"); throw BufferAllocFailure()
        }
        let outBuf = try Self.alloc(ctx.device, slots * hidden * 2)
        let moe = try QwenMoE(context: ctx)
        guard let arg = moe.makeBlobsArgumentBuffer(blobs: blobs, slots: slots) else {
            Issue.record("arg buffer alloc failed"); throw BufferAllocFailure()
        }
        let cb = ctx.queue.makeCommandBuffer()!
        moe.encodeDown(commandBuffer: cb, blobsArgBuffer: arg, blobs: blobs,
                       offsets: offsets, acts: actsBuf, routedOut: outBuf,
                       hidden: hidden, inter: inter, slots: slots)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(outBuf, count: slots * hidden)
        var ref: [Float] = []
        ref.reserveCapacity(slots * hidden)
        for k in 0..<slots {
            let act = Array(actsFp16[(k * inter)..<(k * inter + inter)])
            ref += QwenMoERef.down(downRows: downRows[k], act: act, inter: inter)
                .map { Float($0) }
        }
        let rel = RelError.compute(actual: actual, reference: ref)
        print("\(label): rel=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
        #expect(rel < Tolerance.fp16Reduction,
                "\(label): rel=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
    }

    @Test func down_small() throws {
        try Self.runDown(hidden: 128, inter: 64, slots: 2,
                         seed: 0xA3, label: "down-small")
    }

    @Test func down_real() throws {
        try Self.runDown(hidden: 2048, inter: 512, slots: 8,
                         seed: 0xA4, label: "down-real")
    }

    // MARK: - Step 2: shared expert (one-slot pass) + S13 scalar gate

    /// The shared expert runs the same two kernels with `slots = 1` over one
    /// resident blob; `sg = sigmoid(W_sg * x)` comes from the S13 router
    /// chain (`QwenRouter.encodeSigmoid`, FP16 [1]). Checks the full
    /// SwiGLU chain against the reference and the gate against a CPU
    /// sigmoid.
    @Test func sharedExpert_real() throws {
        let hidden = 2048, inter = 512
        var rng = SeedTree(0xA5).key("shared-real")
        let offsets = QwenMoE.expertOffsets(hidden: hidden, inter: inter)

        let g = Self.randomRows(inter, hidden, rng: &rng)
        let u = Self.randomRows(inter, hidden, rng: &rng)
        let d = Self.randomRows(hidden, inter, rng: &rng)
        let ctx = try MetalContext()
        let blob = Self.makeExpertBlob(gate: g, up: u, down: d,
                                  hidden: hidden, inter: inter, offsets: offsets)
        let blobBuf = try Self.alloc(ctx.device, blob.count)
        blobBuf.contents().copyMemory(from: blob, byteCount: blob.count)

        let xFp16 = (0..<hidden).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let xRef = xFp16.map { Float($0) }
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16) else {
            Issue.record("alloc failed"); throw BufferAllocFailure()
        }
        let actsBuf = try Self.alloc(ctx.device, inter * 2)
        let sharedOutBuf = try Self.alloc(ctx.device, hidden * 2)

        // S13 scalar gate: one FP16 logit -> qwen_sigmoid_f16 -> sg.
        let logit = Float16(rng.uniform(-4.0, 4.0))
        guard let logitBuf = Fp16Buffer.make(ctx.device, halves: [logit]),
              let sgBuf = Fp16Buffer.make(ctx.device, count: 1) else {
            Issue.record("alloc failed"); throw BufferAllocFailure()
        }

        let moe = try QwenMoE(context: ctx)
        let router = try QwenRouter(context: ctx)
        guard let arg = moe.makeBlobsArgumentBuffer(blobs: [blobBuf], slots: 1) else {
            Issue.record("arg buffer alloc failed"); throw BufferAllocFailure()
        }
        let cb = ctx.queue.makeCommandBuffer()!
        moe.encodeGateUpSilu(commandBuffer: cb, blobsArgBuffer: arg, blobs: [blobBuf],
                             offsets: offsets, x: xBuf, acts: actsBuf,
                             hidden: hidden, inter: inter, slots: 1)
        moe.encodeDown(commandBuffer: cb, blobsArgBuffer: arg, blobs: [blobBuf],
                       offsets: offsets, acts: actsBuf, routedOut: sharedOutBuf,
                       hidden: hidden, inter: inter, slots: 1)
        router.encodeSigmoid(commandBuffer: cb, x: logitBuf, out: sgBuf, count: 1)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(sharedOutBuf, count: hidden)
        let act = QwenMoERef.gateUpSilu(gate: g, up: u, x: xRef, hidden: hidden)
        let ref = QwenMoERef.down(downRows: d, act: act, inter: inter).map { Float($0) }
        let rel = RelError.compute(actual: actual, reference: ref)
        print("shared-real: rel=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
        #expect(rel < Tolerance.fp16Reduction,
                "shared-real: rel=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, ref))")

        let sgActual = Fp16Buffer.readHalf(sgBuf, count: 1)[0]
        let sgRef = Float16(1.0 / (1.0 + exp(-Float(logit))))
        #expect(abs(Float(sgActual) - Float(sgRef)) < 1e-3,
                "sg kernel=\(sgActual) ref=\(sgRef) logit=\(logit)")
    }

    /// The C5 offsets must reproduce the byte sizes of contract 3.4 for the
    /// real shapes: 589.824 B per matrix, 1.769.472 B per expert blob.
    @Test func expertBlobLayout_matchesPlan34() {
        #expect(QwenMoE.expertBlobBytes(hidden: 2048, inter: 512) == 1_769_472)
        let o = QwenMoE.expertOffsets(hidden: 2048, inter: 512)
        #expect(o.gateWOff == 0)
        #expect(o.gateSOff == 524_288)
        #expect(o.gateBOff == 557_056)
        #expect(o.upWOff == 589_824)
        #expect(o.downWOff == 1_179_648)
        #expect(o.downBOff == 1_736_704)
    }
}
