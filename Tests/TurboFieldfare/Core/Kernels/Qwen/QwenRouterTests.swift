import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// S13: MoE router chain for Qwen 3.5 — `DequantInt8GEMV` (W_gate, 256 rows
/// × 2048 cols) feeds FP16 logits into the new `qwen_router_topk` kernel
/// (softmax FP32 over 256, top-8, renormalization; ties -> lowest index).
/// Compared against `QwenHeadRef.softmaxTopK`, which runs the identical
/// greedy algorithm but with CPU libm `exp` and serial FP sums.
///
/// The tie tests pin the documented selection rule: with exact FP16 ties the
/// surviving order must be by ascending expert index, including right at the
/// top-K boundary. The last tests cover the shared-expert scalar gate:
/// `DequantInt8GEMV` at m = 1 (W_sg, 1 × 2048) plus `qwen_sigmoid_f16`,
/// against `QwenHeadRef.sigmoid`. Tolerances per contract C8.
@Suite struct QwenRouterTests {

    private struct BufferAllocFailure: Error {}

    private static func alloc(_ device: MTLDevice, _ bytes: Int) throws -> MTLBuffer {
        guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            Issue.record("alloc failed (\(bytes) bytes)")
            throw BufferAllocFailure()
        }
        return buf
    }

    private static func readUInt32(_ buf: MTLBuffer, _ count: Int) -> [Int] {
        let p = buf.contents().bindMemory(to: UInt32.self, capacity: count)
        return (0..<count).map { Int(p[$0]) }
    }

    private static func readFloat(_ buf: MTLBuffer, _ count: Int) -> [Float] {
        let p = buf.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { p[$0] }
    }

    /// Runs `qwen_router_topk` over FP16 logits and returns the raw outputs.
    private static func runTopK(logits: [Float16], topK: Int)
        throws -> (indices: [Int], weights: [Float])
    {
        let experts = logits.count
        let ctx = try MetalContext()
        let router = try QwenRouter(context: ctx)
        guard let logitsBuf = Fp16Buffer.make(ctx.device, halves: logits) else {
            Issue.record("alloc failed")
            throw BufferAllocFailure()
        }
        let idxBuf  = try Self.alloc(ctx.device, topK * MemoryLayout<UInt32>.size)
        let wBuf    = try Self.alloc(ctx.device, topK * MemoryLayout<Float>.size)

        let cb = ctx.queue.makeCommandBuffer()!
        router.encodeTopK(commandBuffer: cb,
                          logits: logitsBuf, indices: idxBuf, weights: wBuf,
                          experts: experts, topK: topK)
        cb.commit(); cb.waitUntilCompleted()
        return (Self.readUInt32(idxBuf, topK), Self.readFloat(wBuf, topK))
    }

    // MARK: - Small shapes

    @Test func routerTopK_small16_top4() throws {
        var rng = SeedTree(0xF01).key("qwen-router-small-e16-k4")
        let logits = (0..<16).map { _ in Float16(rng.uniform(-5.0, 5.0)) }
        let (indices, weights) = try Self.runTopK(logits: logits, topK: 4)

        let ref = QwenHeadRef.softmaxTopK(logits: logits.map { Float($0) },
                                          topK: 4)
        #expect(indices == ref.indices,
                "small tie order: got \(indices) want \(ref.indices)")
        let rel = RelError.compute(actual: weights, reference: ref.weights)
        #expect(rel < Tolerance.fp16Reduction, "small weights: rel=\(rel)")
    }

    // MARK: - Tie rule (documented: lowest index wins)

    @Test func routerTopK_exactTies_pickLowestIndex() throws {
        // 8 experts, K = 3: max is index 4 (3.0); then the four-way 2.0 tie
        // at indices 1, 2, 5, 7 must resolve ascending -> 1, then 2.
        let logits: [Float16] = [1, 2, 2, 0, 3, 2, 1, 2]
        let (indices, weights) = try Self.runTopK(logits: logits, topK: 3)
        #expect(indices == [4, 1, 2], "4-way tie: got \(indices)")

        // The renormalized weights must sum to ~1 and follow descending order.
        let sum = weights.reduce(0, +)
        #expect(abs(sum - 1.0) < 1e-5, "weights sum=\(sum)")
        #expect(weights[0] >= weights[1] && weights[1] >= weights[2],
                "weights not descending: \(weights)")

        // Boundary tie: seven logits strictly win (15..9 at indices 0..6),
        // then the 8th slot is a two-way tie at value 7 between indices 8 and
        // 9 (index 7 is a low distractor) — index 8 takes the last slot, 9 is
        // left out. Also catches a naive "first K ascending" selection.
        let boundaryLogits: [Float16] =
            [15, 14, 13, 12, 11, 10, 9, 0, 7, 7, 0, 0, 0, 0, 0, 0]
        let (boundary, _) = try Self.runTopK(logits: boundaryLogits, topK: 8)
        #expect(boundary == [0, 1, 2, 3, 4, 5, 6, 8],
                "boundary tie: got \(boundary)")
    }

    // MARK: - Real shape: W_gate GEMV -> logits -> top-k

    @Test func routerTopK_real256Experts_top8() throws {
        let experts = 256
        let topK = 8
        let n = 2048
        var rng = SeedTree(0xF02).key("qwen-router-real-e\(experts)-n\(n)")

        let rows = (0..<experts).map { _ in
            Quantization.quantizeInt8Affine(
                (0..<n).map { _ in rng.uniform(-0.5, 0.5) })
        }
        let packed = rows.flatMap(\.packed)
        let scales = rows.flatMap(\.scales)
        let biases = rows.flatMap(\.biases)

        let xFp16 = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let xRef  = xFp16.map { Float($0) }

        let ctx = try MetalContext()
        let gemv = try DequantInt8GEMV(context: ctx)   // (256, 2048): generic pso
        let router = try QwenRouter(context: ctx)
        let wBuf = try Self.alloc(ctx.device, packed.count)
        let sBuf = try Self.alloc(ctx.device, scales.count * MemoryLayout<UInt16>.size)
        let bBuf = try Self.alloc(ctx.device, biases.count * MemoryLayout<UInt16>.size)
        // Fill through the shared-storage pointer.
        packed.withUnsafeBytes { src in
            wBuf.contents().copyMemory(from: src.baseAddress!, byteCount: src.count)
        }
        scales.withUnsafeBytes { src in
            sBuf.contents().copyMemory(from: src.baseAddress!, byteCount: src.count)
        }
        biases.withUnsafeBytes { src in
            bBuf.contents().copyMemory(from: src.baseAddress!, byteCount: src.count)
        }
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let logitsBuf = Fp16Buffer.make(ctx.device, count: experts) else {
            Issue.record("alloc failed")
            throw BufferAllocFailure()
        }
        let idxBuf = try Self.alloc(ctx.device, topK * MemoryLayout<UInt32>.size)
        let wOut   = try Self.alloc(ctx.device, topK * MemoryLayout<Float>.size)

        let cb = ctx.queue.makeCommandBuffer()!
        gemv.encode(commandBuffer: cb,
                    weights: wBuf, scales: sBuf, biases: bBuf,
                    x: xBuf, y: logitsBuf,
                    m: UInt32(experts), n: UInt32(n))
        router.encodeTopK(commandBuffer: cb,
                          logits: logitsBuf, indices: idxBuf, weights: wOut,
                          experts: experts, topK: topK)
        cb.commit(); cb.waitUntilCompleted()

        // Reference logits must round-trip through FP16 — that is exactly what
        // the router kernel reads.
        let refLogits = DequantInt8GemvRef.apply(weightRows: rows, x: xRef, n: n)
            .map { Float(Float16($0)) }
        let ref = QwenHeadRef.softmaxTopK(logits: refLogits, topK: topK)

        let indices = Self.readUInt32(idxBuf, topK)
        #expect(indices == ref.indices,
                "real top-8: got \(indices) want \(ref.indices)")
        let weights = Self.readFloat(wOut, topK)
        let rel = RelError.compute(actual: weights, reference: ref.weights)
        #expect(rel < Tolerance.fp16Reduction, "real weights: rel=\(rel)")
    }

    // MARK: - Shared-expert gate: W_sg GEMV (m = 1) + sigmoid

    /// Elementwise coverage of `qwen_sigmoid_f16` across a range wide enough
    /// to straddle the curve's knee and both saturation tails.
    @Test func sigmoid_elementwise_range() throws {
        let count = 33
        let inputs = (0..<count).map {
            Float16(-8.0 + 16.0 * Float($0) / Float(count - 1))
        }
        let ctx = try MetalContext()
        let router = try QwenRouter(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: inputs),
              let yBuf = Fp16Buffer.make(ctx.device, count: count) else {
            Issue.record("alloc failed")
            throw BufferAllocFailure()
        }
        let cb = ctx.queue.makeCommandBuffer()!
        router.encodeSigmoid(commandBuffer: cb, x: xBuf, out: yBuf, count: count)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenHeadRef.sigmoid(inputs.map { Float($0) })
        let actual = Fp16Buffer.read(yBuf, count: count)
        let rel = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(rel < Tolerance.fp16Reduction,
                "sigmoid sweep: rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// The production chain: one-row 8-bit projection `W_sg` (1 × 2048) on
    /// the generic Int8 GEMV pipeline, then the FP16 sigmoid — exactly how
    /// the runner will compute the scalar shared-expert gate sg (plan 3.5).
    @Test func sharedExpertGate_real_m1_n2048() throws {
        let n = 2048
        var rng = SeedTree(0xF03).key("qwen-sgate-n\(n)")
        // Weights small enough that W_sg * x lands in a moderate range:
        // a far-tail sg would be indistinguishable from 0 or 1 in FP16.
        let row = Quantization.quantizeInt8Affine(
            (0..<n).map { _ in rng.uniform(-0.05, 0.05) })

        let xFp16 = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let xRef  = xFp16.map { Float($0) }

        let ctx = try MetalContext()
        let gemv = try DequantInt8GEMV(context: ctx)   // (1, 2048): generic pso
        let router = try QwenRouter(context: ctx)
        let wBuf = try Self.alloc(ctx.device, row.packed.count)
        let sBuf = try Self.alloc(ctx.device, row.scales.count * MemoryLayout<UInt16>.size)
        let bBuf = try Self.alloc(ctx.device, row.biases.count * MemoryLayout<UInt16>.size)
        row.packed.withUnsafeBytes { src in
            wBuf.contents().copyMemory(from: src.baseAddress!, byteCount: src.count)
        }
        row.scales.withUnsafeBytes { src in
            sBuf.contents().copyMemory(from: src.baseAddress!, byteCount: src.count)
        }
        row.biases.withUnsafeBytes { src in
            bBuf.contents().copyMemory(from: src.baseAddress!, byteCount: src.count)
        }
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: 1),
              let sgBuf = Fp16Buffer.make(ctx.device, count: 1) else {
            Issue.record("alloc failed")
            throw BufferAllocFailure()
        }
        let cb = ctx.queue.makeCommandBuffer()!
        gemv.encode(commandBuffer: cb,
                    weights: wBuf, scales: sBuf, biases: bBuf,
                    x: xBuf, y: yBuf,
                    m: 1, n: UInt32(n))
        router.encodeSigmoid(commandBuffer: cb, x: yBuf, out: sgBuf, count: 1)
        cb.commit(); cb.waitUntilCompleted()

        // Reference: dot product in FP32, the FP16 round-trip the GEMV emits,
        // then the sigmoid in FP32.
        let dot = DequantInt8GemvRef.apply(weightRows: [row], x: xRef, n: n)[0]
        let ref = QwenHeadRef.sigmoid([Float(Float16(dot))])[0]
        let sg = Fp16Buffer.read(sgBuf, count: 1)
        let rel = RelError.compute(actual: sg, reference: [ref])
        #expect(rel < Tolerance.fp16Reduction,
                "sg gate: sg=\(sg) ref=\(ref) rel=\(rel)")
    }
}
