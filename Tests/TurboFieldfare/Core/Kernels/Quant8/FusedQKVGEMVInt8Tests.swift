import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Validates `dequant_int8_qkv_gemv_simd` against the CPU reference and
/// against three separate `DequantInt8GEMV` dispatches. The two GPU paths sum
/// in different orders, so they agree within FP16 reduction tolerance rather
/// than bit for bit.
@Suite struct FusedQKVGEMVInt8Tests {
    private struct Result {
        let q: [Float]
        let k: [Float]
        let v: [Float]
        let qBytes: [UInt8]
        let kBytes: [UInt8]
        let vBytes: [UInt8]
    }

    private static func cpu(_ m: Int8DenseFixtures.Matrix, x: [Float]) -> [Float] {
        m.paddedWeights.withUnsafeBufferPointer { weights in
            m.scales.withUnsafeBufferPointer { scales in
                m.biases.withUnsafeBufferPointer { biases in
                    DequantInt8PackedGemvRef.apply(
                        weights: weights.baseAddress! + m.weightOffset,
                        scales: scales.baseAddress!,
                        biases: biases.baseAddress!,
                        rows: m.rowCount,
                        x: x,
                        n: m.n)
                }
            }
        }
    }

    /// Runs the fused kernel; `v == nil` binds K again as V, as the runner
    /// does for full-attention layers.
    private static func fused(context: MetalContext,
                              q: Int8DenseFixtures.Matrix,
                              k: Int8DenseFixtures.Matrix,
                              v: Int8DenseFixtures.Matrix?,
                              x: [Float16]) throws -> Result {
        let device = context.device
        let kernel = try FusedQKVGEMVInt8(context: context)
        let qBuffers = try Int8DenseFixtures.buffers(q, device: device)
        let kBuffers = try Int8DenseFixtures.buffers(k, device: device)
        let vBuffers = try v.map { try Int8DenseFixtures.buffers($0, device: device) } ?? kBuffers
        let xBuffer = try Int8DenseFixtures.halfBuffer(x, device: device)
        let qRows = q.rowCount
        let kvRows = k.rowCount
        // Outputs land at a non-zero offset, as K/V do in the cache slots.
        let outPad = 3
        let qOut = try Int8DenseFixtures.halfBuffer(count: outPad + qRows, device: device)
        let kOut = try Int8DenseFixtures.halfBuffer(count: outPad + kvRows, device: device)
        let vOut = try Int8DenseFixtures.halfBuffer(count: outPad + kvRows, device: device)
        let outOffset = outPad * MemoryLayout<Float16>.size

        try Int8DenseFixtures.run(context) { cb in
            kernel.encode(commandBuffer: cb,
                          qWeights: qBuffers.weights, qWeightsOffset: qBuffers.weightOffset,
                          qScales: qBuffers.scales, qBiases: qBuffers.biases,
                          kWeights: kBuffers.weights, kWeightsOffset: kBuffers.weightOffset,
                          kScales: kBuffers.scales, kBiases: kBuffers.biases,
                          vWeights: vBuffers.weights, vWeightsOffset: vBuffers.weightOffset,
                          vScales: vBuffers.scales, vBiases: vBuffers.biases,
                          x: xBuffer,
                          qOut: qOut, qOutOffset: outOffset,
                          kOut: kOut, kOutOffset: outOffset,
                          vOut: vOut, vOutOffset: outOffset,
                          qRows: UInt32(qRows),
                          kvRows: UInt32(kvRows),
                          n: UInt32(q.n))
        }
        func read(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
            Array(Fp16Buffer.read(buffer, count: outPad + count).dropFirst(outPad))
        }
        return Result(q: read(qOut, qRows), k: read(kOut, kvRows), v: read(vOut, kvRows),
                      qBytes: Int8DenseFixtures.halfBytes(qOut, offset: outOffset, count: qRows),
                      kBytes: Int8DenseFixtures.halfBytes(kOut, offset: outOffset, count: kvRows),
                      vBytes: Int8DenseFixtures.halfBytes(vOut, offset: outOffset, count: kvRows))
    }

    private static func separateGEMV(context: MetalContext,
                                     _ m: Int8DenseFixtures.Matrix,
                                     x: [Float16]) throws -> [Float] {
        let device = context.device
        let gemv = try DequantInt8GEMV(context: context)
        let buffers = try Int8DenseFixtures.buffers(m, device: device)
        let xBuffer = try Int8DenseFixtures.halfBuffer(x, device: device)
        let y = try Int8DenseFixtures.halfBuffer(count: m.rowCount, device: device)
        try Int8DenseFixtures.run(context) { cb in
            gemv.encode(commandBuffer: cb,
                        weights: buffers.weights, weightsOffset: buffers.weightOffset,
                        scales: buffers.scales, biases: buffers.biases,
                        x: xBuffer, y: y,
                        m: UInt32(m.rowCount), n: UInt32(m.n))
        }
        return Fp16Buffer.read(y, count: m.rowCount)
    }

    private static func expectClose(_ actual: [Float], _ reference: [Float],
                                    _ label: String,
                                    tolerance: Float = Tolerance.fp16Reduction) {
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < tolerance,
                "\(label): rel=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, reference))")
    }

    private static func checkSeparateProjections(qRows: Int, kvRows: Int, n: Int,
                                                 seed: UInt64,
                                                 weightOffset: Int) throws {
        var rng = SplitMix64(seed: seed)
        let q = Int8DenseFixtures.matrix(rows: qRows, n: n, rng: &rng,
                                         range: -0.5...0.5, weightOffset: weightOffset)
        let k = Int8DenseFixtures.matrix(rows: kvRows, n: n, rng: &rng,
                                         range: -0.5...0.5, weightOffset: weightOffset)
        let v = Int8DenseFixtures.matrix(rows: kvRows, n: n, rng: &rng,
                                         range: -0.5...0.5, weightOffset: weightOffset)
        let x = Int8DenseFixtures.randomHalves(count: n, rng: &rng)
        let xRef = x.map(Float.init)

        let context = try MetalContext()
        let result = try Self.fused(context: context, q: q, k: k, v: v, x: x)
        let label = "q=\(qRows) kv=\(kvRows) n=\(n) offset=\(weightOffset)"
        Self.expectClose(result.q, Self.cpu(q, x: xRef), "\(label) Q vs CPU")
        Self.expectClose(result.k, Self.cpu(k, x: xRef), "\(label) K vs CPU")
        Self.expectClose(result.v, Self.cpu(v, x: xRef), "\(label) V vs CPU")
        Self.expectClose(result.q, try Self.separateGEMV(context: context, q, x: x),
                         "\(label) Q vs DequantInt8GEMV")
        Self.expectClose(result.k, try Self.separateGEMV(context: context, k, x: x),
                         "\(label) K vs DequantInt8GEMV")
        Self.expectClose(result.v, try Self.separateGEMV(context: context, v, x: x),
                         "\(label) V vs DequantInt8GEMV")
    }

    @Test func smallShapeMatchesReferenceAndSeparateGEMVs() throws {
        try Self.checkSeparateProjections(qRows: 64, kvRows: 32, n: 128,
                                          seed: 0x8A01, weightOffset: 0)
    }

    /// Sliding-window layer shape (q 4096, k = v 2048, N 2816), weights bound
    /// at a 4-but-not-16-aligned offset.
    @Test func slidingWindowProductionShapeAtFourAlignedOffset() throws {
        try Self.checkSeparateProjections(qRows: 4096, kvRows: 2048, n: 2816,
                                          seed: 0x8A02,
                                          weightOffset: Int8DenseFixtures.offsetFourNotSixteen)
    }

    /// Full-attention layer shape (q 8192, k 1024, N 2816) without v_proj:
    /// K is bound again as V, so V must equal K bit for bit.
    @Test func fullAttentionProductionShapeReusesKAsV() throws {
        var rng = SplitMix64(seed: 0x8A03)
        let offset = Int8DenseFixtures.offsetFourNotSixteen
        let q = Int8DenseFixtures.matrix(rows: 8192, n: 2816, rng: &rng,
                                         range: -0.5...0.5, weightOffset: offset)
        let k = Int8DenseFixtures.matrix(rows: 1024, n: 2816, rng: &rng,
                                         range: -0.5...0.5, weightOffset: offset)
        let x = Int8DenseFixtures.randomHalves(count: 2816, rng: &rng)
        let xRef = x.map(Float.init)

        let context = try MetalContext()
        let result = try Self.fused(context: context, q: q, k: k, v: nil, x: x)
        #expect(result.vBytes == result.kBytes)
        Self.expectClose(result.q, Self.cpu(q, x: xRef), "full Q vs CPU")
        Self.expectClose(result.k, Self.cpu(k, x: xRef), "full K vs CPU")
    }

    /// Row counts that do not fill the last threadgroup and group counts that
    /// leave a remainder after the 4-group blocks (3, 5, 11 groups).
    @Test(arguments: [64, 192, 320, 704, 2112])
    func remainderGroupsAndPartialThreadgroups(n: Int) throws {
        try Self.checkSeparateProjections(qRows: 40, kvRows: 12, n: n,
                                          seed: 0x8A10 &+ UInt64(n),
                                          weightOffset: 4)
    }
}
