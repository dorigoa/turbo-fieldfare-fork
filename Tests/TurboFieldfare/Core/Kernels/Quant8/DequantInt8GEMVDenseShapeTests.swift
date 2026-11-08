import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Validates `DequantInt8GEMV` on the `affine8` dense decode shapes that now
/// have specialized pipelines: `o_proj` for sliding-window (N = 4096) and
/// full-attention (N = 8192) layers. The logits head shape (M = 262,144) is
/// covered by `Int8DenseFullVocabularyTests`.
@Suite struct DequantInt8GEMVDenseShapeTests {
    /// Builds one matrix and checks it at every weight offset in `offsets`.
    private static func runAndCompare(m: Int, n: Int, seed: UInt64,
                                      offsets: [Int]) throws {
        var rng = SeedTree(seed).key("int8-dense-gemv-m\(m)-n\(n)")
        let base = Int8DenseFixtures.matrix(rows: m, n: n, rng: &rng)
        let x = Int8DenseFixtures.randomHalves(count: n, rng: &rng)
        let reference = base.paddedWeights.withUnsafeBufferPointer { weights in
            base.scales.withUnsafeBufferPointer { scales in
                base.biases.withUnsafeBufferPointer { biases in
                    DequantInt8PackedGemvRef.apply(
                        weights: weights.baseAddress!,
                        scales: scales.baseAddress!,
                        biases: biases.baseAddress!,
                        rows: m,
                        x: x.map(Float.init),
                        n: n)
                }
            }
        }

        let context = try MetalContext()
        let gemv = try DequantInt8GEMV(context: context)
        let xBuffer = try Int8DenseFixtures.halfBuffer(x, device: context.device)
        for offset in offsets {
            let matrix = Int8DenseFixtures.rebased(base, weightOffset: offset)
            let buffers = try Int8DenseFixtures.buffers(matrix, device: context.device)
            let y = try Int8DenseFixtures.halfBuffer(count: m, device: context.device)
            try Int8DenseFixtures.run(context) { cb in
                gemv.encode(commandBuffer: cb,
                            weights: buffers.weights, weightsOffset: buffers.weightOffset,
                            scales: buffers.scales, biases: buffers.biases,
                            x: xBuffer, y: y,
                            m: UInt32(m), n: UInt32(n))
            }
            let actual = Fp16Buffer.read(y, count: m)
            let rel = RelError.compute(actual: actual, reference: reference)
            let maxAbs = RelError.maxAbsDiff(actual, reference)
            #expect(rel < Tolerance.fp16Reduction,
                    "M=\(m) N=\(n) offset=\(offset): rel=\(rel) maxAbs=\(maxAbs)")
        }
    }

    /// Sliding-window `o_proj`, at a 4-but-not-16-aligned offset and at an odd
    /// one: the byte-wise kernel accepts any weight offset.
    @Test func oProjSlidingWindowShape() throws {
        try Self.runAndCompare(m: 2816, n: 4096, seed: 0x8B01,
                               offsets: [Int8DenseFixtures.offsetFourNotSixteen, 3])
    }

    @Test func oProjFullAttentionShape() throws {
        try Self.runAndCompare(m: 2816, n: 8192, seed: 0x8B02,
                               offsets: [Int8DenseFixtures.offsetFourNotSixteen])
    }
}
