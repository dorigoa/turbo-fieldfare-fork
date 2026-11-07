import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Validates the greedy 8-bit LM head (RMSNorm → INT8 rows → argmax) on
/// vocabulary subsets. The production vocabulary (262,144 rows) is covered by
/// `Int8DenseFullVocabularyTests`.
@Suite struct LMHeadChainInt8Tests {
    private static let rmsEps: Float = 1e-6

    private static func cpuGreedy(hidden: [Float16],
                                  normWeightBF16: [UInt16],
                                  matrix: Int8DenseFixtures.Matrix)
        -> (index: Int, best: Float, runnerUp: Float) {
        let normed = GreedyHeadRef.normalizedInput(hidden: hidden,
                                                   normWeightBF16: normWeightBF16,
                                                   eps: rmsEps)
        let logits = matrix.paddedWeights.withUnsafeBufferPointer { weights in
            matrix.scales.withUnsafeBufferPointer { scales in
                matrix.biases.withUnsafeBufferPointer { biases in
                    DequantInt8PackedGemvRef.apply(weights: weights.baseAddress! + matrix.weightOffset,
                                                   scales: scales.baseAddress!,
                                                   biases: biases.baseAddress!,
                                                   rows: matrix.rowCount,
                                                   x: normed,
                                                   n: matrix.n)
                }
            }
        }
        return GreedyHeadRef.argmax(logits)
    }

    private static func gpuGreedy(hidden: [Float16],
                                  normWeightBF16: [UInt16],
                                  matrix: Int8DenseFixtures.Matrix,
                                  hiddenOffset: Int = 0) throws -> UInt32 {
        let context = try MetalContext()
        let d = matrix.n
        let vocab = matrix.rowCount
        let chain = try LMHeadChainInt8(context: context, maxD: d, maxVocab: vocab)
        let buffers = try Int8DenseFixtures.buffers(matrix, device: context.device)
        let hiddenBuffer = try Int8DenseFixtures.halfBuffer(hidden, device: context.device)
        let norm = try Int8DenseFixtures.bytesBuffer(normWeightBF16, device: context.device)
        let output = try Int8DenseFixtures.bytesBuffer([UInt32.max], device: context.device)
        try Int8DenseFixtures.run(context) { cb in
            chain.encodeGreedyDecode(commandBuffer: cb,
                                     hidden: hiddenBuffer,
                                     hiddenOffset: hiddenOffset,
                                     normWeight: norm,
                                     weights: buffers.weights,
                                     weightsOffset: buffers.weightOffset,
                                     scales: buffers.scales,
                                     biases: buffers.biases,
                                     outToken: output,
                                     d: UInt32(d),
                                     vocab: UInt32(vocab),
                                     rmsEps: rmsEps)
        }
        return output.contents().load(as: UInt32.self)
    }

    @Test func randomRowsMatchCPUReference() throws {
        let d = 64
        let v = 1024
        var rng = SplitMix64(seed: 0x1A48_4EAD_2026_1107)
        let hidden = Int8DenseFixtures.randomHalves(count: d, rng: &rng)
        let norm = (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let matrix = Int8DenseFixtures.matrix(rows: v, n: d, rng: &rng)

        let reference = Self.cpuGreedy(hidden: hidden, normWeightBF16: norm, matrix: matrix)
        let result = try Self.gpuGreedy(hidden: hidden, normWeightBF16: norm, matrix: matrix)
        #expect(result == UInt32(reference.index),
                "margin=\(reference.best - reference.runnerUp)")
    }

    /// Production hidden size on a 4,096-row slice of the vocabulary, with the
    /// table bound at a 4-but-not-16-aligned offset.
    @Test func productionWidthVocabularySubsetMatchesCPUReference() throws {
        let d = 2816
        let v = 4096
        var rng = SeedTree(0x8C01).key("lm-head-int8-subset")
        let hidden = Int8DenseFixtures.randomHalves(count: d, rng: &rng)
        let norm = (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let matrix = Int8DenseFixtures.matrix(rows: v, n: d, rng: &rng,
                                              weightOffset: Int8DenseFixtures.offsetFourNotSixteen)

        let reference = Self.cpuGreedy(hidden: hidden, normWeightBF16: norm, matrix: matrix)
        let margin = reference.best - reference.runnerUp
        // The fixture must have a decisive winner, or equality would hinge on
        // summation order rather than on the kernel being right.
        #expect(margin > 1e-2, "fixture margin=\(margin)")
        let result = try Self.gpuGreedy(hidden: hidden, normWeightBF16: norm, matrix: matrix)
        #expect(result == UInt32(reference.index), "margin=\(margin)")
    }

    @Test func allInvalidCandidatesFallBackToZero() throws {
        let d = 64
        let v = 1024
        let hidden = [Float16](repeating: .nan, count: d)
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let matrix = Int8DenseFixtures.matrix(
            constantRows: Int8DenseFixtures.constantRows(count: v, n: d) { _ in 0.25 },
            n: d)

        let result = try Self.gpuGreedy(hidden: hidden, normWeightBF16: norm, matrix: matrix)
        #expect(result == 0)
    }

    /// Planted winners near the end of vocabularies whose size is and is not
    /// a multiple of the eight rows per threadgroup.
    @Test(arguments: [1024, 2048, 4096, 1001])
    func plantedWinnerAcrossVocabSizes(v: Int) throws {
        let d = 64
        let target = v - 3
        let hidden = (0..<d).map { Float16(0.25 + Float($0 % 7) * 0.01) }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let matrix = Int8DenseFixtures.matrix(
            constantRows: Int8DenseFixtures.constantRows(count: v, n: d) { row in
                row == target ? 1 : -0.5
            },
            n: d,
            weightOffset: 4)

        let reference = Self.cpuGreedy(hidden: hidden, normWeightBF16: norm, matrix: matrix)
        #expect(reference.index == target)
        let result = try Self.gpuGreedy(hidden: hidden, normWeightBF16: norm, matrix: matrix)
        #expect(result == UInt32(target))
    }

    @Test func tiesResolveToLowestIndex() throws {
        let d = 64
        let v = 1024
        let hidden = (0..<d).map { Float16(0.2 + Float($0 % 5) * 0.02) }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let matrix = Int8DenseFixtures.matrix(
            constantRows: Int8DenseFixtures.constantRows(count: v, n: d) { row in
                (row == 17 || row == 311) ? 1 : -0.25
            },
            n: d)

        let result = try Self.gpuGreedy(hidden: hidden, normWeightBF16: norm, matrix: matrix)
        #expect(result == 17)
    }

    @Test func readsSelectedHiddenRow() throws {
        let d = 64
        let v = 1024
        let rowStride = d + 16
        let selectedRow = 3
        let target = 777
        var hiddenRows = [Float16](repeating: 0, count: 5 * rowStride)
        for row in 0..<5 {
            for index in 0..<d {
                let value = row == selectedRow
                    ? Float(0.25 + Float(index % 7) * 0.01)
                    : Float(-0.4 - Float(index % 5) * 0.01)
                hiddenRows[row * rowStride + index] = Float16(value)
            }
        }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let matrix = Int8DenseFixtures.matrix(
            constantRows: Int8DenseFixtures.constantRows(count: v, n: d) { row in
                row == target ? 1 : -0.25
            },
            n: d)

        let result = try Self.gpuGreedy(
            hidden: hiddenRows,
            normWeightBF16: norm,
            matrix: matrix,
            hiddenOffset: selectedRow * rowStride * MemoryLayout<Float16>.stride)
        #expect(result == UInt32(target))
    }
}
