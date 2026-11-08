import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Validates `embed_lookup_int8`, the 8-bit twin of the tied embedding lookup.
/// The vocabulary-sized table (V = 262,144) is covered by
/// `Int8DenseFullVocabularyTests`.
@Suite struct EmbedLookupInt8Tests {
    private static let vocab = 16

    private struct Table {
        let matrix: Int8DenseFixtures.Matrix
        let buffers: Int8DenseFixtures.MatrixBuffers
    }

    private static func table(d: Int, seed: UInt64, device: MTLDevice,
                              weightOffset: Int) throws -> Table {
        var rng = SeedTree(seed).key("embed-lookup-int8-table-d\(d)")
        let matrix = Int8DenseFixtures.matrix(rows: vocab, n: d, rng: &rng,
                                              weightOffset: weightOffset)
        return Table(matrix: matrix,
                     buffers: try Int8DenseFixtures.buffers(matrix, device: device))
    }

    private static func lookup(_ table: Table,
                               context: MetalContext,
                               kernel: EmbedLookupInt8,
                               token: Int,
                               outScale: Float,
                               outOffsetElements: Int = 0) throws -> [Float] {
        let d = table.matrix.n
        let out = try Int8DenseFixtures.halfBuffer(count: outOffsetElements + d,
                                                   device: context.device)
        try Int8DenseFixtures.run(context) { cb in
            kernel.encode(commandBuffer: cb,
                          table: table.buffers.weights,
                          tableOffset: table.buffers.weightOffset,
                          scales: table.buffers.scales,
                          biases: table.buffers.biases,
                          out: out,
                          outOffset: outOffsetElements * MemoryLayout<Float16>.size,
                          tokenId: UInt32(token),
                          d: UInt32(d),
                          outScale: outScale)
        }
        return Array(Fp16Buffer.read(out, count: outOffsetElements + d)
            .dropFirst(outOffsetElements))
    }

    private static func reference(_ table: Table, token: Int, outScale: Float) -> [Float] {
        let m = table.matrix
        return EmbedLookupInt8Ref.apply(
            tablePacked: Array(m.paddedWeights.dropFirst(m.weightOffset)),
            tableScales: m.scales,
            tableBiases: m.biases,
            tokenId: token,
            d: m.n,
            outScale: outScale)
    }

    /// Production hidden size with the `sqrt(H)` scale, at the first and last
    /// tokens of the table and with the table bound at a 4-but-not-16-aligned
    /// offset.
    @Test(arguments: [0, vocab - 1, 9])
    func productionWidthWithSqrtScaleMatchesReference(token: Int) throws {
        let d = 2816
        let context = try MetalContext()
        let kernel = try EmbedLookupInt8(context: context)
        let table = try Self.table(d: d, seed: 0x8E01, device: context.device,
                                   weightOffset: Int8DenseFixtures.offsetFourNotSixteen)
        let outScale = Float(d).squareRoot()

        let actual = try Self.lookup(table, context: context, kernel: kernel,
                                     token: token, outScale: outScale)
        let reference = Self.reference(table, token: token, outScale: outScale)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < Tolerance.quantInt8,
                "token=\(token) rel=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, reference))")
    }

    /// `outScale = 1` writes the raw dequantized row, here at a non-zero
    /// output offset and an odd-sized group count.
    @Test func unitScaleWritesRawDequantAtOutputOffset() throws {
        let d = 192
        let context = try MetalContext()
        let kernel = try EmbedLookupInt8(context: context)
        let table = try Self.table(d: d, seed: 0x8E02, device: context.device,
                                   weightOffset: 0)
        let token = Self.vocab - 1

        let actual = try Self.lookup(table, context: context, kernel: kernel,
                                     token: token, outScale: 1.0,
                                     outOffsetElements: 5)
        let raw = Quantization.dequantizeInt8Affine(table.matrix.rows[token], n: d)
        let rel = RelError.compute(actual: actual, reference: raw)
        #expect(rel < Tolerance.quantInt8, "rel=\(rel)")
    }

    /// The INT8 path keeps the full 0...255 code range: 0 and 255 decode to the
    /// group's bias and bias + 255 * scale.
    @Test func extremeCodesDecodeExactly() throws {
        let d = 64
        let context = try MetalContext()
        let kernel = try EmbedLookupInt8(context: context)
        var packed = [UInt8](repeating: 0, count: 2 * d)
        for i in 0..<d { packed[d + i] = i.isMultiple(of: 2) ? 0 : 255 }
        let scale = Quantization.bf16Bits(0.0078125)
        let bias = Quantization.bf16Bits(-1.0)
        let matrix = Int8DenseFixtures.Matrix(
            rows: [Quantization.Int8AffineRow(packed: Array(packed[0..<d]),
                                              scales: [scale], biases: [bias]),
                   Quantization.Int8AffineRow(packed: Array(packed[d..<(2 * d)]),
                                              scales: [scale], biases: [bias])],
            n: d,
            paddedWeights: packed,
            weightOffset: 0,
            scales: [scale, scale],
            biases: [bias, bias])
        let table = Table(matrix: matrix,
                          buffers: try Int8DenseFixtures.buffers(matrix, device: context.device))

        let actual = try Self.lookup(table, context: context, kernel: kernel,
                                     token: 1, outScale: 1.0)
        for i in 0..<d {
            let expected: Float = i.isMultiple(of: 2) ? -1.0 : 255 * 0.0078125 - 1.0
            #expect(actual[i] == expected, "i=\(i) got \(actual[i]) expected \(expected)")
        }
    }
}
