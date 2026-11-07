import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// `PrefillEmbedLookupInt8` against the INT8 CPU reference
/// (`EmbedLookupRef.apply`) at the production width.
@Suite struct PrefillEmbedLookupInt8Tests {
    private static let vocab = 61
    private static let d = 2816
    private static let groupsPerRow = d / Quantization.groupSize

    private struct Table {
        let packed: [UInt8]
        let scales: [UInt16]
        let biases: [UInt16]
    }

    /// Random rows quantized with `quantizeInt8Affine`, so every group holds
    /// the codes 0 and 255.
    private static func makeTable(seed: UInt64) -> Table {
        var rng = SeedTree(seed).key("prefill-int8-embed-table")
        var packed = [UInt8]()
        var scales = [UInt16]()
        var biases = [UInt16]()
        for _ in 0..<vocab {
            let row = (0..<d).map { _ in rng.uniform(-0.1, 0.1) }
            let q = Quantization.quantizeInt8Affine(row)
            packed.append(contentsOf: q.packed)
            scales.append(contentsOf: q.scales)
            biases.append(contentsOf: q.biases)
        }
        return Table(packed: packed, scales: scales, biases: biases)
    }

    /// Token ids that start and end on the vocabulary edges (0 and V - 1) and
    /// repeat ids in between.
    private static func makeTokens(count: Int, seed: UInt64) -> [UInt32] {
        var rng = SeedTree(seed).key("prefill-int8-embed-tokens-\(count)")
        var tokens = (0..<count).map { _ in UInt32(rng.next() % UInt64(vocab)) }
        tokens[0] = 0
        if count > 1 {
            tokens[count - 1] = UInt32(vocab - 1)
        }
        if count > 3 {
            tokens[count / 2] = tokens[1]
        }
        return tokens
    }

    private static func check(table: Table,
                              tokens: [UInt32],
                              outScale: Float,
                              tableOffset: Int,
                              scalesOffset: Int,
                              biasesOffset: Int,
                              outRowOffset: Int,
                              label: String) throws {
        typealias Buffers = PrefillInt8ProjectionBuffers
        let t = tokens.count
        let context = try MetalContext()
        let embed = try PrefillEmbedLookupInt8(context: context)
        let outCount = (outRowOffset + t + 1) * d
        guard let tableBuffer = Buffers.makeBuffer(device: context.device,
                                                   values: table.packed,
                                                   prefixBytes: tableOffset),
              let scalesBuffer = Buffers.makeBuffer(device: context.device,
                                                    values: table.scales,
                                                    prefixBytes: scalesOffset),
              let biasesBuffer = Buffers.makeBuffer(device: context.device,
                                                    values: table.biases,
                                                    prefixBytes: biasesOffset),
              let tokenBuffer = Buffers.makeBuffer(device: context.device, values: tokens),
              let out = Buffers.makeSentinelOutput(device: context.device, count: outCount),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed for \(label)")
            return
        }
        embed.encode(commandBuffer: commandBuffer,
                     table: tableBuffer, tableOffset: tableOffset,
                     scales: scalesBuffer, scalesOffset: scalesOffset,
                     biases: biasesBuffer, biasesOffset: biasesOffset,
                     tokens: tokenBuffer,
                     out: out, outOffset: outRowOffset * d * MemoryLayout<Float16>.stride,
                     t: UInt32(t),
                     d: UInt32(d),
                     outScale: outScale)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil, "\(label): \(String(describing: commandBuffer.error))")

        let all = Fp16Buffer.read(out, count: outCount)
        for (row, token) in tokens.enumerated() {
            let reference = EmbedLookupRef.apply(tablePacked: table.packed,
                                                 tableScales: table.scales,
                                                 tableBiases: table.biases,
                                                 tokenId: Int(token),
                                                 d: d).map { $0 * outScale }
            let start = (outRowOffset + row) * d
            let actual = Array(all[start..<(start + d)])
            let rel = RelError.compute(actual: actual, reference: reference)
            #expect(rel <= Tolerance.quantInt8,
                    "\(label) row \(row) token \(token): rel=\(rel)")
        }
        let before = Buffers.overwrittenCount(out, from: 0, count: outRowOffset * d)
        let after = Buffers.overwrittenCount(out, from: (outRowOffset + t) * d, count: outCount)
        #expect(before == 0 && after == 0,
                "\(label): \(before) elements written before and \(after) after the block")
    }

    /// Every chunk length of the plan with the Gemma 4 output scale
    /// `sqrt(D)`, the table 3 bytes into its buffer (any offset is valid) and
    /// scales and biases 2 mod 4.
    @Test func blockMatchesReferenceAtEveryChunkLength() throws {
        let table = Self.makeTable(seed: 0x5410)
        for t in PrefillInt8Projection.tokenCounts {
            try Self.check(table: table,
                           tokens: Self.makeTokens(count: t, seed: 0x5411),
                           outScale: Float(Self.d).squareRoot(),
                           tableOffset: 3,
                           scalesOffset: 2,
                           biasesOffset: 6,
                           outRowOffset: 0,
                           label: "T=\(t)")
        }
    }

    /// The raw dequantized rows (`outScale` 1) written at a row offset; the
    /// rows around the block stay untouched.
    @Test func unscaledBlockAtAnOutputOffset() throws {
        let table = Self.makeTable(seed: 0x5412)
        try Self.check(table: table,
                       tokens: [UInt32(Self.vocab - 1), 7, 0, 7],
                       outScale: 1,
                       tableOffset: 0,
                       scalesOffset: 0,
                       biasesOffset: 0,
                       outRowOffset: 2,
                       label: "unscaled")
    }

    /// Codes 0 and 255 land exactly on `bias` and `255 · scale + bias` when
    /// both are representable in FP16.
    @Test func extremeCodesAreExact() throws {
        let groups = Self.groupsPerRow
        var packed = [UInt8](repeating: 0, count: Self.d)
        for index in packed.indices where (index / 7).isMultiple(of: 2) {
            packed[index] = 255
        }
        let scale: Float = 0.0078125      // 2^-7
        let bias: Float = -1.0
        let table = Table(packed: packed,
                          scales: [UInt16](repeating: Quantization.bf16Bits(scale), count: groups),
                          biases: [UInt16](repeating: Quantization.bf16Bits(bias), count: groups))
        let context = try MetalContext()
        let embed = try PrefillEmbedLookupInt8(context: context)
        typealias Buffers = PrefillInt8ProjectionBuffers
        guard let tableBuffer = Buffers.makeBuffer(device: context.device, values: table.packed),
              let scalesBuffer = Buffers.makeBuffer(device: context.device, values: table.scales),
              let biasesBuffer = Buffers.makeBuffer(device: context.device, values: table.biases),
              let tokenBuffer = Buffers.makeBuffer(device: context.device, values: [UInt32(0)]),
              let out = Fp16Buffer.make(context.device, count: Self.d),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        embed.encode(commandBuffer: commandBuffer,
                     table: tableBuffer, scales: scalesBuffer, biases: biasesBuffer,
                     tokens: tokenBuffer, out: out,
                     t: 1, d: UInt32(Self.d), outScale: 2)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let actual = Fp16Buffer.read(out, count: Self.d)
        let expected = packed.map { (Float($0) * scale + bias) * 2 }
        #expect(actual == expected)
        #expect(Set(actual) == [-2.0, (255 * scale + bias) * 2])
    }
}
