import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The 8-bit dense kernels on the production tied embedding table:
/// V = 262,144 rows × D = 2,816 bytes (≈ 738 MB), laid out like the resident
/// blob (weights, then BF16 scales, then BF16 biases in one buffer).
///
/// One test builds the table once and checks every consumer: the embedding
/// lookup at both edge tokens, the greedy head (specialized pipeline) and the
/// logits head through `PrefillFinalRowHeadInt8` (specialized
/// `DequantInt8GEMV` shape). The table lives in an `MTLBuffer`, not the Swift
/// heap. Quantizing 738 M values through `quantizeInt8Affine` would take
/// minutes in a debug build, so the codes are drawn uniformly from 0...255
/// with per-group BF16 scales and biases in the range `quantizeInt8Affine`
/// produces for uniform(-1, 1) rows (scale ≈ 2/255, bias ≈ -1).
@Suite struct Int8DenseFullVocabularyTests {
    private static let d = 2816
    private static let vocab = 262_144
    private static let rmsEps: Float = 1e-6

    private struct Table {
        let blob: MTLBuffer
        let weightsOffset: Int
        let scalesOffset: Int
        let biasesOffset: Int

        var weights: UnsafePointer<UInt8> {
            UnsafePointer((blob.contents() + weightsOffset).assumingMemoryBound(to: UInt8.self))
        }
        var scales: UnsafePointer<UInt16> {
            UnsafePointer((blob.contents() + scalesOffset).assumingMemoryBound(to: UInt16.self))
        }
        var biases: UnsafePointer<UInt16> {
            UnsafePointer((blob.contents() + biasesOffset).assumingMemoryBound(to: UInt16.self))
        }
    }

    private static func makeTable(device: MTLDevice) throws -> Table {
        let groups = d / Quantization.groupSize
        // Weights at a 4-but-not-16-aligned offset; scales right after them at
        // a 2-but-not-4-aligned offset.
        let weightsOffset = Int8DenseFixtures.offsetFourNotSixteen
        let weightsEnd = weightsOffset + vocab * d
        let scalesOffset = weightsEnd + 2
        let biasesOffset = scalesOffset + vocab * groups * MemoryLayout<UInt16>.size
        let length = biasesOffset + vocab * groups * MemoryLayout<UInt16>.size
        guard let blob = device.makeBuffer(length: (length + 7) & ~7,
                                           options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        let raw = blob.contents()
        var rng = SeedTree(0x8F01).key("int8-full-vocabulary-table")
        for word in 0..<((weightsEnd + 7) / 8) {
            raw.storeBytes(of: rng.next(), toByteOffset: word * 8, as: UInt64.self)
        }
        let scales = (raw + scalesOffset).assumingMemoryBound(to: UInt16.self)
        let biases = (raw + biasesOffset).assumingMemoryBound(to: UInt16.self)
        for i in 0..<(vocab * groups) {
            scales[i] = Quantization.bf16Bits(rng.uniform(0.0070, 0.0078))
            biases[i] = Quantization.bf16Bits(rng.uniform(-1.0, -0.9))
        }
        return Table(blob: blob, weightsOffset: weightsOffset,
                     scalesOffset: scalesOffset, biasesOffset: biasesOffset)
    }

    @Test func productionVocabularyEmbeddingAndHeads() throws {
        let context = try MetalContext()
        let device = context.device
        let table = try Self.makeTable(device: device)

        // Embedding lookup at the first, last and an interior token.
        let embed = try EmbedLookupInt8(context: context)
        let outScale = Float(Self.d).squareRoot()
        for token in [0, Self.vocab - 1, 123_457] {
            let out = try Int8DenseFixtures.halfBuffer(count: Self.d, device: device)
            try Int8DenseFixtures.run(context) { cb in
                embed.encode(commandBuffer: cb,
                             table: table.blob, tableOffset: table.weightsOffset,
                             scales: table.blob, scalesOffset: table.scalesOffset,
                             biases: table.blob, biasesOffset: table.biasesOffset,
                             out: out,
                             tokenId: UInt32(token),
                             d: UInt32(Self.d),
                             outScale: outScale)
            }
            let actual = Fp16Buffer.read(out, count: Self.d)
            let reference = EmbedLookupInt8Ref.apply(table: table.weights,
                                                     scales: table.scales,
                                                     biases: table.biases,
                                                     tokenId: token,
                                                     d: Self.d,
                                                     outScale: outScale)
            let rel = RelError.compute(actual: actual, reference: reference)
            #expect(rel < Tolerance.quantInt8, "embedding token=\(token) rel=\(rel)")
        }

        // Head input. The CPU reference takes the GPU RMSNorm output, so the
        // greedy comparison isolates the INT8 row pass from FP16 rounding
        // differences in the norm.
        var rng = SeedTree(0x8F02).key("int8-full-vocabulary-hidden")
        let hidden = Int8DenseFixtures.randomHalves(count: Self.d, rng: &rng)
        let normBits = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let hiddenBuffer = try Int8DenseFixtures.halfBuffer(hidden, device: device)
        let normBuffer = try Int8DenseFixtures.bytesBuffer(normBits, device: device)
        let normed = try Int8DenseFixtures.halfBuffer(count: Self.d, device: device)
        let rms = try RMSNorm(context: context)
        try Int8DenseFixtures.run(context) { cb in
            rms.encodeBF16W(commandBuffer: cb, x: hiddenBuffer, weight: normBuffer,
                            out: normed, d: UInt32(Self.d), eps: Self.rmsEps)
        }
        let normedValues = Fp16Buffer.read(normed, count: Self.d)
        let cpuNormed = GreedyHeadRef.normalizedInput(hidden: hidden,
                                                      normWeightBF16: normBits,
                                                      eps: Self.rmsEps)
        #expect(RelError.compute(actual: normedValues, reference: cpuNormed) < Tolerance.fp16Reduction)

        let logits = DequantInt8PackedGemvRef.apply(weights: table.weights,
                                                    scales: table.scales,
                                                    biases: table.biases,
                                                    rows: Self.vocab,
                                                    x: normedValues,
                                                    n: Self.d)
        let reference = GreedyHeadRef.argmax(logits)
        let margin = reference.best - reference.runnerUp
        #expect(margin > 1e-2, "fixture margin=\(margin)")

        // Greedy head: specialized D = 2816, V = 262144 pipeline.
        let chain = try LMHeadChainInt8(context: context)
        let token = try Int8DenseFixtures.bytesBuffer([UInt32.max], device: device)
        try Int8DenseFixtures.run(context) { cb in
            chain.encodeGreedyDecode(commandBuffer: cb,
                                     hidden: hiddenBuffer,
                                     normWeight: normBuffer,
                                     weights: table.blob,
                                     weightsOffset: table.weightsOffset,
                                     scales: table.blob,
                                     scalesOffset: table.scalesOffset,
                                     biases: table.blob,
                                     biasesOffset: table.biasesOffset,
                                     outToken: token,
                                     d: UInt32(Self.d),
                                     vocab: UInt32(Self.vocab),
                                     rmsEps: Self.rmsEps)
        }
        let greedy = token.contents().load(as: UInt32.self)
        #expect(greedy == UInt32(reference.index),
                "greedy=\(greedy) reference=\(reference.index) margin=\(margin)")

        // Logits head: `DequantInt8GEMV` at the specialized M = 262144 shape.
        let finalRowHead = try PrefillFinalRowHeadInt8(context: context)
        let gpuLogitsBuffer = try Int8DenseFixtures.halfBuffer(count: Self.vocab, device: device)
        try Int8DenseFixtures.run(context) { cb in
            finalRowHead.encodeLogits(commandBuffer: cb,
                                      hiddenBlock: hiddenBuffer,
                                      row: 0,
                                      rowStrideElements: Self.d,
                                      normWeight: normBuffer,
                                      weights: table.blob,
                                      weightsOffset: table.weightsOffset,
                                      scales: table.blob,
                                      scalesOffset: table.scalesOffset,
                                      biases: table.blob,
                                      biasesOffset: table.biasesOffset,
                                      logits: gpuLogitsBuffer,
                                      d: UInt32(Self.d),
                                      vocab: UInt32(Self.vocab),
                                      rmsEps: Self.rmsEps)
        }
        let gpuLogits = Fp16Buffer.read(gpuLogitsBuffer, count: Self.vocab)
        let rel = RelError.compute(actual: gpuLogits, reference: logits)
        #expect(rel < Tolerance.fp16Reduction,
                "logits rel=\(rel) maxAbs=\(RelError.maxAbsDiff(gpuLogits, logits))")
        // FP16 logits may merge near-ties, but their argmax must be a top
        // candidate within one FP16 step of the FP32 maximum.
        let logitsPick = GreedyHeadRef.argmax(gpuLogits).index
        let fp16Step = Float(Float16(reference.best).ulp)
        #expect(logits[logitsPick] >= reference.best - fp16Step,
                "logits argmax=\(logitsPick) reference=\(reference.index)")
    }
}
