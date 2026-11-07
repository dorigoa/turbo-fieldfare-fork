import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Validates `PrefillFinalRowHeadInt8` against the scalar path it fuses
/// (RMSNorm of the selected row, then `DequantInt8GEMV`) and against the CPU
/// reference.
@Suite struct PrefillFinalRowHeadInt8Tests {
    @Test func finalRowHeadLogitsMatchScalarPathAndReference() throws {
        let rows = 5
        let selectedRow = 3
        let d = 128
        let rowStride = d + 17
        let vocab = 96
        let eps: Float = 1e-6
        let weightOffset = Int8DenseFixtures.offsetFourNotSixteen
        var rng = SeedTree(0x8D01).key("prefill-final-row-head-int8")

        var hidden = [Float16](repeating: 0, count: rows * rowStride)
        for row in 0..<rows {
            for i in 0..<d {
                hidden[row * rowStride + i] = Float16(rng.uniform(-1.0, 1.0))
            }
        }
        let normBits = (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let matrix = Int8DenseFixtures.matrix(rows: vocab, n: d, rng: &rng,
                                              range: -0.5...0.5, weightOffset: weightOffset)

        let context = try MetalContext()
        let scalarNorm = try RMSNorm(context: context)
        let scalarHead = try DequantInt8GEMV(context: context)
        let finalRowHead = try PrefillFinalRowHeadInt8(context: context, maxD: d)
        let device = context.device
        let hiddenBuffer = try Int8DenseFixtures.halfBuffer(hidden, device: device)
        let normBuffer = try Int8DenseFixtures.bytesBuffer(normBits, device: device)
        let buffers = try Int8DenseFixtures.buffers(matrix, device: device)
        let scalarNormed = try Int8DenseFixtures.halfBuffer(count: d, device: device)
        let scalarLogits = try Int8DenseFixtures.halfBuffer(count: vocab, device: device)
        let logitPad = 7
        let blockLogits = try Int8DenseFixtures.halfBuffer(count: logitPad + vocab,
                                                           device: device)

        try Int8DenseFixtures.run(context) { cb in
            scalarNorm.encodeBF16W(commandBuffer: cb,
                                   x: hiddenBuffer,
                                   xOffset: selectedRow * rowStride * MemoryLayout<Float16>.size,
                                   weight: normBuffer,
                                   out: scalarNormed,
                                   d: UInt32(d),
                                   eps: eps)
            scalarHead.encode(commandBuffer: cb,
                              weights: buffers.weights,
                              weightsOffset: weightOffset,
                              scales: buffers.scales,
                              biases: buffers.biases,
                              x: scalarNormed,
                              y: scalarLogits,
                              m: UInt32(vocab),
                              n: UInt32(d))
            finalRowHead.encodeLogits(commandBuffer: cb,
                                      hiddenBlock: hiddenBuffer,
                                      row: selectedRow,
                                      rowStrideElements: rowStride,
                                      normWeight: normBuffer,
                                      weights: buffers.weights,
                                      weightsOffset: weightOffset,
                                      scales: buffers.scales,
                                      biases: buffers.biases,
                                      logits: blockLogits,
                                      logitsOffset: logitPad * MemoryLayout<Float16>.size,
                                      d: UInt32(d),
                                      vocab: UInt32(vocab),
                                      rmsEps: eps)
        }

        let scalar = Fp16Buffer.read(scalarLogits, count: vocab)
        let block = Array(Fp16Buffer.read(blockLogits, count: logitPad + vocab)
            .dropFirst(logitPad))
        #expect(block == scalar, "maxAbs=\(RelError.maxAbsDiff(block, scalar))")

        let selected = Array(hidden[(selectedRow * rowStride)..<(selectedRow * rowStride + d)])
        let normed = GreedyHeadRef.normalizedInput(hidden: selected,
                                                   normWeightBF16: normBits, eps: eps)
        let reference = DequantInt8GemvRef.apply(weightRows: matrix.rows, x: normed, n: d)
        let rel = RelError.compute(actual: block, reference: reference)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "rel=\(rel) maxAbs=\(RelError.maxAbsDiff(block, reference))")
    }
}
