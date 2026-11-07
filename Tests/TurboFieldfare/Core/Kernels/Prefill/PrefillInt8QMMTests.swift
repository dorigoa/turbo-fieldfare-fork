import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// `PrefillInt8QMM` against the FP32 CPU reference and against
/// `DequantInt8GEMV` run once per token row, the path the runner uses for
/// short chunks.
@Suite struct PrefillInt8QMMTests {
    typealias Projection = PrefillInt8Projection
    typealias Buffers = PrefillInt8ProjectionBuffers

    /// Runs the QMM once per token count, each into its own sentinel-filled
    /// output, and checks every requested row against the CPU reference and
    /// the repeated GEMV, and that no row past `t` was written.
    private static func check(_ projection: Projection,
                              label: String,
                              tokenCounts: [Int],
                              weightsOffset: Int = 0,
                              scalesOffset: Int = 0,
                              biasesOffset: Int = 0,
                              xOffset: Int = 0) throws {
        let n = projection.rows
        let k = projection.columns
        let tokens = projection.tokens
        precondition(tokenCounts.allSatisfy { $0 <= tokens })
        let context = try MetalContext()
        let qmm = try PrefillInt8QMM(context: context)
        let gemv = try DequantInt8GEMV(context: context)
        guard let buffers = Buffers(device: context.device,
                                    projection: projection,
                                    weightsOffset: weightsOffset,
                                    scalesOffset: scalesOffset,
                                    biasesOffset: biasesOffset,
                                    xOffset: xOffset),
              let gemvOut = Fp16Buffer.make(context.device, count: tokens * n),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed for \(label)")
            return
        }
        let half = MemoryLayout<Float16>.stride
        for row in 0..<tokens {
            gemv.encode(commandBuffer: commandBuffer,
                        weights: buffers.weights, weightsOffset: weightsOffset,
                        scales: buffers.scales, scalesOffset: scalesOffset,
                        biases: buffers.biases, biasesOffset: biasesOffset,
                        x: buffers.x, xOffset: xOffset + row * k * half,
                        y: gemvOut, yOffset: row * n * half,
                        m: UInt32(n),
                        n: UInt32(k))
        }
        var outputs: [(t: Int, buffer: MTLBuffer)] = []
        for t in tokenCounts {
            guard let output = Buffers.makeSentinelOutput(device: context.device,
                                                          count: tokens * n) else {
                Issue.record("output allocation failed for \(label)")
                return
            }
            qmm.encode(commandBuffer: commandBuffer,
                       weights: buffers.weights, weightsOffset: weightsOffset,
                       scales: buffers.scales, scalesOffset: scalesOffset,
                       biases: buffers.biases, biasesOffset: biasesOffset,
                       x: buffers.x, xOffset: xOffset,
                       y: output,
                       t: t,
                       n: n,
                       k: k)
            outputs.append((t, output))
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil, "\(label): \(String(describing: commandBuffer.error))")

        let reference = projection.cpuReference()
        let repeated = Fp16Buffer.read(gemvOut, count: tokens * n)
        let repeatedRel = RelError.compute(actual: repeated, reference: reference)
        #expect(repeatedRel <= Tolerance.fp16Reduction,
                "\(label): repeated GEMV against CPU rel=\(repeatedRel)")
        for (t, output) in outputs {
            let count = t * n
            let actual = Fp16Buffer.read(output, count: count)
            let expected = Array(reference[0..<count])
            let rows = Array(repeated[0..<count])
            let cpuRel = RelError.compute(actual: actual, reference: expected)
            let cpuMaxAbs = RelError.maxAbsDiff(actual, expected)
            let gemvRel = RelError.compute(actual: actual, reference: rows)
            let gemvMaxAbs = RelError.maxAbsDiff(actual, rows)
            #expect(cpuRel <= Tolerance.fp16Reduction,
                    "\(label) T=\(t): CPU rel=\(cpuRel) maxAbs=\(cpuMaxAbs)")
            #expect(gemvRel <= Tolerance.fp16Reduction,
                    "\(label) T=\(t): GEMV rel=\(gemvRel) maxAbs=\(gemvMaxAbs)")
            let allFinite = actual.allSatisfy { $0.isFinite }
            #expect(allFinite, "\(label) T=\(t): non-finite output")
            let stray = Buffers.overwrittenCount(output, from: count, count: tokens * n)
            #expect(stray == 0, "\(label) T=\(t): \(stray) elements written past row \(t)")
        }
    }

    /// Every projection of plan §2.4 that the prefill runs as a dense QMM, at
    /// every chunk length of the plan. Weights sit 4 mod 16 bytes into their
    /// buffer, scales and biases 2 mod 4 (plan C5).
    @Test func productionShapesMatchCPUAndRepeatedGEMV() throws {
        for (index, shape) in Projection.productionShapes.enumerated() {
            let projection = Projection.random(rows: shape.rows,
                                               columns: shape.columns,
                                               tokens: Projection.tokenCounts.max()!,
                                               seed: 0x5400 + UInt64(index))
            try Self.check(projection,
                           label: shape.name,
                           tokenCounts: Projection.tokenCounts,
                           weightsOffset: 20,
                           scalesOffset: 2,
                           biasesOffset: 6)
        }
    }

    /// Rows and tokens that do not fill an 8 x 8 threadgroup, the smallest
    /// and an odd number of groups, and activations at a 2-byte offset that
    /// is not 8-byte aligned.
    @Test func irregularShapesMatchCPUAndRepeatedGEMV() throws {
        let shapes = [
            (rows: 1, columns: 64, tokens: 1),
            (rows: 35, columns: 64, tokens: 17),
            (rows: 129, columns: 128, tokens: 3),
            (rows: 65, columns: 192, tokens: 9),
            (rows: 7, columns: 2112, tokens: 40),
        ]
        for (index, shape) in shapes.enumerated() {
            let projection = Projection.random(rows: shape.rows,
                                               columns: shape.columns,
                                               tokens: shape.tokens,
                                               seed: 0x5480 + UInt64(index),
                                               weightRange: 0.5)
            try Self.check(projection,
                           label: "irregular \(shape.rows)x\(shape.columns)",
                           tokenCounts: Array(Set([1, shape.tokens])).sorted(),
                           weightsOffset: 4 * index,
                           xOffset: 2 * index)
        }
    }

    /// Every byte value with negative, tiny and subnormal BF16 scales and
    /// biases.
    @Test func adversarialAffineValuesMatchCPU() throws {
        let projection = Projection.adversarial(rows: 67, columns: 320, tokens: 33)
        try Self.check(projection,
                       label: "adversarial",
                       tokenCounts: [1, 32, 33],
                       weightsOffset: 12,
                       scalesOffset: 2,
                       biasesOffset: 4)
    }
}
