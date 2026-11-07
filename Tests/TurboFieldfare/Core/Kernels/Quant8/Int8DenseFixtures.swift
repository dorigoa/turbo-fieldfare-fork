import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Fixtures shared by the 8-bit dense decode kernel tests.
///
/// Matrices come from `Quantization.quantizeInt8Affine` over seeded uniform
/// rows. The packed weights can sit at a non-zero byte offset: the plan's C5
/// lets the 8-bit kernels assume only a 4-byte-aligned weight base, so the
/// tests bind weights at `offsetFourNotSixteen`, which is 4-aligned but not
/// 16-aligned.
enum Int8DenseFixtures {
    /// A weight offset that is a multiple of 4 but not of 16.
    static let offsetFourNotSixteen = 20

    struct Matrix {
        let rows: [Quantization.Int8AffineRow]
        let n: Int
        /// Packed weights preceded by `weightOffset` zero bytes.
        let paddedWeights: [UInt8]
        let weightOffset: Int
        let scales: [UInt16]
        let biases: [UInt16]

        var rowCount: Int { rows.count }
    }

    static func matrix(rows: Int,
                       n: Int,
                       rng: inout SplitMix64,
                       range: ClosedRange<Float> = -1.0...1.0,
                       weightOffset: Int = 0) -> Matrix {
        precondition(n % Quantization.groupSize == 0)
        let groups = n / Quantization.groupSize
        var quantized: [Quantization.Int8AffineRow] = []
        quantized.reserveCapacity(rows)
        var padded = [UInt8](repeating: 0, count: weightOffset + rows * n)
        var scales = [UInt16](repeating: 0, count: rows * groups)
        var biases = [UInt16](repeating: 0, count: rows * groups)
        for row in 0..<rows {
            let values = (0..<n).map { _ in
                rng.uniform(range.lowerBound, range.upperBound)
            }
            let q = Quantization.quantizeInt8Affine(values)
            let start = weightOffset + row * n
            padded.replaceSubrange(start..<(start + n), with: q.packed)
            scales.replaceSubrange((row * groups)..<((row + 1) * groups), with: q.scales)
            biases.replaceSubrange((row * groups)..<((row + 1) * groups), with: q.biases)
            quantized.append(q)
        }
        return Matrix(rows: quantized, n: n, paddedWeights: padded,
                      weightOffset: weightOffset, scales: scales, biases: biases)
    }

    /// The same matrix with its packed weights moved to `weightOffset`.
    static func rebased(_ m: Matrix, weightOffset: Int) -> Matrix {
        var padded = [UInt8](repeating: 0, count: weightOffset)
        padded.append(contentsOf: m.paddedWeights[m.weightOffset...])
        return Matrix(rows: m.rows, n: m.n, paddedWeights: padded,
                      weightOffset: weightOffset, scales: m.scales, biases: m.biases)
    }

    /// Rows whose every element is `value(row)`, so a test can plant a
    /// winner or a tie for the greedy head.
    static func constantRows(count: Int,
                             n: Int,
                             value: (Int) -> Float) -> [Quantization.Int8AffineRow] {
        (0..<count).map { row in
            Quantization.quantizeInt8Affine([Float](repeating: value(row), count: n))
        }
    }

    static func matrix(constantRows rows: [Quantization.Int8AffineRow],
                       n: Int,
                       weightOffset: Int = 0) -> Matrix {
        let groups = n / Quantization.groupSize
        var padded = [UInt8](repeating: 0, count: weightOffset + rows.count * n)
        var scales = [UInt16](repeating: 0, count: rows.count * groups)
        var biases = [UInt16](repeating: 0, count: rows.count * groups)
        for (row, q) in rows.enumerated() {
            let start = weightOffset + row * n
            padded.replaceSubrange(start..<(start + n), with: q.packed)
            scales.replaceSubrange((row * groups)..<((row + 1) * groups), with: q.scales)
            biases.replaceSubrange((row * groups)..<((row + 1) * groups), with: q.biases)
        }
        return Matrix(rows: rows, n: n, paddedWeights: padded,
                      weightOffset: weightOffset, scales: scales, biases: biases)
    }

    struct MatrixBuffers {
        let weights: MTLBuffer
        let scales: MTLBuffer
        let biases: MTLBuffer
        let weightOffset: Int
    }

    static func buffers(_ matrix: Matrix, device: MTLDevice) throws -> MatrixBuffers {
        MatrixBuffers(weights: try bytesBuffer(matrix.paddedWeights, device: device),
                      scales: try bytesBuffer(matrix.scales, device: device),
                      biases: try bytesBuffer(matrix.biases, device: device),
                      weightOffset: matrix.weightOffset)
    }

    static func bytesBuffer<T>(_ values: [T], device: MTLDevice) throws -> MTLBuffer {
        let length = max(values.count * MemoryLayout<T>.stride, 1)
        let buffer = values.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: length,
                              options: .storageModeShared)
        }
        guard let buffer else { throw MetalError.noDevice }
        return buffer
    }

    static func halfBuffer(_ values: [Float16], device: MTLDevice) throws -> MTLBuffer {
        guard let buffer = Fp16Buffer.make(device, halves: values) else {
            throw MetalError.noDevice
        }
        return buffer
    }

    static func halfBuffer(count: Int, device: MTLDevice) throws -> MTLBuffer {
        guard let buffer = Fp16Buffer.make(device, count: count) else {
            throw MetalError.noDevice
        }
        return buffer
    }

    static func randomHalves(count: Int,
                             rng: inout SplitMix64,
                             range: ClosedRange<Float> = -1.0...1.0) -> [Float16] {
        (0..<count).map { _ in Float16(rng.uniform(range.lowerBound, range.upperBound)) }
    }

    static func run(_ context: MetalContext,
                    _ encode: (MTLCommandBuffer) throws -> Void) throws {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.noQueue
        }
        try encode(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer)
    }

    static func halfBytes(_ buffer: MTLBuffer, offset: Int = 0, count: Int) -> [UInt8] {
        Array(UnsafeBufferPointer(
            start: (buffer.contents() + offset).assumingMemoryBound(to: UInt8.self),
            count: count * MemoryLayout<Float16>.size))
    }
}
