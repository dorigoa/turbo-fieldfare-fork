import Accelerate
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// INT8 affine weights, FP16 activations and an FP32 CPU reference for the
/// 8-bit prefill projection tests (`PrefillInt8QMM`, `MPPPrefillInt8QMM`).
///
/// Weights are MLX affine 8-bit (plan C5): `rows * columns` bytes, byte `k` of
/// a row is column `k`, and a BF16 scale and bias per group of 64 columns.
struct PrefillInt8Projection {
    let rows: Int
    let columns: Int
    let tokens: Int
    let packed: [UInt8]
    let scales: [UInt16]
    let biases: [UInt16]
    let x: [Float16]

    var groups: Int { columns / Quantization.groupSize }

    /// Production projection shapes of plan §2.4 as (rows, columns).
    static let productionShapes: [(name: String, rows: Int, columns: Int)] = [
        ("swa-q", 4096, 2816),
        ("full-q", 8192, 2816),
        ("swa-kv", 2048, 2816),
        ("full-kv", 1024, 2816),
        ("swa-o", 2816, 4096),
        ("full-o", 2816, 8192),
        ("shared-gate-up", 2112, 2816),
        ("shared-down", 2816, 2112),
    ]

    /// Prefill chunk lengths the plan asks for: one row, partial and whole
    /// 32-row MPP admission, and multiple 64-row tiles.
    static let tokenCounts = [1, 31, 32, 128, 256]

    /// Random weights quantized with `Quantization.quantizeInt8Affine` and
    /// random activations, both from `SeedTree`.
    static func random(rows: Int,
                       columns: Int,
                       tokens: Int,
                       seed: UInt64,
                       weightRange: Float = 0.05,
                       activationRange: Float = 1.0) -> Self {
        precondition(columns % Quantization.groupSize == 0)
        let tree = SeedTree(seed)
        var weightRNG = tree.key("prefill-int8-w-\(rows)x\(columns)")
        let groups = columns / Quantization.groupSize
        var packed = [UInt8]()
        var scales = [UInt16]()
        var biases = [UInt16]()
        packed.reserveCapacity(rows * columns)
        scales.reserveCapacity(rows * groups)
        biases.reserveCapacity(rows * groups)
        var raw = [Float](repeating: 0, count: columns)
        for _ in 0..<rows {
            for index in 0..<columns {
                raw[index] = weightRNG.uniform(-weightRange, weightRange)
            }
            let row = Quantization.quantizeInt8Affine(raw)
            packed.append(contentsOf: row.packed)
            scales.append(contentsOf: row.scales)
            biases.append(contentsOf: row.biases)
        }
        var activationRNG = tree.key("prefill-int8-x-\(tokens)x\(columns)")
        let x = (0..<(tokens * columns)).map { _ in
            Float16(activationRNG.uniform(-activationRange, activationRange))
        }
        return Self(rows: rows, columns: columns, tokens: tokens,
                    packed: packed, scales: scales, biases: biases, x: x)
    }

    /// Deterministic bytes over the full 0...255 range with adversarial BF16
    /// scales and biases (negative, tiny and subnormal), like the 4-bit MPP
    /// twin test.
    static func adversarial(rows: Int, columns: Int, tokens: Int) -> Self {
        precondition(columns % Quantization.groupSize == 0)
        let groups = columns / Quantization.groupSize
        let packed = (0..<(rows * columns)).map {
            UInt8(truncatingIfNeeded: $0 &* 37 &+ 0x29)
        }
        let scaleTable: [UInt16] = [
            Quantization.bf16Bits(0.0004),
            Quantization.bf16Bits(-0.0006),
            0x0001,
            0x8001,
        ]
        let biasTable: [UInt16] = [
            Quantization.bf16Bits(-0.05),
            Quantization.bf16Bits(0.03),
            0x0001,
            0x8001,
        ]
        var scales = [UInt16](repeating: 0, count: rows * groups)
        var biases = [UInt16](repeating: 0, count: rows * groups)
        for row in 0..<rows {
            for group in 0..<groups {
                scales[row * groups + group] = scaleTable[(row + group) % scaleTable.count]
                biases[row * groups + group] = biasTable[(row * 3 + group) % biasTable.count]
            }
        }
        let x = (0..<(tokens * columns)).map {
            Float16(Float(($0 * 11) % 29 - 14) / 64.0)
        }
        return Self(rows: rows, columns: columns, tokens: tokens,
                    packed: packed, scales: scales, biases: biases, x: x)
    }

    /// FP32 reference `Y[t, row] = Σ_k (q · scale + bias) · x[t, k]`, laid out
    /// `[t * rows + row]` like the kernels' output, for every token.
    ///
    /// Dequantizes the whole matrix once with vDSP, then multiplies with
    /// `vDSP_mmul`: a different staging and summation order from the kernels.
    func cpuReference() -> [Float] {
        var weights = [Float](repeating: 0, count: rows * columns)
        let groupSize = Quantization.groupSize
        packed.withUnsafeBufferPointer { q in
            weights.withUnsafeMutableBufferPointer { w in
                for row in 0..<rows {
                    for group in 0..<groups {
                        let base = row * columns + group * groupSize
                        var scale = Quantization.bf16ToFloat(scales[row * groups + group])
                        var bias = Quantization.bf16ToFloat(biases[row * groups + group])
                        vDSP_vfltu8(q.baseAddress! + base, 1,
                                    w.baseAddress! + base, 1,
                                    vDSP_Length(groupSize))
                        vDSP_vsmsa(w.baseAddress! + base, 1, &scale, &bias,
                                   w.baseAddress! + base, 1,
                                   vDSP_Length(groupSize))
                    }
                }
            }
        }
        let activations = x.map { Float($0) }
        var transposed = [Float](repeating: 0, count: columns * tokens)
        vDSP_mtrans(activations, 1, &transposed, 1,
                    vDSP_Length(columns), vDSP_Length(tokens))
        var rowMajorByWeight = [Float](repeating: 0, count: rows * tokens)
        vDSP_mmul(weights, 1, transposed, 1, &rowMajorByWeight, 1,
                  vDSP_Length(rows), vDSP_Length(tokens), vDSP_Length(columns))
        var output = [Float](repeating: 0, count: tokens * rows)
        vDSP_mtrans(rowMajorByWeight, 1, &output, 1,
                    vDSP_Length(tokens), vDSP_Length(rows))
        return output
    }
}

/// Metal buffers for one `PrefillInt8Projection`, with optional leading bytes
/// so tests can place each operand at a chosen offset.
struct PrefillInt8ProjectionBuffers {
    let weights: MTLBuffer
    let scales: MTLBuffer
    let biases: MTLBuffer
    let x: MTLBuffer
    let weightsOffset: Int
    let scalesOffset: Int
    let biasesOffset: Int
    let xOffset: Int

    /// FP16 bit pattern that marks output elements no kernel wrote.
    static let sentinel: UInt16 = 0xFBFF  // -65504

    init?(device: MTLDevice,
          projection: PrefillInt8Projection,
          weightsOffset: Int = 0,
          scalesOffset: Int = 0,
          biasesOffset: Int = 0,
          xOffset: Int = 0) {
        guard let weights = Self.makeBuffer(device: device, values: projection.packed,
                                            prefixBytes: weightsOffset),
              let scales = Self.makeBuffer(device: device, values: projection.scales,
                                           prefixBytes: scalesOffset),
              let biases = Self.makeBuffer(device: device, values: projection.biases,
                                           prefixBytes: biasesOffset),
              let x = Self.makeBuffer(device: device, values: projection.x,
                                      prefixBytes: xOffset) else {
            return nil
        }
        self.weights = weights
        self.scales = scales
        self.biases = biases
        self.x = x
        self.weightsOffset = weightsOffset
        self.scalesOffset = scalesOffset
        self.biasesOffset = biasesOffset
        self.xOffset = xOffset
    }

    static func makeBuffer<T>(device: MTLDevice,
                              values: [T],
                              prefixBytes: Int = 0) -> MTLBuffer? {
        let byteCount = values.count * MemoryLayout<T>.stride
        guard let buffer = device.makeBuffer(length: max(1, prefixBytes + byteCount),
                                             options: .storageModeShared) else {
            return nil
        }
        memset(buffer.contents(), 0, buffer.length)
        values.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            buffer.contents().advanced(by: prefixBytes)
                .copyMemory(from: baseAddress, byteCount: byteCount)
        }
        return buffer
    }

    /// An FP16 output buffer of `count` elements filled with `sentinel`.
    static func makeSentinelOutput(device: MTLDevice, count: Int) -> MTLBuffer? {
        let values = [UInt16](repeating: sentinel, count: count)
        return makeBuffer(device: device, values: values)
    }

    /// Number of elements in `[from, count)` that no longer hold the sentinel.
    static func overwrittenCount(_ buffer: MTLBuffer, from: Int, count: Int) -> Int {
        let bits = buffer.contents().assumingMemoryBound(to: UInt16.self)
        var overwritten = 0
        for index in from..<count where bits[index] != sentinel {
            overwritten += 1
        }
        return overwritten
    }
}
