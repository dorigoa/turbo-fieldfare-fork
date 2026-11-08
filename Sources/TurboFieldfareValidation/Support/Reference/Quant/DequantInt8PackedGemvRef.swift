import Foundation
import Accelerate
import TurboFieldfare

/// FP32 reference for the INT8-affine GEMV `y = W * x` over a packed,
/// row-major table in the layout the kernels read: `[rows, n]` unsigned bytes
/// and `[rows, n/64]` BF16 scales and biases (as `UInt16` bit patterns).
///
/// Unlike `DequantInt8GemvRef`, it takes raw pointers, so a vocabulary-sized
/// table (262,144 × 2,816 bytes) can stay in an `MTLBuffer` instead of the
/// Swift heap. Each row is widened with `vDSP_vfltu8`, every group of 64 is
/// dotted with `vDSP_dotpr`, and the group's scale and bias apply as
/// `s * Σ q·x + b * Σ x`. The kernels interleave the unpack with the FMA chain
/// and reduce across a SIMD group: different staging, different summation
/// order.
public enum DequantInt8PackedGemvRef {
    public static func apply(weights: UnsafePointer<UInt8>,
                             scales: UnsafePointer<UInt16>,
                             biases: UnsafePointer<UInt16>,
                             rows: Int,
                             x: [Float],
                             n: Int) -> [Float] {
        precondition(rows > 0, "rows must be positive")
        precondition(x.count == n, "x must have n elements")
        precondition(n % Quantization.groupSize == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        let groupSize = Quantization.groupSize
        let groups = n / groupSize

        var groupSums = [Float](repeating: 0, count: groups)
        x.withUnsafeBufferPointer { px in
            for g in 0..<groups {
                vDSP_sve(px.baseAddress! + g * groupSize, 1,
                         &groupSums[g], vDSP_Length(groupSize))
            }
        }

        var y = [Float](repeating: 0, count: rows)
        var widened = [Float](repeating: 0, count: n)
        x.withUnsafeBufferPointer { px in
            widened.withUnsafeMutableBufferPointer { pq in
                let q = pq.baseAddress!
                let input = px.baseAddress!
                for row in 0..<rows {
                    vDSP_vfltu8(weights + row * n, 1, q, 1, vDSP_Length(n))
                    var acc: Float = 0
                    for g in 0..<groups {
                        var dot: Float = 0
                        vDSP_dotpr(q + g * groupSize, 1,
                                   input + g * groupSize, 1,
                                   &dot, vDSP_Length(groupSize))
                        let scale = Quantization.bf16ToFloat(scales[row * groups + g])
                        let bias = Quantization.bf16ToFloat(biases[row * groups + g])
                        acc += scale * dot + bias * groupSums[g]
                    }
                    y[row] = acc
                }
            }
        }
        return y
    }
}
