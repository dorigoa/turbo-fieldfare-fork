import Foundation
import Accelerate

/// FP32 CPU references for the Qwen 3.5 MoE normalization kernels (plan S12).
///
/// Deliberately reaches the same answers through a different operation
/// sequence than the Metal kernels: `vDSP_svesq` for the sums of squares and
/// scalar `squareRoot` + division for the reciprocal, while the kernels use a
/// per-thread `fma` chain, `simd_sum` tree reduction and `rsqrt`. Different
/// summation tree, different rounding chain, same math — which is what makes
/// these useful comparators. Inputs are expected already FP16-rounded (what
/// the kernels read from device memory); outputs are compared after rounding
/// the GPU result back to Float.
public enum QwenNormRef {

    /// One row: y[i] = x[i] * w'[i] / sqrt(mean(x^2) + eps),
    /// with w' = w or 1 + w.
    public static func rmsNorm(x: [Float], weight: [Float],
                               eps: Float, onePlusW: Bool) -> [Float] {
        precondition(x.count == weight.count, "x and weight must match length")
        let d = x.count
        var sumSq: Float = 0
        x.withUnsafeBufferPointer { p in
            vDSP_svesq(p.baseAddress!, 1, &sumSq, vDSP_Length(d))
        }
        let inv = 1.0 / (sumSq / Float(d) + eps).squareRoot()
        var y = [Float](repeating: 0, count: d)
        for i in 0..<d {
            var w = weight[i]
            if onePlusW { w += 1.0 }
            y[i] = x[i] * inv * w
        }
        return y
    }

    /// `x.count / headDim` contiguous heads sharing `weight`:
    /// per head, y[i] = x[i] * w'[i] / sqrt(mean(x_h^2) + eps).
    public static func perHeadRMSNorm(x: [Float], weight: [Float],
                                      headDim: Int,
                                      eps: Float,
                                      onePlusW: Bool) -> [Float] {
        precondition(x.count % headDim == 0, "x must be a whole number of heads")
        precondition(weight.count == headDim, "weight must span one head")
        var y = [Float](repeating: 0, count: x.count)
        var offset = 0
        while offset < x.count {
            let head = Array(x[offset..<offset + headDim])
            let headY = rmsNorm(x: head, weight: weight, eps: eps, onePlusW: onePlusW)
            for i in 0..<headDim { y[offset + i] = headY[i] }
            offset += headDim
        }
        return y
    }

    /// Gated per-head RMSNorm: per head of `headDim`,
    /// y[i] = (o[i] / sqrt(mean(o_h^2) + eps)) * w[i] * silu(z[i]).
    /// The `linear_attn.norm` weight is always plain `w` (plan 3.3, punto 1);
    /// silu runs in FP32, matching the MLX RMSNormGated golden (S8).
    public static func gatedPerHeadRMSNorm(o: [Float], z: [Float],
                                           weight: [Float],
                                           headDim: Int,
                                           eps: Float) -> [Float] {
        precondition(o.count == z.count, "o and z must match length")
        precondition(o.count % headDim == 0, "o must be a whole number of heads")
        precondition(weight.count == headDim, "weight must span one head")
        let normed = perHeadRMSNorm(x: o, weight: weight, headDim: headDim,
                                    eps: eps, onePlusW: false)
        var y = [Float](repeating: 0, count: o.count)
        for i in 0..<o.count {
            let zv = z[i]
            y[i] = normed[i] * (zv / (1.0 + exp(-zv)))
        }
        return y
    }

    /// Per-head L2 norm with a SUM (not mean) under the root:
    /// y[i] = x[i] / sqrt(sum(x_h^2) + eps).
    public static func perHeadL2Norm(x: [Float], headDim: Int, eps: Float) -> [Float] {
        precondition(x.count % headDim == 0, "x must be a whole number of heads")
        var y = [Float](repeating: 0, count: x.count)
        var offset = 0
        while offset < x.count {
            var sumSq: Float = 0
            x.withUnsafeBufferPointer { p in
                vDSP_svesq(p.baseAddress! + offset, 1, &sumSq, vDSP_Length(headDim))
            }
            let inv = 1.0 / (sumSq + eps).squareRoot()
            for i in 0..<headDim { y[offset + i] = x[offset + i] * inv }
            offset += headDim
        }
        return y
    }
}
