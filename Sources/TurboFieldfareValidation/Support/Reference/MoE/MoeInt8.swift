import Foundation
import Accelerate
import TurboFieldfare

/// FP32 reference for the routed MoE FFN with MLX affine 8-bit experts: the
/// INT8 sibling of `MoeRef`.
///
/// Composed from `DequantInt8GemvRef` (bulk affine dequant, then
/// `vDSP_dotpr`) and `MoeRef.geluTanh`. The kernels fuse the byte unpack,
/// the per-group scale and bias and the gate/up/down chain, and round the
/// F-dim activations to FP16 between the two phases; this reference
/// materializes each projection in FP32 and keeps the activations in FP32.
/// Different staging and summation order, so a bug in either side does not
/// replicate.
public enum MoeInt8Ref {
    /// Runs one FFN block: `down(gelu(gate(x)) * up(x))`.
    /// `gateRows` / `upRows` are F-by-D affine INT8. `downRows` is D-by-F.
    public static func runFFN(
        gateRows: [Quantization.Int8AffineRow],
        upRows: [Quantization.Int8AffineRow],
        downRows: [Quantization.Int8AffineRow],
        x: [Float],
        d: Int,
        f: Int
    ) -> [Float] {
        precondition(gateRows.count == f, "gateRows must be F=\(f)")
        precondition(upRows.count == f, "upRows must be F=\(f)")
        precondition(downRows.count == d, "downRows must be D=\(d)")
        precondition(x.count == d)

        let gateOut = DequantInt8GemvRef.apply(weightRows: gateRows, x: x, n: d)
        let upOut = DequantInt8GemvRef.apply(weightRows: upRows, x: x, n: d)
        let gated = MoeRef.geluTanh(gateOut)
        var act = [Float](repeating: 0, count: f)
        vDSP_vmul(gated, 1, upOut, 1, &act, 1, vDSP_Length(f))
        return DequantInt8GemvRef.apply(weightRows: downRows, x: act, n: f)
    }

    /// `y = residual + Σ wₑ · routed_e(x)` over the selected experts, in slot
    /// order. The routed-only half of Gemma 4's parallel MoE block.
    public static func applyStreamedRouted(
        x: [Float],
        residual: [Float],
        routedGate: [[Quantization.Int8AffineRow]],
        routedUp: [[Quantization.Int8AffineRow]],
        routedDown: [[Quantization.Int8AffineRow]],
        indices: [Int],
        routingWeights: [Float],
        d: Int,
        f: Int
    ) -> [Float] {
        precondition(indices.count == routingWeights.count)
        precondition(residual.count == d)

        var y = residual
        for slot in 0..<indices.count {
            let e = indices[slot]
            let out = runFFN(
                gateRows: routedGate[e],
                upRows: routedUp[e],
                downRows: routedDown[e],
                x: x, d: d, f: f)
            var scale = routingWeights[slot]
            var scaled = [Float](repeating: 0, count: d)
            vDSP_vsmul(out, 1, &scale, &scaled, 1, vDSP_Length(d))
            vDSP_vadd(y, 1, scaled, 1, &y, 1, vDSP_Length(d))
        }
        return y
    }
}
