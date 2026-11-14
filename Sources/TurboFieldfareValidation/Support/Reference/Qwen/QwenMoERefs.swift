import Foundation
import TurboFieldfare

/// FP32 CPU references for the Qwen 3.5 MoE kernels (plan S17, math from
/// plan 3.5 "Blocco MoE", INT4 affine coding from contract C5):
///
///   act[slot][i] = silu(gate_i . x) * (up_i . x)      (FP16 rounding after
///                                                       the product)
///   routed[slot] = W_down_slot * act[slot]            (FP16 rounding)
///   y[d] = residual[d] + sum_k w_k * routed[k][d] + sg * shared[d]
///
/// The references bulk-dequantize each row with
/// `Quantization.dequantizeInt4Affine` and dot with `vDSP_dotpr` (via
/// `DequantInt4GemvRef`) — independent staging and summation order from the
/// kernel's per-group SIMD factoring, so the comparison catches dequant and
/// indexing bugs. FP16 round-trips are applied at the same intermediate
/// points as the kernel chain (act, routed/shared outputs, final y).
public enum QwenMoERef {

    public static func silu(_ v: Float) -> Float {
        return v / (1.0 + exp(-v))
    }

    /// Stage 1a for ONE expert: SiLU-multiply activations, FP16 rounded.
    /// `gate` and `up` are `inter` rows of `hidden` columns.
    public static func gateUpSilu(
        gate: [Quantization.Int4AffineRow],
        up: [Quantization.Int4AffineRow],
        x: [Float],
        hidden: Int
    ) -> [Float16] {
        precondition(gate.count == up.count, "gate/up row count mismatch")
        let g = DequantInt4GemvRef.apply(weightRows: gate, x: x, n: hidden)
        let u = DequantInt4GemvRef.apply(weightRows: up, x: x, n: hidden)
        return (0..<gate.count).map { Float16(silu(g[$0]) * u[$0]) }
    }

    /// Stage 1b for ONE expert: down projection of an FP16 activation, FP16
    /// rounded. `down` is `hidden` rows of `inter` columns.
    public static func down(
        downRows: [Quantization.Int4AffineRow],
        act: [Float16],
        inter: Int
    ) -> [Float16] {
        let a = act.map { Float($0) }
        let y = DequantInt4GemvRef.apply(weightRows: downRows, x: a, n: inter)
        return y.map { Float16($0) }
    }

    /// Stage 2: MoE block output for one token, FP32 accumulation over the
    /// FP16 intermediates, FP16 write. `routingW` is Float32 [slots] and
    /// `sg` the FP16 shared-expert scalar gate.
    public static func residual(
        residual: [Float16],
        routed: [[Float16]],
        shared: [Float16],
        routingW: [Float],
        sg: Float16
    ) -> [Float16] {
        precondition(routed.count == routingW.count,
                     "one routing weight per routed expert expected")
        let n = residual.count
        precondition(shared.count == n, "shared output width mismatch")
        for r in routed { precondition(r.count == n, "routed width mismatch") }
        var y = [Float16](repeating: 0, count: n)
        for d in 0..<n {
            var acc = Float(residual[d])
            for (k, r) in routed.enumerated() {
                acc += routingW[k] * Float(r[d])
            }
            acc += Float(sg) * Float(shared[d])
            y[d] = Float16(acc)
        }
        return y
    }
}
