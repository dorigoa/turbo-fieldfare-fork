import Foundation

/// Compile-time baseline for the Qwen 3.5-MoE hybrid block (D2, C4).
/// `manifest.json -> qwen35` must match this field-by-field at load time;
/// mismatches throw `ModelError.archMismatch` with the `qwen35.`-prefixed
/// field name, the same gate `ArchConfig` provides for `arch`.
public struct QwenHybridConfig: Sendable, Equatable {
    public let linearNumKeyHeads: Int
    public let linearNumValueHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearConvKernel: Int
    public let rmsNormEps: Double
    /// True when the checkpoint's norm weights already carry the +1 offset,
    /// so the runtime applies `x̂ · w` instead of `x̂ · (1 + w)`. Measured in
    /// S1 against the pinned checkpoint.
    public let normWeightsShifted: Bool
    public let attentionOutputGate: Bool
    public let sharedExpertGate: Bool

    public init(
        linearNumKeyHeads: Int,
        linearNumValueHeads: Int,
        linearKeyHeadDim: Int,
        linearValueHeadDim: Int,
        linearConvKernel: Int,
        rmsNormEps: Double,
        normWeightsShifted: Bool,
        attentionOutputGate: Bool,
        sharedExpertGate: Bool
    ) {
        self.linearNumKeyHeads = linearNumKeyHeads
        self.linearNumValueHeads = linearNumValueHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernel = linearConvKernel
        self.rmsNormEps = rmsNormEps
        self.normWeightsShifted = normWeightsShifted
        self.attentionOutputGate = attentionOutputGate
        self.sharedExpertGate = sharedExpertGate
    }

    /// Canonical Qwen 3.5-MoE 35B-A3B hybrid block, checked against the
    /// installed model manifest (C4).
    public static let qwen36_35B_A3B = QwenHybridConfig(
        linearNumKeyHeads: 16,
        linearNumValueHeads: 32,
        linearKeyHeadDim: 128,
        linearValueHeadDim: 128,
        linearConvKernel: 4,
        rmsNormEps: 1e-6,
        normWeightsShifted: true,
        attentionOutputGate: true,
        sharedExpertGate: true
    )
}
