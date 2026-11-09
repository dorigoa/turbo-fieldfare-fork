import Foundation
import TurboFieldfareFormat

/// Qwen 3.5-MoE reader for `config.json`, selected by `ArchInfo.load` when the
/// root `model_type` is `qwen3_5_moe` (S5). The Gemma reader is untouched.
///
/// The reader is fail-closed: every value destined for `arch` (D1) or the
/// `qwen35` manifest block (C4) must equal the constant pinned from the D1/C4
/// tables and the S1 measurements, or the config is rejected. A checkpoint
/// that drifted from the pinned revision must fail at repack time rather than
/// write a manifest the runtime refuses at load (D2).
extension ArchInfo {

    static func loadQwen35(root: [String: Any], configPath: String) throws -> ArchInfo {
        guard let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        func fail(_ detail: String) -> RepackError {
            RepackError.configJsonInvalid(path: configPath, detail: detail)
        }
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw fail("missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw fail("missing \(k)")
            }
            return n
        }
        func b(_ k: String) throws -> Bool {
            guard let v = tc[k] as? Bool else { throw fail("missing \(k)") }
            return v
        }
        func s(_ k: String) throws -> String {
            guard let v = tc[k] as? String else { throw fail("missing \(k)") }
            return v
        }
        func pin<T: Equatable>(_ actual: T, _ expected: T, _ k: String) throws {
            guard actual == expected else {
                throw fail("\(k) = \(actual), expected \(expected)")
            }
        }

        guard (tc["model_type"] as? String) == Qwen35Pinned.textModelType else {
            throw fail("text_config.model_type != \(Qwen35Pinned.textModelType)")
        }

        // Geometry that lands in `manifest.json -> arch` (D1).
        try pin(try i("hidden_size"), Qwen35Pinned.hiddenSize, "hidden_size")
        try pin(try i("vocab_size"), Qwen35Pinned.vocabSize, "vocab_size")
        try pin(try i("num_hidden_layers"), Qwen35Pinned.numLayers, "num_hidden_layers")
        try pin(try i("shared_expert_intermediate_size"),
                Qwen35Pinned.ffnIntermediate, "shared_expert_intermediate_size")
        try pin(try i("moe_intermediate_size"),
                Qwen35Pinned.moeIntermediateSize, "moe_intermediate_size")
        try pin(try i("num_attention_heads"), Qwen35Pinned.numHeads, "num_attention_heads")
        try pin(try i("num_key_value_heads"), Qwen35Pinned.numKVHeads, "num_key_value_heads")
        try pin(try i("head_dim"), Qwen35Pinned.headDim, "head_dim")
        try pin(try b("tie_word_embeddings"), false, "tie_word_embeddings")
        try pin(try i("num_experts"), Qwen35Pinned.numExperts, "num_experts")
        try pin(try i("num_experts_per_tok"), Qwen35Pinned.topKExperts, "num_experts_per_tok")
        try pin(try s("hidden_act"), Qwen35Pinned.hiddenActivation, "hidden_act")
        try pin(try i("full_attention_interval"),
                Qwen35Pinned.fullAttentionInterval, "full_attention_interval")

        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        guard layerTypes.count == Qwen35Pinned.numLayers else {
            throw fail("layer_types has \(layerTypes.count) entries, expected \(Qwen35Pinned.numLayers)")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for (index, kind) in layerTypes.enumerated() {
            let isFull = kind == "full_attention"
            guard isFull || kind == "linear_attention" else {
                throw fail("layer_types[\(index)] = \(kind)")
            }
            guard isFull == (index % Qwen35Pinned.fullAttentionInterval
                                == Qwen35Pinned.fullAttentionInterval - 1) else {
                throw fail("layer_types[\(index)] breaks the \(Qwen35Pinned.fullAttentionInterval)-layer attention interval")
            }
            mask.append(isFull ? 1 : 0)
        }

        let rope = (root["rope_parameters"] as? [String: Any])
            ?? (tc["rope_parameters"] as? [String: Any]) ?? [:]
        try pin((rope["rope_theta"] as? NSNumber)?.doubleValue ?? .nan,
                Qwen35Pinned.ropeTheta, "rope_parameters.rope_theta")
        try pin((rope["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? .nan,
                Qwen35Pinned.partialRotaryFactor, "rope_parameters.partial_rotary_factor")

        // Hybrid block destined for `manifest.json -> qwen35` (C4).
        try pin(try i("linear_num_key_heads"),
                Qwen35Pinned.linearNumKeyHeads, "linear_num_key_heads")
        try pin(try i("linear_num_value_heads"),
                Qwen35Pinned.linearNumValueHeads, "linear_num_value_heads")
        try pin(try i("linear_key_head_dim"),
                Qwen35Pinned.linearKeyHeadDim, "linear_key_head_dim")
        try pin(try i("linear_value_head_dim"),
                Qwen35Pinned.linearValueHeadDim, "linear_value_head_dim")
        try pin(try i("linear_conv_kernel_dim"),
                Qwen35Pinned.linearConvKernel, "linear_conv_kernel_dim")
        try pin(try d("rms_norm_eps"), Qwen35Pinned.rmsNormEps, "rms_norm_eps")
        try pin(try b("attn_output_gate"),
                Qwen35Pinned.attentionOutputGate, "attn_output_gate")

        return ArchInfo(
            hiddenSize: Qwen35Pinned.hiddenSize,
            intermediateSize: Qwen35Pinned.ffnIntermediate,
            moeIntermediateSize: Qwen35Pinned.moeIntermediateSize,
            numHeads: Qwen35Pinned.numHeads,
            numKVHeads: Qwen35Pinned.numKVHeads,
            numFullKVHeads: Qwen35Pinned.numFullKVHeads,
            headDim: Qwen35Pinned.headDim,
            fullHeadDim: Qwen35Pinned.fullHeadDim,
            vocabSize: Qwen35Pinned.vocabSize,
            slidingWindow: Qwen35Pinned.slidingWindow,
            finalLogitSoftcap: Qwen35Pinned.finalLogitSoftcap,
            ropeTheta: Qwen35Pinned.ropeTheta,
            fullRopeTheta: Qwen35Pinned.fullRopeTheta,
            partialRotaryFactor: Qwen35Pinned.partialRotaryFactor,
            numLayers: Qwen35Pinned.numLayers,
            numExperts: Qwen35Pinned.numExperts,
            topKExperts: Qwen35Pinned.topKExperts,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: Qwen35Pinned.hiddenActivation,
            modelFamily: GTurboModelFamily.qwen35Moe.rawValue,
            qwen35: GTurboManifestQwen35V1(
                linearNumKeyHeads: Qwen35Pinned.linearNumKeyHeads,
                linearNumValueHeads: Qwen35Pinned.linearNumValueHeads,
                linearKeyHeadDim: Qwen35Pinned.linearKeyHeadDim,
                linearValueHeadDim: Qwen35Pinned.linearValueHeadDim,
                linearConvKernel: Qwen35Pinned.linearConvKernel,
                rmsNormEps: Qwen35Pinned.rmsNormEps,
                normWeightsShifted: Qwen35Pinned.normWeightsShifted,
                attentionOutputGate: Qwen35Pinned.attentionOutputGate,
                sharedExpertGate: Qwen35Pinned.sharedExpertGate))
    }
}

/// Values pinned from the plan (D1, C4) and the S1 measurements. A real
/// `config.json` must agree with every one of them.
private enum Qwen35Pinned {
    static let textModelType = "qwen3_5_moe_text"
    // `arch` (D1).
    static let hiddenSize = 2048
    static let vocabSize = 248_320
    static let numLayers = 40
    static let ffnIntermediate = 512
    static let moeIntermediateSize = 512
    static let numHeads = 16
    static let numKVHeads = 2
    static let numFullKVHeads = 2
    static let headDim = 256
    static let fullHeadDim = 256
    static let slidingWindow = 0
    static let finalLogitSoftcap = 0.0
    static let ropeTheta = 10_000_000.0
    static let fullRopeTheta = 10_000_000.0
    static let partialRotaryFactor = 0.25
    static let numExperts = 256
    static let topKExperts = 8
    static let hiddenActivation = "silu"
    static let fullAttentionInterval = 4
    // Hybrid block (C4). `normWeightsShifted` is the S1 measurement (mean
    // input_layernorm 1.031): the MLX conversion already added the +1 offset,
    // so the runtime applies `x̂ · w`. `sharedExpertGate` has no config key;
    // the checkpoint ships a `shared_expert_gate` tensor, so C4 pins it true.
    static let linearNumKeyHeads = 16
    static let linearNumValueHeads = 32
    static let linearKeyHeadDim = 128
    static let linearValueHeadDim = 128
    static let linearConvKernel = 4
    static let rmsNormEps = 1e-6
    static let normWeightsShifted = true
    static let attentionOutputGate = true
    static let sharedExpertGate = true
}
