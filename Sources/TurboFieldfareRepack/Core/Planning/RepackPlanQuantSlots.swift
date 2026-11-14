import Foundation

/// The manifest's five quantization slots, derived from the tensors a plan
/// actually copies. Nothing is assumed: a slot with no quantized tensor, a slot
/// whose tensors disagree on bit width, or a quantized resident tensor that
/// belongs to no slot all fail the install before a manifest is written.
enum RepackQuantSlot: String, CaseIterable, Sendable {
    case embedding
    case attention
    case router
    case sharedExpert
    case routedExpert

    /// The slot a quantized resident weight belongs to, or nil. Gemma 4 and
    /// Qwen (S6) names are disjoint, so one mapping serves both families:
    /// Qwen adds `lm_head` to the embedding, the `linear_attn` projections to
    /// the attention, `mlp.gate`/`mlp.shared_expert_gate` to the router, and
    /// `mlp.shared_expert.{gate,up,down}_proj` to the shared expert.
    static func residentSlot(for name: String) -> RepackQuantSlot? {
        if name == "language_model.model.embed_tokens.weight"
            || name == "language_model.lm_head.weight" {
            return .embedding
        }
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"]
        where name.hasSuffix(".self_attn.\(projection).weight") {
            return .attention
        }
        for projection in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
        where name.hasSuffix(".linear_attn.\(projection).weight") {
            return .attention
        }
        if name.hasSuffix(".router.proj.weight")
            || name.hasSuffix(".mlp.gate.weight")
            || name.hasSuffix(".mlp.shared_expert_gate.weight") {
            return .router
        }
        for projection in ["gate_proj", "up_proj", "down_proj"]
        where name.hasSuffix(".mlp.\(projection).weight")
            || name.hasSuffix(".mlp.shared_expert.\(projection).weight") {
            return .sharedExpert
        }
        return nil
    }
}

extension RepackPlan {
    /// Weight bits per manifest slot, read from every quantized tensor the plan
    /// copies.
    func manifestQuantBitWidths() throws -> GTurboJSON.QuantBitWidths {
        var bitsBySlot: [RepackQuantSlot: Int] = [:]
        func record(_ bits: Int, slot: RepackQuantSlot, tensor: String) throws {
            if let previous = bitsBySlot.updateValue(bits, forKey: slot), previous != bits {
                throw RepackError.configurationInvalid(
                    detail: "quantization slot \(slot.rawValue) mixes \(previous)-bit and "
                        + "\(bits)-bit weights (\(tensor))")
            }
        }

        for entry in resident.entries {
            guard let spec = entry.quantSpec else { continue }
            guard let slot = RepackQuantSlot.residentSlot(for: entry.name) else {
                throw RepackError.configurationInvalid(
                    detail: "quantized resident tensor \(entry.name) has no manifest quant slot")
            }
            try record(spec.bits, slot: slot, tensor: entry.name)
        }
        for layer in layers {
            for slice in layer.subTensors where slice.component == "weights" {
                guard let bits = slice.bitsForWeights else {
                    throw RepackError.configurationInvalid(
                        detail: "layer \(layer.layerIndex) \(slice.role) weights carry no bit width")
                }
                try record(bits, slot: .routedExpert, tensor: slice.sourceTensor.name)
            }
        }

        func bits(_ slot: RepackQuantSlot) throws -> Int {
            guard let value = bitsBySlot[slot] else {
                throw RepackError.configurationInvalid(
                    detail: "no quantized tensor supplies manifest quant slot \(slot.rawValue)")
            }
            return value
        }
        return GTurboJSON.QuantBitWidths(
            embedding: try bits(.embedding),
            attention: try bits(.attention),
            router: try bits(.router),
            sharedExpert: try bits(.sharedExpert),
            routedExpert: try bits(.routedExpert))
    }
}
