import Foundation
import TurboFieldfareFormat

// Qwen 3.5-MoE runtime schema validation (S18, D2). `Model.load` dispatches
// here when the manifest carries `modelFamily == "qwen3_5_moe"`. The manifest
// `arch`/`qwen35` blocks were already pinned against `ArchConfig.qwen36_35B_A3B`
// and `QwenHybridConfig.qwen36_35B_A3B` by `ManifestReader.validate` (S3);
// this function validates the resident index against the same constants:
// every tensor of plan section 3.3, with the shapes of section 3.4.
// Gemma's `validateRuntimeSchema` is untouched.
extension Model {

    /// Model family from the manifest (D1). Absence of `modelFamily` means
    /// Gemma; the format codec has already rejected unknown family strings.
    public var family: ModelFamily {
        Model.family(of: manifest)
    }

    static func family(of manifest: Manifest) -> ModelFamily {
        guard let name = manifest.modelFamily else { return .gemma4 }
        return ModelFamily(rawValue: name) == .qwen35Moe ? .qwen35Moe : .gemma4
    }

    // MARK: - Qwen resident accessors (3.3)

    /// The pinned hybrid block for a loaded Qwen model (C4/D2); `nil` for
    /// Gemma. `ManifestReader.validate` already checked `manifest.qwen35`
    /// field-by-field against this constant at load time.
    public var qwenHybridConfig: QwenHybridConfig? {
        guard family == .qwen35Moe else { return nil }
        return .qwen36_35B_A3B
    }

    /// Qwen does not tie the embeddings: the LM head is its own quantized
    /// tensor, validated against the embedding slot at load time.
    /// Valid only for Qwen models.
    public var qwenLMHead: TensorView {
        try! resident(name: "language_model.lm_head.weight")
    }

    public func qwenInProjQKV(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_qkv.weight")
    }
    public func qwenInProjZ(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_z.weight")
    }
    public func qwenInProjA(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_a.weight")
    }
    public func qwenInProjB(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_b.weight")
    }
    public func qwenConv1D(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.conv1d.weight")
    }
    public func qwenALog(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.A_log")
    }
    public func qwenDtBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.dt_bias")
    }
    public func qwenDeltaNetNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.norm.weight")
    }
    public func qwenOutProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.out_proj.weight")
    }

    /// MoE router (`mlp.gate`, 8-bit slot), not to be confused with Gemma's
    /// `router(layer:)` (`router.proj`) nor with `qwenSharedExpertGateScalar`.
    public func qwenRouter(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.gate.weight")
    }
    public func qwenSharedExpertGate(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert.gate_proj.weight")
    }
    public func qwenSharedExpertUp(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert.up_proj.weight")
    }
    public func qwenSharedExpertDown(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert.down_proj.weight")
    }
    /// The scalar shared-expert gate `mlp.shared_expert_gate` (1×hidden).
    public func qwenSharedExpertGateScalar(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert_gate.weight")
    }

    static func validateQwenRuntimeSchema(residentIndex: ResidentIndex,
                                          layout: PackedExpertsLayout,
                                          manifest: Manifest,
                                          config: ArchConfig,
                                          hybrid: QwenHybridConfig) throws {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required by the executable runtime schema")
        }
        let profile = try QuantProfile(manifestQuant: quant)
        guard config.fullAttentionLayerMask.count == config.numLayers else {
            throw ModelError.indexCorrupt(
                detail: "fullAttentionLayerMask does not cover every layer")
        }

        func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) byte count overflows UInt64")
            }
            return value
        }

        func checkedIntMultiply(_ lhs: Int, _ rhs: Int, field: String) throws -> Int {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) dimension overflows Int")
            }
            return value
        }

        func requireBF16(_ name: String, dims: [Int]) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            guard !dims.isEmpty, dims.count <= 4, dims.allSatisfy({ $0 > 0 }) else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            var shape: [UInt32] = dims.compactMap { UInt32(exactly: $0) }
            guard shape.count == dims.count else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            var elementCount: UInt64 = 1
            for dimension in shape {
                elementCount = try checkedMultiply(elementCount, UInt64(dimension), field: name)
            }
            while shape.count < 4 { shape.append(0) }
            let expectedBytes = try checkedMultiply(
                elementCount, UInt64(MemoryLayout<UInt16>.size), field: name)
            guard entry.dtype == GTurboFormatV1.DType.bf16.rawValue,
                  entry.shape.0 == shape[0],
                  entry.shape.1 == shape[1],
                  entry.shape.2 == shape[2],
                  entry.shape.3 == shape[3],
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) does not match the required BF16 schema")
            }
        }

        func affineSizes(rows: Int,
                         columns: Int,
                         slot: ManifestQuantSlot,
                         field: String) throws -> (shape: (UInt32, UInt32), weight: UInt64, aux: UInt64) {
            guard let r = UInt32(exactly: rows), let c = UInt32(exactly: columns),
                  r > 0, c > 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has invalid dimensions")
            }
            guard slot.weightBits == 4 || slot.weightBits == 8,
                  slot.groupSize > 0,
                  columns % slot.groupSize == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has unsupported affine quantization")
            }
            let elements = try checkedMultiply(UInt64(rows), UInt64(columns), field: field)
            let bitCount = try checkedMultiply(elements, UInt64(slot.weightBits), field: field)
            guard bitCount % 8 == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) packed byte count is fractional")
            }
            let groups = UInt64(columns / slot.groupSize)
            let auxElements = try checkedMultiply(UInt64(r), groups, field: field)
            let auxBytes = try checkedMultiply(
                auxElements, UInt64(MemoryLayout<UInt16>.size), field: field)
            return (shape: (r, c), weight: bitCount / 8, aux: auxBytes)
        }

        func requireAffine(_ name: String,
                           rows: Int,
                           columns: Int,
                           slot: ManifestQuantSlot) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            let expected = try affineSizes(
                rows: rows, columns: columns, slot: slot, field: name)
            let primaryAlignment: UInt64 = slot.weightBits == 4
                ? UInt64(MemoryLayout<UInt16>.alignment)
                : 1
            guard entry.dtype == GTurboFormatV1.DType.u32.rawValue,
                  entry.shape.0 == expected.shape.0,
                  entry.shape.1 == expected.shape.1,
                  entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expected.weight,
                  entry.scaleSize == expected.aux,
                  entry.biasSize == expected.aux,
                  entry.fileOffset % primaryAlignment == 0,
                  entry.scaleOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0,
                  entry.biasOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) affine metadata mismatch: dtype=\(entry.dtype), shape=[\(entry.shape.0),\(entry.shape.1),\(entry.shape.2),\(entry.shape.3)], bytes=\(entry.sizeBytes), scales=\(entry.scaleSize), biases=\(entry.biasSize), expected shape=[\(expected.shape.0),\(expected.shape.1),0,0], bytes=\(expected.weight), aux=\(expected.aux)")
            }
            if profile == .affine8 {
                let alignment = affine8ResidentWeightAlignment
                guard entry.fileOffset % alignment == 0,
                      residentIndex.header.indexSize % alignment == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "\(name) starts at byte \(entry.fileOffset) of model_weights.bin "
                            + "(resident payload at \(residentIndex.header.indexSize)); the 8-bit "
                            + "kernels need every quantized resident weight on a "
                            + "\(alignment)-byte boundary.")
                }
            }
        }

        // Top level. Qwen does not tie the embeddings: `lm_head` is its own
        // quantized tensor, on the embedding slot (3.3).
        try requireAffine("language_model.model.embed_tokens.weight",
                          rows: config.vocabSize, columns: config.hiddenSize,
                          slot: quant.embedding)
        try requireAffine("language_model.lm_head.weight",
                          rows: config.vocabSize, columns: config.hiddenSize,
                          slot: quant.embedding)
        try requireBF16("language_model.model.norm.weight", dims: [config.hiddenSize])

        let valueDim = try checkedIntMultiply(
            hybrid.linearNumValueHeads, hybrid.linearValueHeadDim, field: "linear value dim")
        let linearKeyRows = try checkedIntMultiply(
            hybrid.linearNumKeyHeads, hybrid.linearKeyHeadDim, field: "linear key dim")
        let (qkvDoubled, qkvMulOverflow) = linearKeyRows.multipliedReportingOverflow(by: 2)
        let (qkvRows, qkvAddOverflow) = qkvDoubled.addingReportingOverflow(valueDim)
        guard !qkvMulOverflow, !qkvAddOverflow else {
            throw ModelError.indexCorrupt(detail: "linear qkv dimension overflows Int")
        }
        let queryProjectionRows = try checkedIntMultiply(
            config.numHeads, config.fullHeadDim, field: "attention query dim")
        let kvProjectionRows = try checkedIntMultiply(
            config.numKVHeads, config.fullHeadDim, field: "attention key/value dim")
        let gateFactor = hybrid.attentionOutputGate ? 2 : 1

        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            try requireBF16("\(prefix).input_layernorm.weight", dims: [config.hiddenSize])
            try requireBF16("\(prefix).post_attention_layernorm.weight",
                            dims: [config.hiddenSize])

            if config.fullAttentionLayerMask[layer] != 0 {
                // Full attention (3.4): q_proj carries the output gate, so it
                // has 2 · numHeads · fullHeadDim rows when
                // `qwen35.attentionOutputGate` is set.
                try requireAffine("\(prefix).self_attn.q_proj.weight",
                                  rows: queryProjectionRows * gateFactor,
                                  columns: config.hiddenSize,
                                  slot: quant.attention)
                try requireAffine("\(prefix).self_attn.k_proj.weight",
                                  rows: kvProjectionRows, columns: config.hiddenSize,
                                  slot: quant.attention)
                try requireAffine("\(prefix).self_attn.v_proj.weight",
                                  rows: kvProjectionRows, columns: config.hiddenSize,
                                  slot: quant.attention)
                try requireAffine("\(prefix).self_attn.o_proj.weight",
                                  rows: config.hiddenSize, columns: queryProjectionRows,
                                  slot: quant.attention)
                try requireBF16("\(prefix).self_attn.q_norm.weight",
                                dims: [config.fullHeadDim])
                try requireBF16("\(prefix).self_attn.k_norm.weight",
                                dims: [config.fullHeadDim])
            } else {
                // Linear attention (Gated DeltaNet, 3.4).
                try requireAffine("\(prefix).linear_attn.in_proj_qkv.weight",
                                  rows: qkvRows, columns: config.hiddenSize,
                                  slot: quant.attention)
                try requireAffine("\(prefix).linear_attn.in_proj_z.weight",
                                  rows: valueDim, columns: config.hiddenSize,
                                  slot: quant.attention)
                try requireAffine("\(prefix).linear_attn.in_proj_a.weight",
                                  rows: hybrid.linearNumValueHeads,
                                  columns: config.hiddenSize,
                                  slot: quant.attention)
                try requireAffine("\(prefix).linear_attn.in_proj_b.weight",
                                  rows: hybrid.linearNumValueHeads,
                                  columns: config.hiddenSize,
                                  slot: quant.attention)
                try requireBF16("\(prefix).linear_attn.conv1d.weight",
                                dims: [qkvRows, hybrid.linearConvKernel, 1])
                try requireBF16("\(prefix).linear_attn.A_log",
                                dims: [hybrid.linearNumValueHeads])
                try requireBF16("\(prefix).linear_attn.dt_bias",
                                dims: [hybrid.linearNumValueHeads])
                try requireBF16("\(prefix).linear_attn.norm.weight",
                                dims: [hybrid.linearValueHeadDim])
                try requireAffine("\(prefix).linear_attn.out_proj.weight",
                                  rows: config.hiddenSize, columns: valueDim,
                                  slot: quant.attention)
            }

            // MoE block, every layer (3.4).
            try requireAffine("\(prefix).mlp.gate.weight",
                              rows: config.numExperts, columns: config.hiddenSize,
                              slot: quant.router)
            try requireAffine("\(prefix).mlp.shared_expert.gate_proj.weight",
                              rows: config.intermediateSize, columns: config.hiddenSize,
                              slot: quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.up_proj.weight",
                              rows: config.intermediateSize, columns: config.hiddenSize,
                              slot: quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.down_proj.weight",
                              rows: config.hiddenSize, columns: config.intermediateSize,
                              slot: quant.sharedExpert)
            if hybrid.sharedExpertGate {
                try requireAffine("\(prefix).mlp.shared_expert_gate.weight",
                                  rows: 1, columns: config.hiddenSize,
                                  slot: quant.router)
            }
        }

        // Routed experts: identical roles and uniformity to the Gemma schema.
        let routedShapes: [(String, Int, Int)] = [
            ("gate", config.moeIntermediateSize, config.hiddenSize),
            ("up", config.moeIntermediateSize, config.hiddenSize),
            ("down", config.hiddenSize, config.moeIntermediateSize),
        ]
        for layer in layout.layers {
            guard let reference = layer.experts.first else {
                throw ModelError.indexCorrupt(
                    detail: "routed layer \(layer.layer) has no experts")
            }
            for (role, rows, columns) in routedShapes {
                let sizes = try affineSizes(
                    rows: rows, columns: columns,
                    slot: quant.routedExpert,
                    field: "routed layer \(layer.layer) \(role)")
                let expectedRoles: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
                    (role, "U32", [sizes.shape.0, sizes.shape.1],
                     quant.routedExpert.weightBits, sizes.weight,
                     UInt64(MemoryLayout<UInt32>.alignment)),
                    ("\(role)_scales", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                    ("\(role)_biases", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                ]
                for (name, dtype, shape, bits, size, alignment) in expectedRoles {
                    guard let expected = reference.subTensors[name] else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) is missing role \(name)")
                    }
                    let (end, overflow) = expected.offset.addingReportingOverflow(expected.size)
                    guard expected.dtype == dtype,
                          expected.shape == shape,
                          expected.bits == bits,
                          expected.size == size,
                          expected.offset % alignment == 0,
                          !overflow,
                          end <= reference.size,
                          end <= UInt64(UInt32.max) + 1 else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) does not match the required schema")
                    }
                    for expert in layer.experts.dropFirst()
                        where expert.subTensors[name] != expected {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) metadata differs across experts")
                    }
                }
            }
        }
    }
}
