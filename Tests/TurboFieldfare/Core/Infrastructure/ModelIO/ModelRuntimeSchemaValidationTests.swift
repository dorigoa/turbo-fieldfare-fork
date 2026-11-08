import Metal
import Testing

@testable import TurboFieldfare

@Suite struct ModelRuntimeSchemaValidationTests {
    @Test func rejectsMissingRequiredResidentTensor() throws {
        let model = try loadToy()
        var entries = model.residentIndex.entries
        entries["language_model.model.embed_tokens.weight"] = nil
        let index = ResidentIndex(header: model.residentIndex.header, entries: entries)

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: index,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsMissingRoutedTensorRole() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 0) { expert in
            var tensors = expert.subTensors
            tensors["gate"] = nil
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsNonuniformRoutedTensorMetadata() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 1) { expert in
            var tensors = expert.subTensors
            let gate = tensors["gate"]!
            tensors["gate"] = SubTensorEntry(
                offset: gate.offset + 1,
                size: gate.size,
                dtype: gate.dtype,
                shape: gate.shape,
                bits: gate.bits)
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsRoutedTensorRangeOutsideUInt32() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 0) { expert in
            var tensors = expert.subTensors
            let gate = tensors["gate"]!
            tensors["gate"] = SubTensorEntry(
                offset: UInt64(UInt32.max),
                size: 2,
                dtype: gate.dtype,
                shape: gate.shape,
                bits: gate.bits)
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsMalformedResidentAffineMetadata() throws {
        let model = try loadToy()
        let name = "language_model.model.embed_tokens.weight"
        let entry = try #require(model.residentIndex.entries[name])
        let invalidEntries = [
            copy(entry, dtype: 1),
            copy(entry, shape: (entry.shape.0, entry.shape.1 + 1, 0, 0)),
            copy(entry, sizeBytes: entry.sizeBytes + 1),
            copy(entry, scaleSize: entry.scaleSize + 2),
            copy(entry, fileOffset: entry.fileOffset + 1),
            copy(entry, scaleOffset: entry.scaleOffset + 1),
            copy(entry, biasSize: entry.biasSize + 2),
            copy(entry, biasOffset: entry.biasOffset + 1),
        ]

        for invalid in invalidEntries {
            var entries = model.residentIndex.entries
            entries[name] = invalid
            let index = ResidentIndex(header: model.residentIndex.header, entries: entries)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: index,
                    layout: model.packedExpertsLayout,
                    manifest: model.manifest,
                    config: .gemma4Toy())
            }
        }
    }

    @Test func rejectsMalformedRoutedAffineMetadata() throws {
        let model = try loadToy()
        let gate = try #require(model.packedExpertsLayout.layers[0].experts[0].subTensors["gate"])
        let invalidEntries = [
            SubTensorEntry(offset: gate.offset, size: gate.size, dtype: "BF16",
                           shape: gate.shape, bits: gate.bits),
            SubTensorEntry(offset: gate.offset, size: gate.size, dtype: gate.dtype,
                           shape: [gate.shape[0], gate.shape[1] + 1], bits: gate.bits),
            SubTensorEntry(offset: gate.offset, size: gate.size, dtype: gate.dtype,
                           shape: gate.shape, bits: 8),
            SubTensorEntry(offset: gate.offset, size: gate.size + 1, dtype: gate.dtype,
                           shape: gate.shape, bits: gate.bits),
            SubTensorEntry(offset: gate.offset + 1, size: gate.size, dtype: gate.dtype,
                           shape: gate.shape, bits: gate.bits),
        ]

        for invalid in invalidEntries {
            let layout = replacingRoleForAllExperts(
                model.packedExpertsLayout, layer: 0, role: "gate", with: invalid)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: model.residentIndex,
                    layout: layout,
                    manifest: model.manifest,
                    config: .gemma4Toy())
            }
        }
    }

    @Test func rejectsExecutableLoadWithoutQuantMetadata() throws {
        let directory = try ModelLoaderTests.writeToySynthetic(includeQuant: false)
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: ModelError.self) {
            try Model.load(directoryURL: directory, device: device, expecting: .gemma4Toy())
        }
    }

    @Test func rejectsOverflowingDerivedDimensions() throws {
        let model = try loadToy()
        let toy = ArchConfig.gemma4Toy()
        let config = ArchConfig(
            hiddenSize: toy.hiddenSize,
            intermediateSize: toy.intermediateSize,
            moeIntermediateSize: toy.moeIntermediateSize,
            numHeads: Int.max,
            numKVHeads: toy.numKVHeads,
            numFullKVHeads: toy.numFullKVHeads,
            headDim: 2,
            fullHeadDim: toy.fullHeadDim,
            vocabSize: toy.vocabSize,
            slidingWindow: toy.slidingWindow,
            finalLogitSoftcap: toy.finalLogitSoftcap,
            ropeTheta: toy.ropeTheta,
            fullRopeTheta: toy.fullRopeTheta,
            partialRotaryFactor: toy.partialRotaryFactor,
            numLayers: toy.numLayers,
            numExperts: toy.numExperts,
            topKExperts: toy.topKExperts,
            tieWordEmbeddings: toy.tieWordEmbeddings,
            attentionKEqV: toy.attentionKEqV,
            fullAttentionLayerMask: toy.fullAttentionLayerMask,
            hiddenActivation: toy.hiddenActivation)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: config)
        }
    }

    // MARK: - affine8 resident alignment (D11)

    /// One quantized resident weight per manifest slot the rule covers.
    private static let affine8QuantizedResidentWeights = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.1.self_attn.o_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.1.router.proj.weight",
    ]

    @Test func acceptsAnAffine8InstallAlignedToSixteenBytes() throws {
        let directory = try AffineToyModel.write(profile: .affine8)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: directory, device: device,
                                   expecting: AffineToyModel.config)
        #expect(model.quantProfile == .affine8)
        let quantized = model.residentIndex.entries.values.filter { $0.dtype == 0 }
        #expect(!quantized.isEmpty)
        #expect(quantized.allSatisfy {
            $0.fileOffset % Model.affine8ResidentWeightAlignment == 0
        })
    }

    @Test(arguments: affine8QuantizedResidentWeights)
    func rejectsAnAffine8ResidentWeightOffTheSixteenByteGrid(name: String) throws {
        let model = try loadAffineToy(.affine8)
        for shift: UInt64 in [2, 4, 8] {
            let index = try shiftingResidentWeight(model.residentIndex, name: name, by: shift)
            let detail = try #require(schemaFailure(
                residentIndex: index, layout: model.packedExpertsLayout,
                manifest: model.manifest, config: AffineToyModel.config))
            #expect(detail.contains(name))
            #expect(detail.contains("16-byte"))
            #expect(detail.contains("reinstall"))
        }
    }

    @Test func refusesToLoadAMisalignedAffine8Install() throws {
        let name = "language_model.model.layers.0.self_attn.k_proj.weight"
        let directory = try AffineToyModel.write(profile: .affine8, shiftedResidentWeight: name)
        let device = try #require(MTLCreateSystemDefaultDevice())
        do {
            _ = try Model.load(directoryURL: directory, device: device,
                               expecting: AffineToyModel.config)
            Issue.record("a misaligned affine8 install loaded")
        } catch let error as ModelError {
            let detail = String(describing: error)
            #expect(detail.contains(name))
            #expect(detail.contains("reinstall"))
        }
    }

    /// The rule is `affine8` only: a 4-bit install keeps its 2-byte weight
    /// grid, 4-bit weights off the 16-byte grid still load, and an 8-bit
    /// router in an `affine4` model keeps its byte grid.
    @Test func affine4KeepsItsResidentAlignmentRules() throws {
        let model = try loadAffineToy(.affine4)
        let projection = "language_model.model.layers.0.self_attn.q_proj.weight"
        let router = "language_model.model.layers.0.router.proj.weight"
        for (name, shift) in [(projection, UInt64(2)), (projection, 4), (router, 1), (router, 4)] {
            let index = try shiftingResidentWeight(model.residentIndex, name: name, by: shift)
            #expect(schemaFailure(residentIndex: index, layout: model.packedExpertsLayout,
                                      manifest: model.manifest,
                                      config: AffineToyModel.config) == nil,
                    "\(name) shifted by \(shift)")
        }
        let odd = try shiftingResidentWeight(model.residentIndex, name: projection, by: 1)
        #expect(schemaFailure(residentIndex: odd, layout: model.packedExpertsLayout,
                                  manifest: model.manifest,
                                  config: AffineToyModel.config) != nil)
    }

    /// The routed checks the INT8 routed kernels rely on (weights on 4 bytes,
    /// scales and biases on 2) already hold for both widths.
    @Test func rejectsAffine8RoutedRolesOffTheirGrid() throws {
        let model = try loadAffineToy(.affine8)
        let gate = try #require(model.packedExpertsLayout.layers[0].experts[0].subTensors["gate"])
        let scales = try #require(
            model.packedExpertsLayout.layers[0].experts[0].subTensors["gate_scales"])
        for (role, entry) in [
            ("gate", SubTensorEntry(offset: gate.offset + 2, size: gate.size, dtype: gate.dtype,
                                    shape: gate.shape, bits: gate.bits)),
            ("gate_scales", SubTensorEntry(offset: scales.offset + 1, size: scales.size,
                                           dtype: scales.dtype, shape: scales.shape,
                                           bits: scales.bits)),
        ] {
            let layout = replacingRoleForAllExperts(model.packedExpertsLayout,
                                                    layer: 0, role: role, with: entry)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: model.residentIndex,
                    layout: layout,
                    manifest: model.manifest,
                    config: AffineToyModel.config)
            }
        }
    }

    private func loadAffineToy(_ profile: QuantProfile) throws -> Model {
        let directory = try AffineToyModel.write(profile: profile)
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try Model.load(directoryURL: directory, device: device,
                              expecting: AffineToyModel.config)
    }

    /// Moves one resident weight forward by `shift` bytes, keeping its size,
    /// scales and biases, so only its alignment changes.
    private func shiftingResidentWeight(_ index: ResidentIndex,
                                        name: String,
                                        by shift: UInt64) throws -> ResidentIndex {
        let entry = try #require(index.entries[name])
        var entries = index.entries
        entries[name] = copy(entry, fileOffset: entry.fileOffset + shift)
        return ResidentIndex(header: index.header, entries: entries)
    }

    /// The schema failure's text, or nil when the schema is accepted.
    private func schemaFailure(residentIndex: ResidentIndex,
                               layout: PackedExpertsLayout,
                               manifest: Manifest,
                               config: ArchConfig) -> String? {
        do {
            try Model.validateRuntimeSchema(residentIndex: residentIndex,
                                            layout: layout,
                                            manifest: manifest,
                                            config: config)
            return nil
        } catch {
            return String(describing: error)
        }
    }

    private func loadToy() throws -> Model {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try Model.load(
            directoryURL: directory,
            device: device,
            expecting: .gemma4Toy())
    }

    private func replacingExpert(
        _ layout: PackedExpertsLayout,
        layer layerIndex: Int,
        expert expertIndex: Int,
        transform: (ExpertEntry) -> ExpertEntry
    ) -> PackedExpertsLayout {
        var layers = layout.layers
        let layer = layers[layerIndex]
        var experts = layer.experts
        experts[expertIndex] = transform(experts[expertIndex])
        layers[layerIndex] = LayerLayout(
            layer: layer.layer,
            file: layer.file,
            experts: experts)
        return PackedExpertsLayout(
            expertStride: layout.expertStride,
            numLayers: layout.numLayers,
            expertsPerLayer: layout.expertsPerLayer,
            layers: layers)
    }

    private func replacingRoleForAllExperts(
        _ layout: PackedExpertsLayout,
        layer layerIndex: Int,
        role: String,
        with replacement: SubTensorEntry
    ) -> PackedExpertsLayout {
        var layers = layout.layers
        let layer = layers[layerIndex]
        let experts = layer.experts.map { expert in
            var tensors = expert.subTensors
            tensors[role] = replacement
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }
        layers[layerIndex] = LayerLayout(layer: layer.layer, file: layer.file, experts: experts)
        return PackedExpertsLayout(
            expertStride: layout.expertStride,
            numLayers: layout.numLayers,
            expertsPerLayer: layout.expertsPerLayer,
            layers: layers)
    }

    private func copy(
        _ entry: ResidentIndexEntry,
        dtype: UInt8? = nil,
        fileOffset: UInt64? = nil,
        sizeBytes: UInt64? = nil,
        shape: (UInt32, UInt32, UInt32, UInt32)? = nil,
        scaleOffset: UInt64? = nil,
        scaleSize: UInt64? = nil,
        biasOffset: UInt64? = nil,
        biasSize: UInt64? = nil
    ) -> ResidentIndexEntry {
        ResidentIndexEntry(
            name: entry.name,
            dtype: dtype ?? entry.dtype,
            fileOffset: fileOffset ?? entry.fileOffset,
            sizeBytes: sizeBytes ?? entry.sizeBytes,
            shape: shape ?? entry.shape,
            scaleOffset: scaleOffset ?? entry.scaleOffset,
            scaleSize: scaleSize ?? entry.scaleSize,
            biasOffset: biasOffset ?? entry.biasOffset,
            biasSize: biasSize ?? entry.biasSize)
    }
}
