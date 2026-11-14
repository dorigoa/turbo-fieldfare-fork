import Foundation
import Metal
import Testing

@testable import TurboFieldfare

/// S18: `Model.load` must accept the installed Qwen `.gturbo`, pick the
/// pinned Qwen constants by manifest family (D2), and the runtime schema
/// validator must reject tampered Qwen indexes. Every case here needs the
/// local model, so the suite is gated on contract C9.
@Suite(.enabled(if: QwenTestResources.hasModel))
struct QwenModelLoadTests {

    private struct Artifacts {
        let directory: URL
        let manifest: Manifest
        let index: ResidentIndex
        let layout: PackedExpertsLayout
    }

    private func artifacts() throws -> Artifacts {
        let directory = try #require(QwenTestResources.modelURL)
        let manifestData = try Data(
            contentsOf: directory.appendingPathComponent("manifest.json"))
        // Gemma's `expecting` on purpose: a Qwen manifest ignores it (D2).
        let manifest = try ManifestReader.decode(
            data: manifestData, expecting: .gemma4_26B_A4B)
        let index = try ResidentIndexReader.load(
            fileURL: directory.appendingPathComponent("model_weights.bin"))
        let layoutData = try Data(
            contentsOf: directory.appendingPathComponent("packed_experts/layout.json"))
        let layout = try PackedExpertsLayoutReader.decode(data: layoutData, manifest: manifest)
        return Artifacts(directory: directory, manifest: manifest, index: index, layout: layout)
    }

    private func tampered(_ artifacts: Artifacts,
                          name: String,
                          _ transform: (ResidentIndexEntry) -> ResidentIndexEntry?) throws
        -> ResidentIndex
    {
        var entries = artifacts.index.entries
        let entry = try #require(entries[name])
        entries[name] = transform(entry)
        return ResidentIndex(header: artifacts.index.header, entries: entries)
    }

    private func copy(
        _ entry: ResidentIndexEntry,
        dtype: UInt8? = nil,
        sizeBytes: UInt64? = nil,
        shape: (UInt32, UInt32, UInt32, UInt32)? = nil,
        scaleSize: UInt64? = nil
    ) -> ResidentIndexEntry {
        ResidentIndexEntry(
            name: entry.name,
            dtype: dtype ?? entry.dtype,
            fileOffset: entry.fileOffset,
            sizeBytes: sizeBytes ?? entry.sizeBytes,
            shape: shape ?? entry.shape,
            scaleOffset: entry.scaleOffset,
            scaleSize: scaleSize ?? entry.scaleSize,
            biasOffset: entry.biasOffset,
            biasSize: entry.biasSize)
    }

    // MARK: - Real load (Fatto quando: annotare il tempo nel report)

    @Test("the installed Qwen model loads and switches to the Qwen constants")
    func loadsRealQwenModel() throws {
        let directory = try #require(QwenTestResources.modelURL)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let start = ContinuousClock.now
        let model = try Model.load(directoryURL: directory, device: device)
        let elapsed = start.duration(to: ContinuousClock.now)
        print("S18 real Qwen Model.load time: \(elapsed)")

        #expect(model.family == .qwen35Moe)
        #expect(model.config == .qwen36_35B_A3B)
        #expect(model.qwenHybridConfig == .qwen36_35B_A3B)
        #expect(model.manifest.modelFamily == ModelFamily.qwen35Moe.rawValue)

        // The LM head is a separate tensor from the embedding (no tie).
        let head = model.qwenLMHead
        let embedding = model.embedding
        #expect(head.shape == (248_320, 2_048, 0, 0))
        #expect(head.offset != embedding.offset)
        #expect(head.length == embedding.length)

        // Per-layer accessors on one linear layer, one attention layer.
        let qkv = try model.qwenInProjQKV(layer: 0)
        #expect(qkv.shape == (8_192, 2_048, 0, 0))
        #expect(try model.qwenInProjZ(layer: 0).shape == (4_096, 2_048, 0, 0))
        #expect(try model.qwenInProjA(layer: 0).shape == (32, 2_048, 0, 0))
        #expect(try model.qwenConv1D(layer: 0).shape == (8_192, 4, 1, 0))
        #expect(try model.qwenALog(layer: 0).shape == (32, 0, 0, 0))
        #expect(try model.qwenDtBias(layer: 0).shape == (32, 0, 0, 0))
        #expect(try model.qwenDeltaNetNorm(layer: 0).shape == (128, 0, 0, 0))
        #expect(try model.qwenOutProj(layer: 0).shape == (2_048, 4_096, 0, 0))
        #expect(try model.qProj(layer: 3).shape == (8_192, 2_048, 0, 0))
        #expect(try model.kProj(layer: 3).shape == (512, 2_048, 0, 0))
        #expect(try model.oProj(layer: 3).shape == (2_048, 4_096, 0, 0))
        #expect(try model.qNorm(layer: 3).shape == (256, 0, 0, 0))
        #expect(try model.qwenRouter(layer: 0).shape == (256, 2_048, 0, 0))
        #expect(try model.qwenSharedExpertGate(layer: 0).shape == (512, 2_048, 0, 0))
        #expect(try model.qwenSharedExpertDown(layer: 0).shape == (2_048, 512, 0, 0))
        #expect(try model.qwenSharedExpertGateScalar(layer: 0).shape == (1, 2_048, 0, 0))
    }

    // MARK: - Runtime schema (3.3 tensors, 3.4 shapes)

    @Test("the installed index passes the Qwen runtime schema")
    func installedIndexPassesQwenSchema() throws {
        let artifacts = try artifacts()
        try Model.validateQwenRuntimeSchema(
            residentIndex: artifacts.index,
            layout: artifacts.layout,
            manifest: artifacts.manifest,
            config: .qwen36_35B_A3B,
            hybrid: .qwen36_35B_A3B)
    }

    @Test("the Gemma schema rejects the Qwen index")
    func gemmaSchemaRejectsQwenIndex() throws {
        let artifacts = try artifacts()
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: artifacts.index,
                layout: artifacts.layout,
                manifest: artifacts.manifest,
                config: .gemma4_26B_A4B)
        }
    }

    @Test("missing Qwen tensors are rejected")
    func missingTensorsRejected() throws {
        let artifacts = try artifacts()
        let names = [
            "language_model.lm_head.weight",
            "language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.in_proj_z.weight",
            "language_model.model.layers.0.linear_attn.in_proj_a.weight",
            "language_model.model.layers.0.linear_attn.in_proj_b.weight",
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "language_model.model.layers.0.linear_attn.A_log",
            "language_model.model.layers.0.linear_attn.dt_bias",
            "language_model.model.layers.0.linear_attn.norm.weight",
            "language_model.model.layers.0.linear_attn.out_proj.weight",
            "language_model.model.layers.3.self_attn.q_proj.weight",
            "language_model.model.layers.3.self_attn.q_norm.weight",
            "language_model.model.layers.39.self_attn.k_proj.weight",
            "language_model.model.layers.0.mlp.gate.weight",
            "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
            "language_model.model.layers.0.mlp.shared_expert_gate.weight",
        ]
        for name in names {
            var entries = artifacts.index.entries
            entries[name] = nil
            let index = ResidentIndex(header: artifacts.index.header, entries: entries)
            #expect(throws: ModelError.self, "removing \(name) must throw") {
                try Model.validateQwenRuntimeSchema(
                    residentIndex: index,
                    layout: artifacts.layout,
                    manifest: artifacts.manifest,
                    config: .qwen36_35B_A3B,
                    hybrid: .qwen36_35B_A3B)
            }
        }
    }

    @Test("wrong dtype, shape or affine sizes are rejected")
    func malformedEntriesRejected() throws {
        let artifacts = try artifacts()

        // `dt_bias` must stay BF16 (the installed checkpoint stores it so).
        let dtBias = "language_model.model.layers.0.linear_attn.dt_bias"
        let index = try tampered(artifacts, name: dtBias) {
            copy($0, dtype: 3) // fp32
        }
        #expect(throws: ModelError.self) {
            try validate(artifacts, index: index)
        }

        // `in_proj_z` must have valueDim rows.
        let inProjZ = "language_model.model.layers.0.linear_attn.in_proj_z.weight"
        let zIndex = try tampered(artifacts, name: inProjZ) {
            copy($0, shape: (2_048, $0.shape.1, 0, 0))
        }
        #expect(throws: ModelError.self) {
            try validate(artifacts, index: zIndex)
        }

        // The conv1d shape must be [8192, 4, 1].
        let conv = "language_model.model.layers.0.linear_attn.conv1d.weight"
        let convIndex = try tampered(artifacts, name: conv) {
            copy($0, shape: (8_192, 1, 4, 0))
        }
        #expect(throws: ModelError.self) {
            try validate(artifacts, index: convIndex)
        }

        // `lm_head` affine scale sizes must match the embedding slot.
        let lmHead = "language_model.lm_head.weight"
        let headIndex = try tampered(artifacts, name: lmHead) {
            copy($0, scaleSize: $0.scaleSize + 2)
        }
        #expect(throws: ModelError.self) {
            try validate(artifacts, index: headIndex)
        }

        // The router is the 8-bit slot: a 4-bit byte count must be rejected.
        let router = "language_model.model.layers.0.mlp.gate.weight"
        let routerIndex = try tampered(artifacts, name: router) {
            copy($0, sizeBytes: $0.sizeBytes / 2)
        }
        #expect(throws: ModelError.self) {
            try validate(artifacts, index: routerIndex)
        }

        // q_proj carries the output gate: 8192 rows, not 4096.
        let qProj = "language_model.model.layers.3.self_attn.q_proj.weight"
        let qIndex = try tampered(artifacts, name: qProj) {
            copy($0, shape: (4_096, $0.shape.1, 0, 0))
        }
        #expect(throws: ModelError.self) {
            try validate(artifacts, index: qIndex)
        }
    }

    @Test("missing routed expert roles are rejected")
    func missingRoutedRoleRejected() throws {
        let artifacts = try artifacts()
        var layers = artifacts.layout.layers
        let layer = layers[5]
        let experts = layer.experts.map { expert in
            var tensors = expert.subTensors
            tensors["up"] = nil
            return ExpertEntry(
                expert: expert.expert,
                physicalRank: expert.physicalRank,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }
        layers[5] = LayerLayout(layer: layer.layer, file: layer.file, experts: experts)
        let layout = PackedExpertsLayout(
            expertStride: artifacts.layout.expertStride,
            numLayers: artifacts.layout.numLayers,
            expertsPerLayer: artifacts.layout.expertsPerLayer,
            layers: layers)
        #expect(throws: ModelError.self) {
            try Model.validateQwenRuntimeSchema(
                residentIndex: artifacts.index,
                layout: layout,
                manifest: artifacts.manifest,
                config: .qwen36_35B_A3B,
                hybrid: .qwen36_35B_A3B)
        }
    }

    private func validate(_ artifacts: Artifacts, index: ResidentIndex) throws {
        try Model.validateQwenRuntimeSchema(
            residentIndex: index,
            layout: artifacts.layout,
            manifest: artifacts.manifest,
            config: .qwen36_35B_A3B,
            hybrid: .qwen36_35B_A3B)
    }
}
