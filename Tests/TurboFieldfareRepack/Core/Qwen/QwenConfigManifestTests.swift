import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// S5: the pinned Qwen checkpoint `config.json` yields exactly the D1 `arch`
/// and the C4 `qwen35` block, mutated configs are rejected, and the Gemma
/// reader path is untouched.
@Suite struct QwenConfigManifestTests {

    /// `Tests/QwenFixtures/checkpoint/config.json`, saved by S1 from the
    /// pinned checkpoint revision.
    private static let fixtureConfigPath = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Core/Qwen
        .deletingLastPathComponent()   // Core
        .deletingLastPathComponent()   // TurboFieldfareRepack
        .deletingLastPathComponent()   // Tests
        .appendingPathComponent("QwenFixtures/checkpoint/config.json")
        .path

    private func loadFixture() throws -> ArchInfo {
        try ArchInfo.load(configPath: Self.fixtureConfigPath)
    }

    // MARK: - D1: `arch` from the real config

    @Test func realConfigYieldsThePinnedD1Arch() throws {
        let arch = try loadFixture()
        #expect(arch.hiddenSize == 2048)
        #expect(arch.intermediateSize == 512)
        #expect(arch.moeIntermediateSize == 512)
        #expect(arch.numHeads == 16)
        #expect(arch.numKVHeads == 2)
        #expect(arch.numFullKVHeads == 2)
        #expect(arch.headDim == 256)
        #expect(arch.fullHeadDim == 256)
        #expect(arch.vocabSize == 248_320)
        #expect(arch.slidingWindow == 0)
        #expect(arch.finalLogitSoftcap == 0)
        #expect(arch.ropeTheta == 10_000_000)
        #expect(arch.fullRopeTheta == 10_000_000)
        #expect(arch.partialRotaryFactor == 0.25)
        #expect(arch.numLayers == 40)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 8)
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.attentionKEqV == false)
        #expect(arch.hiddenActivation == "silu")
        // 1 in layers 3, 7, …, 39; 0 elsewhere (D1).
        #expect(arch.fullAttentionLayerMask.count == 40)
        #expect(arch.fullAttentionLayerMask == (0..<40).map { UInt8($0 % 4 == 3 ? 1 : 0) })
        #expect(arch.modelFamily == GTurboModelFamily.qwen35Moe.rawValue)
    }

    // MARK: - C4: the `qwen35` block

    @Test func realConfigYieldsTheC4HybridBlock() throws {
        let arch = try loadFixture()
        let expected = GTurboManifestQwen35V1(
            linearNumKeyHeads: 16, linearNumValueHeads: 32,
            linearKeyHeadDim: 128, linearValueHeadDim: 128,
            linearConvKernel: 4, rmsNormEps: 1e-6,
            normWeightsShifted: true,          // S1 measurement
            attentionOutputGate: true, sharedExpertGate: true)
        #expect(arch.qwen35 == expected)
    }

    // MARK: - Fail-closed: mutated configs are rejected

    private func load(mutating mutate: (inout [String: Any]) -> Void) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: Self.fixtureConfigPath))
        var root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        mutate(&root)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("S5-QwenConfig-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("config.json").path
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: root).write(to: URL(fileURLWithPath: path))
        return try ArchInfo.load(configPath: path)
    }

    private static func mutateTextConfig(_ key: String, to value: Any) -> (inout [String: Any]) -> Void {
        { root in
            var tc = root["text_config"] as? [String: Any] ?? [:]
            tc[key] = value
            root["text_config"] = tc
        }
    }

    private static func mutateRope(_ key: String, to value: Any) -> (inout [String: Any]) -> Void {
        { root in
            var tc = root["text_config"] as? [String: Any] ?? [:]
            var rope = tc["rope_parameters"] as? [String: Any] ?? [:]
            rope[key] = value
            tc["rope_parameters"] = rope
            root["text_config"] = tc
        }
    }

    /// Sendable stand-in for JSON scalars in parameterized arguments.
    private enum ConfigValue: Sendable {
        case i(Int), d(Double), b(Bool), s(String)
        var anyValue: Any {
            switch self {
            case .i(let v): return v
            case .d(let v): return v
            case .b(let v): return v
            case .s(let v): return v
            }
        }
    }

    @Test(arguments: [
        ("hidden_size", ConfigValue.i(2047)),
        ("vocab_size", ConfigValue.i(248_321)),
        ("num_hidden_layers", ConfigValue.i(41)),
        ("num_experts", ConfigValue.i(255)),
        ("num_experts_per_tok", ConfigValue.i(6)),
        ("num_attention_heads", ConfigValue.i(8)),
        ("num_key_value_heads", ConfigValue.i(1)),
        ("head_dim", ConfigValue.i(128)),
        ("moe_intermediate_size", ConfigValue.i(513)),
        ("shared_expert_intermediate_size", ConfigValue.i(1024)),
        ("hidden_act", ConfigValue.s("relu")),
        ("tie_word_embeddings", ConfigValue.b(true)),
        ("full_attention_interval", ConfigValue.i(5)),
        ("model_type", ConfigValue.s("qwen3_5_moe_text_v2")),
        ("linear_num_key_heads", ConfigValue.i(17)),
        ("linear_num_value_heads", ConfigValue.i(33)),
        ("linear_key_head_dim", ConfigValue.i(64)),
        ("linear_value_head_dim", ConfigValue.i(256)),
        ("linear_conv_kernel_dim", ConfigValue.i(3)),
        ("rms_norm_eps", ConfigValue.d(1e-5)),
        ("attn_output_gate", ConfigValue.b(false)),
    ])
    private func mutatedTextConfigFieldIsRejected(key: String, value: ConfigValue) throws {
        var threw = false
        do {
            _ = try load(mutating: Self.mutateTextConfig(key, to: value.anyValue))
        } catch let error as RepackError {
            threw = true
            if case .configJsonInvalid = error {} else {
                Issue.record("wrong RepackError case: \(error)")
            }
        }
        #expect(threw, "config with \(key) mutated must be rejected")
    }

    @Test(arguments: [
        "rope_theta", "partial_rotary_factor",
    ])
    func mutatedRopeParameterIsRejected(key: String) throws {
        let value: Any = key == "rope_theta" ? 100_000_000.0 : 0.5
        #expect(throws: RepackError.self) {
            try load(mutating: Self.mutateRope(key, to: value))
        }
    }

    @Test func missingHybridFieldIsRejected() throws {
        #expect(throws: RepackError.self) {
            try load(mutating: Self.mutateTextConfig("linear_key_head_dim", to: NSNull()))
        }
    }

    @Test func brokenAttentionIntervalIsRejected() throws {
        // Flipping one layer breaks the three-linear/one-full pattern (3.2).
        var layerTypes = (0..<40).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" }
        layerTypes[1] = "full_attention"
        #expect(throws: RepackError.self) {
            try load(mutating: Self.mutateTextConfig("layer_types", to: layerTypes))
        }
    }

    @Test func qwenRootWithoutTextConfigIsRejected() throws {
        #expect(throws: RepackError.self) {
            try load(mutating: { root in root["text_config"] = nil })
        }
    }

    // MARK: - The Gemma reader path is untouched

    /// A synthetic Gemma-style `config.json` (same shape the repack Gemma
    /// fixtures use) written to a temporary file and loaded through the
    /// shared `ArchInfo.load` entry point.
    private func loadGemmaFixture() throws -> ArchInfo {
        let textConfig: [String: Any] = [
            "hidden_size": 32, "intermediate_size": 64, "moe_intermediate_size": 16,
            "num_attention_heads": 4, "num_key_value_heads": 2, "num_global_key_value_heads": 2,
            "head_dim": 8, "global_head_dim": 8, "vocab_size": 64, "num_hidden_layers": 4,
            "num_experts": 8, "top_k_experts": 2, "sliding_window": 4,
            "final_logit_softcapping": 30.0,
            "rope_parameters": [
                "sliding_attention": ["rope_theta": 10000.0, "rope_type": "default"],
                "full_attention": ["rope_theta": 1000000.0, "rope_type": "proportional",
                                   "partial_rotary_factor": 0.25],
            ],
            "layer_types": ["full_attention", "sliding_attention",
                            "full_attention", "sliding_attention"],
            "tie_word_embeddings": true, "attention_k_eq_v": true,
            "hidden_activation": "gelu_pytorch_tanh",
        ]
        let root: [String: Any] = [
            "model_type": "gemma4", "text_config": textConfig,
        ]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("S5-GemmaConfig-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("config.json").path
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: root).write(to: URL(fileURLWithPath: path))

        return try ArchInfo.load(configPath: path)
    }

    @Test func gemmaConfigKeepsTheGemmaReaderPath() throws {
        let arch = try loadGemmaFixture()
        #expect(arch.modelFamily == nil)
        #expect(arch.qwen35 == nil)
        #expect(arch.hiddenSize == 32)
        #expect(arch.ropeTheta == 10_000)      // sliding-attention theta, as before
        #expect(arch.fullRopeTheta == 1_000_000)
        #expect(arch.fullAttentionLayerMask == [1, 0, 1, 0])
    }

    // MARK: - The manifest written for Qwen (D1 + C4)

    @Test func qwenManifestCarriesFamilyBlockAndArch() throws {
        let arch = try loadFixture()
        let plan = RepackPlan(
            arch: arch, baseMode: "affine", baseGroupSize: 64,
            bitsOverrideCount: 0,
            resident: ResidentFilePlan(
                path: "/fixture/model_weights.bin", entries: [],
                stringTable: [], stringTableOffsets: [],
                indexSize: GTurboFormatV1.alignmentBytes, residentSize: 0),
            layers: [], matchedModelID: nil,
            excludedMultimodalTensorNames: [],
            modelFamily: arch.modelFamily, qwen35: arch.qwen35)
        #expect(plan.modelFamily == GTurboModelFamily.qwen35Moe.rawValue)
        #expect(plan.qwen35 == arch.qwen35)

        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "fixture/qwen36",
            sourceSnapshotHash: String(repeating: "a", count: 64),
            files: [],
            expertsPerLayer: arch.numExperts,
            numLayers: arch.numLayers,
            expertStride: GTurboFormatV1.alignmentBytes,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 4, attention: 4, router: 8,
                sharedExpert: 4, routedExpert: 4))
        // GTurboManifestCodec.decode validates (C4 cross-check included).
        let manifest = try GTurboManifestCodec.decode(data)
        #expect(manifest.modelFamily == "qwen3_5_moe")
        #expect(manifest.qwen35 == arch.qwen35)
        #expect(manifest.arch.hiddenSize == 2048)
        #expect(manifest.arch.vocabSize == 248_320)
        #expect(manifest.arch.numLayers == 40)
        #expect(manifest.arch.ffnIntermediate == 512)
        #expect(manifest.arch.moeIntermediateSize == 512)
        #expect(manifest.arch.numHeads == 16)
        #expect(manifest.arch.numKVHeads == 2)
        #expect(manifest.arch.numFullKVHeads == 2)
        #expect(manifest.arch.headDim == 256)
        #expect(manifest.arch.fullHeadDim == 256)
        #expect(manifest.arch.slidingWindow == 0)
        #expect(manifest.arch.finalLogitSoftcap == 0)
        #expect(manifest.arch.ropeTheta == 10_000_000)
        #expect(manifest.arch.fullRopeTheta == 10_000_000)
        #expect(manifest.arch.partialRotaryFactor == 0.25)
        #expect(manifest.arch.numExperts == 256)
        #expect(manifest.arch.topKExperts == 8)
        #expect(manifest.arch.hiddenActivation == "silu")
        #expect(manifest.arch.fullAttentionLayerMask
                == (0..<40).map { $0 % 4 == 3 ? 1 : 0 })
    }

    @Test func gemmaPlanStillWritesAPlainManifest() throws {
        // A Gemma plan leaves both fields at their `nil` defaults: the
        // manifest must stay the plain Gemma shape (byte-identity against the
        // v1 fixtures is covered by GTurboFormatCompatibilityTests).
        let gemmaPlanFields = try loadGemmaFixture()
        let plan = RepackPlan(
            arch: gemmaPlanFields, baseMode: "affine", baseGroupSize: 64,
            bitsOverrideCount: 0,
            resident: ResidentFilePlan(
                path: "/fixture/model_weights.bin", entries: [],
                stringTable: [], stringTableOffsets: [],
                indexSize: GTurboFormatV1.alignmentBytes, residentSize: 0),
            layers: [], matchedModelID: nil,
            excludedMultimodalTensorNames: [])
        #expect(plan.modelFamily == nil)
        #expect(plan.qwen35 == nil)
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "fixture/gemma",
            sourceSnapshotHash: String(repeating: "b", count: 64),
            files: [],
            expertsPerLayer: plan.arch.numExperts,
            numLayers: plan.arch.numLayers,
            expertStride: GTurboFormatV1.alignmentBytes,
            bitWidths: GTurboJSON.QuantBitWidths(
                embedding: 4, attention: 4, router: 8,
                sharedExpert: 4, routedExpert: 4))
        let manifest = try GTurboManifestCodec.decode(data)
        #expect(manifest.modelFamily == nil)
        #expect(manifest.qwen35 == nil)
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["modelFamily"] == nil)
        #expect(object["qwen35"] == nil)
    }
}
