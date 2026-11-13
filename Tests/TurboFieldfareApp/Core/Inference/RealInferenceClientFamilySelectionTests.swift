import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// S25 step 4: the decode-service session picks the tokenizer, the runner and
/// the prompt renderer from the manifest's family, and refuses to open an
/// image companion for a Qwen pack (D3, D12) — all decided from `manifest.json`
/// alone, with no weights on disk.
@Suite struct RealInferenceClientFamilySelectionTests {

    /// Just the manifest: the family read is schema-only, so no weights,
    /// layout or receipt files are needed.
    private func writeQwenManifest() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("family-qwen-\(UUID().uuidString).gturbo")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        func slot(_ weightBits: Int) -> [String: Any] {
            ["weightBits": weightBits, "scheme": "affine", "scaleType": "BF16",
             "biasType": "BF16", "groupSize": Quantization.groupSize]
        }
        let zeroSHA = String(repeating: "0", count: 64)
        let manifest: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true],
            "modelID": "test/qwen36-35b-a3b",
            "modelFamily": "qwen3_5_moe",
            "qwen35": [
                "linearNumKeyHeads": 16, "linearNumValueHeads": 32,
                "linearKeyHeadDim": 128, "linearValueHeadDim": 128,
                "linearConvKernel": 4, "rmsNormEps": 1e-6,
                "normWeightsShifted": true, "attentionOutputGate": true,
                "sharedExpertGate": true,
            ],
            "arch": [
                "hiddenSize": 2048, "ffnIntermediate": 512, "moeIntermediateSize": 512,
                "numHeads": 16, "numKVHeads": 2, "numFullKVHeads": 2,
                "headDim": 256, "fullHeadDim": 256, "vocabSize": 248_320,
                "slidingWindow": 0, "finalLogitSoftcap": 0,
                "ropeTheta": 10_000_000, "fullRopeTheta": 10_000_000,
                "partialRotaryFactor": 0.25, "numLayers": 40,
                "numExperts": 256, "topKExperts": 8,
                "tieWordEmbeddings": false, "attentionKEqV": false,
                "hiddenActivation": "silu",
                "fullAttentionLayerMask": (0..<40).map { $0 % 4 == 3 ? 1 : 0 },
            ],
            "quant": [
                "embedding": slot(4), "attention": slot(4), "router": slot(8),
                "sharedExpert": slot(4), "routedExpert": slot(4),
            ],
            "files": [
                "model_weights.bin": ["size": 0, "sha256": zeroSHA],
                "packed_experts/layout.json": ["size": 2, "sha256": zeroSHA],
            ],
            "expertsPerLayer": 256,
            "numLayers": 40,
            "expertStride": UInt64(getpagesize()),
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    /// The same manifest-only trick for Gemma: no `modelFamily` key at all,
    /// which must read as Gemma (D1).
    private func writeGemmaManifest() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("family-gemma-\(UUID().uuidString).gturbo")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let arch = ArchConfig.gemma4_26B_A4B
        func slot(_ weightBits: Int) -> [String: Any] {
            ["weightBits": weightBits, "scheme": "affine", "scaleType": "bf16",
             "biasType": "bf16", "groupSize": Quantization.groupSize]
        }
        var files: [String: Any] = [
            "model_weights.bin": ["size": 0, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 2, "sha256": String(repeating: "0", count: 64)],
        ]
        for layer in 0..<arch.numLayers {
            files[String(format: "packed_experts/layer_%02d.bin", layer)] = [
                "size": 0, "sha256": String(repeating: "0", count: 64),
            ]
        }
        let manifest: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true],
            "modelID": "test/gemma-4-26b-a4b",
            "sourceSnapshotHash": "sha256:" + String(repeating: "0", count: 64),
            "quant": [
                "embedding": slot(4),
                "attention": slot(4),
                "router": slot(8),
                "sharedExpert": slot(4),
                "routedExpert": slot(4),
            ],
            "arch": [
                "hiddenSize": arch.hiddenSize,
                "ffnIntermediate": arch.intermediateSize,
                "moeIntermediateSize": arch.moeIntermediateSize,
                "numHeads": arch.numHeads,
                "numKVHeads": arch.numKVHeads,
                "numFullKVHeads": arch.numFullKVHeads,
                "headDim": arch.headDim,
                "fullHeadDim": arch.fullHeadDim,
                "vocabSize": arch.vocabSize,
                "slidingWindow": arch.slidingWindow,
                "finalLogitSoftcap": arch.finalLogitSoftcap,
                "ropeTheta": arch.ropeTheta,
                "fullRopeTheta": arch.fullRopeTheta,
                "partialRotaryFactor": arch.partialRotaryFactor,
                "numLayers": arch.numLayers,
                "numExperts": arch.numExperts,
                "topKExperts": arch.topKExperts,
                "tieWordEmbeddings": arch.tieWordEmbeddings,
                "attentionKEqV": arch.attentionKEqV,
                "hiddenActivation": arch.hiddenActivation,
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
            ],
            "files": files,
            "expertsPerLayer": arch.numExperts,
            "numLayers": arch.numLayers,
            "expertStride": UInt64(getpagesize()),
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    private func manifest(in directory: URL) throws -> Manifest {
        try ManifestReader.load(directoryURL: directory, expecting: .gemma4_26B_A4B)
    }

    @Test func qwenManifestSelectsQwenFamilyWithoutImagePack() throws {
        let directory = try writeQwenManifest()
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = try manifest(in: directory)

        #expect(RealInferenceSession.family(of: manifest) == .qwen35Moe)
        // D12: the session never opens a tower for a Qwen pack.
        #expect(!RealInferenceSession.opensImagePack(for: .qwen35Moe))
    }

    @Test func gemmaManifestWithoutFamilyKeySelectsGemmaWithImagePack() throws {
        let directory = try writeGemmaManifest()
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = try manifest(in: directory)

        #expect(RealInferenceSession.family(of: manifest) == .gemma4)
        #expect(RealInferenceSession.opensImagePack(for: .gemma4))
    }

    /// A `ChatTokenizer` that is neither family's concrete type: it exists
    /// only so the builder's fail-closed casts are reachable.
    private struct MismatchedTokenizer: ChatTokenizer, @unchecked Sendable {
        var vocabSize: Int { 1 }
        var stopTokenIDs: Set<Int32> { [] }
        var endOfTurnID: Int32 { 0 }
        var structuralMarkerIDs: Set<Int32> { [] }
        var toolResultWaitID: Int32? { nil }
        func encode(_ text: String, addBOS: Bool) -> [Int32] { [] }
        func decode(_ ids: [Int32], skipSpecialTokens: Bool) -> String { "" }
        func makeStreamingDetokenizer(barrierTokenIDs: Set<Int32>) -> any StreamingDetokenizer {
            NullDetokenizer()
        }
    }

    private struct NullDetokenizer: StreamingDetokenizer {
        mutating func push(_ id: Int32) -> String { "" }
        mutating func flush() -> String { "" }
    }

    /// The renderer builder fails closed when the cached tokenizer belongs to
    /// the other family — it never hands a conversation a renderer that would
    /// render the wrong chat template.
    @Test func promptRendererRefusesTokenizerOfOtherFamily() throws {
        let tokenizer = MismatchedTokenizer()
        #expect(throws: (any Error).self) {
            try RealInferenceSession.promptRenderer(family: .qwen35Moe, tokenizer: tokenizer)
        }
        #expect(throws: (any Error).self) {
            try RealInferenceSession.promptRenderer(family: .gemma4, tokenizer: tokenizer)
        }
    }
}
