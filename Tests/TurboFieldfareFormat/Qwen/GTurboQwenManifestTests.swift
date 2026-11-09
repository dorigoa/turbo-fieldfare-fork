import Foundation
import Testing
@testable import TurboFieldfareFormat

/// Format-level contract for the optional Qwen manifest fields (D1, C4).
@Suite struct GTurboQwenManifestTests {

    private static let zeroSHA = String(repeating: "0", count: 64)

    /// The D1 "mapped" Qwen values at the smallest layer count a fixture needs.
    private static func qwenArchV1(numLayers: Int = 2) -> GTurboManifestArchV1 {
        GTurboManifestArchV1(
            hiddenSize: 2048, ffnIntermediate: 512, moeIntermediateSize: 512,
            numHeads: 16, numKVHeads: 2, numFullKVHeads: 2,
            headDim: 256, fullHeadDim: 256, vocabSize: 248_320,
            slidingWindow: 0, finalLogitSoftcap: 0,
            ropeTheta: 10_000_000, fullRopeTheta: 10_000_000,
            partialRotaryFactor: 0.25, numLayers: numLayers,
            numExperts: 256, topKExperts: 8,
            tieWordEmbeddings: false, attentionKEqV: false,
            hiddenActivation: "silu",
            fullAttentionLayerMask: (0..<numLayers).map { $0 % 4 == 3 ? 1 : 0 })
    }

    private static func qwen35Block() -> GTurboManifestQwen35V1 {
        GTurboManifestQwen35V1(
            linearNumKeyHeads: 16, linearNumValueHeads: 32,
            linearKeyHeadDim: 128, linearValueHeadDim: 128,
            linearConvKernel: 4, rmsNormEps: 1e-6,
            normWeightsShifted: true, attentionOutputGate: true,
            sharedExpertGate: true)
    }

    private static func qwenManifest() -> GTurboManifestV1 {
        let slot = GTurboManifestQuantSlotV1(
            weightBits: 4, scheme: "affine", scaleType: "BF16",
            biasType: "BF16", groupSize: 64)
        return GTurboManifestV1(
            flags: [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            modelID: "fixture/qwen36",
            sourceSnapshotHash: "snapshot",
            arch: qwenArchV1(),
            quant: GTurboManifestQuantV1(
                embedding: slot, attention: slot, router: slot,
                sharedExpert: slot, routedExpert: slot),
            files: [
                "model_weights.bin": GTurboManifestFileV1(size: 16_448, sha256: zeroSHA),
                "packed_experts/layout.json": GTurboManifestFileV1(size: 1, sha256: zeroSHA),
            ],
            expertsPerLayer: 256,
            numLayers: 2,
            expertStride: GTurboFormatV1.alignmentBytes,
            bitWidthOverridesHonored: nil,
            modelFamily: GTurboModelFamily.qwen35Moe.rawValue,
            qwen35: qwen35Block())
    }

    @Test func gemmaManifestsKeepTheNewFieldsNil() throws {
        // The initializer defaults must leave existing construction sites and
        // Gemma wire data untouched.
        let root: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true],
            "modelID": "fixture/gemma",
            "arch": [
                "hiddenSize": 64, "ffnIntermediate": 128, "moeIntermediateSize": 32,
                "numHeads": 4, "numKVHeads": 2, "numFullKVHeads": 1,
                "headDim": 16, "fullHeadDim": 32, "vocabSize": 1024,
                "slidingWindow": 128, "finalLogitSoftcap": 30,
                "ropeTheta": 10_000, "fullRopeTheta": 1_000_000,
                "partialRotaryFactor": 0.25, "numLayers": 1, "numExperts": 2,
                "topKExperts": 1, "tieWordEmbeddings": true, "attentionKEqV": true,
                "hiddenActivation": "gelu_pytorch_tanh", "fullAttentionLayerMask": [0],
            ],
            "files": [
                "model_weights.bin": ["size": 16_448, "sha256": Self.zeroSHA],
                "packed_experts/layout.json": ["size": 1, "sha256": Self.zeroSHA],
            ],
            "expertsPerLayer": 2, "numLayers": 1,
            "expertStride": GTurboFormatV1.alignmentBytes,
        ]
        let data = try JSONSerialization.data(withJSONObject: root)
        let manifest = try GTurboManifestCodec.decode(data)
        #expect(manifest.modelFamily == nil)
        #expect(manifest.qwen35 == nil)

        // Re-encoding must not introduce the new keys.
        let reencoded = String(decoding: try GTurboManifestCodec.encode(manifest), as: UTF8.self)
        #expect(!reencoded.contains("modelFamily"))
        #expect(!reencoded.contains("qwen35"))
    }

    @Test func qwenManifestRoundTripsThroughTheCodec() throws {
        let manifest = Self.qwenManifest()
        let data = try GTurboManifestCodec.encode(manifest)
        let decoded = try GTurboManifestCodec.decode(data)
        #expect(decoded == manifest)
        #expect(decoded.modelFamily == "qwen3_5_moe")
        let block = try #require(decoded.qwen35)
        #expect(block.linearNumValueHeads == 32)
        #expect(block.normWeightsShifted)
        // A Qwen slidingWindow of 0 is legal only for its family (D1).
        #expect(decoded.arch.slidingWindow == 0)
    }

    @Test func unknownModelFamilyIsRejectedWithAClearMessage() throws {
        var manifest = Self.qwenManifest()
        manifest = replaceFamily(in: manifest, with: "qwen2_tiny")
        #expect(throws: GTurboFormatError.invalid(
            field: "manifest.modelFamily",
            reason: "unknown model family \"qwen2_tiny\"")) {
            try GTurboManifestCodec.validate(manifest)
        }
    }

    @Test func qwenFamilyRequiresTheQwen35Block() throws {
        var manifest = Self.qwenManifest()
        manifest = replaceFamily(in: manifest,
                                 with: GTurboModelFamily.qwen35Moe.rawValue,
                                 dropQwen35: true)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.validate(manifest)
        }
    }

    @Test func qwen35BlockWithoutFamilyIsRejected() throws {
        let data = try GTurboManifestCodec.encode(Self.qwenManifest())
        var root = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        root.removeValue(forKey: "modelFamily")
        let stripped = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.invalid(
            field: "manifest.qwen35",
            reason: "present without manifest.modelFamily")) {
            _ = try GTurboManifestCodec.decode(stripped)
        }
    }

    @Test func invalidQwen35ValuesAreRejected() throws {
        let data = try GTurboManifestCodec.encode(Self.qwenManifest())
        var root = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        var block = try #require(root["qwen35"] as? [String: Any])
        block["linearNumValueHeads"] = 0
        root["qwen35"] = block
        let edited = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.self) {
            _ = try GTurboManifestCodec.decode(edited)
        }
    }

    /// Rebuilds a manifest with a different family string (the fields are
    /// `let`, so reconstruction is the way).
    private func replaceFamily(in manifest: GTurboManifestV1, with name: String,
                               dropQwen35: Bool = false) -> GTurboManifestV1 {
        GTurboManifestV1(
            magic: manifest.magic,
            versionMajor: manifest.versionMajor,
            versionMinor: manifest.versionMinor,
            flags: manifest.flags,
            modelID: manifest.modelID,
            sourceSnapshotHash: manifest.sourceSnapshotHash,
            arch: manifest.arch,
            quant: manifest.quant,
            files: manifest.files,
            expertsPerLayer: manifest.expertsPerLayer,
            numLayers: manifest.numLayers,
            expertStride: manifest.expertStride,
            bitWidthOverridesHonored: manifest.bitWidthOverridesHonored,
            modelFamily: name,
            qwen35: dropQwen35 ? nil : manifest.qwen35)
    }
}
