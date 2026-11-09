import Testing
import Foundation
@testable import TurboFieldfare

/// Reader-level contract for Qwen manifests (D1, D2, C4): validation runs
/// against the pinned Qwen constants, mismatches name the field, and every
/// other path stays Gemma's.
@Suite struct ManifestReaderQwenTests {

    private static let zeroSHA = String(repeating: "0", count: 64)

    /// A manifest dictionary with the pinned Qwen 36 values, written into a
    /// temp directory the reader can load.
    private static func writeQwenManifest(rootOverrides: [String: Any] = [:],
                                          archOverrides: [String: Any] = [:],
                                          qwen35Overrides: [String: Any] = [:]) throws
        -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-qwen-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        var arch: [String: Any] = [
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
        ]
        for (k, v) in archOverrides { arch[k] = v }

        var qwen35: [String: Any] = [
            "linearNumKeyHeads": 16, "linearNumValueHeads": 32,
            "linearKeyHeadDim": 128, "linearValueHeadDim": 128,
            "linearConvKernel": 4, "rmsNormEps": 1e-6,
            "normWeightsShifted": true, "attentionOutputGate": true,
            "sharedExpertGate": true,
        ]
        for (k, v) in qwen35Overrides { qwen35[k] = v }

        func slot(_ bits: Int) -> [String: Any] {
            ["weightBits": bits, "scheme": "affine", "scaleType": "BF16",
             "biasType": "BF16", "groupSize": 64]
        }

        var root: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": "qwen/qwen36",
            "modelFamily": "qwen3_5_moe",
            "qwen35": qwen35,
            "arch": arch,
            "quant": [
                "embedding": slot(4), "attention": slot(4), "router": slot(8),
                "sharedExpert": slot(4), "routedExpert": slot(4),
            ],
            "files": [
                "model_weights.bin": ["size": 1024, "sha256": zeroSHA],
                "packed_experts/layout.json": ["size": 1024, "sha256": zeroSHA],
            ],
            "expertsPerLayer": 256,
            "numLayers": 40,
            "expertStride": 16_384,
        ]
        for (k, v) in rootOverrides { root[k] = v }

        let data = try JSONSerialization.data(
            withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    @Test func qwenManifestValidatesAgainstThePinnedQwenConstants() throws {
        // D2: `expecting` keeps the Gemma default and must be ignored.
        let dir = try Self.writeQwenManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir,
                                               expecting: .gemma4_26B_A4B)
        #expect(manifest.modelFamily == "qwen3_5_moe")
        let block = try #require(manifest.qwen35)
        #expect(block == ManifestQwen35(
            linearNumKeyHeads: 16, linearNumValueHeads: 32,
            linearKeyHeadDim: 128, linearValueHeadDim: 128,
            linearConvKernel: 4, rmsNormEps: 1e-6,
            normWeightsShifted: true, attentionOutputGate: true,
            sharedExpertGate: true))
        #expect(manifest.arch.numLayers == 40)
    }

    @Test func qwen35MismatchReportsTheBlockFieldName() throws {
        let dir = try Self.writeQwenManifest(qwen35Overrides: ["linearNumValueHeads": 33])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.archMismatch(
            field: "qwen35.linearNumValueHeads", expected: "32", actual: "33")) {
            try ManifestReader.load(directoryURL: dir, expecting: .gemma4_26B_A4B)
        }
    }

    @Test func qwenArchMismatchReportsTheArchFieldName() throws {
        let dir = try Self.writeQwenManifest(archOverrides: ["numHeads": 32])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.archMismatch(
            field: "numHeads", expected: "16", actual: "32")) {
            try ManifestReader.load(directoryURL: dir, expecting: .gemma4_26B_A4B)
        }
    }

    @Test func unknownModelFamilyIsRejectedWithAClearMessage() throws {
        let dir = try Self.writeQwenManifest(rootOverrides: ["modelFamily": "qwen2_tiny"])
        defer { try? FileManager.default.removeItem(at: dir) }
        var caught: ModelError?
        do {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .gemma4_26B_A4B)
        } catch let error as ModelError {
            caught = error
        }
        let message = caught?.description ?? "<no error>"
        #expect(message.contains("qwen2_tiny"), "message was: \(message)")
    }

    @Test func qwenFamilyWithoutTheBlockIsRejected() throws {
        let dir = try Self.writeQwenManifest(rootOverrides: [:])
        defer { try? FileManager.default.removeItem(at: dir) }
        // Strip the block from the written JSON.
        let url = dir.appendingPathComponent("manifest.json")
        var root = try #require(JSONSerialization.jsonObject(
            with: try Data(contentsOf: url)) as? [String: Any])
        root.removeValue(forKey: "qwen35")
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        #expect(throws: ModelError.self) {
            try ManifestReader.load(directoryURL: dir, expecting: .gemma4_26B_A4B)
        }
    }

    @Test func qwenManifestStillRequiresQuantMetadata() throws {
        let dir = try Self.writeQwenManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("manifest.json")
        var root = try #require(JSONSerialization.jsonObject(
            with: try Data(contentsOf: url)) as? [String: Any])
        root.removeValue(forKey: "quant")
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        #expect(throws: ModelError.self) {
            try ManifestReader.load(directoryURL: dir, expecting: .gemma4_26B_A4B)
        }
    }

    @Test func qwenArchValidatedAgainstGemmaExpectationsFailsOnHiddenSize() throws {
        // The closed failure D1 describes: the same Qwen `arch` compared with
        // the Gemma baseline diverges on the first field, before any other
        // mismatch can matter.
        let qwenJSON = try JSONSerialization.data(
            withJSONObject: [
                "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
                "flags": ["streamingPresent": true],
                "modelID": "qwen/qwen36",
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
                "files": [
                    "model_weights.bin": ["size": 1024, "sha256": Self.zeroSHA],
                    "packed_experts/layout.json": ["size": 1024, "sha256": Self.zeroSHA],
                ],
                "expertsPerLayer": 256, "numLayers": 40, "expertStride": 16_384,
            ])
        let manifest = try JSONDecoder().decode(Manifest.self, from: qwenJSON)
        #expect(throws: ModelError.archMismatch(
            field: "hiddenSize", expected: "2816", actual: "2048")) {
            try ManifestReader.validate(manifest, against: .gemma4_26B_A4B)
        }
    }

    @Test func strippedQwenManifestNeverLoadsAsGemma() throws {
        // A Qwen manifest with the new keys removed (what a pre-Qwen reader
        // sees) still fails closed: the generic v1 gate rejects the Qwen
        // `arch` values before any Gemma comparison could accept them.
        let dir = try Self.writeQwenManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("manifest.json")
        var root = try #require(JSONSerialization.jsonObject(
            with: try Data(contentsOf: url)) as? [String: Any])
        root.removeValue(forKey: "modelFamily")
        root.removeValue(forKey: "qwen35")
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        #expect(throws: ModelError.self) {
            try ManifestReader.load(directoryURL: dir, expecting: .gemma4_26B_A4B)
        }
    }
}
