import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// S23 step 1: a Qwen pack selects the Qwen serving identity before any
/// model work. The manifest family decides the tokenizer and the runner,
/// the API model ID is the manifest's own (D9), and a client naming any
/// other model is refused (`model_not_found`) rather than served Qwen.
@Suite("Server Qwen model identity")
struct ServerQwenModelIdentityTests {

    private static let zeroSHA = String(repeating: "0", count: 64)
    private static let qwenManifestModelID = "mlx-community/Qwen3.6-35B-A3B-4bit"

    /// A manifest-only directory carrying the pinned Qwen values, valid for
    /// `ManifestReader.load` and therefore for `ServerModelVariant.read`.
    /// No weights are written: the identity comes from the manifest alone.
    private static func writeQwenManifestDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-qwen-identity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let arch: [String: Any] = [
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
        let qwen35: [String: Any] = [
            "linearNumKeyHeads": 16, "linearNumValueHeads": 32,
            "linearKeyHeadDim": 128, "linearValueHeadDim": 128,
            "linearConvKernel": 4, "rmsNormEps": 1e-6,
            "normWeightsShifted": true, "attentionOutputGate": true,
            "sharedExpertGate": true,
        ]
        func slot(_ bits: Int) -> [String: Any] {
            ["weightBits": bits, "scheme": "affine", "scaleType": "BF16",
             "biasType": "BF16", "groupSize": 64]
        }
        let root: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": qwenManifestModelID,
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

        let data = try JSONSerialization.data(
            withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    /// The read that gates the whole session: the manifest names the family,
    /// and a Qwen pack answers to its manifest ID, never to a Gemma name.
    @Test func qwenManifestReadsWithItsFamilyAndServesItsOwnID() throws {
        let directory = try Self.writeQwenManifestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let variant = try #require(try ServerModelVariant.read(modelDirectory: directory))
        #expect(variant.family == .qwen35Moe)
        #expect(variant.profile == .affine4)
        #expect(variant.manifestModelID == Self.qwenManifestModelID)
        #expect(variant.defaultAPIModelID == Self.qwenManifestModelID)
    }

    /// The default `--model-id`, end to end from the manifest: a Qwen pack
    /// is listed and served under its own ID, and the Gemma default is
    /// never assumed for it.
    @Test func qwenManifestDecidesTheDefaultModelID() throws {
        let directory = try Self.writeQwenManifestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let arguments = try ServerArguments.parse(
            ["--model", directory.path, "--max-context", "4096"],
            hostMemoryBytes: 32 << 30, environment: [:])
        #expect(arguments.modelVariant?.family == .qwen35Moe)
        #expect(arguments.modelID == Self.qwenManifestModelID)
    }

    /// The ready-time log line names the family next to the variant; the
    /// fields scripts read before are unchanged and the family is appended.
    @Test func modelOpenedLogAppendsTheFamily() {
        let fourBit = ServerModelVariant(
            profile: .affine4, manifestModelID: "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(ServerLog.modelOpenedMessage(fourBit)
            == "model opened variant=4-bit "
            + "manifest_model_id=mlx-community/gemma-4-26b-a4b-it-4bit family=gemma4")
        let qwen = ServerModelVariant(
            profile: .affine4,
            manifestModelID: Self.qwenManifestModelID,
            family: .qwen35Moe)
        #expect(ServerLog.modelOpenedMessage(qwen)
            == "model opened variant=4-bit "
            + "manifest_model_id=\(Self.qwenManifestModelID) family=qwen3_5_moe")
    }

    /// The listing names the Qwen ID; requests naming any other model —
    /// including every Gemma name — get `model_not_found` rather than Qwen.
    @Test func qwenModelIDIsListedAndOtherIDsAreRefused() async throws {
        let modelID = ServerModelVariant(
            profile: .affine4,
            manifestModelID: Self.qwenManifestModelID,
            family: .qwen35Moe).defaultAPIModelID
        #expect(modelID == Self.qwenManifestModelID)
        let server = TurboFieldfareHTTPServer(
            modelID: modelID,
            queueLimit: 1,
            backend: IdentityStubBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let models = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        let listing = try #require(
            JSONSerialization.jsonObject(with: models) as? [String: Any])
        let entries = try #require(listing["data"] as? [[String: Any]])
        #expect(entries.map { $0["id"] as? String } == [modelID])

        func complete(model: String) async throws -> (Int?, String) {
            var request = URLRequest(
                url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "model": model,
                "messages": [["role": "user", "content": "hi"]],
            ])
            let (data, response) = try await URLSession.shared.data(for: request)
            return ((response as? HTTPURLResponse)?.statusCode,
                    String(decoding: data, as: UTF8.self))
        }
        let served = try await complete(model: modelID)
        #expect(served.0 == 200)
        for foreign in [ServerModelVariant.legacyAPIModelID, "gemma-4-26b-a4b-it",
                        "mlx-community/Qwen3.6-35B-A3B-8bit", "some-other-model"] {
            let refused = try await complete(model: foreign)
            #expect(refused.0 == 404, "model \(foreign) must be refused")
            #expect(refused.1.contains("model_not_found"))
        }

        try await server.shutdown()
    }
}

private actor IdentityStubBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
    }
}
