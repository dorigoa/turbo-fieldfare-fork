import Darwin
import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// A chat with Qwen loaded has to be saved like a chat with Gemma.
///
/// Two ways of arriving at the same installed Qwen model: launching with it
/// already selected, and switching to it from Gemma inside one session. Both
/// end with the same model on disk, the same fake runtime and the same send,
/// so a difference in what reaches the store is the app's, not the setup's.
/// Nothing is downloaded and no model is loaded: the install is metadata only
/// and the runtime is `FakeInferenceClient`.
@Suite(.serialized) struct AppModelQwenHistoryTests {
    private let thirtyTwoGigabyteHost: UInt64 = 32 << 30

    /// Control: the launch path reads the model's identity with Qwen's own
    /// install descriptor.
    @MainActor @Test func aChatIsSavedWhenTheAppLaunchesWithQwen() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let qwen = root.appendingPathComponent(AppModelVariant.qwen36.defaultDirectoryName,
                                               isDirectory: true)
        try makeCompleteQwenInstall(at: qwen)
        var settings = MacAppSettings()
        settings.modelVariant = .qwen36
        settings.setSelectedModelPath(qwen.path, for: .qwen36)
        try MacAppSettingsFileStore.save(settings, at: AppDataLocation.settingsURL(in: root))
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = makeModel(root: root, client: client)
        try #require(model.modelVariant == .qwen36)
        try #require(model.modelPathText == qwen.path)
        try #require(model.installationStatus == .complete)

        #expect(model.conversationIdentity != nil,
                "the launch path must read the Qwen model's identity")
        try await sendOneTurn(model, client: client, directory: qwen)

        #expect(model.storedConversationID != nil)
        let store = try #require(model.conversationStore)
        #expect(try await store.list().count == 1)
    }

    /// The reported bug: after switching to Qwen inside a session, the turn
    /// is answered but never written, and nothing on screen says so.
    @MainActor @Test func aChatIsSavedAfterSwitchingToQwenFromGemma() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let fourBit = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        _ = try makeCompleteModelInstall("qwen-history-gemma", at: fourBit)
        let client = FakeInferenceClient(eventDelay: .milliseconds(1))
        let model = makeModel(root: root, client: client)
        try #require(model.modelVariant == .int4)
        try #require(model.installationStatus == .complete)
        let qwen = model.modelURL(for: .qwen36)
        try makeCompleteQwenInstall(at: qwen)

        await model.setModelVariant(.qwen36)

        try #require(model.modelVariant == .qwen36)
        try #require(model.modelPathText == qwen.path)
        try #require(model.installationStatus == .complete,
                     "the Qwen install must pass Qwen's own probe")
        #expect(model.conversationIdentity != nil,
                "switching to Qwen must read the Qwen model's identity")
        try await sendOneTurn(model, client: client, directory: qwen)

        #expect(model.storedConversationID != nil,
                "the turn was answered, so it must have been saved")
        let store = try #require(model.conversationStore)
        #expect(try await store.list().count == 1)
    }

    // MARK: - Helpers

    @MainActor
    private func makeModel(root: URL, client: FakeInferenceClient) -> AppModel {
        AppModel(appDataDirectory: root,
                 environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                 client: client,
                 installerFactory: { MockModelInstallerClient(descriptor: .descriptor(for: $0)) },
                 settingsPersistenceEnabled: true,
                 hostMemoryBytes: thirtyTwoGigabyteHost)
    }

    /// Puts the fake runtime in the loaded state the app expects, the way
    /// `AppModelHistoryTests.readyModel` does, then sends one prompt and
    /// waits for the send and for any write it started.
    @MainActor
    private func sendOneTurn(_ model: AppModel, client: FakeInferenceClient,
                             directory: URL) async throws {
        try await client.ensureLoaded(
            modelDirectory: directory,
            maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions,
            forceLogitsHead: true) { _ in }
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        model.maxNewTokensOverride = 8
        try await model.waitForHistory()
        model.promptText = "qwen exchange"
        model.send()
        await SendWaiting.turnEnds(model)
        try #require(model.conversation.committedTurns == 1,
                     "the fake runtime must have answered the turn")
        try await model.waitForStoredTurns(atLeast: model.conversation.turns.count)
    }

    private func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppModelQwenHistory-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.standardizedFileURL
    }

    /// A metadata-only Qwen install that passes `AppModelInstallationProbe`
    /// with Qwen's descriptor: the manifest of
    /// `RealInferenceClientFamilySelectionTests` carrying Qwen's pinned source
    /// hash, plus the layout file and a receipt bound to this directory.
    private func makeCompleteQwenInstall(at location: URL) throws {
        let directory = location.standardizedFileURL
        let experts = directory.appendingPathComponent("packed_experts", isDirectory: true)
        try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: experts.appendingPathComponent("layout.json"))
        func slot(_ weightBits: Int) -> [String: Any] {
            ["weightBits": weightBits, "scheme": "affine", "scaleType": "BF16",
             "biasType": "BF16", "groupSize": Quantization.groupSize]
        }
        let zeroSHA = String(repeating: "0", count: 64)
        let manifest: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true],
            "modelID": AppModelVariant.qwen36.repoID,
            "sourceSnapshotHash": "sha256:"
                + AppModelInstallDescriptor.descriptor(for: .qwen36).sourceIndexSHA256,
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
        let manifestData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try manifestData.write(to: directory.appendingPathComponent("manifest.json"))
        let receipt: [String: Any] = [
            "schemaVersion": 1,
            "manifestSha256": Sha256Verifier.hashData(manifestData),
            "modelDirectoryPath": directory.path,
            "verificationTimestamp": "2026-11-18T00:00:00Z",
            "toolVersion": "TurboFieldfareAppCoreTests",
            "files": [:],
        ]
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("verified-install.json"))
    }
}
