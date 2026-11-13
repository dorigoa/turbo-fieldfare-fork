import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Choosing a variant moves everything that depends on it together: the
/// location, the installer and its download size, the installed-model probe
/// and memory admission. Each test drives the switch through `AppModel` with
/// fake installers and a fake inference client; nothing is downloaded and no
/// model is loaded.
@Suite(.serialized) struct AppModelVariantTests {
    private let thirtyTwoGigabyteHost: UInt64 = 32 << 30
    private let sixteenGigabyteHost: UInt64 = 16 << 30
    private let eightGigabyteHost: UInt64 = 8 << 30

    @MainActor @Test func switchingVariantMovesTheLocationInstallerProbeAndSettings() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let fourBit = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        _ = try makeCompleteModelInstall("switch", at: fourBit)
        let client = MockLifecycleInferenceClient()
        let model = makeModel(root: root, client: client, host: thirtyTwoGigabyteHost)
        try #require(model.modelVariant == .int4)
        try #require(model.isModelInstalled)
        #expect(model.modelPathText == fourBit.path)
        #expect(model.installDescriptor == .descriptor(for: .int4))
        model.loadState = .ready(modelDirectory: fourBit, loadSeconds: 0)

        await model.setModelVariant(.int8)

        let eightBit = root.appendingPathComponent("gemma4-8bit.gturbo", isDirectory: true)
        #expect(model.modelVariant == .int8)
        #expect(model.modelPathText == eightBit.path)
        #expect(model.installDescriptor == .descriptor(for: .int8))
        #expect(model.installationStatus == .missing)
        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
        #expect(model.canInstallModel, "the other variant's download must be offered")
        #expect(model.presentation.label == "Model required")
        let saved = try #require(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root)))
        #expect(saved.modelVariant == .int8)
        #expect(saved.selectedModelPath == fourBit.path, "the legacy key must keep naming the 4-bit model")
        #expect(saved.selectedModelPathInt8 == eightBit.path)

        await model.setModelVariant(.int4)

        #expect(model.modelVariant == .int4)
        #expect(model.modelPathText == fourBit.path)
        #expect(model.installDescriptor == .descriptor(for: .int4))
        #expect(model.isModelInstalled)
        let back = try #require(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root)))
        #expect(back.modelVariant == .int4)
        #expect(back.selectedModelPath == fourBit.path)
        #expect(back.selectedModelPathInt8 == eightBit.path)
    }

    /// Loading goes to the chosen variant's directory; that the decode service
    /// then sizes it from the manifest is `DecodeLoadAdmissionTests`.
    @MainActor @Test func anInstalledEightBitModelLoadsFromItsOwnDirectory() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let eightBit = root.appendingPathComponent("gemma4-8bit.gturbo", isDirectory: true)
        _ = try makeCompleteModelInstall("load-8", variant: .int8, at: eightBit)
        let client = MockLifecycleInferenceClient()
        let model = makeModel(root: root, client: client, host: thirtyTwoGigabyteHost)

        await model.setModelVariant(.int8)

        #expect(model.isModelInstalled)
        #expect(model.installationStatus == .complete)
        try #require(model.canLoadModel)
        model.loadModel()
        await client.waitForEnsureLoadedCallCount(1)
        #expect(client.ensureLoadedCalls.last?.0.standardizedFileURL.path == eightBit.path)
    }

    /// U1: below the variant's threshold the load is blocked, not merely
    /// discouraged, with the admission rule's own sentence. The download stays
    /// possible.
    @MainActor @Test func anEightGigabyteMacCanDownloadButNotLoadTheEightBitModel() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let client = MockLifecycleInferenceClient()
        let model = makeModel(root: root, client: client, host: eightGigabyteHost)
        try #require(model.modelVariantMemoryRefusal == nil)

        await model.setModelVariant(.int8)

        #expect(model.canInstallModel)
        let refusal = try #require(model.modelVariantMemoryRefusal)
        #expect(refusal.contains("8-bit"))
        #expect(refusal.contains("4,096"))
        #expect(refusal.contains("16 GB"))
        #expect(refusal.contains("8 GB"))

        _ = try makeCompleteModelInstall("refuse-8", variant: .int8,
                                         at: URL(fileURLWithPath: model.modelPathText))
        model.recheckModelAtCurrentLocation()
        try #require(model.isModelInstalled)
        #expect(!model.canLoadModel)
        #expect(model.presentation.label == "Needs more memory")
        #expect(model.presentation.detail == refusal)
        #expect(model.presentation.primaryAction == nil)
        model.loadModel()
        #expect(client.ensureLoadedCallCount() == 0)
        // Nothing fits, so nothing is clamped: the chosen context is kept for
        // the switch back, and the picker shows it rather than an empty menu.
        #expect(model.maxContextTokens == AppContextLengthOption.defaultOption.tokens)
        #expect(model.contextClampNotice == nil)
        #expect(model.contextOptions == [.defaultOption])
        #expect(model.contextOptionsNote == nil)

        await model.setModelVariant(.int4)
        #expect(model.modelVariantMemoryRefusal == nil)
        #expect(model.contextOptions.contains(.oneTwentyEightK))
    }

    /// The clamp follows the variant: a context the 4-bit model can back on
    /// this Mac is one the 8-bit model cannot, and the switch says so.
    @MainActor @Test func switchingVariantReclampsTheContextWithTheNotice() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try MacAppSettingsFileStore.save(MacAppSettings(contextTokens: 262_144, expertCacheSlots: 32),
                                         at: AppDataLocation.settingsURL(in: root))
        let model = makeModel(root: root, client: MockLifecycleInferenceClient(), host: sixteenGigabyteHost)
        try #require(model.maxContextTokens == 262_144)
        try #require(model.contextClampNotice == nil)

        await model.setModelVariant(.int8)

        #expect(model.maxContextTokens == 131_072)
        let notice = try #require(model.contextClampNotice)
        #expect(notice.contains("262,144"))
        #expect(notice.contains("128K"))
        #expect(!model.contextOptions.contains(.twoFiftySixK))
        #expect(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root))?.contextTokens
            == 131_072)

        // Back on the 4-bit model the notice no longer describes anything:
        // it quoted the 8-bit need.
        await model.setModelVariant(.int4)
        #expect(model.contextClampNotice == nil)
        #expect(model.contextOptions.contains(.twoFiftySixK))
    }

    /// With the location fixed for the launch, the variant is the installed
    /// model's own, the picker is off with a reason, and the user's saved
    /// choice on disk is left as it was.
    @MainActor @Test func aModelPathOverrideReadsTheVariantFromTheManifestAndLocksThePicker() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let eightBit = try makeCompleteModelInstall("override-8", variant: .int8)
        defer { try? FileManager.default.removeItem(at: eightBit) }
        let model = AppModel(appDataDirectory: root,
                             environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path,
                                           "TURBO_FIELDFARE_MODEL_PATH": eightBit.path],
                             client: MockLifecycleInferenceClient(),
                             installerFactory: Self.fakeInstallers,
                             settingsPersistenceEnabled: true,
                             hostMemoryBytes: thirtyTwoGigabyteHost)

        #expect(model.modelVariant == .int8)
        #expect(model.isModelInstalled)
        #expect(model.installDescriptor == .descriptor(for: .int8))
        #expect(!model.canChangeModelVariant)
        #expect(model.modelVariantLockReason?.contains("TURBO_FIELDFARE_MODEL_PATH") == true)

        await model.setModelVariant(.int4)
        #expect(model.modelVariant == .int8)
        #expect(model.modelPathText == eightBit.standardizedFileURL.path)

        model.setTextSize(.large)
        let saved = try #require(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root)))
        #expect(saved.textSize == .large)
        #expect(saved.modelVariant == .int4)
        #expect(saved.selectedModelPathInt8 == nil)
    }

    @MainActor @Test func theVariantPickerFollowsTheLocationRules() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = makeModel(root: root, client: MockLifecycleInferenceClient(), host: thirtyTwoGigabyteHost)
        #expect(model.canChangeModelVariant)
        #expect(model.modelVariantLockReason == nil)
        model.loadState = .loading(.validatingDirectory)
        #expect(!model.canChangeModelVariant)
        await model.setModelVariant(.int8)
        #expect(model.modelVariant == .int4, "a variant change ran during a load")
    }

    /// Choosing an existing model of the other variant is a variant switch,
    /// and each variant then remembers where its own model is.
    @MainActor @Test func choosingAnExistingModelOfTheOtherVariantSwitchesToIt() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let fourBit = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        _ = try makeCompleteModelInstall("existing-4", at: fourBit)
        let elsewhere = root.appendingPathComponent("elsewhere", isDirectory: true)
        let eightBit = elsewhere.appendingPathComponent("gemma4-8bit.gturbo", isDirectory: true)
        _ = try makeCompleteModelInstall("existing-8", variant: .int8, at: eightBit)
        let model = makeModel(root: root, client: MockLifecycleInferenceClient(), host: thirtyTwoGigabyteHost)

        await model.reviewExistingModelLocation(elsewhere)

        #expect(model.modelLocationDiagnostic == nil)
        #expect(model.modelVariant == .int8)
        #expect(model.modelPathText == eightBit.path)
        #expect(model.isModelInstalled)
        #expect(model.installDescriptor == .descriptor(for: .int8))
        let saved = try #require(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root)))
        #expect(saved.modelVariant == .int8)
        #expect(saved.selectedModelPath == fourBit.path)
        #expect(saved.selectedModelPathInt8 == eightBit.path)

        await model.setModelVariant(.int4)
        #expect(model.modelPathText == fourBit.path)
        await model.setModelVariant(.int8)
        #expect(model.modelPathText == eightBit.path, "the 8-bit location was forgotten")
    }

    /// A download location names the variant's own directory, and is saved
    /// under that variant's key only.
    @MainActor @Test func aDownloadLocationUsesTheVariantsDirectoryName() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = makeModel(root: root, client: MockLifecycleInferenceClient(), host: thirtyTwoGigabyteHost)
        let fourBitPath = model.modelPathText
        await model.setModelVariant(.int8)
        let parent = root.appendingPathComponent("downloads", isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)

        await model.selectModelLocation(parent, intent: .download)

        #expect(model.modelLocationDiagnostic == nil)
        #expect(model.modelPathText == parent.appendingPathComponent("gemma4-8bit.gturbo").path)
        let saved = try #require(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root)))
        #expect(saved.selectedModelPathInt8 == model.modelPathText)
        #expect(saved.selectedModelPath == fourBitPath)
    }

    /// The download runs through the chosen variant's installer, and its
    /// completion is checked against that variant's checkpoint.
    @MainActor @Test func installingRunsTheSelectedVariantsInstaller() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        // What a real installer would leave behind. Elsewhere than the
        // destination, so the destination still reads as not installed when
        // the install starts; the event then names where the model ended up.
        let eightBit = try makeCompleteModelInstall("install-8", variant: .int8)
        defer { try? FileManager.default.removeItem(at: eightBit) }
        // The 4-bit fake ends without an `.installed` event, so if it were the
        // one that ran, the install would fail as incomplete.
        let fourBitInstaller = MockModelInstallerClient(descriptor: .descriptor(for: .int4))
        let eightBitInstaller = MockModelInstallerClient(events: [.installed(eightBit)],
                                                         descriptor: .descriptor(for: .int8))
        let installers: [AppModelVariant: MockModelInstallerClient] = [.int4: fourBitInstaller,
                                                                       .int8: eightBitInstaller]
        let model = AppModel(appDataDirectory: root,
                             environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                             client: MockLifecycleInferenceClient(),
                             installerFactory: { installers[$0]! },
                             settingsPersistenceEnabled: true,
                             hostMemoryBytes: thirtyTwoGigabyteHost)
        await model.setModelVariant(.int8)
        try #require(model.canInstallModel)

        model.installModel()
        try await waitUntil {
            await MainActor.run { !model.installState.isInstalling && model.installState != .idle }
        }

        // Accepted by the 8-bit probe, which is the installer's descriptor.
        #expect(model.installState == .installed(modelDirectory: eightBit.standardizedFileURL))
        #expect(model.isModelInstalled)
        #expect(model.modelVariant == .int8)
        #expect(!fourBitInstaller.cancelCalled)
    }

    @MainActor @Test func aConversationFromTheOtherVariantIsNamedButStillNotContinued() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        _ = try makeCompleteModelInstall("history-variant",
                                         modelID: AppModelVariant.int4.repoID, at: directory)
        let current = Self.identity(.int4)
        let store = ConversationStore(rootURL: root.appendingPathComponent("conversations", isDirectory: true))
        try await store.activate()
        let fromEightBit = try await store.create(
            title: "eight", identity: Self.identity(.int8),
            session: ConversationSessionSettings(contextTokens: 8_192, expertCacheSlots: 16,
                                                 visionResidencyPolicy: "onDemand"),
            sampling: ConversationSampling(temperature: 0, topKEnabled: true, topK: 64,
                                           topPEnabled: true, topP: 0.95, maxNewTokens: 64))
        _ = try await store.append([
            .turn(ConversationTurnRecord(role: .user, at: Date(), text: "stored", tokens: [1])),
            .turn(ConversationTurnRecord(role: .assistant, at: Date(), text: "answer", tokens: [2],
                                         boundary: ConversationBoundary(tokens: [3], needsReplay: false))),
        ], to: fromEightBit.id)

        let model = AppModel(modelDirectory: directory,
                             appDataDirectory: root,
                             client: FakeInferenceClient(),
                             installer: MockModelInstallerClient(),
                             settingsPersistenceEnabled: true,
                             conversationIdentityProvider: { _ in current },
                             conversationStoreProvider: { _ in store },
                             hostMemoryBytes: thirtyTwoGigabyteHost)
        try await model.waitForHistory(count: 1)
        let meta = try #require(model.history.entry(fromEightBit.id))
        #expect(model.continuability(of: meta) == .cannotReplay(reason: .differentModel))
        #expect(model.recordedOtherVariant(of: meta) == .int8)

        model.openConversation(id: fromEightBit.id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        #expect(model.openedConversationState == .cannotReplay(reason: .differentModel))
        #expect(model.openedConversationRecordedVariant == .int8)
    }

    /// Before anything is installed there is no identity to compare with; the
    /// variant the app is set to stands in for it.
    @MainActor @Test func withoutAnInstalledModelTheSettingDecidesWhichVariantIsOther() throws {
        let model = AppModel(client: MockLifecycleInferenceClient(), installer: MockModelInstallerClient())
        #expect(model.modelVariant == .int4)
        #expect(model.recordedOtherVariant(of: Self.meta(.int8)) == .int8)
        #expect(model.recordedOtherVariant(of: Self.meta(.int4)) == nil)
        #expect(model.recordedOtherVariant(of: Self.meta(modelID: "google/gemma-4-26B-A4B-it")) == nil)
    }

    @Test func theStaticRefusalSpeaksOnlyForAVariantThatFitsNoContext() throws {
        #expect(AppModel.modelVariantMemoryRefusal(variant: .int4, hostMemoryBytes: eightGigabyteHost) == nil)
        #expect(AppModel.modelVariantMemoryRefusal(variant: .int8, hostMemoryBytes: sixteenGigabyteHost) == nil)
        let refusal = try #require(AppModel.modelVariantMemoryRefusal(variant: .int8,
                                                                      hostMemoryBytes: eightGigabyteHost))
        #expect(refusal == "The 8-bit model can't be loaded on this Mac. "
            + AppContextLengthOption.fourK.needDescription(hostMemoryBytes: eightGigabyteHost,
                                                           footprint: .affine8Provisional))
        // The clamp leaves a context alone when no size would fit anyway.
        let kept = AppModel.admittedContext(8_192, hostMemoryBytes: eightGigabyteHost, variant: .int8)
        #expect(kept.tokens == 8_192)
        #expect(kept.notice == nil)
        // And with the 4-bit footprint it is exactly the old clamp.
        let fourBit = AppModel.admittedContext(262_144, hostMemoryBytes: eightGigabyteHost, variant: .int4)
        let legacy = AppModel.admittedContext(262_144, hostMemoryBytes: eightGigabyteHost)
        #expect(fourBit.tokens == legacy.tokens)
        #expect(fourBit.notice == legacy.notice)
    }

    /// S24: Qwen3.6 answers admission with its family's geometry — 128K needs
    /// more than 8 GB — and the clamp for the variant matches that arithmetic,
    /// not the 4-bit Gemma one that would admit the load.
    @Test func theQwenVariantIsAdmittedByItsOwnGeometry() throws {
        #expect(AppModel.modelVariantMemoryRefusal(variant: .qwen36,
                                                   hostMemoryBytes: eightGigabyteHost) == nil)
        #expect(AppModel.modelVariantMemoryRefusal(variant: .qwen36,
                                                   hostMemoryBytes: sixteenGigabyteHost) == nil)
        let note = try #require(AppModel.contextOptionsNote(hostMemoryBytes: eightGigabyteHost,
                                                            footprint: .qwen36Provisional,
                                                            config: ArchConfig.qwen36_35B_A3B))
        #expect(note.contains("131,072"))
        #expect(note.contains("needs 16 GB"))
        // The same host on the old 4-bit Gemma figures hid nothing at 128K.
        #expect(AppModel.contextOptionsNote(hostMemoryBytes: eightGigabyteHost)
            .map { !$0.contains("131,072") } ?? true)
        // 128K is too big for 8 GB (8,657,043,456), 64K is not (7,314,866,176):
        // the clamp falls back to the largest admitted row and says so once.
        let clamped = AppModel.admittedContext(131_072, hostMemoryBytes: eightGigabyteHost,
                                               variant: .qwen36)
        #expect(clamped.tokens == 65_536)
        #expect(clamped.notice?.contains("131,072") == true)
        #expect(clamped.notice?.contains("Qwen3.6") == false) // the need sentence names memory
        let kept = AppModel.admittedContext(32_768, hostMemoryBytes: eightGigabyteHost,
                                            variant: .qwen36)
        #expect(kept.tokens == 32_768)
        #expect(kept.notice == nil)
    }

    @Test func theContextNoteUsesTheVariantsFootprint() throws {
        #expect(AppModel.contextOptionsNote(hostMemoryBytes: sixteenGigabyteHost) == nil)
        #expect(AppModel.contextOptionsNote(hostMemoryBytes: sixteenGigabyteHost,
                                            footprint: .affine8Provisional) == nil)
        let note = try #require(AppModel.contextOptionsNote(hostMemoryBytes: sixteenGigabyteHost,
                                                            expertCacheSlots: 32,
                                                            footprint: .affine8Provisional))
        #expect(note.contains("262,144"))
        #expect(!note.contains("131,072"))
        #expect(AppModel.contextOptionsNote(hostMemoryBytes: sixteenGigabyteHost, expertCacheSlots: 32) == nil)
    }

    /// S24, U4: the Qwen variant opens on its family's sampling, and the
    /// Gemma variants keep theirs — the same sliders, two sets of values.
    @MainActor @Test func theQwenVariantOpensWithItsFamilySampling() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try MacAppSettingsFileStore.save(MacAppSettings(modelVariant: .qwen36),
                                         at: AppDataLocation.settingsURL(in: root))
        let model = makeModel(root: root, client: MockLifecycleInferenceClient(), host: sixteenGigabyteHost)
        #expect(model.modelVariant == .qwen36)
        #expect(model.temperature == 1.0)
        #expect(model.topK == 20)
        #expect(model.topP == 0.95)

        await model.setModelVariant(.int4)
        #expect(model.temperature == 0.2)
        #expect(model.topK == 64)
        #expect(model.topP == 0.95)

        await model.setModelVariant(.qwen36)
        #expect(model.temperature == 1.0)
        #expect(model.topK == 20)
    }

    /// A sampling the user changed stays with the family it was changed on,
    /// under that family's keys: the untouched family keeps answering with its
    /// defaults, and a save keeps both records apart.
    @MainActor @Test func aChangedSamplingStaysWithItsFamily() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = makeModel(root: root, client: MockLifecycleInferenceClient(), host: sixteenGigabyteHost)
        #expect(model.temperature == 0.2)
        model.temperature = 0.5
        model.topK = 30

        await model.setModelVariant(.qwen36)
        #expect(model.temperature == 1.0, "Qwen took the values changed on Gemma")
        #expect(model.topK == 20)
        model.temperature = 0.7

        await model.setModelVariant(.int4)
        #expect(model.temperature == 0.5, "the Gemma edit was lost over the switch")
        #expect(model.topK == 30)

        await model.setModelVariant(.qwen36)
        #expect(model.temperature == 0.7, "the Qwen edit was lost over the switch")

        let saved = try #require(try MacAppSettingsFileStore.load(at: AppDataLocation.settingsURL(in: root)))
        #expect(saved.temperature == 0.5)
        #expect(saved.topK == 30)
        #expect(saved.temperatureQwen36 == 0.7)
        #expect(saved.topKQwen36 == 20)
    }

    // MARK: - Helpers

    private static let fakeInstallers: @Sendable (AppModelVariant) -> any AppModelInstallerClient = {
        MockModelInstallerClient(descriptor: .descriptor(for: $0))
    }

    @MainActor
    private func makeModel(root: URL, client: MockLifecycleInferenceClient, host: UInt64) -> AppModel {
        AppModel(appDataDirectory: root,
                 environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                 client: client,
                 installerFactory: Self.fakeInstallers,
                 settingsPersistenceEnabled: true,
                 hostMemoryBytes: host)
    }

    private func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppModelVariant-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.standardizedFileURL
    }

    private static func identity(_ variant: AppModelVariant) -> ConversationIdentity {
        ConversationIdentity(modelID: variant.repoID,
                             sourceSnapshotHash: "sha256:" + variant.sourceIndexSHA256,
                             templateIdentity: GFTokenizer.chatTemplateIdentity,
                             imageProcessingVersion: VisionImageProcessing.version)
    }

    private static func meta(_ variant: AppModelVariant) -> ConversationMeta {
        meta(modelID: variant.repoID)
    }

    private static func meta(modelID: String) -> ConversationMeta {
        ConversationMeta(
            id: UUID(), title: "t", createdAt: Date(), updatedAt: Date(),
            turnCount: 2, kvTokens: 10, imageCount: 0,
            identity: ConversationIdentity(modelID: modelID,
                                           sourceSnapshotHash: "sha256:00",
                                           templateIdentity: GFTokenizer.chatTemplateIdentity,
                                           imageProcessingVersion: VisionImageProcessing.version),
            session: ConversationSessionSettings(contextTokens: 8_192, expertCacheSlots: 16,
                                                 visionResidencyPolicy: "onDemand"),
            sampling: ConversationSampling(temperature: 0.2, topKEnabled: true, topK: 64,
                                           topPEnabled: true, topP: 0.95, maxNewTokens: 8_192))
    }
}
