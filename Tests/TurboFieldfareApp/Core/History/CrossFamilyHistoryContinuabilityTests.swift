import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// S25 step 7: the two catalog families separate stored chats the way the
/// quantization variants already did — one `modelID`, one rule. The identities
/// here are not hand-built: each comes out of an installed manifest through
/// the app's own provider, so the test proves the whole chain (manifest →
/// `ConversationIdentity` → `ConversationContinuability`) rather than one
/// field chosen to differ.
@Suite struct CrossFamilyConversationIdentityTests {

    private func install(_ tag: String, _ variant: AppModelVariant) throws -> URL {
        try makeCompleteModelInstall(
            tag, variant: variant, modelID: variant.repoID,
            at: FileManager.default.temporaryDirectory
                .appendingPathComponent("cross-family-\(tag)-\(UUID().uuidString).gturbo",
                                       isDirectory: true))
    }

    private func meta(recordedWith identity: ConversationIdentity) -> ConversationMeta {
        ConversationMeta(
            id: UUID(), title: "t", createdAt: Date(), updatedAt: Date(),
            turnCount: 2, kvTokens: 100, imageCount: 0,
            identity: identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "onDemand"),
            sampling: ConversationSampling(
                temperature: 0.2, topKEnabled: true, topK: 64,
                topPEnabled: true, topP: 0.95, maxNewTokens: 4_096))
    }

    @Test func eachFamilyManifestCarriesItsOwnIdentityAndRefusesTheOther() throws {
        let qwen = try install("qwen-identity", .qwen36)
        let gemma = try install("gemma-identity", .int4)
        defer {
            try? FileManager.default.removeItem(at: qwen)
            try? FileManager.default.removeItem(at: gemma)
        }

        let qwenIdentity = try ConversationIdentity.forModelDirectory(qwen)
        let gemmaIdentity = try ConversationIdentity.forModelDirectory(gemma)
        #expect(qwenIdentity.modelID == AppModelVariant.qwen36.repoID)
        #expect(gemmaIdentity.modelID == AppModelVariant.int4.repoID)
        // Both modelIDs name a catalog variant, so a refused row can say
        // "recorded with the Qwen3.6 model" instead of "a different model".
        #expect(AppModelVariant.variant(forRepoID: qwenIdentity.modelID) == .qwen36)
        #expect(AppModelVariant.variant(forRepoID: gemmaIdentity.modelID) == .int4)

        // Cross-family: refused in both directions, on the model alone.
        #expect(ConversationContinuability.evaluate(
            meta: meta(recordedWith: gemmaIdentity), currentContext: 8_192,
            identity: qwenIdentity) == .cannotReplay(reason: .differentModel))
        #expect(ConversationContinuability.evaluate(
            meta: meta(recordedWith: qwenIdentity), currentContext: 8_192,
            identity: gemmaIdentity) == .cannotReplay(reason: .differentModel))
        // Same family continues: the refusal is the family, not the fixture.
        #expect(ConversationContinuability.evaluate(
            meta: meta(recordedWith: qwenIdentity), currentContext: 8_192,
            identity: qwenIdentity) == .continuable)
        #expect(ConversationContinuability.evaluate(
            meta: meta(recordedWith: gemmaIdentity), currentContext: 8_192,
            identity: gemmaIdentity) == .continuable)
    }
}

/// The same rule one level up: with one family installed, a chat recorded
/// with the other opens readable and stays read-only — a Gemma chat under
/// Qwen, and a Qwen chat under Gemma. The running model reads its identity
/// from the directory's own manifest, exactly as it does outside tests.
@Suite(.serialized) struct CrossFamilyHistoryReadOnlyTests {
    private let thirtyTwoGigabyteHost: UInt64 = 32 << 30

    @MainActor
    @Test(arguments: [
        (loaded: AppModelVariant.qwen36, recorded: AppModelVariant.int4),
        (loaded: AppModelVariant.int4, recorded: AppModelVariant.qwen36),
    ])
    func aChatRecordedWithTheOtherFamilyOpensButDoesNotContinue(
        _ pair: (loaded: AppModelVariant, recorded: AppModelVariant)
    ) async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("cross-family-history-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let loadedDirectory = try makeCompleteModelInstall(
            "loaded", variant: pair.loaded, modelID: pair.loaded.repoID,
            at: root.appendingPathComponent("loaded-\(pair.loaded.shortLabel)", isDirectory: true))
        let recordedDirectory = try makeCompleteModelInstall(
            "recorded", variant: pair.recorded, modelID: pair.recorded.repoID,
            at: root.appendingPathComponent("recorded-\(pair.recorded.shortLabel)", isDirectory: true))

        // The chat carries exactly the identity its own install produces.
        let recordedIdentity = try ConversationIdentity.forModelDirectory(recordedDirectory)
        let store = ConversationStore(
            rootURL: root.appendingPathComponent("conversations", isDirectory: true))
        try await store.activate()
        let chat = try await store.create(
            title: "cross-family", identity: recordedIdentity,
            session: ConversationSessionSettings(contextTokens: 8_192, expertCacheSlots: 16,
                                                 visionResidencyPolicy: "onDemand"),
            sampling: ConversationSampling(temperature: 0, topKEnabled: true, topK: 64,
                                           topPEnabled: true, topP: 0.95, maxNewTokens: 64))
        _ = try await store.append([
            .turn(ConversationTurnRecord(role: .user, at: Date(), text: "stored", tokens: [1])),
            .turn(ConversationTurnRecord(role: .assistant, at: Date(), text: "answer", tokens: [2],
                                         boundary: ConversationBoundary(tokens: [3], needsReplay: false))),
        ], to: chat.id)

        // No injected identity: the installed manifest decides who is running.
        let model = AppModel(
            modelDirectory: loadedDirectory,
            appDataDirectory: root,
            client: FakeInferenceClient(),
            settingsPersistenceEnabled: true,
            conversationStoreProvider: { _ in store },
            hostMemoryBytes: thirtyTwoGigabyteHost)
        try await model.waitForHistory(count: 1)
        #expect(model.modelVariant == pair.loaded)
        #expect(model.conversationIdentity?.modelID == pair.loaded.repoID)

        let meta = try #require(model.history.entry(chat.id))
        #expect(model.continuability(of: meta) == .cannotReplay(reason: .differentModel))
        #expect(model.recordedOtherVariant(of: meta) == pair.recorded)

        model.openConversation(id: chat.id)
        try await waitUntil { await !model.transcriptHistory.isEmpty }
        #expect(model.openedConversationState == .cannotReplay(reason: .differentModel))
        #expect(model.openedConversationRecordedVariant == pair.recorded)
        // Read-only, not gone: the transcript is drawn whole from disk.
        #expect(model.transcriptHistory.map(\.user.text) == ["stored"])
    }
}
