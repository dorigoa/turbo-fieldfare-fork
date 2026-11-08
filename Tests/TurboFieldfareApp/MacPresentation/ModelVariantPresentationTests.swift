import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation

/// D7 and U3: a chat recorded with the other variant still cannot continue.
/// What changes is the copy — it names the variant, and the setting that would
/// continue the chat — so the user is not left guessing which "different
/// model" it was.
@Suite struct ModelVariantHistoryCopyTests {
    @Test(arguments: AppModelVariant.allCases)
    func aRowNamesTheVariantItWasRecordedWith(_ variant: AppModelVariant) {
        #expect(ConversationRowPresentation.subtitle(
            meta(), state: .cannotReplay(reason: .differentModel), recordedVariant: variant)
            == "recorded with the \(variant.shortLabel) model")
    }

    /// Without a known variant the old wording stands, and only the
    /// different-model reason ever names one.
    @Test func otherwiseTheWordingIsUnchanged() {
        #expect(ConversationRowPresentation.subtitle(
            meta(), state: .cannotReplay(reason: .differentModel))
            == "recorded with a different model")
        #expect(ConversationRowPresentation.subtitle(
            meta(), state: .cannotReplay(reason: .differentCheckpoint), recordedVariant: .int4)
            == "recorded with a different checkpoint")
        #expect(ConversationRowPresentation.subtitle(
            meta(kvTokens: 6_120), state: .continuable, recordedVariant: .int8)
            == ConversationRowPresentation.subtitle(meta(kvTokens: 6_120), state: .continuable))
    }

    @Test func theNoticeNamesTheVariantAndTheSettingThatContinuesIt() throws {
        let content = try #require(ConversationStateNotice.content(
            for: .cannotReplay(reason: .differentModel), currentContext: 8_192,
            recordedVariant: .int4))
        #expect(content.title == "This chat can't be continued.")
        #expect(content.message.contains("recorded with the 4-bit model"))
        #expect(content.message.contains("Model Variant to 4-bit"))
        // Still a re-read, still named as a different conversation.
        #expect(content.offersReread)
        #expect(content.requiredContext == nil)
        #expect(content.message.contains("re-read"))
        #expect(content.message.contains("different conversation"))
    }

    @Test func theNoticeIsUnchangedWithoutAVariant() throws {
        let plain = try #require(ConversationStateNotice.content(
            for: .cannotReplay(reason: .differentModel), currentContext: 8_192))
        #expect(plain.message.contains("It was recorded with a different model"))
        let checkpoint = try #require(ConversationStateNotice.content(
            for: .cannotReplay(reason: .differentCheckpoint), currentContext: 8_192,
            recordedVariant: .int8))
        #expect(!checkpoint.message.contains("8-bit"))
    }

    private func meta(kvTokens: Int? = 3_000) -> ConversationMeta {
        ConversationMeta(
            id: UUID(), title: "t", createdAt: Date(), updatedAt: Date(),
            turnCount: 2, kvTokens: kvTokens,
            identity: ConversationIdentity(modelID: AppModelVariant.int8.repoID,
                                           sourceSnapshotHash: "sha256:00",
                                           templateIdentity: "template",
                                           imageProcessingVersion: 1),
            session: ConversationSessionSettings(contextTokens: 8_192, expertCacheSlots: 16,
                                                 visionResidencyPolicy: "on-demand"),
            sampling: ConversationSampling(temperature: 0.2, topKEnabled: true, topK: 64,
                                           topPEnabled: true, topP: 0.95, maxNewTokens: 8_192))
    }
}

@Suite struct ModelVariantAccessibilityTests {
    @Test func eachVariantOptionHasItsOwnIdentifier() {
        #expect(AccessibilityID.settingsModelVariant.rawValue == "settings.modelVariant")
        #expect(AccessibilityID.modelVariantOption(.int4).rawValue == "settings.modelVariant.fourBit")
        #expect(AccessibilityID.modelVariantOption(.int8).rawValue == "settings.modelVariant.eightBit")
        #expect(AccessibilityID.inspectorVariant.rawValue == "inspector.variant")
        let options = Set(AppModelVariant.allCases.map(AccessibilityID.modelVariantOption))
        #expect(options.count == AppModelVariant.allCases.count)
    }
}

/// An installed model this Mac cannot back resolves to a state without a Load
/// action, so neither the banner nor the empty transcript offers a button the
/// model would then refuse.
@Suite struct ModelVariantPresentationStateTests {
    @Test func aRefusedModelOffersNoLoad() {
        let refusal = "The 8-bit model can't be loaded on this Mac. "
            + "A 4,096-token context needs 16 GB of memory; this Mac has 9 GB."
        let state = AppPresentationState.resolve(snapshot(memoryRefusal: refusal))
        #expect(state.label == "Needs more memory")
        #expect(state.detail == refusal)
        #expect(state.severity == .warning)
        #expect(state.primaryAction == nil)
        #expect(state.conversationAction == nil)
    }

    @Test func withoutARefusalTheInstalledModelStillOffersLoad() {
        let state = AppPresentationState.resolve(snapshot(memoryRefusal: nil))
        #expect(state.label == "Installed · Not loaded")
        #expect(state.primaryAction == .load)
    }

    /// The download stays possible below the threshold (U1).
    @Test func aRefusalDoesNotHideTheInstall() {
        var missing = snapshot(memoryRefusal: "refused")
        missing.requiresInstallation = true
        missing.installReadiness = .ready(AppModelInstallRequirement(requiredBytes: 1, availableBytes: 2))
        let state = AppPresentationState.resolve(missing)
        #expect(state.label == "Model required")
        #expect(state.primaryAction == .install)
    }

    private func snapshot(memoryRefusal: String?) -> AppPresentationSnapshot {
        AppPresentationSnapshot(requiresInstallation: false,
                                installState: .idle,
                                installReadiness: .installed,
                                loadState: .notLoaded,
                                hasStaleRuntime: false,
                                isRunning: false,
                                isGenerationCancellationPending: false,
                                generationPhase: .idle,
                                memoryRefusal: memoryRefusal)
    }
}
