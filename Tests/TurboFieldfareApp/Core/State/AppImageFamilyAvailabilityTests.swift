import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// Which families can take image input (D12, S24).
///
/// Qwen3.6 has no image tower, so no companion pack, runtime flag or GPU can
/// make images real for it. The app must say so with the same sentence at
/// every point of contact — and Gemma must keep behaving exactly as before.
@Suite(.serialized) struct AppImageFamilyAvailabilityTests {
    private func withVisionEnabled<T>(_ body: () throws -> T) rethrows -> T {
        let previous = ProcessInfo.processInfo.environment[
            "TURBO_FIELDFARE_VISION_RUNTIME"]
        setenv("TURBO_FIELDFARE_VISION_RUNTIME", "1", 1)
        defer {
            if let previous {
                setenv("TURBO_FIELDFARE_VISION_RUNTIME", previous, 1)
            } else {
                unsetenv("TURBO_FIELDFARE_VISION_RUNTIME")
            }
        }
        return try body()
    }

    @Test func theImageTowerBelongsToTheFamily() {
        #expect(AppModelVariant.int4.supportsImages)
        #expect(AppModelVariant.int8.supportsImages)
        #expect(!AppModelVariant.qwen36.supportsImages,
                "the Qwen3.6 checkpoint has no trained image tower")
    }

    /// The whole point of the family gate: with a companion pack on disk and
    /// the runtime flag on, Qwen still never offers images; Gemma still does.
    @MainActor
    @Test func theQwenFamilyNeverOffersImageInputEvenWithACompanionPack() throws {
        let qwen = try makeCompleteModelInstall("family-qwen", variant: .qwen36)
        defer { try? FileManager.default.removeItem(at: qwen) }
        try installVisionCompanion(forTextModel: qwen)
        let model = AppModel(modelDirectory: qwen)
        #expect(model.modelVariant == .qwen36)
        #expect(model.isModelInstalled)
        #expect(model.isVisionPackInstalled,
                "the fixture must really place a pack beside the model")

        withVisionEnabled {
            #expect(!model.isImageInputAvailable,
                    "a family with no image tower must never accept images")
        }

        // Gemma control: the same gate leaves the trained family untouched.
        let gemma = try makeVisionReadyModelInstall("family-gemma")
        defer { try? FileManager.default.removeItem(at: gemma) }
        let gemmaModel = AppModel(modelDirectory: gemma)
        #expect(gemmaModel.modelVariant == .int4)
        withVisionEnabled {
            #expect(gemmaModel.isImageInputAvailable,
                    "Gemma's image path must not change")
        }
    }

    /// A drag or paste that reaches the model anyway (the editor gate is a
    /// view, not a wall) is refused with the family sentence, and leaves no
    /// staged copy behind — never "restore the companion pack", which for
    /// this family describes a repair that cannot exist.
    @MainActor
    @Test func anImageDroppedOnQwenIsRefusedWithTheFamilyReason() throws {
        let qwen = try makeCompleteModelInstall("family-drop", variant: .qwen36)
        defer { try? FileManager.default.removeItem(at: qwen) }
        try installVisionCompanion(forTextModel: qwen)

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("family-drop-src-\(UUID().uuidString)",
                                    isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root,
                                                withIntermediateDirectories: true)
        let source = root.appendingPathComponent("picture.png")
        try Data("fixture".utf8).write(to: source)

        let storeRoot = root.appendingPathComponent("store", isDirectory: true)
        let store = AppImageAttachmentStore(
            directoryURL: storeRoot.appendingPathComponent("staged", isDirectory: true))
        let model = AppModel(modelDirectory: qwen, attachmentStore: store)

        withVisionEnabled { model.addImages([source]) }

        #expect(model.imageAttachments.isEmpty)
        #expect(model.imageAttachmentError
                    == "Image input is not available for this model.",
                "the refusal must name the family reason, not a missing pack")
        let staged = FileManager.default.enumerator(at: store.directoryURL,
                                                    includingPropertiesForKeys: nil)
        #expect(staged == nil || !staged!.lazy.contains { _ in true },
                "a refused image must not leave a staged copy behind")
    }
}
