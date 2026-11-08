import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Image support with the 8-bit model (S10, option A): the companion pack is
/// still downloaded from the 4-bit checkpoint, because the tower is BF16 in
/// both and only the projector differs, and it is bound to the 8-bit text
/// model's manifest. Each variant keeps its own `<stem>.vision.gturbo`, and a
/// pack bound to the other variant is never accepted.
@Suite(.serialized) struct AppModelVariantVisionTests {
    private let thirtyTwoGigabyteHost: UInt64 = 32 << 30

    @MainActor @Test func eachVariantProbesItsOwnCompanionFromTheFourBitSource() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let fourBit = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let eightBit = root.appendingPathComponent("gemma4-8bit.gturbo", isDirectory: true)
        _ = try makeCompleteModelInstall("vision-4", variant: .int4, at: fourBit)
        _ = try makeCompleteModelInstall("vision-8", variant: .int8, at: eightBit)
        try installFourBitSourcePack(boundTo: fourBit)
        let model = makeModel(root: root)
        try #require(model.modelVariant == .int4)
        #expect(model.visionInstallationStatus == .complete)

        await model.setModelVariant(.int8)

        #expect(model.modelPathText == eightBit.path)
        #expect(model.visionInstallationStatus == .missing,
                "the 4-bit model's pack must not count as the 8-bit model's")
        #expect(!model.hasVisionPackDirectory)
        #expect(model.visionInstallDescriptor == .visionCompanion)
        #expect(model.visionInstallDescriptor.repoID == AppModelVariant.int4.repoID,
                "option A downloads the image pack from the 4-bit checkpoint")

        let companion = try installFourBitSourcePack(boundTo: eightBit)
        #expect(companion.lastPathComponent == "gemma4-8bit.vision.gturbo")
        model.refreshVisionInstallReadiness()
        #expect(model.visionInstallationStatus == .complete)
        #expect(model.hasVisionPackDirectory)

        await model.setModelVariant(.int4)

        #expect(model.visionInstallationStatus == .complete)
        #expect(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("gemma4.vision.gturbo").path))
    }

    /// Fail-closed: a pack bound to the 4-bit manifest, put where the 8-bit
    /// model looks for its companion, is reported as needing repair rather
    /// than accepted.
    @MainActor @Test func aPackBoundToTheOtherVariantIsRefused() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let fourBit = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let eightBit = root.appendingPathComponent("gemma4-8bit.gturbo", isDirectory: true)
        _ = try makeCompleteModelInstall("vision-wrong-4", variant: .int4, at: fourBit)
        _ = try makeCompleteModelInstall("vision-wrong-8", variant: .int8, at: eightBit)
        try installFourBitSourcePack(
            boundTo: fourBit,
            at: root.appendingPathComponent("gemma4-8bit.vision.gturbo", isDirectory: true))

        let status = AppVisionPackInstallationProbe.status(at: eightBit)

        guard case .partial = status else {
            Issue.record("a pack bound to the 4-bit manifest was reported as \(status)")
            return
        }
        #expect(AppVisionPackInstallationProbe.status(at: fourBit) == .missing)
    }

    // MARK: - Helpers

    @discardableResult
    private func installFourBitSourcePack(boundTo text: URL, at location: URL? = nil) throws -> URL {
        try installVisionCompanion(forTextModel: text,
                                   sourceRepoID: AppModelVariant.int4.repoID,
                                   sourceIndexSha256: AppModelVariant.int4.sourceIndexSHA256,
                                   at: location)
    }

    @MainActor
    private func makeModel(root: URL) -> AppModel {
        AppModel(appDataDirectory: root,
                 environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                 client: MockLifecycleInferenceClient(),
                 installerFactory: { MockModelInstallerClient(descriptor: .descriptor(for: $0)) },
                 settingsPersistenceEnabled: true,
                 hostMemoryBytes: thirtyTwoGigabyteHost)
    }

    private func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppModelVariantVision-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.standardizedFileURL
    }
}
