import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Each variant has its own download, its own directory and its own probe,
/// all taken from the catalog: a figure copied by hand into the app is one the
/// installer and the repack command can disagree with.
@Suite struct AppModelVariantCatalogTests {
    @Test(arguments: AppModelVariant.allCases)
    func theDescriptorIsTheCatalogEntry(_ variant: AppModelVariant) {
        let descriptor = AppModelInstallDescriptor.descriptor(for: variant)
        #expect(descriptor.displayName == variant.displayName)
        #expect(descriptor.repoID == variant.repoID)
        #expect(descriptor.revision == variant.revision)
        #expect(descriptor.sourceIndexSHA256 == variant.sourceIndexSHA256)
        #expect(descriptor.approximateDownloadBytes == variant.approximateDownloadBytes)
        #expect(descriptor.installedBytes == variant.installedBytes)
        #expect(descriptor.reserveBytes == variant.reserveBytes)
    }

    /// `.default` is what every caller meant before variants existed, so it
    /// keeps the 4-bit figures exactly.
    @Test func theDefaultDescriptorIsStillTheFourBitDownload() {
        let fourBit = AppModelInstallDescriptor.default
        #expect(fourBit == .descriptor(for: .int4))
        #expect(fourBit.displayName == "Gemma 4 26B-A4B IT 4-bit")
        #expect(fourBit.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(fourBit.revision == "0d77464eeb233a2da68ebf9d7dc4edaac7db956d")
        #expect(fourBit.sourceIndexSHA256
            == "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13")
        #expect(fourBit.approximateDownloadBytes == 14_620_479_420)
        #expect(fourBit.installedBytes == 14_291_921_884)
        #expect(fourBit.reserveBytes == 1_073_741_824)
        #expect(AppModelInstallDescriptor.descriptor(for: .int8) != fourBit)
    }

    @Test func eachVariantSizesAdmissionByItsOwnProfile() {
        #expect(AppModelVariant.int4.quantProfile == .affine4)
        #expect(AppModelVariant.int8.quantProfile == .affine8)
        #expect(AppModelVariant.int4.modelFootprint == .affine4Measured)
        #expect(AppModelVariant.int8.modelFootprint == .affine8Provisional)
    }

    /// S24: the two large downloads carry their caveat, derived from the
    /// catalog so the installer, the disk check and the copy never disagree.
    @Test func theLargeDownloadsCarryAnInstallNote() {
        #expect(AppModelVariant.int4.installNote == nil)
        #expect(AppModelVariant.int8.installNote == "~27 GB download; 16 GB memory recommended")
        #expect(AppModelVariant.qwen36.installNote
            == "~20 GB download; 16 GB memory recommended")
    }

    /// S24 lifted the S4 filter: the pickers list every catalog variant, so
    /// Qwen3.6 is selectable alongside the Gemma pair. `gemma4Variants` stays
    /// as the catalog's own view of the Gemma entries.
    @Test func theAppOffersEveryCatalogVariant() {
        #expect(AppModelVariant.allCases == [.int4, .int8, .qwen36])
        #expect(AppModelVariant.gemma4Variants == [.int4, .int8])
        #expect(AppModelVariant.int4.family == nil)
        #expect(AppModelVariant.int8.family == nil)
        #expect(AppModelVariant.qwen36.family != nil)
        #expect(AppModelVariant.qwen36.quantProfile == .affine4)
    }
}

@Suite struct AppModelVariantProbeTests {
    @Test(arguments: AppModelVariant.allCases)
    func theVariantIsReadFromTheManifestNotTheFolderName(_ variant: AppModelVariant) throws {
        let directory = try makeCompleteModelInstall("detect-\(variant.rawValue)", variant: variant)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(AppModelInstallationProbe.detectVariant(at: directory) == variant)
        #expect(AppModelInstallationProbe.status(anyVariantAt: directory) == .complete)
        #expect(AppModelInstallationProbe.status(at: directory, descriptor: .descriptor(for: variant))
            == .complete)
    }

    /// The probe for one variant still refuses the other: a download of the
    /// 8-bit model must not mistake a 4-bit install for its own.
    @Test func aVariantsProbeRefusesTheOtherVariant() throws {
        let eightBit = try makeCompleteModelInstall("probe-other", variant: .int8)
        defer { try? FileManager.default.removeItem(at: eightBit) }
        guard case .partial(let reason) = AppModelInstallationProbe.status(at: eightBit) else {
            Issue.record("the 4-bit probe accepted an 8-bit install")
            return
        }
        #expect(reason.contains("4-bit"))
    }

    @Test func noManifestOrAnUnknownCheckpointHasNoVariant() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("detect-missing-\(UUID()).gturbo")
        #expect(AppModelInstallationProbe.detectVariant(at: missing) == nil)
        #expect(AppModelInstallationProbe.status(anyVariantAt: missing) == .missing)

        let unknown = try makeCompleteModelInstall("detect-unknown")
        defer { try? FileManager.default.removeItem(at: unknown) }
        let manifestURL = unknown.appendingPathComponent("manifest.json")
        var manifest = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
        manifest["sourceSnapshotHash"] = "sha256:" + String(repeating: "e", count: 64)
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys]).write(to: manifestURL)
        #expect(AppModelInstallationProbe.detectVariant(at: unknown) == nil)
        guard case .partial(let reason) = AppModelInstallationProbe.status(anyVariantAt: unknown) else {
            Issue.record("an unknown checkpoint was accepted as a model")
            return
        }
        #expect(reason.contains("not a supported variant"))
    }
}

@Suite struct AppModelVariantLocationTests {
    @Test func aCheckoutKeepsEachVariantInItsOwnScratchDirectory() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TurboFieldfareApp/Mac"]
        let resolve = { (variant: AppModelVariant) in
            AppModelLocation.resolve(
                explicitURL: nil,
                executableURL: URL(fileURLWithPath: "/repo/.build/debug/TurboFieldfareMac"),
                currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
                applicationSupportURL: URL(fileURLWithPath: "/support"),
                fileExists: files.contains,
                variant: variant).path
        }
        #expect(resolve(.int4) == "/repo/scratch/gemma4.gturbo")
        #expect(resolve(.int8) == "/repo/scratch/gemma4-8bit.gturbo")
        #expect(resolve(.qwen36) == "/repo/scratch/qwen36.gturbo")
    }

    @Test func aStandaloneAppKeepsEachVariantInApplicationSupport() {
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/TurboFieldfareMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false },
            variant: .int8)
        #expect(result.path == "/support/TurboFieldfare/gemma4-8bit.gturbo")
        let qwen = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/TurboFieldfareMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false },
            variant: .qwen36)
        #expect(qwen.path == "/support/TurboFieldfare/qwen36.gturbo")
    }

    /// The 8-bit model goes where the 4-bit one is, under its own name, so
    /// the two never share a directory and a vision pack never sits between.
    @Test func anUnchosenVariantLivesBesideTheFourBitModel() {
        let fourBit = URL(fileURLWithPath: "/Users/someone/LLM/TurboFieldfare/gemma4.gturbo")
        #expect(AppModelLocation.defaultURL(for: .int8, besideFourBit: fourBit).path
            == "/Users/someone/LLM/TurboFieldfare/gemma4-8bit.gturbo")
        #expect(AppModelLocation.defaultURL(for: .qwen36, besideFourBit: fourBit).path
            == "/Users/someone/LLM/TurboFieldfare/qwen36.gturbo")
        #expect(AppModelLocation.defaultURL(for: .int4, besideFourBit: fourBit) == fourBit)
    }

    @Test func anExistingModelOfEitherVariantIsAccepted() throws {
        for variant in AppModelVariant.allCases {
            let directory = try makeCompleteModelInstall("existing-\(variant.rawValue)", variant: variant)
            defer { try? FileManager.default.removeItem(at: directory) }
            try AppModelLocation.validate(directory, intent: .existing)
            try AppModelLocation.validate(directory, intent: .existing, variant: .int4)
        }
    }

    @Test func choosingTheParentOfAnEightBitModelNamesItsFolder() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("variant-parent-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try makeCompleteModelInstall("variant-parent", variant: .int8,
                                         at: root.appendingPathComponent("gemma4-8bit.gturbo"))
        do {
            try AppModelLocation.validate(root, intent: .existing)
            Issue.record("a parent folder was accepted as a model")
        } catch {
            #expect(String(describing: error).contains("Choose the gemma4-8bit.gturbo folder inside this folder"))
        }
    }

    /// A download destination is checked against the variant being
    /// downloaded: a completed 4-bit install there is not "already done".
    @Test func aDownloadRefusesADestinationHoldingTheOtherVariant() throws {
        let fourBit = try makeCompleteModelInstall("download-other")
        defer { try? FileManager.default.removeItem(at: fourBit) }
        try AppModelLocation.validate(fourBit, intent: .download, variant: .int4)
        do {
            try AppModelLocation.validate(fourBit, intent: .download, variant: .int8)
            Issue.record("an 8-bit download accepted a 4-bit install as its destination")
        } catch {
            #expect(String(describing: error).contains("not a supported completed model"))
        }
    }
}

@Suite struct ExistingModelVariantSelectionTests {
    @Test func eachCandidateShowsItsVariantAndWhatChoosingItChanges() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("existing-variants-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try makeCompleteModelInstall("review-4", at: root.appendingPathComponent("gemma4.gturbo"))
        _ = try makeCompleteModelInstall("review-8", variant: .int8,
                                         at: root.appendingPathComponent("gemma4-8bit.gturbo"))

        let review = try ExistingModelSelection.inspect(root)
        let byName = Dictionary(uniqueKeysWithValues: review.candidates.map {
            ($0.textURL.lastPathComponent, $0)
        })
        let fourBit = try #require(byName["gemma4.gturbo"])
        let eightBit = try #require(byName["gemma4-8bit.gturbo"])
        #expect(fourBit.variant == .int4)
        #expect(eightBit.variant == .int8)
        #expect(fourBit.canSelect && eightBit.canSelect)
        #expect(eightBit.variantSummary == "8-bit model")
        #expect(eightBit.variantSwitchNote(from: .int4)
            == "Using this model switches Model Variant to 8-bit.")
        #expect(eightBit.variantSwitchNote(from: .int8) == nil)
        #expect(fourBit.variantSwitchNote(from: .int4) == nil)
        // Two usable models: the user picks, nothing is chosen for them.
        #expect(review.initialSelection == nil)
    }
}
