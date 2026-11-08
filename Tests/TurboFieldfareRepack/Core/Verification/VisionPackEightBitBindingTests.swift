import Foundation
import Testing
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// Image support with the 8-bit text model (S10, option A). The companion pack
/// comes from the pinned 4-bit checkpoint — the tower is BF16 in both, only the
/// projector differs — and is bound to the 8-bit model's manifest, which is
/// what the verifier and the runtime check. These cases pin that a 4-bit-source
/// pack beside `gemma4-8bit.gturbo` verifies and activates, and that the
/// binding still refuses the other variant's text model.
@Suite struct VisionPackEightBitBindingTests {
    @Test func aFourBitSourcePackVerifiesAgainstTheEightBitTextModel() throws {
        let fixture = try Fixture.make(tag: "verify")
        defer { fixture.remove() }
        let pack = try fixture.writePack(boundTo: fixture.eightBit, in: fixture.eightBitCompanion)

        let verification = try VisionPackVerifier.verify(
            directory: pack, textModelDirectory: fixture.eightBit)

        #expect(verification.sourceIndexSha256 == SupportedModelVariant.int4.sourceIndexSHA256)
        #expect(verification.sourceRevision == SupportedModelVariant.int4.revision)
        #expect(verification.compatibleTextManifestSha256 == fixture.eightBitManifestSHA)
    }

    @Test func thePackBoundToOneVariantIsRefusedForTheOther() throws {
        let fixture = try Fixture.make(tag: "refuse")
        defer { fixture.remove() }
        let eightBitPack = try fixture.writePack(boundTo: fixture.eightBit,
                                                 in: fixture.eightBitCompanion)
        let fourBitPack = try fixture.writePack(boundTo: fixture.fourBit,
                                                in: fixture.fourBitCompanion)

        for (pack, text) in [(eightBitPack, fixture.fourBit), (fourBitPack, fixture.eightBit)] {
            let error = #expect(throws: RepackError.self) {
                try VisionPackVerifier.verify(directory: pack, textModelDirectory: text)
            }
            #expect("\(error.map { "\($0)" } ?? "")".contains("different text checkpoint"))
        }
        _ = try VisionPackVerifier.verify(directory: fourBitPack, textModelDirectory: fixture.fourBit)
    }

    /// What `--activate-vision-install` and the app do after a download: the
    /// pinned 4-bit source identity activates a pack prepared for the 8-bit
    /// model.
    @Test func aPreparedPackActivatesBesideTheEightBitModelWithTheFourBitSource() throws {
        let fixture = try Fixture.make(tag: "activate")
        defer { fixture.remove() }
        let paths = try RemoteInstallPaths(outputDirectory: fixture.eightBitCompanion.path)
        _ = try fixture.writePack(
            boundTo: fixture.eightBit,
            in: URL(fileURLWithPath: paths.partialDirectory, isDirectory: true),
            receiptDirectory: fixture.eightBitCompanion)
        try RemoteInstallCheckpoint(
            repoID: SupportedModelSource.repoID,
            requestedRevision: SupportedModelSource.revision,
            resolvedCommit: SupportedModelVariant.int4.revision,
            sourceIndexSHA256: SupportedModelVariant.int4.sourceIndexSHA256,
            planFingerprint: String(repeating: "c", count: 64),
            totalSourceBytes: 2).write(to: paths.checkpointFile,
                                       parentDirectory: paths.parentDirectory)

        try RemoteVisionPackInstaller.activatePrepared(
            outputDirectory: fixture.eightBitCompanion.path,
            textModelDirectory: fixture.eightBit.path,
            repoID: SupportedModelSource.repoID,
            requestedRevision: SupportedModelSource.revision)

        #expect(FileManager.default.fileExists(atPath: fixture.eightBitCompanion.path))
        #expect(!FileManager.default.fileExists(atPath: paths.partialDirectory))
        #expect(!FileManager.default.fileExists(atPath: paths.checkpointFile))
        _ = try VisionPackVerifier.verify(directory: fixture.eightBitCompanion,
                                          textModelDirectory: fixture.eightBit)
    }
}

private struct Fixture {
    let parent: URL
    let fourBit: URL
    let eightBit: URL
    let eightBitManifestSHA: String

    var fourBitCompanion: URL { parent.appendingPathComponent("gemma4.vision.gturbo", isDirectory: true) }
    var eightBitCompanion: URL { parent.appendingPathComponent("gemma4-8bit.vision.gturbo", isDirectory: true) }

    static func make(tag: String) throws -> Fixture {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-8bit-\(tag)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        let fourBit = parent.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let eightBit = parent.appendingPathComponent("gemma4-8bit.gturbo", isDirectory: true)
        _ = try writeTextModel(.int4, at: fourBit)
        let eightBitSHA = try writeTextModel(.int8, at: eightBit)
        return Fixture(parent: parent, fourBit: fourBit, eightBit: eightBit,
                       eightBitManifestSHA: eightBitSHA)
    }

    /// The text manifest a real install of `variant` records: its repo ID, its
    /// pinned source hash and its quantization profile.
    private static func writeTextModel(_ variant: SupportedModelVariant, at directory: URL) throws -> String {
        let bits = variant == .int8 ? 8 : 4
        func slot(_ bits: Int) -> GTurboManifestQuantSlotV1 {
            GTurboManifestQuantSlotV1(weightBits: bits, scheme: "affine", scaleType: "bf16",
                                      biasType: "bf16", groupSize: 64)
        }
        let manifest = GTurboManifestV1(
            flags: [:], modelID: variant.repoID,
            sourceSnapshotHash: "sha256:" + variant.sourceIndexSHA256,
            arch: GTurboManifestArchV1(
                hiddenSize: 1, ffnIntermediate: 1, moeIntermediateSize: 1,
                numHeads: 1, numKVHeads: 1, numFullKVHeads: 1,
                headDim: 1, fullHeadDim: 1, vocabSize: 1,
                slidingWindow: 1, finalLogitSoftcap: 1,
                ropeTheta: 1, fullRopeTheta: 1, partialRotaryFactor: 1,
                numLayers: 1, numExperts: 1, topKExperts: 1,
                tieWordEmbeddings: true, attentionKEqV: true,
                hiddenActivation: "gelu", fullAttentionLayerMask: [0]),
            quant: GTurboManifestQuantV1(embedding: slot(bits), attention: slot(bits),
                                         router: slot(8), sharedExpert: slot(bits),
                                         routedExpert: slot(bits)),
            files: [:], expertsPerLayer: 1, numLayers: 1, expertStride: 16_384,
            bitWidthOverridesHonored: nil)
        let data = try GTurboManifestCodec.encode(manifest)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try data.write(to: directory.appendingPathComponent("manifest.json"))
        return hash(data)
    }

    /// A pack whose tower comes from the pinned 4-bit checkpoint, bound to
    /// `text`'s manifest, written into `directory`; the receipt names
    /// `receiptDirectory` (the final location) when the pack is still partial.
    func writePack(boundTo text: URL, in directory: URL, receiptDirectory: URL? = nil) throws -> URL {
        let textManifestData = try Data(contentsOf: text.appendingPathComponent("manifest.json"))
        let textManifest = try GTurboManifestCodec.decode(textManifestData)
        let textSHA = Self.hash(textManifestData)
        guard let textSource = textManifest.sourceSnapshotHash else {
            throw CocoaError(.fileReadCorruptFile)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let weights = Data(repeating: 0, count: 16_384)
        let processor = Data("{}".utf8)
        let files = [
            GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
                size: UInt64(weights.count), sha256: Self.hash(weights)),
            GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(
                size: UInt64(processor.count), sha256: Self.hash(processor)),
        ]
        let source = SupportedModelVariant.int4
        let manifest = GTurboVisionManifestV1(
            modelID: source.repoID, sourceRevision: source.revision,
            sourceIndexSha256: source.sourceIndexSHA256,
            processorConfigSha256: Self.hash(processor),
            compatibleTextSourceSnapshotHash: textSource,
            compatibleTextManifestSha256: textSHA,
            files: files,
            tensors: [.init(name: "vision.weight", executionPosition: 0,
                            offset: 0, size: 2, shape: [1], dtype: .bf16)])
        let manifestData = try GTurboVisionManifestCodec.encode(manifest)
        let manifestSHA = Self.hash(manifestData)
        let receipt = GTurboVisionReceiptV1(
            manifestSha256: manifestSHA,
            companionDirectoryPath: (receiptDirectory ?? directory).standardizedFileURL.path,
            compatibleTextManifestSha256: textSHA,
            sourceRepoID: source.repoID, sourceRevision: source.revision,
            verificationTimestamp: "fixture", toolVersion: "fixture",
            files: files.merging([
                GTurboVisionFormatV1.manifestFile: GTurboManifestFileV1(
                    size: UInt64(manifestData.count), sha256: manifestSHA),
            ]) { _, new in new })
        try weights.write(to: directory.appendingPathComponent(GTurboVisionFormatV1.weightsFile))
        try processor.write(to: directory.appendingPathComponent(GTurboVisionFormatV1.processorFile))
        try manifestData.write(to: directory.appendingPathComponent(GTurboVisionFormatV1.manifestFile))
        try GTurboVisionReceiptCodec.encode(receipt).write(
            to: directory.appendingPathComponent(GTurboVisionFormatV1.receiptFile))
        return directory
    }

    func remove() {
        try? FileManager.default.removeItem(at: parent)
    }

    private static func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }
}
