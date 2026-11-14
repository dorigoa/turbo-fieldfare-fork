import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite struct SupportedModelVariantTests {
    /// `SupportedModelSource` is public API and the 4-bit default; turning it
    /// into a facade over the catalog must not move a single value.
    @Test func defaultSourceKeepsTheFourBitPin() {
        #expect(SupportedModelSource.displayName == "Gemma 4 26B-A4B IT 4-bit")
        #expect(SupportedModelSource.repoID == "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(SupportedModelSource.revision == "0d77464eeb233a2da68ebf9d7dc4edaac7db956d")
        #expect(SupportedModelSource.sourceIndexSHA256
                == "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13")
        #expect(SupportedModelSource.approximateDownloadBytes == 14_620_479_420)
        #expect(SupportedModelSource.installedBytes == 14_291_921_884)
        #expect(SupportedModelSource.reserveBytes == 1_073_741_824)
        #expect(SupportedModelSource.variant == .int4)
    }

    @Test func eightBitVariantIsPinnedByRevisionAndIndex() {
        let variant = SupportedModelVariant.int8
        #expect(variant.repoID == "mlx-community/gemma-4-26b-a4b-it-8bit")
        #expect(variant.revision == "33c6d23798a0af159529890f79329206dbfbd73c")
        #expect(variant.sourceIndexSHA256
                == "4b96ec862d7ae3f8d150b22b295cefcc9b5ef5e83f172f530dc1f2d2c11339cc")
        #expect(variant.defaultDirectoryName == "gemma4-8bit.gturbo")
        #expect(variant.recommendedMinimumHostMemoryBytes == 16 << 30)
        // The routed-expert files alone are 24,285,020,160 bytes.
        #expect(variant.installedBytes > 24_285_020_160)
        #expect(variant.approximateDownloadBytes > 24_265_359_360)
    }

    /// Pinned from a metadata-only dry run of the 8-bit source (S1). The
    /// parts are the figures that run printed; see the catalog comments.
    @Test func eightBitSizesComeFromTheDryRun() {
        let variant = SupportedModelVariant.int8
        let payloadRanges: UInt64 = 26_939_947_964
        let sourceMetadata: UInt64 = 413_177
        let tokenizerDownload: UInt64 = 32_189_832
        #expect(variant.approximateDownloadBytes
                == payloadRanges + sourceMetadata + tokenizerDownload)

        let residentFile: UInt64 = 2_545_592_786
        let expertFiles: UInt64 = 30 * 809_500_672
        let layout: UInt64 = 8_465_996
        let tokenizerInstalled: UInt64 = 32_202_012
        let measured = residentFile + expertFiles + layout + tokenizerInstalled
        #expect(variant.installedBytes == 26_871_300_000)
        // Room for manifest.json and verified-install.json (about 14 KB).
        #expect(variant.installedBytes >= measured + 16_384)
        #expect(variant.installedBytes - measured < 100_000)
    }

    /// Pinned from a metadata-only dry run of the Qwen source (S6): the
    /// parts are the figures that run printed; see the catalog comments.
    /// The minimum host memory keeps the 8-bit Mac class figure because a
    /// dry run measures disk sizes, not memory.
    @Test func qwen36VariantIsPinnedWithDryRunSizes() {
        let variant = SupportedModelVariant.qwen36
        #expect(variant.displayName == "Qwen3.6 35B-A3B 4-bit")
        #expect(variant.shortLabel == "Qwen3.6")
        #expect(variant.repoID == "mlx-community/Qwen3.6-35B-A3B-4bit")
        #expect(variant.revision == "38740b847e4cb78f352aba30aa41c76e08e6eb46")
        #expect(variant.sourceIndexSHA256
                == "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea")

        let payloadRanges: UInt64 = 20_146_183_200
        let sourceMetadata: UInt64 = 513_665
        let tokenizerDownload: UInt64 = 19_998_246
        #expect(variant.approximateDownloadBytes
                == payloadRanges + sourceMetadata + tokenizerDownload)

        let residentFile: UInt64 = 1_389_476_096
        let expertFiles: UInt64 = 40 * 452_984_832
        let layout: UInt64 = 22_493_846
        let tokenizerInstalled: UInt64 = 20_021_837
        let measured = residentFile + expertFiles + layout + tokenizerInstalled
        #expect(variant.installedBytes == 19_551_400_000)
        // The gap covers manifest.json and verified-install.json (a few KB).
        #expect(variant.installedBytes > measured)
        #expect(variant.installedBytes - measured < 100_000)
        #expect(variant.defaultDirectoryName == "qwen36.gturbo")
        #expect(variant.recommendedMinimumHostMemoryBytes == 16 << 30)
    }

    @Test func qwen36IsFoundByEveryIdentityItLeavesBehind() {
        let variant = SupportedModelVariant.qwen36
        #expect(SupportedModelVariant.variant(forSourceIndexSHA256: variant.sourceIndexSHA256)
                == variant)
        #expect(SupportedModelVariant.variant(
                    forSourceSnapshotHash: "sha256:" + variant.sourceIndexSHA256) == variant)
        #expect(SupportedModelVariant.variant(forRepoID: variant.repoID) == variant)
        #expect(SourceFingerprint.modelID(
                    forIndexSha256: variant.sourceIndexSHA256) == variant.repoID)
    }

    /// The manifest family splits the catalog (C1): no key is Gemma, and the
    /// app's model selectors list the Gemma entries until S24 lifts the filter.
    @Test func familiesSplitTheCatalogAndOnlyGemmaIsAppSelectable() {
        #expect(SupportedModelVariant.int4.family == nil)
        #expect(SupportedModelVariant.int8.family == nil)
        #expect(SupportedModelVariant.qwen36.family == .qwen35Moe)
        #expect(SupportedModelVariant.gemma4Variants == [.int4, .int8])
        #expect(SupportedModelVariant.allCases.count == 3)
    }

    @Test func variantsAreFoundByEveryIdentityTheyLeaveBehind() {
        for variant in SupportedModelVariant.allCases {
            #expect(SupportedModelVariant.variant(forSourceIndexSHA256: variant.sourceIndexSHA256)
                    == variant)
            #expect(SupportedModelVariant.variant(
                        forSourceIndexSHA256: variant.sourceIndexSHA256.uppercased()) == variant)
            #expect(SupportedModelVariant.variant(
                        forSourceSnapshotHash: "sha256:" + variant.sourceIndexSHA256) == variant)
            #expect(SupportedModelVariant.variant(forRepoID: variant.repoID) == variant)
        }
        #expect(SupportedModelVariant.variant(forSourceSnapshotHash:
                    SupportedModelVariant.int4.sourceIndexSHA256) == nil)
        #expect(SupportedModelVariant.variant(forSourceIndexSHA256: String(repeating: "0", count: 64))
                == nil)
        #expect(SupportedModelVariant.variant(forRepoID: "mlx-community/gemma-4-26b-a4b-it-6bit")
                == nil)
    }

    @Test func fingerprintsAdmitBothPinnedSources() {
        #expect(SourceFingerprint.knownFingerprints.count == SupportedModelVariant.allCases.count)
        for variant in SupportedModelVariant.allCases {
            #expect(SourceFingerprint.modelID(forIndexSha256: variant.sourceIndexSHA256)
                    == variant.repoID)
        }
    }

    @Test func installOptionsCarryTheVariantPin() {
        let output = URL(fileURLWithPath: "/tmp/gemma4-8bit.gturbo")
        let options = SupportedModelVariant.int8.installOptions(
            outputDirectory: output, overwrite: false, token: nil)
        #expect(options.repoID == SupportedModelVariant.int8.repoID)
        #expect(options.revision == SupportedModelVariant.int8.revision)
        #expect(options.requireKnownSource)
        #expect(options.outputDir == output.path)
        #expect(options.minFreeReserveBytes == SupportedModelVariant.int8.reserveBytes)
        #expect(!options.resume)
    }
}
