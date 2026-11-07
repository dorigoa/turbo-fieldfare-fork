import Foundation
import Testing
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
