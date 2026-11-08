import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareCLICore

/// The CLI reads the variant from the installed manifest, sizes its memory
/// warning with that variant's footprint and names it in the footer, without
/// changing the timing line anything already parses.
@Suite struct CLIModelVariantTests {
    static let fourBit = CLIModelVariant(
        profile: .affine4, manifestModelID: "mlx-community/gemma-4-26b-a4b-it-4bit")
    static let eightBit = CLIModelVariant(
        profile: .affine8, manifestModelID: "mlx-community/gemma-4-26b-a4b-it-8bit")

    @Test func readsTheVariantFromTheInstalledManifest() throws {
        let (eightBitDirectory, _) = try ManifestReaderTests.writeToyManifest(
            ["quant": ManifestReaderTests.quantAffine8(),
             "modelID": "mlx-community/gemma-4-26b-a4b-it-8bit"],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: eightBitDirectory) }
        let eightBit = try CLIModelVariant.read(modelDirectory: eightBitDirectory)
        #expect(eightBit == Self.eightBit)
        #expect(eightBit.label == "8-bit")
        #expect(eightBit.footprint == .affine8Provisional)
        #expect(eightBit.footprint
            == (try ModelFootprint.forModelDirectory(eightBitDirectory)))

        let (fourBitDirectory, _) = try ManifestReaderTests.writeToyManifest(
            ["quant": ManifestReaderTests.quant(),
             "modelID": "mlx-community/gemma-4-26b-a4b-it-4bit"],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: fourBitDirectory) }
        let fourBit = try CLIModelVariant.read(modelDirectory: fourBitDirectory)
        #expect(fourBit == Self.fourBit)
        #expect(fourBit.label == "4-bit")
        #expect(fourBit.footprint == .affine4Measured)
    }

    /// The historical shared-expert pack (shared expert at 8 bits) is still
    /// the 4-bit variant.
    @Test func eightBitSharedExpertIsStillFourBit() throws {
        let (directory, _) = try ManifestReaderTests.writeToyManifest(
            ["quant": ManifestReaderTests.quant(sharedExpertBits: 8)],
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(try CLIModelVariant.read(modelDirectory: directory).label == "4-bit")
    }

    @Test func manifestWithoutQuantizationIsRefused() throws {
        let (directory, _) = try ManifestReaderTests.writeToyManifest(
            config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: ModelError.self) {
            _ = try CLIModelVariant.read(modelDirectory: directory)
        }
    }

    @Test func directoryWithoutManifestIsRefused() {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-variant-missing-\(UUID().uuidString)")
        #expect(throws: (any Error).self) {
            _ = try CLIModelVariant.read(modelDirectory: directory)
        }
    }

    /// Per the user's decision U1, an 8 GB Mac is below the 8-bit estimate
    /// even at 4K, where the 4-bit model fits; the warning says which model
    /// set the need.
    @Test func memoryWarningFollowsTheVariant() {
        let eightGB: UInt64 = 8 << 30
        #expect(Self.fourBit.memoryWarning(
            maxContext: 4_096, hostMemoryBytes: eightGB, expertCacheSlots: 16) == nil)
        let need = ContextAdmission.needDescription(
            config: .gemma4_26B_A4B, maxContext: 4_096, hostMemoryBytes: eightGB,
            expertCacheSlots: 16, footprint: .affine8Provisional)
        #expect(Self.eightBit.memoryWarning(
            maxContext: 4_096, hostMemoryBytes: eightGB, expertCacheSlots: 16)
            == "[memory: 8-bit model: \(need) Continuing.]")
        #expect(Self.eightBit.memoryWarning(
            maxContext: 4_096, hostMemoryBytes: 16 << 30, expertCacheSlots: 16) == nil)
    }

    /// The 4-bit warning keeps the sentence and the numbers it printed before;
    /// only the variant is new.
    @Test func fourBitWarningKeepsItsNumbers() {
        let eightGB: UInt64 = 8 << 30
        let need = ContextAdmission.needDescription(
            config: .gemma4_26B_A4B, maxContext: 131_072, hostMemoryBytes: eightGB,
            expertCacheSlots: 32)
        #expect(Self.fourBit.memoryWarning(
            maxContext: 131_072, hostMemoryBytes: eightGB, expertCacheSlots: 32)
            == "[memory: 4-bit model: \(need) Continuing.]")
    }

    @Test func eightBitSlotsAreChargedAtTheirOwnStride() {
        let config = ArchConfig.gemma4_26B_A4B
        let fourBitNeed = ContextAdmission.minimumHostMemoryBytes(
            config: config, maxContext: 32_768, expertCacheSlots: 32,
            footprint: .affine4Measured)
        #expect(Self.fourBit.memoryWarning(
            maxContext: 32_768, hostMemoryBytes: fourBitNeed, expertCacheSlots: 32) == nil)
        #expect(Self.eightBit.memoryWarning(
            maxContext: 32_768, hostMemoryBytes: fourBitNeed, expertCacheSlots: 32) != nil)
    }

    /// The timing line is byte-for-byte the pre-variant footer and stays the
    /// last line; the variant gets a line of its own just before it.
    @Test func footerAddsAVariantLineBeforeTheUnchangedTimingLine() {
        let footer = timingFooter(reason: "endOfTurn", prefillTokens: 42, newTokens: 5,
                                  decodeSeconds: 2, variant: Self.eightBit)
        #expect(footer == "\n[variant=8-bit manifest_model_id=mlx-community/gemma-4-26b-a4b-it-8bit]\n"
            + "[stop=endOfTurn prefill=42tok new=5tok decode=2.00s tok/s=2.500]\n")
        let lines = footer.split(separator: "\n")
        #expect(lines.last?.hasPrefix("[stop=") == true)
        #expect(lines.filter { $0.hasPrefix("[stop=") }.count == 1)

        let fourBit = timingFooter(reason: "maxTokens", prefillTokens: 7, newTokens: 0,
                                   decodeSeconds: 0, variant: Self.fourBit)
        #expect(fourBit == "\n[variant=4-bit manifest_model_id=mlx-community/gemma-4-26b-a4b-it-4bit]\n"
            + "[stop=maxTokens prefill=7tok new=0tok decode=0.00s tok/s=0.000]\n")
    }

    @Test func helpNamesTheVariants() {
        #expect(Args.usage.contains("4-bit or 8-bit"))
        #expect(Args.usage.contains("variant=4-bit or variant=8-bit"))
    }
}
