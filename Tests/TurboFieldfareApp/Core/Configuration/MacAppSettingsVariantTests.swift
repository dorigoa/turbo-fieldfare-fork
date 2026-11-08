import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// `modelVariant` and `selectedModelPathInt8` are additive: a file written
/// before them reads as the 4-bit model, and a file written after them still
/// gives an older build the 4-bit location it knows, under the version it
/// accepts. Neither direction may cost the user their other settings.
@Suite struct MacAppSettingsVariantTests {
    @Test func aFileWithoutTheVariantKeysReadsAsTheFourBitModel() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        try Data("""
        {"version":4,"selectedModelPath":"/models/gemma4.gturbo",\(Self.requiredFields(contextTokens: 16_384)),"textSize":125}
        """.utf8).write(to: file)

        let settings = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(settings.modelVariant == .int4)
        #expect(settings.selectedModelPathInt8 == nil)
        #expect(settings.selectedModelPath == "/models/gemma4.gturbo")
        #expect(settings.selectedModelPath(for: .int4) == "/models/gemma4.gturbo")
        #expect(settings.selectedModelPath(for: .int8) == nil)
        #expect(settings.contextTokens == 16_384)
        #expect(settings.textSize == .large)
    }

    @Test(arguments: AppModelVariant.allCases)
    func theVariantAndBothLocationsRoundTrip(_ variant: AppModelVariant) throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        let written = MacAppSettings(selectedModelPath: "/models/gemma4.gturbo",
                                     modelVariant: variant,
                                     selectedModelPathInt8: "/models/gemma4-8bit.gturbo")
        try MacAppSettingsFileStore.save(written, at: file)

        let read = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(read == written)
        #expect(read.selectedModelPath(for: .int8) == "/models/gemma4-8bit.gturbo")
    }

    /// What an older build reads: the version it accepts, the 4-bit location
    /// under the key it knows, and two keys it ignores. Decoding through a
    /// struct that has never heard of the variant stands in for that build.
    @Test func anOlderBuildReadingANewFileStillFindsTheFourBitModel() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        try MacAppSettingsFileStore.save(MacAppSettings(selectedModelPath: "/models/gemma4.gturbo",
                                                        modelVariant: .int8,
                                                        selectedModelPathInt8: "/models/gemma4-8bit.gturbo"),
                                         at: file)

        struct OlderBuildSettings: Decodable {
            let version: Int
            let selectedModelPath: String?
        }
        let older = try JSONDecoder().decode(OlderBuildSettings.self, from: Data(contentsOf: file))
        #expect(older.version == MacAppSettings.currentVersion)
        #expect(older.selectedModelPath == "/models/gemma4.gturbo")
        let raw = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        #expect(raw["modelVariant"] as? String == "int8")
        #expect(raw["selectedModelPathInt8"] as? String == "/models/gemma4-8bit.gturbo")
    }

    /// The same terms as `textSize`: a newer build's variant falls back to the
    /// 4-bit model instead of discarding the whole file.
    @Test(arguments: ["\"int2\"", "42", "null"])
    func anUnknownVariantFallsBackToFourBitWithoutResettingPreferences(_ value: String) throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        try Data("""
        {"version":4,"selectedModelPath":"/models/gemma4.gturbo","modelVariant":\(value),\(Self.requiredFields(topK: 12))}
        """.utf8).write(to: file)

        let settings = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(settings.modelVariant == .int4)
        #expect(settings.topK == 12)
        #expect(settings.selectedModelPath == "/models/gemma4.gturbo")
    }

    @Test(arguments: ["relative/gemma4-8bit.gturbo", "/models/nul\u{0}.gturbo"])
    func anInvalidEightBitLocationIsRefusedLikeTheFourBitOne(_ path: String) {
        #expect(!MacAppSettings(selectedModelPathInt8: path).isValid())
        #expect(MacAppSettings(selectedModelPathInt8: "/models/gemma4-8bit.gturbo").isValid())
    }

    @Test func bothLocationsAreStandardizedOnLoad() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        try MacAppSettingsFileStore.save(MacAppSettings(selectedModelPath: "/models/./a/../gemma4.gturbo",
                                                        selectedModelPathInt8: "/models/b/../gemma4-8bit.gturbo"),
                                         at: file)
        let settings = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(settings.selectedModelPath == "/models/gemma4.gturbo")
        #expect(settings.selectedModelPathInt8 == "/models/gemma4-8bit.gturbo")
    }

    /// Someone who only ever chose an 8-bit location has chosen; the legacy
    /// 4-bit search must not run over that choice on the next launch.
    @Test func anEightBitLocationAloneSkipsTheLegacySearch() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = AppDataLocation.settingsURL(in: root)
        let written = MacAppSettings(modelVariant: .int8,
                                     selectedModelPathInt8: "/models/gemma4-8bit.gturbo")
        try MacAppSettingsFileStore.save(written, at: file)

        let bootstrap = AppDataBootstrap(root: root,
                                         legacyModel: root.appendingPathComponent("gemma4.gturbo"),
                                         environment: ["TURBO_FIELDFARE_APP_DATA_PATH": root.path],
                                         enabled: true)
        #expect(bootstrap.failure == nil)
        #expect(bootstrap.settings.selectedModelPath == nil)
        #expect(bootstrap.settings.selectedModelPathInt8 == "/models/gemma4-8bit.gturbo")
        #expect(try MacAppSettingsFileStore.load(at: file) == written)
    }

    /// The keys a version 4 file always carries; everything after them in
    /// the schema is optional.
    private static func requiredFields(contextTokens: Int = 8_192, topK: Int = 64) -> String {
        "\"contextTokens\":\(contextTokens),\"expertCacheSlots\":16,\"temperature\":0.2,"
            + "\"topKEnabled\":true,\"topK\":\(topK),\"topPEnabled\":true,\"topP\":0.95,"
            + "\"prefillEnabled\":true"
    }

    private func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("settings-variant-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
