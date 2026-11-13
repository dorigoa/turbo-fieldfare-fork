import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// S24, U4: sampling defaults per family. The Qwen keys are additive and
/// optional — a file that never saw them keeps answering with the family's
/// own values, and one that carries them round-trips the user's changes
/// without moving anything in the Gemma keys that predate variants.
@Suite struct MacAppSettingsSamplingTests {

    @Test func theFamiliesCarryThePlannedDefaults() {
        let qwen = AppModelVariant.qwen36.defaultSampling
        #expect(qwen.temperature == 1.0)
        #expect(qwen.topK == 20)
        #expect(qwen.topP == 0.95)
        #expect(qwen.topKEnabled && qwen.topPEnabled)
        let gemma = AppModelVariant.int4.defaultSampling
        #expect(gemma.temperature == 0.2)
        #expect(gemma.topK == 64)
        #expect(gemma.topP == 0.95)
        #expect(AppModelVariant.int8.defaultSampling == gemma)
    }

    /// A file written before the Qwen keys existed reads as it did, and the
    /// Qwen family answers with its defaults — there is nothing of the user's
    /// for it yet.
    @Test func aFileWithoutTheQwenSamplingKeysReadsAsTheFamilyDefaults() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        try Data("""
        {"version":4,"selectedModelPath":"/models/gemma4.gturbo",\
        "contextTokens":8192,"expertCacheSlots":16,"temperature":0.5,\
        "topKEnabled":false,"topK":40,"topPEnabled":true,"topP":0.9,\
        "prefillEnabled":true}
        """.utf8).write(to: file)

        let settings = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(settings.temperatureQwen36 == nil)
        let qwen = settings.sampling(for: .qwen36)
        #expect(qwen == AppModelVariant.qwen36.defaultSampling)
        let gemma = settings.sampling(for: .int4)
        #expect(gemma.temperature == 0.5)
        #expect(!gemma.topKEnabled)
        #expect(gemma.topK == 40)
        #expect(gemma.topP == 0.9)
        #expect(settings.sampling(for: .int8) == gemma)
    }

    /// A changed Qwen sampling is stored under its own keys, survives a save
    /// and reload, and leaves the Gemma keys exactly as they were.
    @Test func changedQwenSamplingRoundTripsUnderItsOwnKeys() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        var settings = MacAppSettings(temperature: 0.2, topK: 64, topP: 0.95)
        settings.setSampling(AppSamplingValues(temperature: 0.5, topKEnabled: true, topK: 30,
                                               topPEnabled: false, topP: 0.8),
                             for: .qwen36)
        try MacAppSettingsFileStore.save(settings, at: file)

        let read = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(read.temperatureQwen36 == 0.5)
        #expect(read.topKQwen36 == 30)
        #expect(read.sampling(for: .qwen36).temperature == 0.5)
        #expect(read.sampling(for: .qwen36).topK == 30)
        #expect(!read.sampling(for: .qwen36).topPEnabled)
        #expect(read.temperature == 0.2)
        #expect(read.topK == 64)
        #expect(read.topP == 0.95)
        let raw = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        #expect(raw["temperatureQwen36"] as? Double == 0.5)
    }

    /// A Qwen sampling equal to the family defaults was never really moved:
    /// the overrides are dropped, so the family's values keep steering.
    @Test func qwenSamplingAtTheDefaultsStaysUnsaved() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        var settings = MacAppSettings()
        settings.setSampling(AppSamplingValues(temperature: 0.9, topKEnabled: true, topK: 20,
                                               topPEnabled: true, topP: 0.95),
                             for: .qwen36)
        #expect(settings.temperatureQwen36 == 0.9)
        settings.setSampling(AppModelVariant.qwen36.defaultSampling, for: .qwen36)
        #expect(settings.temperatureQwen36 == nil)
        try MacAppSettingsFileStore.save(settings, at: file)

        let raw = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        #expect(raw["temperatureQwen36"] == nil)
        let read = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(read.sampling(for: .qwen36) == AppModelVariant.qwen36.defaultSampling)
    }

    /// The two Gemma variants share the keys they had before variants; a Gemma
    /// edit is stored where Gemma has always been stored, not per variant.
    @Test func aGemmaEditLandsInTheKeysThatPredateVariants() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        var settings = MacAppSettings()
        settings.setSampling(AppSamplingValues(temperature: 0.8, topKEnabled: true, topK: 50,
                                               topPEnabled: true, topP: 0.9),
                             for: .int8)
        try MacAppSettingsFileStore.save(settings, at: file)

        let read = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(read.temperature == 0.8)
        #expect(read.topK == 50)
        #expect(read.sampling(for: .int4).temperature == 0.8)
        #expect(read.temperatureQwen36 == nil)
        #expect(read.sampling(for: .qwen36) == AppModelVariant.qwen36.defaultSampling)
    }

    /// Overrides are written as a complete set; half of one names values no
    /// slider could have produced, which is settings recovery, not guessing.
    @Test(arguments: ["\"temperatureQwen36\":1.5",
                      "\"temperatureQwen36\":1.5,\"topKEnabledQwen36\":true",
                      "\"temperatureQwen36\":3.0,\"topKEnabledQwen36\":true,\"topKQwen36\":20,"
                          + "\"topPEnabledQwen36\":true,\"topPQwen36\":0.95"])
    func aDamagedQwenSamplingRecordIsRefused(_ fields: String) throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        try Data("""
        {"version":4,"contextTokens":8192,"expertCacheSlots":16,"temperature":0.2,\
        "topKEnabled":true,"topK":64,"topPEnabled":true,"topP":0.95,\
        "prefillEnabled":true,\(fields)}
        """.utf8).write(to: file)
        #expect(throws: (any Error).self) {
            _ = try MacAppSettingsFileStore.load(at: file)
        }
    }

    /// S25, D11: the thinking key is additive and optional. A file that never
    /// saw it answers with the shipped defaults — Qwen runs its frame open,
    /// Gemma has no frame and answers yes anyway, same as an unset key.
    @Test func theThinkingKeyDefaultsToTheFamilyAnswers() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        try Data("""
        {"version":4,"selectedModelPath":"/models/gemma4.gturbo",\
        "contextTokens":8192,"expertCacheSlots":16,"temperature":0.5,\
        "topKEnabled":false,"topK":40,"topPEnabled":true,"topP":0.9,\
        "prefillEnabled":true}
        """.utf8).write(to: file)

        let settings = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(settings.thinkingEnabledQwen36 == nil)
        #expect(settings.thinkingEnabled(for: .qwen36))
        #expect(settings.thinkingEnabled(for: .int4))
        #expect(settings.thinkingEnabled(for: .int8))
    }

    /// Turning Qwen's frame off stores its own key, survives a save and
    /// reload, and moves nothing in the records that predate variants.
    @Test func aQwenThinkingChangeRoundTripsUnderItsOwnKey() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        var settings = MacAppSettings()
        settings.setThinkingEnabled(false, for: .qwen36)
        try MacAppSettingsFileStore.save(settings, at: file)

        let read = try #require(try MacAppSettingsFileStore.load(at: file))
        #expect(read.thinkingEnabledQwen36 == false)
        #expect(!read.thinkingEnabled(for: .qwen36))
        #expect(read.thinkingEnabled(for: .int4))
        #expect(read.thinkingEnabled(for: .int8))
        let raw = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        #expect(raw["thinkingEnabledQwen36"] as? Bool == false)
    }

    /// A frame turned back on is the shipped state again, so its key goes
    /// away and a later change of the default keeps steering; Gemma has no
    /// key for a switch it can never show, so writes for it are dropped.
    @Test func thinkingAtTheDefaultStaysUnsavedAndGemmaHasNoKey() throws {
        var settings = MacAppSettings()
        settings.setThinkingEnabled(false, for: .qwen36)
        #expect(settings.thinkingEnabledQwen36 == false)
        settings.setThinkingEnabled(true, for: .qwen36)
        #expect(settings.thinkingEnabledQwen36 == nil)

        settings.setThinkingEnabled(false, for: .int4)
        settings.setThinkingEnabled(false, for: .int8)
        #expect(settings.thinkingEnabledQwen36 == nil)
        #expect(settings.thinkingEnabled(for: .int4))

        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(MacAppSettings.fileName)
        var saved = MacAppSettings()
        saved.setThinkingEnabled(false, for: .int8)
        try MacAppSettingsFileStore.save(saved, at: file)
        let raw = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        #expect(raw["thinkingEnabledQwen36"] == nil)
    }

    private func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("settings-sampling-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
