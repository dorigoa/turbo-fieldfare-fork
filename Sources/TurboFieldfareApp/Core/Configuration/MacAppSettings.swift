import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

struct MacAppSettings: Codable, Equatable, Sendable {
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 4

    var version: Int = currentVersion
    var selectedModelPath: String?
    var migratedHistorySources: [String] = []
    var contextTokens: Int = AppContextLengthOption.eightK.tokens
    var expertCacheSlots: Int = 16
    var temperature: Double = 0.2
    var topKEnabled: Bool = true
    var topK: Int = 64
    var topPEnabled: Bool = true
    var topP: Double = 0.95
    var prefillEnabled: Bool = true
    var newlineShortcut: AppNewlineShortcut = .return
    var showPromptExamples: Bool = true
    var textSize: AppTextSize = .standard
    /// Whether the list of chats is showing. Remembered because hiding it is a
    /// choice about how the window looks, and a window that forgot it every
    /// launch would be making that choice again for the user each time.
    var sidebarVisible: Bool = true
    /// Whether the Inspector is showing. Remembered for the same reason the
    /// sidebar is: it is a choice about the window, not about one session.
    var inspectorVisible: Bool = true
    var visionResidencyPolicy: VisionResidencyPolicy = .onDemand
    var rdadvisePolicy: AppRDAdvicePolicy = .off
    var loadModelOnLaunch: Bool = false
    /// Which checkpoint the app installs and loads. Additive, like the fields
    /// above: a file without it was written by a build that only knew the
    /// 4-bit model, so absent means `int4` and costs no version bump.
    var modelVariant: SupportedModelVariant = .int4
    /// Where the 8-bit model lives. `selectedModelPath` stays the 4-bit
    /// location, so an older build reading this file still finds its model
    /// where it left it and never loads a checkpoint it cannot run.
    var selectedModelPathInt8: String?
    /// Where the Qwen3.6 model lives, remembered the same way.
    var selectedModelPathQwen36: String?
    /// Sampling saved for Qwen3.6, remembered the same way. Absent means the
    /// user has never moved it from that family's own defaults, so every
    /// launch follows `defaultSampling` again (U4, S24). The Gemma keys above
    /// are not optional because they predate variants: Gemma's own values have
    /// always been stored, changed or not.
    var temperatureQwen36: Double?
    var topKEnabledQwen36: Bool?
    var topKQwen36: Int?
    var topPEnabledQwen36: Bool?
    var topPQwen36: Double?
    /// Whether Qwen3.6 runs with its thinking frame open (D11, S25). Absent
    /// means the user never moved it from the family default, which is on —
    /// the same reading as the optional sampling keys above. Gemma has no
    /// thinking frame, so it has no such key.
    var thinkingEnabledQwen36: Bool?

    private enum CodingKeys: String, CodingKey {
        case version
        case selectedModelPath
        case migratedHistorySources
        case contextTokens
        case expertCacheSlots
        case temperature
        case topKEnabled
        case topK
        case topPEnabled
        case topP
        case prefillEnabled
        case newlineShortcut
        case showPromptExamples
        case textSize
        case sidebarVisible
        case inspectorVisible
        case visionResidencyPolicy
        case rdadvisePolicy
        case loadModelOnLaunch
        case modelVariant
        case selectedModelPathInt8
        case selectedModelPathQwen36
        case temperatureQwen36
        case topKEnabledQwen36
        case topKQwen36
        case topPEnabledQwen36
        case topPQwen36
        case thinkingEnabledQwen36
    }

    init(version: Int = currentVersion,
         selectedModelPath: String? = nil,
         migratedHistorySources: [String] = [],
         contextTokens: Int = AppContextLengthOption.eightK.tokens,
         expertCacheSlots: Int = 16,
         temperature: Double = 0.2,
         topKEnabled: Bool = true,
         topK: Int = 64,
         topPEnabled: Bool = true,
         topP: Double = 0.95,
         prefillEnabled: Bool = true,
         newlineShortcut: AppNewlineShortcut = .return,
         showPromptExamples: Bool = true,
         textSize: AppTextSize = .standard,
         sidebarVisible: Bool = true,
         inspectorVisible: Bool = true,
         visionResidencyPolicy: VisionResidencyPolicy = .onDemand,
         rdadvisePolicy: AppRDAdvicePolicy = .off,
         loadModelOnLaunch: Bool = false,
         modelVariant: SupportedModelVariant = .int4,
         selectedModelPathInt8: String? = nil,
         selectedModelPathQwen36: String? = nil,
         temperatureQwen36: Double? = nil,
         topKEnabledQwen36: Bool? = nil,
         topKQwen36: Int? = nil,
         topPEnabledQwen36: Bool? = nil,
         topPQwen36: Double? = nil,
         thinkingEnabledQwen36: Bool? = nil) {
        self.version = version
        self.selectedModelPath = selectedModelPath
        self.migratedHistorySources = migratedHistorySources
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.prefillEnabled = prefillEnabled
        self.newlineShortcut = newlineShortcut
        self.showPromptExamples = showPromptExamples
        self.textSize = textSize
        self.sidebarVisible = sidebarVisible
        self.inspectorVisible = inspectorVisible
        self.visionResidencyPolicy = visionResidencyPolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.loadModelOnLaunch = loadModelOnLaunch
        self.modelVariant = modelVariant
        self.selectedModelPathInt8 = selectedModelPathInt8
        self.selectedModelPathQwen36 = selectedModelPathQwen36
        self.temperatureQwen36 = temperatureQwen36
        self.topKEnabledQwen36 = topKEnabledQwen36
        self.topKQwen36 = topKQwen36
        self.topPEnabledQwen36 = topPEnabledQwen36
        self.topPQwen36 = topPQwen36
        self.thinkingEnabledQwen36 = thinkingEnabledQwen36
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        selectedModelPath = try container.decodeIfPresent(String.self, forKey: .selectedModelPath)
        migratedHistorySources = try container.decodeIfPresent([String].self, forKey: .migratedHistorySources) ?? []
        contextTokens = try container.decode(Int.self, forKey: .contextTokens)
        expertCacheSlots = try container.decode(Int.self, forKey: .expertCacheSlots)
        temperature = try container.decode(Double.self, forKey: .temperature)
        topKEnabled = try container.decode(Bool.self, forKey: .topKEnabled)
        topK = try container.decode(Int.self, forKey: .topK)
        topPEnabled = try container.decode(Bool.self, forKey: .topPEnabled)
        topP = try container.decode(Double.self, forKey: .topP)
        prefillEnabled = try container.decode(Bool.self, forKey: .prefillEnabled)
        newlineShortcut = try container.decodeIfPresent(
            AppNewlineShortcut.self,
            forKey: .newlineShortcut) ?? .return
        showPromptExamples = try container.decodeIfPresent(
            Bool.self,
            forKey: .showPromptExamples) ?? true
        // A damaged appearance preference must not reach settings-wide recovery,
        // which disables settings writes and requires explicit recovery.
        // Foundation can throw a non-DecodingError for a fractional integer.
        if container.contains(.textSize) {
            do {
                textSize = try container.decode(AppTextSize.self, forKey: .textSize)
            } catch {
                FileHandle.standardError.write(Data(
                    "Invalid Mac app setting textSize: unsupported value or type; using 100%.\n".utf8))
                textSize = .standard
            }
        }
        // Additive, like every field around it: absent means the default, so
        // no version bump and no rewrite of a file an older build can still
        // read.
        sidebarVisible = try container.decodeIfPresent(
            Bool.self,
            forKey: .sidebarVisible) ?? true
        inspectorVisible = try container.decodeIfPresent(
            Bool.self,
            forKey: .inspectorVisible) ?? true
        visionResidencyPolicy = try container.decodeIfPresent(
            VisionResidencyPolicy.self,
            forKey: .visionResidencyPolicy) ?? .onDemand
        rdadvisePolicy = try container.decodeIfPresent(
            AppRDAdvicePolicy.self,
            forKey: .rdadvisePolicy) ?? .off
        loadModelOnLaunch = try container.decodeIfPresent(
            Bool.self,
            forKey: .loadModelOnLaunch) ?? false
        // A variant this build does not know — written by a newer one — falls
        // back to the 4-bit model rather than to settings-wide recovery, the
        // same terms as `textSize`. `selectedModelPath` is the 4-bit location
        // whatever the file says, so the fallback still names a real install.
        if container.contains(.modelVariant) {
            do {
                modelVariant = try container.decode(SupportedModelVariant.self, forKey: .modelVariant)
            } catch {
                FileHandle.standardError.write(Data(
                    "Invalid Mac app setting modelVariant: unsupported value or type; using 4-bit.\n".utf8))
                modelVariant = .int4
            }
        }
        selectedModelPathInt8 = try container.decodeIfPresent(
            String.self,
            forKey: .selectedModelPathInt8)
        selectedModelPathQwen36 = try container.decodeIfPresent(
            String.self,
            forKey: .selectedModelPathQwen36)
        temperatureQwen36 = try container.decodeIfPresent(Double.self, forKey: .temperatureQwen36)
        topKEnabledQwen36 = try container.decodeIfPresent(Bool.self, forKey: .topKEnabledQwen36)
        topKQwen36 = try container.decodeIfPresent(Int.self, forKey: .topKQwen36)
        topPEnabledQwen36 = try container.decodeIfPresent(Bool.self, forKey: .topPEnabledQwen36)
        topPQwen36 = try container.decodeIfPresent(Double.self, forKey: .topPQwen36)
        thinkingEnabledQwen36 = try container.decodeIfPresent(Bool.self, forKey: .thinkingEnabledQwen36)
    }

    /// The sampling to run `variant` with: what its family saved, or the
    /// family's own defaults where the user has never changed anything (U4).
    func sampling(for variant: SupportedModelVariant) -> AppSamplingValues {
        switch variant {
        case .int4, .int8:
            return AppSamplingValues(temperature: temperature, topKEnabled: topKEnabled, topK: topK,
                                     topPEnabled: topPEnabled, topP: topP)
        case .qwen36:
            guard let temperature = temperatureQwen36, let topKEnabled = topKEnabledQwen36,
                  let topK = topKQwen36, let topPEnabled = topPEnabledQwen36,
                  let topP = topPQwen36 else { return variant.defaultSampling }
            return AppSamplingValues(temperature: temperature, topKEnabled: topKEnabled, topK: topK,
                                     topPEnabled: topPEnabled, topP: topP)
        }
    }

    /// Saves `values` as `variant`'s sampling. Where Qwen ends up exactly at
    /// its family defaults the saved overrides are dropped, so a state the
    /// user never moved stays able to follow a later change of the defaults
    /// (S24); the Gemma keys have no such notion — they were always stored.
    mutating func setSampling(_ values: AppSamplingValues, for variant: SupportedModelVariant) {
        switch variant {
        case .int4, .int8:
            temperature = values.temperature
            topKEnabled = values.topKEnabled
            topK = values.topK
            topPEnabled = values.topPEnabled
            topP = values.topP
        case .qwen36:
            guard values != variant.defaultSampling else {
                temperatureQwen36 = nil
                topKEnabledQwen36 = nil
                topKQwen36 = nil
                topPEnabledQwen36 = nil
                topPQwen36 = nil
                return
            }
            temperatureQwen36 = values.temperature
            topKEnabledQwen36 = values.topKEnabled
            topKQwen36 = values.topK
            topPEnabledQwen36 = values.topPEnabled
            topPQwen36 = values.topP
        }
    }

    /// Whether `variant` runs its generations with a thinking frame (D11,
    /// S25). Only families that have one answer: Qwen3.6 reads its saved key,
    /// where absent means the family default of on; Gemma has no frame, so it
    /// always says yes — the same answer as an unset key.
    func thinkingEnabled(for variant: SupportedModelVariant) -> Bool {
        switch variant {
        case .int4, .int8: true
        case .qwen36: thinkingEnabledQwen36 ?? true
        }
    }

    /// Stores `variant`'s thinking choice. Only Qwen3.6 has a frame to store
    /// one for; its default is remembered as absence, so a family left as it
    /// ships keeps following a later change of that default, exactly like the
    /// sampling override above.
    mutating func setThinkingEnabled(_ enabled: Bool, for variant: SupportedModelVariant) {
        switch variant {
        case .int4, .int8: break
        case .qwen36: thinkingEnabledQwen36 = enabled ? nil : false
        }
    }

    /// The saved location of `variant`'s install, if one was ever chosen.
    func selectedModelPath(for variant: SupportedModelVariant) -> String? {
        switch variant {
        case .int4: selectedModelPath
        case .int8: selectedModelPathInt8
        case .qwen36: selectedModelPathQwen36
        }
    }

    mutating func setSelectedModelPath(_ path: String?, for variant: SupportedModelVariant) {
        switch variant {
        case .int4: selectedModelPath = path
        case .int8: selectedModelPathInt8 = path
        case .qwen36: selectedModelPathQwen36 = path
        }
    }

    /// Schema routing happens before decoding; value validation never replaces a damaged file.
    func isValid() -> Bool {
        Self.isValidModelPath(selectedModelPath)
            && Self.isValidModelPath(selectedModelPathInt8)
            && Self.isValidModelPath(selectedModelPathQwen36)
            && AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
            && Self.isValidQwenSampling(temperatureQwen36, topKEnabledQwen36, topKQwen36,
                                        topPEnabledQwen36, topPQwen36)
    }

    /// An override is written as a complete set by `setSampling`; a file with
    /// only part of it names values no slider could have produced, which is
    /// the same case as any other invalid value.
    private static func isValidQwenSampling(_ temperature: Double?, _ topKEnabled: Bool?,
                                            _ topK: Int?, _ topPEnabled: Bool?,
                                            _ topP: Double?) -> Bool {
        let present = [temperature != nil, topKEnabled != nil, topK != nil,
                       topPEnabled != nil, topP != nil]
        guard (present.allSatisfy { $0 } || present.allSatisfy { !$0 }) else { return false }
        guard let temperature, let topK, let topP else { return true }
        return temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
    }

    private static func isValidModelPath(_ path: String?) -> Bool {
        path.map { $0.hasPrefix("/") && !$0.contains("\0") } ?? true
    }
}

/// Just enough of the file to route on, so a newer build's schema cannot throw
/// before its version has been read.
private struct VersionStamp: Decodable {
    let version: Int
}

enum MacAppSettingsFileStore {
    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(MacAppSettings.fileName, isDirectory: false)
    }

    static func load(at fileURL: URL) throws -> MacAppSettings? {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        let data = try Data(contentsOf: fileURL)
        let stamp = try JSONDecoder().decode(VersionStamp.self, from: data)
        guard stamp.version <= MacAppSettings.currentVersion else {
            throw SettingsError.unsupportedVersion(stamp.version)
        }
        var settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
        guard settings.isValid() else { throw SettingsError.invalidValues }
        settings.version = MacAppSettings.currentVersion
        for variant in SupportedModelVariant.allCases {
            if let path = settings.selectedModelPath(for: variant) {
                settings.setSelectedModelPath(URL(fileURLWithPath: path).standardizedFileURL.path,
                                              for: variant)
            }
        }
        return settings
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             fileManager: FileManager = .default) -> MacAppSettings {
        do {
            if let settings = try load(at: fileURL(forModelDirectory: modelDirectory)) {
                let original = try JSONDecoder().decode(VersionStamp.self, from: Data(contentsOf: fileURL(forModelDirectory: modelDirectory)))
                if original.version < MacAppSettings.currentVersion {
                    try save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
                }
                return settings
            }
            let settings = MacAppSettings()
            try save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
            return settings
        } catch {
            FileHandle.standardError.write(Data("Reading Mac app settings failed: \(error)\n".utf8))
            return MacAppSettings()
        }
    }

    enum SettingsError: Error, CustomStringConvertible {
        case unsupportedVersion(Int)
        case invalidValues
        var description: String {
            switch self {
            case .unsupportedVersion(let version): "Settings version \(version) requires a newer app. The original file was preserved."
            case .invalidValues: "Settings contain invalid values. The original file was preserved."
            }
        }
    }

    static func save(_ settings: MacAppSettings,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw SettingsError.invalidValues }
        try save(settings, at: fileURL(forModelDirectory: modelDirectory), fileManager: fileManager)
    }

    static func save(_ settings: MacAppSettings, at fileURL: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw SettingsError.invalidValues }
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try Posix.atomicWrite(data, to: fileURL.path,
                              durableIn: fileURL.deletingLastPathComponent().path,
                              uniqueTemporary: true)
    }

}
