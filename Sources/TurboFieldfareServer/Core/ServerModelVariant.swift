import Foundation
import TurboFieldfare

/// The served model's variant, read from its manifest.
///
/// The variant is never a flag. A 4-bit and an 8-bit pack share every tensor
/// name and differ only in `manifest.quant`, the one record that cannot
/// disagree with the weights it describes. The server needs it before the
/// load, to size memory admission and to choose the API model ID, and after
/// it, to say what it is serving.
public struct ServerModelVariant: Equatable, Sendable {
    public let profile: QuantProfile
    /// `manifest.modelID`: the Hugging Face repository the pack was built from.
    public let manifestModelID: String

    public init(profile: QuantProfile, manifestModelID: String) {
        self.profile = profile
        self.manifestModelID = manifestModelID
    }

    /// The API model ID from before the 8-bit variant existed. A 4-bit model
    /// keeps it, so every client already configured for it keeps working.
    public static let legacyAPIModelID = "gemma-4-26b-a4b-it"

    /// "4-bit" or "8-bit", the names the installer and the Mac app use.
    public var label: String { "\(profile.denseWeightBits)-bit" }

    /// What admission charges this model beyond its KV cache.
    public var footprint: ModelFootprint { .forProfile(profile) }

    /// The `model` requests must name when `--model-id` is not given. An
    /// 8-bit model answers to its manifest's ID, so a client still configured
    /// for the 4-bit name is refused instead of being served another variant
    /// without notice.
    public var defaultAPIModelID: String {
        switch profile {
        case .affine4: Self.legacyAPIModelID
        case .affine8: manifestModelID
        }
    }

    /// Reads `manifest.json` alone, a few kilobytes, and never the weights.
    ///
    /// nil when the directory has no manifest: such a directory cannot load at
    /// all, and the load reports that with the precise reason, as it did before
    /// variants existed, rather than this read turning it into a usage error.
    /// A manifest that is present but unreadable throws.
    public static func read(modelDirectory: URL,
                            config: ArchConfig = .gemma4_26B_A4B) throws -> ServerModelVariant? {
        let manifestURL = modelDirectory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }
        let manifest = try ManifestReader.load(directoryURL: modelDirectory, expecting: config)
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required to size the model's memory")
        }
        return ServerModelVariant(profile: try QuantProfile(manifestQuant: quant),
                                  manifestModelID: manifest.modelID)
    }
}
