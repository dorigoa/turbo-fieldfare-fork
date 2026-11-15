import Foundation

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
public enum SourceFingerprint {
    public static let knownFingerprints: [String: String] = Dictionary(
        uniqueKeysWithValues: SupportedModelVariant.allCases.map {
            ($0.repoID, $0.sourceIndexSHA256)
        })

    /// The checkpoints an image companion may be cut from: one per Gemma
    /// variant, which includes a checkpoint that is no longer a text variant.
    public static let knownVisionFingerprints: [String: String] = Dictionary(
        uniqueKeysWithValues: VisionCompanionSource.all.map {
            ($0.repoID, $0.sourceIndexSHA256)
        })

    /// Returns the recognised model ID for a given index.json SHA-256, or nil.
    public static func modelID(forIndexSha256 sha256Hex: String) -> String? {
        modelID(forIndexSha256: sha256Hex, in: knownFingerprints)
    }

    /// The image source an index.json SHA-256 belongs to, or nil.
    public static func visionModelID(forIndexSha256 sha256Hex: String) -> String? {
        modelID(forIndexSha256: sha256Hex, in: knownVisionFingerprints)
    }

    static func modelID(forIndexSha256 sha256Hex: String,
                        in fingerprints: [String: String]) -> String? {
        for (id, sha) in fingerprints where sha == sha256Hex { return id }
        return nil
    }
}
