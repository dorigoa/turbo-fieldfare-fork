import Foundation

/// Model family declared by `manifest.json -> modelFamily` (C1). A manifest
/// without the key is Gemma 4, the only family v1 shipped with, so every
/// pre-Qwen reader and fixture stays valid.
///
/// Raw values are wire strings; an unlisted value must be rejected at read
/// time, never silently treated as Gemma (S3).
package enum GTurboModelFamily: String, Codable, Equatable, Sendable {
    case qwen35Moe = "qwen3_5_moe"
}
