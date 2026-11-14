import Foundation

/// Shared locators for Qwen tests (contract C9).
///
/// Qwen cases that need the installed model, the tokenizer sidecar files, or
/// the golden fixtures ask this helper and gate themselves with
/// `@Test(.enabled(if: …))` / a suite-level `.enabled(if:)`, so a checkout
/// without the local `scratch/` artifacts still runs green.
///
/// Everything is derived from `#filePath` (contract C7's fixture convention),
/// so the locators do not depend on the working directory SwiftPM happens to
/// use when running the tests.
enum QwenTestResources {
    /// Repository root, derived from this file's location:
    /// `Tests/TurboFieldfare/Core/Runtime/Qwen/QwenTestResources.swift`.
    static let repoRoot: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 { url.deleteLastPathComponent() }
        return url
    }()

    /// `Tests/QwenFixtures` — the golden fixtures committed to git (C7).
    /// Always present; does not need an installed model.
    static var fixtures: URL {
        repoRoot.appendingPathComponent("Tests/QwenFixtures", isDirectory: true)
    }

    /// `scratch/qwen36.gturbo` when its manifest exists, otherwise `nil`.
    static var modelURL: URL? {
        let model = repoRoot.appendingPathComponent("scratch/qwen36.gturbo", isDirectory: true)
        return FileManager.default.fileExists(
            atPath: model.appendingPathComponent("manifest.json").path) ? model : nil
    }

    /// The tokenizer folder: the `.gturbo` sidecar `tokenizer/` first, then
    /// the local `scratch/qwen36-tokenizer` copy kept by `tools/fetch_tokenizer.py`
    /// (C9). `nil` when neither folder holds a `tokenizer.json`.
    static var tokenizerFolder: URL? {
        let candidates = [
            modelURL?.appendingPathComponent("tokenizer", isDirectory: true),
            repoRoot.appendingPathComponent("scratch/qwen36-tokenizer", isDirectory: true),
        ]
        for case let folder? in candidates
        where FileManager.default.fileExists(
            atPath: folder.appendingPathComponent("tokenizer.json").path) {
            return folder
        }
        return nil
    }

    /// Convenience for `@Test(.enabled(if: …))`.
    static var hasModel: Bool { modelURL != nil }
    static var hasTokenizer: Bool { tokenizerFolder != nil }
}
