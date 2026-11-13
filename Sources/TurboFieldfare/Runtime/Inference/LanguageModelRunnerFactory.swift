import Foundation
import Metal

/// The runner surface the CLI, the server, and the app call into across both
/// families (D3): `reset`, `continuationPosition`, and
/// `prepareForContinuation` come from `ContinuableLogitProducer`, plus the
/// context window and the rewind both runners already expose publicly. The
/// conformances below are empty for that reason — no member changes.
public protocol LanguageModelRunner: ContinuableLogitProducer {
    var maxContext: Int { get }
    /// Rewind the caches to `position`. For Qwen (D6) only the saved prefix
    /// snapshots are legal targets; anything else throws
    /// `QwenRewindError.rewindUnsupported` and the caller re-prefills.
    func rewind(to position: Int) throws
}

extension RealForwardRunner: LanguageModelRunner {}
extension QwenHybridForwardRunner: LanguageModelRunner {}

/// Chooses the runner for the family a loaded model belongs to (D3) and the
/// tokenizer for the family a caller asks for (D7, D13).
public enum LanguageModelRunnerFactory {
    /// `RealForwardRunner` for Gemma, which takes the runtime configuration;
    /// `QwenHybridForwardRunner` for Qwen, whose init has no such parameter,
    /// so for Qwen the runtime configuration does not reach the runner (S22–S24).
    public static func make(model: Model,
                            context: MetalContext,
                            maxContext: Int,
                            runtimeConfiguration: RuntimeConfiguration = .production) throws -> any LanguageModelRunner {
        switch Model.family(of: model.manifest) {
        case .gemma4:
            return try RealForwardRunner(model: model, context: context,
                                         maxContext: maxContext,
                                         runtimeConfiguration: runtimeConfiguration)
        case .qwen35Moe:
            return try QwenHybridForwardRunner(model: model, context: context,
                                               maxContext: maxContext)
        }
    }

    /// `GFTokenizer.load(forModelDirectory:)` or `QwenTokenizer.load(forModelDirectory:)`.
    public static func loadTokenizer(family: ModelFamily,
                                     forModelDirectory modelDirectory: URL) async throws -> any ChatTokenizer {
        switch family {
        case .gemma4:
            return try await GFTokenizer.load(forModelDirectory: modelDirectory)
        case .qwen35Moe:
            return try await QwenTokenizer.load(forModelDirectory: modelDirectory)
        }
    }
}
