import Foundation
import TurboFieldfare

/// Whether the service may attempt a load at the requested context.
///
/// Pure and separate from the command loop so the decision is testable without
/// a socket, a model or a machine of the size in question. The service is the
/// process that would be killed: a 262,144-token KV is charged in full the
/// moment a command buffer binds it, so an unbacked context is not a load that
/// runs slowly — it is a jetsam kill with no event on the wire, which the app
/// can only report as a lost connection.
enum DecodeLoadAdmission {
    /// Diagnostics only. Set it to run a context this rule refuses and find out
    /// what actually happens on the machine, which is the measurement the rule
    /// itself came from.
    static let overrideEnvironmentKey = "TURBO_FIELDFARE_ALLOW_UNBACKED_CONTEXT"

    /// The message to answer the load with, or nil to proceed.
    ///
    /// `footprint` is the model's own; the default is the 4-bit pack's, which
    /// is what every caller meant before the 8-bit variant existed.
    static func refusal(maxContextTokens: Int,
                        hostMemoryBytes: UInt64,
                        environment: [String: String], expertCacheSlots: Int = 16,
                        footprint: ModelFootprint = .affine4Measured) -> String? {
        decide(maxContextTokens: maxContextTokens,
               hostMemoryBytes: hostMemoryBytes,
               environment: environment,
               expertCacheSlots: expertCacheSlots,
               footprint: { footprint })
    }

    /// The same rule for the model about to be loaded, sized from its
    /// manifest: an 8-bit pack maps about twice the resident weights and fills
    /// slots twice the size, so the 4-bit figures would admit a context the
    /// host cannot back.
    ///
    /// Reads only `manifest.json`, and only once the ceiling and the override
    /// have been decided. A manifest that cannot be read is a refusal, not a
    /// fallback to the 4-bit footprint: guessing low is exactly the kill this
    /// rule exists to prevent.
    static func refusal(modelDirectory: URL,
                        maxContextTokens: Int,
                        hostMemoryBytes: UInt64,
                        environment: [String: String], expertCacheSlots: Int = 16) -> String? {
        decide(maxContextTokens: maxContextTokens,
               hostMemoryBytes: hostMemoryBytes,
               environment: environment,
               expertCacheSlots: expertCacheSlots,
               footprint: { try ModelFootprint.forModelDirectory(modelDirectory) })
    }

    private static func decide(maxContextTokens: Int,
                               hostMemoryBytes: UInt64,
                               environment: [String: String], expertCacheSlots: Int,
                               footprint readFootprint: () throws -> ModelFootprint) -> String? {
        let config = ArchConfig.gemma4_26B_A4B
        // The ceiling is the checkpoint's, not the host's, so no override
        // reaches it: memory cannot make a position the model was never
        // trained to address valid.
        guard maxContextTokens <= config.maxPositionEmbeddings else {
            return ModelError.contextExceedsModel(
                requested: maxContextTokens,
                maximum: config.maxPositionEmbeddings).description
        }
        guard environment[overrideEnvironmentKey] != "1" else { return nil }
        let footprint: ModelFootprint
        do {
            footprint = try readFootprint()
        } catch {
            return "The model's memory needs could not be read from its manifest: \(error)"
        }
        guard case .needsMemory = ContextAdmission.availability(
            config: config,
            maxContext: maxContextTokens,
            hostMemoryBytes: hostMemoryBytes, expertCacheSlots: expertCacheSlots,
            footprint: footprint) else { return nil }
        return ContextAdmission.needDescription(config: config,
                                                maxContext: maxContextTokens,
                                                hostMemoryBytes: hostMemoryBytes, expertCacheSlots: expertCacheSlots,
                                                footprint: footprint)
    }
}
