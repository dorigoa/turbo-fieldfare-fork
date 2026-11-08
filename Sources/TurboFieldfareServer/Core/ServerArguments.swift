import Foundation
import TurboFieldfare

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    public let modelID: String
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let visionPack: String?
    public let visionResidency: VisionResidencyPolicy
    /// The variant the model's manifest recorded at parse time; nil when the
    /// directory had no manifest, which the load then reports.
    public let modelVariant: ServerModelVariant?

    public static let allowedMaxContext = [
        4_096, 8_192, 16_384, 32_768, 65_536, 98_304, 131_072, 196_608, 262_144,
    ]
    public static let unbackedContextOverrideVariable = "TURBO_FIELDFARE_ALLOW_UNBACKED_CONTEXT"

    public static let usage = """
    usage: TurboFieldfareServer --model <completed .gturbo directory> [options]

      --model <dir>              Required model directory.
      --vision-pack <dir>        Vision companion pack (default beside text model).
      --vision-residency <on-demand|keep-ready>
                                 Routed-expert residency during vision (default on-demand).
      --port <1...65535>         Loopback port (default 8080).
      --model-id <id>            API model identifier (default gemma-4-26b-a4b-it
                                 for a 4-bit model; an 8-bit model uses its
                                 manifest's model ID).
      --max-context <tokens>     4096, 8192, 16384, 32768, 65536, 98304,
                                 131072, 196608, or 262144 (default 16384).
                                 Contexts exceeding the host memory budget are refused;
                                 the budget follows the model's variant.
                                 TURBO_FIELDFARE_ALLOW_UNBACKED_CONTEXT=1 overrides this.
      --queue-limit <count>      Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                                 Prompt KV reuse mode (default single-prefix).
      --expert-cache-slots <n>   Expert-cache slots: \(RuntimeConfiguration.allowedValueList(RuntimeConfiguration.allowedExpertCacheSlots)) (default 16).
      --expert-cache-policy <s>  Expert-cache policy: lfu or lru (default lfu).
      --prefill on|off           Enable or disable chunked prompt prefill (default on).
                                 Chunked prefill requires 16 or more cache slots.
      --prefill-chunk-tokens <n|auto>
                                 Prefill chunk size: \(RuntimeConfiguration.allowedValueList(RuntimeConfiguration.allowedPrefillChunkTokens, alsoAccepting: ["auto"]))
                                 (default 128). Each chunk re-reads the routed
                                 expert pool, so larger chunks read less; auto
                                 runs at the cap, 256, which prefills every
                                 prompt in the same spans a per-request size
                                 would. Prefill scratch is sized from the chunk,
                                 so the cap holds about 32.5 MB of it against
                                 16.4 MB at 128.
      --rdadvise <s>             Read-advice policy: off, default, bounded, or adaptive
                                 (default off).
      --help                     Show this help.
    """

    // Mirrors the CLI's runtime flags so both binaries accept the same options
    // with the same validation, instead of the server pinning production
    // defaults. RuntimeConfiguration traps on unsupported values, so every
    // bound is checked here before the initializer runs.
    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool = true
    ) throws -> RuntimeConfiguration {
        guard RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw ServerArgumentError.notAllowed(
                flag: "--expert-cache-slots",
                allowed: RuntimeConfiguration.allowedExpertCacheSlots)
        }
        guard RuntimeConfiguration.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw ServerArgumentError.notAllowed(
                flag: "--prefill-chunk-tokens",
                allowed: RuntimeConfiguration.allowedPrefillChunkTokens,
                alsoAccepting: ["auto"])
        }
        guard prefillPolicy == .off
                || expertCacheSlots >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        else {
            throw ServerArgumentError.invalid(
                "--expert-cache-slots \(expertCacheSlots) requires --prefill off")
        }
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            rdadvisePolicy: rdadvisePolicy,
            prefillEnabled: prefillPolicy == .chunked,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead)
    }

    public static func parse(
        _ input: [String],
        hostMemoryBytes: UInt64 = ContextAdmission.hostMemoryBytes,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        readModelVariant: (URL) throws -> ServerModelVariant? = {
            try ServerModelVariant.read(modelDirectory: $0)
        }
    ) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var modelID: String?
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var visionPack: String?
        var visionResidency: VisionResidencyPolicy = .onDemand
        var expertCacheSlots = 16
        var expertCachePolicy = RuntimeExpertCachePolicy.lfu
        var prefillPolicy = RuntimePrefillPolicy.chunked
        var prefillChunkTokens = 128
        var rdadvisePolicy = RDAdvicePolicyMode.off
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelID = value
            case "--max-context":
                guard let parsed = Int(value),
                      allowedMaxContext.contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--vision-pack":
                visionPack = value
            case "--vision-residency":
                guard let parsed = VisionResidencyPolicy(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--vision-residency must be on-demand or keep-ready")
                }
                visionResidency = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ServerArgumentError.notAllowed(
                        flag: flag,
                        allowed: RuntimeConfiguration.allowedExpertCacheSlots)
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                guard let parsed = RuntimeExpertCachePolicy(rawValue: value) else {
                    throw ServerArgumentError.invalid("--expert-cache-policy must be lfu or lru")
                }
                expertCachePolicy = parsed
            case "--prefill":
                switch value {
                case "on": prefillPolicy = .chunked
                case "off": prefillPolicy = .off
                default: throw ServerArgumentError.invalid("--prefill must be on or off")
                }
            case "--prefill-chunk-tokens":
                // `auto` is an alias for the cap here, and nothing downstream
                // learns it was spelled that way. A per-request size is the
                // smallest allowed size that covers the span, so the cap
                // prefills every prompt in exactly the spans that size would,
                // and the KV ring is sized from the cap either way. The prefill
                // scratch is the one thing a per-request size changes: it is
                // allocated from the chunk, about 125.5 KB per token over a
                // fixed 328 KB, and the
                // server would reallocate it on every size change.
                if value == "auto" {
                    prefillChunkTokens = PrefillRuntimeConfig.maxChunkTokens
                    break
                }
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ServerArgumentError.notAllowed(
                        flag: flag,
                        allowed: RuntimeConfiguration.allowedPrefillChunkTokens,
                        alsoAccepting: ["auto"])
                }
                prefillChunkTokens = parsed
            case "--rdadvise":
                guard let parsed = RDAdvicePolicyMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--rdadvise must be off, default, bounded, or adaptive")
                }
                rdadvisePolicy = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        // Same resolution as the load, so both read the same manifest.
        let modelVariant: ServerModelVariant?
        do {
            modelVariant = try readModelVariant(URL(fileURLWithPath: model).standardizedFileURL)
        } catch {
            throw ServerArgumentError.invalid("--model manifest is not usable: \(error)")
        }
        // Refused here rather than at load: the KV allocation for a context the
        // host cannot back fails deep inside the runtime, after the model has
        // already started loading, with an allocator error that says nothing
        // about how much memory the choice actually needs. An 8-bit model
        // needs more than the 4-bit figures. Without a manifest the load fails
        // anyway, so the 4-bit figure every model was admitted with before
        // variants existed decides nothing that matters.
        let config = ArchConfig.gemma4_26B_A4B
        let footprint = modelVariant?.footprint ?? .affine4Measured
        if case .needsMemory = ContextAdmission.availability(config: config,
                                                             maxContext: maxContext,
                                                             hostMemoryBytes: hostMemoryBytes, expertCacheSlots: expertCacheSlots,
                                                             footprint: footprint),
           environment[unbackedContextOverrideVariable] != "1" {
            let need = ContextAdmission.needDescription(config: config,
                                                        maxContext: maxContext,
                                                        hostMemoryBytes: hostMemoryBytes, expertCacheSlots: expertCacheSlots,
                                                        footprint: footprint)
            throw ServerArgumentError.invalid(
                (modelVariant.map { "\($0.label) model: " } ?? "")
                    + need
                    + " Set \(unbackedContextOverrideVariable)=1 to start anyway.")
        }
        return ServerArguments(model: model,
                               port: port,
                               modelID: modelID
                                   ?? modelVariant?.defaultAPIModelID
                                   ?? ServerModelVariant.legacyAPIModelID,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               expertCacheSlots: expertCacheSlots,
                               expertCachePolicy: expertCachePolicy,
                               prefillPolicy: prefillPolicy,
                               prefillChunkTokens: prefillChunkTokens,
                               rdadvisePolicy: rdadvisePolicy,
                               visionPack: visionPack,
                               visionResidency: visionResidency,
                               modelVariant: modelVariant)
    }

    /// The line printed once the port is open. Fields are only ever appended,
    /// so a script that reads the earlier ones keeps working. `model` is the
    /// API model ID requests must name; `manifest_model_id` is the source the
    /// loaded pack was built from.
    public func readyLine(servedVariant: ServerModelVariant, visionCapability: String) -> String {
        "TurboFieldfareServer ready at http://127.0.0.1:\(port) model=\(modelID) "
            + "context=\(maxContext) prompt_cache=\(promptCacheMode.rawValue) "
            + "vision=\(visionCapability) vision_residency=\(visionResidency.rawValue) "
            + "variant=\(servedVariant.label) manifest_model_id=\(servedVariant.manifestModelID)"
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}

extension ServerArgumentError {
    static func notAllowed(flag: String,
                           allowed: [Int],
                           alsoAccepting aliases: [String] = []) -> ServerArgumentError {
        .invalid("\(flag) must be "
            + RuntimeConfiguration.allowedValueList(allowed, alsoAccepting: aliases))
    }
}
