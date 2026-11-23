import Foundation
import TurboFieldfare

/// The sampling a request generates with: what `/props` reports as the
/// defaults and `/slots` as the running request's settings.
///
/// The defaults are the ones `OpenAIRequestValidator` applies when a request
/// names none, read from here so the two cannot drift apart.
public struct ServerSamplingSettings: Equatable, Sendable {
    /// D10/U4: the sampling defaults are the family's reference generation.
    /// Gemma's long-standing 0.2/64/0.95 is untouched; Qwen answers with the
    /// checkpoint's 1.0/20/0.95, the same figures the CLI uses (S22).
    static func defaultTemperature(for family: ModelFamily) -> Float {
        family == .qwen35Moe ? 1.0 : 0.2
    }

    static func defaultTopK(for family: ModelFamily) -> Int {
        family == .qwen35Moe ? 20 : 64
    }

    static let defaultTopP: Float = 0.95
    static let defaultRepetitionPenalty: Float = 1
    static let defaultMaxTokens = 4_096

    let temperature: Float
    let topK: Int
    let topP: Float
    let repetitionPenalty: Float
    /// nil draws a fresh seed per request.
    let seed: UInt64?
    let maxTokens: Int
    let stop: [String]
    let stream: Bool

    static func defaults(for family: ModelFamily) -> ServerSamplingSettings {
        ServerSamplingSettings(temperature: defaultTemperature(for: family),
                               topK: defaultTopK(for: family),
                               topP: defaultTopP,
                               repetitionPenalty: defaultRepetitionPenalty,
                               seed: nil,
                               maxTokens: defaultMaxTokens,
                               stop: [],
                               stream: false)
    }

    init(temperature: Float, topK: Int, topP: Float, repetitionPenalty: Float,
         seed: UInt64?, maxTokens: Int, stop: [String], stream: Bool) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.maxTokens = maxTokens
        self.stop = stop
        self.stream = stream
    }

    /// The settings a validated request carries. The validator always sets
    /// Top-K and Top-P; the fallbacks only keep this total.
    init(request: ValidatedChatRequest) {
        let config = request.generationConfig
        self.init(temperature: config.temperature,
                  topK: config.topK ?? Self.defaultTopK(for: .gemma4),
                  topP: config.topP ?? 1,
                  repetitionPenalty: config.repetitionPenalty,
                  seed: config.seed,
                  maxTokens: request.maximumCompletionTokens,
                  stop: config.stopStrings,
                  stream: request.stream)
    }
}

/// What the loaded model is, as `GET /props` names it: the paths main
/// resolved and the chat template installed beside the tokenizer.
public struct ServerModelDescription: Equatable, Sendable {
    public let path: String
    /// The model's original Jinja template; empty when none is installed.
    public let chatTemplate: String

    public init(path: String = "", chatTemplate: String = "") {
        self.path = path
        self.chatTemplate = chatTemplate
    }

    /// Reads `chat_template.jinja` from the tokenizer folder the model
    /// session loaded, where both families install it. A missing or
    /// unreadable template is reported as empty rather than failing startup:
    /// the session has already refused a model that needs one.
    public static func load(modelDirectory: URL, family: ModelFamily) -> ServerModelDescription {
        let folder = family == .qwen35Moe
            ? QwenTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
            : GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        let template = folder
            .flatMap { folder in
                try? String(contentsOf: folder.appendingPathComponent("chat_template.jinja"),
                            encoding: .utf8)
            }
        return ServerModelDescription(path: modelDirectory.standardizedFileURL.path,
                                      chatTemplate: template ?? "")
    }
}

/// `GET /props` and `GET /slots` in llama-server's shape, for clients that
/// probe a local server for its context size and state before using it.
///
/// llama.cpp documents these in `tools/server/README.md`; there is no other
/// standard. The fields follow that README and the llama.cpp web UI's
/// `ApiLlamaCppServerProps` and `ApiSlotData` types, so `next_token` is an
/// object as they describe it. Only what this server knows is reported:
/// - one slot, because generation is serialized; the queue behind it is in
///   `/metrics`;
/// - the samplers this runtime has, in the order it applies them, and the
///   ones it lacks at the values that switch them off in llama.cpp;
/// - no prompt or generated text, and no field this server cannot fill
///   truthfully (`bos_token`, `eos_token`, `media_marker`).
struct LlamaServerProperties: Sendable {
    /// llama-server's default seed, which it reports when each request draws
    /// its own.
    static let randomSeed: UInt64 = 4_294_967_295
    static let buildInfo = "turbofieldfare"
    /// llama.cpp's names for the stages of `Sampler`: the repetition penalty,
    /// then Top-P over the full distribution, Top-K over what survives, and
    /// temperature on the final draw.
    static let samplers = ["penalties", "top_p", "top_k", "temperature"]

    let modelID: String
    let family: ModelFamily
    let contextTokens: Int
    let visionReady: Bool
    let model: ServerModelDescription

    /// The `GET /props` object. `default_generation_settings` is the idle
    /// slot with the defaults a request gets when it names none, as
    /// llama-server reports it; the live state is in `/slots`.
    func props() -> [String: Any] {
        var settings = slot(processing: false, running: nil)
        settings["prompt"] = ""
        if var nextToken = settings["next_token"] as? [String: Any] {
            nextToken["stopping_word"] = ""
            settings["next_token"] = nextToken
        }
        return [
            "default_generation_settings": settings,
            "total_slots": 1,
            "model_alias": modelID,
            "model_path": model.path,
            "modalities": [
                "vision": visionReady,
                "audio": false,
                "video": false,
            ],
            "endpoint_slots": true,
            "endpoint_props": false,
            "endpoint_metrics": true,
            "chat_template": model.chatTemplate,
            "chat_template_caps": [String: Any](),
            "build_info": Self.buildInfo,
            "is_sleeping": false,
        ]
    }

    /// The `GET /slots` array: the single slot.
    func slots(processing: Bool, running: ServerMetrics.Running?) -> [[String: Any]] {
        [slot(processing: processing, running: running)]
    }

    /// The slot. A running request shows its own settings and progress;
    /// otherwise the defaults, with no task (`id_task` -1). `processing` is
    /// the coordinator's: true from the moment a request holds the slot, a
    /// little before its first progress report.
    func slot(processing: Bool, running: ServerMetrics.Running?) -> [String: Any] {
        let settings = running?.settings ?? .defaults(for: family)
        var slot: [String: Any] = [
            "id": 0,
            "id_task": running.map { Int($0.id) } ?? -1,
            "n_ctx": contextTokens,
            "speculative": false,
            "is_processing": processing,
            "params": Self.params(settings),
            "next_token": [
                "has_next_token": running != nil,
                // Not tracked: generated text is never inspected here.
                "has_new_line": false,
                "n_remain": running.map { max(settings.maxTokens - $0.generated, 0) } ?? -1,
                "n_decoded": running?.generated ?? 0,
            ] as [String: Any],
        ]
        if let total = running?.promptTotal {
            slot["n_prompt_tokens"] = total
        }
        return slot
    }

    static func params(_ settings: ServerSamplingSettings) -> [String: Any] {
        [
            "n_predict": settings.maxTokens,
            "max_tokens": settings.maxTokens,
            "seed": settings.seed ?? randomSeed,
            "temperature": number(settings.temperature),
            "top_k": settings.topK,
            "top_p": number(settings.topP),
            "repeat_penalty": number(settings.repetitionPenalty),
            "stop": settings.stop,
            "stream": settings.stream,
            "samplers": samplers,
            // Samplers this runtime does not have, at llama.cpp's "off".
            "min_p": 0.0,
            "typical_p": 1.0,
            "top_n_sigma": -1.0,
            "xtc_probability": 0.0,
            "dynatemp_range": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "mirostat": 0,
        ]
    }

    /// The value as written, `0.95` rather than the `0.949999988079071` a
    /// Float widens to.
    private static func number(_ value: Float) -> Double {
        Double(value.description) ?? Double(value)
    }
}
