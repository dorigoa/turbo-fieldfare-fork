import Foundation
import Hub
import Tokenizers

public enum QwenTokenizerError: Error, CustomStringConvertible {
    case missingTokenizerConfig
    case unsupportedPipeline(actual: String)
    case missingSpecialToken(String)
    case invalidTokenID(token: String, id: Int)
    case invalidChatTemplate(String)

    public var description: String {
        switch self {
        case .missingTokenizerConfig:
            return "tokenizer_config.json is missing or unreadable"
        case .unsupportedPipeline(let actual):
            return "tokenizer pipeline is not the pinned Qwen 3.6 byte-level BPE sequence "
                + "(NFC normalizer, Sequence[Split(byte-level regex), ByteLevel], BPE model, "
                + "ByteLevel decoder and post-processor); found: \(actual)"
        case .missingSpecialToken(let t):
            return "tokenizer missing required special token: \(t)"
        case .invalidTokenID(let token, let id):
            return "tokenizer declares out-of-range ID \(id) for token \(token)"
        case .invalidChatTemplate(let reason):
            return "chat template continuation error: \(reason)"
        }
    }
}
/// Qwen 3.6 tokenizer wrapper (design decision D7).
///
/// Loads the tokenizer sidecar of a `.gturbo` model directory through the
/// same `LanguageModelConfigurationFromHub` path `GFTokenizer` uses, verifies
/// the pinned pipeline from `tokenizer.json` itself, and fails closed on
/// anything else.
///
/// The pinned pipeline (measured in S2): NFC normalizer,
/// `Sequence[Split(cl100k-style byte-level regex, Isolated, invert = false),
/// ByteLevel(add_prefix_space = false, trim_offsets = false,
/// use_regex = false)]` pre-tokenizer, BPE model over 248.044 byte-level
/// entries, ByteLevel decoder and ByteLevel post-processor. The 33
/// `added_tokens` are matched literally before the BPE (the nine markers the
/// runtime needs are NOT in `model.vocab`), so markers only ever arrive as
/// whole `added_tokens`. Qwen has no BOS and the post-processor injects
/// nothing, so `encode` adds no special tokens.
///
/// Decoding is a push-loop over ``QwenDetokenizer`` (GPT-2 byte-level), never
/// the library's `decode`, whose `clean_up_tokenization_spaces` pass rewrites
/// model output — the same reason `GFTokenizer.decode` bypasses it.
public struct QwenTokenizer: @unchecked Sendable {
    /// Model ID of the pinned source (manifest `modelID`, decision D9). Used
    /// for naming only; the tokenizer always loads from local files.
    public static let modelID = "mlx-community/Qwen3.6-35B-A3B-4bit"

    /// The `Split` pattern pinned by S2 (`tokenizer.json`'s and
    /// `tokenizer_config.json`'s `pretokenize_regex`, byte-identical).
    static let pinnedPretokenizeRegex =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// The nine marker names from S2, exactly as declared in `added_tokens`.
    /// The `"<"` is written as a Swift unicode escape so this source never
    /// embeds raw special-token control sequences.
    public static let endoftextToken = "\u{3C}|endoftext|>"
    public static let imStartToken = "\u{3C}|im_start|>"
    public static let imEndToken = "\u{3C}|im_end|>"
    public static let toolCallStartToken = "\u{3C}tool_call>"
    public static let toolCallEndToken = "\u{3C}/tool_call>"
    public static let toolResponseToken = "\u{3C}tool_response>"
    public static let toolResponseEndToken = "\u{3C}/tool_response>"
    public static let thinkToken = "\u{3C}think>"
    public static let thinkEndToken = "\u{3C}/think>"

    // MARK: - Special token IDs (resolved from `added_tokens` at load)

    public let endoftextID: Int32
    public let imStartID: Int32
    public let imEndID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    public let thinkID: Int32
    public let thinkEndID: Int32
    /// EOS is `<|im_end|>` and pad is `<|endoftext|>` per the pinned
    /// `tokenizer_config.json`; Qwen has no BOS.
    public let eosID: Int32
    public let padID: Int32
    /// Generation stops on `<|im_end|>` and `<|endoftext|>`.
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int

    /// The markers that structure assistant output (chat turns, think block,
    /// tool blocks). Streaming treats them as detokenizer barriers so a
    /// byte-level run never spans one — the Qwen counterpart of
    /// `GFTokenizer.structuralMarkerIDs`.
    public var structuralMarkerIDs: Set<Int32> {
        [imStartID, imEndID, toolCallStartID, toolCallEndID,
         toolResponseID, toolResponseEndID, thinkID, thinkEndID]
    }

    /// `added_tokens[special == true]` IDs — exactly what the library's
    /// `decode(skipSpecialTokens: true)` filters before its decoder chain.
    /// Note the tool/think markers are NOT special here: they decode as text
    /// in both modes, matching the pinned tokenizer and the S2 golden.
    let specialTokenIDs: Set<Int32>
    /// All 33 `added_tokens` IDs: decoded literally, they cut byte-level runs.
    let addedTokenIDs: Set<Int32>

    @usableFromInline
    let tokenizer: any Tokenizer

    /// The pinned `chat_template.jinja` compiler, loaded with the tokenizer
    /// files (S10/D8). Chat rendering and continuations go through it.
    let chatFormat: QwenChatFormat

    // MARK: - Loading

    /// Load the tokenizer from a folder holding `tokenizer.json` +
    /// `tokenizer_config.json` (the `.gturbo` `tokenizer/` sidecar or the
    /// plan's pinned local copy). There is deliberately no pretrained-load
    /// entry point: the tokenizer is part of the installed model directory
    /// and must not be fetched from the Hub at runtime (C6, D7).
    public static func load(from folder: URL) async throws -> QwenTokenizer {
        try await make(from: folder)
    }

    /// Qwen model directory -> its `tokenizer/` sidecar, then the
    /// `TURBO_FIELDFARE_TOKENIZER_DIR` override, mirroring
    /// `GFTokenizer.tokenizerFolder(forModelDirectory:)`.
    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }
        guard let override = environment["TURBO_FIELDFARE_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> QwenTokenizer {
        guard let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) else {
            throw QwenTokenizerError.missingTokenizerConfig
        }
        return try await load(from: folder)
    }

    static func loadUncached(from folder: URL) async throws -> QwenTokenizer {
        try await make(from: folder)
    }

    /// Build from the raw tokenizer configs so the pipeline and the
    /// special-token set can be validated against `tokenizer.json` itself.
    /// Same fetch/caching path `AutoTokenizer.from(pretrained:)` uses
    /// internally (and the one `GFTokenizer.make` uses). The chat template
    /// travels with the tokenizer files (C6), so a folder without
    /// `chat_template.jinja` fails at load like any other missing input.
    private static func make(from folder: URL) async throws -> QwenTokenizer {
        let folder = folder.standardizedFileURL
        let hub = LanguageModelConfigurationFromHub(modelFolder: folder)
        guard let tokenizerConfig = try await hub.tokenizerConfig else {
            throw QwenTokenizerError.missingTokenizerConfig
        }
        let tokenizerData = try await hub.tokenizerData
        let underlying = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        let chatFormat = try QwenChatFormat.load(from: folder)
        return try QwenTokenizer(
            tokenizer: underlying, tokenizerData: tokenizerData, chatFormat: chatFormat)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    // MARK: - Pipeline verification

    /// Reject a tokenizer whose declared pipeline is not the pinned Qwen 3.6
    /// byte-level BPE sequence measured in S2.
    ///
    /// Like `GFTokenizer.verifyDecoderConfiguration`, this reads the
    /// declaration structurally from `tokenizer.json`, so any foreign
    /// component fails at load instead of decoding subtly wrong text. The
    /// behavioral agreement with the library is pinned by the golden tests.
    static func verifyPipeline(_ tokenizerData: Config) throws {
        let normalizer = tokenizerData["normalizer"]
        let preTokenizer = tokenizerData["preTokenizer"]
        let steps = preTokenizer.pretokenizers.array(or: [])
        let model = tokenizerData["model"]
        let decoder = tokenizerData["decoder"]
        let postProcessor = tokenizerData["postProcessor"]

        let pipelineOK =
            normalizer.type.string() == "NFC"
            && preTokenizer.type.string() == "Sequence"
            && steps.count == 2
            && steps[0].type.string() == "Split"
            && steps[0].behavior.string() == "Isolated"
            && steps[0].invert.boolean(or: true) == false
            && steps[0].pattern.Regex.string() == pinnedPretokenizeRegex
            && steps[1].type.string() == "ByteLevel"
            && steps[1].addPrefixSpace.boolean(or: true) == false
            && steps[1].trimOffsets.boolean(or: true) == false
            && steps[1].useRegex.boolean(or: true) == false
            && model.type.string() == "BPE"
            && decoder.type.string() == "ByteLevel"
            && decoder.addPrefixSpace.boolean(or: true) == false
            && decoder.trimOffsets.boolean(or: true) == false
            && decoder.useRegex.boolean(or: true) == false
            && postProcessor.type.string() == "ByteLevel"
            && postProcessor.useRegex.boolean(or: true) == false
        guard pipelineOK else {
            let found = "normalizer=\(normalizer.type.string() ?? "?") "
                + "preTokenizer=\(preTokenizer.type.string() ?? "?")"
                + "[\(steps.map { $0.type.string() ?? "?" }.joined(separator: ","))] "
                + "model=\(model.type.string() ?? "?") "
                + "decoder=\(decoder.type.string() ?? "?") "
                + "postProcessor=\(postProcessor.type.string() ?? "?")"
            throw QwenTokenizerError.unsupportedPipeline(actual: found)
        }
    }

    // MARK: - Special-token resolution

    /// Config-supplied IDs are full-range `Int`; a trapping `Int32(_:)` would
    /// crash on a corrupt tokenizer instead of taking the typed error path.
    private static func int32ID(_ token: String, _ id: Int) throws -> Int32 {
        guard let value = Int32(exactly: id) else {
            throw QwenTokenizerError.invalidTokenID(token: token, id: id)
        }
        return value
    }

    /// Resolve the nine S2 marker names from `tokenizer.json`'s
    /// `added_tokens` — the authoritative source for them, since they are
    /// not in `model.vocab` and BPE would answer `<unk>`.
    private static func requireAddedTokenIDs(_ tokenizerData: Config) throws -> [String: Int32] {
        var resolved: [String: Int32] = [:]
        for added in tokenizerData["addedTokens"].array(or: []) {
            guard let content = added.content.string(),
                  let id = added["id"].integer() else { continue }
            resolved[content] = try int32ID(content, id)
        }
        for name in [Self.endoftextToken, Self.imStartToken, Self.imEndToken,
                     Self.toolCallStartToken, Self.toolCallEndToken,
                     Self.toolResponseToken, Self.toolResponseEndToken,
                     Self.thinkToken, Self.thinkEndToken]
        where resolved[name] == nil {
            throw QwenTokenizerError.missingSpecialToken(name)
        }
        return resolved
    }

    public init(tokenizer: any Tokenizer,
                tokenizerData: Config,
                chatFormat: QwenChatFormat) throws {
        self.tokenizer = tokenizer
        self.chatFormat = chatFormat
        try Self.verifyPipeline(tokenizerData)

        // `added_tokens[special == true]` reproduces the ID set the library's
        // `decode(skipSpecialTokens: true)` filters; the full `added_tokens`
        // ID set marks the literals that cut byte-level runs at decode.
        var specials: Set<Int32> = []
        var added: Set<Int32> = []
        for entry in tokenizerData["addedTokens"].array(or: []) {
            guard let id = entry["id"].integer() else { continue }
            let value = try Self.int32ID(entry.content.string() ?? "added token", id)
            added.insert(value)
            if entry["special"].boolean(or: false) { specials.insert(value) }
        }
        self.specialTokenIDs = specials
        self.addedTokenIDs = added

        // `requireAddedTokenIDs` guarantees each name is present.
        let ids = try Self.requireAddedTokenIDs(tokenizerData)
        self.endoftextID = ids[Self.endoftextToken]!
        self.imStartID = ids[Self.imStartToken]!
        self.imEndID = ids[Self.imEndToken]!
        self.toolCallStartID = ids[Self.toolCallStartToken]!
        self.toolCallEndID = ids[Self.toolCallEndToken]!
        self.toolResponseID = ids[Self.toolResponseToken]!
        self.toolResponseEndID = ids[Self.toolResponseEndToken]!
        self.thinkID = ids[Self.thinkToken]!
        self.thinkEndID = ids[Self.thinkEndToken]!
        self.eosID = self.imEndID
        self.padID = self.endoftextID
        self.stopTokenIDs = [self.imEndID, self.endoftextID]
        // Pinned by 3.1/D1: 248.044 BPE entries + 276 reserved slots.
        self.vocabSize = 248_320
    }

    // MARK: - Encode / decode

    /// Encode UTF-8 text to token IDs.
    ///
    /// Qwen has no BOS token (`bos_token = null` in `tokenizer_config.json`)
    /// and its ByteLevel post-processor injects nothing, so `addBOS` is
    /// ignored — the parameter exists only to keep one encode signature
    /// across model families (see D13, which adopts it for `ChatTokenizer`).
    /// Library-side, `addSpecialTokens: false` mirrors the pinned
    /// `add_bos_token = false` config: `added_tokens` are still matched
    /// literally inside the text, they are just never injected.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        _ = addBOS
        return tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips the
    /// `added_tokens[special == true]` IDs, matching the library's filter.
    ///
    /// A push-loop over ``QwenDetokenizer``, so batch decode and the
    /// generation-loop streaming decode agree by construction.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = false) -> String {
        var detokenizer = QwenDetokenizer(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
        var output = ""
        for id in ids {
            output += detokenizer.push(id)
        }
        output += detokenizer.flush()
        return output
    }

    // MARK: - Chat template (S10, D8)

    /// The Qwen counterpart of `GFTokenizer.applyChatTemplate`: render the
    /// conversation with the generation prompt attached and no tool schemas,
    /// through the pinned `chat_template.jinja`.
    ///
    /// `enableThinking` defaults to `true`, the checkpoint's own default;
    /// passing `false` selects the template's empty-think generation prompt.
    public func applyChatTemplate(_ messages: [GFTokenizer.Message],
                                  enableThinking: Bool = true) throws -> String {
        try chatFormat.render(
            messages: messages.map(Self.chatMessage),
            tools: nil,
            addGenerationPrompt: true,
            enableThinking: enableThinking)
    }

    /// The Qwen counterpart of `GFTokenizer.encodeToolChat`: render with tool
    /// schemas and the generation prompt, then encode without injected specials.
    public func encodeToolChat(messages: [GFTokenizer.Message],
                               tools: [GFTokenizer.FunctionDefinition],
                               enableThinking: Bool = true) throws -> [Int32] {
        let rendered = try chatFormat.render(
            messages: messages.map(Self.chatMessage),
            tools: tools.map(Self.toolSpec),
            addGenerationPrompt: true,
            enableThinking: enableThinking)
        return encode(rendered, addBOS: false)
    }

    /// Tokens to append after a completed assistant turn to start a new user
    /// turn — the Qwen counterpart of `GFTokenizer.encodeTextContinuation`.
    ///
    /// The cached assistant turn ended on the model's `im_end` stop token
    /// (which the runtime does not keep in the KV cache), so the continuation
    /// re-emits it, then opens the user turn the way the template renders one
    /// (content trimmed, like the template's own trim) and ends with the
    /// generation prompt for the requested thinking mode. A special-token
    /// boundary sits on both sides of every piece here, so splitting the
    /// encode this way yields exactly the IDs one encode of the joined text
    /// would produce.
    public func encodeTextContinuation(userContent: String,
                                       enableThinking: Bool = true) -> [Int32] {
        let content = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        return [imEndID] + encode(
            "\n\(Self.imStartToken)user\n\(content)\(Self.imEndToken)\n"
                + "\(Self.imStartToken)assistant\n" + Self.generationPrompt(enableThinking: enableThinking),
            addBOS: false)
    }

    /// Tokens to append after a cached assistant turn that ended in tool
    /// calls, once the tool results arrive — the Qwen counterpart of
    /// `GFTokenizer.encodeToolResultContinuation`, with the same algorithm.
    ///
    /// Render the conversation with and without the incoming tool turns; the
    /// continuation is the suffix of the second render that follows the last
    /// `tool_call`-delimited block of the assistant turn. Because the Qwen
    /// template closes the assistant turn before opening the tool-response
    /// user turn, that suffix starts at the cached turn's closing `im_end`
    /// (the stop token the runtime dropped), where Gemma's starts at its
    /// tool-response marker. Uniqueness fallback as in Gemma: when the two
    /// renders disagree before the call block, only a unique match of the
    /// exact block tokens is trusted.
    public func encodeToolResultContinuation(
        cachedMessages: [GFTokenizer.Message],
        assistant: GFTokenizer.Message,
        incomingMessages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        enableThinking: Bool = true
    ) throws -> [Int32] {
        let prefix = try encodeToolChat(
            messages: cachedMessages + [assistant],
            tools: tools,
            enableThinking: enableThinking)
        let full = try encodeToolChat(
            messages: incomingMessages,
            tools: tools,
            enableThinking: enableThinking)
        let callCount = assistant.toolCalls.count
        let starts = prefix.indices.filter { prefix[$0] == toolCallStartID }
        guard callCount > 0, starts.count >= callCount,
              let callEnd = prefix.lastIndex(of: toolCallEndID) else {
            throw QwenTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is missing")
        }
        let callStart = starts[starts.count - callCount]
        let callSequence = Array(prefix[callStart...callEnd])
        // Same reasoning as Gemma's: a verbatim-repeated call renders an
        // identical token sequence, so requiring a unique match dropped the
        // cache; the boundary is positional when both renders agree through
        // the call block.
        let suffixStart: Int
        if full.count > callEnd, full[...callEnd] == prefix[...callEnd] {
            suffixStart = callEnd + 1
        } else {
            let matches = full.subsequenceStartIndices(matching: callSequence)
            guard matches.count == 1 else {
                throw QwenTokenizerError.invalidChatTemplate(
                    "cached assistant tool-call boundary is ambiguous")
            }
            suffixStart = matches[0] + callSequence.count
        }
        let suffix = Array(full[suffixStart...])
        guard suffix.first == imEndID else {
            throw QwenTokenizerError.invalidChatTemplate(
                "tool-result continuation does not begin at the KV boundary")
        }
        return suffix
    }

    // MARK: - Chat mapping helpers

    /// Map a runtime message onto the template's representation. Roles pass
    /// through by name; `reasoning_content` is never attached from here, so
    /// older turns re-render without their reasoning (the pinned default,
    /// see the S2 report — continuations are what preserve it).
    private static func chatMessage(_ message: GFTokenizer.Message) -> QwenChatMessage {
        QwenChatMessage(
            role: message.role.rawValue,
            content: message.content,
            name: message.name,
            toolCallID: message.toolCallID,
            toolCalls: message.toolCalls.map {
                QwenToolCall(id: $0.id, name: $0.name, arguments: QwenChatValue($0.arguments))
            })
    }

    /// Tool schema in the shape the template consumes, with the key order the
    /// pinned goldens render (`type` first, then `function`).
    private static func toolSpec(_ tool: GFTokenizer.FunctionDefinition) -> QwenChatValue {
        .object([
            (key: "type", value: .string("function")),
            (key: "function", value: .object([
                (key: "name", value: .string(tool.name)),
                (key: "description", value: .string(tool.description)),
                (key: "parameters", value: QwenChatValue(tool.parameters)),
            ])),
        ])
    }

    /// The generation prompt tail the template emits after the assistant
    /// turn opening.
    private static func generationPrompt(enableThinking: Bool) -> String {
        enableThinking
            ? thinkToken + "\n"
            : thinkToken + "\n\n" + thinkEndToken + "\n\n"
    }
}
private extension Array where Element: Equatable {
    func subsequenceStartIndices(matching needle: [Element]) -> [Int] {
        guard !needle.isEmpty, needle.count <= count else { return [] }
        return indices.dropLast(needle.count - 1).filter { start in
            self[start..<(start + needle.count)].elementsEqual(needle)
        }
    }
}




