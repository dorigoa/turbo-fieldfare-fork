import Foundation
import Synchronization
import TurboFieldfare

/// The Anthropic Messages API (`POST /v1/messages`), which Claude Code and the
/// Anthropic SDKs speak, served by the same pipeline as Chat Completions.
///
/// A Messages request is translated into the Chat Completions request this
/// server already decodes and validates, so both APIs share one validator, one
/// queue, one prompt cache, and one set of logs and metrics; only the wire
/// format of the answer differs. The translation is deliberately lenient where
/// the OpenAI path is strict: Anthropic clients send fields this server has no
/// use for (`metadata`, `cache_control`, beta fields), and they are ignored
/// rather than refused. Anything the server cannot represent and that would
/// change the answer is either refused with an error or replaced by text that
/// tells the model what was left out; nothing is dropped silently.
enum AnthropicMessages {
    static let messagesPath = "/v1/messages"
    static let countTokensPath = "/v1/messages/count_tokens"
    static let toolResultImagePlaceholder =
        "[image omitted: this server cannot pass images inside tool results to the model]"
    static let documentPlaceholder =
        "[document omitted: this server does not support document content blocks]"

    // MARK: Request translation

    /// The Chat Completions JSON for a Messages request body. `counting` is
    /// the `count_tokens` form, which carries no `max_tokens`.
    static func chatRequestJSON(from body: Data,
                                family: ModelFamily,
                                counting: Bool = false) throws -> Data {
        guard body.contains(where: { ![0x20, 0x09, 0x0A, 0x0D].contains($0) }) else {
            throw ServerRequestError.invalid(message: "request body is empty",
                                             param: nil, code: "invalid_json")
        }
        let root: JSONValue
        do {
            root = try JSONDecoder().decode(JSONValue.self, from: body)
        } catch let error as DecodingError {
            throw ServerRequestError.decoding(error, body: body)
        }
        guard case .object(let request) = root else {
            throw invalid("request body must be a JSON object", nil)
        }

        var chat: [String: JSONValue] = [:]
        if let model = request["model"], model != .null {
            guard case .string = model else { throw invalid(#""model" must be a string"#, "model") }
            chat["model"] = model
        }

        if counting {
            chat["max_completion_tokens"] = .integer(1)
        } else {
            guard let maxTokens = request["max_tokens"], maxTokens != .null else {
                throw invalid(#"missing required field "max_tokens""#, "max_tokens")
            }
            guard let value = integer(maxTokens), value > 0 else {
                throw invalid(#""max_tokens" must be a positive integer"#, "max_tokens")
            }
            chat["max_completion_tokens"] = .integer(Int64(value))
        }

        for (field, target) in [("temperature", "temperature"), ("top_p", "top_p")] {
            if let value = request[field], value != .null {
                guard isNumber(value) else { throw invalid("\"\(field)\" must be a number", field) }
                chat[target] = value
            }
        }
        if let topK = request["top_k"], topK != .null {
            guard let value = integer(topK) else {
                throw invalid(#""top_k" must be an integer"#, "top_k")
            }
            chat["top_k"] = .integer(Int64(value))
        }
        if let stops = request["stop_sequences"], stops != .null {
            guard case .array(let values) = stops,
                  values.allSatisfy({ if case .string = $0 { true } else { false } }) else {
                throw invalid(#""stop_sequences" must be an array of strings"#, "stop_sequences")
            }
            if !values.isEmpty { chat["stop"] = .array(values) }
        }
        if !counting, let stream = request["stream"], stream != .null {
            guard case .bool = stream else { throw invalid(#""stream" must be true or false"#, "stream") }
            chat["stream"] = stream
        }

        var systemPrompt = ""
        if let system = request["system"], system != .null {
            systemPrompt = try systemText(system)
        }
        guard case .array(let input)? = request["messages"] else {
            throw invalid(#""messages" must be an array"#, "messages")
        }
        var tools = try ToolSet(request["tools"])
        var conversation: [JSONValue] = []
        for (index, message) in input.enumerated() {
            let path = "messages[\(index)]"
            if case .string("system")? = message.objectValue?["role"] {
                let text = try systemTurn(message, path: path, tools: &tools)
                guard !text.isEmpty else { continue }
                if conversation.isEmpty {
                    // Ahead of the conversation it is simply more system prompt.
                    systemPrompt += (systemPrompt.isEmpty ? "" : "\n\n") + text
                } else {
                    conversation.append(.object([
                        "role": .string("user"),
                        "content": .string("<system-reminder>\n\(text)\n</system-reminder>"),
                    ]))
                }
                continue
            }
            tools.loadReferences(in: message)
            conversation += try translateMessage(message, path: path)
        }
        var messages: [JSONValue] = systemPrompt.isEmpty ? [] : [
            .object(["role": .string("system"), "content": .string(systemPrompt)]),
        ]
        messages += mergingAdjacentUserText(conversation)
        chat["messages"] = .array(messages)

        let translatedTools = try tools.active.map { entry in
            try translateTool(entry.tool, path: entry.path)
        }
        if !translatedTools.isEmpty { chat["tools"] = .array(translatedTools) }
        if let choice = request["tool_choice"], choice != .null {
            // `any` and a named tool would force a call, which this server
            // cannot do; they run as `auto`, where the model still sees every
            // tool and decides.
            guard case .string(let type)? = choice.objectValue?["type"] else {
                throw invalid(#""tool_choice" must be an object with a "type""#, "tool_choice")
            }
            switch type {
            case "auto", "any", "tool": chat["tool_choice"] = .string("auto")
            case "none": chat["tool_choice"] = .string("none")
            default: throw invalid("tool_choice type \"\(type)\" is not supported", "tool_choice")
            }
        }

        // Only Qwen3.6 has a thinking phase; Gemma runs with it off, and its
        // chat request refuses both fields. The budget maps onto the three
        // depths the server enforces, and anything larger reasons unbounded.
        if family == .qwen35Moe, let thinking = request["thinking"], thinking != .null {
            guard case .string(let type)? = thinking.objectValue?["type"] else {
                throw invalid(#""thinking" must be an object with a "type""#, "thinking")
            }
            switch type {
            case "disabled":
                chat["chat_template_kwargs"] = .object(["enable_thinking": .bool(false)])
            case "enabled", "adaptive":
                chat["chat_template_kwargs"] = .object(["enable_thinking": .bool(true)])
                if let budget = thinking.objectValue?["budget_tokens"].flatMap(integer),
                   let depth = ReasoningDepth.allCases.first(where: { budget <= $0.budgetTokens }) {
                    chat["reasoning_effort"] = .string(depth.rawValue)
                }
            default:
                throw invalid("thinking type \"\(type)\" is not supported", "thinking")
            }
        }

        return try JSONEncoder().encode(JSONValue.object(chat))
    }

    private static func systemText(_ system: JSONValue) throws -> String {
        switch system {
        case .string(let text):
            return text
        case .array(let blocks):
            return try blocks.enumerated().compactMap { index, block -> String? in
                guard case .string("text")? = block.objectValue?["type"],
                      case .string(let text)? = block.objectValue?["text"] else {
                    throw invalid("system[\(index)] must be a text block", "system")
                }
                return text.isEmpty ? nil : text
            }.joined(separator: "\n\n")
        default:
            throw invalid(#""system" must be a string or an array of text blocks"#, "system")
        }
    }

    /// One Messages turn as Chat Completions messages. A user turn's tool
    /// results become `tool` messages, ahead of whatever else it says, which
    /// is the order the conversation happened in.
    private static func translateMessage(_ message: JSONValue, path: String) throws -> [JSONValue] {
        guard case .object(let fields) = message else {
            throw invalid("\(path) must be an object", path)
        }
        guard case .string(let role)? = fields["role"], role == "user" || role == "assistant" else {
            let seen: String = {
                if case .string(let value)? = fields["role"] { return "\"\(value.prefix(32))\"" }
                return "missing"
            }()
            throw invalid("\"\(path).role\" is \(seen); it must be user, assistant, or system",
                          "\(path).role")
        }
        let blocks: [JSONValue]
        switch fields["content"] {
        case .string(let text)?:
            blocks = [.object(["type": .string("text"), "text": .string(text)])]
        case .array(let list)?:
            blocks = list
        default:
            throw invalid("\"\(path).content\" must be a string or an array of content blocks",
                          "\(path).content")
        }

        var texts: [String] = []
        var images: [JSONValue] = []
        var toolCalls: [JSONValue] = []
        var toolResults: [JSONValue] = []
        for (index, block) in blocks.enumerated() {
            let blockPath = "\(path).content[\(index)]"
            guard case .object(let item) = block, case .string(let type)? = item["type"] else {
                throw invalid("\(blockPath) must be a content block with a type", blockPath)
            }
            switch (role, type) {
            case (_, "text"):
                guard case .string(let text)? = item["text"] else {
                    throw invalid("\(blockPath).text must be a string", blockPath)
                }
                if !text.isEmpty { texts.append(text) }
            case ("user", "image"):
                images.append(try imagePart(item, path: blockPath))
            case (_, "document"):
                texts.append(documentPlaceholder)
            case ("user", "tool_result"):
                guard case .string(let id)? = item["tool_use_id"] else {
                    throw invalid("\(blockPath).tool_use_id must be a string", blockPath)
                }
                toolResults.append(.object([
                    "role": .string("tool"),
                    "tool_call_id": .string(id),
                    "content": .string(try toolResultText(item["content"], path: blockPath)),
                ]))
            case ("assistant", "tool_use"):
                guard case .string(let id)? = item["id"], case .string(let name)? = item["name"] else {
                    throw invalid("\(blockPath) needs a string id and name", blockPath)
                }
                let input = item["input"] ?? .object([:])
                let arguments = String(decoding: try JSONEncoder().encode(input), as: UTF8.self)
                toolCalls.append(.object([
                    "id": .string(id),
                    "type": .string("function"),
                    "function": .object(["name": .string(name), "arguments": .string(arguments)]),
                ]))
            case ("assistant", "thinking"), ("assistant", "redacted_thinking"):
                // Earlier reasoning is not fed back: the chat templates drop
                // it from history too.
                continue
            default:
                throw invalid("content block type \"\(type)\" is not supported in a \(role) message",
                              "\(blockPath).type")
            }
        }

        let text = texts.joined(separator: "\n\n")
        if role == "assistant" {
            var out: [String: JSONValue] = ["role": .string("assistant"), "content": .string(text)]
            if !toolCalls.isEmpty {
                out["tool_calls"] = .array(toolCalls)
                if text.isEmpty { out["content"] = nil }
            }
            return [.object(out)]
        }
        var out = toolResults
        if !images.isEmpty {
            let parts = (text.isEmpty ? [] : [JSONValue.object([
                "type": .string("text"), "text": .string(text)])]) + images
            out.append(.object(["role": .string("user"), "content": .array(parts)]))
        } else if !text.isEmpty || toolResults.isEmpty {
            out.append(.object(["role": .string("user"), "content": .string(text)]))
        }
        return out
    }

    /// A `role: "system"` turn, which Claude Code sends inside the
    /// conversation: an instruction that holds until the next user message,
    /// a per-turn statement such as the effort level, or tools added and
    /// removed. The chat templates take a system message only at the start,
    /// so the turn's text is returned for the caller to place, its tool
    /// changes go to `tools`, and its control fields (`output_config`,
    /// `clear_at`) have no counterpart here and are ignored.
    private static func systemTurn(_ message: JSONValue,
                                   path: String,
                                   tools: inout ToolSet) throws -> String {
        let blocks: [JSONValue]
        switch message.objectValue?["content"] {
        case nil, .null?:
            return ""
        case .string(let text)?:
            return text
        case .array(let list)?:
            blocks = list
        default:
            throw invalid("\"\(path).content\" must be a string or an array of blocks",
                          "\(path).content")
        }
        var texts: [String] = []
        for (index, block) in blocks.enumerated() {
            let blockPath = "\(path).content[\(index)]"
            let item = block.objectValue ?? [:]
            switch item["type"] {
            case .string("text")?:
                if case .string(let text)? = item["text"], !text.isEmpty { texts.append(text) }
            case .string("tool_addition")?:
                try tools.add(item["tool"], path: "\(blockPath).tool")
            case .string("tool_removal")?:
                tools.remove(item["tool"])
            default:
                throw invalid("system content block type is not supported", "\(blockPath).type")
            }
        }
        return texts.joined(separator: "\n\n")
    }

    /// Consecutive user messages with plain text become one, so a system
    /// turn turned reminder never leaves two user turns in a row.
    private static func mergingAdjacentUserText(_ messages: [JSONValue]) -> [JSONValue] {
        var result: [JSONValue] = []
        for message in messages {
            if case .object(let current) = message,
               current["role"] == .string("user"),
               case .string(let text)? = current["content"],
               case .object(var previous)? = result.last,
               previous["role"] == .string("user"),
               case .string(let earlier)? = previous["content"] {
                previous["content"] = .string(earlier.isEmpty ? text : "\(earlier)\n\n\(text)")
                result[result.count - 1] = .object(previous)
            } else {
                result.append(message)
            }
        }
        return result
    }

    /// The tools a request makes available: the top-level `tools`, minus
    /// those marked `defer_loading` until something loads them, plus and
    /// minus what system turns add and remove. One list for the whole
    /// conversation, since the templates render tools once, up front.
    struct ToolSet {
        private var order: [String] = []
        private var tools: [String: (tool: JSONValue, path: String)] = [:]
        private var enabled: Set<String> = []

        init(_ list: JSONValue?) throws {
            switch list {
            case nil, .null?:
                return
            case .array(let entries)?:
                for (index, entry) in entries.enumerated() {
                    guard case .string(let name)? = entry.objectValue?["name"] else {
                        throw invalid("tools[\(index)].name must be a string", "tools[\(index)].name")
                    }
                    insert(name, entry, path: "tools[\(index)]")
                    if entry.objectValue?["defer_loading"] != .bool(true) { enabled.insert(name) }
                }
            default:
                throw invalid(#""tools" must be an array"#, "tools")
            }
        }

        var active: [(tool: JSONValue, path: String)] {
            order.filter(enabled.contains).compactMap { tools[$0] }
        }

        mutating func add(_ tool: JSONValue?, path: String) throws {
            let fields = tool?.objectValue ?? [:]
            switch fields["type"] {
            case .string("tool_definition")?:
                guard case .object(let definition)? = fields["definition"],
                      case .string(let name)? = definition["name"] else {
                    throw invalid("\(path).definition needs a name", "\(path).definition")
                }
                insert(name, .object(definition), path: "\(path).definition")
                enabled.insert(name)
            case .string("tool_reference")?:
                if case .string(let name)? = fields["name"], tools[name] != nil {
                    enabled.insert(name)
                }
            default:
                throw invalid("\(path).type must be tool_definition or tool_reference", path)
            }
        }

        mutating func remove(_ tool: JSONValue?) {
            if case .string(let name)? = tool?.objectValue?["name"] { enabled.remove(name) }
        }

        /// Deferred tools a tool result loaded with a `tool_reference` block.
        mutating func loadReferences(in message: JSONValue) {
            guard case .array(let blocks)? = message.objectValue?["content"] else { return }
            for block in blocks where block.objectValue?["type"] == .string("tool_result") {
                guard case .array(let inner)? = block.objectValue?["content"] else { continue }
                for item in inner where item.objectValue?["type"] == .string("tool_reference") {
                    if case .string(let name)? = item.objectValue?["tool_name"], tools[name] != nil {
                        enabled.insert(name)
                    }
                }
            }
        }

        private mutating func insert(_ name: String, _ tool: JSONValue, path: String) {
            if tools[name] == nil { order.append(name) }
            tools[name] = (tool, path)
        }
    }

    /// An image block as a data-URL `image_url` part. Whether images can be
    /// read at all is the validator's call: without an image pack it refuses
    /// the request rather than answering as though no image had been sent.
    private static func imagePart(_ block: [String: JSONValue], path: String) throws -> JSONValue {
        guard case .object(let source)? = block["source"],
              case .string(let kind)? = source["type"] else {
            throw invalid("\(path).source must be an object with a type", "\(path).source")
        }
        let url: String
        switch kind {
        case "base64":
            guard case .string(let media)? = source["media_type"],
                  case .string(let data)? = source["data"] else {
                throw invalid("\(path).source needs media_type and data", "\(path).source")
            }
            url = "data:\(media);base64,\(data)"
        case "url":
            guard case .string(let link)? = source["url"] else {
                throw invalid("\(path).source.url must be a string", "\(path).source")
            }
            url = link
        default:
            throw invalid("image source type \"\(kind)\" is not supported", "\(path).source")
        }
        return .object(["type": .string("image_url"), "image_url": .object(["url": .string(url)])])
    }

    /// A tool result's content as text. Chat Completions tool messages carry
    /// text only, so an image or document in a result becomes a line telling
    /// the model it was left out.
    private static func toolResultText(_ content: JSONValue?, path: String) throws -> String {
        switch content {
        case nil, .null?:
            return ""
        case .string(let text)?:
            return text
        case .array(let blocks)?:
            return try blocks.enumerated().map { index, block -> String in
                switch block.objectValue?["type"] {
                case .string("text")?:
                    guard case .string(let text)? = block.objectValue?["text"] else {
                        throw invalid("\(path).content[\(index)].text must be a string", path)
                    }
                    return text
                case .string("image")?:
                    return toolResultImagePlaceholder
                case .string("document")?:
                    return documentPlaceholder
                case .string("tool_reference")?:
                    // The tool itself joins the tool list (`ToolSet`).
                    let name = block.objectValue?["tool_name"].flatMap {
                        if case .string(let name) = $0 { return name } else { return nil }
                    } ?? "a tool"
                    return "[\(name) is now available]"
                default:
                    throw invalid("\(path).content[\(index)] has an unsupported block type", path)
                }
            }.joined(separator: "\n")
        default:
            throw invalid("\(path).content must be a string or an array of blocks", path)
        }
    }

    private static func translateTool(_ tool: JSONValue, path: String) throws -> JSONValue {
        guard case .object(let fields) = tool else { throw invalid("\(path) must be an object", path) }
        if case .string(let type)? = fields["type"], type != "custom" {
            throw invalid("tool type \"\(type)\" is not supported; only custom tools are", "\(path).type")
        }
        guard case .string(let name)? = fields["name"] else {
            throw invalid("\(path).name must be a string", "\(path).name")
        }
        let description: String? = {
            if case .string(let text)? = fields["description"] { return text }
            return nil
        }()
        let schema = fields["input_schema"] ?? .object(["type": .string("object")])
        let adapted = toolParameters(schema, name: name, description: description)
        var function: [String: JSONValue] = ["name": .string(name), "parameters": adapted.parameters]
        if let text = adapted.description { function["description"] = .string(text) }
        return .object(["type": .string("function"), "function": .object(function)])
    }

    // MARK: Tool schemas

    /// Tool parameters the chat templates can render. Claude Code's tools use
    /// full JSON Schema, and the server renders a smaller subset: every node
    /// typed, unions only of string constants or one type plus null. The
    /// schema is first reduced to that subset; if the validator still cannot
    /// take it, the tool keeps an open object and its original schema goes
    /// into the description, so the model still reads every parameter.
    static func toolParameters(_ schema: JSONValue,
                               name: String,
                               description: String?) -> (parameters: JSONValue, description: String?) {
        let reduced = reduce(schema)
        let candidate = OpenAITool(type: "function",
                                   function: OpenAIFunctionDefinition(name: name,
                                                                      description: description,
                                                                      parameters: reduced))
        if (try? OpenAIRequestValidator.validateTool(candidate)) != nil {
            return (reduced, description)
        }
        let original = (try? JSONEncoder().encode(schema)).map { String(decoding: $0, as: UTF8.self) }
            ?? "{}"
        let note = "Input JSON Schema: \(original)"
        let combined = description.map { "\($0)\n\n\(note)" } ?? note
        return (.object(["type": .string("object"), "properties": .object([:])]), combined)
    }

    private static let schemaTypes: Set<String> = [
        "string", "number", "integer", "boolean", "array", "object", "null",
    ]

    static func reduce(_ schema: JSONValue) -> JSONValue {
        guard case .object(var node) = schema else {
            // `true`, `{}`-like booleans and anything else: any value, which
            // a string describes best.
            return .object(["type": .string("string")])
        }
        for key in ["$schema", "$id", "$defs", "definitions", "$ref", "$comment"] {
            node[key] = nil
        }
        if case .array(let parts)? = node["allOf"] {
            node["allOf"] = nil
            for case .object(let part) in parts {
                node = merge(node, part)
            }
        }
        for keyword in ["anyOf", "oneOf"] {
            guard case .array(let branches)? = node[keyword] else { continue }
            node = reduceUnion(node, keyword: keyword, branches: branches)
        }
        if node["anyOf"] == nil && node["oneOf"] == nil {
            let nullable: Bool = {
                if case .array(let types)? = node["type"] { return types.contains(.string("null")) }
                return false
            }()
            let type = JSONValue.string(reducedType(node))
            node["type"] = nullable ? .array([type, .string("null")]) : type
        }
        if case .object(let properties)? = node["properties"] {
            node["properties"] = .object(properties.mapValues(reduce))
        }
        switch node["items"] {
        case .object?:
            node["items"] = reduce(node["items"]!)
        case .array(let tuple)?:
            node["items"] = tuple.first.map(reduce) ?? .object(["type": .string("string")])
        case .bool?:
            node["items"] = .object(["type": .string("string")])
        default:
            break
        }
        return .object(node)
    }

    /// A union the templates can render stays a union (string constants, or
    /// one schema plus null); any other collapses to its common type, or to
    /// its first branch when the branches disagree.
    private static func reduceUnion(_ node: [String: JSONValue],
                                    keyword: String,
                                    branches: [JSONValue]) -> [String: JSONValue] {
        var node = node
        let objects = branches.compactMap(\.objectValue)
        let isNull: ([String: JSONValue]) -> Bool = { $0["type"] == .string("null") }
        let concrete = objects.filter { !isNull($0) }
        let hasNull = objects.count != concrete.count
        let stringConstants = concrete.count == objects.count && objects.count >= 2
            && objects.allSatisfy {
                Set($0.keys).isSubset(of: ["const", "type"]) && $0["type"] == .string("string")
                    && { if case .string? = $0["const"] { true } else { false } }($0)
            }
        if stringConstants { return node }
        if objects.count == 2, hasNull, concrete.count == 1 {
            node[keyword] = .array([reduce(.object(concrete[0])), .object(["type": .string("null")])])
            return node
        }
        node[keyword] = nil
        let types = Set(concrete.compactMap { branch -> String? in
            if case .string(let type)? = branch["type"] { return type }
            return nil
        })
        if types.count == 1, let type = types.first, concrete.allSatisfy({ $0["type"] != nil }) {
            node["type"] = .string(type)
            if type == "object" {
                var properties: [String: JSONValue] = [:]
                for branch in concrete {
                    if case .object(let more)? = branch["properties"] {
                        properties.merge(more) { first, _ in first }
                    }
                }
                node["properties"] = .object(properties)
            } else if type == "array", let items = concrete.first?["items"] {
                node["items"] = items
            } else if type == "string" {
                let constants = concrete.compactMap { $0["const"] }
                if constants.count == concrete.count { node["enum"] = .array(constants) }
            }
        } else if let first = concrete.first {
            node = merge(node, first)
        }
        if hasNull, case .string(let type)? = node["type"] {
            node["type"] = .array([.string(type), .string("null")])
        }
        return node
    }

    private static func reducedType(_ node: [String: JSONValue]) -> String {
        switch node["type"] {
        case .string(let type)? where schemaTypes.contains(type) && type != "null":
            return type
        case .array(let types)?:
            for case .string(let type) in types where type != "null" && schemaTypes.contains(type) {
                return type
            }
        default:
            break
        }
        if node["properties"] != nil { return "object" }
        if node["items"] != nil { return "array" }
        let sample: JSONValue? = node["const"] ?? {
            if case .array(let values)? = node["enum"] { return values.first }
            return nil
        }()
        switch sample {
        case .bool?: return "boolean"
        case .integer?, .unsignedInteger?: return "integer"
        case .number?, .decimal?: return "number"
        default: return "string"
        }
    }

    /// `base` with `extra`'s keywords added; `base` wins where both speak,
    /// except that properties and required lists are combined.
    private static func merge(_ base: [String: JSONValue],
                              _ extra: [String: JSONValue]) -> [String: JSONValue] {
        var result = base
        for (key, value) in extra {
            switch (key, result[key], value) {
            case ("properties", .object(let mine)?, .object(let theirs)):
                result[key] = .object(mine.merging(theirs) { first, _ in first })
            case ("required", .array(let mine)?, .array(let theirs)):
                result[key] = .array(mine + theirs.filter { !mine.contains($0) })
            default:
                if result[key] == nil { result[key] = value }
            }
        }
        return result
    }

    // MARK: Responses

    static func stopReason(_ finishReason: String) -> String {
        switch finishReason {
        case "length": "max_tokens"
        case "tool_calls": "tool_use"
        default: "end_turn"
        }
    }

    /// Anthropic counts cached prompt tokens apart from `input_tokens`, so the
    /// two add up to the prompt.
    static func usage(_ usage: OpenAIUsage) -> [String: Any] {
        let cached = usage.promptTokensDetails.cachedTokens
        return [
            "input_tokens": max(usage.promptTokens - cached, 0),
            "output_tokens": usage.completionTokens,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        ]
    }

    static func toolInput(_ call: ParsedToolCall) -> Any {
        (try? JSONSerialization.jsonObject(with: Data(call.argumentsJSON.utf8))) ?? [String: Any]()
    }

    static func message(id: String, model: String, completion: ServerCompletion) -> [String: Any] {
        var content: [[String: Any]] = []
        if !completion.content.isEmpty {
            content.append(["type": "text", "text": completion.content])
        }
        for call in completion.toolCalls {
            content.append(["type": "tool_use", "id": call.id, "name": call.name,
                            "input": toolInput(call)])
        }
        return [
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stopReason(completion.finishReason),
            "stop_sequence": NSNull(),
            "usage": usage(completion.usage),
        ]
    }

    static func errorType(status: UInt) -> String {
        switch status {
        case 400, 405, 415: "invalid_request_error"
        case 401: "authentication_error"
        case 403: "permission_error"
        case 404: "not_found_error"
        case 413: "request_too_large"
        case 429: "rate_limit_error"
        case 503, 529: "overloaded_error"
        default: "api_error"
        }
    }

    static func errorObject(status: UInt, message: String) -> [String: Any] {
        ["type": "error", "error": ["type": errorType(status: status), "message": message]]
    }

    /// `GET /v1/models` in Anthropic's shape, answered when the request
    /// carries an `anthropic-version` header.
    static func modelList(modelID: String) -> [String: Any] {
        [
            "data": [[
                "type": "model",
                "id": modelID,
                "display_name": modelID,
                "created_at": "1970-01-01T00:00:00Z",
            ]],
            "has_more": false,
            "first_id": modelID,
            "last_id": modelID,
        ]
    }

    private static func integer(_ value: JSONValue) -> Int? {
        switch value {
        case .integer(let number): Int(exactly: number)
        case .unsignedInteger(let number): Int(exactly: number)
        case .number(let number): number.rounded() == number ? Int(exactly: number) : nil
        case .decimal(let number):
            { (double: Double) in double.rounded() == double ? Int(exactly: double) : nil }(
                NSDecimalNumber(decimal: number).doubleValue)
        default: nil
        }
    }

    private static func isNumber(_ value: JSONValue) -> Bool {
        switch value {
        case .integer, .unsignedInteger, .number, .decimal: true
        default: false
        }
    }

    private static func invalid(_ message: String, _ param: String?) -> ServerRequestError {
        .invalid(message: message, param: param, code: "invalid_request")
    }
}

/// The server-sent events of one streamed Messages response, in the order
/// Anthropic defines: `message_start`, then per content block a start, its
/// deltas and a stop, then `message_delta` with the stop reason and usage, and
/// `message_stop`. Text opens a block on its first delta; a tool call closes
/// any open text block and is sent whole, as one `input_json_delta`.
final class AnthropicStreamEncoder: Sendable {
    private struct State {
        var nextIndex = 0
        var openText: Int?
    }

    let id: String
    let model: String
    private let state = Mutex(State())

    init(id: String, model: String) {
        self.id = id
        self.model = model
    }

    func messageStart(inputTokens: Int) -> String {
        Self.event("message_start", [
            "type": "message_start",
            "message": [
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [Any](),
                "stop_reason": NSNull(),
                "stop_sequence": NSNull(),
                "usage": [
                    "input_tokens": inputTokens,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                ],
            ],
        ])
    }

    func text(_ delta: String) -> String {
        guard !delta.isEmpty else { return "" }
        return state.withLock { state in
            var out = ""
            let index: Int
            if let open = state.openText {
                index = open
            } else {
                index = state.nextIndex
                state.nextIndex += 1
                state.openText = index
                out += Self.event("content_block_start", [
                    "type": "content_block_start",
                    "index": index,
                    "content_block": ["type": "text", "text": ""],
                ])
            }
            out += Self.event("content_block_delta", [
                "type": "content_block_delta",
                "index": index,
                "delta": ["type": "text_delta", "text": delta],
            ])
            return out
        }
    }

    func toolCall(_ call: ParsedToolCall) -> String {
        state.withLock { state in
            var out = Self.closeText(&state)
            let index = state.nextIndex
            state.nextIndex += 1
            out += Self.event("content_block_start", [
                "type": "content_block_start",
                "index": index,
                "content_block": ["type": "tool_use", "id": call.id, "name": call.name,
                                  "input": [String: Any]()],
            ])
            out += Self.event("content_block_delta", [
                "type": "content_block_delta",
                "index": index,
                "delta": ["type": "input_json_delta", "partial_json": call.argumentsJSON],
            ])
            out += Self.event("content_block_stop", ["type": "content_block_stop", "index": index])
            return out
        }
    }

    func finish(_ completion: ServerCompletion) -> String {
        state.withLock { state in
            Self.closeText(&state)
                + Self.event("message_delta", [
                    "type": "message_delta",
                    "delta": [
                        "stop_reason": AnthropicMessages.stopReason(completion.finishReason),
                        "stop_sequence": NSNull(),
                    ],
                    "usage": AnthropicMessages.usage(completion.usage),
                ])
                + Self.event("message_stop", ["type": "message_stop"])
        }
    }

    static let ping = event("ping", ["type": "ping"])

    static func error(status: UInt, message: String) -> String {
        event("error", AnthropicMessages.errorObject(status: status, message: message))
    }

    private static func closeText(_ state: inout State) -> String {
        guard let open = state.openText else { return "" }
        state.openText = nil
        return event("content_block_stop", ["type": "content_block_stop", "index": open])
    }

    static func event(_ name: String, _ object: [String: Any]) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
            ?? Data("{}".utf8)
        return "event: \(name)\ndata: \(String(decoding: data, as: UTF8.self))\n\n"
    }
}

/// The prompt's token count once the request is prepared, for
/// `message_start`, which is sent before generation reports anything.
final class PreparedTokenCount: Sendable {
    private let count = Mutex<Int?>(nil)

    func set(_ value: Int?) { count.withLock { $0 = value } }
    var value: Int? { count.withLock { $0 } }
}
