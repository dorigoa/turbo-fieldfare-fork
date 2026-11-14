import Foundation
import Jinja

/// Errors raised while loading or rendering the pinned Qwen chat template.
public enum QwenChatFormatError: Error, CustomStringConvertible {
    case missingTemplate(path: String)
    case unreadableTemplate(path: String, reason: String)
    case renderFailed(reason: String)
    case invalidJSON(reason: String)

    public var description: String {
        switch self {
        case .missingTemplate(let path):
            return "chat template file is missing: \(path)"
        case .unreadableTemplate(let path, let reason):
            return "chat template file is unreadable (\(path)): \(reason)"
        case .renderFailed(let reason):
            return "chat template rendering failed: \(reason)"
        case .invalidJSON(let reason):
            return "invalid JSON in chat template input: \(reason)"
        }
    }
}

/// A JSON value that keeps object key order.
///
/// The template embeds tool schemas with `|tojson`, and the pinned golden
/// renders object keys in their original document order. Neither `JSONValue`
/// (a Swift dictionary) nor Jinja's `Value(any:)` (which sorts keys) can
/// carry that order, so chat rendering uses this ordered form end to end:
/// the golden fixtures and any caller that parsed raw JSON text keep the
/// caller's key order, while values converted from `JSONValue` keep an
/// arbitrary (but self-consistent) object order, as before.
public indirect enum QwenChatValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int64)
    case double(Double)
    case string(String)
    case array([QwenChatValue])
    /// Ordered key/value pairs; duplicate keys keep the last value.
    case object([(key: String, value: QwenChatValue)])

    /// The Jinja representation, preserving this value's object key order.
    func jinjaValue() -> Jinja.Value {
        switch self {
        case .null: return .null
        case .bool(let value): return .boolean(value)
        case .int(let value): return .int(Int(value))
        case .double(let value): return .double(value)
        case .string(let value): return .string(value)
        case .array(let values): return .array(values.map { $0.jinjaValue() })
        case .object(let pairs):
            var object = OrderedDictionary<String, Jinja.Value>()
            for pair in pairs { object[pair.key] = pair.value.jinjaValue() }
            return .object(object)
        }
    }

    /// Convert a `JSONValue` (whose objects are unordered Swift dictionaries)
    /// into the ordered form, with object keys sorted. Two equal `JSONValue`
    /// instances held as separate dictionaries can iterate in different
    /// orders in the same process, and chat continuations compare renders
    /// built from such separate instances (see
    /// `QwenTokenizer.encodeToolResultContinuation`), so the conversion must
    /// canonicalize the order rather than follow it. Golden renders keep the
    /// fixture's document order because they arrive through `parseJSON`.
    public init(_ value: JSONValue) {
        switch value {
        case .null: self = .null
        case .bool(let flag): self = .bool(flag)
        case .integer(let i): self = .int(i)
        case .unsignedInteger(let u): self = .int(Int64(u))
        case .decimal(let d): self = .double(NSDecimalNumber(decimal: d).doubleValue)
        case .number(let d): self = .double(d)
        case .string(let s): self = .string(s)
        case .array(let items): self = .array(items.map(QwenChatValue.init))
        case .object(let entries):
            self = .object(entries.sorted { $0.key < $1.key }
                .map { (key: $0.key, value: QwenChatValue($0.value)) })
        }
    }

    /// Parse strict JSON keeping object key order (used by the golden tests
    /// and by any caller holding raw JSON text).
    public static func parseJSON(_ text: String) throws -> QwenChatValue {
        var parser = JSONOrderParser(Array(text.utf8))
        let value = try parser.parseValue()
        try parser.finish()
        return value
    }

    // Object payloads are labeled-tuple arrays, which synthesis cannot
    // compare, so equality is spelled out (key order is part of equality).
    public static func == (lhs: QwenChatValue, rhs: QwenChatValue) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null): return true
        case (.bool(let a), .bool(let b)): return a == b
        case (.int(let a), .int(let b)): return a == b
        case (.double(let a), .double(let b)): return a == b
        case (.string(let a), .string(let b)): return a == b
        case (.array(let a), .array(let b)): return a == b
        case (.object(let a), .object(let b)):
            guard a.count == b.count else { return false }
            for (left, right) in zip(a, b)
            where left.key != right.key || left.value != right.value { return false }
            return true
        default: return false
        }
    }
}

/// Minimal strict JSON scanner preserving object key order.
private struct JSONOrderParser {
    private let bytes: [UInt8]
    private var index = 0

    init(_ bytes: [UInt8]) { self.bytes = bytes }

    mutating func finish() throws {
        skipWhitespace()
        guard index == bytes.count else {
            throw QwenChatFormatError.invalidJSON(reason: "trailing characters")
        }
    }

    mutating func parseValue() throws -> QwenChatValue {
        skipWhitespace()
        guard let byte = peek() else {
            throw QwenChatFormatError.invalidJSON(reason: "unexpected end of input")
        }
        switch byte {
        case UInt8(ascii: "{"): return try parseObject()
        case UInt8(ascii: "["): return try parseArray()
        case UInt8(ascii: "\""): return .string(try parseString())
        case UInt8(ascii: "t"): try expectWord("true"); return .bool(true)
        case UInt8(ascii: "f"): try expectWord("false"); return .bool(false)
        case UInt8(ascii: "n"): try expectWord("null"); return .null
        default: return try parseNumber()
        }
    }

    private mutating func parseObject() throws -> QwenChatValue {
        _ = advance() // {
        var pairs: [(key: String, value: QwenChatValue)] = []
        skipWhitespace()
        if peek() == UInt8(ascii: "}") {
            _ = advance()
            return .object(pairs)
        }
        while true {
            skipWhitespace()
            guard peek() == UInt8(ascii: "\"") else {
                throw QwenChatFormatError.invalidJSON(reason: "expected object key")
            }
            let key = try parseString()
            skipWhitespace()
            guard advance() == UInt8(ascii: ":") else {
                throw QwenChatFormatError.invalidJSON(reason: "expected ':'")
            }
            let value = try parseValue()
            if let last = pairs.lastIndex(where: { $0.key == key }) {
                pairs[last] = (key: key, value: value)
            } else {
                pairs.append((key: key, value: value))
            }
            skipWhitespace()
            switch advance() {
            case UInt8(ascii: ","): continue
            case UInt8(ascii: "}"): return .object(pairs)
            default:
                throw QwenChatFormatError.invalidJSON(reason: "expected ',' or '}'")
            }
        }
    }

    private mutating func parseArray() throws -> QwenChatValue {
        _ = advance() // [
        var items: [QwenChatValue] = []
        skipWhitespace()
        if peek() == UInt8(ascii: "]") {
            _ = advance()
            return .array(items)
        }
        while true {
            items.append(try parseValue())
            skipWhitespace()
            switch advance() {
            case UInt8(ascii: ","): continue
            case UInt8(ascii: "]"): return .array(items)
            default:
                throw QwenChatFormatError.invalidJSON(reason: "expected ',' or ']'")
            }
        }
    }

    private mutating func parseString() throws -> String {
        _ = advance() // opening quote
        var units: [UInt16] = []
        while true {
            guard let byte = advance() else {
                throw QwenChatFormatError.invalidJSON(reason: "unterminated string")
            }
            switch byte {
            case UInt8(ascii: "\""):
                return String(decoding: units, as: UTF16.self)
            case UInt8(ascii: "\\"):
                guard let escape = advance() else {
                    throw QwenChatFormatError.invalidJSON(reason: "unterminated escape")
                }
                switch escape {
                case UInt8(ascii: "\""): units.append(0x22)
                case UInt8(ascii: "\\"): units.append(0x5C)
                case UInt8(ascii: "/"): units.append(0x2F)
                case UInt8(ascii: "b"): units.append(0x08)
                case UInt8(ascii: "f"): units.append(0x0C)
                case UInt8(ascii: "n"): units.append(0x0A)
                case UInt8(ascii: "r"): units.append(0x0D)
                case UInt8(ascii: "t"): units.append(0x09)
                case UInt8(ascii: "u"): units.append(try parseHex4())
                default:
                    throw QwenChatFormatError.invalidJSON(reason: "unknown escape")
                }
            default:
                // Raw UTF-8: re-collect the sequence and append its UTF-16 units.
                var sequence: [UInt8] = [byte]
                let extra = Self.continuationBytes(of: byte)
                for _ in 0..<extra {
                    guard let next = advance() else {
                        throw QwenChatFormatError.invalidJSON(reason: "truncated UTF-8")
                    }
                    sequence.append(next)
                }
                units.append(contentsOf: Array(String(decoding: sequence, as: UTF8.self).utf16))
            }
        }
    }

    private mutating func parseHex4() throws -> UInt16 {
        var value: UInt16 = 0
        for _ in 0..<4 {
            guard let byte = advance(), let digit = Self.hexDigit(byte) else {
                throw QwenChatFormatError.invalidJSON(reason: "bad unicode escape")
            }
            value = value << 4 | digit
        }
        return value
    }

    private mutating func parseNumber() throws -> QwenChatValue {
        let start = index
        var isDouble = false
        if peek() == UInt8(ascii: "-") { _ = advance() }
        scan: while let byte = peek() {
            switch byte {
            case UInt8(ascii: "0")...UInt8(ascii: "9"):
                _ = advance()
            case UInt8(ascii: "."), UInt8(ascii: "e"), UInt8(ascii: "E"):
                isDouble = true
                _ = advance()
            case UInt8(ascii: "+"), UInt8(ascii: "-"):
                // Only valid right after an exponent marker; a dangling one is
                // rewound below so the caller reports the real error.
                _ = advance()
            default:
                break scan
            }
        }
        var end = index
        var text = String(decoding: bytes[start..<end], as: UTF8.self)
        while let last = text.last, last == "+" || last == "-" || last == "e" || last == "E" || last == "." {
            text.removeLast()
            end -= 1
        }
        index = end
        guard !text.isEmpty else {
            throw QwenChatFormatError.invalidJSON(reason: "expected a value")
        }
        if isDouble {
            guard let value = Double(text) else {
                throw QwenChatFormatError.invalidJSON(reason: "bad number \(text)")
            }
            return .double(value)
        }
        guard let value = Int64(text) else {
            guard let value = Double(text) else {
                throw QwenChatFormatError.invalidJSON(reason: "bad number \(text)")
            }
            return .double(value)
        }
        return .int(value)
    }

    private mutating func expectWord(_ word: String) throws {
        for unit in word.utf8 {
            guard advance() == unit else {
                throw QwenChatFormatError.invalidJSON(reason: "expected \(word)")
            }
        }
    }

    private mutating func skipWhitespace() {
        while let byte = peek(),
              byte == UInt8(ascii: " ") || byte == UInt8(ascii: "\n")
                || byte == UInt8(ascii: "\r") || byte == UInt8(ascii: "\t") {
            _ = advance()
        }
    }

    private func peek() -> UInt8? { index < bytes.count ? bytes[index] : nil }

    @discardableResult
    private mutating func advance() -> UInt8? {
        defer { if index < bytes.count { index += 1 } }
        return index < bytes.count ? bytes[index] : nil
    }

    private static func hexDigit(_ byte: UInt8) -> UInt16? {
        switch byte {
        case UInt8(ascii: "0")...UInt8(ascii: "9"): return UInt16(byte - UInt8(ascii: "0"))
        case UInt8(ascii: "a")...UInt8(ascii: "f"): return UInt16(byte - UInt8(ascii: "a") + 10)
        case UInt8(ascii: "A")...UInt8(ascii: "F"): return UInt16(byte - UInt8(ascii: "A") + 10)
        default: return nil
        }
    }

    private static func continuationBytes(of lead: UInt8) -> Int {
        switch lead {
        case 0xC0...0xDF: return 1
        case 0xE0...0xEF: return 2
        case 0xF0...0xF7: return 3
        default: return 0
        }
    }
}

/// One historical tool call inside an assistant turn (template `tool_calls`).
public struct QwenToolCall: Sendable, Equatable {
    public let id: String
    public let name: String
    /// Arguments as an ordered JSON object; the template iterates
    /// `arguments|items`, so key order is emitted verbatim.
    public let arguments: QwenChatValue

    public init(id: String, name: String, arguments: QwenChatValue) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

/// One message in the template's representation (design decision D8).
///
/// `reasoningContent` maps to the template's `reasoning_content` variable:
/// present only when the caller supplies it, exactly like `apply_chat_template`
/// on the pinned checkpoint. An assistant turn without it whose content
/// contains a closing think marker still gets its reasoning extracted by the
/// template itself.
public struct QwenChatMessage: Sendable {
    public let role: String
    public let content: String?
    public let reasoningContent: String?
    public let name: String?
    public let toolCallID: String?
    public let toolCalls: [QwenToolCall]

    public init(role: String,
                content: String?,
                reasoningContent: String? = nil,
                name: String? = nil,
                toolCallID: String? = nil,
                toolCalls: [QwenToolCall] = []) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.name = name
        self.toolCallID = toolCallID
        self.toolCalls = toolCalls
    }

    func jinjaValue() -> Jinja.Value {
        var object: OrderedDictionary<String, Jinja.Value> = [:]
        object["role"] = .string(role)
        object["content"] = content.map(Jinja.Value.string) ?? .null
        if let reasoningContent { object["reasoning_content"] = .string(reasoningContent) }
        if let name { object["name"] = .string(name) }
        if let toolCallID { object["tool_call_id"] = .string(toolCallID) }
        if !toolCalls.isEmpty {
            object["tool_calls"] = .array(toolCalls.map { call in
                var callObject: OrderedDictionary<String, Jinja.Value> = [:]
                callObject["id"] = .string(call.id)
                callObject["type"] = .string("function")
                var function: OrderedDictionary<String, Jinja.Value> = [:]
                function["name"] = .string(call.name)
                function["arguments"] = call.arguments.jinjaValue()
                callObject["function"] = .object(function)
                return .object(callObject)
            })
        }
        return .object(object)
    }
}

/// Renders the pinned checkpoint's `chat_template.jinja` with the same Jinja
/// engine `GFTokenizer.encodeToolChat` already uses (design decision D8).
///
/// The template is loaded from the `chat_template.jinja` beside the tokenizer
/// files (contract C6); the pinned tokenizer_config declares no inline
/// template, so it is passed to the engine as a literal. The context mirrors
/// what `apply_chat_template` passes on the pinned checkpoint: `messages`,
/// `tools`, `add_generation_prompt`, `enable_thinking`, and `preserve_thinking`
/// only when set (its absence is the standard "drop reasoning in older turns"
/// behavior, see the S2 report). All 16 golden conversations in
/// `Tests/QwenFixtures/chat/chat_golden.json` render byte-identically.
public struct QwenChatFormat: Sendable {
    /// File name the repack copies into the `tokenizer/` sidecar (C6).
    public static let templateFileName = "chat_template.jinja"

    private let template: Template

    /// Compile a template string (the engine `swift-transformers` bundles).
    public init(jinjaTemplate: String) throws {
        do {
            self.template = try Template(jinjaTemplate)
        } catch {
            throw QwenChatFormatError.renderFailed(
                reason: "template did not compile: \(error)")
        }
    }

    /// Load the template file shipped beside the tokenizer files in `folder`.
    public static func load(from folder: URL) throws -> QwenChatFormat {
        let url = folder.appendingPathComponent(templateFileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw QwenChatFormatError.missingTemplate(path: url.path)
        }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw QwenChatFormatError.unreadableTemplate(path: url.path, reason: "not UTF-8")
        }
        return try QwenChatFormat(jinjaTemplate: text)
    }

    /// Render one conversation to the exact prompt string the pinned template
    /// produces for these arguments.
    public func render(messages: [QwenChatMessage],
                       tools: [QwenChatValue]? = nil,
                       addGenerationPrompt: Bool = true,
                       enableThinking: Bool = true,
                       preserveThinking: Bool = false) throws -> String {
        var context: [String: Jinja.Value] = [
            "messages": .array(messages.map { $0.jinjaValue() }),
            "add_generation_prompt": .boolean(addGenerationPrompt),
            "enable_thinking": .boolean(enableThinking),
        ]
        if let tools, !tools.isEmpty {
            context["tools"] = .array(tools.map { $0.jinjaValue() })
        }
        if preserveThinking {
            context["preserve_thinking"] = .boolean(true)
        }
        do {
            return try template.render(context, environment: Self.pythonTojsonEnvironment())
        } catch {
            throw QwenChatFormatError.renderFailed(reason: "\(error)")
        }
    }

    /// An environment whose `tojson` filter matches the Python one the
    /// checkpoint rendered the goldens with: object keys keep their document
    /// order, separators are `", "` / `": "`, and non-ASCII characters stay
    /// literal. swift-jinja's built-in uses `JSONEncoder` with `sortedKeys`
    /// and compact separators, which renames the golden bytes of every
    /// `tools` schema and non-string tool argument, so it is replaced here
    /// (the engine looks up environment variables before built-in filters).
    private static func pythonTojsonEnvironment() -> Environment {
        let environment = Environment()
        environment["tojson"] = .function { arguments, _, _ in
            .string(pythonJSON(arguments.first ?? .null))
        }
        return environment
    }

    /// Serialize a Jinja value the way Python's `json.dumps(ensure_ascii=False)` does.
    private static func pythonJSON(_ value: Jinja.Value) -> String {
        switch value {
        case .boolean(let flag): return flag ? "true" : "false"
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .string(let text): return pythonJSONString(text)
        case .array(let items):
            return "[" + items.map(pythonJSON).joined(separator: ", ") + "]"
        case .object(let entries):
            // `OrderedDictionary` iteration keeps the document key order.
            return "{" + entries
                .map { pythonJSONString($0.key) + ": " + pythonJSON($0.value) }
                .joined(separator: ", ") + "}"
        default:
            // `.null`, `.undefined` and unserializable callables all render
            // as `null`, matching what the engine's built-in falls back to.
            return "null"
        }
    }

    /// Quote a string the way Python's `json.dumps(ensure_ascii=False)` does.
    private static func pythonJSONString(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}
