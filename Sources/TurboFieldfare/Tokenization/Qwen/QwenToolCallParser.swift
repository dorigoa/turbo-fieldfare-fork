import Foundation

/// Errors for `QwenToolCallParser`.
public enum QwenToolCallParserError: Error, Equatable {
    case malformed
    case unknownTool(String)
    case oversized
}

/// Parses one Qwen XML tool call (plan 3.6) into `ParsedToolCall`:
///
/// ```text
/// [tool_call open marker]
/// <function=NAME>
/// <parameter=PARAM>
/// VALUE
/// </parameter>
/// </function>
/// [tool_call close marker]
/// ```
///
/// The two bracketed markers are the `tool_call` special tokens of the pinned
/// vocabulary (see `Self.callOpen`/`callClose`); the inner tags are plain
/// text. A parameter VALUE is
/// first tried as JSON (numbers, booleans, objects, lists); anything else is
/// a string. Exactly one leading and one trailing newline are removed before
/// that attempt — they are layout, not content.
public struct QwenToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public static let callOpen = "\u{3C}|tool_call|\u{3E}"
    public static let callClose = "\u{3C}/tool_call|\u{3E}"

    public init() {}

    public func parse(_ text: String,
                      allowedTools: Set<String>,
                      id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= Self.maximumBytes else {
            throw QwenToolCallParserError.oversized
        }
        var body = Substring(text)
        if body.hasPrefix(Self.callOpen) { body = body.dropFirst(Self.callOpen.count) }
        if body.hasSuffix(Self.callClose) {
            body = body.dropLast(Self.callClose.count)
        }
        var parser = Parser(body)
        parser.skipWhitespace()
        guard parser.take("<function=") else {
            throw QwenToolCallParserError.malformed
        }
        let name = try parser.identifier(until: ">")
        guard parser.take(">") else { throw QwenToolCallParserError.malformed }
        guard allowedTools.contains(name) else {
            throw QwenToolCallParserError.unknownTool(name)
        }

        var arguments: [String: JSONValue] = [:]
        while true {
            parser.skipWhitespace()
            if parser.take("</function>") { break }
            guard parser.take("<parameter=") else {
                throw QwenToolCallParserError.malformed
            }
            let parameter = try parser.identifier(until: ">")
            guard parser.take(">"), !arguments.keys.contains(parameter) else {
                throw QwenToolCallParserError.malformed
            }
            arguments[parameter] = try parser.parameterValue()
        }
        parser.skipWhitespace()
        guard parser.isAtEnd else { throw QwenToolCallParserError.malformed }
        let object = JSONValue.object(arguments)
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: object,
                              argumentsJSON: try object.encoded())
    }
}

private struct Parser {
    private let characters: [Character]
    private var index = 0

    init(_ text: Substring) {
        characters = Array(text)
    }

    var isAtEnd: Bool { index == characters.count }

    /// Whitespace between tags is layout. A value never goes through here:
    /// its text is taken verbatim between `<parameter=…>` and `</parameter>`.
    mutating func skipWhitespace() {
        while index < characters.count, characters[index].isWhitespace { index += 1 }
    }

    mutating func take(_ literal: String) -> Bool {
        let value = Array(literal)
        guard index + value.count <= characters.count,
              Array(characters[index..<(index + value.count)]) == value else {
            return false
        }
        index += value.count
        return true
    }

    mutating func identifier(until end: Character) throws -> String {
        let start = index
        while index < characters.count {
            let character = characters[index]
            guard character.isLetter || character.isNumber
                    || "_-.$".contains(character) else { break }
            index += 1
        }
        guard index > start, index < characters.count, characters[index] == end else {
            throw QwenToolCallParserError.malformed
        }
        return String(characters[start..<index])
    }

    /// The text between `<parameter=NAME>` and the next `</parameter>`, with
    /// exactly one leading and one trailing newline removed, then parsed.
    mutating func parameterValue() throws -> JSONValue {
        let start = index
        while index < characters.count,
              !starts(with: "</parameter>") { index += 1 }
        guard index < characters.count else {
            throw QwenToolCallParserError.malformed
        }
        var value = String(characters[start..<index])
        index += "</parameter>".count
        if value.hasPrefix("\n") { value.removeFirst() }
        if value.hasSuffix("\n") { value.removeLast() }
        return try Self.coerce(value)
    }

    /// Try JSON first (numbers, booleans, objects, lists per plan 3.6; a
    /// valid `null` or quoted string decodes through the same path); if the
    /// text is not valid JSON it stays a plain string.
    static func coerce(_ value: String) throws -> JSONValue {
        if let decoded = try? JSONDecoder().decode(JSONValue.self,
                                                   from: Data(value.utf8)) {
            return decoded
        }
        return .string(value)
    }

    func starts(with literal: String) -> Bool {
        let value = Array(literal)
        return index + value.count <= characters.count
            && Array(characters[index..<(index + value.count)]) == value
    }
}
