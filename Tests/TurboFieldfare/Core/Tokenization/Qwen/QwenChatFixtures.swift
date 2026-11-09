import Foundation
@testable import TurboFieldfare

/// Parses `Tests/QwenFixtures/chat/chat_golden.json` into the types the chat
/// format consumes (contract C7's fixture convention, S10).
///
/// The file is parsed with `QwenChatValue.parseJSON`, not `JSONDecoder`,
/// because the goldens pin object key order (`tools` and tool-call
/// `arguments` re-appear verbatim through the template's `|tojson`), which
/// `JSONDecoder`'s dictionary would lose.
enum QwenChatFixtures {
    struct GoldenCase {
        var name: String
        var messages: [QwenChatMessage]
        var tools: [QwenChatValue]?
        var enableThinking: Bool
        var addGenerationPrompt: Bool
        var rendered: String
        var ids: [Int32]
    }

    enum Failure: Error, CustomStringConvertible {
        case malformed(String)

        var description: String {
            switch self {
            case .malformed(let reason): return "malformed chat golden fixture: \(reason)"
            }
        }
    }

    static func goldenCases() throws -> [GoldenCase] {
        let url = QwenTestResources.fixtures
            .appendingPathComponent("chat", isDirectory: true)
            .appendingPathComponent("chat_golden.json")
        let text = try String(contentsOf: url, encoding: .utf8)
        guard case .array(let entries) = try QwenChatValue.parseJSON(text) else {
            throw Failure.malformed("top level is not an array")
        }
        return try entries.map { entry in
            guard case .string(let name)? = lookup(entry, "name") else {
                throw Failure.malformed("case without a name")
            }
            guard case .array(let messageValues)? = lookup(entry, "messages") else {
                throw Failure.malformed("\(name): no messages array")
            }
            guard case .string(let rendered)? = lookup(entry, "rendered") else {
                throw Failure.malformed("\(name): no rendered string")
            }
            guard case .array(let idValues)? = lookup(entry, "ids") else {
                throw Failure.malformed("\(name): no ids array")
            }
            var ids: [Int32] = []
            for id in idValues {
                guard case .int(let value) = id, let int32 = Int32(exactly: value) else {
                    throw Failure.malformed("\(name): bad id \(id)")
                }
                ids.append(int32)
            }
            var tools: [QwenChatValue]?
            if case .array(let toolValues)? = lookup(entry, "tools") {
                tools = toolValues
            }
            var enableThinking = true
            if case .bool(let flag)? = lookup(entry, "enable_thinking") {
                enableThinking = flag
            }
            let addGenerationPrompt: Bool
            if case .bool(let flag)? = lookup(entry, "add_generation_prompt") {
                addGenerationPrompt = flag
            } else {
                addGenerationPrompt = true
            }
            return GoldenCase(
                name: name,
                messages: try messageValues.map { try message($0, case: name) },
                tools: tools,
                enableThinking: enableThinking,
                addGenerationPrompt: addGenerationPrompt,
                rendered: rendered,
                ids: ids)
        }
    }

    static func message(_ value: QwenChatValue, case name: String = "?") throws -> QwenChatMessage {
        guard case .string(let role)? = lookup(value, "role") else {
            throw Failure.malformed("\(name): message without a role")
        }
        var content: String?
        if case .string(let text)? = lookup(value, "content") { content = text }
        var reasoningContent: String?
        if case .string(let text)? = lookup(value, "reasoning_content") { reasoningContent = text }
        var callName: String?
        if case .string(let text)? = lookup(value, "name") { callName = text }
        var toolCallID: String?
        if case .string(let text)? = lookup(value, "tool_call_id") { toolCallID = text }
        var toolCalls: [QwenToolCall] = []
        if case .array(let calls)? = lookup(value, "tool_calls") {
            for call in calls {
                guard case .string(let callID)? = lookup(call, "id"),
                      let function = lookup(call, "function"),
                      case .string(let functionName)? = lookup(function, "name") else {
                    throw Failure.malformed("\(name): malformed tool call")
                }
                let arguments = lookup(function, "arguments") ?? .object([])
                toolCalls.append(QwenToolCall(id: callID, name: functionName, arguments: arguments))
            }
        }
        return QwenChatMessage(role: role,
                               content: content,
                               reasoningContent: reasoningContent,
                               name: callName,
                               toolCallID: toolCallID,
                               toolCalls: toolCalls)
    }

    static func lookup(_ value: QwenChatValue, _ key: String) -> QwenChatValue? {
        guard case .object(let pairs) = value else { return nil }
        return pairs.first { $0.key == key }?.value
    }
}
