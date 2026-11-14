import Foundation
import Testing
@testable import TurboFieldfare

/// Tail verification for multi-turn golden conversations (S10 step 4): for
/// every split after a cached assistant turn, "previous prompt + response +
/// continuation" must reproduce the next full template render.
///
/// The pinned template drops a history assistant turn's whole think block
/// (S2 report), so a *text* continuation matches the next render only after
/// that block is removed from the reconstructed text; the removed cases are
/// the annotated "template removes reasoning" ones. A *tool-result*
/// continuation needs no removal: the calling turn sits after the last user
/// query, where the template keeps the think block (and both sides of the
/// comparison re-render through `GFTokenizer.Message`, which never carries
/// `reasoning_content`, so a cached reasoning is dropped identically on both
/// sides — see `reasoning_with_tools_loop` in the S10 report).
@Suite(.enabled(if: QwenTestResources.hasTokenizer))
struct QwenChatContinuationTests {
    @Test("Multi-turn continuations reproduce the next prompt")
    func continuationsMatchTemplateTail() async throws {
        let folder = try #require(QwenTestResources.tokenizerFolder)
        let format = try QwenChatFormat.load(from: folder)
        let tokenizer = try await QwenTokenizer.load(from: folder)

        let thinkOpen = QwenTokenizer.thinkToken + "\n"
        let emptyThink = thinkOpen + "\n" + QwenTokenizer.thinkEndToken + "\n\n"
        let cachedTurnClose = QwenTokenizer.imEndToken + "\n"

        for golden in try QwenChatFixtures.goldenCases() {
            let messages = golden.messages
            let enableThinking = golden.enableThinking
            let definitions = try golden.tools?.map(Self.functionDefinition) ?? []
            let toolValues = definitions.map(Self.toolSpec)

            for split in 1..<messages.count {
                let assistant = messages[split - 1]
                guard assistant.role == "assistant" else { continue }
                let nextRole = messages[split].role

                if assistant.toolCalls.isEmpty, nextRole == "user" {
                    let (reasoning, content) = Self.templateReasoning(of: assistant)
                    let previous = try format.render(
                        messages: Array(messages[0..<split - 1]),
                        tools: toolValues.isEmpty ? nil : toolValues,
                        addGenerationPrompt: true,
                        enableThinking: enableThinking)
                    // What the model appends after the generation prompt's
                    // open think marker, and the think block the next render
                    // strips from this now-historical turn.
                    let response = reasoning + "\n" + QwenTokenizer.thinkEndToken + "\n\n" + content
                    let cache = previous + response
                    let strippedBlock = thinkOpen + reasoning + "\n"
                        + QwenTokenizer.thinkEndToken + "\n\n"
                    var stripped = cache
                    if let range = stripped.range(of: strippedBlock) {
                        stripped.replaceSubrange(range, with: "")
                    } else {
                        Issue.record("think block missing from cache text for \(golden.name)[\(split)]")
                    }
                    let next = try format.render(
                        messages: Array(messages[0...split]),
                        tools: toolValues.isEmpty ? nil : toolValues,
                        addGenerationPrompt: true,
                        enableThinking: enableThinking)
                    #expect(next.hasPrefix(stripped),
                            "reasoning-stripped continuation text differs for \(golden.name)[\(split)]")
                    let bridge = tokenizer.encodeTextContinuation(
                        userContent: messages[split].content ?? "",
                        enableThinking: enableThinking)
                    #expect(bridge.first == tokenizer.imEndID,
                            "text continuation must re-emit the dropped stop token for \(golden.name)[\(split)]")
                    let expected = tokenizer.encode(stripped, addBOS: false) + bridge
                    #expect(tokenizer.encode(next, addBOS: false) == expected,
                            "text continuation tokens differ for \(golden.name)[\(split)]")
                } else if !assistant.toolCalls.isEmpty, nextRole == "tool" {
                    guard !definitions.isEmpty else {
                        Issue.record("tool-result continuation golden without tools: \(golden.name)")
                        continue
                    }
                    let previous = try format.render(
                        messages: Array(messages[0..<split - 1]),
                        tools: toolValues,
                        addGenerationPrompt: true,
                        enableThinking: enableThinking)
                    // The assistant turn sits after the last user query, so
                    // a no-generation-prompt render keeps its think block:
                    // everything after the marker is the generated response,
                    // including the tool-call XML.
                    let history = try format.render(
                        messages: Array(messages[0...split - 1]),
                        tools: toolValues,
                        addGenerationPrompt: false,
                        enableThinking: enableThinking)
                    let marker = enableThinking ? thinkOpen : emptyThink
                    let markerRange = try #require(history.range(of: marker, options: .backwards))
                    let cache = previous + String(history[markerRange.upperBound...])
                    let next = try format.render(
                        messages: messages,
                        tools: toolValues,
                        addGenerationPrompt: true,
                        enableThinking: enableThinking)
                    #expect(next.hasPrefix(cache),
                            "tool-result cache is not a prefix of the next prompt for \(golden.name)[\(split)]")
                    guard cache.hasSuffix(cachedTurnClose) else {
                        Issue.record("tool-result cache does not close with the stop token for \(golden.name)[\(split)]")
                        continue
                    }
                    // The runtime stops at the stop token and drops it; the
                    // continuation re-emits it.
                    let kvText = String(cache.dropLast(cachedTurnClose.count))
                    let bridge: [Int32]
                    do {
                        bridge = try tokenizer.encodeToolResultContinuation(
                            cachedMessages: messages[0..<split - 1].map(Self.gfMessage),
                            assistant: Self.gfMessage(assistant),
                            incomingMessages: messages.map(Self.gfMessage),
                            tools: definitions,
                            enableThinking: enableThinking)
                    } catch {
                        Issue.record("encodeToolResultContinuation failed for \(golden.name)[\(split)]: \(error)")
                        continue
                    }
                    #expect(bridge.first == tokenizer.imEndID,
                            "tool-result continuation must re-emit the dropped stop token for \(golden.name)[\(split)]")
                    let expected = tokenizer.encode(kvText, addBOS: false) + bridge
                    #expect(tokenizer.encode(next, addBOS: false) == expected,
                            "tool-result continuation tokens differ for \(golden.name)[\(split)]")
                }
            }
        }
    }

    // MARK: - Fixture mapping helpers

    /// The reasoning/content split the pinned template performs itself for a
    /// message without a `reasoning_content` field.
    private static func templateReasoning(of message: QwenChatMessage) -> (String, String) {
        let content = message.content ?? ""
        if let reasoning = message.reasoningContent {
            return (reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
                    content.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let thinkEnd = QwenTokenizer.thinkEndToken
        if content.contains(thinkEnd) {
            let reasoning = content.components(separatedBy: thinkEnd).first?
                .components(separatedBy: QwenTokenizer.thinkToken).last ?? ""
            let visible = content.components(separatedBy: thinkEnd).last ?? ""
            return (reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
                    visible.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return ("", content.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private static func gfMessage(_ message: QwenChatMessage) throws -> GFTokenizer.Message {
        let role = try #require(GFTokenizer.Role(rawValue: message.role))
        return GFTokenizer.Message(
            role: role,
            content: message.content,
            toolCalls: message.toolCalls.map {
                GFTokenizer.HistoricalToolCall(id: $0.id, name: $0.name, arguments: jsonValue($0.arguments))
            },
            toolCallID: message.toolCallID,
            name: message.name)
    }

    private static func functionDefinition(_ tool: QwenChatValue) throws -> GFTokenizer.FunctionDefinition {
        guard let function = QwenChatFixtures.lookup(tool, "function"),
              case .string(let name)? = QwenChatFixtures.lookup(function, "name"),
              case .string(let description)? = QwenChatFixtures.lookup(function, "description") else {
            throw QwenChatFixtures.Failure.malformed("tool without a function name/description")
        }
        return GFTokenizer.FunctionDefinition(
            name: name,
            description: description,
            parameters: jsonValue(QwenChatFixtures.lookup(function, "parameters") ?? .object([])))
    }

    /// Tool schema in the shape the template consumes, mirroring the
    /// tokenizer's internal mapping so both render with one dictionary order.
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

    private static func jsonValue(_ value: QwenChatValue) -> JSONValue {
        switch value {
        case .null: return .null
        case .bool(let flag): return .bool(flag)
        case .int(let number): return .integer(number)
        case .double(let number): return .number(number)
        case .string(let text): return .string(text)
        case .array(let values): return .array(values.map(jsonValue))
        case .object(let pairs):
            return .object(Dictionary(uniqueKeysWithValues: pairs.map { ($0.key, jsonValue($0.value)) }))
        }
    }
}
