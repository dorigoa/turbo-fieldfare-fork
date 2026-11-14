import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// S23 step 5: a scripted decode through the Qwen branch of the server.
///
/// The logits are scripted so the decode emits a chosen assistant text,
/// tokenized by the pinned Qwen tokenizer. Each test validates a real request
/// with the Qwen family, then runs the same sequence as the Qwen branch of
/// `ServerModelSession.generate`: `runRawCompletion` through `any
/// ChatTokenizer`, every token delta and the tail into a
/// `QwenAssistantDecoder` built from the request's tools and thinking flag,
/// `finish()`, and the finish reason rule. The server test target cannot see
/// `TurboFieldfareValidationSupport`, so the scripted producer is local.
@Suite("Server Qwen scripted generation",
       .enabled(if: QwenServerTestResources.hasTokenizer))
struct ServerQwenScriptedGenerationTests {
    // Chat markers written with an escaped first character (2.3).
    private static let thinkEnd = "\u{3C}/think>"
    private static let callOpen = "\u{3C}tool_call>"
    private static let callClose = "\u{3C}/tool_call>"

    private static let weatherTool =
        #"[{"type":"function","function":{"name":"get_weather","description":"Current weather","#
        + #""parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#

    private let tokenizer: QwenTokenizer

    init() async throws {
        tokenizer = try await QwenTokenizer.load(
            from: try #require(QwenServerTestResources.tokenizerFolder))
    }

    private struct Outcome {
        var content = ""
        var calls: [ParsedToolCall] = []
        var rawReason: StopReason?
        var finishReason = ""
    }

    private func validated(_ fields: String = "") throws -> ValidatedChatRequest {
        let extra = fields.isEmpty ? "" : "," + fields
        let body = #"{"model":"m","messages":[{"role":"user","content":"x"}]"# + extra + "}"
        let decoder = JSONDecoder()
        decoder.userInfo[OpenAIChatRequest.modelFamilyUserInfoKey] = ModelFamily.qwen35Moe
        return try OpenAIRequestValidator.validate(
            decoder.decode(OpenAIChatRequest.self, from: Data(body.utf8)),
            modelID: "m",
            family: .qwen35Moe)
    }

    /// The Qwen branch of `ServerModelSession.generate`, from the prompt to
    /// the finish reason, with the decode scripted to emit `assistantText`
    /// followed by `im_end`.
    private func run(_ request: ValidatedChatRequest,
                     assistantText: String) async throws -> Outcome {
        let chatTokenizer: any ChatTokenizer = tokenizer
        let promptIds = chatTokenizer.encode("go", addBOS: false)
        let script = chatTokenizer.encode(assistantText, addBOS: false) + [tokenizer.imEndID]
        let producer = ScriptedQwenProducer(vocabSize: tokenizer.vocabSize,
                                            promptCount: promptIds.count,
                                            script: script,
                                            fallback: tokenizer.imEndID)
        let decoder = QwenAssistantDecoder(
            allowedTools: Set(request.tools.map(\.name)),
            insideReasoning: request.enableThinking)
        var config = request.generationConfig
        config.maxNewTokens = script.count + 4
        config.stopStrings = []
        let context = try MetalContext()
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        var outcome = Outcome()
        var decodingError: Error?
        let result = try await runRawCompletion(
            producer: producer, tokenizer: chatTokenizer, promptIds: promptIds,
            config: config, context: context, scratch: scratch,
            prefillConfig: .off) { progress in
                guard decodingError == nil else { return }
                do {
                    let events: [StructuredAssistantEvent]
                    switch progress {
                    case .prefill:
                        events = []
                    case .token(_, _, let delta):
                        events = try decoder.consume(delta)
                    case .tail(let text):
                        events = text.isEmpty ? [] : try decoder.consume(text)
                    }
                    for event in events {
                        switch event {
                        case .content(let text): outcome.content += text
                        case .toolCall(let call): outcome.calls.append(call)
                        }
                    }
                } catch {
                    decodingError = error
                }
            }
        if let decodingError { throw decodingError }
        for event in try decoder.finish() {
            switch event {
            case .content(let text): outcome.content += text
            case .toolCall(let call): outcome.calls.append(call)
            }
        }
        outcome.rawReason = result.reason
        // The finish reason rule of `generate`.
        if !outcome.calls.isEmpty {
            outcome.finishReason = "tool_calls"
        } else if result.reason == .maxTokens {
            outcome.finishReason = "length"
        } else {
            outcome.finishReason = "stop"
        }
        return outcome
    }

    /// The parser's markers are the vocabulary's own `tool_call` tokens: a
    /// marker the checkpoint never emits would let every real tool call
    /// through as content.
    @Test func parserMarkersAreTheVocabularyToolCallTokens() {
        #expect(QwenToolCallParser.callOpen == Self.callOpen)
        #expect(QwenToolCallParser.callClose == Self.callClose)
        #expect(tokenizer.encode(QwenToolCallParser.callOpen, addBOS: false)
                == [tokenizer.toolCallStartID])
        #expect(tokenizer.encode(QwenToolCallParser.callClose, addBOS: false)
                == [tokenizer.toolCallEndID])
    }

    @Test func reasoningStaysOutOfContent() async throws {
        let request = try validated()
        #expect(request.enableThinking)
        let outcome = try await run(
            request,
            assistantText: "The user wants the capital of France.\n"
                + Self.thinkEnd + "\n\nParis is the capital.")
        #expect(outcome.content.trimmingCharacters(in: .whitespacesAndNewlines)
                == "Paris is the capital.")
        #expect(!outcome.content.contains("capital of France"))
        #expect(!outcome.content.contains("think>"))
        #expect(outcome.calls.isEmpty)
        #expect(outcome.rawReason == .endOfTurn)
        #expect(outcome.finishReason == "stop")
    }

    @Test func xmlToolCallBecomesAnOpenAIToolCall() async throws {
        let request = try validated(#""tools":"# + Self.weatherTool)
        let outcome = try await run(
            request,
            assistantText: "I should ask the weather tool.\n" + Self.thinkEnd + "\n\n"
                + Self.callOpen + "\n\u{3C}function=get_weather>\n\u{3C}parameter=city>\nParis\n"
                + "\u{3C}/parameter>\n\u{3C}/function>\n" + Self.callClose)
        let call = try #require(outcome.calls.first)
        #expect(outcome.calls.count == 1)
        #expect(call.name == "get_weather")
        #expect(call.arguments == .object(["city": .string("Paris")]))
        let arguments = try JSONSerialization.jsonObject(with: Data(call.argumentsJSON.utf8))
        #expect((arguments as? [String: String]) == ["city": "Paris"])
        #expect(outcome.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!outcome.content.contains("weather tool"))
        // Qwen closes a tool turn on im_end: no tool-result wait token.
        #expect(outcome.rawReason == .endOfTurn)
        #expect(outcome.finishReason == "tool_calls")
    }

    @Test func thinkingOffPassesContentThrough() async throws {
        let request = try validated(#""chat_template_kwargs":{"enable_thinking":false}"#)
        #expect(!request.enableThinking)
        let outcome = try await run(request, assistantText: "Paris.")
        #expect(outcome.content.trimmingCharacters(in: .whitespacesAndNewlines) == "Paris.")
        #expect(outcome.calls.isEmpty)
        #expect(outcome.finishReason == "stop")
    }

    @Test func undeclaredToolFailsClosed() async throws {
        let request = try validated(#""tools":"# + Self.weatherTool)
        await #expect(throws: (any Error).self) {
            _ = try await run(
                request,
                assistantText: "Calling.\n" + Self.thinkEnd + "\n\n" + Self.callOpen
                    + "\n\u{3C}function=delete_files>\n\u{3C}parameter=path>\n/\n"
                    + "\u{3C}/parameter>\n\u{3C}/function>\n" + Self.callClose)
        }
    }
}

/// Logits that make the decode emit `script`: the call after the last
/// prompt token picks `script[0]`, each later call the next entry, then
/// `fallback`. Prefill runs token by token (`prefillConfig: .off`), so the
/// call count lines up with the prompt length.
private final class ScriptedQwenProducer: LogitProducer, @unchecked Sendable {
    private let vocabSize: Int
    private let promptCount: Int
    private let script: [Int32]
    private let fallback: Int32
    private var calls = 0

    init(vocabSize: Int, promptCount: Int, script: [Int32], fallback: Int32) {
        self.vocabSize = vocabSize
        self.promptCount = promptCount
        self.script = script
        self.fallback = fallback
    }

    func reset() { calls = 0 }

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        let step = calls - (promptCount - 1)
        calls += 1
        let next = step >= 0 && step < script.count ? script[step] : fallback
        let values = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
        for index in 0..<vocabSize { values[index] = Float16(-30.0) }
        values[Int(next)] = Float16(30.0)
    }
}
