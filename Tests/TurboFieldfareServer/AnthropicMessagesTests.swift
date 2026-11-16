import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

private actor MessagesBackend: ServerInferenceBackend {
    let completion: ServerCompletion
    let events: [ServerInferenceEvent]

    init(content: String = "hello",
         toolCalls: [ParsedToolCall] = [],
         finishReason: String = "stop") {
        completion = ServerCompletion(
            content: content,
            toolCalls: toolCalls,
            finishReason: finishReason,
            usage: OpenAIUsage(promptTokens: 30, completionTokens: 5, totalTokens: 35,
                               cachedTokens: 20))
        events = (content.isEmpty ? [] : [.content(content)]) + toolCalls.map { .toolCall($0) }
    }

    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        ServerPreparedRequest(request: request, promptIDs: [1, 2, 3, 4, 5, 6, 7])
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        for event in events { onEvent(event) }
        return completion
    }
}

private func readCall() -> ParsedToolCall {
    ParsedToolCall(id: "call_000000000000000000000001", name: "Read",
                   arguments: .object(["file_path": .string("/tmp/a")]),
                   argumentsJSON: #"{"file_path":"/tmp/a"}"#)
}

/// The Messages API on top of the chat pipeline: requests translate into
/// chat requests the validator accepts, and answers come back in Anthropic's
/// shapes.
@Suite("Anthropic Messages", .serialized)
struct AnthropicMessagesTests {
    // MARK: Translation

    private static func translate(_ body: String,
                                  family: ModelFamily = .gemma4,
                                  counting: Bool = false) throws -> OpenAIChatRequest {
        let json = try AnthropicMessages.chatRequestJSON(from: Data(body.utf8), family: family,
                                                         counting: counting)
        return try OpenAIChatRequest.decodeRequest(from: json, family: family, acceptAnyModel: true)
    }

    private static func validated(_ body: String,
                                  family: ModelFamily = .gemma4) throws -> ValidatedChatRequest {
        try OpenAIRequestValidator.validate(try translate(body, family: family),
                                            modelID: "m", family: family, acceptAnyModel: true)
    }

    private static func error(_ body: String, counting: Bool = false) -> ServerRequestError? {
        do {
            _ = try AnthropicMessages.chatRequestJSON(from: Data(body.utf8), family: .gemma4,
                                                      counting: counting)
            return nil
        } catch {
            return error as? ServerRequestError
        }
    }

    @Test func systemBlocksAndUserTextBecomeChatMessages() throws {
        let request = try Self.validated(#"""
        {"model":"claude-sonnet","max_tokens":64000,"stream":true,
         "system":[{"type":"text","text":"You are a coding agent.","cache_control":{"type":"ephemeral"}},
                   {"type":"text","text":"Be brief."}],
         "messages":[{"role":"user","content":[
            {"type":"text","text":"<system-reminder>context</system-reminder>"},
            {"type":"text","text":"Fix the bug.","cache_control":{"type":"ephemeral"}}]}],
         "metadata":{"user_id":"u"},"temperature":1}
        """#)
        #expect(request.messages.count == 2)
        #expect(request.messages[0].role == .system)
        #expect(request.messages[0].content == "You are a coding agent.\n\nBe brief.")
        #expect(request.messages[1].role == .user)
        #expect(request.messages[1].content
            == "<system-reminder>context</system-reminder>\n\nFix the bug.")
        #expect(request.stream)
        #expect(request.maximumCompletionTokens == 64_000)
        #expect(request.generationConfig.temperature == 1)
    }

    /// A tool round trip as Claude Code sends it: the assistant's call, then
    /// a user turn carrying the result and more text.
    @Test func toolUseAndToolResultBecomeACallAndAToolMessage() throws {
        let request = try Self.validated(#"""
        {"model":"m","max_tokens":100,
         "tools":[{"name":"Read","description":"Reads a file",
                   "input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},
                                   "required":["file_path"],"additionalProperties":false,
                                   "$schema":"http://json-schema.org/draft-07/schema#"}}],
         "messages":[
          {"role":"user","content":"Read /tmp/a"},
          {"role":"assistant","content":[
             {"type":"thinking","thinking":"hmm","signature":"x"},
             {"type":"text","text":"Reading it."},
             {"type":"tool_use","id":"call_1","name":"Read","input":{"file_path":"/tmp/a"}}]},
          {"role":"user","content":[
             {"type":"tool_result","tool_use_id":"call_1",
              "content":[{"type":"text","text":"line 1"},{"type":"image","source":{}}]},
             {"type":"text","text":"Now summarize."}]}]}
        """#)
        #expect(request.tools.map(\.name) == ["Read"])
        let roles = request.messages.map(\.role)
        #expect(roles == [.user, .assistant, .tool, .user])
        #expect(request.messages[1].content == "Reading it.")
        #expect(request.messages[1].toolCalls.map(\.id) == ["call_1"])
        #expect(request.messages[1].toolCalls.first?.arguments
            == .object(["file_path": .string("/tmp/a")]))
        #expect(request.messages[2].toolCallID == "call_1")
        #expect(request.messages[2].content
            == "line 1\n" + AnthropicMessages.toolResultImagePlaceholder)
        #expect(request.messages[3].content == "Now summarize.")
    }

    @Test func aTurnOfOnlyToolResultsAddsNoEmptyUserMessage() throws {
        let request = try Self.validated(#"""
        {"model":"m","max_tokens":10,"messages":[
          {"role":"user","content":"go"},
          {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]},
          {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}
        """#)
        #expect(request.messages.map(\.role) == [.user, .assistant, .tool])
        #expect(request.messages[1].content == nil)
    }

    @Test func maxTokensIsRequiredExceptForCounting() throws {
        let body = #"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#
        #expect(Self.error(body)?.envelope.error.message == #"missing required field "max_tokens""#)
        #expect(Self.error(body)?.envelope.error.param == "max_tokens")
        #expect(Self.error(body, counting: true) == nil)
    }

    @Test(arguments: [
        (#"{"max_tokens":1,"messages":[{"role":"system","content":"x"}]}"#, "messages[0].role"),
        (#"{"max_tokens":1,"messages":[{"role":"user","content":[{"type":"search_result"}]}]}"#,
         "messages[0].content[0].type"),
        (#"{"max_tokens":0,"messages":[]}"#, "max_tokens"),
        (#"{"max_tokens":1,"messages":[],"tools":[{"type":"web_search_20250305","name":"w"}]}"#,
         "tools[0].type"),
    ])
    func refusalsNameTheAnthropicField(body: String, param: String) {
        #expect(Self.error(body)?.envelope.error.param == param)
    }

    @Test func toolChoiceThatForcesACallRunsAsAuto() throws {
        for type in ["auto", "any", "tool"] {
            let request = try Self.translate(#"""
            {"max_tokens":5,"messages":[{"role":"user","content":"x"}],
             "tool_choice":{"type":"\#(type)","name":"Read"}}
            """#)
            #expect(request.toolChoice == .string("auto"))
        }
        let none = try Self.translate(#"""
        {"max_tokens":5,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"none"}}
        """#)
        #expect(none.toolChoice == .string("none"))
    }

    @Test func thinkingMapsToQwenControlsAndIsIgnoredForGemma() throws {
        func qwen(_ thinking: String) throws -> ValidatedChatRequest {
            try Self.validated(#"""
            {"max_tokens":5,"messages":[{"role":"user","content":"x"}],"thinking":\#(thinking)}
            """#, family: .qwen35Moe)
        }
        #expect(try qwen(#"{"type":"disabled"}"#).enableThinking == false)
        let medium = try qwen(#"{"type":"enabled","budget_tokens":1000}"#)
        #expect(medium.enableThinking)
        #expect(medium.reasoningBudgetTokens == ReasoningDepth.medium.budgetTokens)
        #expect(try qwen(#"{"type":"enabled","budget_tokens":31999}"#).reasoningBudgetTokens == nil)
        // Gemma's chat request refuses both fields; the translation leaves
        // them out instead of failing the request.
        _ = try Self.validated(#"""
        {"max_tokens":5,"messages":[{"role":"user","content":"x"}],
         "thinking":{"type":"enabled","budget_tokens":2000}}
        """#)
    }

    @Test func imageBlocksBecomeDataURLParts() throws {
        let request = try Self.translate(#"""
        {"max_tokens":5,"messages":[{"role":"user","content":[
          {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},
          {"type":"text","text":"What is this?"}]}]}
        """#)
        guard case .parts(let parts)? = request.messages.first?.content else {
            Issue.record("expected content parts")
            return
        }
        #expect(parts.map(\.type) == ["text", "image_url"])
        #expect(parts.last?.imageURL?.url == "data:image/png;base64,AAAA")
    }

    // MARK: Tool schemas

    /// Schemas shaped like Claude Code's own tools, which use JSON Schema
    /// features the chat templates cannot render.
    @Test(arguments: [
        // Typed throughout, with keywords the reduction must leave alone.
        #"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"command":{"type":"string","description":"The command"},"timeout":{"type":"number","maximum":600000},"run_in_background":{"type":"boolean"}},"required":["command"],"additionalProperties":false}"#,
        // A union of an array and an object, and an untyped property.
        #"{"type":"object","properties":{"files":{"anyOf":[{"type":"array","items":{"type":"object","properties":{"path":{"type":"string"}}}},{"type":"object","additionalProperties":{"type":"string"}}]},"payload":{"description":"anything"}}}"#,
        // A string union that mixes a constant with a pattern.
        #"{"type":"object","properties":{"contract":{"anyOf":[{"type":"string","const":"latest"},{"type":"string","pattern":"^[0-9]+$"}]},"limit":{"type":["integer","null"]}}}"#,
        // Nested arrays of objects with min and max items and propertyNames.
        #"{"type":"object","properties":{"questions":{"type":"array","minItems":1,"maxItems":4,"items":{"type":"object","properties":{"options":{"type":"array","items":{"type":"object","properties":{"label":{"type":"string"}},"required":["label"]}}}}},"answers":{"type":"object","propertyNames":{"type":"string"},"additionalProperties":{"type":"string"}}}}"#,
    ])
    func claudeCodeShapedSchemasReduceToRenderableOnes(schema: String) throws {
        let value = try JSONDecoder().decode(JSONValue.self, from: Data(schema.utf8))
        let reduced = AnthropicMessages.toolParameters(value, name: "Tool", description: "d")
        // Reduced, not replaced by the fallback.
        #expect(reduced.description == "d")
        _ = try OpenAIRequestValidator.validateTool(OpenAITool(
            type: "function",
            function: OpenAIFunctionDefinition(name: "Tool", description: reduced.description,
                                               parameters: reduced.parameters)))
    }

    /// A schema the reduction cannot make renderable still reaches the model:
    /// the tool takes an open object and the description carries the schema.
    @Test func anUnrenderableSchemaFallsBackToTheDescription() throws {
        let schema = try JSONDecoder().decode(JSONValue.self, from: Data(
            #"{"type":"object","properties":{"bad key!":{"type":"string"}}}"#.utf8))
        let result = AnthropicMessages.toolParameters(schema, name: "Odd", description: "Does odd things.")
        #expect(result.parameters == .object(["type": .string("object"), "properties": .object([:])]))
        let description = try #require(result.description)
        #expect(description.hasPrefix("Does odd things.\n\nInput JSON Schema: "))
        #expect(description.contains("bad key!"))
    }

    // MARK: Responses

    @Test func messageCarriesBlocksStopReasonAndSplitUsage() throws {
        let completion = ServerCompletion(
            content: "Reading.", toolCalls: [readCall()], finishReason: "tool_calls",
            usage: OpenAIUsage(promptTokens: 100, completionTokens: 9, totalTokens: 109,
                               cachedTokens: 60))
        let object = AnthropicMessages.message(id: "msg_1", model: "m", completion: completion)
        #expect(object["type"] as? String == "message")
        #expect(object["stop_reason"] as? String == "tool_use")
        let content = try #require(object["content"] as? [[String: Any]])
        #expect(content.map { $0["type"] as? String } == ["text", "tool_use"])
        #expect((content[1]["input"] as? [String: Any])?["file_path"] as? String == "/tmp/a")
        let usage = try #require(object["usage"] as? [String: Int])
        #expect(usage == ["input_tokens": 40, "output_tokens": 9,
                          "cache_read_input_tokens": 60, "cache_creation_input_tokens": 0])
        #expect(AnthropicMessages.stopReason("length") == "max_tokens")
        #expect(AnthropicMessages.stopReason("stop") == "end_turn")
    }

    @Test func streamEventsFollowAnthropicsOrder() {
        let encoder = AnthropicStreamEncoder(id: "msg_1", model: "m")
        let completion = ServerCompletion(
            content: "Hi there", toolCalls: [readCall()], finishReason: "tool_calls",
            usage: OpenAIUsage(promptTokens: 10, completionTokens: 4, totalTokens: 14))
        let stream = encoder.messageStart(inputTokens: 10)
            + encoder.text("Hi") + encoder.text("") + encoder.text(" there")
            + encoder.toolCall(readCall())
            + encoder.finish(completion)
        let events = stream.split(separator: "\n").filter { $0.hasPrefix("event: ") }
            .map { String($0.dropFirst(7)) }
        #expect(events == [
            "message_start",
            "content_block_start", "content_block_delta", "content_block_delta",
            "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop",
        ])
        #expect(stream.contains(#""index":0,"type":"content_block_stop""#))
        #expect(stream.contains(#""partial_json":"{\"file_path\":\"\/tmp\/a\"}""#)
            || stream.contains(#""partial_json":"{\"file_path\":\"/tmp/a\"}""#))
        #expect(stream.contains(#""stop_reason":"tool_use""#))
        #expect(stream.hasSuffix("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
    }

    // MARK: Over HTTP

    private static func post(_ port: Int, _ path: String, _ body: String,
                             headers: [String: String] = [:]) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        request.httpBody = Data(body.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, try #require(response as? HTTPURLResponse))
    }

    private static func server(_ backend: MessagesBackend = MessagesBackend(),
                               acceptAnyModel: Bool = true) async throws -> (TurboFieldfareHTTPServer, Int) {
        let server = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1,
                                              backend: backend, acceptAnyModel: acceptAnyModel)
        let channel = try await server.start(port: 0)
        return (server, try #require(channel.localAddress?.port))
    }

    @Test func messagesAnswersInAnthropicShape() async throws {
        let (server, port) = try await Self.server()
        let (data, response) = try await Self.post(port, "/v1/messages?beta=true", #"""
        {"model":"claude-sonnet-4","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}
        """#, headers: ["anthropic-version": "2023-06-01", "x-api-key": "local"])
        #expect(response.statusCode == 200)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["type"] as? String == "message")
        #expect(object["model"] as? String == "test-model")
        #expect((object["id"] as? String)?.hasPrefix("msg_") == true)
        #expect(object["stop_reason"] as? String == "end_turn")
        let content = try #require(object["content"] as? [[String: Any]])
        #expect(content.first?["text"] as? String == "hello")
        try await server.shutdown()
    }

    @Test func messagesStreamsServerSentEvents() async throws {
        let (server, port) = try await Self.server(MessagesBackend(toolCalls: [readCall()],
                                                                   finishReason: "tool_calls"))
        let (data, response) = try await Self.post(port, "/v1/messages", #"""
        {"max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}
        """#)
        #expect(response.statusCode == 200)
        #expect(response.value(forHTTPHeaderField: "content-type") == "text/event-stream")
        let text = String(decoding: data, as: UTF8.self)
        let events = text.split(separator: "\n").filter { $0.hasPrefix("event: ") }
            .map { String($0.dropFirst(7)) }
        #expect(events.first == "message_start")
        #expect(events.contains("ping"))
        #expect(events.suffix(2) == ["message_delta", "message_stop"])
        #expect(text.contains(#""text":"hello""#))
        #expect(text.contains(#""stop_reason":"tool_use""#))
        // The prepared prompt's size goes out in message_start.
        #expect(text.contains(#""input_tokens":7"#))
        try await server.shutdown()
    }

    @Test func refusalsUseAnthropicErrors() async throws {
        let (server, port) = try await Self.server(acceptAnyModel: false)
        let (data, response) = try await Self.post(port, "/v1/messages", #"""
        {"model":"claude-sonnet-4","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}
        """#)
        #expect(response.statusCode == 404)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["type"] as? String == "error")
        #expect((object["error"] as? [String: Any])?["type"] as? String == "not_found_error")

        let (missing, missingResponse) = try await Self.post(port, "/v1/messages", #"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#)
        #expect(missingResponse.statusCode == 400)
        let error = try #require(JSONSerialization.jsonObject(with: missing) as? [String: Any])
        let detail = try #require(error["error"] as? [String: Any])
        #expect(detail["type"] as? String == "invalid_request_error")
        #expect(detail["message"] as? String == #"missing required field "max_tokens""#)
        try await server.shutdown()
    }

    @Test func countTokensRendersWithoutGenerating() async throws {
        let (server, port) = try await Self.server()
        let (data, response) = try await Self.post(port, "/v1/messages/count_tokens", #"""
        {"model":"m","messages":[{"role":"user","content":"hi"}]}
        """#)
        #expect(response.statusCode == 200)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Int])
        #expect(object == ["input_tokens": 7])
        try await server.shutdown()
    }

    @Test func modelListFollowsTheAskingAPI() async throws {
        let (server, port) = try await Self.server()
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/models")!)
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        let anthropic = try JSONSerialization.jsonObject(
            with: try await URLSession.shared.data(for: request).0) as? [String: Any]
        #expect(anthropic?["has_more"] as? Bool == false)
        #expect(((anthropic?["data"] as? [[String: Any]])?.first)?["type"] as? String == "model")
        let openAI = try JSONSerialization.jsonObject(
            with: try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0) as? [String: Any]
        #expect(openAI?["object"] as? String == "list")
        try await server.shutdown()
    }
}
