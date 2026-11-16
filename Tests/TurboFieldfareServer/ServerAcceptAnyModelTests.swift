import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

private actor EchoBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
    }
}

/// `--accept-any-model` serves whatever `model` a request names, or none, as
/// llama-server does. Without it the name must match, as it always had to.
@Suite("Accept any model", .serialized)
struct ServerAcceptAnyModelTests {
    private static func parse(_ extra: [String]) throws -> ServerArguments {
        try ServerArguments.parse(["--model", "model.gturbo"] + extra,
                                  hostMemoryBytes: 64 << 30, environment: [:],
                                  readModelVariant: { _ in nil })
    }

    @Test func flagIsOffByDefaultAndTakesNoValue() throws {
        #expect(try Self.parse([]).acceptAnyModel == false)
        let first = try Self.parse(["--accept-any-model", "--port", "8181"])
        #expect(first.acceptAnyModel)
        #expect(first.port == 8181)
        let last = try Self.parse(["--port", "8181", "--accept-any-model"])
        #expect(last.acceptAnyModel)
        #expect(last.port == 8181)
        #expect(ServerArguments.usage.contains("--accept-any-model"))
    }

    @Test func decodingAcceptsAMissingModelOnlyUnderTheFlag() throws {
        let body = Data(#"{"messages":[{"role":"user","content":"hi"}]}"#.utf8)
        #expect(throws: ServerRequestError.self) {
            _ = try OpenAIChatRequest.decodeRequest(from: body, family: .gemma4)
        }
        let request = try OpenAIChatRequest.decodeRequest(
            from: body, family: .gemma4, acceptAnyModel: true)
        #expect(request.model == "")
        // A model that is there must still be a string.
        #expect(throws: ServerRequestError.invalid(
            message: #""model" must be a string"#, param: "model", code: "invalid_json")) {
            _ = try OpenAIChatRequest.decodeRequest(
                from: Data(#"{"model":7,"messages":[]}"#.utf8),
                family: .gemma4, acceptAnyModel: true)
        }
    }

    @Test func validatorSkipsTheNameCheckOnlyUnderTheFlag() throws {
        let request = try OpenAIChatRequest.decodeRequest(
            from: Data(#"{"model":"other","messages":[{"role":"user","content":"hi"}]}"#.utf8),
            family: .gemma4)
        #expect(throws: ServerRequestError.unknownModel) {
            _ = try OpenAIRequestValidator.validate(request, modelID: "served")
        }
        _ = try OpenAIRequestValidator.validate(request, modelID: "served", acceptAnyModel: true)
    }

    @Test(arguments: [
        #"{"messages":[{"role":"user","content":"hi"}]}"#,
        #"{"model":"default_model","messages":[{"role":"user","content":"hi"}]}"#,
        #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#,
    ])
    func serverAnswersAnyModelAndNamesItsOwn(body: String) async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: EchoBackend(),
            acceptAnyModel: true)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(body.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["model"] as? String == "test-model")
        try await server.shutdown()
    }

    @Test func serverStillRefusesOtherModelsByDefault() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: EchoBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"{"model":"default_model","messages":[{"role":"user","content":"hi"}]}"#.utf8)
        let (_, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 404)
        try await server.shutdown()
    }
}
