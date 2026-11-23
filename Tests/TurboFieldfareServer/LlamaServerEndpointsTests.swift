import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// Reports progress, then holds the generation until released, so `/slots`
/// can be read while a request occupies the slot.
private actor HeldGenerationBackend: ServerInferenceBackend {
    private var continuation: CheckedContinuation<Void, Never>?

    func release() {
        continuation?.resume()
        continuation = nil
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.prefill(done: 10, total: 10))
        onEvent(.decoded(count: 3))
        await withCheckedContinuation { continuation = $0 }
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 10, completionTokens: 3, totalTokens: 13))
    }
}

private actor ImmediateBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        ServerCompletion(content: "", toolCalls: [], finishReason: "stop",
                         usage: OpenAIUsage(promptTokens: 1, completionTokens: 0, totalTokens: 1))
    }
}

private func getJSON(_ port: Int, _ path: String) async throws -> (status: Int, json: Any) {
    let (data, response) = try await URLSession.shared.data(
        from: URL(string: "http://127.0.0.1:\(port)\(path)")!)
    let status = (response as? HTTPURLResponse)?.statusCode ?? 0
    return (status, try JSONSerialization.jsonObject(with: data))
}

private func onlySlot(_ json: Any) throws -> [String: Any] {
    let slots = try #require(json as? [[String: Any]])
    try #require(slots.count == 1)
    return slots[0]
}

@Suite("llama-server /props and /slots", .serialized)
struct LlamaServerEndpointsTests {
    @Test func propsDescribesTheModelAndItsDefaults() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ImmediateBackend(),
            visionCapability: "ready",
            metrics: ServerMetrics(maxContextTokens: 16_384),
            model: ServerModelDescription(path: "/models/test.gturbo",
                                          chatTemplate: "{{ messages }}"))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (status, json) = try await getJSON(port, "/props")
        #expect(status == 200)
        let props = try #require(json as? [String: Any])
        #expect(props["total_slots"] as? Int == 1)
        #expect(props["model_alias"] as? String == "test-model")
        #expect(props["model_path"] as? String == "/models/test.gturbo")
        #expect(props["chat_template"] as? String == "{{ messages }}")
        #expect(props["build_info"] as? String == "turbofieldfare")
        #expect(props["is_sleeping"] as? Bool == false)
        let modalities = try #require(props["modalities"] as? [String: Any])
        #expect(modalities["vision"] as? Bool == true)
        #expect(modalities["audio"] as? Bool == false)

        let settings = try #require(props["default_generation_settings"] as? [String: Any])
        #expect(settings["n_ctx"] as? Int == 16_384)
        #expect(settings["id_task"] as? Int == -1)
        #expect(settings["is_processing"] as? Bool == false)
        #expect(settings["prompt"] as? String == "")
        let next = try #require(settings["next_token"] as? [String: Any])
        #expect(next["stopping_word"] as? String == "")
        #expect(next["n_remain"] as? Int == -1)
        let params = try #require(settings["params"] as? [String: Any])
        #expect(params["temperature"] as? Double == 0.2)
        #expect(params["top_k"] as? Int == 64)
        #expect(params["top_p"] as? Double == 0.95)
        #expect(params["n_predict"] as? Int == 4_096)
        #expect(params["seed"] as? UInt64 == LlamaServerProperties.randomSeed)
        #expect(params["samplers"] as? [String] == ["penalties", "top_p", "top_k", "temperature"])

        try await server.shutdown()
    }

    /// Qwen reports its own reference sampling, the defaults the validator
    /// applies to a Qwen request that names none.
    @Test func qwenDefaultsAreTheValidatorDefaults() throws {
        let properties = LlamaServerProperties(
            modelID: "qwen", family: .qwen35Moe, contextTokens: 8_192,
            visionReady: false, model: ServerModelDescription())
        let settings = try #require(properties.props()["default_generation_settings"]
            as? [String: Any])
        let params = try #require(settings["params"] as? [String: Any])
        #expect(params["temperature"] as? Double == 1.0)
        #expect(params["top_k"] as? Int == 20)

        let request = try OpenAIRequestValidator.validate(
            OpenAIChatRequest.decodeRequest(
                from: Data(#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#.utf8),
                family: .qwen35Moe, acceptAnyModel: true),
            modelID: "qwen", family: .qwen35Moe, acceptAnyModel: true,
            preStagedImages: [:], attachmentLease: nil)
        #expect(ServerSamplingSettings(request: request) == .defaults(for: .qwen35Moe))
    }

    @Test func slotsFollowTheRunningRequest() async throws {
        let backend = HeldGenerationBackend()
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend,
            metrics: ServerMetrics(maxContextTokens: 4_096))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let idle = try onlySlot(try await getJSON(port, "/slots").json)
        #expect(idle["is_processing"] as? Bool == false)
        #expect(idle["id_task"] as? Int == -1)
        #expect(idle["n_ctx"] as? Int == 4_096)
        #expect(try await getJSON(port, "/slots?fail_on_no_slot=1").status == 200)

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],
         "temperature":0.7,"max_tokens":100,"seed":7}
        """#.utf8)
        let completion = Task { try await URLSession.shared.data(for: request) }

        var busy: [String: Any] = [:]
        for _ in 0..<250 {
            busy = try onlySlot(try await getJSON(port, "/slots").json)
            let next = busy["next_token"] as? [String: Any]
            if busy["is_processing"] as? Bool == true, next?["n_decoded"] as? Int == 3 { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(busy["is_processing"] as? Bool == true)
        #expect(busy["id_task"] as? Int == 1)
        #expect(busy["n_prompt_tokens"] as? Int == 10)
        let next = try #require(busy["next_token"] as? [String: Any])
        #expect(next["has_next_token"] as? Bool == true)
        #expect(next["n_decoded"] as? Int == 3)
        #expect(next["n_remain"] as? Int == 97)
        let params = try #require(busy["params"] as? [String: Any])
        #expect(params["temperature"] as? Double == 0.7)
        #expect(params["n_predict"] as? Int == 100)
        #expect(params["seed"] as? Int == 7)

        let refused = try await getJSON(port, "/slots?fail_on_no_slot=1")
        #expect(refused.status == 503)
        let error = try #require((refused.json as? [String: Any])?["error"] as? [String: Any])
        #expect(error["type"] as? String == "unavailable_error")

        await backend.release()
        let (_, response) = try await completion.value
        #expect((response as? HTTPURLResponse)?.statusCode == 200)

        let after = try onlySlot(try await getJSON(port, "/slots").json)
        #expect(after["is_processing"] as? Bool == false)
        #expect(after["id_task"] as? Int == -1)
        #expect(try await getJSON(port, "/slots?fail_on_no_slot=1").status == 200)

        try await server.shutdown()
    }

    @Test func propsAndSlotsRefuseOtherMethods() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: ImmediateBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        for path in ["/props", "/slots"] {
            var post = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
            post.httpMethod = "POST"
            let (_, response) = try await URLSession.shared.data(for: post)
            #expect((response as? HTTPURLResponse)?.statusCode == 405)
        }
        try await server.shutdown()
    }

    @Test func modelDescriptionReadsTheInstalledTemplate() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("props-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenizer = directory.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))

        let missing = ServerModelDescription.load(modelDirectory: directory, family: .gemma4)
        #expect(missing.chatTemplate == "")
        #expect(missing.path == directory.standardizedFileURL.path)

        try Data("{% for m in messages %}{% endfor %}".utf8)
            .write(to: tokenizer.appendingPathComponent("chat_template.jinja"))
        for family in [ModelFamily.gemma4, .qwen35Moe] {
            let loaded = ServerModelDescription.load(modelDirectory: directory, family: family)
            #expect(loaded.chatTemplate == "{% for m in messages %}{% endfor %}")
        }
    }
}
