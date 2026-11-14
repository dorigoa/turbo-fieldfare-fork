import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// S23 step 4 (D12): images with Qwen fail closed at both ends the server
/// can see them - an explicit `--vision-pack` is a startup mistake for a
/// Qwen pack, and a request carrying an image is refused with the plan's
/// 400 before any token or GPU work. An image is never accepted and then
/// ignored.
@Suite("Server Qwen image refusal")
struct ServerQwenImageRefusalTests {

    /// The check runs on the manifest family alone, before the tokenizer or
    /// the weights open, so a manifest-only directory is enough to prove it
    /// and nothing heavy is loaded.
    @Test func explicitVisionPackWithQwenRefusesStartup() async throws {
        let directory = try ServerQwenModelIdentityTests.writeQwenManifestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let ghostPack = directory.appendingPathComponent("ghost.vision.gturbo")
        do {
            _ = try await ServerModelSession.load(
                modelDirectory: directory,
                maxContext: 4096,
                visionPackURL: ghostPack,
                runtimeConfiguration: .production)
            Issue.record("a Qwen pack accepted an explicit vision pack")
        } catch let error as ServerModelSessionError {
            guard case .visionUnsupportedForQwen(let path) = error else {
                Issue.record("expected visionUnsupportedForQwen, got \(error)")
                return
            }
            // The refusal names the path that caused it: an operator's typo
            // must fail loudly at startup, not serve text without vision.
            #expect(path == ghostPack.path)
            #expect(error.description.contains("vision is not available for this model"))
        }
    }

    /// The wire contract of the session's image guard: the exact D12 message
    /// reaches the client as a 400 with `image_not_supported` on `messages`.
    /// The session's own refusal of a real Qwen request is proven with the
    /// installed model in the S23 report (curl proof).
    @Test func imageRefusalAnswersFourHundredWithTheD12Body() async throws {
        let modelID = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let server = TurboFieldfareHTTPServer(
            modelID: modelID,
            queueLimit: 1,
            backend: QwenImageRefusingBackend(),
            family: .qwen35Moe)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": modelID,
            "messages": [["role": "user", "content": "describe the picture"]],
        ])
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        let body = String(decoding: data, as: UTF8.self)
        #expect(body.contains("image input is not available for this model"))
        #expect(body.contains("image_not_supported"))
        #expect(body.contains(#""param":"messages""#))
        try await server.shutdown()
    }

    /// A real image reaches the streaming body parser first, which refuses
    /// images before staging any byte. Under Qwen it must answer with the
    /// D12 body too, not the companion-pack message of Gemma: the S23 curl
    /// proof caught that gap.
    @Test func streamedImageIsRefusedWithTheD12Body() async throws {
        let modelID = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let server = TurboFieldfareHTTPServer(
            modelID: modelID,
            queueLimit: 1,
            backend: QwenImageRefusingBackend(),
            family: .qwen35Moe)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            (#"{"model":"\#(modelID)","messages":[{"role":"user","content":["#
             + #"{"type":"text","text":"what is this?"},"#
             + #"{"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}}]}]}"#).utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        let body = String(decoding: data, as: UTF8.self)
        #expect(body.contains("image input is not available for this model"))
        #expect(body.contains("image_not_supported"))
        #expect(!body.contains("vision_unavailable"))
        try await server.shutdown()
    }

    /// Gemma keeps its companion-pack refusal from the same parser.
    @Test func gemmaStreamedImageKeepsTheVisionMessage() throws {
        let parser = StreamingChatRequestBody(visionCapability: "missing")
        var buffer = ByteBuffer(bytes: Data(
            #"{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}}]}]}"#.utf8))
        #expect(throws: ServerRequestError.invalid(
            message: "image support is unavailable",
            param: "messages", code: "vision_unavailable")) {
            try parser.feed(&buffer)
        }
    }
}

/// Mirrors the session's `prepare` guard verbatim, so the test pins the
/// message the session throws, not just any 400.
private actor QwenImageRefusingBackend: ServerInferenceBackend {
    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        throw ServerRequestError.invalid(
            message: "image input is not available for this model",
            param: "messages", code: "image_not_supported")
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        Issue.record("a refused image request reached generation")
        return ServerCompletion(
            content: "unexpected",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}
