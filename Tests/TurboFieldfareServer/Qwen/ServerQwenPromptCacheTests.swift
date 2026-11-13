import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// Contract C9: tokenizer-backed cases enable themselves only when the
/// installed model (or the kept local tokenizer copy) is present, so a
/// checkout without `scratch/` artifacts stays green. The server test target
/// cannot see `QwenTestResources` (it lives in another test module), so this
/// is the same locator, re-declared file-locally.
private enum QwenServerTestResources {
    /// Repository root, derived from this file's location:
    /// `Tests/TurboFieldfareServer/Qwen/ServerQwenPromptCacheTests.swift`.
    static var repoRoot: URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url
    }

    static var tokenizerFolder: URL? {
        let model = repoRoot.appendingPathComponent("scratch/qwen36.gturbo", isDirectory: true)
        let candidates = [
            model.appendingPathComponent("tokenizer", isDirectory: true),
            repoRoot.appendingPathComponent("scratch/qwen36-tokenizer", isDirectory: true),
        ]
        for folder in candidates
        where FileManager.default.fileExists(
            atPath: folder.appendingPathComponent("tokenizer.json").path) {
            return folder
        }
        return nil
    }

    static var hasTokenizer: Bool { tokenizerFolder != nil }
}

/// S23 step 3: the prompt cache serves Qwen under D6 - reuse only as a
/// continuation at the exact end of the cached prefix, bridged by the Qwen
/// tokenizer with the request's thinking frame. Gemma's rules are asserted
/// unchanged from the same fixtures, so the family branch cannot leak.
@Suite("Server Qwen prompt cache",
       .enabled(if: QwenServerTestResources.hasTokenizer))
struct ServerQwenPromptCacheTests {

    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: false,
        templateSHA256: "template")

    private func qwenTokenizer() async throws -> QwenTokenizer {
        try await QwenTokenizer.load(from: try #require(QwenServerTestResources.tokenizerFolder))
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            imageIdentities: [],
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: kvBacked.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: kvBacked.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary])
    }

    // MARK: - Text continuation

    @Test(arguments: [true, false])
    func textContinuationResumesWithTheRequestsThinkingFrame(enableThinking: Bool) async throws {
        let tokenizer = try await qwenTokenizer()
        let initial = request(messages: [GFTokenizer.Message(role: .user, content: "first")])
        var cache = ServerPromptCache()
        #expect(cache.publish(
            domain: domain, request: initial, content: "answer", calls: [],
            result: rawResult(kvBacked: [1, 2], boundary: tokenizer.endOfTurnID,
                              reason: .endOfTurn)) == .published)

        let continuation = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let bridge = QwenPromptCacheBridge(tokenizer: tokenizer,
                                           enableThinking: enableThinking)
        guard case .hit(let effective, let cached) = cache.match(
            domain: domain, request: continuation, renderedPromptIDs: nil,
            bridge: bridge, family: .qwen35Moe
        ) else {
            Issue.record("expected a Qwen text continuation hit")
            return
        }
        // The Qwen turn closed on `im_end`, which never entered the KV, so
        // the bridge re-emits it and the boundary check holds; only the
        // suffix after the cached position is prefilled (D6).
        let expectedBridge = tokenizer.encodeTextContinuation(
            userContent: "second", enableThinking: enableThinking)
        #expect(cached == 2)
        #expect(effective == [1, 2] + expectedBridge)
        #expect(effective[2] == tokenizer.endOfTurnID)
    }

    // A cached tool turn closed differently per family: Gemma on the
    // tool-response marker (`.toolCalls`), Qwen on `im_end` (`.endOfTurn`).
    // The guard has to accept exactly the family's own closure, never the
    // other's, or a resumed prompt would not be the one the client sent.
    @Test(arguments: [
        (ModelFamily.gemma4, StopReason.endOfTurn),
        (ModelFamily.gemma4, StopReason.maxTokens),
        (ModelFamily.qwen35Moe, StopReason.toolCalls),
    ])
    func foreignToolTurnClosureMisses(family: ModelFamily, reason: StopReason) async throws {
        let tokenizer = try await qwenTokenizer()
        let tools = [GFTokenizer.FunctionDefinition(
            name: "read", description: "Read a file.",
            parameters: .object(["type": .string("object")]))]
        let historical = GFTokenizer.HistoricalToolCall(
            id: "call_1", name: "read",
            arguments: .object(["path": .string("/tmp/x")]))
        let call = ParsedToolCall(id: "call_1", name: "read",
                                  arguments: historical.arguments,
                                  argumentsJSON: #"{"path":"/tmp/x"}"#)
        let initial = request(
            messages: [GFTokenizer.Message(role: .user, content: "first")], tools: tools)
        var cache = ServerPromptCache()
        #expect(cache.publish(
            domain: domain, request: initial, content: "", calls: [call],
            result: rawResult(kvBacked: [1, 2, 3], boundary: tokenizer.endOfTurnID,
                              reason: reason)) == .published)

        let continuation = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
            GFTokenizer.Message(role: .assistant, content: nil, toolCalls: [historical]),
            GFTokenizer.Message(role: .tool, content: "file text", toolCallID: "call_1"),
        ], tools: tools)
        let bridge = QwenPromptCacheBridge(tokenizer: tokenizer, enableThinking: true)
        #expect(cache.match(domain: domain, request: continuation, renderedPromptIDs: nil,
                            bridge: bridge, family: family).missReason
                    == .unsupportedContinuation)
    }

    @Test func toolResultContinuationResumesOnQwenEndOfTurn() async throws {
        let tokenizer = try await qwenTokenizer()
        let tools = [GFTokenizer.FunctionDefinition(
            name: "read", description: "Read a file.",
            parameters: .object(["type": .string("object")]))]
        let historical = GFTokenizer.HistoricalToolCall(
            id: "call_1", name: "read",
            arguments: .object(["path": .string("/tmp/x")]))
        let call = ParsedToolCall(id: "call_1", name: "read",
                                  arguments: historical.arguments,
                                  argumentsJSON: #"{"path":"/tmp/x"}"#)
        let initial = request(
            messages: [GFTokenizer.Message(role: .user, content: "first")], tools: tools)
        var cache = ServerPromptCache()
        // A Qwen tool turn ends `.endOfTurn` (no tool-result wait token),
        // and its uncommitted boundary is `im_end`.
        #expect(cache.publish(
            domain: domain, request: initial, content: "", calls: [call],
            result: rawResult(kvBacked: [1, 2, 3], boundary: tokenizer.endOfTurnID,
                              reason: .endOfTurn)) == .published)

        let continuation = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
            GFTokenizer.Message(role: .assistant, content: nil, toolCalls: [historical]),
            GFTokenizer.Message(role: .tool, content: "file text", toolCallID: "call_1"),
        ], tools: tools)
        let bridge = QwenPromptCacheBridge(tokenizer: tokenizer, enableThinking: true)
        guard case .hit(let effective, let cached) = cache.match(
            domain: domain, request: continuation, renderedPromptIDs: nil,
            bridge: bridge, family: .qwen35Moe
        ) else {
            Issue.record("expected a Qwen tool-result continuation hit")
            return
        }
        // The Qwen template closes the assistant turn before opening the
        // tool-response turn, so the bridge starts right at the dropped
        // `im_end` and carries the tool results in the Qwen form.
        let expectedBridge = try tokenizer.encodeToolResultContinuation(
            cachedMessages: initial.messages,
            assistant: GFTokenizer.Message(role: .assistant, content: nil,
                                           toolCalls: [historical]),
            incomingMessages: continuation.messages,
            tools: tools)
        #expect(cached == 3)
        #expect(effective == [1, 2, 3] + expectedBridge)
        #expect(expectedBridge.first == tokenizer.endOfTurnID)
    }
}
