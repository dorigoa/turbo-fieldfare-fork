import Foundation
import Testing
@testable import TurboFieldfare

/// Every golden conversation in `Tests/QwenFixtures/chat/chat_golden.json`
/// (generated in S2 with the pinned checkpoint's tokenizer) must render
/// through `QwenChatFormat` byte-identically, and the pinned token IDs must
/// equal `QwenTokenizer.encode` of that exact text (S10 "Fatto quando").
@Suite(.enabled(if: QwenTestResources.hasTokenizer))
struct QwenChatGoldenTests {
    @Test("All golden conversations render byte-identically")
    func renderMatchesGolden() throws {
        let folder = try #require(QwenTestResources.tokenizerFolder)
        let format = try QwenChatFormat.load(from: folder)
        for golden in try QwenChatFixtures.goldenCases() {
            let text = try format.render(
                messages: golden.messages,
                tools: golden.tools,
                addGenerationPrompt: golden.addGenerationPrompt,
                enableThinking: golden.enableThinking)
            #expect(text == golden.rendered, "rendered text differs for \(golden.name)")
        }
    }

    @Test("Golden IDs equal QwenTokenizer.encode of the rendered text")
    func idsMatchGolden() async throws {
        let folder = try #require(QwenTestResources.tokenizerFolder)
        let tokenizer = try await QwenTokenizer.load(from: folder)
        for golden in try QwenChatFixtures.goldenCases() {
            let ids = tokenizer.encode(golden.rendered, addBOS: false)
            #expect(ids == golden.ids, "token IDs differ for \(golden.name)")
        }
    }
}
