import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The reasoning cap against the pinned Qwen tokenizer and the real assistant
/// decoder, with no model weights: the frame the budget forces has to be the
/// one `QwenReasoningSplitter` recognises, or the cap would close the block in
/// the KV while the reader still saw everything as reasoning.
@Suite(.enabled(if: QwenTestResources.hasTokenizer))
struct QwenReasoningBudgetTests {
    let tokenizer: QwenTokenizer

    init() async throws {
        tokenizer = try await QwenTokenizer.load(
            from: try #require(QwenTestResources.tokenizerFolder))
    }

    /// The budget's frame is the marker token plus the template's separator,
    /// and it detokenizes to the text the splitter's state machine expects.
    @Test func budgetFrameIsTheTemplatesOwnClose() throws {
        let budget = try #require(tokenizer.reasoningBudget(depth: .low))
        #expect(budget.tokens == ReasoningDepth.low.budgetTokens)
        #expect(budget.endTokenID == tokenizer.thinkEndID)
        #expect(budget.endMarkerText == QwenTokenizer.thinkEndToken)
        #expect(budget.closeTokenIDs.first == tokenizer.thinkEndID)
        #expect(budget.closeTokenIDs.count >= 2)
        // Streamed through the generation detokenizer — the same one the loop
        // builds — the frame is exactly `</think>` and the separator.
        var detokenizer = tokenizer.makeStreamingDetokenizer(
            barrierTokenIDs: tokenizer.structuralMarkerIDs)
        var text = ""
        for id in budget.closeTokenIDs { text += detokenizer.push(id) }
        text += detokenizer.flush()
        #expect(text == "</think>\n\n")
    }

    /// End to end over the real tokenizer: a model that never closes its block
    /// reasons to the cap, the loop forces the frame, and the real decoder puts
    /// the reasoning on the reasoning callback and the answer in `.content` —
    /// the D11 invariant the cap must not break.
    @Test func cappedTurnYieldsAnAnswerAndNoReasoningInContent() async throws {
        let budget = try #require(tokenizer.reasoningBudget(tokens: 4))
        let separator = try #require(budget.closeTokenIDs.last)
        let thinking = try #require(
            tokenizer.tokenizer.convertTokenToId("Ġhmm").map(Int32.init),
            "pinned vocab is missing the token used as reasoning filler")
        let answerIDs = tokenizer.encode("Paris.", addBOS: false)
        let firstAnswer = try #require(answerIDs.first)

        // Reasoning forever, until the forced separator arrives: then the
        // answer, then the turn's end. The producer only ever sees the
        // separator because the loop really fed the frame through the KV.
        let producer = ScriptedLogitProducer(vocabSize: tokenizer.vocabSize) { input, _ in
            if input == separator { return .argmax(firstAnswer) }
            if let index = answerIDs.firstIndex(of: input), input != thinking {
                let next = index + 1
                return .argmax(next < answerIDs.count
                    ? answerIDs[next] : self.tokenizer.imEndID)
            }
            return .argmax(thinking)
        }

        let chatTokenizer: any ChatTokenizer = tokenizer
        let prompt = try tokenizer.applyChatTemplate(
            [GFTokenizer.Message(role: .user, content: "Capital of France?")],
            enableThinking: true)
        let promptIDs = chatTokenizer.encode(prompt, addBOS: false)
        let context = try MetalContext()
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: tokenizer.vocabSize)

        // The decoder starts inside the block, as the renderer's generation
        // prompt left it.
        let decoder = QwenAssistantDecoder(allowedTools: [], insideReasoning: true)
        let reasoning = ReasoningCollector()
        decoder.onReasoning = { reasoning.append($0) }
        var content = ""
        var decoderError: (any Error)?
        var ids: [Int32] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: chatTokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 64, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off,
            reasoningBudget: budget) { progress in
                switch progress {
                case .token(_, let id, let delta):
                    ids.append(id)
                    guard !delta.isEmpty, decoderError == nil else { break }
                    do {
                        for event in try decoder.consume(delta) {
                            if case .content(let text) = event { content += text }
                        }
                    } catch { decoderError = error }
                case .tail(let tail):
                    guard decoderError == nil else { break }
                    do {
                        var events = try decoder.consume(tail)
                        events += try decoder.finish()
                        for event in events {
                            if case .content(let text) = event { content += text }
                        }
                    } catch { decoderError = error }
                case .prefill: break
                }
            }

        #expect(decoderError == nil)
        #expect(result.reason == .endOfTurn)
        // Four reasoning tokens, then the frame, then the answer.
        #expect(Array(ids.prefix(4)) == Array(repeating: thinking, count: 4))
        #expect(Array(ids.dropFirst(4).prefix(budget.closeTokenIDs.count))
            == budget.closeTokenIDs)
        // The answer is visible content; the reasoning never is, and the
        // marker the loop forced is not content either.
        #expect(content == "Paris.")
        #expect(!content.contains("</think>"))
        #expect(!content.contains("hmm"))
        #expect(reasoning.text.contains("hmm"))
    }

    /// The reasoning callback is `@Sendable`; this is the suite's sink for it.
    final class ReasoningCollector: @unchecked Sendable {
        private var buffer = ""
        func append(_ text: String) { buffer += text }
        var text: String { buffer }
    }
}
