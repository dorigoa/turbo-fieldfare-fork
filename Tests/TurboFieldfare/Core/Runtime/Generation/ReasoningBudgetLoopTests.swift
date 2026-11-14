import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The reasoning cap as the generation loop runs it, with no model: a toy
/// tokenizer whose vocabulary has a reasoning frame, and a scripted producer
/// that answers whatever token it was last fed. Because the producer is fed the
/// forced tokens too, these cases also prove the closing frame really enters
/// the KV and conditions what the model writes next.
@Suite struct ReasoningBudgetLoopTests {
    // A toy vocabulary. Reasoning tokens are 10…; the frame is 7 and 8.
    private static let thinkEnd: Int32 = 7
    private static let separator: Int32 = 8
    private static let imEnd: Int32 = 9
    private static let answer: Int32 = 20
    private static let vocab = 64

    /// The smallest `ChatTokenizer` the loop needs: id-to-text with the frame
    /// decoding as the literal marker, which is how the pinned Qwen tokenizer
    /// behaves for `</think>` in both skip modes.
    private struct ToyTokenizer: ChatTokenizer, @unchecked Sendable {
        var vocabSize: Int { ReasoningBudgetLoopTests.vocab }
        var stopTokenIDs: Set<Int32> { [ReasoningBudgetLoopTests.imEnd] }
        var endOfTurnID: Int32 { ReasoningBudgetLoopTests.imEnd }
        var structuralMarkerIDs: Set<Int32> {
            [ReasoningBudgetLoopTests.thinkEnd, ReasoningBudgetLoopTests.imEnd]
        }
        var toolResultWaitID: Int32? { nil }

        static func text(for id: Int32) -> String {
            switch id {
            case ReasoningBudgetLoopTests.thinkEnd: "</think>"
            case ReasoningBudgetLoopTests.separator: "\n\n"
            case ReasoningBudgetLoopTests.imEnd: ""
            case ReasoningBudgetLoopTests.answer: "Paris."
            default: "r\(id)"
            }
        }

        func encode(_ text: String, addBOS: Bool) -> [Int32] { [10] }
        func decode(_ ids: [Int32], skipSpecialTokens: Bool) -> String {
            ids.map(Self.text(for:)).joined()
        }
        func makeStreamingDetokenizer(barrierTokenIDs: Set<Int32>) -> any StreamingDetokenizer {
            ToyDetokenizer()
        }
    }

    private struct ToyDetokenizer: StreamingDetokenizer {
        mutating func push(_ id: Int32) -> String { ToyTokenizer.text(for: id) }
        mutating func flush() -> String { "" }
    }

    private func budget(tokens: Int) throws -> ReasoningBudget {
        try #require(ReasoningBudget(tokens: tokens,
                                     endTokenID: Self.thinkEnd,
                                     endMarkerText: "</think>",
                                     closeTokenIDs: [Self.thinkEnd, Self.separator]))
    }

    /// Runs the loop over a producer driven by `next`, which maps the token it
    /// was last fed to the token it answers with.
    private func run(budget: ReasoningBudget?,
                     maxNewTokens: Int,
                     next: @escaping @Sendable (Int32) -> Int32)
        async throws -> (ids: [Int32], text: String, result: RawDecodeResult) {
        let context = try MetalContext()
        let tokenizer = ToyTokenizer()
        let producer = ScriptedLogitProducer(vocabSize: Self.vocab) { input, _ in
            .argmax(next(input))
        }
        let scratch = try RawCompletionScratch(context: context, vocab: Self.vocab)
        var ids: [Int32] = []
        var text = ""
        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer as any ChatTokenizer,
            promptIds: [10],
            config: GenerationConfig(maxNewTokens: maxNewTokens, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .off,
            reasoningBudget: budget) { progress in
                switch progress {
                case .token(_, let id, let delta):
                    ids.append(id)
                    text += delta
                case .tail(let tail): text += tail
                case .prefill: break
                }
            }
        return (ids, text, result)
    }

    /// A model that never closes its block — the S27 behaviour — reasons until
    /// the cap, and the loop then forces the closing frame into the stream and
    /// lets the answer out. The producer is fed the frame, so the answer token
    /// here is the one scripted to follow the separator: the proof that the
    /// model's continuation is conditioned on a closed block.
    @Test func capClosesABlockTheModelNeverCloses() async throws {
        let run = try await run(budget: try budget(tokens: 3), maxNewTokens: 32) { input in
            switch input {
            case Self.separator: Self.answer
            case Self.answer: Self.imEnd
            default: 11  // reasoning, forever
            }
        }
        #expect(run.ids == [11, 11, 11, Self.thinkEnd, Self.separator, Self.answer])
        #expect(run.result.reason == .endOfTurn)
        // Decision 3: the frame does not count toward the budget, but its
        // tokens are real tokens — three reasoning, two frame, one answer, plus
        // the stop token the loop consumes at the boundary.
        #expect(run.result.newTokens == 7)
        #expect(run.text == "r11r11r11</think>\n\nParis.")
        // The frame and the answer are in the KV, in order, after the prompt.
        #expect(run.result.kvBackedTokenIDs
            == [10, 11, 11, 11, Self.thinkEnd, Self.separator, Self.answer])
    }

    /// Decision 1: a model that closes its own block before the cap is left
    /// alone — no frame is forced, and the stream carries exactly one close.
    @Test func capStaysOutOfTheWayWhenTheModelClosesItself() async throws {
        let run = try await run(budget: try budget(tokens: 5), maxNewTokens: 32) { input in
            switch input {
            case 11: Self.thinkEnd
            case Self.thinkEnd: Self.answer
            case Self.answer: Self.imEnd
            default: 11
            }
        }
        #expect(run.ids == [11, Self.thinkEnd, Self.answer])
        #expect(run.ids.filter { $0 == Self.thinkEnd }.count == 1)
        #expect(run.result.reason == .endOfTurn)
    }

    /// Decision 2: whatever the model writes after the forced close is content,
    /// and no second frame is forced however long it keeps going.
    @Test func noSecondFrameIsForcedAfterTheFirst() async throws {
        let run = try await run(budget: try budget(tokens: 2), maxNewTokens: 12) { _ in 11 }
        #expect(run.result.reason == .maxTokens)
        #expect(run.ids.filter { $0 == Self.thinkEnd }.count == 1)
        #expect(run.ids.prefix(4) == [11, 11, Self.thinkEnd, Self.separator])
        // Everything after the frame is the model's own writing, uninterrupted.
        #expect(run.ids.dropFirst(4).allSatisfy { $0 == 11 })
    }

    /// Decision 4: the response limit wins. Reached on the same step the cap
    /// would fire, the turn ends there with no answer — exactly as it does
    /// today — and no frame is forced into a turn that is already over.
    @Test func responseLimitWinsOverTheCap() async throws {
        let run = try await run(budget: try budget(tokens: 3), maxNewTokens: 3) { _ in 11 }
        #expect(run.ids == [11, 11, 11])
        #expect(run.result.reason == .maxTokens)
        #expect(!run.ids.contains(Self.thinkEnd))
    }

    /// No budget is the unbounded path every surface had before the cap
    /// existed, and every Gemma turn: the loop forces nothing at all.
    @Test func noBudgetLeavesTheLoopUnchanged() async throws {
        let run = try await run(budget: nil, maxNewTokens: 8) { _ in 11 }
        #expect(run.ids == Array(repeating: 11, count: 8))
        #expect(run.result.reason == .maxTokens)
        #expect(!run.ids.contains(Self.thinkEnd))
    }

    /// A one-token budget is the shallowest cap that still lets the model
    /// start: one reasoning token, then the frame.
    @Test func oneTokenBudgetClosesAfterTheFirstReasoningToken() async throws {
        let run = try await run(budget: try budget(tokens: 1), maxNewTokens: 16) { input in
            switch input {
            case Self.separator: Self.answer
            case Self.answer: Self.imEnd
            default: 11
            }
        }
        #expect(run.ids == [11, Self.thinkEnd, Self.separator, Self.answer])
        #expect(run.result.reason == .endOfTurn)
    }
}
