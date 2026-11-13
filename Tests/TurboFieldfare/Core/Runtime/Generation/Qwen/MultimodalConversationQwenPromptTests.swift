import Foundation
import Metal
import Synchronization
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// S25 step 2: the family-general `MultimodalConversation` entry point with
/// the Qwen prompt renderer, driven by a scripted runner — two turns, the
/// second resuming the cache without a reset, and the D12 image refusal.

/// A scripted `LanguageModelRunner`: it delegates logits to
/// `ScriptedLogitProducer` and records the continuation contract the turn
/// loop exercises, so a test can prove the second turn resumed at the
/// recorded position and that no `reset()` happened after the first turn.
/// The failure a scripted `produce` injects once the KV has already moved,
/// so a turn fails the way a mid-decode hardware or sampler failure does.
struct ScriptedDecodeFailure: Error, Equatable {}

private final class ScriptedConversationRunner: LanguageModelRunner,
                                                ContextWindowReporting,
                                                @unchecked Sendable {
    struct State {
        var position = 0
        var resetCount = 0
        var prepared: [Int] = []
        var produced = 0
        var rewindThrows = false
        /// The next `produce` call throws once, after committing its token.
        var failNextProduce = false
    }

    private let script: ScriptedLogitProducer
    private let state = Mutex(State())
    let maxContext: Int

    init(vocabSize: Int, maxContext: Int,
         step: @escaping @Sendable (Int32, Int) -> ScriptedLogitProducer.Step) {
        self.script = ScriptedLogitProducer(vocabSize: vocabSize, step: step)
        self.maxContext = maxContext
    }

    var snapshot: State { state.withLock { $0 } }

    func setRewindThrows(_ value: Bool) {
        state.withLock { $0.rewindThrows = value }
    }

    func failNextProduction() {
        state.withLock { $0.failNextProduce = true }
    }

    func reset() {
        state.withLock { $0.position = 0; $0.resetCount += 1 }
        script.reset()
    }

    var continuationPosition: Int { state.withLock { $0.position } }

    func prepareForContinuation(expectedPosition: Int) throws {
        try state.withLock {
            guard $0.position == expectedPosition else {
                throw GeneratorError.invalidContinuation(
                    "scripted runner at \($0.position), expected \(expectedPosition)")
            }
            $0.prepared.append(expectedPosition)
        }
    }

    func rewind(to position: Int) throws {
        if state.withLock({ $0.rewindThrows }) {
            throw QwenRewindError.rewindUnsupported(position: position)
        }
        state.withLock { $0.position = position }
    }

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await script.produce(token: token, position: position, into: logits)
        let fail = state.withLock { (s: inout State) -> Bool in
            s.position = position + 1; s.produced += 1
            let fail = s.failNextProduce
            s.failNextProduce = false
            return fail
        }
        // After the commit: the runner's cursor has moved, so the conversation
        // sees the same "KV advanced, then the run threw" shape a real
        // mid-decode failure leaves behind.
        if fail { throw ScriptedDecodeFailure() }
    }
}

@Suite(.enabled(if: QwenTestResources.hasTokenizer))
struct MultimodalConversationQwenPromptTests {
    let tokenizer: QwenTokenizer
    let context: MetalContext

    init() async throws {
        tokenizer = try await QwenTokenizer.load(
            from: try #require(QwenTestResources.tokenizerFolder))
        context = try MetalContext()
    }

    /// The single-byte token for a printable byte (GPT-2 byte-level
    /// alphabet: printable bytes map to themselves), as in the S21 loop test.
    private func byteID(_ byte: Character) throws -> Int32 {
        try #require(tokenizer.tokenizer.convertTokenToId(String(byte)).map(Int32.init),
                     "pinned vocab is missing the byte token \(byte)")
    }

    /// A real-but-tiny `Model` from the loader suite's synthetic `model.gturbo`.
    /// The conversation only reads `config.vocabSize` on `restore` and the
    /// vision path; with a scripted runner and text turns the weights are inert.
    private func toyModel() throws -> Model {
        let directory = try ModelLoaderTests.writeToySynthetic()
        return try Model.load(directoryURL: directory, device: context.device,
                              expecting: .gemma4Toy())
    }

    /// `a`, `b`, then `im_end`, whatever the prompt ended with.
    private func scriptNext(a: Int32, b: Int32, end: Int32)
        -> @Sendable (Int32, Int) -> ScriptedLogitProducer.Step {
            { input, _ in
                if input == a { return .argmax(b) }
                if input == b { return .argmax(end) }
                return .argmax(a)
            }
        }

    @Test("two Qwen turns: the second continues the cache with no reset")
    func twoTurnsResumeWithoutReset() async throws {
        let a = try byteID("a"), b = try byteID("b")
        let end = tokenizer.imEndID
        let runner = ScriptedConversationRunner(
            vocabSize: tokenizer.vocabSize, maxContext: 8192,
            step: scriptNext(a: a, b: b, end: end))
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        let renderer = QwenConversationPromptRenderer(tokenizer: tokenizer)
        let conversation = MultimodalConversation(
            model: try toyModel(), context: context,
            renderer: renderer, tokenizer: tokenizer, runner: runner,
            scratch: scratch, maxContext: 8192)

        let first = try await conversation.send(
            parts: [.text("one")],
            config: GenerationConfig(maxNewTokens: 8, temperature: 0),
            prefillConfig: .off)
        #expect(first.text == "ab")
        #expect(first.reason == .endOfTurn)
        // First turn: `.reset` start, no continuation contract yet.
        var snapshot = runner.snapshot
        #expect(snapshot.resetCount == 1)
        #expect(snapshot.prepared.isEmpty)

        let second = try await conversation.send(
            parts: [.text("two")],
            config: GenerationConfig(maxNewTokens: 8, temperature: 0),
            prefillConfig: .off)
        #expect(second.text == "ab")
        #expect(second.reason == .endOfTurn)
        // Second turn: the renderer's continuation tokens joined the existing
        // KV; the loop validated the recorded position instead of resetting.
        snapshot = runner.snapshot
        #expect(snapshot.resetCount == 1)
        #expect(snapshot.prepared == [first.kvTokens])
        #expect(second.cachedTokens == first.kvTokens)
        // The continuation tokens — and only they — went on top of the KV.
        #expect(second.promptTokens - second.cachedTokens > 0)
        // KV and runner cursor still agree for the next turn.
        #expect(runner.continuationPosition == second.kvTokens)
    }

    @Test("a Qwen turn carrying an image is refused before any model work (D12)")
    func imageTurnRefusedBeforeWork() async throws {
        let runner = ScriptedConversationRunner(
            vocabSize: tokenizer.vocabSize, maxContext: 8192) { _, _ in .argmax(0) }
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        let conversation = MultimodalConversation(
            model: try toyModel(), context: context,
            renderer: QwenConversationPromptRenderer(tokenizer: tokenizer),
            tokenizer: tokenizer, runner: runner,
            scratch: scratch, maxContext: 8192)
        await #expect(throws: MultimodalConversationError.self) {
            try await conversation.send(
                parts: [.text("a picture"), .image],
                images: [URL(fileURLWithPath: "/nonexistent.jpg")],
                config: GenerationConfig(maxNewTokens: 8, temperature: 0),
                prefillConfig: .off)
        }
        let snapshot = runner.snapshot
        #expect(snapshot.produced == 0)
        #expect(snapshot.resetCount == 0)
        #expect(snapshot.prepared.isEmpty)
    }

    @Test("the renderer forwards `enableThinking` to both turn shapes (D11)")
    func rendererCarriesThinkingSwitch() throws {
        let text = "Hi"
        let on = QwenConversationPromptRenderer(tokenizer: tokenizer, enableThinking: true)
        let off = QwenConversationPromptRenderer(tokenizer: tokenizer, enableThinking: false)
        #expect(try on.firstTurnTokens(text: text) ==
                    tokenizer.encode(
                        try tokenizer.applyChatTemplate(
                            [GFTokenizer.Message(role: .user, content: text)],
                            enableThinking: true),
                        addBOS: false))
        #expect(try on.continuationTokens(userText: text) ==
                    tokenizer.encodeTextContinuation(userContent: text, enableThinking: true))
        #expect(try off.continuationTokens(userText: text) ==
                    tokenizer.encodeTextContinuation(userContent: text, enableThinking: false))
        // The switch must actually change the prompt, or step 5's flag would
        // be dead on arrival.
        #expect(try on.firstTurnTokens(text: text) != off.firstTurnTokens(text: text))
        #expect(try on.continuationTokens(userText: text) != off.continuationTokens(userText: text))
    }

    /// D6: DeltaNet state cannot rewind, so the recovery rewind throws
    /// `rewindUnsupported`. The conversation must rebuild the KV from the
    /// recorded tokens (reset + replay, the restore pattern) instead of
    /// condemning the lineage.
    @Test("a mid-turn failure replays the recorded lineage when rewind is unsupported (D6)")
    func failedTurnReplaysRecordedLineageAfterUnsupportedRewind() async throws {
        let a = try byteID("a"), b = try byteID("b")
        let end = tokenizer.imEndID
        let runner = ScriptedConversationRunner(
            vocabSize: tokenizer.vocabSize, maxContext: 8192,
            step: scriptNext(a: a, b: b, end: end))
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        let conversation = MultimodalConversation(
            model: try toyModel(), context: context,
            renderer: QwenConversationPromptRenderer(tokenizer: tokenizer),
            tokenizer: tokenizer, runner: runner,
            scratch: scratch, maxContext: 8192)

        let first = try await conversation.send(
            parts: [.text("one")],
            config: GenerationConfig(maxNewTokens: 8, temperature: 0),
            prefillConfig: .off)
        let recorded = first.kvTokens

        // The next turn fails mid-run — after the KV has moved — and the
        // runner refuses the recovery rewind (D6).
        runner.setRewindThrows(true)
        runner.failNextProduction()
        await #expect(throws: ScriptedDecodeFailure.self) {
            try await conversation.send(
                parts: [.text("two")],
                config: GenerationConfig(maxNewTokens: 8, temperature: 0),
                prefillConfig: .off)
        }
        // The fallback rebuilt the KV from the recorded tokens: the cursor
        // sits at the recorded length, the record agrees, and the lineage is
        // not broken.
        #expect(runner.continuationPosition == recorded)
        #expect(await conversation.kvTokenCount == recorded)
        #expect(await conversation.isUsable)
        // One reset for the first turn's `.reset` start, one for the replay;
        // the failed turn itself resumed and was never rewound.
        var snapshot = runner.snapshot
        #expect(snapshot.resetCount == 2)
        #expect(snapshot.prepared == [recorded])

        // The following turn resumes on the rebuilt KV like any other.
        let third = try await conversation.send(
            parts: [.text("three")],
            config: GenerationConfig(maxNewTokens: 8, temperature: 0),
            prefillConfig: .off)
        #expect(third.text == "ab")
        #expect(third.reason == .endOfTurn)
        #expect(third.cachedTokens == recorded)
        snapshot = runner.snapshot
        #expect(snapshot.resetCount == 2)
        #expect(snapshot.prepared == [recorded, recorded])
        #expect(runner.continuationPosition == third.kvTokens)
    }

    /// The second rewind call site: the stop-string trim. The turn completed
    /// cleanly, so a refused rewind must rebuild the trimmed KV and still
    /// return the turn, not lose it.
    @Test("a stop-string trim replays the trimmed lineage when rewind is unsupported (D6)")
    func stopStringTrimReplaysTrimmedLineageAfterUnsupportedRewind() async throws {
        let a = try byteID("a"), b = try byteID("b"), c = try byteID("c")
        let runner = ScriptedConversationRunner(
            vocabSize: tokenizer.vocabSize, maxContext: 8192) { input, _ in
                if input == a { return .argmax(b) }
                if input == b { return .argmax(c) }
                return .argmax(a)
            }
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        let conversation = MultimodalConversation(
            model: try toyModel(), context: context,
            renderer: QwenConversationPromptRenderer(tokenizer: tokenizer),
            tokenizer: tokenizer, runner: runner,
            scratch: scratch, maxContext: 8192)
        // `b` is committed to the KV as an invisible stop-prefix, `c` completes
        // "bc" and stops: one committed hidden token has to be trimmed.
        let config = GenerationConfig(maxNewTokens: 8, temperature: 0, stopStrings: ["bc"])
        runner.setRewindThrows(true)

        let first = try await conversation.send(
            parts: [.text("one")], config: config, prefillConfig: .off)
        #expect(first.text == "a")
        #expect(first.reason == .stopString)
        // The turn survived the refused rewind: the record holds the trimmed
        // KV, and the runner's cursor was rebuilt to match it.
        #expect(await conversation.kvTokenCount == first.kvTokens)
        #expect(runner.continuationPosition == first.kvTokens)
        #expect(await conversation.isUsable)
        // First turn's `.reset` start + the trim replay.
        #expect(runner.snapshot.resetCount == 2)

        // The next turn resumes on the trimmed KV, and its own hidden token is
        // trimmed the same way.
        let second = try await conversation.send(
            parts: [.text("two")], config: config, prefillConfig: .off)
        #expect(second.text == "a")
        #expect(second.reason == .stopString)
        #expect(second.cachedTokens == first.kvTokens)
        let snapshot = runner.snapshot
        #expect(snapshot.resetCount == 3)
        #expect(snapshot.prepared == [first.kvTokens])
        #expect(runner.continuationPosition == second.kvTokens)
    }

    /// The contract the two call sites above rely on: the helpers pass an
    /// unsupported rewind through un-wrapped so the actor can fall back, while
    /// every other rewind failure still arrives as a lineage loss.
    @Test("the KV recovery helpers propagate an unsupported rewind (D6)")
    func kvRecoveryPropagatesRewindUnsupported() {
        #expect(throws: QwenRewindError.self) {
            try MultimodalConversationKVRecovery.restoreAfterFailure(
                positionBefore: 280,
                generationError: CancellationError(),
                rewind: { _ in throw QwenRewindError.rewindUnsupported(position: 280) },
                reset: { Issue.record("unsupported rewind must not silently reset") })
        }
        #expect(throws: QwenRewindError.self) {
            _ = try MultimodalConversationKVRecovery.trimHiddenStopTokens(
                [1, 2, 3],
                withheld: 1,
                rewind: { _ in throw QwenRewindError.rewindUnsupported(position: 2) },
                reset: { Issue.record("unsupported trim rewind must not silently reset") })
        }
        // Any other failure keeps its old wrapped shape (the suite
        // `MultimodalConversationKVRecoveryTests` covers the wrapper itself;
        // this pins down that only `rewindUnsupported` escapes).
        #expect(throws: MultimodalConversationError.self) {
            try MultimodalConversationKVRecovery.restoreAfterFailure(
                positionBefore: 280,
                generationError: CancellationError(),
                rewind: { _ in throw GeneratorError.invalidContinuation("fixture") },
                reset: { Issue.record("failed rewind must not silently reset") })
        }
    }
}
