import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// S21: the family-agnostic generation loop driven by `any ChatTokenizer`,
/// and the real-model proof that the factory builds the Qwen runner and
/// tokenizer end to end.

/// Scripted decode with the Qwen tokenizer: the loop stops on `im_end`,
/// reports `.endOfTurn`, and streams only the visible text.
@Suite(.enabled(if: QwenTestResources.hasTokenizer))
struct QwenChatTokenizerGenerationTests {
    let tokenizer: QwenTokenizer

    init() async throws {
        tokenizer = try await QwenTokenizer.load(
            from: try #require(QwenTestResources.tokenizerFolder))
    }

    /// The single-byte token for a printable byte (GPT-2 byte-level
    /// alphabet: printable bytes map to themselves).
    private func byteID(_ byte: Character) throws -> Int32 {
        try #require(tokenizer.tokenizer.convertTokenToId(String(byte)).map(Int32.init),
                     "pinned vocab is missing the byte token \(byte)")
    }

    @Test("the raw completion loop stops on im_end and reports endOfTurn")
    func stopsOnImEnd() async throws {
        let seq: [Int32] = [try byteID("h"), try byteID("i"), try byteID("d")]
        let end = tokenizer.imEndID
        let next: [Int32: Int32] = {
            var n: [Int32: Int32] = [:]
            for (i, id) in seq.enumerated() { n[id] = (i + 1 < seq.count) ? seq[i + 1] : end }
            return n
        }()
        let first = try #require(seq.first)
        let producer = ScriptedLogitProducer(vocabSize: tokenizer.vocabSize) { input, _ in
            .argmax(next[input] ?? first)
        }
        // Through the existential: the call must resolve to the
        // `any ChatTokenizer` overload.
        let chatTokenizer: any ChatTokenizer = tokenizer
        let promptIds = chatTokenizer.encode("go", addBOS: false)
        let ctx = try MetalContext()
        let scratch = try RawCompletionScratch(context: ctx, vocab: tokenizer.vocabSize)
        var tokenIDs: [Int32] = []
        var deltas: [String] = []
        var tails: [String] = []
        let result = try await runRawCompletion(producer: producer, tokenizer: chatTokenizer,
                                                promptIds: promptIds,
                                                config: GenerationConfig(maxNewTokens: 8, temperature: 0),
                                                context: ctx, scratch: scratch,
                                                prefillConfig: .off) { progress in
            switch progress {
            case .token(_, let id, let delta):
                tokenIDs.append(id)
                deltas.append(delta)
            case .tail(let text): tails.append(text)
            case .prefill: break
            }
        }
        #expect(result.reason == .endOfTurn)
        #expect(result.newTokens == seq.count + 1)
        // The stop token itself is consumed at the boundary and never
        // surfaces as a `.token` event: only the three visible tokens do.
        #expect(tokenIDs == seq)
        // The visible text is the three byte tokens; the skipped `im_end`
        // contributes nothing, and nothing is held back at the stop.
        #expect(deltas.joined() == "hid")
        #expect(tails.joined().isEmpty)
    }
}

/// S21 step 4 real proof: the factory builds the Qwen runner and tokenizer
/// from the installed model, and `runRawCompletion` decodes the 32 greedy
/// tokens of p0 that `mlx-lm` produced in S8.
@Suite(.enabled(if: QwenTestResources.hasModel))
struct QwenFactoryGenerationTests {
    let model: Model
    let context: MetalContext
    let modelDirectory: URL

    init() async throws {
        let directory = try #require(QwenTestResources.modelURL)
        let device = try #require(MTLCreateSystemDefaultDevice())
        model = try Model.load(directoryURL: directory, device: device)
        context = try MetalContext()
        modelDirectory = directory
    }

    @Test("the factory runs 32 greedy tokens of p0 through the raw completion loop")
    func factoryGreedyP0() async throws {
        let runner = try LanguageModelRunnerFactory.make(
            model: model, context: context, maxContext: 2048)
        #expect(runner is QwenHybridForwardRunner)
        let tokenizer: any ChatTokenizer = try await LanguageModelRunnerFactory.loadTokenizer(
            family: .qwen35Moe, forModelDirectory: modelDirectory)

        let promptData = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/prompts.json"))
        let prompts = try JSONSerialization.jsonObject(with: promptData) as! [[String: Any]]
        let entry = try #require(prompts.first { ($0["name"] as? String) == "p0" })
        let promptIds = try #require(entry["ids"] as? [Int]).map(Int32.init)

        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize)
        var ids: [Int32] = []
        let result = try await runRawCompletion(
            producer: runner, tokenizer: tokenizer, promptIds: promptIds,
            config: GenerationConfig(maxNewTokens: 32, temperature: 0),
            context: context, scratch: scratch) { progress in
            if case .token(_, let id, _) = progress { ids.append(id) }
        }
        #expect(result.newTokens == 32)
        #expect(result.reason == .maxTokens)

        let goldenData = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/greedy.json"))
        let golden = try JSONSerialization.jsonObject(with: goldenData) as! [String: [Int]]
        let expected = try #require(golden["p0"]).map(Int32.init)
        let firstDiff = zip(ids, expected).first { $0 != $1 }
        #expect(firstDiff == nil, "p0 greedy diverges from the mlx-lm golden: actual \(firstDiff?.0 ?? -1) vs golden \(firstDiff?.1 ?? -1)")
        if let diff = firstDiff {
            print("S21 step 4 p0 divergence: actual \(diff.0) vs golden \(diff.1)")
        }
        print("S21 step 4 p0 greedy via the factory: 32 tokens, reason \(result.reason)")
    }
}
