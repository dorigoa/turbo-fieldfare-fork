import Foundation
import Metal
import TurboFieldfare

/// The Qwen branch of the CLI run (S22, D3). The runner and the tokenizer
/// come from `LanguageModelRunnerFactory`, the same factory the Gemma branch
/// uses; the family is the one the installed manifest declared, and the
/// footer names it. Text in, text out: images are refused up front (D12).
func runQwenFamily(args: Args,
                   input: PromptInput,
                   modelURL: URL,
                   stdout: FileHandle,
                   stderr: FileHandle,
                   variant: CLIModelVariant) async -> RunResult {
    do {
        // D12: Qwen packs have no vision companion. Refuse before any GPU
        // work, fail-closed, the same rule the Gemma path applies to a
        // missing companion pack.
        if !args.images.isEmpty || input.hasImages {
            return errored(stderr, "image input is not available for this model", 2)
        }
        let chatTokenizer: any ChatTokenizer = try await LanguageModelRunnerFactory.loadTokenizer(
            family: .qwen35Moe, forModelDirectory: modelURL)
        guard let tokenizer = chatTokenizer as? QwenTokenizer else {
            return errored(stderr, "tokenizer does not match the model family", 1)
        }
        // D11 / U3: reasoning is on by default; `--no-thinking` ends the
        // generation prompt after an empty think block instead of inside one.
        // The split only exists in chat mode, where the generation prompt
        // carries the think frame; a raw prompt streams verbatim, the way
        // the S8 golden and the Gemma raw mode do.
        let enableThinking = !args.noThinking
        let chatMode: Bool
        var promptIds: [Int32]
        switch input {
        case .raw(let text):
            chatMode = false
            // Qwen has no BOS token; `encode` ignores `addBOS` by design.
            promptIds = tokenizer.encode(text, addBOS: false)
        case .messages(let messages):
            chatMode = true
            // The S10 Qwen chat format, rendered through the pinned
            // `chat_template.jinja` with the generation prompt attached.
            let rendered = try tokenizer.applyChatTemplate(
                messages, enableThinking: enableThinking)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        case .multimodal:
            return errored(stderr, "image input is not available for this model", 2)
        }
        guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        guard promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }

        // `auto` resolves from the prompt's real length; the prompt is
        // already encoded, so no cheap estimate is needed here.
        var effectiveArgs = args
        if args.resolvesPrefillChunkAuto {
            effectiveArgs.prefillChunkTokens =
                PrefillRuntimeConfig.autoChunkTokens(promptTokens: promptIds.count)
            if !args.quiet {
                let line = "[prefill chunk auto: "
                    + "\(effectiveArgs.prefillChunkTokens) tokens for a "
                    + "\(promptIds.count)-token prompt]\n"
                stderr.write(Data(line.utf8))
            }
        }
        let args = effectiveArgs
        let runtime = try args.resolvedRuntimeConfiguration(
            forceLogitsHead: false, imagePrompt: false)
        // Before the model load: an invalid sampling combination is refused
        // in milliseconds, the way the Gemma parse-level check does.
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = try qwenGenerationConfig(args: args, maxNewTokens: effectiveMaxNew)

        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner = try LanguageModelRunnerFactory.make(
            model: model,
            context: context,
            maxContext: args.maxContext)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)

        // The S11 parser splits the chat stream (D11): reasoning never
        // reaches stdout; it streams to stderr through the callback. The
        // CLI has no tool surface, so no tool is allowed: a tool-call block
        // the model emits anyway is a closed failure, and the run stops at
        // it. Chat mode only — a raw prompt has no think frame to split.
        let decoder = chatMode
            ? QwenAssistantDecoder(allowedTools: [], insideReasoning: enableThinking)
            : nil
        if let decoder {
            let reasoningSink = QwenReasoningSink(stderr: stderr)
            decoder.onReasoning = { text in reasoningSink.write(text) }
        }
        var decoderError: (any Error)?

        // The reasoning cap. Only a chat turn has a think frame to bound, and
        // only one whose prompt opened it: with `--no-thinking` the parser has
        // already refused a depth, and without `--reasoning-depth` the budget
        // is nil, which is the unbounded run this CLI has always done.
        let reasoningBudget = chatMode && enableThinking
            ? args.reasoningDepth.flatMap(tokenizer.reasoningBudget(depth:))
            : nil

        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: chatTokenizer,
            promptIds: promptIds,
            multimodalInput: nil,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig,
            reasoningBudget: reasoningBudget,
            shouldStop: { decoderError != nil }) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    guard !delta.isEmpty, decoderError == nil else { break }
                    if let decoder {
                        do {
                            for event in try decoder.consume(delta) {
                                if case .content(let text) = event, !text.isEmpty {
                                    stdout.write(Data(text.utf8))
                                }
                            }
                        } catch {
                            decoderError = error
                        }
                    } else {
                        stdout.write(Data(delta.utf8))
                    }
                case .tail(let tail):
                    guard decoderError == nil else { break }
                    if let decoder {
                        do {
                            // The detokenizer's held remainder is the last
                            // fragment of the stream; the finish flush is
                            // after it, in that order.
                            var events = try decoder.consume(tail)
                            events += try decoder.finish()
                            for event in events {
                                if case .content(let text) = event, !text.isEmpty {
                                    stdout.write(Data(text.utf8))
                                }
                            }
                        } catch {
                            decoderError = error
                        }
                    } else {
                        stdout.write(Data(tail.utf8))
                    }
                }
            }
        if let error = decoderError {
            return errored(stderr, "\(error)", 1)
        }

        if !args.quiet {
            // The loaded model's own record, not the pre-load read: the footer
            // names what actually ran, family included.
            let footer = timingFooter(
                reason: String(describing: stats.reason),
                prefillTokens: stats.prefillTokens,
                newTokens: stats.newTokens,
                decodeSeconds: stats.decodeSeconds,
                variant: CLIModelVariant(profile: model.quantProfile,
                                         manifestModelID: model.modelID,
                                         family: model.family))
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch let error as ArgsError {
        return errored(stderr, "\(error)", 2)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

/// The `@Sendable` reasoning callback writes through here; the file handle
/// it holds is owned by the run that installed it.
final class QwenReasoningSink: @unchecked Sendable {
    private let stderr: FileHandle

    init(stderr: FileHandle) { self.stderr = stderr }

    func write(_ text: String) {
        guard !text.isEmpty else { return }
        stderr.write(Data(text.utf8))
    }
}

/// The generation config for a Qwen run (D10, U4): a sampling knob the user
/// left untouched takes the value from the checkpoint's own
/// `generation_config.json` (temperature 1.0, top-k 20, top-p 0.95); a knob
/// the user set keeps their value.
func qwenGenerationConfig(args: Args, maxNewTokens: Int) throws -> GenerationConfig {
    let temperature = args.temperatureSet ? args.temperature : 1.0
    let topK = args.topKSet ? args.topK : 20
    let topP = args.topPSet ? args.topP : 0.95
    // The parse-level check only sees the raw flag values, where Gemma's
    // defaults are what fill the gaps; this is the family-aware one.
    guard !(temperature > 0 && topK == nil && topP.map({ $0 < 1 }) == true) else {
        throw ArgsError.invalidValue(
            flag: "--top-p",
            value: "\(topP ?? 0) requires --top-k between 1 and 256")
    }
    return GenerationConfig(
        maxNewTokens: maxNewTokens,
        temperature: temperature,
        topK: topK,
        topP: topP,
        repetitionPenalty: args.repetitionPenalty,
        seed: args.seed,
        stopStrings: args.stops,
        extraStopTokens: [])
}
