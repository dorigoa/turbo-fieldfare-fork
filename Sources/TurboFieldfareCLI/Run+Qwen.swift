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
        var promptIds: [Int32]
        switch input {
        case .raw(let text):
            // Qwen has no BOS token; `encode` ignores `addBOS` by design.
            promptIds = tokenizer.encode(text, addBOS: false)
        case .messages(let messages):
            // The S10 Qwen chat format, rendered through the pinned
            // `chat_template.jinja` with the generation prompt attached.
            let rendered = try tokenizer.applyChatTemplate(messages)
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
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = try qwenGenerationConfig(args: args, maxNewTokens: effectiveMaxNew)

        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: chatTokenizer,
            promptIds: promptIds,
            multimodalInput: nil,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
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

/// The generation config for a Qwen run: the user's flags, exactly as the
/// Gemma branch applies them.
func qwenGenerationConfig(args: Args, maxNewTokens: Int) throws -> GenerationConfig {
    GenerationConfig(
        maxNewTokens: maxNewTokens,
        temperature: args.temperature,
        topK: args.topK,
        topP: args.topP,
        repetitionPenalty: args.repetitionPenalty,
        seed: args.seed,
        stopStrings: args.stops,
        extraStopTokens: [])
}
