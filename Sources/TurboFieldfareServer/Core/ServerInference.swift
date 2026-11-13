import CryptoKit
import Foundation
import TurboFieldfare

public enum ServerInferenceEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Equatable, Sendable {
    public let content: String
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage
    /// Wall time the runtime spent prefilling the prompt and decoding the
    /// completion, as `runRawCompletion` already measures them around those two
    /// phases; nothing is timed here. Defaulted to zero so a backend that does
    /// not decode through the runtime — the test doubles, the compatibility
    /// harness — reports no timing rather than a fabricated one.
    public let prefillSeconds: Double
    public let decodeSeconds: Double

    public init(content: String,
                toolCalls: [ParsedToolCall],
                finishReason: String,
                usage: OpenAIUsage,
                prefillSeconds: Double = 0,
                decodeSeconds: Double = 0) {
        self.content = content
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
    }
}

enum StructuredOutputFailureKind: String, Equatable, Sendable {
    case decoderConsume = "decoder_consume"
    case decoderFinish = "decoder_finish"
    case orphanToolResponse = "orphan_tool_response"
}

enum StructuredOutputFailureCause: String, Equatable, Sendable {
    case malformed
    case unknownTool = "unknown_tool"
    case oversized
    case unexpected
    case none

    static func classify(_ error: Error) -> Self {
        guard let parserError = error as? GemmaToolCallParserError else {
            return .unexpected
        }
        switch parserError {
        case .malformed: return .malformed
        case .unknownTool: return .unknownTool
        case .oversized: return .oversized
        }
    }
}

struct StructuredOutputFailureDiagnostics: Equatable, Sendable {
    let renderedPromptTokens: Int
    let effectivePromptTokens: Int
    let resultPromptTokens: Int
    let cachedPromptTokens: Int
    let computedPrefillTokens: Int
    let completionTokens: Int
    let maxCompletionTokens: Int
    let rawStop: String
    let kvPosition: Int
    let kvBackedTokens: Int
    let boundaryTokens: Int
    let decodedCalls: Int
    let visibleBytes: Int
    let stopStringMatched: Bool
    let toolStartCount: Int
    let toolEndCount: Int
    let toolResponseCount: Int
    let toolResponseEndCount: Int
    let lastToolStartOffset: Int
    let lastToolEndOffset: Int
    let lastToolResponseOffset: Int
    let lastToolResponseEndOffset: Int
    let effectiveCountMatchesResult: Bool
    let effectivePrefixMatchesKV: Bool
    let kvPositionMatchesHistory: Bool
    let completionCountMatchesHistory: Bool
    let prefillAccountingMatches: Bool
    let renderedPromptHash: String
    let effectivePromptHash: String
    let generatedHash: String

    init(
        renderedPromptIDs: [Int32],
        effectivePromptIDs: [Int32],
        result: RawDecodeResult,
        maxCompletionTokens: Int,
        decodedCalls: Int,
        visibleBytes: Int,
        stopStringMatched: Bool,
        toolStartID: Int32,
        toolEndID: Int32,
        toolResponseID: Int32,
        toolResponseEndID: Int32
    ) {
        let safePrefillCount = min(
            max(result.prefillTokens, 0),
            result.kvBackedTokenIDs.count)
        let committedGenerated = result.kvBackedTokenIDs.dropFirst(safePrefillCount)
        let boundary = result.uncommittedBoundaryTokenIDs[...]
        let generatedSegments = [committedGenerated, boundary]

        var toolStartCount = 0
        var toolEndCount = 0
        var toolResponseCount = 0
        var toolResponseEndCount = 0
        var lastToolStartOffset = -1
        var lastToolEndOffset = -1
        var lastToolResponseOffset = -1
        var lastToolResponseEndOffset = -1
        var offset = 0
        for segment in generatedSegments {
            for tokenID in segment {
                if tokenID == toolStartID {
                    toolStartCount += 1
                    lastToolStartOffset = offset
                }
                if tokenID == toolEndID {
                    toolEndCount += 1
                    lastToolEndOffset = offset
                }
                if tokenID == toolResponseID {
                    toolResponseCount += 1
                    lastToolResponseOffset = offset
                }
                if tokenID == toolResponseEndID {
                    toolResponseEndCount += 1
                    lastToolResponseEndOffset = offset
                }
                offset += 1
            }
        }

        let (prefillAccounted, prefillOverflow) = result.cachedPromptTokens
            .addingReportingOverflow(result.computedPrefillTokens)

        self.renderedPromptTokens = renderedPromptIDs.count
        self.effectivePromptTokens = effectivePromptIDs.count
        self.resultPromptTokens = result.prefillTokens
        self.cachedPromptTokens = result.cachedPromptTokens
        self.computedPrefillTokens = result.computedPrefillTokens
        self.completionTokens = result.newTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.rawStop = Self.rawStop(result.reason)
        self.kvPosition = result.kvPosition
        self.kvBackedTokens = result.kvBackedTokenIDs.count
        self.boundaryTokens = result.uncommittedBoundaryTokenIDs.count
        self.decodedCalls = decodedCalls
        self.visibleBytes = visibleBytes
        self.stopStringMatched = stopStringMatched
        self.toolStartCount = toolStartCount
        self.toolEndCount = toolEndCount
        self.toolResponseCount = toolResponseCount
        self.toolResponseEndCount = toolResponseEndCount
        self.lastToolStartOffset = lastToolStartOffset
        self.lastToolEndOffset = lastToolEndOffset
        self.lastToolResponseOffset = lastToolResponseOffset
        self.lastToolResponseEndOffset = lastToolResponseEndOffset
        self.effectiveCountMatchesResult = effectivePromptIDs.count == result.prefillTokens
        self.effectivePrefixMatchesKV = result.kvBackedTokenIDs.count >= effectivePromptIDs.count
            && result.kvBackedTokenIDs.prefix(effectivePromptIDs.count)
                .elementsEqual(effectivePromptIDs)
        self.kvPositionMatchesHistory = result.kvPosition == result.kvBackedTokenIDs.count
        self.completionCountMatchesHistory = offset == result.newTokens
        self.prefillAccountingMatches = !prefillOverflow
            && prefillAccounted == result.prefillTokens
        self.renderedPromptHash = Self.i32leSHA256([renderedPromptIDs[...]])
        self.effectivePromptHash = Self.i32leSHA256([effectivePromptIDs[...]])
        self.generatedHash = Self.i32leSHA256(generatedSegments)
    }

    static func i32leSHA256(_ segments: [ArraySlice<Int32>]) -> String {
        var hasher = SHA256()
        var bytes = Data()
        bytes.reserveCapacity(4_096)
        for segment in segments {
            for tokenID in segment {
                var littleEndian = UInt32(bitPattern: tokenID).littleEndian
                withUnsafeBytes(of: &littleEndian) {
                    bytes.append(contentsOf: $0)
                }
                if bytes.count == 4_096 {
                    hasher.update(data: bytes)
                    bytes.removeAll(keepingCapacity: true)
                }
            }
        }
        if !bytes.isEmpty { hasher.update(data: bytes) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    var logDescription: String {
        [
            "rendered_prompt_tokens=\(renderedPromptTokens)",
            "effective_prompt_tokens=\(effectivePromptTokens)",
            "result_prompt_tokens=\(resultPromptTokens)",
            "cached_prompt_tokens=\(cachedPromptTokens)",
            "computed_prefill_tokens=\(computedPrefillTokens)",
            "completion_tokens=\(completionTokens)",
            "max_completion_tokens=\(maxCompletionTokens)",
            "raw_stop=\(rawStop)",
            "kv_position=\(kvPosition)",
            "kv_backed_tokens=\(kvBackedTokens)",
            "boundary_tokens=\(boundaryTokens)",
            "decoded_calls=\(decodedCalls)",
            "visible_bytes=\(visibleBytes)",
            "stop_string_matched=\(stopStringMatched)",
            "tool_start_count=\(toolStartCount)",
            "tool_end_count=\(toolEndCount)",
            "tool_response_count=\(toolResponseCount)",
            "tool_response_end_count=\(toolResponseEndCount)",
            "last_tool_start_offset=\(lastToolStartOffset)",
            "last_tool_end_offset=\(lastToolEndOffset)",
            "last_tool_response_offset=\(lastToolResponseOffset)",
            "last_tool_response_end_offset=\(lastToolResponseEndOffset)",
            "effective_count_matches_result=\(effectiveCountMatchesResult)",
            "effective_prefix_matches_kv=\(effectivePrefixMatchesKV)",
            "kv_position_matches_history=\(kvPositionMatchesHistory)",
            "completion_count_matches_history=\(completionCountMatchesHistory)",
            "prefill_accounting_matches=\(prefillAccountingMatches)",
            "rendered_prompt_i32le_sha256=\(renderedPromptHash)",
            "effective_prompt_i32le_sha256=\(effectivePromptHash)",
            "generated_i32le_sha256=\(generatedHash)",
        ].joined(separator: " ")
    }

    private static func rawStop(_ reason: StopReason) -> String {
        switch reason {
        case .eos: "eos"
        case .endOfTurn: "end_of_turn"
        case .maxTokens: "max_tokens"
        case .stopString: "stop_string"
        case .cancelled: "cancelled"
        case .toolCalls: "tool_calls"
        }
    }
}

struct StructuredOutputFailure: Error, CustomDebugStringConvertible, Sendable {
    let kind: StructuredOutputFailureKind
    let cause: StructuredOutputFailureCause
    let diagnostics: StructuredOutputFailureDiagnostics

    var debugDescription: String {
        "structured_output_failure kind=\(kind.rawValue) "
            + "cause=\(cause.rawValue) \(diagnostics.logDescription)"
    }
}

public struct ServerPreparedRequest: Sendable {
    public let request: ValidatedChatRequest
    fileprivate let promptIDs: [Int32]?

    public var promptTokenCount: Int? { promptIDs?.count }

    init(request: ValidatedChatRequest, promptIDs: [Int32]? = nil) {
        self.request = request
        self.promptIDs = promptIDs
    }
}

public protocol ServerInferenceBackend: Sendable {
    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
    func generate(_ prepared: ServerPreparedRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

public extension ServerInferenceBackend {
    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        ServerPreparedRequest(request: request)
    }

    func generate(
        _ prepared: ServerPreparedRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        try await generate(prepared.request, onEvent: onEvent)
    }
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var admittedCount = 0
    private var active = false
    private var waiters: [Waiter] = []
    private var shuttingDown = false

    public init(queueLimit: Int) {
        self.queueLimit = queueLimit
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await runPreparing(
            onQueued: onQueued,
            prepare: { () },
            operation: { _ in try await operation() })
    }

    func runPreparing<Prepared: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        prepare: @escaping @Sendable () async throws -> Prepared,
        operation: @escaping @Sendable (Prepared) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        guard admittedCount <= queueLimit else { throw ServerRequestError.queueFull }
        admittedCount += 1
        defer { admittedCount -= 1 }

        let prepared = try await prepare()
        try Task.checkCancellation()
        try await acquire(onQueued: onQueued)
        defer { release() }
        return try await operation(prepared)
    }

    private func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if !active {
            active = true
            return
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            active = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { active }
}

/// How the server reads the images of a request: one plan per image, and the
/// encode reads the plan the count came from.
///
/// Planning opens the file, sniffs it, parses its metadata and — for a JPEG —
/// walks its markers. Counting with a plan and then letting `encodeImage(at:)`
/// re-plan internally did all of that a second time per image, and the two
/// opens saw two files: one rewritten in between produced a count the laid-out
/// span no longer matched, and the turn died on `placeholderMismatch` naming
/// nothing the client did.
///
/// Both multimodal paths go through here so the pattern has one implementation
/// to keep correct rather than one per path.
enum ServerRequestImages {
    /// An image planned once: the count that lays out its placeholder span,
    /// and — while the plan behind that count is still open — the bytes the
    /// count was taken from.
    struct Planned {
        let url: URL
        let softTokenCount: Int
        let plan: VisionImagePlan?
    }

    /// How many of a request's images keep their plan open at once.
    ///
    /// A plan owns the descriptor its ImageIO source reads through, and what
    /// bounds a request's image count is the context budget alone — an
    /// 8,192-token context admits some 2,700 single-token images. One
    /// descriptor each would exhaust the process table and take the listening
    /// socket down with it, so past this many the plan is released after
    /// counting and remade at encode time. Real requests carry a handful of
    /// images and never reach it.
    static let maximumOpenPlans = 32

    /// Plans every image before any of them is encoded.
    ///
    /// Two reasons for the order: the counts that lay out the placeholder spans
    /// then come from the same open files the encodes read, and a request whose
    /// last image cannot be read at all is refused before the tower has run on
    /// the images ahead of it.
    static func plans(
        for urls: [URL],
        with preprocessor: Gemma4ImagePreprocessor,
        maximumOpenPlans: Int = ServerRequestImages.maximumOpenPlans,
        checkCancellation: () throws -> Void = {}
    ) throws -> [Planned] {
        var planned: [Planned] = []
        planned.reserveCapacity(urls.count)
        for url in urls {
            // Planning a JPEG walks its whole scan, so an abandoned request
            // stops at the next image rather than reading all of them first.
            try checkCancellation()
            let plan = try preprocessor.plan(fileURL: url)
            planned.append(Planned(
                url: url,
                softTokenCount: plan.geometry.softTokenCount,
                plan: planned.count < maximumOpenPlans ? plan : nil))
        }
        return planned
    }

    /// What an image's encode reads through.
    enum Source {
        /// The plan the image's count was taken from, still open.
        case plan(VisionImagePlan)
        /// The file again, for an image the open bound released.
        case reopened(URL)
    }

    /// An image is encoded from the plan its count was taken from. Only one
    /// whose plan the open bound released is read a second time.
    ///
    /// Returned rather than dispatched through caller-supplied closures:
    /// Swift 6.2's region isolation rejects an actor-isolated closure that
    /// captures a session's `VisionRuntime` and `Model` and is handed to a
    /// nonisolated callee, which broke the build on Xcode 26.0-26.3.
    static func source(for image: Planned) -> Source {
        guard let plan = image.plan else { return .reopened(image.url) }
        return .plan(plan)
    }

    /// Every image of a request planned, each still paired with the id its
    /// features are filed under.
    ///
    /// The pairing is the reason this is not two statements at the call site:
    /// `[UUID: URL]` has no order, so the ids and the plans have to be taken
    /// from one snapshot. Pairing them from two walks of the dictionary files
    /// one image's features under another's id, and when both project to the
    /// same soft-token count nothing downstream can notice.
    ///
    /// Planning every image before the caller encodes any is the other half:
    /// the spans are laid out from the same open files the encodes read, and a
    /// request whose last image cannot be read at all is refused before the
    /// tower has run on the ones ahead of it.
    static func plannedEntries(
        _ imageFiles: [UUID: URL],
        with preprocessor: Gemma4ImagePreprocessor,
        maximumOpenPlans: Int = ServerRequestImages.maximumOpenPlans,
        checkCancellation: () throws -> Void = {}
    ) throws -> [(id: UUID, image: Planned)] {
        let entries = Array(imageFiles)
        let planned = try plans(
            for: entries.map { $0.value },
            with: preprocessor,
            maximumOpenPlans: maximumOpenPlans,
            checkCancellation: checkCancellation)
        return zip(entries, planned).map { (id: $0.key, image: $1) }
    }

    /// A file the caller supplied that cannot be read is a bad request, not a
    /// server fault, and it stays one wherever the read fails. Admission reads
    /// dimensions with stream verification off; the encode re-plans with it on,
    /// so a truncated upload is admitted and only fails the second read.
    /// Without this that second failure reached the generic handler as a 500,
    /// which official clients retry and 4xx they do not.
    static func requestError(forUnreadable error: any Error) -> any Error {
        if error is ServerRequestError || error is CancellationError { return error }
        return ServerRequestError.invalid(
            message: "image could not be read: \(error)",
            param: "messages", code: "invalid_image")
    }
}

/// Load-time failures that are neither a manifest error nor a client error.
enum ServerModelSessionError: Error, CustomStringConvertible {
    /// A Qwen pack has no vision companion (D12): an explicit `--vision-pack`
    /// for it is a startup mistake, refused with the path that caused it.
    case visionUnsupportedForQwen(path: String)
    /// A family's tokenizer was reached from the other family's path. The
    /// session loads both fields from the manifest's family, so reaching one
    /// is a wiring bug in the server, never a request outcome.
    case qwenChatUnavailable

    var description: String {
        switch self {
        case .visionUnsupportedForQwen(let path):
            "vision is not available for this model; \(path) cannot be used with a Qwen pack"
        case .qwenChatUnavailable:
            "Qwen chat serving is not available in this build yet"
        }
    }
}

public actor ServerModelSession: ServerInferenceBackend {
    private let context: MetalContext
    private let model: Model
    /// Gemma's tokenizer; nil for a Qwen session (D3). Gemma-specific paths
    /// (vision, the Gemma decoder, the Gemma cache bridges) go through
    /// `requireGemmaTokenizer`, the family branch picks the tokenizer it needs.
    private let tokenizer: GFTokenizer?
    private let qwenTokenizer: QwenTokenizer?
    /// The tokenizer the shared completion loop runs on, whichever family.
    /// Both families' tokenizers are Sendable structs; the intersection keeps
    /// the actor able to hand it to the nonisolated completion loop.
    private let chatTokenizer: any ChatTokenizer & Sendable
    /// The runner the S21 factory built for the family (D3).
    private let runner: any LanguageModelRunner
    private let scratch: RawCompletionScratch
    private let prefillConfig: PrefillRuntimeConfig
    private let maxContext: Int
    private let promptCacheMode: ServerPromptCacheMode
    private let promptCacheDomain: ServerPromptCacheDomain
    private var promptCache = ServerPromptCache()
    public nonisolated let visionCapability: String
    /// The loaded model's own variant, for the ready line.
    public nonisolated let modelVariant: ServerModelVariant
    /// The family the loaded manifest declared, for the request validator
    /// and the log (D3, D9).
    public nonisolated let family: ModelFamily
    private let visionRuntime: VisionRuntime?
    private let visionResidencyPolicy: VisionResidencyPolicy

    /// What the cached KV was actually produced by. Six configuration fields
    /// plus every `TURBO_FIELDFARE_*` variable in the environment: those select
    /// kernels - about nineteen vision switches alone - and one of them set to
    /// `0` silently picks a different attention path. Leaving them out meant a
    /// prefix built by one kernel set could be reused under another, with
    /// nothing recording that anything had changed.
    static func runtimeIdentityString(
        runtime: RuntimeConfiguration,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        let configuration = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
        ]
        let switches = environment
            .filter { $0.key.hasPrefix("TURBO_FIELDFARE_") }
            .map { "\($0.key)=\($0.value)" }
            .sorted()
        return (configuration + switches).joined(separator: ":")
    }

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            visionPackURL: URL? = nil,
                            visionResidencyPolicy: VisionResidencyPolicy = .onDemand,
                            promptCacheMode: ServerPromptCacheMode = .singlePrefix,
                            runtimeConfiguration: RuntimeConfiguration) async throws -> ServerModelSession {
        // The manifest names the family before anything else opens: the
        // tokenizer and the runner are family-specific (D3), and a Qwen pack
        // must not be read by Gemma's tokenizer codec nor run by Gemma's
        // runner. The read is `manifest.json` alone, a few kilobytes, and it
        // never touches the weights; a missing manifest keeps today's flow,
        // where the load below reports the precise reason.
        let manifestVariant = try ServerModelVariant.read(modelDirectory: modelDirectory)
        let family = manifestVariant?.family ?? .gemma4
        // A Qwen pack has no vision companion at all (D12): an explicit
        // `--vision-pack` for it is a startup mistake, refused with the path.
        if family == .qwen35Moe, let visionPackURL {
            throw ServerModelSessionError.visionUnsupportedForQwen(
                path: visionPackURL.path)
        }
        let tokenizerFolder = family == .qwen35Moe
            ? QwenTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
            : GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw family == .qwen35Moe
                ? QwenTokenizerError.missingTokenizerConfig
                : GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            if family == .qwen35Moe {
                throw QwenTokenizerError.invalidChatTemplate(
                    "chat_template.jinja is missing from the tokenizer folder")
            }
            throw GFTokenizerError.missingToolTemplate
        }
        let chatTokenizer: any ChatTokenizer & Sendable
        if family == .qwen35Moe {
            chatTokenizer = try await QwenTokenizer.load(from: tokenizerFolder)
        } else {
            chatTokenizer = try await GFTokenizer.load(from: tokenizerFolder)
        }
        let tokenizer = chatTokenizer as? GFTokenizer
        let qwenTokenizer = chatTokenizer as? QwenTokenizer
        let context = try MetalContext()
        let runtime = runtimeConfiguration
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        // The loaded model's own record, family included: `Model.load` has
        // already validated the manifest's family against the pinned
        // constants, so the family read from the manifest above is the family
        // that will actually run.
        let modelVariant = ServerModelVariant(profile: model.quantProfile,
                                              manifestModelID: model.modelID,
                                              family: family)
        ServerLog.modelOpened(modelVariant)
        // The one factory that chooses the runner by family (D3, S21): the
        // server no longer builds `RealForwardRunner` itself.
        let runner = try LanguageModelRunnerFactory.make(
            model: model,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize)
        let templateDigest = SHA256.hash(data: try Data(contentsOf: templateURL))
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = Self.runtimeIdentityString(runtime: runtime)
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: PrefillKVStorageMode.fp16.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        let visionRuntime: VisionRuntime?
        let visionCapability: String
        // An explicit pack path is an operator's statement that the pack is
        // there; a typo must fail loudly at startup, not serve text with vision
        // silently missing.
        if let visionPackURL,
           !FileManager.default.fileExists(atPath: visionPackURL.path) {
            throw VisionPackError.packNotFound(visionPackURL.path)
        }
        // A model directory without the .gturbo suffix has no companion
        // location; that only means vision is missing, never that a
        // text-serving load must fail.
        let resolvedVisionPackURL = visionPackURL
            ?? (try? VisionPackLocation.companionURL(forTextModel: modelDirectory))
        if family == .qwen35Moe {
            // D12: no image tower, no companion pack, nothing to probe:
            // Qwen serving is text-only, reported as `missing` by `/health`
            // and `["text"]` by `/v1/models`.
            visionRuntime = nil
            visionCapability = "missing"
        } else if let hardwareCapability = Self.hardwareVisionCapability(
            supportsVisionRuntime: VisionRuntime.isSupported(on: context.device)) {
            visionRuntime = nil
            visionCapability = hardwareCapability
            ServerLog.visionRuntimeUnsupported()
        } else if let resolvedVisionPackURL,
           FileManager.default.fileExists(atPath: resolvedVisionPackURL.path) {
            do {
                visionRuntime = try VisionRuntime.open(
                    textModelURL: modelDirectory,
                    context: context,
                    visionPackURL: resolvedVisionPackURL)
                visionCapability = "ready"
            } catch {
                visionRuntime = nil
                visionCapability = Self.unavailableVisionCapability(for: error)
                if visionCapability == "unsupported" {
                    ServerLog.visionRuntimeUnsupported(
                        at: resolvedVisionPackURL, error: error)
                } else {
                    ServerLog.visionPackInvalid(at: resolvedVisionPackURL, error: error)
                }
            }
        } else {
            visionRuntime = nil
            visionCapability = "missing"
        }
        return ServerModelSession(context: context,
                                  model: model,
                                  tokenizer: tokenizer,
                                  qwenTokenizer: qwenTokenizer,
                                  chatTokenizer: chatTokenizer,
                                  family: modelVariant.family,
                                  runner: runner,
                                  scratch: scratch,
                                  prefillConfig: runtime.prefillConfig,
                                  maxContext: maxContext,
                                  promptCacheMode: promptCacheMode,
                                  promptCacheDomain: promptCacheDomain,
                                  visionRuntime: visionRuntime,
                                  visionCapability: visionCapability,
                                  visionResidencyPolicy: visionResidencyPolicy,
                                  modelVariant: modelVariant)
    }

    static func hardwareVisionCapability(
        supportsVisionRuntime: Bool
    ) -> String? {
        supportsVisionRuntime ? nil : "unsupported"
    }

    static func unavailableVisionCapability(for error: Error) -> String {
        guard let runtimeError = error as? VisionRuntimeError else {
            return "invalid"
        }
        if case .unsupportedKernel = runtimeError {
            return "unsupported"
        }
        return "invalid"
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: GFTokenizer?,
                 qwenTokenizer: QwenTokenizer?,
                 chatTokenizer: any ChatTokenizer & Sendable,
                 family: ModelFamily,
                 runner: any LanguageModelRunner,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 maxContext: Int,
                 promptCacheMode: ServerPromptCacheMode,
                 promptCacheDomain: ServerPromptCacheDomain,
                 visionRuntime: VisionRuntime?,
                 visionCapability: String,
                 visionResidencyPolicy: VisionResidencyPolicy,
                 modelVariant: ServerModelVariant) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.qwenTokenizer = qwenTokenizer
        self.chatTokenizer = chatTokenizer
        self.family = family
        self.runner = runner
        self.scratch = scratch
        self.prefillConfig = prefillConfig
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
        self.visionRuntime = visionRuntime
        self.visionResidencyPolicy = visionResidencyPolicy
        self.visionCapability = visionCapability
        self.modelVariant = modelVariant
    }

    /// The Gemma tokenizer, for paths that exist only for Gemma: vision,
    /// the Gemma decoder, and the Gemma cache bridges. Reaching it from a
    /// Qwen session is a bug, not a request outcome; the interim guard
    /// refuses in the meantime.
    private var gemmaTokenizer: GFTokenizer {
        get throws {
            guard let tokenizer else {
                throw ServerModelSessionError.qwenChatUnavailable
            }
            return tokenizer
        }
    }

    /// The Qwen tokenizer, for paths that exist only for Qwen: the pinned
    /// chat template, the tool-call stream parser, and the Qwen cache
    /// bridges. Like the Gemma accessor, reaching it from the other family's
    /// path is a bug, not a request outcome.
    private var qwenChatTokenizer: QwenTokenizer {
        get throws {
            guard let qwenTokenizer else {
                throw ServerModelSessionError.qwenChatUnavailable
            }
            return qwenTokenizer
        }
    }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let prepared = try await prepare(request)
        return try await generate(prepared, onEvent: onEvent)
    }

    public func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        if !request.imageFiles.isEmpty || request.multimodalMessages != nil {
            let softTokens = try imageSoftTokenCounts(request)
            try validateImageTokenBudget(request, softTokens: softTokens)
        }
        return ServerPreparedRequest(
            request: request,
            promptIDs: request.multimodalMessages == nil ? try renderPrompt(request) : nil)
    }

    private func imagePreprocessor(_ visionRuntime: VisionRuntime) -> Gemma4ImagePreprocessor {
        Gemma4ImagePreprocessor(device: context.device, config: visionRuntime.config)
    }

    /// The encode side of a planned image, bound to this session's runtime.
    private func encodeTurnImage(
        _ image: ServerRequestImages.Planned, visionRuntime: VisionRuntime
    ) throws -> VisionFeatures {
        switch ServerRequestImages.source(for: image) {
        case .plan(let plan):
            return try visionRuntime.encodeImage(
                plan: plan,
                languageModel: model,
                residencyPolicy: visionResidencyPolicy,
                checkCancellation: { try Task.checkCancellation() })
        case .reopened(let url):
            return try visionRuntime.encodeImage(
                at: url,
                languageModel: model,
                residencyPolicy: visionResidencyPolicy,
                checkCancellation: { try Task.checkCancellation() })
        }
    }

    /// Projected token count per image from headers alone; no pixel is decoded
    /// and no scan is walked.
    private func imageSoftTokenCounts(
        _ request: ValidatedChatRequest
    ) throws -> [Int] {
        guard let visionRuntime else {
            throw ServerRequestError.invalid(
                message: "image support is unavailable",
                param: "messages", code: "vision_unavailable")
        }
        let preprocessor = imagePreprocessor(visionRuntime)
        var counts: [Int] = []
        counts.reserveCapacity(request.imageFiles.count)
        for url in request.imageFiles.values {
            do {
                let geometry = try preprocessor.admissionGeometry(fileURL: url)
                counts.append(geometry.softTokenCount)
            } catch {
                throw ServerRequestImages.requestError(forUnreadable: error)
            }
        }
        return counts
    }

    /// Rejects an image request that cannot fit the context before any pixel is
    /// decoded or any tower runs.
    private func validateImageTokenBudget(
        _ request: ValidatedChatRequest, softTokens: [Int]
    ) throws {
        let textTokens = try renderPrompt(request).count
        guard ServerImageTokenBudget.fits(
            softTokenCounts: softTokens,
            textTokens: textTokens,
            maxContext: maxContext) else {
            throw ServerImageTokenBudget.rejection(
                imageCount: request.imageFiles.count,
                softTokenCounts: softTokens,
                textTokens: textTokens,
                maxContext: maxContext)
        }
    }

    /// The trailing user turn rendered as a continuation, with only its own
    /// images encoded. Token counts come from geometry, so the turn's shape is
    /// known before any encode.
    private func multimodalContinuation(
        request: ValidatedChatRequest,
        cachedMessageCount: Int
    ) async throws -> MultimodalPrefillInput {
        guard let messages = request.multimodalMessages, let visionRuntime,
              messages.count == cachedMessageCount + 2,
              let turn = messages.last, turn.role == .user,
              turn.toolCalls.isEmpty, turn.toolCallID == nil else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        var parts: [MultimodalContinuationPart] = []
        var urls: [URL] = []
        for part in turn.content {
            switch part {
            case .text(let value):
                parts.append(.text(value))
            case .image(let id):
                guard let url = request.imageFiles[id] else {
                    throw MultimodalPromptRendererError.missingImage(id)
                }
                parts.append(.image)
                urls.append(url)
            }
        }
        let images = try ServerRequestImages.plans(
            for: urls,
            with: imagePreprocessor(visionRuntime),
            checkCancellation: { try Task.checkCancellation() })
        let bridge = try gemmaTokenizer.encodeMultimodalUserContinuation(
            textAndImages: parts, imageTokenCounts: images.map(\.softTokenCount))
        var spans: [MultimodalImageSpan] = []
        for (range, image) in zip(bridge.imageTokenRanges, images) {
            try Task.checkCancellation()
            let features = try encodeTurnImage(image, visionRuntime: visionRuntime)
            guard features.tokenCount == range.count else {
                // Geometry disagreed with the encode; refuse rather than inject
                // a span of the wrong length.
                throw MultimodalPromptRendererError.placeholderMismatch
            }
            spans.append(MultimodalImageSpan(tokenRange: range, features: features))
        }
        return try MultimodalPrefillInput(
            effectiveTokenIDs: bridge.effectiveTokenIDs,
            embeddingTokenIDs: bridge.embeddingTokenIDs,
            imageSpans: spans)
    }

    private func renderMultimodal(
        _ request: ValidatedChatRequest
    ) async throws -> MultimodalPrefillInput {
        guard let messages = request.multimodalMessages, let visionRuntime else {
            throw ServerRequestError.invalid(
                message: "image support is unavailable",
                param: "messages", code: "vision_unavailable")
        }
        do {
            var features: [UUID: VisionFeatures] = [:]
            // Scoped, so the plans — up to `maximumOpenPlans` live ImageIO
            // descriptors — are released before the render walks the whole
            // history, rather than held across it.
            do {
                let planned: [(id: UUID, image: ServerRequestImages.Planned)]
                do {
                    planned = try ServerRequestImages.plannedEntries(
                        request.imageFiles,
                        with: imagePreprocessor(visionRuntime),
                        checkCancellation: { try Task.checkCancellation() })
                } catch {
                    throw ServerRequestImages.requestError(forUnreadable: error)
                }
                features.reserveCapacity(planned.count)
                for entry in planned {
                    try Task.checkCancellation()
                    features[entry.id] = try encodeTurnImage(
                        entry.image, visionRuntime: visionRuntime)
                }
            }
            return try MultimodalPromptRenderer.render(
                messages: messages,
                featuresByID: features,
                tokenizer: try gemmaTokenizer,
                tools: request.tools)
        } catch let error as MultimodalPromptRendererError {
            throw Self.clientError(for: error) ?? error
        }
    }

    /// Renderer failures a client can cause with its own message content are
    /// request rejections, not server errors; the rest stay opaque and surface
    /// as 500s because they mean the server's own layout went wrong.
    private static func clientError(
        for error: MultimodalPromptRendererError
    ) -> ServerRequestError? {
        switch error {
        case .reservedImageMarker:
            .invalid(
                message: "message text contains a reserved image placeholder token",
                param: "messages", code: "invalid_message")
        case .placeholderMismatch:
            .invalid(
                message: "message content could not be aligned with its images",
                param: "messages", code: "invalid_message")
        case .emptyMessages, .emptyContent:
            .invalid(
                message: "multimodal messages must not be empty",
                param: "messages", code: "invalid_message")
        case .missingImage, .unexpectedImage:
            nil
        }
    }

    public func generate(
        _ prepared: ServerPreparedRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let request = prepared.request
        var completionStarted = false
        var completed = false
        defer {
            // Rendering, planning and image encoding touch no KV state, so a
            // request that fails or is cancelled before the completion starts
            // leaves the cache exactly as the last completed request left it.
            // Image encoding is seconds long, so this window is not rare.
            if completionStarted, !completed {
                promptCache.invalidate()
                runner.reset()
            }
        }
        let needsToolTemplate = usesToolTemplate(request)
        let isMultimodal = request.multimodalMessages != nil
        if isMultimodal, visionRuntime == nil {
            throw ServerRequestError.invalid(
                message: "image support is unavailable",
                param: "messages", code: "vision_unavailable")
        }

        // Matched before any image is encoded. A multimodal hit continues from
        // the cached KV, so the images already inside that prefix are never
        // re-encoded; only a full-prefill miss needs them.
        let renderedTextIDs: [Int32]? = isMultimodal
            ? nil : try (prepared.promptIDs ?? renderPrompt(request))
        // Interim within S23: the Qwen prompt cache is wired by the
        // section's third step (reuse only as D6 allows). Until it is, a
        // Qwen request always takes the full-prefill path, which is always
        // legal: D6 guarantees a cold prefill works for every prompt.
        let promptCacheActive = promptCacheMode == .singlePrefix && family == .gemma4
        var cacheMatch: ServerPromptCacheMatch = .miss
        if promptCacheActive {
            cacheMatch = promptCache.match(
                domain: promptCacheDomain,
                request: request,
                renderedPromptIDs: renderedTextIDs,
                tokenizer: try gemmaTokenizer)
        }
        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        var multimodalInput: MultimodalPrefillInput?
        switch cacheMatch {
        case .hit(let effective, let cached):
            effectivePromptIDs = effective
            completionStart = .resume(cachedPromptTokens: cached)
            multimodalInput = nil
        case .renderThenResume(let cached):
            // Build the new turn as a continuation on the cached KV rather than
            // re-rendering history: the KV holds tokens the model generated, and
            // a fresh render re-tokenises that assistant text, so the two cannot
            // be compared. Only this turn's images are encoded.
            do {
                let bridge = try await multimodalContinuation(
                    request: request,
                    cachedMessageCount: promptCache.inputMessageCount ?? 0)
                effectivePromptIDs = (promptCache.kvBackedTokenIDs ?? [])
                    + bridge.effectiveTokenIDs
                multimodalInput = bridge
                completionStart = .resume(cachedPromptTokens: cached)
            } catch is CancellationError {
                // A disconnect mid-plan or mid-encode is not a bridge failure:
                // the cached prefix is untouched and stays valid for the retry.
                throw CancellationError()
            } catch {
                ServerLog.promptCacheMiss(reason: .continuationBridgeUnavailable)
                promptCache.invalidate()
                let rendered = try await renderMultimodal(request)
                effectivePromptIDs = rendered.effectiveTokenIDs
                multimodalInput = rendered
                completionStart = .reset
            }
        case .miss(let reason):
            if promptCacheActive {
                ServerLog.promptCacheMiss(reason: reason)
            }
            promptCache.invalidate()
            if let renderedTextIDs {
                effectivePromptIDs = renderedTextIDs
                multimodalInput = nil
            } else {
                let rendered = try await renderMultimodal(request)
                effectivePromptIDs = rendered.effectiveTokenIDs
                multimodalInput = rendered
            }
            completionStart = .reset
        }
        // What the template renders versus what actually prefills: on a text
        // cache hit the two genuinely differ, and that difference is the point
        // of the rendered/effective split in the failure diagnostics. A
        // multimodal resume has no full render to compare against, so the
        // effective ids stand in.
        let renderedPromptIDs = renderedTextIDs ?? effectivePromptIDs
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []

        // Gemma's structured decoder exists only where the tool template
        // was rendered (its boundary markers are the template's); Qwen's
        // stream parser runs for every Qwen request, tools or not, because
        // the template's think frame is always there and its splitter is
        // what keeps reasoning out of `content` (D11). The parser drops
        // reasoning: the OpenAI surface has no channel for it, exactly as
        // Gemma's thought channel is dropped today.
        let needsGemmaToolTemplate = family == .gemma4 && needsToolTemplate
        let gemmaDecoder = needsGemmaToolTemplate
            ? StructuredAssistantDecoder(
                tokenizer: try gemmaTokenizer,
                allowedTools: Set(request.tools.map(\.name)))
            : nil
        let qwenDecoder = family == .qwen35Moe
            ? QwenAssistantDecoder(
                allowedTools: Set(request.tools.map(\.name)),
                insideReasoning: request.enableThinking)
            : nil
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var content = ""
        var calls: [ParsedToolCall] = []
        var decodingError: Error?
        var shouldStop = false

        completionStarted = true
        let result = try await runRawCompletion(
            producer: runner,
            tokenizer: chatTokenizer,
            promptIds: effectivePromptIDs,
            multimodalInput: multimodalInput,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: completionStart,
            shouldStop: { shouldStop }) { progress in
                guard decodingError == nil else { return }
                do {
                    func handle(_ events: [StructuredAssistantEvent]) {
                        for event in events {
                            switch event {
                            case .content(let text):
                                let visible = stopMatcher.push(text)
                                if !visible.isEmpty {
                                    content += visible
                                    onEvent(.content(visible))
                                }
                                if stopMatcher.isStopped { shouldStop = true }
                            case .toolCall(let call):
                                calls.append(call)
                                onEvent(.toolCall(call))
                            }
                        }
                    }
                    switch progress {
                    case .prefill:
                        break
                    case .token(_, let tokenID, let delta):
                        let events: [StructuredAssistantEvent]
                        if let gemmaDecoder {
                            events = try gemmaDecoder.consume(tokenID: tokenID, delta: delta)
                        } else if let qwenDecoder {
                            events = try qwenDecoder.consume(delta)
                        } else {
                            events = delta.isEmpty ? [] : [StructuredAssistantEvent.content(delta)]
                        }
                        handle(events)
                    case .tail(let text):
                        // The flush tail is not tied to a token ID, so it must
                        // go through the decoder's channel state explicitly;
                        // appending it directly would leak text held back
                        // inside the thought channel or a tool call. The
                        // Qwen parser flushes at `finish` below, so here it
                        // only takes the tail as one more fragment.
                        let events: [StructuredAssistantEvent]
                        if let gemmaDecoder {
                            events = try gemmaDecoder.consumeTail(text)
                        } else if let qwenDecoder {
                            events = text.isEmpty ? [] : try qwenDecoder.consume(text)
                        } else {
                            events = text.isEmpty ? [] : [StructuredAssistantEvent.content(text)]
                        }
                        handle(events)
                    }
                } catch {
                    decodingError = error
                    shouldStop = true
                }
        }
        func structuredFailure(
            kind: StructuredOutputFailureKind,
            cause: StructuredOutputFailureCause
        ) throws -> StructuredOutputFailure {
            // The diagnostics name the boundary tokens of the family that
            // actually ran: both tokenizers carry the four marker IDs under
            // the same names, so the field shape is shared and only the
            // values are the family's own.
            let markers: (start: Int32, end: Int32, response: Int32, responseEnd: Int32)
            if let qwenTokenizer {
                markers = (qwenTokenizer.toolCallStartID, qwenTokenizer.toolCallEndID,
                           qwenTokenizer.toolResponseID, qwenTokenizer.toolResponseEndID)
            } else {
                let tokenizer = try gemmaTokenizer
                markers = (tokenizer.toolCallStartID, tokenizer.toolCallEndID,
                           tokenizer.toolResponseID, tokenizer.toolResponseEndID)
            }
            return StructuredOutputFailure(
                kind: kind,
                cause: cause,
                diagnostics: StructuredOutputFailureDiagnostics(
                    renderedPromptIDs: renderedPromptIDs ?? effectivePromptIDs,
                    effectivePromptIDs: effectivePromptIDs,
                    result: result,
                    maxCompletionTokens: config.maxNewTokens,
                    decodedCalls: calls.count,
                    visibleBytes: content.utf8.count,
                    stopStringMatched: stopMatcher.isStopped,
                    toolStartID: markers.start,
                    toolEndID: markers.end,
                    toolResponseID: markers.response,
                    toolResponseEndID: markers.responseEnd))
        }
        if let decodingError {
            throw try structuredFailure(
                kind: .decoderConsume,
                cause: .classify(decodingError))
        }
        do {
            try gemmaDecoder?.finish()
            // The Qwen parser's finish flush is the last word of the stream,
            // after the tail fragment, in that order (the CLI runs the same
            // sequence, S22): held text that never completed a tag surfaces
            // as content, and an unclosed tool-call block throws. The events
            // go through the same channel as the streamed ones; this mirrors
            // the `handle` inside the progress closure above, which cannot
            // leave that closure's region (Swift 6 region isolation).
            if let qwenDecoder {
                for event in try qwenDecoder.finish() {
                    switch event {
                    case .content(let text):
                        let visible = stopMatcher.push(text)
                        if !visible.isEmpty {
                            content += visible
                            onEvent(.content(visible))
                        }
                        if stopMatcher.isStopped { shouldStop = true }
                    case .toolCall(let call):
                        calls.append(call)
                        onEvent(.toolCall(call))
                    }
                }
            }
        } catch {
            throw try structuredFailure(
                kind: .decoderFinish,
                cause: .classify(error))
        }
        // The orphan-tool-response failure is a Gemma template fact: its
        // tool template parks the stream on the tool-response marker, so a
        // `.toolCalls` stop with nothing parsed can only be the marker with
        // no calls behind it. Qwen turns close on `im_end` and its parser
        // has already thrown on any malformed or unclosed call block.
        if needsGemmaToolTemplate, result.reason == .toolCalls, calls.isEmpty {
            throw try structuredFailure(kind: .orphanToolResponse, cause: .none)
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        // Publishing an image turn is safe now that the entry carries per-message
        // image identity: a later turn whose images differ is refused by
        // `imagesDiverged` rather than resumed onto a KV built from a different
        // picture.
        if promptCacheActive {
            let publication = promptCache.publish(
                domain: promptCacheDomain,
                request: request,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopMatcher.isStopped)
            ServerLog.promptCachePublication(publication)
        }
        completed = true
        return ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               totalTokens: result.prefillTokens + result.newTokens,
                               cachedTokens: result.cachedPromptTokens),
            prefillSeconds: result.prefillSeconds,
            decodeSeconds: result.decodeSeconds)
    }

    private func renderPrompt(_ request: ValidatedChatRequest) throws -> [Int32] {
        let promptIDs: [Int32]
        switch family {
        case .gemma4:
            if usesToolTemplate(request) {
                promptIDs = try gemmaTokenizer.encodeToolChat(
                    messages: request.messages,
                    tools: request.tools)
            } else {
                let rendered = try gemmaTokenizer.applyChatTemplate(request.messages)
                promptIDs = try gemmaTokenizer.encode(rendered, addBOS: false)
            }
        case .qwen35Moe:
            // D8: the pinned `chat_template.jinja` renders the whole
            // conversation, tool schemas included, so tools reach the model
            // as template text and need no special template of their own;
            // the conversation is encoded once either way. The thinking
            // frame follows the validated request (D11).
            let tokenizer = try qwenChatTokenizer
            if usesToolTemplate(request) {
                promptIDs = try tokenizer.encodeToolChat(
                    messages: request.messages,
                    tools: request.tools,
                    enableThinking: request.enableThinking)
            } else {
                let rendered = try tokenizer.applyChatTemplate(
                    request.messages, enableThinking: request.enableThinking)
                promptIDs = tokenizer.encode(rendered, addBOS: false)
            }
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return promptIDs
    }

    private func usesToolTemplate(_ request: ValidatedChatRequest) -> Bool {
        !request.tools.isEmpty || request.messages.contains {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }
    }
}
