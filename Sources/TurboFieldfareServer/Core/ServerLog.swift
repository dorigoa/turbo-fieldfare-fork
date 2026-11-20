import Foundation
import Synchronization

/// Where the server's log lines go (`--log-output`). Fatal errors, the usage
/// text and the `--host` exposure warning stay on stderr either way.
public enum ServerLogOutput: String, CaseIterable, Sendable {
    case stdout
    case stderr
}

public enum ServerLog {
    /// stdout by default, so `> server.log` captures the log with the ready
    /// line; stderr keeps the log apart from the ready line for a script that
    /// reads stdout.
    private static let output = Mutex(ServerLogOutput.stdout)

    public static func setOutput(_ destination: ServerLogOutput) {
        output.withLock { $0 = destination }
    }

    static var currentOutput: ServerLogOutput { output.withLock { $0 } }

    static func accepted(id: String, streaming: Bool) {
        write("request \(id) accepted streaming=\(streaming)")
    }

    static func prepared(id: String, promptTokens: Int?) {
        let count = promptTokens.map(String.init) ?? "backend-managed"
        write("request \(id) prepared prompt=\(count)")
    }

    static func queued(id: String) {
        write("request \(id) queued")
    }

    static func generating(id: String) {
        write("request \(id) generating")
    }

    /// A running request's progress over the last interval, as
    /// `RequestProgressLog` words it. Counts and speeds only.
    static func progress(_ line: String) {
        write(line)
    }

    static func completed(id: String,
                          duration: Duration,
                          completion: ServerCompletion) {
        write(completedMessage(id: id, duration: duration, completion: completion))
    }

    /// Split out so the throughput arithmetic can be checked without a log sink
    /// and without a decode. The prefill `tok/s` divides by the tokens actually
    /// prefilled: a prompt-cache hit contributes no prefill work, so counting
    /// cached tokens as prefilled would report a rate the runtime never
    /// achieved. Carries counts and durations only, never content.
    static func completedMessage(id: String,
                                 duration: Duration,
                                 completion: ServerCompletion) -> String {
        let usage = completion.usage
        // Clamped because both counts come from the runtime: a cached count
        // above the prompt would otherwise log a negative token count and a
        // negative rate rather than the zero work that was done.
        let computedPrefillTokens = max(
            usage.promptTokens - usage.promptTokensDetails.cachedTokens, 0)
        let prefillTokensPerSecond = completion.prefillSeconds > 0
            ? Double(computedPrefillTokens) / completion.prefillSeconds
            : 0
        let decodeTokensPerSecond = completion.decodeSeconds > 0
            ? Double(usage.completionTokens) / completion.decodeSeconds
            : 0
        return "request \(id) completed in \(format(duration)) "
            + "prompt=\(usage.promptTokens) "
            + "cached=\(usage.promptTokensDetails.cachedTokens) "
            + "completion=\(usage.completionTokens) "
            + "pp=\(formatSeconds(completion.prefillSeconds)) "
            + "tok/s=\(formatRate(prefillTokensPerSecond)) "
            + "tg=\(formatSeconds(completion.decodeSeconds)) "
            + "tok/s=\(formatRate(decodeTokensPerSecond)) "
            + "finish=\(completion.finishReason)"
    }

    /// Written once the weights are open and before the runtime is built,
    /// which allocates the KV cache and, for a variant this build cannot run,
    /// fails: either way the operator sees which variant was opened. Carries
    /// the variant and the manifest's model ID only.
    static func modelOpened(_ variant: ServerModelVariant) {
        write(modelOpenedMessage(variant))
    }

    static func modelOpenedMessage(_ variant: ServerModelVariant) -> String {
        // `family` is appended, so scripts reading the earlier fields keep
        // working (S23, D3: the operator must see which family was opened).
        "model opened variant=\(variant.label) "
            + "manifest_model_id=\(variant.manifestModelID) family=\(variant.family.rawValue)"
    }

    static func visionPackInvalid(at url: URL, error: Error) {
        write("vision pack at \(url.path) is invalid: \(String(reflecting: error))")
    }

    static func visionRuntimeUnsupported(at url: URL, error: Error) {
        write("vision runtime for pack at \(url.path) is unsupported: "
            + String(reflecting: error))
    }

    static func visionRuntimeUnsupported() {
        write("vision runtime is unsupported: the image tower requires an M2 or newer Mac")
    }

    /// The session is an actor, so generation is serialized: this line always
    /// belongs to the request between the preceding `generating` and the
    /// following `completed`. Carries a reason code only, never prompt content.
    static func promptCacheMiss(reason: ServerPromptCacheMissReason) {
        write(promptCacheMissMessage(reason: reason))
    }

    static func promptCacheMissMessage(reason: ServerPromptCacheMissReason) -> String {
        "prompt cache miss reason=\(reason.rawValue)"
    }

    static func promptCachePublication(_ result: ServerPromptCachePublishResult) {
        if let message = promptCachePublicationMessage(result) { write(message) }
    }

    static func promptCachePublicationMessage(_ result: ServerPromptCachePublishResult) -> String? {
        guard case .rejected(let reason) = result else { return nil }
        return "prompt cache publication rejected reason=\(reason.rawValue)"
    }

    /// A request whose client stopped listening. Kept distinct from `failed`
    /// because a cancel is not a server fault and should not read as one, and
    /// added because excluding it from logging altogether left the request's last
    /// line as `generating` forever: an operator could not tell a running request
    /// from an abandoned one, or from a crashed one. Carries no content.
    static func cancelled(id: String, phase: String, duration: Duration) {
        write(cancelledMessage(id: id, phase: phase, duration: duration))
    }

    /// Split out so the wording can be checked without a log sink: this line
    /// must stay distinguishable from `failed`, must name the phase, and must
    /// carry no prompt or generated content.
    static func cancelledMessage(id: String, phase: String, duration: Duration) -> String {
        "request \(id) cancelled by client in \(format(duration)) phase=\(phase)"
    }

    /// A request refused before it was accepted: a body that does not
    /// decode, a field the validator refuses, a route or method the server
    /// does not serve. Without this line such a request left no trace at all,
    /// and the client's error was the only record of it. The reason is the
    /// error message the client received, which names fields and never quotes
    /// request content.
    static func rejected(status: UInt,
                         method: String,
                         path: String,
                         error: OpenAIErrorEnvelope.Detail) {
        write(rejectedMessage(status: status, method: method, path: path, error: error))
    }

    static func rejectedMessage(status: UInt,
                                method: String,
                                path: String,
                                error: OpenAIErrorEnvelope.Detail) -> String {
        "request rejected status=\(status) method=\(method) path=\(path) "
            + "code=\(error.code)"
            + (error.param.map { " param=\($0)" } ?? "")
            + " reason=\(error.message)"
    }

    static func failed(id: String,
                       phase: String,
                       status: UInt,
                       error: Error) {
        write("request \(id) failed phase=\(phase) status=\(status) "
            + "error=\(String(reflecting: error))")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return formatSeconds(seconds)
    }

    private static func formatSeconds(_ seconds: Double) -> String {
        String(format: "%.3fs", seconds)
    }

    private static func formatRate(_ rate: Double) -> String {
        String(format: "%.3f", rate)
    }

    /// One `write` per line, so lines from concurrent requests never
    /// interleave mid-line, and nothing is held in a buffer: a redirected log
    /// is current while the server runs.
    private static func write(_ message: String) {
        let line = Data("[\(Date().formatted(.iso8601))] \(message)\n".utf8)
        switch currentOutput {
        case .stdout: FileHandle.standardOutput.write(line)
        case .stderr: FileHandle.standardError.write(line)
        }
    }
}
