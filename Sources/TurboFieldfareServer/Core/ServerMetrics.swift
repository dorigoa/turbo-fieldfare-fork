import Darwin
import Darwin.Mach
import Foundation
import Synchronization
import TurboFieldfare

/// What `GET /metrics` reports, in the Prometheus text exposition format.
///
/// Every figure comes from something the server already measures: the token
/// counts and the prefill and decode times are the ones `runRawCompletion`
/// returns and the `completed` log line prints, the queue state is the
/// coordinator's, and memory is read from the kernel at scrape time. Nothing
/// here times the runtime, so a scrape cannot slow generation down.
///
/// Counters are cumulative from the moment the server opened its port. Speeds
/// over a window are ratios of counters, for example
/// `rate(turbofieldfare_generation_tokens_total[5m])
///  / rate(turbofieldfare_decode_seconds_total[5m])`; the `last_*` gauges give
/// the same figures for the most recent request without a query.
///
/// The counters move only when a request completes, which for a long prompt or
/// a long reply can be minutes. The `running_*` gauges show the request that is
/// generating while it runs, from the runtime's progress events (prefill per
/// chunk, decode per token: the ones the progress log reads), so a scraper
/// polling every few seconds sees what the server is doing and how fast.
public final class ServerMetrics: Sendable {
    public static let contentType = "text/plain; version=0.0.4; charset=utf-8"

    /// Upper bounds for the time from accepting a request to its first output.
    static let firstOutputBuckets: [Double] = [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300]
    /// Upper bounds for the time from accepting a request to its completion.
    static let durationBuckets: [Double] = [1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1_800]
    static let finishReasons = ["stop", "length", "tool_calls"]

    struct Histogram: Equatable {
        let bounds: [Double]
        private(set) var bucketCounts: [UInt64]
        private(set) var sum: Double = 0
        private(set) var count: UInt64 = 0

        init(bounds: [Double]) {
            self.bounds = bounds
            self.bucketCounts = Array(repeating: 0, count: bounds.count)
        }

        mutating func observe(_ value: Double) {
            for index in bounds.indices where value <= bounds[index] {
                bucketCounts[index] += 1
            }
            sum += value
            count += 1
        }
    }

    struct State {
        var startTime: Double?
        var completed: UInt64 = 0
        var cancelled: UInt64 = 0
        var failed: UInt64 = 0
        var failuresByStatus: [UInt: UInt64] = [:]
        var finishReasons: [String: UInt64] = [:]
        var promptTokens: UInt64 = 0
        var cachedPromptTokens: UInt64 = 0
        var prefillTokens: UInt64 = 0
        var generationTokens: UInt64 = 0
        var prefillSeconds: Double = 0
        var decodeSeconds: Double = 0
        var lastContextTokens: Int?
        var lastPrefillTokensPerSecond: Double?
        var lastDecodeTokensPerSecond: Double?
        var firstOutput = Histogram(bounds: ServerMetrics.firstOutputBuckets)
        var duration = Histogram(bounds: ServerMetrics.durationBuckets)
        /// Requests that started generating, the running one included.
        var generationsStarted: UInt64 = 0
        /// The request generating now; nil when idle.
        var running: Running?
    }

    struct ProgressSample {
        let time: ContinuousClock.Instant
        let count: Int
    }

    /// The request generating now, as its progress events report it.
    /// Generation is serialized, so there is at most one.
    struct Running {
        let id: UInt64
        var decoding = false
        /// Prompt tokens in the KV so far, prompt-cache hits included, of the
        /// prompt's total: the runtime's own count, reported per chunk.
        var promptDone: Int?
        var promptTotal: Int?
        var generated = 0
        /// The first and the latest report of each phase, which time it. The
        /// first report opens the phase, so its tokens (the cached prefix, the
        /// first chunk) are not divided by time spent before it.
        var prefillFirst: ProgressSample?
        var prefillLatest: ProgressSample?
        var decodeFirst: ProgressSample?
        var decodeLatest: ProgressSample?

        /// Average over the chunks so far, as the progress log's prefill line.
        var prefillTokensPerSecond: Double? {
            Self.rate(prefillFirst, prefillLatest)
        }

        /// Average since the first generated token.
        var decodeTokensPerSecond: Double? {
            Self.rate(decodeFirst, decodeLatest)
        }

        private static func rate(_ first: ProgressSample?, _ latest: ProgressSample?) -> Double? {
            guard let first, let latest, latest.count > first.count else { return nil }
            let seconds = ServerMetrics.seconds(first.time.duration(to: latest.time))
            return seconds > 0 ? Double(latest.count - first.count) / seconds : nil
        }
    }

    private let state = Mutex(State())
    /// Labels of `turbofieldfare_model_info`, rendered in key order.
    let modelLabels: [String: String]
    let maxContextTokens: Int?

    public init(modelLabels: [String: String] = [:], maxContextTokens: Int? = nil) {
        self.modelLabels = modelLabels
        self.maxContextTokens = maxContextTokens
    }

    /// The labels `main` gives `turbofieldfare_model_info`. The expert-cache
    /// and early-read settings are Gemma's alone: the Qwen runner takes no
    /// runtime configuration, so reporting them for Qwen would name a setting
    /// that is not in effect.
    public static func modelLabels(arguments: ServerArguments,
                                   variant: ServerModelVariant,
                                   visionCapability: String) -> [String: String] {
        var labels = [
            "model": arguments.modelID,
            "variant": variant.label,
            "family": variant.family.rawValue,
            "manifest_model_id": variant.manifestModelID,
            "vision": visionCapability,
            "prompt_cache": arguments.promptCacheMode.rawValue,
        ]
        if variant.family == .gemma4 {
            labels["expert_cache_slots"] = String(arguments.expertCacheSlots)
            labels["early_expert_read"] = arguments.earlyExpertRead.rawValue
        }
        return labels
    }

    func markStarted(at date: Date = Date()) {
        state.withLock { $0.startTime = date.timeIntervalSince1970 }
    }

    /// A request that produced an answer. The prefill rate divides by the
    /// tokens actually prefilled, as the `completed` log line does, and the
    /// `last_*` rates move only when that phase did work: a reply with no
    /// prefilled or no generated token has no speed to report.
    func recordCompletion(_ completion: ServerCompletion,
                          duration: Duration,
                          timeToFirstOutput: Duration?) {
        let usage = completion.usage
        let cached = usage.promptTokensDetails.cachedTokens
        let computed = max(usage.promptTokens - cached, 0)
        state.withLock { state in
            state.completed += 1
            state.finishReasons[completion.finishReason, default: 0] += 1
            state.promptTokens += UInt64(max(usage.promptTokens, 0))
            state.cachedPromptTokens += UInt64(max(cached, 0))
            state.prefillTokens += UInt64(computed)
            state.generationTokens += UInt64(max(usage.completionTokens, 0))
            state.prefillSeconds += max(completion.prefillSeconds, 0)
            state.decodeSeconds += max(completion.decodeSeconds, 0)
            state.lastContextTokens = usage.promptTokens + usage.completionTokens
            if computed > 0, completion.prefillSeconds > 0 {
                state.lastPrefillTokensPerSecond = Double(computed) / completion.prefillSeconds
            }
            if usage.completionTokens > 0, completion.decodeSeconds > 0 {
                state.lastDecodeTokensPerSecond =
                    Double(usage.completionTokens) / completion.decodeSeconds
            }
            state.duration.observe(Self.seconds(duration))
            if let timeToFirstOutput {
                state.firstOutput.observe(Self.seconds(timeToFirstOutput))
            }
        }
    }

    func recordCancellation() {
        state.withLock { $0.cancelled += 1 }
    }

    /// A request starts generating: from now on it is the running one, in
    /// prefill until its first generated token. Returns the id `endRunning`
    /// takes.
    @discardableResult
    func beginRunning() -> UInt64 {
        state.withLock { state in
            state.generationsStarted += 1
            state.running = Running(id: state.generationsStarted)
            return state.generationsStarted
        }
    }

    /// A `.prefill` or `.decoded` event of the running request; anything else,
    /// or no running request, is ignored.
    func recordProgress(_ event: ServerInferenceEvent, at now: ContinuousClock.Instant = .now) {
        state.withLock { state in
            guard var running = state.running else { return }
            switch event {
            case .prefill(let done, let total):
                let sample = ProgressSample(time: now, count: done)
                if running.prefillFirst == nil { running.prefillFirst = sample }
                running.prefillLatest = sample
                running.promptDone = done
                running.promptTotal = total
            case .decoded(let count):
                let sample = ProgressSample(time: now, count: count)
                if running.decodeFirst == nil { running.decodeFirst = sample }
                running.decodeLatest = sample
                running.decoding = true
                running.generated = count
            case .content, .toolCall:
                return
            }
            state.running = running
        }
    }

    /// The request `id` stopped generating, however it ended. A later request
    /// that already began is left alone.
    func endRunning(_ id: UInt64) {
        state.withLock { state in
            if state.running?.id == id { state.running = nil }
        }
    }

    func recordFailure(status: UInt) {
        state.withLock {
            $0.failed += 1
            $0.failuresByStatus[status, default: 0] += 1
        }
    }

    /// The exposition text. `running`, `waiting` and `memory` are read by the
    /// caller at scrape time; everything else is the state recorded so far.
    func render(queueLimit: Int,
                running: Bool,
                waiting: Int,
                memory: ServerProcessMemory?) -> String {
        let state = self.state.withLock { $0 }
        var out = MetricsWriter()

        out.family("turbofieldfare_model_info", .gauge,
                   "The loaded model and the settings it runs with.")
        out.sample("turbofieldfare_model_info",
                   labels: modelLabels.sorted { $0.key < $1.key }, 1)
        if let maxContextTokens {
            out.family("turbofieldfare_context_max_tokens", .gauge,
                       "Context length the server was started with (--max-context).")
            out.sample("turbofieldfare_context_max_tokens", maxContextTokens)
        }
        out.family("turbofieldfare_queue_limit", .gauge,
                   "Requests that may wait behind the running one (--queue-limit).")
        out.sample("turbofieldfare_queue_limit", queueLimit)
        if let startTime = state.startTime {
            out.family("turbofieldfare_start_time_seconds", .gauge,
                       "Unix time at which the server opened its port, after the model loaded.")
            out.sample("turbofieldfare_start_time_seconds", startTime)
        }

        out.family("turbofieldfare_requests_running", .gauge,
                   "Requests generating now: 0 or 1, since generation is serialized.")
        out.sample("turbofieldfare_requests_running", running ? 1 : 0)
        out.family("turbofieldfare_requests_waiting", .gauge,
                   "Prepared requests waiting for the running one to finish.")
        out.sample("turbofieldfare_requests_waiting", waiting)

        out.family("turbofieldfare_generations_started_total", .counter,
                   "Requests that started generating, the running one included: "
                       + "when it changes, the running_* gauges are about another request.")
        out.sample("turbofieldfare_generations_started_total", state.generationsStarted)
        out.family("turbofieldfare_running_phase", .gauge,
                   "What the server is doing now, 1 on that phase: idle, prefill "
                       + "(reading the prompt) or decode (generating).")
        let phase = state.running.map { $0.decoding ? "decode" : "prefill" } ?? "idle"
        for name in ["idle", "prefill", "decode"] {
            out.sample("turbofieldfare_running_phase", labels: [("phase", name)],
                       name == phase ? 1 : 0)
        }
        if let running = state.running {
            if let total = running.promptTotal, let done = running.promptDone {
                out.family("turbofieldfare_running_prompt_tokens", .gauge,
                           "Prompt length of the running request.")
                out.sample("turbofieldfare_running_prompt_tokens", total)
                out.family("turbofieldfare_running_prompt_done_tokens", .gauge,
                           "Prompt tokens of the running request in the KV so far, "
                               + "prompt-cache hits included, reported per chunk.")
                out.sample("turbofieldfare_running_prompt_done_tokens", done)
            }
            out.family("turbofieldfare_running_generated_tokens", .gauge,
                       "Tokens the running request has generated so far, reasoning included.")
            out.sample("turbofieldfare_running_generated_tokens", running.generated)
            if let rate = running.prefillTokensPerSecond {
                out.family("turbofieldfare_running_prefill_tokens_per_second", .gauge,
                           "Prefill speed of the running request: the average over its "
                               + "chunks so far, timed by when they landed.")
                out.sample("turbofieldfare_running_prefill_tokens_per_second", rate)
            }
            if let rate = running.decodeTokensPerSecond {
                out.family("turbofieldfare_running_decode_tokens_per_second", .gauge,
                           "Generation speed of the running request since its first token.")
                out.sample("turbofieldfare_running_decode_tokens_per_second", rate)
            }
        }

        out.family("turbofieldfare_requests_total", .counter,
                   "Chat completion requests by how they ended.")
        out.sample("turbofieldfare_requests_total", labels: [("result", "completed")],
                   state.completed)
        out.sample("turbofieldfare_requests_total", labels: [("result", "cancelled")],
                   state.cancelled)
        out.sample("turbofieldfare_requests_total", labels: [("result", "failed")],
                   state.failed)
        out.family("turbofieldfare_request_failures_total", .counter,
                   "Failed chat completion requests by HTTP status.")
        for (status, count) in state.failuresByStatus.sorted(by: { $0.key < $1.key }) {
            out.sample("turbofieldfare_request_failures_total",
                       labels: [("status", String(status))], count)
        }
        out.family("turbofieldfare_completions_total", .counter,
                   "Completed requests by finish_reason.")
        let reasons = Set(Self.finishReasons).union(state.finishReasons.keys).sorted()
        for reason in reasons {
            out.sample("turbofieldfare_completions_total",
                       labels: [("finish_reason", reason)],
                       state.finishReasons[reason, default: 0])
        }

        out.family("turbofieldfare_prompt_tokens_total", .counter,
                   "Prompt tokens of completed requests, prompt-cache hits included.")
        out.sample("turbofieldfare_prompt_tokens_total", state.promptTokens)
        out.family("turbofieldfare_prompt_cached_tokens_total", .counter,
                   "Prompt tokens reused from the prompt cache instead of prefilled.")
        out.sample("turbofieldfare_prompt_cached_tokens_total", state.cachedPromptTokens)
        out.family("turbofieldfare_prefill_tokens_total", .counter,
                   "Prompt tokens actually prefilled: prompt minus cached.")
        out.sample("turbofieldfare_prefill_tokens_total", state.prefillTokens)
        out.family("turbofieldfare_generation_tokens_total", .counter,
                   "Tokens generated by completed requests.")
        out.sample("turbofieldfare_generation_tokens_total", state.generationTokens)
        out.family("turbofieldfare_prefill_seconds_total", .counter,
                   "Time the runtime spent prefilling prompts.")
        out.sample("turbofieldfare_prefill_seconds_total", state.prefillSeconds)
        out.family("turbofieldfare_decode_seconds_total", .counter,
                   "Time the runtime spent generating tokens.")
        out.sample("turbofieldfare_decode_seconds_total", state.decodeSeconds)

        if let tokens = state.lastContextTokens {
            out.family("turbofieldfare_last_context_tokens", .gauge,
                       "Context the last completed request filled: prompt plus generated tokens.")
            out.sample("turbofieldfare_last_context_tokens", tokens)
        }
        if let rate = state.lastPrefillTokensPerSecond {
            out.family("turbofieldfare_last_prefill_tokens_per_second", .gauge,
                       "Prefill speed of the last request that prefilled any token.")
            out.sample("turbofieldfare_last_prefill_tokens_per_second", rate)
        }
        if let rate = state.lastDecodeTokensPerSecond {
            out.family("turbofieldfare_last_decode_tokens_per_second", .gauge,
                       "Generation speed of the last request that generated any token.")
            out.sample("turbofieldfare_last_decode_tokens_per_second", rate)
        }

        out.histogram("turbofieldfare_time_to_first_output_seconds",
                      "From accepting a request to its first content or tool call, "
                          + "queueing and prefill included.",
                      state.firstOutput)
        out.histogram("turbofieldfare_request_duration_seconds",
                      "From accepting a request to its completion.",
                      state.duration)

        if let memory {
            out.family("turbofieldfare_memory_footprint_bytes", .gauge,
                       "Memory charged to the process (phys_footprint): runtime buffers "
                           + "and KV cache, without the mapped weights.")
            out.sample("turbofieldfare_memory_footprint_bytes", memory.footprintBytes)
            out.family("turbofieldfare_memory_mapped_resident_bytes", .gauge,
                       "File-backed pages resident in RAM: the mapped weights the kernel "
                           + "can drop under pressure.")
            out.sample("turbofieldfare_memory_mapped_resident_bytes",
                       memory.mappedResidentBytes)
            out.family("process_resident_memory_bytes", .gauge,
                       "Resident memory size in bytes.")
            out.sample("process_resident_memory_bytes", memory.residentBytes)
        }
        return out.text
    }

    fileprivate static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
    }
}

/// The process's memory as `task_info` reports it, read at scrape time.
/// The footprint is the measure the Mac app's HUD shows for the decode
/// service; adding the mapped-resident bytes counts each resident page once,
/// as `AppMemorySampler.occupiedSample` does.
struct ServerProcessMemory: Equatable, Sendable {
    let footprintBytes: UInt64
    let mappedResidentBytes: UInt64
    let residentBytes: UInt64

    static func current() -> ServerProcessMemory? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return ServerProcessMemory(footprintBytes: UInt64(info.phys_footprint),
                                   mappedResidentBytes: UInt64(info.external),
                                   residentBytes: UInt64(info.resident_size))
    }
}

/// The first output of a request, stamped by whichever event arrives first.
final class FirstOutputClock: Sendable {
    private let instant = Mutex<ContinuousClock.Instant?>(nil)

    func mark() {
        instant.withLock { if $0 == nil { $0 = .now } }
    }

    var value: ContinuousClock.Instant? { instant.withLock { $0 } }
}

private struct MetricsWriter {
    enum Kind: String {
        case counter
        case gauge
        case histogram
    }

    private(set) var text = ""

    mutating func family(_ name: String, _ kind: Kind, _ help: String) {
        text += "# HELP \(name) \(help)\n# TYPE \(name) \(kind.rawValue)\n"
    }

    mutating func sample(_ name: String,
                         labels: [(String, String)] = [],
                         _ value: Int) {
        line(name, labels: labels, String(value))
    }

    mutating func sample(_ name: String,
                         labels: [(String, String)] = [],
                         _ value: UInt64) {
        line(name, labels: labels, String(value))
    }

    mutating func sample(_ name: String,
                         labels: [(String, String)] = [],
                         _ value: Double) {
        line(name, labels: labels, Self.format(value))
    }

    mutating func histogram(_ name: String, _ help: String, _ histogram: ServerMetrics.Histogram) {
        family(name, .histogram, help)
        for (bound, count) in zip(histogram.bounds, histogram.bucketCounts) {
            line(name + "_bucket", labels: [("le", Self.format(bound))], String(count))
        }
        line(name + "_bucket", labels: [("le", "+Inf")], String(histogram.count))
        line(name + "_sum", labels: [], Self.format(histogram.sum))
        line(name + "_count", labels: [], String(histogram.count))
    }

    private mutating func line(_ name: String, labels: [(String, String)], _ value: String) {
        text += name
        if !labels.isEmpty {
            text += "{"
                + labels.map { "\($0.0)=\"\(Self.escape($0.1))\"" }.joined(separator: ",")
                + "}"
        }
        text += " \(value)\n"
    }

    /// Shortest round-trip form, which the Prometheus parser reads back
    /// exactly. Callers never pass a non-finite value.
    static func format(_ value: Double) -> String {
        value.isFinite ? "\(value)" : "0"
    }

    static func escape(_ value: String) -> String {
        value.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
    }
}
