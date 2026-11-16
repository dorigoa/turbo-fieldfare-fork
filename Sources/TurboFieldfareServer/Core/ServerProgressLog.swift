import Foundation
import Synchronization

/// The progress of one running request, sampled into a log line at a fixed
/// interval (`--progress-log-interval`, 3 s by default).
///
/// The `completed` line gives a request's speed only once it ends, which for
/// a long reply or a long prompt can be minutes; this reports progress while
/// it runs. Every count comes from the runtime's own progress reports:
/// prefill per chunk, decode per token.
///
/// The two phases are reported differently because they report at different
/// grains. Decode reports every token, so a decode line gives the rate over
/// exactly the interval since the previous line, and a stall shows as a slow
/// interval. Prefill reports once per chunk, and a chunk can take longer than
/// the interval: 256 tokens at 47 tok/s is 5.4 s against 3 s. Rated per
/// interval, that printed the same `prompt=` with `pp_tok_s=0.0` line after
/// line between chunks, which read as a prefill stuck in a loop. A prefill
/// line is therefore written only when a chunk has landed, its rate is the
/// average over the chunks so far, timed by when they landed, with the share
/// done and an estimate of the time left; a prefill with no new chunk for
/// 30 s says so instead.
final class RequestProgressLog: Sendable {
    private enum Phase {
        case prefill
        case decode
    }

    private struct Sample {
        let time: ContinuousClock.Instant
        let count: Int
    }

    private struct State {
        var phase: Phase?
        var promptTotal: Int?
        var prefilled = 0
        var decoded = 0
        /// The start of the interval the next decode line reports: the
        /// previous line, or the first report of the phase.
        var since: Sample?
        /// The first and the latest prefill report, which time the chunks.
        var prefillFirst: Sample?
        var prefillLatest: Sample?
        /// The prefill count and time of the last prefill line.
        var prefillLogged: Sample?
        /// When the current stretch without a chunk was last reported.
        var prefillStallReported: ContinuousClock.Instant?
    }

    /// How long a prefill may go without a new chunk before a line says so.
    static let prefillStallNotice: Duration = .seconds(30)

    let id: String
    let maxContextTokens: Int?
    private let state = Mutex(State())

    init(id: String, maxContextTokens: Int?) {
        self.id = id
        self.maxContextTokens = maxContextTokens
    }

    /// Takes a `.prefill` or `.decoded` event; anything else is ignored. The
    /// first report of a phase opens its interval, so the first line of a
    /// phase never counts time spent in the phase before it.
    func record(_ event: ServerInferenceEvent, at now: ContinuousClock.Instant = .now) {
        state.withLock { state in
            switch event {
            case .prefill(let done, let total):
                if state.phase != .prefill {
                    state.phase = .prefill
                    state.prefillFirst = Sample(time: now, count: done)
                    state.prefillLogged = nil
                }
                state.prefillLatest = Sample(time: now, count: done)
                state.prefilled = done
                state.promptTotal = total
            case .decoded(let count):
                if state.phase != .decode {
                    state.phase = .decode
                    state.since = Sample(time: now, count: count)
                }
                state.decoded = count
            case .content, .toolCall:
                break
            }
        }
    }

    /// The line for the interval that ends now, or nil when there is nothing
    /// new to say: before the runtime has reported anything, or during a
    /// prefill whose next chunk has not landed yet.
    func tick(at now: ContinuousClock.Instant = .now) -> String? {
        state.withLock { state in
            switch state.phase {
            case nil:
                return nil
            case .prefill?:
                return prefillLine(&state, now: now)
            case .decode?:
                guard let since = state.since else { return nil }
                let window = Self.seconds(since.time.duration(to: now))
                guard window > 0 else { return nil }
                let rate = Double(state.decoded - since.count) / window
                state.since = Sample(time: now, count: state.decoded)
                var line = "request \(id) decode generated=\(state.decoded) "
                if let promptTotal = state.promptTotal {
                    line += "context=\(promptTotal + state.decoded)"
                        + (maxContextTokens.map { "/\($0)" } ?? "") + " "
                }
                return line + "tok/s=\(Self.rate(rate)) window=\(Self.window(window))"
            }
        }
    }

    private func prefillLine(_ state: inout State, now: ContinuousClock.Instant) -> String? {
        guard let first = state.prefillFirst, let latest = state.prefillLatest else { return nil }
        if let logged = state.prefillLogged, logged.count == latest.count {
            // No chunk since the last line: stay quiet, unless the silence
            // has gone on long enough to be news itself.
            let quiet = latest.time.duration(to: now)
            guard quiet >= Self.prefillStallNotice else { return nil }
            if let reported = state.prefillStallReported,
               reported.duration(to: now) < Self.prefillStallNotice {
                return nil
            }
            state.prefillStallReported = now
            return "request \(id) prefill prompt=\(promptProgress(state)) "
                + "no new chunk for \(String(format: "%.0f", Self.seconds(quiet)))s"
        }
        state.prefillLogged = Sample(time: now, count: latest.count)
        state.prefillStallReported = nil
        var line = "request \(id) prefill prompt=\(promptProgress(state))"
        let elapsed = Self.seconds(first.time.duration(to: latest.time))
        if elapsed > 0, latest.count > first.count {
            let rate = Double(latest.count - first.count) / elapsed
            line += " pp_tok_s=\(Self.rate(rate))"
            if let total = state.promptTotal, total > latest.count {
                line += " eta=" + String(format: "%.0f", Double(total - latest.count) / rate) + "s"
            }
        }
        return line
    }

    private func promptProgress(_ state: State) -> String {
        guard let total = state.promptTotal, total > 0 else { return "\(state.prefilled)" }
        return "\(state.prefilled)/\(total) (\(state.prefilled * 100 / total)%)"
    }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
    }

    private static func rate(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    private static func window(_ seconds: Double) -> String {
        String(format: "%.2fs", seconds)
    }
}
