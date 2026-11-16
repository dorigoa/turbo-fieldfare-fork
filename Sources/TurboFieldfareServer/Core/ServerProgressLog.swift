import Foundation
import Synchronization

/// The progress of one running request, sampled into a log line at a fixed
/// interval (`--progress-log-interval`, 3 s by default).
///
/// The `completed` line gives a request's speed only once it ends, which for
/// a long reply or a long prompt can be minutes; this gives the speed of the
/// last interval while it runs. Each line covers exactly the time since the
/// previous one, so a stall shows as a slow interval rather than being
/// averaged away. Every count comes from the runtime's own progress reports:
/// prefill per chunk, decode per token.
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
        /// The start of the interval the next line reports: the previous
        /// line, or the first report of the current phase.
        var since: Sample?
    }

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
                    state.since = Sample(time: now, count: done)
                }
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

    /// The line for the interval that ends now, or nil before the runtime
    /// has reported anything.
    func tick(at now: ContinuousClock.Instant = .now) -> String? {
        state.withLock { state in
            guard let phase = state.phase, let since = state.since else { return nil }
            let window = Self.seconds(since.time.duration(to: now))
            guard window > 0 else { return nil }
            switch phase {
            case .prefill:
                let rate = Double(state.prefilled - since.count) / window
                state.since = Sample(time: now, count: state.prefilled)
                let total = state.promptTotal.map { "/\($0)" } ?? ""
                return "request \(id) prefill prompt=\(state.prefilled)\(total) "
                    + "pp_tok_s=\(Self.rate(rate)) window=\(Self.window(window))"
            case .decode:
                let rate = Double(state.decoded - since.count) / window
                state.since = Sample(time: now, count: state.decoded)
                var line = "request \(id) decode generated=\(state.decoded) "
                if let promptTotal = state.promptTotal {
                    line += "context=\(promptTotal + state.decoded)"
                        + (maxContextTokens.map { "/\($0)" } ?? "") + " "
                }
                return line + "tg_tok_s=\(Self.rate(rate)) window=\(Self.window(window))"
            }
        }
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
