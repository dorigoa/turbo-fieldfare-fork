import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server progress log")
struct ServerProgressLogTests {
    private let start = ContinuousClock.now

    private func at(_ seconds: Double) -> ContinuousClock.Instant {
        start.advanced(by: .milliseconds(Int64(seconds * 1_000)))
    }

    @Test func nothingIsLoggedBeforeTheRuntimeReports() {
        let log = RequestProgressLog(id: "r", maxContextTokens: 4_096)
        #expect(log.tick(at: at(3)) == nil)
        // Client-facing events are not progress.
        log.record(.content("hi"), at: at(1))
        #expect(log.tick(at: at(6)) == nil)
    }

    /// Each line covers the time since the previous one, and the first line
    /// of a phase starts at that phase's first report.
    @Test func eachLineReportsTheIntervalSinceThePreviousOne() {
        let log = RequestProgressLog(id: "chatcmpl-1", maxContextTokens: 4_096)
        log.record(.prefill(done: 256, total: 1_000), at: at(0.5))
        log.record(.prefill(done: 768, total: 1_000), at: at(1.5))
        #expect(log.tick(at: at(2.5))
            == "request chatcmpl-1 prefill prompt=768/1000 pp_tok_s=256.0 window=2.00s")

        log.record(.prefill(done: 1_000, total: 1_000), at: at(3))
        log.record(.decoded(count: 1), at: at(3.5))
        log.record(.decoded(count: 41), at: at(5))
        // Decode started at 3.5 s: the prefill before it is not decode time.
        #expect(log.tick(at: at(5.5))
            == "request chatcmpl-1 decode generated=41 context=1041/4096 "
            + "tok/s=20.0 window=2.00s")

        log.record(.decoded(count: 161), at: at(8))
        #expect(log.tick(at: at(8.5))
            == "request chatcmpl-1 decode generated=161 context=1161/4096 "
            + "tok/s=40.0 window=3.00s")
    }

    /// A stall is a slow interval, not something averaged away.
    @Test func aStallReadsAsZero() {
        let log = RequestProgressLog(id: "r", maxContextTokens: nil)
        log.record(.prefill(done: 10, total: 10), at: at(0))
        log.record(.decoded(count: 1), at: at(0.5))
        log.record(.decoded(count: 31), at: at(3))
        #expect(log.tick(at: at(3)) == "request r decode generated=31 context=41 "
            + "tok/s=12.0 window=2.50s")
        #expect(log.tick(at: at(6)) == "request r decode generated=31 context=41 "
            + "tok/s=0.0 window=3.00s")
    }

    /// A reply served from a fully cached prompt may report no prefill: the
    /// line then leaves out the context rather than guessing it.
    @Test func withoutPrefillReportsTheContextIsLeftOut() {
        let log = RequestProgressLog(id: "r", maxContextTokens: 4_096)
        log.record(.decoded(count: 1), at: at(0))
        log.record(.decoded(count: 7), at: at(1))
        #expect(log.tick(at: at(3)) == "request r decode generated=7 tok/s=2.0 window=3.00s")
    }

    @Test func intervalFlagDefaultsToThreeSeconds() throws {
        let parse = { (extra: [String]) in
            try ServerArguments.parse(["--model", "model.gturbo"] + extra,
                                      hostMemoryBytes: 64 << 30, environment: [:],
                                      readModelVariant: { _ in nil })
        }
        #expect(try parse([]).progressLogInterval == 3)
        #expect(try parse(["--progress-log-interval", "0"]).progressLogInterval == 0)
        #expect(try parse(["--progress-log-interval", "10"]).progressLogInterval == 10)
        for bad in ["-1", "3601", "1.5", "fast"] {
            #expect(throws: ServerArgumentError.invalid(
                "--progress-log-interval must be between 0 and 3600 seconds")) {
                _ = try parse(["--progress-log-interval", bad])
            }
        }
        #expect(ServerArguments.usage.contains("--progress-log-interval <0...3600>"))
    }
}
