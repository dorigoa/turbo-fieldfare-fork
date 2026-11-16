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

    /// A decode line covers the time since the previous one, and the first
    /// line of a phase starts at that phase's first report. The prefill line
    /// gives the share done, the average rate over the chunks so far, and the
    /// time left.
    @Test func eachLineReportsTheIntervalSinceThePreviousOne() {
        let log = RequestProgressLog(id: "chatcmpl-1", maxContextTokens: 4_096)
        log.record(.prefill(done: 256, total: 1_000), at: at(0.5))
        log.record(.prefill(done: 768, total: 1_000), at: at(1.5))
        #expect(log.tick(at: at(2.5))
            == "request chatcmpl-1 prefill prompt=768/1000 (76%) pp_tok_s=512.0 eta=0s")

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

    /// The case that read as a loop: 256-token chunks landing every 5.4 s
    /// against a 3 s interval. Only intervals in which a chunk landed print a
    /// line, and the rate is the chunks' own, not 0.0 then 85.3.
    @Test func slowChunksPrintOneSteadyLinePerChunk() {
        let log = RequestProgressLog(id: "msg_1", maxContextTokens: 262_144)
        var lines: [String] = []
        var chunk = 0
        for tick in 1...12 {
            let now = Double(tick) * 3
            while Double(chunk + 1) * 5.4 <= now {
                chunk += 1
                log.record(.prefill(done: min(chunk * 256, 4_648), total: 4_648),
                           at: at(Double(chunk) * 5.4))
            }
            if let line = log.tick(at: at(now)) { lines.append(line) }
        }
        #expect(lines.count == chunk)
        #expect(lines.first == "request msg_1 prefill prompt=256/4648 (5%)")
        #expect(lines[1] == "request msg_1 prefill prompt=512/4648 (11%) pp_tok_s=47.4 eta=87s")
        #expect(lines.allSatisfy { !$0.contains("pp_tok_s=0.0") })
        #expect(lines.last == "request msg_1 prefill prompt=1536/4648 (33%) pp_tok_s=47.4 eta=66s")
    }

    /// With no chunk for 30 s the prefill says so, once per 30 s.
    @Test func aPrefillWithNoNewChunkSaysSo() {
        let log = RequestProgressLog(id: "r", maxContextTokens: nil)
        log.record(.prefill(done: 128, total: 1_024), at: at(0))
        #expect(log.tick(at: at(3)) == "request r prefill prompt=128/1024 (12%)")
        #expect(log.tick(at: at(6)) == nil)
        #expect(log.tick(at: at(30)) == "request r prefill prompt=128/1024 (12%) no new chunk for 30s")
        #expect(log.tick(at: at(33)) == nil)
        #expect(log.tick(at: at(60)) == "request r prefill prompt=128/1024 (12%) no new chunk for 60s")
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
