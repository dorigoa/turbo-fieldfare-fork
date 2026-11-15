import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server metrics")
struct ServerMetricsTests {
    private static func completion(prompt: Int,
                                   cached: Int,
                                   generated: Int,
                                   prefillSeconds: Double,
                                   decodeSeconds: Double,
                                   finishReason: String = "stop") -> ServerCompletion {
        ServerCompletion(content: "",
                         toolCalls: [],
                         finishReason: finishReason,
                         usage: OpenAIUsage(promptTokens: prompt,
                                            completionTokens: generated,
                                            totalTokens: prompt + generated,
                                            cachedTokens: cached),
                         prefillSeconds: prefillSeconds,
                         decodeSeconds: decodeSeconds)
    }

    private static func render(_ metrics: ServerMetrics,
                               running: Bool = false,
                               waiting: Int = 0,
                               memory: ServerProcessMemory? = nil) -> String {
        metrics.render(queueLimit: 4, running: running, waiting: waiting, memory: memory)
    }

    private static func lines(_ text: String) -> Set<String> {
        Set(text.split(separator: "\n").map(String.init))
    }

    @Test func freshServerReportsZeroesAndNoSpeeds() {
        let metrics = ServerMetrics(modelLabels: ["variant": "4-bit", "model": "gemma"],
                                    maxContextTokens: 16_384)
        let lines = Self.lines(Self.render(metrics))
        // Labels in key order, so the line is stable across scrapes.
        #expect(lines.contains(#"turbofieldfare_model_info{model="gemma",variant="4-bit"} 1"#))
        #expect(lines.contains("turbofieldfare_context_max_tokens 16384"))
        #expect(lines.contains("turbofieldfare_queue_limit 4"))
        #expect(lines.contains("turbofieldfare_requests_running 0"))
        #expect(lines.contains("turbofieldfare_requests_waiting 0"))
        for result in ["completed", "cancelled", "failed"] {
            #expect(lines.contains(#"turbofieldfare_requests_total{result="\#(result)"} 0"#))
        }
        for reason in ["stop", "length", "tool_calls"] {
            #expect(lines.contains(
                #"turbofieldfare_completions_total{finish_reason="\#(reason)"} 0"#))
        }
        #expect(lines.contains("turbofieldfare_generation_tokens_total 0"))
        #expect(lines.contains(#"turbofieldfare_request_duration_seconds_bucket{le="+Inf"} 0"#))
        // No request yet: a speed or a context fill of zero would be a
        // measurement nobody made, so the gauges are absent.
        #expect(!lines.contains { $0.hasPrefix("turbofieldfare_last_") })
        #expect(!lines.contains { $0.hasPrefix("turbofieldfare_start_time_seconds ") })
        #expect(!lines.contains { $0.hasPrefix("turbofieldfare_memory_") })
    }

    @Test func completionFeedsCountersSpeedsAndHistograms() {
        let metrics = ServerMetrics()
        metrics.recordCompletion(
            Self.completion(prompt: 1_000, cached: 200, generated: 100,
                            prefillSeconds: 2, decodeSeconds: 4),
            duration: .seconds(6),
            timeToFirstOutput: .milliseconds(2_500))
        let lines = Self.lines(Self.render(metrics, running: true, waiting: 2))
        #expect(lines.contains("turbofieldfare_requests_running 1"))
        #expect(lines.contains("turbofieldfare_requests_waiting 2"))
        #expect(lines.contains(#"turbofieldfare_requests_total{result="completed"} 1"#))
        #expect(lines.contains(#"turbofieldfare_completions_total{finish_reason="stop"} 1"#))
        #expect(lines.contains("turbofieldfare_prompt_tokens_total 1000"))
        #expect(lines.contains("turbofieldfare_prompt_cached_tokens_total 200"))
        #expect(lines.contains("turbofieldfare_prefill_tokens_total 800"))
        #expect(lines.contains("turbofieldfare_generation_tokens_total 100"))
        #expect(lines.contains("turbofieldfare_prefill_seconds_total 2.0"))
        #expect(lines.contains("turbofieldfare_decode_seconds_total 4.0"))
        #expect(lines.contains("turbofieldfare_last_context_tokens 1100"))
        // 800 prefilled tokens in 2 s: cached tokens are not prefill work.
        #expect(lines.contains("turbofieldfare_last_prefill_tokens_per_second 400.0"))
        #expect(lines.contains("turbofieldfare_last_decode_tokens_per_second 25.0"))
        // Buckets are cumulative and the bound itself is inclusive.
        #expect(lines.contains(#"turbofieldfare_time_to_first_output_seconds_bucket{le="1.0"} 0"#))
        #expect(lines.contains(#"turbofieldfare_time_to_first_output_seconds_bucket{le="2.5"} 1"#))
        #expect(lines.contains(#"turbofieldfare_time_to_first_output_seconds_bucket{le="+Inf"} 1"#))
        #expect(lines.contains("turbofieldfare_time_to_first_output_seconds_sum 2.5"))
        #expect(lines.contains(#"turbofieldfare_request_duration_seconds_bucket{le="5.0"} 0"#))
        #expect(lines.contains(#"turbofieldfare_request_duration_seconds_bucket{le="10.0"} 1"#))
        #expect(lines.contains("turbofieldfare_request_duration_seconds_count 1"))
    }

    /// A reply served entirely from the prompt cache prefilled nothing, and
    /// one cut off before its first token generated nothing: each keeps the
    /// last real speed instead of reporting zero.
    @Test func speedsMoveOnlyWhenThePhaseDidWork() {
        let metrics = ServerMetrics()
        metrics.recordCompletion(
            Self.completion(prompt: 500, cached: 0, generated: 50,
                            prefillSeconds: 1, decodeSeconds: 2),
            duration: .seconds(3), timeToFirstOutput: nil)
        metrics.recordCompletion(
            Self.completion(prompt: 500, cached: 500, generated: 0,
                            prefillSeconds: 0, decodeSeconds: 0,
                            finishReason: "length"),
            duration: .seconds(1), timeToFirstOutput: nil)
        let lines = Self.lines(Self.render(metrics))
        #expect(lines.contains("turbofieldfare_last_prefill_tokens_per_second 500.0"))
        #expect(lines.contains("turbofieldfare_last_decode_tokens_per_second 25.0"))
        #expect(lines.contains("turbofieldfare_last_context_tokens 500"))
        #expect(lines.contains(#"turbofieldfare_completions_total{finish_reason="length"} 1"#))
        // No output means no first-output observation.
        #expect(lines.contains("turbofieldfare_time_to_first_output_seconds_count 0"))
    }

    @Test func failuresAreCountedByStatusAndCancellationsApart() {
        let metrics = ServerMetrics()
        metrics.recordFailure(status: 429)
        metrics.recordFailure(status: 400)
        metrics.recordFailure(status: 429)
        metrics.recordCancellation()
        let lines = Self.lines(Self.render(metrics))
        #expect(lines.contains(#"turbofieldfare_requests_total{result="failed"} 3"#))
        #expect(lines.contains(#"turbofieldfare_requests_total{result="cancelled"} 1"#))
        #expect(lines.contains(#"turbofieldfare_request_failures_total{status="400"} 1"#))
        #expect(lines.contains(#"turbofieldfare_request_failures_total{status="429"} 2"#))
    }

    @Test func memoryAndStartTimeAppearOnceKnown() {
        let metrics = ServerMetrics()
        metrics.markStarted(at: Date(timeIntervalSince1970: 1_800_000_000))
        let memory = ServerProcessMemory(footprintBytes: 2_000,
                                         mappedResidentBytes: 30_000,
                                         residentBytes: 31_000)
        let lines = Self.lines(Self.render(metrics, memory: memory))
        #expect(lines.contains("turbofieldfare_start_time_seconds 1800000000.0"))
        #expect(lines.contains("turbofieldfare_memory_footprint_bytes 2000"))
        #expect(lines.contains("turbofieldfare_memory_mapped_resident_bytes 30000"))
        #expect(lines.contains("process_resident_memory_bytes 31000"))
        #expect(ServerProcessMemory.current() != nil)
    }

    @Test func labelValuesAreEscaped() {
        let metrics = ServerMetrics(modelLabels: ["model": "a\"b\\c\nd"])
        #expect(Self.render(metrics).contains(#"turbofieldfare_model_info{model="a\"b\\c\nd"} 1"#))
    }

    /// Every sample belongs to a family declared before it, each family is
    /// declared once, and every line parses as the text format expects.
    @Test func expositionIsWellFormed() throws {
        let metrics = ServerMetrics(modelLabels: ["model": "m"], maxContextTokens: 4_096)
        metrics.markStarted()
        metrics.recordCompletion(
            Self.completion(prompt: 10, cached: 2, generated: 3,
                            prefillSeconds: 0.01, decodeSeconds: 0.03),
            duration: .milliseconds(40), timeToFirstOutput: .milliseconds(15))
        metrics.recordFailure(status: 500)
        let text = Self.render(metrics, memory: .current())
        #expect(text.hasSuffix("\n"))
        let sample = try Regex(
            #"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\.)*"(?:,[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\.)*")*\})? (-?[0-9.]+(?:e[-+]?[0-9]+)?)$"#)
        var declared: [String: String] = [:]
        for line in text.split(separator: "\n").map(String.init) {
            if line.hasPrefix("# HELP ") { continue }
            if line.hasPrefix("# TYPE ") {
                let parts = line.split(separator: " ").map(String.init)
                #expect(parts.count == 4)
                #expect(declared[parts[2]] == nil, "family \(parts[2]) declared twice")
                declared[parts[2]] = parts[3]
                continue
            }
            let match = try #require(line.wholeMatch(of: sample), "malformed line: \(line)")
            let name = String(match.output[1].substring ?? "")
            let family = ["_bucket", "_sum", "_count"]
                .first { name.hasSuffix($0) && declared[String(name.dropLast($0.count))] == "histogram" }
                .map { String(name.dropLast($0.count)) } ?? name
            #expect(declared[family] != nil, "sample \(name) has no TYPE")
        }
    }

    @Test func runtimeLabelsAreGemmaOnly() throws {
        let arguments = try ServerArguments.parse(
            ["--model", "model.gturbo", "--early-expert-read", "router",
             "--expert-cache-slots", "32"],
            hostMemoryBytes: 64 << 30, environment: [:],
            readModelVariant: { _ in nil })
        let gemma = ServerModelVariant(
            profile: .affine4, manifestModelID: "mlx-community/gemma-4-26B-A4B-it-qat-4bit")
        let gemmaLabels = ServerMetrics.modelLabels(arguments: arguments, variant: gemma,
                                                    visionCapability: "ready")
        #expect(gemmaLabels["early_expert_read"] == "router")
        #expect(gemmaLabels["expert_cache_slots"] == "32")
        #expect(gemmaLabels["family"] == "gemma4")
        #expect(gemmaLabels["vision"] == "ready")
        #expect(gemmaLabels["prompt_cache"] == "single-prefix")

        // The Qwen runner takes no runtime configuration, so these flags are
        // not in effect there and are not reported.
        let qwen = ServerModelVariant(
            profile: .affine4, manifestModelID: "mlx-community/Qwen3.6-35B-A3B-4bit",
            family: .qwen35Moe)
        let qwenLabels = ServerMetrics.modelLabels(arguments: arguments, variant: qwen,
                                                   visionCapability: "missing")
        #expect(qwenLabels["early_expert_read"] == nil)
        #expect(qwenLabels["expert_cache_slots"] == nil)
        #expect(qwenLabels["family"] == ModelFamily.qwen35Moe.rawValue)
    }
}
