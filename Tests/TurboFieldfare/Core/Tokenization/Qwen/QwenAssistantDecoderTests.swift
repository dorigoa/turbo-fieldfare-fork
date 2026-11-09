import Foundation
import Testing
@testable import TurboFieldfare

/// End-to-end decoding: reasoning through the callback only, content and
/// tool calls as `StructuredAssistantEvent`s, with markers split at any
/// fragment boundary (plan D11, 3.6, S11 card).
@Suite
struct QwenAssistantDecoderTests {
    private struct Outcome: Equatable {
        var content = ""
        var reasoning = ""
        var calls: [ParsedToolCall] = []
    }

    /// Reference cell so the `@Sendable` reasoning callback can accumulate.
    private final class ReasoningBox: @unchecked Sendable {
        var text = ""
    }

    private func block(_ name: String, _ parameter: String) -> String {
        "\(QwenToolCallParser.callOpen)\n<function=\(name)>\n"
            + "<parameter=q>\n\(parameter)\n</parameter>\n</function>\n"
            + "\(QwenToolCallParser.callClose)"
    }

    private func run(_ fragments: [String], insideReasoning: Bool = true) throws
        -> Outcome {
        var decoder = QwenAssistantDecoder(allowedTools: ["search"],
                                           insideReasoning: insideReasoning,
                                           idGenerator: { "call_test" })
        var outcome = Outcome()
        let box = ReasoningBox()
        decoder.onReasoning = { box.text += $0 }
        for fragment in fragments {
            for event in try decoder.consume(fragment) {
                switch event {
                case .content(let text): outcome.content += text
                case .toolCall(let call): outcome.calls.append(call)
                }
            }
        }
        for event in try decoder.finish() {
            switch event {
            case .content(let text): outcome.content += text
            case .toolCall(let call): outcome.calls.append(call)
            }
        }
        outcome.reasoning = box.text
        return outcome
    }

    @Test("reasoning goes to the callback, content and calls to the events")
    func basicStream() throws {
        let outcome = try run(["Deep thought\n</think>\n\nCalling: ",
                               block("search", "Rome"),
                               " Done."])
        #expect(outcome.reasoning == "Deep thought\n")
        #expect(outcome.content == "Calling:  Done.")
        #expect(outcome.calls.count == 1)
        let call = try #require(outcome.calls.first)
        #expect(call.name == "search")
        #expect(call.arguments.objectValue?["q"] == .string("Rome"))
    }

    @Test("content before the call and several calls in one stream")
    func multipleCalls() throws {
        let outcome = try run(["before ", block("search", "one"),
                               " mid ", block("search", "two"), " after"],
                              insideReasoning: false)
        #expect(outcome.calls.count == 2)
        #expect(outcome.calls.map(\.name) == ["search", "search"])
        #expect(outcome.calls.compactMap { $0.arguments.objectValue?["q"] }
            == [.string("one"), .string("two")])
        #expect(outcome.content == "before  mid  after")
        let decoder = QwenAssistantDecoder(allowedTools: ["search"],
                                           insideReasoning: true)
        #expect(!decoder.hasToolCalls)
    }

    @Test("any split point of the stream yields the same outcome")
    func everySplitPosition() throws {
        let stream = "think\n</think>\n\nlead " + block("search", "x")
            + " mid " + block("search", "y") + " tail"
        let reference = try run([stream])
        for split in 0...stream.count {
            let index = stream.index(stream.startIndex, offsetBy: split)
            let outcome = try run([String(stream[..<index]),
                                   String(stream[index...])])
            #expect(outcome == reference, "split \(split)")
        }
        // Even one character per fragment.
        #expect(try run(stream.map(String.init)) == reference)
    }

    @Test("reasoning disabled: no callback, direct content")
    func reasoningDisabled() throws {
        // Prompt ended with the empty think block, so the stream is pure
        // content; a think tag is not searched for outside reasoning.
        let outcome = try run(["Just an ", "answer."], insideReasoning: false)
        #expect(outcome.reasoning == "")
        #expect(outcome.content == "Just an answer.")
        #expect(outcome.calls.isEmpty)
    }

    @Test("malformed tool XML throws and poisons the decoder")
    func malformedPoisons() throws {
        let broken = "\(QwenToolCallParser.callOpen)"
            + "<function=delete></function>\(QwenToolCallParser.callClose)"
        var decoder = QwenAssistantDecoder(allowedTools: ["search"],
                                           insideReasoning: false,
                                           idGenerator: { "call_test" })
        #expect(throws: QwenToolCallParserError.unknownTool("delete")) {
            _ = try decoder.consume(broken)
        }
        // Poisoned: every later call throws again.
        #expect(throws: QwenToolCallParserError.malformed) {
            _ = try decoder.consume("more")
        }
        #expect(throws: QwenToolCallParserError.malformed) {
            try decoder.finish()
        }
        #expect(!decoder.hasToolCalls)
    }

    @Test("a tool call left open at stop throws")
    func unclosedCallThrows() throws {
        var decoder = QwenAssistantDecoder(allowedTools: ["search"],
                                           insideReasoning: false,
                                           idGenerator: { "call_test" })
        _ = try decoder.consume("hi " + QwenToolCallParser.callOpen
                                + "<function=search></function>")
        #expect(throws: QwenToolCallParserError.malformed) {
            try decoder.finish()
        }
    }

    @Test("a half-open marker at stop surfaces as content")
    func heldOpenMarkerFlushesAsContent() throws {
        let outcome = try run(["wait <to"], insideReasoning: false)
        #expect(outcome.content == "wait <to")
    }
}
