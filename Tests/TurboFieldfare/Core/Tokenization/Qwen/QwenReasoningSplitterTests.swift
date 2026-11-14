import Foundation
import Testing
@testable import TurboFieldfare

/// Streaming reasoning/content split for Qwen (plan 3.6, D11): the splitter
/// must rejoin tags split at any fragment boundary and never leak think text
/// into content.
@Suite
struct QwenReasoningSplitterTests {
    private func run(_ fragments: [String], insideReasoning: Bool)
        -> (reasoning: String, content: String) {
        var splitter = QwenReasoningSplitter(insideReasoning: insideReasoning)
        var reasoning = ""
        var content = ""
        func take(_ segments: [QwenReasoningSplitter.Segment]) {
            for segment in segments {
                switch segment {
                case .reasoning(let text): reasoning += text
                case .content(let text): content += text
                }
            }
        }
        for fragment in fragments { take(splitter.consume(fragment)) }
        take(splitter.finish())
        return (reasoning, content)
    }

    @Test("a whole stream splits once the close tag arrives")
    func wholeStream() {
        let result = run(["Deep thought\n</think>\n\nThe answer."],
                         insideReasoning: true)
        #expect(result.reasoning == "Deep thought\n")
        #expect(result.content == "The answer.")
    }

    @Test("close tag split at every position rejoins identically")
    func everySplitPosition() throws {
        let stream = "Think</think>\n\nanswer"
        let reference = run([stream], insideReasoning: true)
        for split in 0...stream.count {
            let index = stream.index(stream.startIndex, offsetBy: split)
            let result = run([String(stream[..<index]), String(stream[index...])],
                             insideReasoning: true)
            #expect(result.reasoning == reference.reasoning, "split \(split)")
            #expect(result.content == reference.content, "split \(split)")
        }
    }

    @Test("every fragment boundary still splits correctly")
    func everyBoundaryOfEveryStream() throws {
        for stream in ["a</think>b", "x</think>\n\n", "</think>", "</think"] {
            let reference = run([stream], insideReasoning: true)
            for split in 0...stream.count {
                let index = stream.index(stream.startIndex, offsetBy: split)
                let result = run([String(stream[..<index]),
                                  String(stream[index...])],
                                 insideReasoning: true)
                #expect(result.reasoning == reference.reasoning,
                        "stream \(stream.debugDescription) split \(split)")
                #expect(result.content == reference.content,
                        "stream \(stream.debugDescription) split \(split)")
            }
        }
    }

    @Test("a disabled prompt (ending </think>\n\n) streams content only")
    func disabledStartsOutside() {
        let result = run(["Just an ", "answer."], insideReasoning: false)
        #expect(result.reasoning == "")
        #expect(result.content == "Just an answer.")
    }

    @Test("the </think> separator: newlines dropped, missing newlines kept")
    func separatorHandling() {
        #expect(run(["t</think>\n\nA"], insideReasoning: true).content == "A")
        // The separator is not there: nothing to drop.
        #expect(run(["t</think>A"], insideReasoning: true).content == "A")
        // Only one newline arrives: not the separator, stays content.
        #expect(run(["t</think>\nA"], insideReasoning: true).content == "\nA")
        // The separator split across three fragments is still dropped.
        #expect(run(["t</think>\n", "\n", "A"], insideReasoning: true).content == "A")
    }

    @Test("finish flushes an unfinished tag as ordinary text")
    func finishFlushesHeldText() {
        #expect(run(["Think</thin"], insideReasoning: true).reasoning == "Think</thin")
        #expect(run(["Think</think>\n"], insideReasoning: true).content == "\n")
        // Nothing leaks: a held '<' inside reasoning is flushed as reasoning.
        let result = run(["a < b"], insideReasoning: true)
        #expect(result.reasoning == "a < b")
    }
}
