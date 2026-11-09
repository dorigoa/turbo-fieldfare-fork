import Foundation

/// Turns streamed Qwen assistant text into the same events the Gemma decoder
/// produces (`StructuredAssistantEvent.content` / `.toolCall`).
///
/// The stream is first split by `QwenReasoningSplitter` into reasoning and
/// content. Reasoning never enters the event list (plan D11 and the S11 card
/// forbid new cases on the public enum): it surfaces through `onReasoning`.
/// Content is scanned for tool-call blocks (plan 3.6, XML format); each
/// complete block is parsed by `QwenToolCallParser` into one `.toolCall`
/// event, and the text around the blocks is emitted as `.content`.
///
/// Like `StructuredAssistantDecoder`, once an error is raised the decoder is
/// poisoned: every further call throws again. `finish()` throws if a tool
/// call block never closed — a generation cut off inside a call is malformed.
public final class QwenAssistantDecoder: @unchecked Sendable {
    private let allowedTools: Set<String>
    private let idGenerator: @Sendable () -> String
    private let splitter: QwenReasoningSplitter

    /// Tool-call scanning state, carried across fragments and segments.
    private enum Scan {
        /// Between blocks; `tail` holds a suffix that may still become the
        /// opening marker.
        case between(tail: String)
        /// Inside a block; `body` holds everything after the opening marker,
        /// `tail` a suffix that may still become the closing marker.
        case inside(body: String, tail: String)
    }

    private var scan: Scan = .between(tail: "")
    private var emittedCalls = 0
    private var failed = false

    /// Reasoning text, streamed through here instead of the event list.
    public var onReasoning: (@Sendable (String) -> Void)?

    /// - Parameters:
    ///   - allowedTools: names accepted for tool-call blocks.
    ///   - insideReasoning: whether the generation prompt ended with
    ///     `<think>\n` (reasoning enabled) or `</think>\n\n` (disabled).
    public init(allowedTools: Set<String>,
                insideReasoning: Bool,
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.allowedTools = allowedTools
        self.idGenerator = idGenerator
        self.splitter = QwenReasoningSplitter(insideReasoning: insideReasoning)
    }

    /// Feed one streamed text fragment; returns the events it completed.
    public func consume(_ fragment: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw QwenToolCallParserError.malformed }
        return try route(splitter.consume(fragment))
    }

    /// Flush at a stop boundary and return the final events: held text that
    /// never completed a tag surfaces as content. Throws if a tool call
    /// block is still open.
    public func finish() throws -> [StructuredAssistantEvent] {
        guard !failed else { throw QwenToolCallParserError.malformed }
        var events = try route(splitter.finish())
        switch scan {
        case .between(let tail):
            scan = .between(tail: "")
            if !tail.isEmpty { events.append(.content(tail)) }
        case .inside:
            failed = true
            throw QwenToolCallParserError.malformed
        }
        return events
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }

    private func route(_ segments: [QwenReasoningSplitter.Segment]) throws
        -> [StructuredAssistantEvent] {
        var events: [StructuredAssistantEvent] = []
        for segment in segments {
            switch segment {
            case .reasoning(let text):
                onReasoning?(text)
            case .content(let text):
                events += try scanContent(text)
            }
        }
        return events
    }

    private func scanContent(_ text: String) throws -> [StructuredAssistantEvent] {
        var events: [StructuredAssistantEvent] = []
        var input = text
        while !input.isEmpty {
            switch scan {
            case .between(var tail):
                tail += input
                input = ""
                if let open = tail.range(of: QwenToolCallParser.callOpen) {
                    let visible = String(tail[..<open.lowerBound])
                    if !visible.isEmpty { events.append(.content(visible)) }
                    scan = .inside(body: "", tail: String(tail[open.upperBound...]))
                } else {
                    let held = Self.heldTagTailLength(of: tail,
                                                      tag: QwenToolCallParser.callOpen)
                    let split = tail.index(tail.endIndex, offsetBy: -held)
                    let visible = String(tail[..<split])
                    if !visible.isEmpty { events.append(.content(visible)) }
                    scan = .between(tail: String(tail[split...]))
                }
            case .inside(var body, var tail):
                tail += input
                input = ""
                if let close = tail.range(of: QwenToolCallParser.callClose) {
                    body += String(tail[..<close.lowerBound])
                    scan = .between(tail: String(tail[close.upperBound...]))
                    do {
                        let call = try QwenToolCallParser().parse(
                            body, allowedTools: allowedTools, id: idGenerator())
                        emittedCalls += 1
                        events.append(.toolCall(call))
                    } catch {
                        failed = true
                        throw error
                    }
                } else {
                    let held = Self.heldTagTailLength(of: tail,
                                                      tag: QwenToolCallParser.callClose)
                    let split = tail.index(tail.endIndex, offsetBy: -held)
                    body += String(tail[..<split])
                    guard body.utf8.count <= QwenToolCallParser.maximumBytes else {
                        failed = true
                        throw QwenToolCallParserError.oversized
                    }
                    scan = .inside(body: body, tail: String(tail[split...]))
                }
            }
        }
        return events
    }

    /// The length of the longest suffix of `text` that is a proper prefix of
    /// `tag`, so a marker split across fragments waits for its last piece.
    private static func heldTagTailLength(of text: String, tag: String) -> Int {
        let characters = Array(text)
        let tagCharacters = Array(tag)
        var length = min(characters.count, tagCharacters.count - 1)
        while length > 0 {
            if Array(characters[(characters.count - length)...])
                == Array(tagCharacters[0..<length]) { break }
            length -= 1
        }
        return length
    }
}
