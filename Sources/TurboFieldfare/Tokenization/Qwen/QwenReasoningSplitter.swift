import Foundation

/// Splits a streamed Qwen assistant response into reasoning and content.
///
/// The chat template (plan 3.6) emits reasoning as `<think>\n…\n</think>\n\n`
/// before the visible content, so the generation prompt always ends either
/// inside a think block (`<think>\n`, reasoning enabled) or after one
/// (`</think>\n\n`, reasoning disabled). The splitter is constructed with that
/// state and then fed arbitrary text fragments from the detokenizer.
///
/// A tag can split across any fragment boundary, so the tail of every
/// fragment that is a proper prefix of a tag is held back until the next
/// fragment proves or rejects it. `finish()` flushes held text as plain
/// reasoning or content: an unfinished tag is just text.
///
/// Invariant (plan D11): text between `<think>` and `</think>` never surfaces
/// as content.
final class QwenReasoningSplitter: @unchecked Sendable {
    enum Segment: Equatable, Sendable {
        case reasoning(String)
        case content(String)
    }

    static let thinkOpen = "<think>"
    static let thinkClose = "</think>"

    private enum State {
        /// Inside the think block, scanning for `</think>`.
        case inside
        /// `</think>` matched; swallowing the `\n\n` separator before content.
        case afterClose
        /// Visible content only.
        case outside
    }

    private var state: State
    /// Text held back at the end of the last fragment: a tag prefix while
    /// `inside`, the separator prefix while `afterClose`, always empty outside.
    private var pending = ""

    /// - Parameter insideReasoning: `true` when the generation prompt ended
    ///   with `<think>\n` (reasoning enabled), `false` when it ended with
    ///   `</think>\n\n` (reasoning disabled, plan 3.6).
    init(insideReasoning: Bool) {
        state = insideReasoning ? .inside : .outside
    }

    func consume(_ fragment: String) -> [Segment] {
        var segments: [Segment] = []
        var text = pending + fragment
        pending = ""
        while true {
            switch state {
            case .inside:
                guard let close = text.range(of: Self.thinkClose) else {
                    let held = Self.heldTagTailLength(of: text, tag: Self.thinkClose)
                    let split = text.index(text.endIndex, offsetBy: -held)
                    let emit = String(text[..<split])
                    if !emit.isEmpty { segments.append(.reasoning(emit)) }
                    pending = String(text[split...])
                    return segments
                }
                let reasoning = String(text[..<close.lowerBound])
                if !reasoning.isEmpty { segments.append(.reasoning(reasoning)) }
                text = String(text[close.upperBound...])
                state = .afterClose
            case .afterClose:
                if text.isEmpty { return segments }  // separator still pending
                if text.hasPrefix("\n\n") {
                    text = String(text.dropFirst(2))
                    state = .outside
                    continue
                }
                if text == "\n" {
                    pending = text  // second newline may arrive next fragment
                    return segments
                }
                // Not the separator: whatever it is becomes content.
                state = .outside
                continue
            case .outside:
                if !text.isEmpty { segments.append(.content(text)) }
                return segments
            }
        }
    }

    /// Flush at a stop boundary. Held text from an unfinished tag or an
    /// unfinished separator surfaces as ordinary reasoning or content.
    func finish() -> [Segment] {
        defer { pending = "" }
        switch state {
        case .inside:
            return pending.isEmpty ? [] : [.reasoning(pending)]
        case .afterClose:
            return pending.isEmpty ? [] : [.content(pending)]
        case .outside:
            return []
        }
    }

    /// The length of the longest suffix of `text` that is a proper prefix of
    /// `tag`, so that a tag split across fragments is not emitted before its
    /// last piece arrives.
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
