import Foundation
import Synchronization

/// The prompt-render surface `MultimodalConversation` asks of a tokenizer
/// family: the tokens for a conversation's first user turn and for every
/// continuation after it (S25 step 2).
///
/// `ChatTokenizer` deliberately does not carry these (D13): on Gemma the two
/// render functions are members of `GFTokenizer` itself, and Qwen's versions
/// take an `enableThinking` switch the conversation has no business knowing.
/// The renderer owns that switch so one request-level setting reaches every
/// turn of the conversation.
public protocol ConversationPromptRenderer: Sendable {
    /// Tokens for the very first user turn of a conversation: the chat
    /// template's opening, the user text, and the generation prompt.
    func firstTurnTokens(text: String) throws -> [Int32]
    /// Tokens appending one more user turn to an existing conversation.
    func continuationTokens(userText: String) throws -> [Int32]
    /// A fresh per-turn assistant-text filter, or nil when the family's
    /// streamed text is already the visible answer (S25 step 5, D11).
    /// Qwen's think frame is a family fact of the template — and the filter's
    /// `insideReasoning` start state is the renderer's own `enableThinking`,
    /// so the two can never disagree.
    func makeAssistantDecoder() -> (any ConversationAssistantDecoder)?
}

public extension ConversationPromptRenderer {
    /// Families that stream text unchanged — today Gemma — get the pre-step-5
    /// path without touching their conformances.
    func makeAssistantDecoder() -> (any ConversationAssistantDecoder)? { nil }
}

/// The text-stream shape the conversation feeds one turn's deltas through:
/// the same surface `QwenAssistantDecoder` has always had (S11), seen as a
/// protocol so the conversation stays family-agnostic (D13).
public protocol ConversationAssistantDecoder: AnyObject {
    /// Reasoning text, reported as it is recognised instead of surfacing in
    /// the returned events (D11).
    var onReasoning: (@Sendable (String) -> Void)? { get set }
    /// Feed one streamed fragment; returns the visible events it completed.
    func consume(_ fragment: String) throws -> [StructuredAssistantEvent]
    /// Flush at the stop boundary; throws on a block that never closed.
    func finish() throws -> [StructuredAssistantEvent]
}

extension QwenAssistantDecoder: ConversationAssistantDecoder {}

/// One turn's D11 text pipeline: raw detokenizer deltas in, visible text out,
/// with the thinking block's open/closed state reported on the way.
///
/// The state transitions are its own reason to exist: the decoder's reasoning
/// callback is `@Sendable`, the run's progress closure captures the turn's
/// mutable text, and a block that the run never closed (stop, error, cut-off)
/// still has to end. A family with no decoder passes fragments through
/// untouched, so the Gemma path is byte-identical to before.
public final class ConversationThinkingStream: @unchecked Sendable {
    private let decoder: (any ConversationAssistantDecoder)?
    /// Whether the current turn is inside its thinking block.
    private let thinkingOpen = Mutex(false)
    private let onThinking: (@Sendable (Bool) -> Void)?

    public init(renderer: any ConversationPromptRenderer,
                onThinking: (@Sendable (Bool) -> Void)? = nil) {
        decoder = renderer.makeAssistantDecoder()
        self.onThinking = onThinking
        // Only the first reasoning fragment of a block opens the state; the
        // decoder calls back synchronously from `consume`, so the ordering
        // against the returned content is the stream's own. The lock lives on
        // `self` (a `Mutex` cannot be copied out), and the capture is weak:
        // the decoder is held by this stream, so a strong capture would close
        // a cycle over every turn's pipeline.
        decoder?.onReasoning = { [weak self] fragment in
            guard !fragment.isEmpty, let self else { return }
            let opened = self.thinkingOpen.withLock { state -> Bool in
                if state { return false }
                state = true
                return true
            }
            if opened { self.onThinking?(true) }
        }
    }

    /// Whether this family filters its stream. Gemma answers false and every
    /// call below is a pass-through, so callers keep forwarding the raw
    /// progress events — including empty deltas — exactly as before.
    public var isFiltering: Bool { decoder != nil }

    /// The visible text one raw delta produced: empty while a tag or the
    /// think block holds it. Throws on a malformed block; the decoder is
    /// poisoned afterwards, so callers must stop the turn.
    public func consume(_ fragment: String) throws -> String {
        guard let decoder else { return fragment }
        var visible = ""
        for event in try decoder.consume(fragment) {
            switch event {
            case .content(let text):
                visible += text
            case .toolCall:
                // No conversation path grants tools to this decoder, so a
                // completed block means the model emitted a tool call nobody
                // asked for: fail closed like the CLI (S22).
                throw QwenToolCallParserError.malformed
            }
        }
        if !visible.isEmpty { closeThinking() }
        return visible
    }

    /// The flush after the final tail fragment: held text that never completed
    /// a tag surfaces, and an unclosed block throws.
    public func finish() throws -> String {
        guard let decoder else { return "" }
        var visible = ""
        for event in try decoder.finish() {
            switch event {
            case .content(let text):
                visible += text
            case .toolCall:
                throw QwenToolCallParserError.malformed
            }
        }
        if !visible.isEmpty { closeThinking() }
        return visible
    }

    /// Ends the turn: a block the run never closed — stopped, failed, or cut
    /// off mid-thought — must not leave the app showing a stale indicator.
    public func closeTurn() { closeThinking() }

    private func closeThinking() {
        let closed = thinkingOpen.withLock { state -> Bool in
            if !state { return false }
            state = false
            return true
        }
        if closed { onThinking?(false) }
    }
}

/// Gemma: exactly the calls `MultimodalConversation` has always made inline
/// (same template, same `addBOS: false`, same continuation helper).
public struct GemmaConversationPromptRenderer: ConversationPromptRenderer {
    private let tokenizer: GFTokenizer

    public init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer
    }

    public func firstTurnTokens(text: String) throws -> [Int32] {
        tokenizer.encode(
            try tokenizer.applyChatTemplate(
                [GFTokenizer.Message(role: .user, content: text)]),
            addBOS: false)
    }

    public func continuationTokens(userText: String) throws -> [Int32] {
        tokenizer.encodeTextContinuation(userContent: userText)
    }
}

/// Qwen: the same shape, with the thinking switch baked into the rendered
/// prompt (D11). `enableThinking: false` asks the template for an answer-only
/// generation prompt.
public struct QwenConversationPromptRenderer: ConversationPromptRenderer {
    private let tokenizer: QwenTokenizer
    /// The switch `applyChatTemplate` and `encodeTextContinuation` both take.
    public let enableThinking: Bool

    public init(tokenizer: QwenTokenizer, enableThinking: Bool = true) {
        self.tokenizer = tokenizer
        self.enableThinking = enableThinking
    }

    public func firstTurnTokens(text: String) throws -> [Int32] {
        tokenizer.encode(
            try tokenizer.applyChatTemplate(
                [GFTokenizer.Message(role: .user, content: text)],
                enableThinking: enableThinking),
            addBOS: false)
    }

    public func continuationTokens(userText: String) throws -> [Int32] {
        tokenizer.encodeTextContinuation(userContent: userText,
                                         enableThinking: enableThinking)
    }

    /// The turn's think frame starts where this renderer's generation prompt
    /// ended (S25 step 5, D11): inside the block with thinking on, after it
    /// with thinking off. The app's conversation grants no tools, so a
    /// tool-call block the model emits anyway is a closed failure (as in the
    /// CLI's Qwen path).
    public func makeAssistantDecoder() -> (any ConversationAssistantDecoder)? {
        QwenAssistantDecoder(allowedTools: [], insideReasoning: enableThinking)
    }
}
