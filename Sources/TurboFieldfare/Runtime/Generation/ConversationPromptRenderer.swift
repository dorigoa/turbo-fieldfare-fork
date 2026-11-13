import Foundation

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
}
