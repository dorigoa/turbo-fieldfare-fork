import Foundation

/// How deeply a reasoning family may think before it has to answer.
///
/// The pinned Qwen3.6 chat template has no `thinking_budget` or
/// `reasoning_effort` variable — its only knob is the boolean
/// `enable_thinking` — so a depth control cannot be a prompt parameter. It is a
/// decode-time cap instead, and these are the levels the user-facing surfaces
/// offer. The numbers come from the S27 step 4 measurement: on this checkpoint
/// all three community-benchmark prompts spent a full 1024-token budget inside
/// a think block that never closed, writing zero bytes of answer. So 1024 is
/// the anchor — the figure the model is known not to finish inside — with a
/// quarter of it as the shallow level and four times it as the long leash.
public enum ReasoningDepth: String, CaseIterable, Sendable, Codable {
    case low
    case medium
    case high

    /// Tokens the model may spend inside its think block at this depth.
    public var budgetTokens: Int {
        switch self {
        case .low: 256
        case .medium: 1_024
        case .high: 4_096
        }
    }

    /// The level whose budget is `tokens`, or nil for a count that is not one
    /// of the three levels. Used where a stored or wire-level token count has
    /// to be shown back as a level.
    public init?(budgetTokens tokens: Int) {
        guard let match = Self.allCases.first(where: { $0.budgetTokens == tokens })
        else { return nil }
        self = match
    }
}

/// A cap on the tokens a turn may spend inside its reasoning block, together
/// with the tokens that close the block when the cap is reached.
///
/// Deliberately free of any notion of "think block": the generation loop stays
/// family-agnostic (D13) and only ever sees "count sampled tokens, and when the
/// count runs out force these tokens into the stream". The Qwen factory below
/// fills in what the frame actually is.
///
/// A budget is passed only for a turn whose generation prompt opened the block —
/// that is, reasoning on. Reasoning off and unbounded reasoning both pass nil,
/// which is why the absence of a budget is exactly the behaviour every surface
/// had before this type existed.
public struct ReasoningBudget: Sendable, Equatable {
    /// Tokens the model may sample inside the block. At least one: the cap is
    /// checked after a token is committed, so a turn always gets to start its
    /// reasoning, and a budget of zero is spelled "reasoning off" instead.
    public let tokens: Int
    /// The single vocabulary token that closes the block, watched in the
    /// sampled stream so a block the model closes itself disarms the cap.
    public let endTokenID: Int32
    /// The same close, as text. Watched alongside `endTokenID` because a
    /// byte-level vocabulary can spell the marker out instead of emitting its
    /// own token, and a second close injected after the model's own would
    /// surface to the reader as visible text.
    public let endMarkerText: String
    /// What to force into the stream when the cap is reached: the closing
    /// marker and the separator the chat template puts after it.
    public let closeTokenIDs: [Int32]

    /// - Returns: nil when the inputs cannot express a cap — a non-positive
    ///   budget, or a frame with nothing to close it — so a caller that cannot
    ///   build one simply runs unbounded instead of failing a generation.
    public init?(tokens: Int, endTokenID: Int32, endMarkerText: String,
                 closeTokenIDs: [Int32]) {
        guard tokens > 0, !closeTokenIDs.isEmpty, !endMarkerText.isEmpty
        else { return nil }
        self.tokens = tokens
        self.endTokenID = endTokenID
        self.endMarkerText = endMarkerText
        self.closeTokenIDs = closeTokenIDs
    }
}

public extension QwenTokenizer {
    /// The budget for a Qwen turn that may spend at most `tokens` reasoning.
    ///
    /// The forced close is `</think>` — one vocabulary token, which decodes as
    /// text in both skip modes, so `QwenReasoningSplitter` sees it — followed by
    /// the `\n\n` the template puts between the block and the answer. Injecting
    /// the separator as well as the marker leaves the splitter in its content
    /// state deterministically, instead of leaving it waiting on whatever the
    /// model happens to emit next.
    func reasoningBudget(tokens: Int) -> ReasoningBudget? {
        ReasoningBudget(tokens: tokens,
                        endTokenID: thinkEndID,
                        endMarkerText: Self.thinkEndToken,
                        closeTokenIDs: [thinkEndID] + encode("\n\n", addBOS: false))
    }

    /// The budget for a depth level.
    func reasoningBudget(depth: ReasoningDepth) -> ReasoningBudget? {
        reasoningBudget(tokens: depth.budgetTokens)
    }
}

/// The loop-local half of the budget: it watches one turn's sampled tokens and
/// says when the closing frame has to be forced.
///
/// Four decisions are recorded here because they are the whole behaviour of the
/// feature, and none of them is derivable from the types:
///
/// 1. **A block the model closes itself disarms the cap permanently.** The
///    first close observed — as `endTokenID`, or as the marker's text, tracked
///    with a rolling tail so a spelling split across tokens is still seen —
///    ends the watch. Nothing is ever forced afterwards.
/// 2. **The cap fires at most once per turn.** Reasoning the model writes after
///    the forced close is content, and it reaches the reader as content. The
///    runtime does not try to detect or suppress a re-opened block, which
///    matches `QwenReasoningSplitter`: it has no transition back into the block
///    either (D11).
/// 3. **The closing frame does not count toward the budget.** The budget limits
///    what the model spends reasoning; the frame is the runtime's own writing.
///    Those tokens do count toward `maxNewTokens`, because they are real tokens
///    in the KV and in the context.
/// 4. **`maxNewTokens` wins.** If the response limit is reached on the step the
///    cap would fire, the turn ends there with no answer, exactly as it does
///    today. A budget only helps when the response limit leaves room after it;
///    a budget at or above `maxNewTokens` is simply never reached, and no
///    reserve is carried to make room for one.
struct ReasoningBudgetGuard {
    private let budget: ReasoningBudget?
    private var spent = 0
    private var armed: Bool
    /// Tail of the decoded stream kept so the marker's text is still matched
    /// when it arrives split across two tokens.
    private var tail = ""

    init(budget: ReasoningBudget?) {
        self.budget = budget
        self.armed = budget != nil
    }

    /// Whether a close could still be forced for this turn.
    var isWatching: Bool { armed }

    /// Feed one *sampled* token and the text it decoded to. Forced tokens never
    /// come back through here: they are not the model's reasoning, and counting
    /// them would let the frame consume the budget it exists to end.
    ///
    /// - Returns: the tokens to force into the stream, or an empty array.
    mutating func push(tokenID: Int32, text: String) -> [Int32] {
        guard armed, let budget else { return [] }
        if tokenID == budget.endTokenID || closedInText(budget: budget, text: text) {
            armed = false
            return []
        }
        spent += 1
        guard spent >= budget.tokens else { return [] }
        armed = false
        return budget.closeTokenIDs
    }

    private mutating func closedInText(budget: ReasoningBudget, text: String) -> Bool {
        guard !text.isEmpty else { return false }
        let window = tail + text
        let keep = max(budget.endMarkerText.count - 1, 0)
        tail = String(window.suffix(keep))
        return window.contains(budget.endMarkerText)
    }
}
