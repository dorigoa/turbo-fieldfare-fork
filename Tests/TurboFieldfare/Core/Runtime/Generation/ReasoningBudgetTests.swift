import Testing
import Foundation
@testable import TurboFieldfare

/// The reasoning cap's semantics, with no model and no GPU: the levels, the
/// budget value, and the four decisions recorded on `ReasoningBudgetGuard`.
@Suite struct ReasoningBudgetTests {
    /// The three levels and the S27-anchored token counts behind them.
    @Test func depthLevelsMapToTheMeasuredBudgets() {
        #expect(ReasoningDepth.low.budgetTokens == 256)
        #expect(ReasoningDepth.medium.budgetTokens == 1_024)
        #expect(ReasoningDepth.high.budgetTokens == 4_096)
        #expect(ReasoningDepth.allCases == [.low, .medium, .high])
    }

    /// A token count round-trips back to its level, and a count that is not a
    /// level answers nil rather than the nearest one.
    @Test func depthRoundTripsThroughItsBudget() {
        for depth in ReasoningDepth.allCases {
            #expect(ReasoningDepth(budgetTokens: depth.budgetTokens) == depth)
        }
        #expect(ReasoningDepth(budgetTokens: 500) == nil)
        #expect(ReasoningDepth(budgetTokens: 0) == nil)
    }

    /// A budget that cannot express a cap is nil, so a caller that fails to
    /// build one runs unbounded instead of failing the generation.
    @Test func budgetRefusesInputsThatCannotExpressACap() {
        #expect(ReasoningBudget(tokens: 0, endTokenID: 7,
                                endMarkerText: "</think>", closeTokenIDs: [7]) == nil)
        #expect(ReasoningBudget(tokens: -1, endTokenID: 7,
                                endMarkerText: "</think>", closeTokenIDs: [7]) == nil)
        #expect(ReasoningBudget(tokens: 4, endTokenID: 7,
                                endMarkerText: "</think>", closeTokenIDs: []) == nil)
        #expect(ReasoningBudget(tokens: 4, endTokenID: 7,
                                endMarkerText: "", closeTokenIDs: [7]) == nil)
        #expect(ReasoningBudget(tokens: 1, endTokenID: 7,
                                endMarkerText: "</think>", closeTokenIDs: [7]) != nil)
    }

    private static let marker = "</think>"
    private static let endID: Int32 = 7
    private static let closeIDs: [Int32] = [7, 8]

    private func budget(tokens: Int) -> ReasoningBudget {
        ReasoningBudget(tokens: tokens, endTokenID: Self.endID,
                        endMarkerText: Self.marker, closeIDs: Self.closeIDs)!
    }

    /// No budget means the guard never asks for anything: the unbounded path,
    /// which is every Gemma turn.
    @Test func noBudgetNeverForcesAnything() {
        var guardState = ReasoningBudgetGuard(budget: nil)
        #expect(!guardState.isWatching)
        for id in Int32(1)...Int32(50) {
            #expect(guardState.push(tokenID: id, text: "r").isEmpty)
        }
    }

    /// The cap fires on the token that reaches it, not before, and the frame it
    /// asks for is the budget's own close.
    @Test func capFiresOnTheTokenThatReachesIt() {
        var guardState = ReasoningBudgetGuard(budget: budget(tokens: 3))
        #expect(guardState.push(tokenID: 1, text: "a").isEmpty)
        #expect(guardState.push(tokenID: 2, text: "b").isEmpty)
        #expect(guardState.push(tokenID: 3, text: "c") == Self.closeIDs)
        #expect(!guardState.isWatching)
    }

    /// Decision 2: the cap fires at most once per turn. Whatever the model
    /// writes after the forced close is content, and no second frame is forced.
    @Test func capFiresAtMostOncePerTurn() {
        var guardState = ReasoningBudgetGuard(budget: budget(tokens: 1))
        #expect(guardState.push(tokenID: 1, text: "a") == Self.closeIDs)
        for id in Int32(2)...Int32(20) {
            #expect(guardState.push(tokenID: id, text: "more").isEmpty)
        }
    }

    /// A budget of one token still lets the model start reasoning: the cap is
    /// checked after a token is committed, never before the first one.
    @Test func budgetOfOneStillSamplesOneReasoningToken() {
        var guardState = ReasoningBudgetGuard(budget: budget(tokens: 1))
        #expect(guardState.isWatching)
        #expect(guardState.push(tokenID: 1, text: "a") == Self.closeIDs)
    }

    /// Decision 1: a block the model closes itself disarms the cap for good,
    /// through the marker's own vocabulary token.
    @Test func modelsOwnCloseTokenDisarmsTheCap() {
        var guardState = ReasoningBudgetGuard(budget: budget(tokens: 3))
        #expect(guardState.push(tokenID: 1, text: "a").isEmpty)
        #expect(guardState.push(tokenID: Self.endID, text: Self.marker).isEmpty)
        #expect(!guardState.isWatching)
        // Well past the budget: nothing is ever forced after the model's close.
        for id in Int32(10)...Int32(30) {
            #expect(guardState.push(tokenID: id, text: "answer").isEmpty)
        }
    }

    /// Decision 1, the other spelling: the marker written as ordinary text
    /// disarms the cap too, so a second close is never injected on top of the
    /// model's own — which the splitter would surface as visible content.
    @Test func markerSpelledAsTextDisarmsTheCap() {
        var guardState = ReasoningBudgetGuard(budget: budget(tokens: 3))
        #expect(guardState.push(tokenID: 100, text: "done " + Self.marker).isEmpty)
        #expect(!guardState.isWatching)
        #expect(guardState.push(tokenID: 101, text: "answer").isEmpty)
        #expect(guardState.push(tokenID: 102, text: "answer").isEmpty)
    }

    /// The text watch keeps a rolling tail, so a marker split across two
    /// tokens is still recognised as the model's own close.
    @Test func markerSplitAcrossTokensDisarmsTheCap() {
        var guardState = ReasoningBudgetGuard(budget: budget(tokens: 5))
        #expect(guardState.push(tokenID: 100, text: "</thi").isEmpty)
        #expect(guardState.push(tokenID: 101, text: "nk>").isEmpty)
        #expect(!guardState.isWatching)
        for id in Int32(102)...Int32(110) {
            #expect(guardState.push(tokenID: id, text: "x").isEmpty)
        }
    }

    /// A marker split so far apart that the rolling tail cannot hold it is not
    /// a close: the cap still fires. The tail is one character short of the
    /// marker by construction, which is what makes the split case above work
    /// without holding unbounded text.
    @Test func textThatOnlyResemblesTheMarkerDoesNotDisarmTheCap() {
        var guardState = ReasoningBudgetGuard(budget: budget(tokens: 3))
        #expect(guardState.push(tokenID: 100, text: "</think").isEmpty)
        #expect(guardState.push(tokenID: 101, text: "ing is fun").isEmpty)
        #expect(guardState.push(tokenID: 102, text: "x") == Self.closeIDs)
    }
}

private extension ReasoningBudget {
    /// Test-local shorthand so the suite reads as the semantics it checks.
    init?(tokens: Int, endTokenID: Int32, endMarkerText: String, closeIDs: [Int32]) {
        self.init(tokens: tokens, endTokenID: endTokenID,
                  endMarkerText: endMarkerText, closeTokenIDs: closeIDs)
    }
}
