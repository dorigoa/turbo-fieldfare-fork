import Foundation
import TurboFieldfare

/// How deeply a reasoning family may think before it answers, as the Inspector
/// offers it (D11).
///
/// The three bounded levels are `ReasoningDepth`'s, so the app, the CLI and the
/// server all mean the same token counts by the same words. `unbounded` is the
/// fourth option and has no runtime budget at all: it is what every Qwen turn
/// did before the cap existed, kept reachable because a user who wants the
/// model to think as long as it likes should be able to say so.
public enum AppReasoningDepth: String, Codable, CaseIterable, Sendable, Identifiable {
    case low
    case medium
    case high
    case unbounded

    public var id: String { rawValue }

    /// The family default, and the depth a Qwen install ships with.
    ///
    /// Medium, not unbounded. S27 step 4 measured this checkpoint spending a
    /// whole 1024-token budget inside a think block that never closed, on all
    /// three benchmark prompts, writing no answer at all; in the app, where the
    /// response limit defaults to the whole remaining context, that is a turn
    /// that reasons for thousands of tokens and decays into repetition. A
    /// default that reproduces it would make the control pointless for the one
    /// surface where it is most needed.
    public static let familyDefault: AppReasoningDepth = .medium

    /// The runtime budget, or nil for an unbounded turn.
    public var budgetTokens: Int? {
        switch self {
        case .low: ReasoningDepth.low.budgetTokens
        case .medium: ReasoningDepth.medium.budgetTokens
        case .high: ReasoningDepth.high.budgetTokens
        case .unbounded: nil
        }
    }

    public var label: String {
        switch self {
        case .unbounded: "Unbounded"
        default: "\(rawValue.capitalized) (\(budgetTokens ?? 0) tokens)"
        }
    }
}
