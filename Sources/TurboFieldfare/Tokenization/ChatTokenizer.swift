import Foundation

/// A streaming detokenizer for generation loops: it emits each token's own
/// contribution to the output as it arrives and holds back text that is only
/// complete at a boundary, which `flush` returns. The family detokenizers
/// (`GFDetokenizer`, `QwenDetokenizer`) already have this exact shape, so
/// they adopt it with empty conformances and stay internal (D13).
public protocol StreamingDetokenizer {
    /// Text contributed by `id`, ready to append to the stream.
    mutating func push(_ id: Int32) -> String
    /// Remainder held back at a stop boundary.
    mutating func flush() -> String
}

/// The tokenizer surface `runRawCompletion`, the server, and the
/// conversations ask of a family (D13). Requirements carry no default
/// values on purpose: `addBOS` and `skipSpecialTokens` have different
/// defaults in the two families (Gemma decode skips specials, Qwen decode
/// keeps them), so callers pass them explicitly.
public protocol ChatTokenizer {
    var vocabSize: Int { get }
    var stopTokenIDs: Set<Int32> { get }
    var endOfTurnID: Int32 { get }
    var structuralMarkerIDs: Set<Int32> { get }
    /// The ID that means "waiting for a tool result": Gemma's
    /// `toolResponseID`; `nil` for Qwen, whose turn closes on `im_end`. A
    /// `toolResponseID` requirement cannot express this: both tokenizers
    /// already have a non-optional `toolResponseID` of a different meaning.
    var toolResultWaitID: Int32? { get }
    func encode(_ text: String, addBOS: Bool) -> [Int32]
    func decode(_ ids: [Int32], skipSpecialTokens: Bool) -> String
    /// The streaming detokenizer for generation, with the structural
    /// markers as barriers so held text never crosses a marker.
    func makeStreamingDetokenizer(barrierTokenIDs: Set<Int32>) -> any StreamingDetokenizer
}

extension GFDetokenizer: StreamingDetokenizer {}
extension QwenDetokenizer: StreamingDetokenizer {}

/// The existing members already satisfy the shared requirements; only the
/// two new-name requirements need witnesses (D13: nothing existing changes).
extension GFTokenizer: ChatTokenizer {
    public var toolResultWaitID: Int32? { toolResponseID }

    /// The shape `runRawCompletion` has always built: Gemma streaming skips
    /// special tokens.
    public func makeStreamingDetokenizer(barrierTokenIDs: Set<Int32>) -> any StreamingDetokenizer {
        GFDetokenizer(tokenizer: self, barrierTokenIDs: barrierTokenIDs)
    }
}

extension QwenTokenizer: ChatTokenizer {
    public var endOfTurnID: Int32 { imEndID }
    public var toolResultWaitID: Int32? { nil }

    /// Qwen generation must drop the special turn markers, so the
    /// detokenizer runs in skip mode even though `decode` keeps them by
    /// default; the barrier flush keeps held bytes from crossing a marker.
    public func makeStreamingDetokenizer(barrierTokenIDs: Set<Int32>) -> any StreamingDetokenizer {
        QwenDetokenizer(tokenizer: self, skipSpecialTokens: true,
                        barrierTokenIDs: barrierTokenIDs)
    }
}
