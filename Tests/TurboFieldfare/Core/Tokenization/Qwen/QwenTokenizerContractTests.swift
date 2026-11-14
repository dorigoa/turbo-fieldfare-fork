import Foundation
import Hub
import Testing
@testable import TurboFieldfare

/// The surface later sections (S10-S12) consume: pinned special-token IDs,
/// stop tokens, skip semantics, and the fail-closed load paths.
@Suite(.enabled(if: QwenTestResources.hasTokenizer))
struct QwenTokenizerContractTests {
    let tokenizer: QwenTokenizer

    init() async throws {
        tokenizer = try await QwenTokenizer.load(
            from: try #require(QwenTestResources.tokenizerFolder))
    }

    @Test("special token IDs match the pinned tokenizer.json values")
    func pinnedIDs() {
        #expect(tokenizer.endoftextID == 248044)
        #expect(tokenizer.imStartID == 248045)
        #expect(tokenizer.imEndID == 248046)
        #expect(tokenizer.toolCallStartID == 248058)
        #expect(tokenizer.toolCallEndID == 248059)
        #expect(tokenizer.toolResponseID == 248066)
        #expect(tokenizer.toolResponseEndID == 248067)
        #expect(tokenizer.thinkID == 248068)
        #expect(tokenizer.thinkEndID == 248069)
        #expect(tokenizer.eosID == tokenizer.imEndID)
        #expect(tokenizer.padID == tokenizer.endoftextID)
        #expect(tokenizer.stopTokenIDs == [tokenizer.imEndID, tokenizer.endoftextID])
        #expect(tokenizer.vocabSize == 248_320)
    }

    @Test("structural markers are the eight turn markers, not endoftext")
    func structuralMarkers() {
        let markers = tokenizer.structuralMarkerIDs
        #expect(markers.count == 8)
        #expect(!markers.contains(tokenizer.endoftextID))
        #expect(markers.isSuperset(of: [tokenizer.imStartID, tokenizer.imEndID,
                                        tokenizer.thinkID, tokenizer.thinkEndID,
                                        tokenizer.toolCallStartID, tokenizer.toolCallEndID]))
    }

    @Test("tool/think markers are not special, so skip keeps them")
    func markersSurviveSkip() {
        let decoded = tokenizer.decode([tokenizer.thinkID, tokenizer.toolCallStartID],
                                       skipSpecialTokens: true)
        #expect(decoded == QwenTokenizer.thinkToken + QwenTokenizer.toolCallStartToken)
        // Special markers really are dropped in the same mode.
        #expect(tokenizer.decode([tokenizer.imStartID, tokenizer.imEndID],
                                 skipSpecialTokens: true) == "")
    }

    // MARK: - Fail-closed loading

    /// A `tokenizer.json` skeleton that passes `verifyPipeline`, so the
    /// added-token checks are the only thing that can reject it.
    private static func pinnedPipelineConfig(addedTokens: Config = []) -> Config {
        [
            "normalizer": ["type": "NFC"],
            "preTokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Split", "behavior": "Isolated", "invert": false,
                 "pattern": ["Regex": Config(QwenTokenizer.pinnedPretokenizeRegex)]],
                ["type": "ByteLevel", "addPrefixSpace": false,
                 "trimOffsets": false, "useRegex": false],
            ]],
            "model": ["type": "BPE"],
            "decoder": ["type": "ByteLevel", "addPrefixSpace": false,
                        "trimOffsets": false, "useRegex": false],
            "postProcessor": ["type": "ByteLevel", "useRegex": false],
            "addedTokens": addedTokens,
        ]
    }

    @Test("a foreign pipeline is rejected with a clear error")
    func rejectsForeignPipeline() {
        let wrongNormalizer: Config = ["normalizer": ["type": "NFD"]]
        #expect(throws: QwenTokenizerError.self) {
            try QwenTokenizer.verifyPipeline(wrongNormalizer)
        }
        let wrongDecoder: Config = [
            "normalizer": ["type": "NFC"],
            "preTokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Split"], ["type": "ByteLevel"]]],
            "model": ["type": "WordPiece"],
            "decoder": ["type": "WordPiece"],
            "postProcessor": ["type": "Template"],
        ]
        #expect(throws: QwenTokenizerError.self) {
            try QwenTokenizer.verifyPipeline(wrongDecoder)
        }
    }

    @Test("a valid pipeline missing a required marker is rejected")
    func rejectsMissingMarker() throws {
        let config = Self.pinnedPipelineConfig(addedTokens: [
            ["content": Config(QwenTokenizer.endoftextToken), "id": 248044, "special": true],
        ])
        #expect(throws: QwenTokenizerError.self) {
            _ = try QwenTokenizer(tokenizer: tokenizer.tokenizer, tokenizerData: config,
                chatFormat: QwenChatFormat(jinjaTemplate: ""))
        }
    }

    @Test("an out-of-range added-token ID is rejected, not trapped")
    func rejectsInvalidTokenID() throws {
        let config = Self.pinnedPipelineConfig(addedTokens: [
            ["content": Config(QwenTokenizer.imStartToken), "id": 248045, "special": true],
            ["content": Config(QwenTokenizer.imEndToken), "id": 3_000_000_000, "special": true],
        ])
        #expect(throws: QwenTokenizerError.self) {
            _ = try QwenTokenizer(tokenizer: tokenizer.tokenizer, tokenizerData: config,
                chatFormat: QwenChatFormat(jinjaTemplate: ""))
        }
    }
}
