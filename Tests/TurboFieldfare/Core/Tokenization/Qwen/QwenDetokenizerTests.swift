import Foundation
import Testing
@testable import TurboFieldfare

/// The streaming contract `QwenDetokenizer` adds over batch decode:
/// byte-level vocabulary tokens, held incomplete UTF-8, and barrier flushes.
@Suite(.enabled(if: QwenTestResources.hasTokenizer))
struct QwenDetokenizerTests {
    let tokenizer: QwenTokenizer

    init() async throws {
        tokenizer = try await QwenTokenizer.load(
            from: try #require(QwenTestResources.tokenizerFolder))
    }

    private func id(for token: String) throws -> Int32 {
        try #require(tokenizer.tokenizer.convertTokenToId(token).map(Int32.init),
                    "pinned vocab is missing the single-byte token \(token.unicodeScalars.first!.value)")
    }

    @Test("codepoints split across single-byte tokens decode whole")
    func splitCodepoints() throws {
        // 'é' is bytes C3 A9, '你' is E4 BD A0; every byte-level vocabulary
        // contains each single byte as its own token (the byte mapped through
        // the GPT-2 alphabet: C3→Ã, A9→©, E4→ä, BD→½, A0→ġ).
        let splits: [(String, [String])] = [
            ("\u{00E9}", ["\u{00C3}", "\u{00A9}"]),
            ("\u{4F60}", ["\u{00E4}", "\u{00BD}", "\u{0142}"]),
        ]
        for (expected, parts) in splits {
            let ids = try parts.map { try id(for: $0) }
            #expect(tokenizer.decode(ids, skipSpecialTokens: false) == expected)

            var stream = QwenDetokenizer(tokenizer: tokenizer, skipSpecialTokens: false)
            var pieces: [String] = []
            for id in ids { pieces.append(stream.push(id)) }
            let tail = stream.flush()
            // Nothing surfaces until the closing byte arrives…
            #expect(pieces.dropLast().allSatisfy { $0 == "" })
            // …and then the codepoint comes out complete.
            #expect(pieces.last == expected && tail == "")
        }
    }

    @Test("a barrier token flushes held bytes even though it is skipped")
    func barrierFlushes() throws {
        let partial = try id(for: "\u{00E4}")  // first byte of '你'
        var stream = QwenDetokenizer(tokenizer: tokenizer,
                                     skipSpecialTokens: true,
                                     barrierTokenIDs: [tokenizer.imEndID])
        #expect(stream.push(partial) == "")
        // The held byte surfaces at the barrier as U+FFFD, like the library's
        // lossy per-segment decode, while the skipped marker itself is "".
        #expect(stream.push(tokenizer.imEndID) == "\u{FFFD}")
        #expect(stream.push(tokenizer.imEndID) == "")
    }

    @Test("a non-special added token flushes the run and decodes literally")
    func addedTokenCutsRun() throws {
        let partial = try id(for: "\u{00E4}")
        var stream = QwenDetokenizer(tokenizer: tokenizer, skipSpecialTokens: true)
        #expect(stream.push(partial) == "")
        #expect(stream.push(tokenizer.thinkID) == "\u{FFFD}" + QwenTokenizer.thinkToken)
        #expect(stream.flush() == "")
    }

    @Test("unknown IDs contribute nothing and special IDs drop silently")
    func unknownAndSpecialIDs() {
        var stream = QwenDetokenizer(tokenizer: tokenizer, skipSpecialTokens: true)
        #expect(stream.push(999_999) == "")
        #expect(stream.push(tokenizer.endoftextID) == "")
        #expect(stream.flush() == "")
    }
}
