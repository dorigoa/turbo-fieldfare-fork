import Foundation
import Tokenizers

/// Streaming detokenizer for Qwen generation loops (S9).
///
/// `QwenTokenizer.decode` is a push-loop over this type, so batch and
/// streaming decode agree by construction — the same shape as
/// `GFDetokenizer`, with the Gemma `Replace/ByteFallback/Fuse` chain replaced
/// by the GPT-2 byte-level map the Qwen ByteLevel decoder declares.
///
/// Semantics, matching the pinned tokenizer's decode:
///
/// 1. Every token string is a sequence of byte-level symbols (GPT-2's
///    `bytes_to_unicode` alphabet). Mapping each symbol back to its byte and
///    concatenating recovers the UTF-8 stream, so per-token fragments
///    compose position-independently.
/// 2. A multi-byte codepoint can split across tokens, so trailing bytes of
///    an incomplete UTF-8 sequence are held until the token that closes it.
///    This reproduces the library's `String(decoding:, as: UTF8.self)` over
///    each segment, byte for byte.
/// 3. `added_tokens` (all 33, special or not) are literals that cut a
///    run: the buffer is flushed, then the marker text is emitted. That
///    matches the library's ByteLevel decoder, which decodes segments
///    between added tokens separately — lossy per segment.
/// 4. `skipSpecialTokens` drops only the `added_tokens[special == true]`
///    IDs before decoding, again matching the library. Unknown IDs
///    (outside vocab) contribute nothing.
///
/// `barrierTokenIDs` flushes the held bytes at structural markers even when
/// the marker itself is skipped, so held text surfaces before the marker is
/// routed — the Qwen counterpart of `GFDetokenizer`'s barrier rule.
///
/// Cost is O(1) per token, independent of how much has been generated.
struct QwenDetokenizer {
    let tokenizer: any Tokenizer
    let skipSpecialTokens: Bool
    private let specialTokenIDs: Set<Int32>
    private let addedTokenIDs: Set<Int32>
    private let barrierTokenIDs: Set<Int32>
    /// Bytes of an incomplete UTF-8 sequence held back (at most 3).
    private var pendingBytes: [UInt8] = []

    init(tokenizer: QwenTokenizer,
         skipSpecialTokens: Bool = true,
         barrierTokenIDs: Set<Int32> = []) {
        self.tokenizer = tokenizer.tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        self.specialTokenIDs = tokenizer.specialTokenIDs
        self.addedTokenIDs = tokenizer.addedTokenIDs
        self.barrierTokenIDs = barrierTokenIDs
    }

    // MARK: - Byte-level alphabet (GPT-2 `bytes_to_unicode`)

    /// Printable bytes map to themselves; the other 68 map, in byte order,
    /// to the codepoints 256...323 (Latin Extended-A, no surrogates).
    static let byteToUnicode: [Character] = {
        var table: [Character] = []
        table.reserveCapacity(256)
        var extra: UInt32 = 256
        for byte in UInt8(0)...UInt8(255) {
            let printable = (33...126).contains(byte)
                || (161...172).contains(byte)
                || (174...255).contains(byte)
            if printable {
                table.append(Character(UnicodeScalar(byte)))
            } else {
                table.append(Character(UnicodeScalar(extra)!))
                extra += 1
            }
        }
        return table
    }()

    static let unicodeToByte: [UnicodeScalar: UInt8] = {
        var map: [UnicodeScalar: UInt8] = [:]
        for (byte, character) in byteToUnicode.enumerated() {
            for scalar in character.unicodeScalars { map[scalar] = UInt8(byte) }
        }
        return map
    }()

    private static let replacement = UnicodeScalar(0xFFFD)!

    // MARK: - Streaming

    /// Text contributed by `id`, ready to append to the stream.
    ///
    /// Returns `""` while a byte-level run is still open and valid; those
    /// bytes come out with the token that closes them, at an added token or
    /// barrier, or at `flush()`.
    mutating func push(_ id: Int32) -> String {
        // An unknown ID contributes nothing and leaves the run open, matching
        // the library, whose decode compactMap-drops unresolvable IDs.
        guard let token = tokenizer.convertIdToToken(Int(id)) else { return "" }
        if skipSpecialTokens, specialTokenIDs.contains(id) {
            return barrierTokenIDs.contains(id) ? flush() : ""
        }
        // Added tokens are emitted literally and cut the run, exactly like
        // the library's decoder splitting its lossy segments at them.
        if addedTokenIDs.contains(id) {
            return flush() + token
        }
        for scalar in token.unicodeScalars {
            if let byte = Self.unicodeToByte[scalar] {
                pendingBytes.append(byte)
            } else {
                // Defensive: the pinned vocabulary is entirely byte-level,
                // so this path is unreachable (the library force-crashes
                // here). Route any unexpected scalar's UTF-8 through the
                // same buffer.
                pendingBytes.append(contentsOf: Array(String(scalar).utf8))
            }
        }
        return drainCompleteUTF8()
    }

    /// Remainder held back at a stop boundary. An incomplete UTF-8 tail
    /// becomes U+FFFD, like the library's lossy per-segment decode.
    mutating func flush() -> String {
        guard !pendingBytes.isEmpty else { return "" }
        let text = String(decoding: pendingBytes, as: UTF8.self)
        pendingBytes.removeAll(keepingCapacity: true)
        return text
    }

    /// Emit the longest complete UTF-8 prefix of the pending bytes and hold
    /// back only the trailing incomplete sequence. Invalid bytes decode as
    /// U+FFFD with resynchronization at the next byte, which is what
    /// `String(decoding:, as: UTF8.self)` does over a whole segment;
    /// overlong encodings (impossible from real text) decode leniently.
    private mutating func drainCompleteUTF8() -> String {
        var output = ""
        let bytes = pendingBytes
        var index = 0
        while index < bytes.count {
            let lead = bytes[index]
            let width: Int
            switch lead {
            case 0x00...0x7F: width = 1
            case 0xC2...0xDF: width = 2
            case 0xE0...0xEF: width = 3
            case 0xF0...0xF4: width = 4
            default: width = 0
            }
            if width == 0 {  // invalid lead byte
                output.unicodeScalars.append(Self.replacement)
                index += 1
                continue
            }
            if index + width > bytes.count { break }  // incomplete tail: wait
            if width == 1 {
                output.append(Character(UnicodeScalar(lead)))
                index += 1
                continue
            }
            var value = UInt32(lead) & UInt32(0xFF >> (width + 1))
            var valid = true
            for byte in bytes[(index + 1)..<(index + width)] {
                guard byte & 0xC0 == 0x80 else { valid = false; break }
                value = (value << 6) | UInt32(byte & 0x3F)
            }
            if valid, let scalar = UnicodeScalar(value) {
                output.unicodeScalars.append(scalar)
                index += width
            } else {
                output.unicodeScalars.append(Self.replacement)
                index += 1
            }
        }
        if index > 0 { pendingBytes.removeFirst(index) }
        return output
    }
}
