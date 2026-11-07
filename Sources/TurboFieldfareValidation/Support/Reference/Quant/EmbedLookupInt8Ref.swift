import Foundation
import TurboFieldfare

/// FP32 reference for `embed_lookup_int8`:
/// `out = dequantize(table[tokenId]) * outScale`.
///
/// The table is `[V, D]` unsigned bytes with `[V, D/64]` BF16 scales and
/// biases (stored as `UInt16` bit patterns). The pointer form reads one row of
/// a vocabulary-sized table that stays in an `MTLBuffer`; the array form
/// bounds-checks small fixtures. Pass `sqrt(hidden_size)` as `outScale` to
/// match Gemma 4's post-embedding scale, or `1.0` for the raw dequant.
public enum EmbedLookupInt8Ref {
    public static func apply(table: UnsafePointer<UInt8>,
                             scales: UnsafePointer<UInt16>,
                             biases: UnsafePointer<UInt16>,
                             tokenId: Int,
                             d: Int,
                             outScale: Float) -> [Float] {
        precondition(d % Quantization.groupSize == 0,
                     "D must be a multiple of \(Quantization.groupSize)")
        precondition(tokenId >= 0, "token must be non-negative")
        let groupsPerRow = d / Quantization.groupSize
        let row = table + tokenId * d
        let rowScales = scales + tokenId * groupsPerRow
        let rowBiases = biases + tokenId * groupsPerRow

        var out = [Float](repeating: 0, count: d)
        for i in 0..<d {
            let g = i / Quantization.groupSize
            let scale = Quantization.bf16ToFloat(rowScales[g])
            let bias = Quantization.bf16ToFloat(rowBiases[g])
            out[i] = (Float(row[i]) * scale + bias) * outScale
        }
        return out
    }

    public static func apply(tablePacked: [UInt8],
                             tableScales: [UInt16],
                             tableBiases: [UInt16],
                             tokenId: Int,
                             d: Int,
                             outScale: Float) -> [Float] {
        let groupsPerRow = d / Quantization.groupSize
        precondition((tokenId + 1) * d <= tablePacked.count, "token out of range")
        precondition((tokenId + 1) * groupsPerRow <= tableScales.count, "scales out of range")
        precondition((tokenId + 1) * groupsPerRow <= tableBiases.count, "biases out of range")
        return tablePacked.withUnsafeBufferPointer { table in
            tableScales.withUnsafeBufferPointer { scales in
                tableBiases.withUnsafeBufferPointer { biases in
                    apply(table: table.baseAddress!,
                          scales: scales.baseAddress!,
                          biases: biases.baseAddress!,
                          tokenId: tokenId,
                          d: d,
                          outScale: outScale)
                }
            }
        }
    }
}
