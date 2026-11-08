import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// Structural checks for the 8-bit dense decode kernels: the wrappers satisfy
/// the plan's C3 protocols, and the shared Metal module keeps its function
/// constants inside S2's 100–109 range with `int8`-named symbols.
@Suite struct Int8DenseKernelContractTests {
    private static var moduleURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboFieldfare/Metal/Quant/dequant_int8_dense.metal")
    }

    @Test func wrappersConformToTheAffineContracts() throws {
        let context = try MetalContext()
        let embed: any AffineEmbedLookupEncoding = try EmbedLookupInt8(context: context)
        let qkv: any AffineFusedQKVEncoding = try FusedQKVGEMVInt8(context: context)
        let head: any AffineGreedyHeadEncoding = try LMHeadChainInt8(context: context,
                                                                     maxD: 64,
                                                                     maxVocab: 64)
        let finalRow: any AffinePrefillFinalRowHeadEncoding =
            try PrefillFinalRowHeadInt8(context: context, maxD: 64)
        let gemv: any AffineGEMVEncoding = try DequantInt8GEMV(context: context)
        #expect(embed is EmbedLookupInt8)
        #expect(qkv is FusedQKVGEMVInt8)
        #expect(head is LMHeadChainInt8)
        #expect(finalRow is PrefillFinalRowHeadInt8)
        #expect(gemv is DequantInt8GEMV)
    }

    @Test func moduleUsesOnlyItsFunctionConstantRange() throws {
        let source = try String(contentsOf: Self.moduleURL, encoding: .utf8)
        let pattern = try NSRegularExpression(pattern: #"function_constant\((\d+)\)"#)
        let range = NSRange(source.startIndex..., in: source)
        let indices = pattern.matches(in: source, range: range).compactMap { match in
            Range(match.range(at: 1), in: source).flatMap { Int(source[$0]) }
        }
        #expect(!indices.isEmpty)
        #expect(indices.allSatisfy { (100...109).contains($0) }, "indices=\(indices)")
        #expect(Set(indices).count == indices.count, "duplicate indices=\(indices)")
    }

    @Test func moduleKernelsCarryTheInt8Name() throws {
        let source = try String(contentsOf: Self.moduleURL, encoding: .utf8)
        let pattern = try NSRegularExpression(
            pattern: #"(?:kernel\s+void|\]\]\s*void)\s+(\w+)\s*\("#)
        let range = NSRange(source.startIndex..., in: source)
        let names = pattern.matches(in: source, range: range).compactMap { match in
            Range(match.range(at: 1), in: source).map { String(source[$0]) }
        }
        #expect(Set(names) == [
            "embed_lookup_int8",
            "dequant_int8_qkv_gemv_simd",
            "lm_head_greedy_int8_rows_chunk_raw",
        ])
        let context = try MetalContext()
        for name in names {
            #expect(context.library.functionNames.contains(name), "\(name) missing")
        }
    }
}
