import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// INT8 twins of the grouped routed-expert prefill phases
/// (`PrefillGroupedRoutedMoE(weightBits: 8)`): module contract, expert layout
/// and width selection. Execution and binding cases live in the extensions.
@Suite struct PrefillGroupedRoutedMoEInt8Tests {
    static let int8Kernels: Set<String> = [
        "prefill_grouped_routed_moe_batched_phase1_int8",
        "prefill_grouped_routed_moe_batched_down_int8",
    ]

    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func metalSource(_ relativePath: String) throws -> String {
        try String(contentsOf: repositoryRoot.appendingPathComponent(relativePath),
                   encoding: .utf8)
    }

    private static func captures(_ pattern: String, in source: String) throws -> [String] {
        let regex = try NSRegularExpression(pattern: pattern)
        let range = NSRange(source.startIndex..., in: source)
        return regex.matches(in: source, range: range).compactMap { match in
            Range(match.range(at: 1), in: source).map { String(source[$0]) }
        }
    }

    /// `[[buffer(N)]]` index for each parameter name of `kernel` in `source`.
    private static func bufferIndices(kernel: String, in source: String) throws -> [String: Int] {
        let header = try #require(
            try captures(#"kernel\s+void\s+"# + kernel + #"\s*\(([^{]*)\)\s*\{"#, in: source).first,
            "kernel \(kernel) not found")
        var indices: [String: Int] = [:]
        for name in try captures(#"(\w+\s*\[\[buffer\(\d+\)\]\])"#, in: header) {
            let parts = try captures(#"(\w+)\s*\[\["#, in: name)
            let index = try captures(#"buffer\((\d+)\)"#, in: name)
            if let parameter = parts.first, let value = index.first.flatMap(Int.init) {
                indices[parameter] = value
            }
        }
        return indices
    }

    // MARK: - Module contract (plan C4)

    @Test func int8ModuleDeclaresItsKernelsAndNoForeignConstants() throws {
        let source = try Self.metalSource("Sources/TurboFieldfare/Metal/Prefill/prefill_moe_int8.metal")
        let kernels = try Self.captures(#"kernel\s+void\s+(\w+)\s*\("#, in: source)
        #expect(Set(kernels) == Self.int8Kernels)
        #expect(kernels.count == Self.int8Kernels.count)

        let context = try MetalContext()
        for name in Self.int8Kernels {
            #expect(context.library.functionNames.contains(name), "\(name) missing")
        }
        // The 4-bit twins are still in the shared library next to them.
        #expect(context.library.functionNames.contains("prefill_grouped_routed_moe_batched_phase1"))
        #expect(context.library.functionNames.contains("prefill_grouped_routed_moe_batched_down"))

        let constants = try Self.captures(#"function_constant\((\d+)\)"#, in: source)
            .compactMap { Int($0) }
        #expect(constants.allSatisfy { (130...139).contains($0) }, "indices=\(constants)")
        let fileScopeConstants = try Self.captures(
            #"(?m)^constant\s+constexpr\s+\w+\s+(\w+)"#, in: source)
        #expect(!fileScopeConstants.isEmpty)
        #expect(fileScopeConstants.allSatisfy { $0.lowercased().contains("int8") },
                "\(fileScopeConstants)")
        let helpers = try Self.captures(#"(?m)^static\s+inline\s+[\w:<>]+\s+(\w+)\s*\("#, in: source)
        #expect(!helpers.isEmpty)
        #expect(helpers.allSatisfy { $0.contains("int8") }, "\(helpers)")
        let types = try Self.captures(#"(?m)^struct\s+(\w+)"#, in: source)
        #expect(types.isEmpty, "the module reuses the prefill.metal structs: \(types)")
    }

    @Test func int8KernelsKeepTheFourBitBufferIndices() throws {
        let fourBit = try Self.metalSource("Sources/TurboFieldfare/Metal/Prefill/prefill.metal")
        let eightBit = try Self.metalSource("Sources/TurboFieldfare/Metal/Prefill/prefill_moe_int8.metal")
        for phase in ["phase1", "down"] {
            let twin = "prefill_grouped_routed_moe_batched_\(phase)"
            let expected = try Self.bufferIndices(kernel: twin, in: fourBit)
            let actual = try Self.bufferIndices(kernel: twin + "_int8", in: eightBit)
            #expect(!expected.isEmpty)
            #expect(actual == expected, "\(phase): \(actual) != \(expected)")
        }
        // The Swift encoder binds these indices for both widths.
        #expect(PrefillGroupedRoutedMoEBufferIndex.hidden == 0)
        #expect(PrefillGroupedRoutedMoEBufferIndex.sortedPairs == 1)
        #expect(PrefillGroupedRoutedMoEBufferIndex.routePartials == 5)
        #expect(PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch == 7)
        #expect(PrefillGroupedRoutedMoEBufferIndex.downScratch == 8)
        #expect(PrefillGroupedRoutedMoEBufferIndex.expertArgumentState == 9)
        #expect(PrefillGroupedRoutedMoEBufferIndex.params == 10)
    }

    // MARK: - Width selection

    @Test func weightBitsDefaultToFourAndAcceptOnlyFourOrEight() throws {
        let context = try MetalContext()
        #expect(try PrefillGroupedRoutedMoE(context: context).weightBits == 4)
        #expect(try PrefillGroupedRoutedMoE(context: context, weightBits: 4).weightBits == 4)
        #expect(try PrefillGroupedRoutedMoE(context: context, weightBits: 8).weightBits == 8)
        #expect(PrefillGroupedRoutedMoE.supportedWeightBits == [4, 8])
        for bits in [0, 1, 2, 3, 5, 6, 7, 16, -4] {
            #expect(throws: PrefillGroupedRoutedMoEError.unsupportedWeightBits(bits)) {
                _ = try PrefillGroupedRoutedMoE(context: context, weightBits: bits)
            }
        }
        #expect(PrefillGroupedRoutedMoEError.unsupportedWeightBits(3).description
                    .contains("expected 4 or 8"))
    }

    // MARK: - Expert layout (plan §2.5)

    @Test func productionLayoutMatchesThePlan() {
        let layout = PrefillInt8ExpertPool.Layout(d: 2816, f: 704)
        #expect(PrefillInt8ExpertPool.offsetList(layout.offsets) == [
            0, 1_982_464, 2_044_416,
            2_106_368, 4_088_832, 4_150_784,
            4_212_736, 6_195_200, 6_257_152,
        ])
        #expect(layout.payloadBytes == 6_319_104)
        #expect(layout.stride == 6_324_224)
        #expect(PrefillInt8ExpertPool.offsetList(layout.offsets).allSatisfy { $0.isMultiple(of: 16) })
        #expect(PrefillGroupedRoutedMoE.int8ExpertOffsetsAreAligned(layout.offsets))
    }

    @Test func int8OffsetRuleNeedsFourByteWeightsAndTwoByteScales() {
        let base = PrefillInt8ExpertPool.Layout(d: 2816, f: 704).offsets
        func shifted(_ keyPath: WritableKeyPath<MoEExpertOffsets, UInt32>,
                     by delta: UInt32) -> MoEExpertOffsets {
            var offsets = base
            offsets[keyPath: keyPath] += delta
            return offsets
        }
        // Weights: any multiple of 4 is enough (plan C5); 2 and 1 are not.
        for keyPath in [\MoEExpertOffsets.gateWOff, \.upWOff, \.downWOff] {
            #expect(PrefillGroupedRoutedMoE.int8ExpertOffsetsAreAligned(shifted(keyPath, by: 4)))
            #expect(!PrefillGroupedRoutedMoE.int8ExpertOffsetsAreAligned(shifted(keyPath, by: 2)))
            #expect(!PrefillGroupedRoutedMoE.int8ExpertOffsetsAreAligned(shifted(keyPath, by: 1)))
        }
        // Scales and biases: BF16, so 2 bytes.
        for keyPath in [\MoEExpertOffsets.gateSOff, \.gateBOff, \.upSOff,
                        \.upBOff, \.downSOff, \.downBOff] {
            #expect(PrefillGroupedRoutedMoE.int8ExpertOffsetsAreAligned(shifted(keyPath, by: 2)))
            #expect(!PrefillGroupedRoutedMoE.int8ExpertOffsetsAreAligned(shifted(keyPath, by: 1)))
        }
    }
}
