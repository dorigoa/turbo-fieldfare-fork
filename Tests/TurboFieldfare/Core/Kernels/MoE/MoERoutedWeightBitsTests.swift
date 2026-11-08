import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// `MoE(routedWeightBits:)` selects the routed kernels by width: 4 keeps the
/// INT4 path, 8 the INT8 twins of `moe_int8.metal`, anything else is refused.
@Suite struct MoERoutedWeightBitsTests {
    typealias Fixtures = MoERoutedFixtures

    private static var int8ModuleURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboFieldfare/Metal/MoE/moe_int8.metal")
    }

    @Test func rejectsUnsupportedWidths() throws {
        let context = try MetalContext()
        for bits in [0, 2, 3, 6, 16] {
            #expect(throws: MoEError.unsupportedRoutedWeightBits(bits)) {
                _ = try MoE(context: context, routedWeightBits: bits)
            }
        }
        #expect(try MoE(context: context).routedWeightBits == 4)
        for profile in QuantProfile.allCases {
            let kernel = try MoE(context: context, routedWeightBits: profile.routedWeightBits)
            #expect(kernel.routedWeightBits == profile.routedWeightBits)
            #expect(MoE.supportedRoutedWeightBits.contains(profile.routedWeightBits))
        }
    }

    struct FourBitShape: Sendable, CustomTestStringConvertible {
        let d: Int
        let f: Int
        var testDescription: String { "D=\(d) F=\(f)" }
    }

    /// The default and the explicit 4-bit kernel produce identical bits on
    /// INT4 blobs and match `MoeRef`, at a generic shape and at the
    /// production shape that selects the specialized INT4 pipelines.
    @Test(arguments: [FourBitShape(d: 128, f: 64), FourBitShape(d: 2816, f: 704)])
    func fourBitPathIsUnchanged(_ shape: FourBitShape) throws {
        let experts = Fixtures.experts(bits: 4, d: shape.d, f: shape.f,
                                       seed: 0x5401, label: "int4-routed-\(shape.d)")
        let inputs = Fixtures.inputs(d: shape.d, seed: 0x5402, label: "int4-routed-inputs")
        let blobs = experts.map { Fixtures.blob($0, layout: .packed) }
        let context = try MetalContext()

        let defaultDriver = try Fixtures.Driver(
            context: context, kernel: try MoE(context: context),
            blobs: blobs, inputs: inputs, d: shape.d, f: shape.f)
        let explicitDriver = try Fixtures.Driver(
            context: context, kernel: try MoE(context: context, routedWeightBits: 4),
            blobs: blobs, inputs: inputs, d: shape.d, f: shape.f)

        let byDefault = try defaultDriver.runFull()
        let explicit = try explicitDriver.runFull()
        let split = try explicitDriver.runSplit(hits: [0, 2, 5, 7], misses: [1, 3, 4, 6])
        let expected = Fixtures.reference(
            experts: experts, x: inputs.x, residual: inputs.residual,
            routingWeights: inputs.routingWeights, d: shape.d, f: shape.f)

        #expect(explicit.acts == byDefault.acts)
        #expect(explicit.y == byDefault.y)
        #expect(split == byDefault.y)
        let error = RelError.compute(actual: byDefault.y, reference: expected)
        #expect(error < Tolerance.fp16ChainedReduction, "rel=\(error)")
    }

    /// Both widths bind the same `RoutedBlobs` argument buffer, so
    /// `makeRoutedArgumentBuffer` and `makeReusedRoutedArgumentBuffer` keep
    /// one layout.
    @Test func bothWidthsShareTheArgumentBufferLayout() throws {
        let context = try MetalContext()
        let names = [
            "moe_phase1_gate_up_act_u16load",
            "moe_phase1_gate_up_act_subset_u16load",
            "moe_phase2_down_reduce_k8",
            "moe_phase1_gate_up_act_int8",
            "moe_phase1_gate_up_act_subset_int8",
            "moe_phase2_down_reduce_k8_int8",
        ]
        let lengths = try names.map { name in
            guard let function = context.library.makeFunction(name: name) else {
                throw MetalError.missingFunction(name)
            }
            return function.makeArgumentEncoder(bufferIndex: 0).encodedLength
        }
        #expect(Set(lengths).count == 1, "lengths=\(lengths)")

        guard let blob = context.device.makeBuffer(length: 64, options: .storageModeShared)
        else { throw MetalError.noDevice }
        let blobs = [MTLBuffer](repeating: blob, count: Fixtures.topK)
        let four = try MoE(context: context, routedWeightBits: 4)
        let eight = try MoE(context: context, routedWeightBits: 8)
        let fourBuffer = four.makeRoutedArgumentBuffer(routedBlobs: blobs, topK: 8)
        let eightBuffer = eight.makeRoutedArgumentBuffer(routedBlobs: blobs, topK: 8)
        #expect(fourBuffer?.length == eightBuffer?.length)
        #expect(eight.makeReusedRoutedArgumentBuffer(routedBlobs: blobs, topK: 8).length
            == lengths[0])
    }

    /// Plan C5: INT8 weights need 4-byte offsets, scales and biases 2-byte.
    @Test func int8OffsetAlignmentRule() {
        #expect(MoE.int8RoutedOffsetsAreAligned(Fixtures.Production8Bit.offsets))
        let shifted = MoEExpertOffsets(
            gateWOff: 20, gateSOff: 98, gateBOff: 130,
            upWOff: 164, upSOff: 226, upBOff: 258,
            downWOff: 292, downSOff: 354, downBOff: 386)
        #expect(MoE.int8RoutedOffsetsAreAligned(shifted))

        var weightOnTwo = shifted
        weightOnTwo.downWOff = 294
        #expect(!MoE.int8RoutedOffsetsAreAligned(weightOnTwo))
        var oddScale = shifted
        oddScale.upSOff = 227
        #expect(!MoE.int8RoutedOffsetsAreAligned(oddScale))
        var oddBias = shifted
        oddBias.gateBOff = 131
        #expect(!MoE.int8RoutedOffsetsAreAligned(oddBias))
    }

    /// §2.5 is derived from the shapes: rows × columns bytes of weights and
    /// rows × columns / 64 BF16 scales and biases, back to back.
    @Test func productionOffsetsFollowFromTheShapes() {
        let d = Fixtures.Production8Bit.d
        let f = Fixtures.Production8Bit.f
        let weights = f * d
        let affine = f * d / Quantization.groupSize * 2
        var cursor = 0
        var derived = [Int]()
        for size in [weights, affine, affine, weights, affine, affine, weights, affine, affine] {
            derived.append(cursor)
            cursor += size
        }
        let o = Fixtures.Production8Bit.offsets
        #expect(derived == [o.gateWOff, o.gateSOff, o.gateBOff, o.upWOff, o.upSOff,
                            o.upBOff, o.downWOff, o.downSOff, o.downBOff].map(Int.init))
        #expect(cursor == Fixtures.Production8Bit.payloadBytes)
        let page = 16_384
        #expect(Fixtures.Production8Bit.expertStride == (cursor + page - 1) / page * page)
    }

    @Test func int8ModuleDeclaresItsKernelsAndNoForeignConstants() throws {
        let source = try String(contentsOf: Self.int8ModuleURL, encoding: .utf8)
        func captures(_ pattern: String) throws -> [String] {
            let regex = try NSRegularExpression(pattern: pattern)
            let range = NSRange(source.startIndex..., in: source)
            return regex.matches(in: source, range: range).compactMap { match in
                Range(match.range(at: 1), in: source).map { String(source[$0]) }
            }
        }
        let kernels = try captures(#"kernel\s+void\s+(\w+)\s*\("#)
        #expect(Set(kernels) == [
            "moe_phase1_gate_up_act_int8",
            "moe_phase1_gate_up_act_subset_int8",
            "moe_phase2_down_reduce_k8_int8",
        ])
        let context = try MetalContext()
        for name in kernels {
            #expect(context.library.functionNames.contains(name), "\(name) missing")
        }
        let constants = try captures(#"function_constant\((\d+)\)"#).compactMap { Int($0) }
        #expect(constants.allSatisfy { (110...119).contains($0) }, "indices=\(constants)")
    }
}
