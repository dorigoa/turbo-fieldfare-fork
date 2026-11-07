import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The 8-bit prefill wrappers fit the plan's kernel contracts (C3) and the
/// `prefill_int8` module keeps to its naming and function-constant rules (C4).
@Suite struct PrefillInt8KernelContractTests {
    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func captures(_ pattern: String, in source: String) throws -> [String] {
        let regex = try NSRegularExpression(pattern: pattern)
        let range = NSRange(source.startIndex..., in: source)
        return regex.matches(in: source, range: range).compactMap { match in
            Range(match.range(at: 1), in: source).map { String(source[$0]) }
        }
    }

    /// The selection S8 makes per profile: one conformer per width behind the
    /// same protocol.
    private static func prefillQMM(for profile: QuantProfile,
                                   context: MetalContext) throws -> any AffinePrefillQMMEncoding {
        switch profile {
        case .affine4: try PrefillInt4QMM(context: context)
        case .affine8: try PrefillInt8QMM(context: context)
        }
    }

    private static func prefillEmbed(for profile: QuantProfile,
                                     context: MetalContext) throws -> any AffinePrefillEmbedEncoding {
        switch profile {
        case .affine4: try PrefillEmbedLookupInt4(context: context)
        case .affine8: try PrefillEmbedLookupInt8(context: context)
        }
    }

    @Test func wrappersConformToTheAffineContracts() throws {
        let context = try MetalContext()
        #expect(try Self.prefillQMM(for: .affine8, context: context) is PrefillInt8QMM)
        #expect(try Self.prefillQMM(for: .affine4, context: context) is PrefillInt4QMM)
        #expect(try Self.prefillEmbed(for: .affine8, context: context) is PrefillEmbedLookupInt8)
        #expect(try Self.prefillEmbed(for: .affine4, context: context) is PrefillEmbedLookupInt4)
    }

    /// Encoding through the protocol, with every argument spelled out as S8
    /// will, gives the same bits as the concrete call with its defaults.
    @Test func encodingThroughTheContractMatchesTheConcreteCall() throws {
        typealias Buffers = PrefillInt8ProjectionBuffers
        let projection = PrefillInt8Projection.random(rows: 96, columns: 192,
                                                      tokens: 33, seed: 0x5420)
        let context = try MetalContext()
        let concrete = try PrefillInt8QMM(context: context)
        let contract = try Self.prefillQMM(for: .affine8, context: context)
        guard let buffers = Buffers(device: context.device, projection: projection),
              let direct = Fp16Buffer.make(context.device, count: 33 * 96),
              let viaContract = Fp16Buffer.make(context.device, count: 33 * 96),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        concrete.encode(commandBuffer: commandBuffer,
                        weights: buffers.weights, scales: buffers.scales,
                        biases: buffers.biases, x: buffers.x, y: direct,
                        t: 33, n: 96, k: 192)
        contract.encode(commandBuffer: commandBuffer,
                        weights: buffers.weights, weightsOffset: 0,
                        scales: buffers.scales, scalesOffset: 0,
                        biases: buffers.biases, biasesOffset: 0,
                        x: buffers.x, xOffset: 0,
                        y: viaContract, yOffset: 0,
                        t: 33, n: 96, k: 192)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(Fp16Buffer.readHalf(direct, count: 33 * 96)
                == Fp16Buffer.readHalf(viaContract, count: 33 * 96))
    }

    @Test func prefillInt8ModuleDeclaresItsKernelsAndNoForeignConstants() throws {
        let url = Self.repositoryRoot
            .appendingPathComponent("Sources/TurboFieldfare/Metal/Prefill/prefill_int8.metal")
        let source = try String(contentsOf: url, encoding: .utf8)
        let kernels = try Self.captures(#"kernel\s+void\s+(\w+)\s*\("#, in: source)
        #expect(Set(kernels) == [
            "prefill_embed_lookup_int8_block",
            "prefill_dequant_int8_qmm_f16_block",
        ])
        let context = try MetalContext()
        for name in kernels {
            #expect(context.library.functionNames.contains(name), "\(name) missing")
        }
        // The 4-bit twins are still in the shared library next to them.
        #expect(context.library.functionNames.contains("prefill_embed_lookup_int4_block"))
        #expect(context.library.functionNames.contains("prefill_dequant_int4_qmm_f16_block"))

        let constants = try Self.captures(#"function_constant\((\d+)\)"#, in: source)
            .compactMap { Int($0) }
        #expect(constants.allSatisfy { (120...129).contains($0) }, "indices=\(constants)")
        let fileScopeConstants = try Self.captures(
            #"(?m)^constant\s+constexpr\s+\w+\s+(\w+)"#, in: source)
        #expect(fileScopeConstants.allSatisfy { $0.lowercased().contains("int8") },
                "\(fileScopeConstants)")
    }
}
