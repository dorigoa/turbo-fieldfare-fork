import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The runner picks its weight kernels by quantization profile.
@Suite struct RealForwardRunnerKernelSelectionTests {
    @Test func affine4KeepsTheFourBitKernels() throws {
        let context = try MetalContext()
        let kernels = try AffineKernelSet.make(profile: .affine4,
                                               context: context,
                                               config: .gemma4_26B_A4B)
        #expect(kernels.profile == .affine4)
        #expect(kernels.embed is EmbedLookupInt4)
        #expect(kernels.gemv is DequantInt4GEMV)
        #expect(kernels.fusedQKV is FusedQKVGEMV)
        #expect(kernels.greedyHead is LMHeadChainInt4)
        #expect(kernels.moe.routedWeightBits == 4)
        #expect(kernels.prefillEmbed is PrefillEmbedLookupInt4)
        #expect(kernels.prefillQMM is PrefillInt4QMM)
        #expect(kernels.prefillMPP is MPPPrefillInt4QMM)
        #expect(kernels.prefillGroupedMoE.weightBits == 4)
        #expect(kernels.prefillFinalRowHead is PrefillFinalRowHeadInt4)
    }

    @Test func affine8SelectsTheEightBitTwins() throws {
        let context = try MetalContext()
        let kernels = try AffineKernelSet.make(profile: .affine8,
                                               context: context,
                                               config: .gemma4_26B_A4B)
        #expect(kernels.profile == .affine8)
        #expect(kernels.embed is EmbedLookupInt8)
        #expect(kernels.gemv is DequantInt8GEMV)
        #expect(kernels.fusedQKV is FusedQKVGEMVInt8)
        #expect(kernels.greedyHead is LMHeadChainInt8)
        #expect(kernels.moe.routedWeightBits == 8)
        #expect(kernels.prefillEmbed is PrefillEmbedLookupInt8)
        #expect(kernels.prefillQMM is PrefillInt8QMM)
        #expect(kernels.prefillMPP is MPPPrefillInt8QMM)
        #expect(kernels.prefillGroupedMoE.weightBits == 8)
        #expect(kernels.prefillFinalRowHead is PrefillFinalRowHeadInt8)
    }
}

/// End-to-end `affine8` runs on `AffineToyModel`, whose 8-bit install
/// dequantizes to exactly the weights of its 4-bit install. The 4-bit runner
/// is the reference: on the same path (decode or chunked prefill) the two
/// profiles differ only by kernel summation order.
@Suite(.serialized) struct RealForwardRunnerAffineProfileTests {
    static let maxContext = 256
    static let promptLength = 80
    static let continuationLength = 4

    /// Seeded prompt plus a few continuation tokens decoded after it.
    static let tokens: [Int32] = {
        var rng = SeedTree(AffineToyModel.defaultSeed).key("prompt")
        return (0..<(promptLength + continuationLength)).map { _ in
            Int32(rng.next() % UInt64(AffineToyModel.config.vocabSize))
        }
    }()

    struct Installs {
        let affine4: URL
        let affine8: URL
    }

    static func writeInstalls() throws -> Installs {
        Installs(affine4: try AffineToyModel.write(profile: .affine4),
                 affine8: try AffineToyModel.write(profile: .affine8))
    }

    static func makeRunner(directory: URL,
                           context: MetalContext,
                           slots: Int,
                           prefillEnabled: Bool,
                           fusedHead: Bool) throws -> RealForwardRunner {
        let model = try Model.load(directoryURL: directory,
                                   device: context.device,
                                   expecting: AffineToyModel.config,
                                   streamingMode: .pread(slotCount: slots))
        // The runner holds the model, which keeps the resident mapping alive.
        return try RealForwardRunner(
            model: model,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: slots,
                                                       prefillEnabled: prefillEnabled,
                                                       prefillChunkTokens: 64,
                                                       forceLogitsHead: !fusedHead))
    }

    static func makeLogitsBuffer(_ context: MetalContext) throws -> MTLBuffer {
        try #require(context.device.makeBuffer(
            length: AffineToyModel.config.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
    }

    static func read(_ logits: MTLBuffer) -> [Float] {
        let pointer = logits.contents().assumingMemoryBound(to: Float16.self)
        return (0..<AffineToyModel.config.vocabSize).map { Float(pointer[$0]) }
    }

    static func argmax(_ values: [Float]) -> Int {
        values.indices.max { values[$0] < values[$1] || (values[$0] == values[$1] && $0 > $1) }!
    }

    /// Gap between the largest and the second-largest logit.
    static func topMargin(_ values: [Float]) -> Float {
        let sorted = values.sorted(by: >)
        return sorted[0] - sorted[1]
    }

    /// Logits after every token of `tokens`, decoded one position at a time.
    static func decodeLogits(directory: URL, context: MetalContext,
                             tokens: [Int32], slots: Int) async throws -> [[Float]] {
        let runner = try makeRunner(directory: directory, context: context,
                                    slots: slots, prefillEnabled: false,
                                    fusedHead: false)
        let logits = try makeLogitsBuffer(context)
        var rows: [[Float]] = []
        for (position, token) in tokens.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
            rows.append(read(logits))
        }
        return rows
    }

    /// Logits after a chunked prefill of the prompt, then after each
    /// continuation token decoded on top of the prefilled KV.
    static func prefillThenDecodeLogits(directory: URL, context: MetalContext,
                                        slots: Int) async throws -> [[Float]] {
        let runner = try makeRunner(directory: directory, context: context,
                                    slots: slots, prefillEnabled: true,
                                    fusedHead: false)
        let logits = try makeLogitsBuffer(context)
        var rows: [[Float]] = []
        let result = try await runner.prefillChunked(
            tokens: tokens[0..<promptLength],
            startPosition: 0,
            outputMode: .logits,
            config: .production(chunkTokens: 64),
            into: logits,
            onProgress: { _ in })
        #expect(result.newPosition == promptLength)
        rows.append(read(logits))
        for position in promptLength..<tokens.count {
            try await runner.produce(token: tokens[position], position: position, into: logits)
            rows.append(read(logits))
        }
        return rows
    }

    static func compare(_ actual: [[Float]], _ reference: [[Float]],
                        label: String) {
        #expect(actual.count == reference.count)
        var worst: Float = 0
        var worstAbs: Float = 0
        var largest: Float = 0
        var identicalRows = 0
        for (index, (a, r)) in zip(actual, reference).enumerated() {
            worstAbs = max(worstAbs, zip(a, r).map { abs($0 - $1) }.max() ?? 0)
            largest = max(largest, r.map(abs).max() ?? 0)
            if a == r { identicalRows += 1 }
            let allFinite = a.allSatisfy { $0.isFinite }
            #expect(allFinite, "\(label) row \(index) has non-finite logits")
            let error = RelError.compute(actual: a, reference: r)
            worst = max(worst, error)
            #expect(error <= Tolerance.fp16ChainedReduction,
                    "\(label) row \(index): rel \(error)")
            // Equal argmax is required only where the reference has a clear
            // winner; a near tie may legitimately flip with summation order.
            if topMargin(r) > 0.05 {
                #expect(argmax(a) == argmax(r),
                        "\(label) row \(index): argmax \(argmax(a)) != \(argmax(r))")
            }
        }
        print("[affine-toy] \(label): max rel \(worst), max abs \(worstAbs), "
                + "max |ref| \(largest), \(identicalRows)/\(actual.count) rows bit-identical")
    }

    /// With 8 slots half the experts are evicted and refetched, so the routed
    /// hit/miss split runs; with 16 every expert stays cached after its
    /// first fetch.
    @Test(arguments: [8, 16])
    func decodeMatchesTheFourBitReference(slots: Int) async throws {
        let installs = try Self.writeInstalls()
        let context = try MetalContext()
        let reference = try await Self.decodeLogits(directory: installs.affine4, context: context,
                                                    tokens: Self.tokens, slots: slots)
        let actual = try await Self.decodeLogits(directory: installs.affine8, context: context,
                                                 tokens: Self.tokens, slots: slots)
        Self.compare(actual, reference, label: "decode affine8 vs affine4, \(slots) slots")
    }

    /// Chunked prefill needs at least 16 slots (two tiles of 8 experts in
    /// flight), the same floor `RuntimeConfiguration` enforces.
    @Test(arguments: [16, 24])
    func chunkedPrefillMatchesTheFourBitReference(slots: Int) async throws {
        let installs = try Self.writeInstalls()
        let context = try MetalContext()
        let reference = try await Self.prefillThenDecodeLogits(directory: installs.affine4,
                                                               context: context, slots: slots)
        let actual = try await Self.prefillThenDecodeLogits(directory: installs.affine8,
                                                            context: context, slots: slots)
        Self.compare(actual, reference, label: "prefill affine8 vs affine4, \(slots) slots")
    }

    /// Chunked prefill and token-by-token decode reach the same last-row
    /// logits through disjoint kernels. The 8-bit gap must stay within the
    /// shared tolerance, like the 4-bit one.
    @Test func chunkedPrefillAgreesWithDecodeForEachProfile() async throws {
        let installs = try Self.writeInstalls()
        let context = try MetalContext()
        for (profile, directory) in [(QuantProfile.affine4, installs.affine4),
                                     (.affine8, installs.affine8)] {
            let decoded = try await Self.decodeLogits(directory: directory, context: context,
                                                      tokens: Self.tokens, slots: 16)
            let prefilled = try await Self.prefillThenDecodeLogits(directory: directory,
                                                                   context: context, slots: 16)
            Self.compare(prefilled, Array(decoded[(Self.promptLength - 1)...]),
                         label: "\(profile.rawValue) prefill vs decode")
        }
    }

    /// The fused greedy heads (decode and prefill) pick the argmax of the
    /// logits head.
    @Test func fusedGreedyHeadPicksTheLogitsArgmax() async throws {
        let installs = try Self.writeInstalls()
        let context = try MetalContext()
        let decoded = try await Self.decodeLogits(directory: installs.affine8, context: context,
                                                  tokens: Self.tokens, slots: 16)
        let prefilled = try await Self.prefillThenDecodeLogits(directory: installs.affine8,
                                                               context: context, slots: 16)

        let decodeRunner = try Self.makeRunner(
            directory: installs.affine8, context: context, slots: 16,
            prefillEnabled: false, fusedHead: true)
        let logits = try Self.makeLogitsBuffer(context)
        for (position, token) in Self.tokens.enumerated() {
            try await decodeRunner.produce(token: token, position: position, into: logits)
            let expected = decoded[position]
            if Self.topMargin(expected) > 0.05 {
                #expect(Int(decodeRunner.lastGreedyToken) == Self.argmax(expected),
                        "decode position \(position)")
            }
        }

        let prefillRunner = try Self.makeRunner(
            directory: installs.affine8, context: context, slots: 16,
            prefillEnabled: true, fusedHead: true)
        let result = try await prefillRunner.prefillChunked(
            tokens: Self.tokens[0..<Self.promptLength],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 64),
            into: logits,
            onProgress: { _ in })
        if Self.topMargin(prefilled[0]) > 0.05 {
            #expect(result.seed == .greedyToken(UInt32(Self.argmax(prefilled[0]))))
        }
    }
}
