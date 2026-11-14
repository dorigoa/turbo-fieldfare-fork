import Testing
import Foundation
@testable import TurboFieldfare

/// S24: admission for the Qwen family. The KV term counts the ten attention
/// layers at 20,480 B per token and nothing for the thirty DeltaNet layers,
/// whose recurrent state is charged once, constant; the non-KV figures and the
/// reserve are provisional until S27 measures them (3.4, D5). Gemma's own
/// answers are pinned by `ContextAdmissionTests`.
@Suite struct ContextAdmissionQwenTests {

    private let config = ArchConfig.qwen36_35B_A3B
    private let footprint = ModelFootprint.qwen36Measured
    private let eightGigabyteHost: UInt64 = 8_589_934_592
    private let sixteenGigabyteHost: UInt64 = 17_179_869_184

    /// The pinned figures the plan fixes for S24, so a later edit moves them
    /// through a test failure and not through a menu caption.
    @Test func theMeasuredFootprintCarriesThePinnedFigures() {
        #expect(footprint.nonKVRuntimeBytes == 1_879_048_192) // 1.75 GiB, measured in S27
        #expect(footprint.linearStateBytes == 67_108_864)     // 64 MiB, constant
        #expect(footprint.hostReserveBytes == 3 << 30)
        #expect(footprint.expertStrideBytes == 1_769_472)
        // The Gemma footprints carry no linear state, so their arithmetic
        // cannot have moved.
        #expect(ModelFootprint.affine4Measured.linearStateBytes == 0)
        #expect(ModelFootprint.affine8Measured.linearStateBytes == 0)
    }

    /// 10 × 2 × 2 × 256 × 2 B = 20,480 B per token (3.4), with no sliding
    /// ring: the linear layers hold DeltaNet state, not KV (D5).
    @Test func kvGrowsAtTwentyThousandFourHundredBytesPerToken() {
        for context in [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144] {
            #expect(ContextAdmission.fp16KVBytes(config: config, maxContext: context)
                    == UInt64(context) * 20_480, "context \(context)")
        }
    }

    /// The plan's four sizes with the provisional figures, 16 slots.
    @Test(arguments: [4_096, 32_768, 131_072, 262_144])
    func minimumHostMemoryAddsKVStateRuntimeAndReserve(context: Int) {
        let minimum = ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: context,
                                                              footprint: footprint)
        #expect(minimum == UInt64(context) * 20_480 + 1_879_048_192 + 67_108_864 + 3_221_225_472)
        // The boundary is inclusive, exactly as for Gemma.
        #expect(ContextAdmission.availability(config: config, maxContext: context,
                                              hostMemoryBytes: minimum - 1, footprint: footprint)
                == .needsMemory(minimumHostBytes: minimum))
        #expect(ContextAdmission.availability(config: config, maxContext: context,
                                              hostMemoryBytes: minimum, footprint: footprint)
                == .available)
    }

    /// An 8 GB Mac backs every context up to 128K on the figures S27 measured
    /// and is refused only the checkpoint's 262,144-token ceiling, which needs
    /// 10,536,091,648 bytes; a 16 GB one backs every size. Before S27 the
    /// provisional non-KV figure refused 128K here too.
    @Test func eightGigabytesBacksUpTo128KAndSixteenBacksEverySize() {
        for context in [4_096, 32_768, 131_072] {
            #expect(ContextAdmission.availability(config: config, maxContext: context,
                                                  hostMemoryBytes: eightGigabyteHost,
                                                  footprint: footprint) == .available,
                    "context \(context)")
        }
        for context in [262_144] {
            #expect(ContextAdmission.availability(config: config, maxContext: context,
                                                  hostMemoryBytes: eightGigabyteHost,
                                                  footprint: footprint) != .available,
                    "context \(context)")
        }
        for context in [4_096, 32_768, 131_072, 262_144] {
            #expect(ContextAdmission.availability(config: config, maxContext: context,
                                                  hostMemoryBytes: sixteenGigabyteHost,
                                                  footprint: footprint) == .available,
                    "context \(context)")
        }
        let message = ContextAdmission.needDescription(config: config, maxContext: 262_144,
                                                       hostMemoryBytes: eightGigabyteHost,
                                                       footprint: footprint)
        #expect(message.contains("needs 16 GB"))
    }

    /// The pinned pack streams each routed expert at 1,769,472 bytes across
    /// 40 layer files (S6), so a larger cache is charged at that stride.
    @Test func extraSlotsAreChargedAtTheQwenStride() {
        let baseline = ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 8_192,
                                                               footprint: footprint)
        let expected = baseline + UInt64((32 - 16) * 40 * 1_769_472)
        #expect(ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 8_192,
                                                        expertCacheSlots: 32, footprint: footprint)
                == expected)
    }

    /// The checkpoint ceiling the family's constant carries: the 262,144 the
    /// plan admits (U2), which the picker can offer.
    @Test func theCheckpointCeilingIsThePlannedOne() {
        #expect(config.maxPositionEmbeddings == 262_144)
        #expect(config.slidingWindow == 0)
        #expect(config.fullAttentionLayerMask.reduce(0, +) == 10)
    }

    // MARK: - From the installed manifest

    @Test func aQwenManifestAnswersWithTheFamilyProfile() throws {
        let directory = try writeQwenManifest()
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(try ModelFootprint.forModelDirectory(directory) == .qwen36Measured)
        let (readConfig, readFootprint) = try ContextAdmission.admissionProfile(forModelDirectory: directory)
        #expect(readConfig == ArchConfig.qwen36_35B_A3B)
        #expect(readFootprint == .qwen36Measured)
    }

    /// Just the manifest, with the pinned Qwen constants (D1, C4): no weights,
    /// layout or receipt are needed to size the model.
    private func writeQwenManifest() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-qwen-admission-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"), withIntermediateDirectories: true)
        func slot(_ bits: Int) -> [String: Any] {
            ["weightBits": bits, "scheme": "affine", "scaleType": "BF16",
             "biasType": "BF16", "groupSize": 64]
        }
        let zeroSHA = String(repeating: "0", count: 64)
        let manifest: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "modelFamily": "qwen3_5_moe",
            "qwen35": [
                "linearNumKeyHeads": 16, "linearNumValueHeads": 32,
                "linearKeyHeadDim": 128, "linearValueHeadDim": 128,
                "linearConvKernel": 4, "rmsNormEps": 1e-6,
                "normWeightsShifted": true, "attentionOutputGate": true,
                "sharedExpertGate": true,
            ],
            "arch": [
                "hiddenSize": 2048, "ffnIntermediate": 512, "moeIntermediateSize": 512,
                "numHeads": 16, "numKVHeads": 2, "numFullKVHeads": 2,
                "headDim": 256, "fullHeadDim": 256, "vocabSize": 248_320,
                "slidingWindow": 0, "finalLogitSoftcap": 0,
                "ropeTheta": 10_000_000, "fullRopeTheta": 10_000_000,
                "partialRotaryFactor": 0.25, "numLayers": 40,
                "numExperts": 256, "topKExperts": 8,
                "tieWordEmbeddings": false, "attentionKEqV": false,
                "hiddenActivation": "silu",
                "fullAttentionLayerMask": (0..<40).map { $0 % 4 == 3 ? 1 : 0 },
            ],
            "quant": [
                "embedding": slot(4), "attention": slot(4), "router": slot(8),
                "sharedExpert": slot(4), "routedExpert": slot(4),
            ],
            "files": [
                "model_weights.bin": ["size": 1024, "sha256": zeroSHA],
                "packed_experts/layout.json": ["size": 1024, "sha256": zeroSHA],
            ],
            "expertsPerLayer": 256,
            "numLayers": 40,
            "expertStride": 16_384,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }
}
