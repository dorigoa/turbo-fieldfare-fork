import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareDecodeService

/// The service is the process the system kills. A 262,144-token KV is charged
/// in full the moment a command buffer binds it — one full-attention layer over
/// sixteen rows already costs 1,085 MiB — so a load this host cannot back does
/// not fail slowly and report itself: it disappears mid-load, and the app can
/// only show a lost connection. Every case here is a load that must be answered
/// before the runner is built.
@Suite struct DecodeLoadAdmissionTests {

    @Test(arguments: [24, 32])
    func cacheGrowthRejects128KOnEightGB(slots: Int) throws {
        let refusal = try #require(DecodeLoadAdmission.refusal(maxContextTokens: 131_072,
            hostMemoryBytes: 8 << 30, environment: [:], expertCacheSlots: slots))
        #expect(refusal.contains("16 GB"))
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 131_072,
            hostMemoryBytes: 16 << 30, environment: [:], expertCacheSlots: slots) == nil)
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 131_072,
            hostMemoryBytes: 8 << 30,
            environment: [DecodeLoadAdmission.overrideEnvironmentKey: "1"],
            expertCacheSlots: slots) == nil)
    }

    private let eightGigabyteHost: UInt64 = 8_589_934_592
    private let twentyFourGigabyteHost: UInt64 = 25_769_803_776

    @Test func acontextThisHostCanBackIsAdmitted() {
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 8_192,
                                            hostMemoryBytes: eightGigabyteHost,
                                            environment: [:]) == nil)
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 262_144,
                                            hostMemoryBytes: twentyFourGigabyteHost,
                                            environment: [:]) == nil)
    }

    @Test func acontextThisHostCannotBackIsRefusedWithTheNeed() throws {
        let refusal = try #require(
            DecodeLoadAdmission.refusal(maxContextTokens: 262_144,
                                        hostMemoryBytes: eightGigabyteHost,
                                        environment: [:]))
        #expect(refusal.contains("262,144"))
        #expect(refusal.contains("16 GB"))
        #expect(refusal.contains("8 GB"))
    }

    /// The measurement the admission rule came from is only reachable by
    /// running a context the rule refuses, so the override exists — explicitly,
    /// and for nothing else.
    @Test func theoverrideAdmitsAnUnbackedContext() {
        #expect(DecodeLoadAdmission.refusal(
            maxContextTokens: 262_144,
            hostMemoryBytes: eightGigabyteHost,
            environment: [DecodeLoadAdmission.overrideEnvironmentKey: "1"]) == nil)
    }

    @Test(arguments: ["0", "", "true", "yes"])
    func theoverrideIsExactlyOneAndNothingElse(_ value: String) {
        #expect(DecodeLoadAdmission.refusal(
            maxContextTokens: 262_144,
            hostMemoryBytes: eightGigabyteHost,
            environment: [DecodeLoadAdmission.overrideEnvironmentKey: value]) != nil,
            "\(value) was treated as the override")
    }

    /// The model ceiling is not a memory question, so the diagnostic override
    /// must not reach it: a position beyond `max_position_embeddings` is
    /// invalid on a 512 GB machine too.
    @Test func themodelCeilingIsRefusedEvenWithTheOverrideSet() throws {
        let ceiling = ArchConfig.gemma4_26B_A4B.maxPositionEmbeddings
        let refusal = try #require(DecodeLoadAdmission.refusal(
            maxContextTokens: ceiling + 1,
            hostMemoryBytes: twentyFourGigabyteHost,
            environment: [DecodeLoadAdmission.overrideEnvironmentKey: "1"]))
        #expect(refusal.contains("262145"))
        #expect(refusal.contains("262144"))
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: ceiling,
                                            hostMemoryBytes: twentyFourGigabyteHost,
                                            environment: [:]) == nil)
    }

    // MARK: - Footprint from the model being loaded

    private let sixteenGigabyteHost: UInt64 = 17_179_869_184

    /// The service sizes the load from the manifest of the model it is about
    /// to open, not from the 4-bit constants: an 8-bit pack on an 8 GB Mac is
    /// refused at every context, and admitted on a 16 GB one.
    @Test(arguments: [4_096, 8_192])
    func aneightBitManifestIsRefusedOnEightGBAndAdmittedOnSixteen(context: Int) throws {
        let directory = try writeManifest(bits: 8)
        defer { try? FileManager.default.removeItem(at: directory) }
        let refusal = try #require(DecodeLoadAdmission.refusal(
            modelDirectory: directory, maxContextTokens: context,
            hostMemoryBytes: eightGigabyteHost, environment: [:]))
        #expect(refusal.contains("16 GB"))
        #expect(refusal.contains("8 GB"))
        #expect(DecodeLoadAdmission.refusal(
            modelDirectory: directory, maxContextTokens: context,
            hostMemoryBytes: sixteenGigabyteHost, environment: [:]) == nil)
        // The same request sized with the 4-bit figures would have gone ahead.
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: context,
                                            hostMemoryBytes: eightGigabyteHost,
                                            environment: [:]) == nil)
    }

    /// A 4-bit manifest gets exactly the answers the service gave before.
    @Test(arguments: [8_192, 131_072, 262_144])
    func afourBitManifestDecidesAsBefore(context: Int) throws {
        let directory = try writeManifest(bits: 4)
        defer { try? FileManager.default.removeItem(at: directory) }
        for host in [eightGigabyteHost, sixteenGigabyteHost, twentyFourGigabyteHost] {
            for slots in [16, 32] {
                #expect(DecodeLoadAdmission.refusal(
                    modelDirectory: directory, maxContextTokens: context,
                    hostMemoryBytes: host, environment: [:], expertCacheSlots: slots)
                    == DecodeLoadAdmission.refusal(maxContextTokens: context,
                                                   hostMemoryBytes: host, environment: [:],
                                                   expertCacheSlots: slots))
            }
        }
    }

    /// Guessing the 4-bit footprint for a model whose manifest cannot be read
    /// is guessing low, which is the kill this rule exists to prevent.
    @Test func amanifestThatCannotBeReadIsARefusal() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("admission-missing-\(UUID()).gturbo")
        let refusal = try #require(DecodeLoadAdmission.refusal(
            modelDirectory: missing, maxContextTokens: 8_192,
            hostMemoryBytes: twentyFourGigabyteHost, environment: [:]))
        #expect(refusal.contains("manifest"))
    }

    /// The override and the ceiling are decided before the manifest is read:
    /// the first skips it, the second is not a memory question.
    @Test func theoverrideAndTheCeilingDoNotNeedTheManifest() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("admission-missing-\(UUID()).gturbo")
        #expect(DecodeLoadAdmission.refusal(
            modelDirectory: missing, maxContextTokens: 8_192,
            hostMemoryBytes: eightGigabyteHost,
            environment: [DecodeLoadAdmission.overrideEnvironmentKey: "1"]) == nil)
        let ceiling = ArchConfig.gemma4_26B_A4B.maxPositionEmbeddings
        let refusal = try #require(DecodeLoadAdmission.refusal(
            modelDirectory: missing, maxContextTokens: ceiling + 1,
            hostMemoryBytes: twentyFourGigabyteHost, environment: [:]))
        #expect(refusal.contains("262145"))
    }

    /// S24: a Qwen manifest is sized with its family's geometry, not Gemma's.
    /// S27 measured that geometry, which moved where the two profiles part:
    /// 128K now fits an 8 GB Mac on either, so the ceiling is the discriminator
    /// — a host that exactly meets Qwen's requirement at 262,144 tokens is
    /// still short of the 4-bit Gemma one. The boundary is inclusive.
    @Test func aQwenManifestIsNotMeasuredWithGemmaGeometry() throws {
        let directory = try writeQwenManifest()
        defer { try? FileManager.default.removeItem(at: directory) }
        let minimum: UInt64 = 262_144 * 20_480 + 1_879_048_192 + 67_108_864 + 3_221_225_472
        // The same host, sized with the 4-bit Gemma constants, refuses.
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 262_144,
                                            hostMemoryBytes: minimum,
                                            environment: [:]) != nil)
        #expect(DecodeLoadAdmission.refusal(modelDirectory: directory, maxContextTokens: 262_144,
                                            hostMemoryBytes: minimum, environment: [:]) == nil)
        #expect(DecodeLoadAdmission.refusal(modelDirectory: directory, maxContextTokens: 262_144,
                                            hostMemoryBytes: minimum - 1, environment: [:]) != nil)
        let refusal = try #require(DecodeLoadAdmission.refusal(
            modelDirectory: directory, maxContextTokens: 262_144,
            hostMemoryBytes: eightGigabyteHost, environment: [:]))
        #expect(refusal.contains("262,144"))
        #expect(refusal.contains("needs 16 GB"))
        // 128K fits an 8 GB Mac on the measured figures (S27); it did not on
        // the provisional ones.
        #expect(DecodeLoadAdmission.refusal(modelDirectory: directory, maxContextTokens: 131_072,
                                            hostMemoryBytes: eightGigabyteHost, environment: [:]) == nil)
        // Small contexts sail through on both profiles.
        #expect(DecodeLoadAdmission.refusal(modelDirectory: directory, maxContextTokens: 4_096,
                                            hostMemoryBytes: eightGigabyteHost, environment: [:]) == nil)
    }

    /// The default footprint parameter is the 4-bit pack's, so callers that
    /// predate variants keep their answers.
    @Test func theexplicitFootprintDefaultsToTheFourBitPack() {
        for context in [8_192, 262_144] {
            #expect(DecodeLoadAdmission.refusal(maxContextTokens: context,
                                                hostMemoryBytes: eightGigabyteHost, environment: [:],
                                                footprint: .affine4Measured)
                == DecodeLoadAdmission.refusal(maxContextTokens: context,
                                               hostMemoryBytes: eightGigabyteHost, environment: [:]))
        }
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 4_096,
                                            hostMemoryBytes: eightGigabyteHost, environment: [:],
                                            footprint: .affine8Provisional) != nil)
    }

    /// Just the manifest: the footprint is read from it alone, so no weights,
    /// layout or receipt are needed to exercise the rule.
    private func writeManifest(bits: Int) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("admission-\(bits)bit-\(UUID().uuidString).gturbo")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let arch = ArchConfig.gemma4_26B_A4B
        func slot(_ weightBits: Int) -> [String: Any] {
            ["weightBits": weightBits, "scheme": "affine", "scaleType": "bf16",
             "biasType": "bf16", "groupSize": Quantization.groupSize]
        }
        var files: [String: Any] = [
            "model_weights.bin": ["size": 0, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 2, "sha256": String(repeating: "0", count: 64)],
        ]
        for layer in 0..<arch.numLayers {
            files[String(format: "packed_experts/layer_%02d.bin", layer)] = [
                "size": 0, "sha256": String(repeating: "0", count: 64),
            ]
        }
        let manifest: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true],
            "modelID": "test/gemma-4-26b-a4b",
            "sourceSnapshotHash": "sha256:" + String(repeating: "0", count: 64),
            "quant": [
                "embedding": slot(bits),
                "attention": slot(bits),
                "router": slot(8),
                "sharedExpert": slot(bits),
                "routedExpert": slot(bits),
            ],
            "arch": [
                "hiddenSize": arch.hiddenSize,
                "ffnIntermediate": arch.intermediateSize,
                "moeIntermediateSize": arch.moeIntermediateSize,
                "numHeads": arch.numHeads,
                "numKVHeads": arch.numKVHeads,
                "numFullKVHeads": arch.numFullKVHeads,
                "headDim": arch.headDim,
                "fullHeadDim": arch.fullHeadDim,
                "vocabSize": arch.vocabSize,
                "slidingWindow": arch.slidingWindow,
                "finalLogitSoftcap": arch.finalLogitSoftcap,
                "ropeTheta": arch.ropeTheta,
                "fullRopeTheta": arch.fullRopeTheta,
                "partialRotaryFactor": arch.partialRotaryFactor,
                "numLayers": arch.numLayers,
                "numExperts": arch.numExperts,
                "topKExperts": arch.topKExperts,
                "tieWordEmbeddings": arch.tieWordEmbeddings,
                "attentionKEqV": arch.attentionKEqV,
                "hiddenActivation": arch.hiddenActivation,
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
            ],
            "files": files,
            "expertsPerLayer": arch.numExperts,
            "numLayers": arch.numLayers,
            "expertStride": UInt64(getpagesize()),
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    /// Just the manifest, with the pinned Qwen constants (D1, C4).
    private func writeQwenManifest() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("admission-qwen-\(UUID().uuidString).gturbo")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        func slot(_ weightBits: Int) -> [String: Any] {
            ["weightBits": weightBits, "scheme": "affine", "scaleType": "BF16",
             "biasType": "BF16", "groupSize": Quantization.groupSize]
        }
        let zeroSHA = String(repeating: "0", count: 64)
        let manifest: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true],
            "modelID": "test/qwen36-35b-a3b",
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
                "model_weights.bin": ["size": 0, "sha256": zeroSHA],
                "packed_experts/layout.json": ["size": 2, "sha256": zeroSHA],
            ],
            "expertsPerLayer": 256,
            "numLayers": 40,
            "expertStride": UInt64(getpagesize()),
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }
}
