import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// The server learns the model's variant from its manifest before the load:
/// the 8-bit pack needs more memory than the 4-bit figures, and it answers to
/// its own API model ID so a client configured for the 4-bit name is not
/// served another variant without notice.
@Suite("Server model variant")
struct ServerModelVariantTests {
    static let fourBit = ServerModelVariant(
        profile: .affine4, manifestModelID: "mlx-community/gemma-4-26b-a4b-it-4bit")
    static let eightBit = ServerModelVariant(
        profile: .affine8, manifestModelID: "mlx-community/gemma-4-26b-a4b-it-8bit")

    private static func parse(_ input: [String],
                              variant: ServerModelVariant?,
                              hostMemoryBytes: UInt64 = 32 << 30,
                              environment: [String: String] = [:]) throws -> ServerArguments {
        try ServerArguments.parse(["--model", "model.gturbo"] + input,
                                  hostMemoryBytes: hostMemoryBytes,
                                  environment: environment,
                                  readModelVariant: { _ in variant })
    }

    @Test func labelsNameTheVariants() {
        #expect(Self.fourBit.label == "4-bit")
        #expect(Self.eightBit.label == "8-bit")
        #expect(Self.fourBit.footprint == .affine4Measured)
        #expect(Self.eightBit.footprint == .affine8Provisional)
    }

    @Test func fourBitKeepsTheLegacyAPIModelID() throws {
        let arguments = try Self.parse([], variant: Self.fourBit)
        #expect(arguments.modelID == "gemma-4-26b-a4b-it")
        #expect(arguments.modelVariant == Self.fourBit)
    }

    @Test func eightBitAnswersToItsManifestModelID() throws {
        let arguments = try Self.parse([], variant: Self.eightBit)
        #expect(arguments.modelID == "mlx-community/gemma-4-26b-a4b-it-8bit")
        #expect(arguments.modelVariant == Self.eightBit)
    }

    @Test(arguments: [fourBit, eightBit])
    func explicitModelIDWinsForEitherVariant(variant: ServerModelVariant) throws {
        let arguments = try Self.parse(["--model-id", "my-model"], variant: variant)
        #expect(arguments.modelID == "my-model")
    }

    /// The order of the flags must not matter: the default is decided after
    /// the whole command line is read.
    @Test func explicitModelIDBeforeTheModelStillWins() throws {
        let arguments = try ServerArguments.parse(
            ["--model-id", "my-model", "--model", "model.gturbo"],
            hostMemoryBytes: 32 << 30, environment: [:],
            readModelVariant: { _ in Self.eightBit })
        #expect(arguments.modelID == "my-model")
    }

    /// A directory without a manifest cannot load; the load says so, exactly
    /// as before variants existed, and parsing behaves as it always did.
    @Test func directoryWithoutManifestParsesAsBefore() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-variant-missing-\(UUID().uuidString)")
        let arguments = try ServerArguments.parse(["--model", directory.path],
                                                  hostMemoryBytes: 32 << 30,
                                                  environment: [:])
        #expect(arguments.modelVariant == nil)
        #expect(arguments.modelID == "gemma-4-26b-a4b-it")
    }

    @Test func unreadableManifestIsRefusedBeforeTheLoad() {
        struct Broken: Error {}
        #expect {
            _ = try ServerArguments.parse(["--model", "model.gturbo"],
                                          hostMemoryBytes: 32 << 30,
                                          environment: [:],
                                          readModelVariant: { _ in throw Broken() })
        } throws: { error in
            guard case ServerArgumentError.invalid(let message) = error else { return false }
            return message.hasPrefix("--model manifest is not usable")
        }
    }

    /// Per the user's decision U1, an 8 GB Mac is refused the 8-bit model even
    /// at the smallest context, while the same request with the 4-bit model
    /// is admitted: the variant, not the flags, decides.
    @Test func eightBitIsRefusedOnEightGBWhereFourBitIsAdmitted() throws {
        let input = ["--max-context", "4096"]
        #expect(try Self.parse(input, variant: Self.fourBit,
                               hostMemoryBytes: 8 << 30).maxContext == 4_096)
        #expect {
            _ = try Self.parse(input, variant: Self.eightBit, hostMemoryBytes: 8 << 30)
        } throws: { error in
            guard case ServerArgumentError.invalid(let message) = error else { return false }
            let need = ContextAdmission.needDescription(
                config: .gemma4_26B_A4B, maxContext: 4_096, hostMemoryBytes: 8 << 30,
                expertCacheSlots: 16, footprint: .affine8Provisional)
            return message == "8-bit model: " + need
                + " Set \(ServerArguments.unbackedContextOverrideVariable)=1 to start anyway."
        }
        #expect(try Self.parse(input, variant: Self.eightBit,
                               hostMemoryBytes: 16 << 30).maxContext == 4_096)
        let overridden = try Self.parse(
            input, variant: Self.eightBit, hostMemoryBytes: 8 << 30,
            environment: [ServerArguments.unbackedContextOverrideVariable: "1"])
        #expect(overridden.modelVariant == Self.eightBit)
    }

    /// Extra slots cost the variant's own stride: a request the 4-bit model
    /// fits can need a larger Mac with the 8-bit one.
    @Test func eightBitSlotsAreChargedAtTheirOwnStride() throws {
        let config = ArchConfig.gemma4_26B_A4B
        let fourBitNeed = ContextAdmission.minimumHostMemoryBytes(
            config: config, maxContext: 32_768, expertCacheSlots: 32,
            footprint: .affine4Measured)
        let eightBitNeed = ContextAdmission.minimumHostMemoryBytes(
            config: config, maxContext: 32_768, expertCacheSlots: 32,
            footprint: .affine8Provisional)
        #expect(eightBitNeed > fourBitNeed)
        let input = ["--max-context", "32768", "--expert-cache-slots", "32"]
        #expect(try Self.parse(input, variant: Self.fourBit,
                               hostMemoryBytes: fourBitNeed).expertCacheSlots == 32)
        #expect(throws: ServerArgumentError.self) {
            _ = try Self.parse(input, variant: Self.eightBit, hostMemoryBytes: fourBitNeed)
        }
        #expect(try Self.parse(input, variant: Self.eightBit,
                               hostMemoryBytes: eightBitNeed).expertCacheSlots == 32)
    }

    @Test func readsTheVariantFromAnInstalledManifest() throws {
        let eightBit = try Self.writeManifest(
            modelID: "mlx-community/gemma-4-26b-a4b-it-8bit", bits: (8, 8, 8, 8, 8))
        defer { try? FileManager.default.removeItem(at: eightBit) }
        #expect(try ServerModelVariant.read(modelDirectory: eightBit) == Self.eightBit)

        let fourBit = try Self.writeManifest(
            modelID: "mlx-community/gemma-4-26b-a4b-it-4bit", bits: (4, 4, 8, 4, 4))
        defer { try? FileManager.default.removeItem(at: fourBit) }
        #expect(try ServerModelVariant.read(modelDirectory: fourBit) == Self.fourBit)

        // The default reader, end to end: the manifest alone decides the
        // admission and the API model ID.
        let admitted = try ServerArguments.parse(
            ["--model", eightBit.path, "--max-context", "4096"],
            hostMemoryBytes: 16 << 30, environment: [:])
        #expect(admitted.modelID == "mlx-community/gemma-4-26b-a4b-it-8bit")
        #expect(admitted.modelVariant == Self.eightBit)
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(
                ["--model", eightBit.path, "--max-context", "4096"],
                hostMemoryBytes: 8 << 30, environment: [:])
        }
        let legacy = try ServerArguments.parse(
            ["--model", fourBit.path, "--max-context", "4096"],
            hostMemoryBytes: 8 << 30, environment: [:])
        #expect(legacy.modelID == "gemma-4-26b-a4b-it")
    }

    /// A manifest mixing the two profiles' bit widths has no kernel set; the
    /// server refuses it before admission rather than sizing it as either.
    @Test func mixedProfileManifestIsRefused() throws {
        let mixed = try Self.writeManifest(
            modelID: "mixed", bits: (8, 8, 8, 8, 4))
        defer { try? FileManager.default.removeItem(at: mixed) }
        #expect(throws: (any Error).self) {
            _ = try ServerModelVariant.read(modelDirectory: mixed)
        }
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerArguments.parse(["--model", mixed.path],
                                          hostMemoryBytes: 32 << 30, environment: [:])
        }
    }

    /// Scripts that read the ready line keep every field they read before;
    /// the variant fields are appended.
    @Test func readyLineAppendsTheVariant() throws {
        let arguments = try Self.parse(["--port", "8181"], variant: Self.eightBit)
        let line = arguments.readyLine(servedVariant: Self.eightBit, visionCapability: "missing")
        let before = "TurboFieldfareServer ready at http://127.0.0.1:8181 "
            + "model=mlx-community/gemma-4-26b-a4b-it-8bit context=16384 "
            + "prompt_cache=single-prefix vision=missing vision_residency=on-demand"
        #expect(line == before
            + " variant=8-bit manifest_model_id=mlx-community/gemma-4-26b-a4b-it-8bit")
    }

    @Test func modelOpenedLogNamesTheVariant() {
        #expect(ServerLog.modelOpenedMessage(Self.fourBit)
            == "model opened variant=4-bit manifest_model_id=mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(ServerLog.modelOpenedMessage(Self.eightBit)
            == "model opened variant=8-bit manifest_model_id=mlx-community/gemma-4-26b-a4b-it-8bit")
    }

    @Test func usageExplainsTheVariantDefault() {
        #expect(ServerArguments.usage.contains("default gemma-4-26b-a4b-it"))
        #expect(ServerArguments.usage.contains("an 8-bit model uses its"))
        #expect(ServerArguments.usage.contains("the budget follows the model's variant"))
    }

    /// A manifest `ManifestReader` accepts for the production architecture,
    /// with only the fields this suite varies. No weights are written: the
    /// variant is read from the manifest alone.
    static func writeManifest(modelID: String,
                              bits: (Int, Int, Int, Int, Int)) throws -> URL {
        let config = ArchConfig.gemma4_26B_A4B
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-variant-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)
        func slot(_ bits: Int) -> [String: Any] {
            ["weightBits": bits, "scheme": "affine", "scaleType": "bf16",
             "biasType": "bf16", "groupSize": Quantization.groupSize]
        }
        let zeros = String(repeating: "0", count: 64)
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": zeros],
            "packed_experts/layout.json": ["size": 1024, "sha256": zeros],
        ]
        for layer in 0..<config.numLayers {
            files["packed_experts/layer_\(layer).bin"] = ["size": 16384, "sha256": zeros]
        }
        let root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": modelID,
            "arch": [
                "hiddenSize": config.hiddenSize,
                "ffnIntermediate": config.intermediateSize,
                "moeIntermediateSize": config.moeIntermediateSize,
                "numHeads": config.numHeads,
                "numKVHeads": config.numKVHeads,
                "numFullKVHeads": config.numFullKVHeads,
                "headDim": config.headDim,
                "fullHeadDim": config.fullHeadDim,
                "vocabSize": config.vocabSize,
                "slidingWindow": config.slidingWindow,
                "finalLogitSoftcap": config.finalLogitSoftcap,
                "ropeTheta": config.ropeTheta,
                "fullRopeTheta": config.fullRopeTheta,
                "partialRotaryFactor": config.partialRotaryFactor,
                "numLayers": config.numLayers,
                "numExperts": config.numExperts,
                "topKExperts": config.topKExperts,
                "tieWordEmbeddings": config.tieWordEmbeddings,
                "attentionKEqV": config.attentionKEqV,
                "hiddenActivation": config.hiddenActivation,
                "fullAttentionLayerMask": config.fullAttentionLayerMask.map { Int($0) },
            ] as [String: Any],
            "quant": [
                "embedding": slot(bits.0),
                "attention": slot(bits.1),
                "router": slot(bits.2),
                "sharedExpert": slot(bits.3),
                "routedExpert": slot(bits.4),
            ],
            "files": files,
            "expertsPerLayer": config.numExperts,
            "numLayers": config.numLayers,
            "expertStride": 16384,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: root, options: [.sortedKeys, .withoutEscapingSlashes])
        try data.write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }
}
