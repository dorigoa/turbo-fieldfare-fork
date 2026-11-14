import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareCLICore

/// S22: the CLI routes by the family the installed manifest declares (D3),
/// refuses image input for Qwen before any model work (D12), and fills the
/// sampling knobs the user left untouched with the checkpoint's own
/// `generation_config.json` values (D10, U4). The routing tests run against
/// a manifest-only directory: the refusal and the variant read never touch
/// weights or a tokenizer, so no installed model is needed.
@Suite struct CLIQwenRoutingTests {

    private static let zeroSHA = String(repeating: "0", count: 64)

    /// A manifest-only directory carrying the pinned Qwen values, valid for
    /// `ManifestReader.load` and therefore for `CLIModelVariant.read`.
    private static func writeQwenManifestDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-qwen-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let arch: [String: Any] = [
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
        ]
        let qwen35: [String: Any] = [
            "linearNumKeyHeads": 16, "linearNumValueHeads": 32,
            "linearKeyHeadDim": 128, "linearValueHeadDim": 128,
            "linearConvKernel": 4, "rmsNormEps": 1e-6,
            "normWeightsShifted": true, "attentionOutputGate": true,
            "sharedExpertGate": true,
        ]
        func slot(_ bits: Int) -> [String: Any] {
            ["weightBits": bits, "scheme": "affine", "scaleType": "BF16",
             "biasType": "BF16", "groupSize": 64]
        }
        let root: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "modelFamily": "qwen3_5_moe",
            "qwen35": qwen35,
            "arch": arch,
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
            withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    private static func capture(_ url: URL) throws -> String {
        String(data: try Data(contentsOf: url), encoding: .utf8) ?? ""
    }

    /// Routing is the manifest's, never a flag: the family read from the
    /// installed pack names itself in the variant footer.
    @Test func qwenManifestNamesTheFamilyInVariantAndFooter() throws {
        let directory = try Self.writeQwenManifestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let variant = try CLIModelVariant.read(modelDirectory: directory)
        #expect(variant.family == .qwen35Moe)
        #expect(variant.label == "4-bit")
        #expect(variant.manifestModelID == "mlx-community/Qwen3.6-35B-A3B-4bit")
        #expect(variant.footerLine
            == "[variant=4-bit family=qwen3_5_moe "
            + "manifest_model_id=mlx-community/Qwen3.6-35B-A3B-4bit]")
    }

    /// D12: an image on a Qwen pack is refused up front, fail-closed, before
    /// any tokenizer or GPU work. The manifest-only directory proves the
    /// refusal needs no model: it happens on the first line of the Qwen
    /// branch, and the same message the server will use.
    @Test func qwenImagePromptIsRefusedBeforeAnyModelWork() async throws {
        let directory = try Self.writeQwenManifestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-qwen-img-out-\(UUID().uuidString)")
        let errURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-qwen-img-err-\(UUID().uuidString)")
        try Data().write(to: outURL)
        try Data().write(to: errURL)
        let stdout = try FileHandle(forWritingTo: outURL)
        let stderr = try FileHandle(forWritingTo: errURL)
        defer { try? stdout.close(); try? stderr.close() }
        let args = Args(model: directory.path,
                        chatPrompt: "what is in this image?",
                        images: ["/nonexistent/missing.png"],
                        maxNew: 8)
        let result = await run(args: args, stdout: stdout, stderr: stderr)
        #expect(result.exitCode == 2)
        #expect(try Self.capture(outURL).isEmpty)
        #expect(try Self.capture(errURL)
            == "error: image input is not available for this model\n")
    }

    /// `--messages-file` reaches the image path with `args.images` empty, so
    /// the refusal must ask the parsed input, not the flag list: an
    /// `image_file` part is refused with the same message and code.
    @Test func qwenMessagesFileImageIsRefusedToo() async throws {
        let directory = try Self.writeQwenManifestDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let messagesURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-qwen-messages-\(UUID().uuidString).json")
        let messages: [[String: Any]] = [[
            "role": "user",
            "content": [
                ["type": "text", "text": "what is in this image?"],
                ["type": "image_file", "path": "/nonexistent/missing.png"],
            ],
        ]]
        let data = try JSONSerialization.data(withJSONObject: messages)
        try data.write(to: messagesURL)
        defer { try? FileManager.default.removeItem(at: messagesURL) }
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-qwen-msg-out-\(UUID().uuidString)")
        let errURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cli-qwen-msg-err-\(UUID().uuidString)")
        try Data().write(to: outURL)
        try Data().write(to: errURL)
        let stdout = try FileHandle(forWritingTo: outURL)
        let stderr = try FileHandle(forWritingTo: errURL)
        defer { try? stdout.close(); try? stderr.close() }
        let args = Args(model: directory.path,
                        messagesFile: messagesURL.path,
                        maxNew: 8)
        let result = await run(args: args, stdout: stdout, stderr: stderr)
        #expect(result.exitCode == 2)
        #expect(try Self.capture(outURL).isEmpty)
        #expect(try Self.capture(errURL)
            == "error: image input is not available for this model\n")
    }

    @Test func noThinkingDefaultsOffAndTheFlagTurnsItOn() throws {
        let plain = try Args.parse(["--model", "m.gturbo", "--chat-prompt", "hi"])
        #expect(!plain.noThinking)
        let off = try Args.parse([
            "--model", "m.gturbo", "--chat-prompt", "hi", "--no-thinking",
        ])
        #expect(off.noThinking)
    }

    /// Reasoning depth (D11): the depth parses to a level, and its absence is the unbounded run
    /// this CLI has always done.
    @Test func reasoningDepthParsesToALevelAndDefaultsToUnbounded() throws {
        let plain = try Args.parse(["--model", "m.gturbo", "--chat-prompt", "hi"])
        #expect(plain.reasoningDepth == nil)
        for depth in ReasoningDepth.allCases {
            let args = try Args.parse([
                "--model", "m.gturbo", "--chat-prompt", "hi",
                "--reasoning-depth", depth.rawValue,
            ])
            #expect(args.reasoningDepth == depth)
        }
    }

    @Test func reasoningDepthRefusesAValueThatIsNotALevel() {
        #expect(throws: ArgsError.invalidValue(flag: "--reasoning-depth", value: "deep")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--chat-prompt", "hi",
                "--reasoning-depth", "deep",
            ])
        }
    }

    /// A depth for a turn that does no reasoning bounds nothing: refused,
    /// rather than silently dropping one of the two flags.
    @Test func reasoningDepthAndNoThinkingAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--no-thinking", "--reasoning-depth")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--chat-prompt", "hi",
                "--no-thinking", "--reasoning-depth", "low",
            ])
        }
    }

    /// The knob-level tracking behind D10: a flag the user passed is recorded
    /// as set, and only a flag they passed; the parse-level defaults stay
    /// Gemma's either way.
    @Test func samplingFlagsRecordWhoSetThem() throws {
        let untouched = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(untouched.temperature == 0.2 && !untouched.temperatureSet)
        #expect(untouched.topK == 64 && !untouched.topKSet)
        #expect(untouched.topP == 0.95 && !untouched.topPSet)

        let passed = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--temperature", "0.3", "--top-k", "8", "--top-p", "0.5",
        ])
        #expect(passed.temperature == 0.3 && passed.temperatureSet)
        #expect(passed.topK == 8 && passed.topKSet)
        #expect(passed.topP == 0.5 && passed.topPSet)
    }

    /// U4: on Qwen, every knob the user left untouched takes the checkpoint's
    /// own value; the user's knobs survive untouched.
    @Test func qwenSamplingDefaultsFollowTheCheckpoint() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        let config = try qwenGenerationConfig(args: arguments, maxNewTokens: 32)
        #expect(config.maxNewTokens == 32)
        #expect(config.temperature == 1.0)
        #expect(config.topK == 20)
        #expect(config.topP == 0.95)

        let mine = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--temperature", "0", "--top-p", "1",
        ])
        let kept = try qwenGenerationConfig(args: mine, maxNewTokens: 8)
        #expect(kept.temperature == 0)
        #expect(kept.topK == 20)   // untouched: still the checkpoint's
        #expect(kept.topP == 1.0)  // set by the user: theirs
    }

    /// The family-aware companion of the parse-level `--top-p` check: a
    /// disabled top-k cannot pair with an effective top-p under one (D10).
    @Test func qwenSamplingRefusesTopPWithoutTopK() {
        let arguments = Args(model: "m.gturbo", prompt: "hi",
                             topK: nil, topP: 0.95,
                             topKSet: true)
        #expect(throws: ArgsError.invalidValue(
            flag: "--top-p",
            value: "0.95 requires --top-k between 1 and 256")) {
            _ = try qwenGenerationConfig(args: arguments, maxNewTokens: 8)
        }
    }
}
