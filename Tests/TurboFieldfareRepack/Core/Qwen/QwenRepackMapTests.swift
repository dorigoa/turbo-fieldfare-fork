import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// S6: a tiny synthetic Qwen3.5-MoE snapshot (2 layers — one linear-attention,
/// one full-attention — 4 experts, hidden 64) exercises the full planner +
/// writer map: classification, quant slots, C5 expert packing, vision
/// exclusion, the manifest, byte-faithful resident writes for BF16 and FP32
/// tensors, and the fail-closed rejection of off-allowlist names.
@Suite struct QwenRepackMapTests {

    // Synthetic geometry (3.3 shapes, miniature sizes).
    private static let hidden = 64
    private static let groupSize = 64
    private static let numExperts = 4
    private static let numLayers = 2
    private static let vocab = 512

    private struct Tensor {
        let name: String
        let dtype: String      // safetensors header dtype
        let shape: [Int]
        let bytes: [UInt8]
    }

    private struct Snapshot {
        let dir: String
        let shardPath: String
    }

    private func temporaryDirectory(_ tag: String) -> String {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("S6-QwenMap-\(tag)-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url.path
    }

    // MARK: - Snapshot builder

    /// Build the Qwen synthetic snapshot: config.json (base 4-bit affine +
    /// 8-bit `mlp.gate`/`mlp.shared_expert_gate` overrides, like the real
    /// checkpoint), one safetensors shard and the index. `extraTensorNames`
    /// appends fail-closed probes (no payloads needed beyond the tensor).
    private func buildSnapshot(at dir: String,
                               extraTensors: [Tensor] = []) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)

        var tensors: [Tensor] = []
        var rng = SplitMix64(seed: 0x5EED_06C5_0000_0004)

        func payload(_ count: Int) -> [UInt8] {
            var bytes = [UInt8](repeating: 0, count: count)
            for i in 0..<count { bytes[i] = UInt8(rng.next() & 0xFF) }
            return bytes
        }
        func quantized(_ base: String, outerShape: [Int], innerLogical: Int, bits: Int) {
            let factor = 32 / bits
            precondition(innerLogical % Self.groupSize == 0 && innerLogical % factor == 0)
            let shape = outerShape + [innerLogical / factor]
            tensors.append(Tensor(name: base + ".weight", dtype: "U32",
                                  shape: shape, bytes: payload(shape.reduce(1, *) * 4)))
            let companion = outerShape + [innerLogical / Self.groupSize]
            tensors.append(Tensor(name: base + ".scales", dtype: "BF16",
                                  shape: companion, bytes: payload(companion.reduce(1, *) * 2)))
            tensors.append(Tensor(name: base + ".biases", dtype: "BF16",
                                  shape: companion, bytes: payload(companion.reduce(1, *) * 2)))
        }
        func plain(_ name: String, dtype: String, shape: [Int]) {
            let elementBytes = dtype == "F32" ? 4 : 2
            tensors.append(Tensor(name: name, dtype: dtype,
                                  shape: shape, bytes: payload(shape.reduce(1, *) * elementBytes)))
        }

        // Top level (3.3): embedding and lm_head quantized, final norm plain.
        quantized("language_model.model.embed_tokens", outerShape: [Self.vocab],
                  innerLogical: Self.hidden, bits: 4)
        quantized("language_model.lm_head", outerShape: [Self.vocab],
                  innerLogical: Self.hidden, bits: 4)
        plain("language_model.model.norm.weight", dtype: "BF16", shape: [Self.hidden])

        // Layer 0: linear attention (DeltaNet) block.
        let l0 = "language_model.model.layers.0"
        plain(l0 + ".input_layernorm.weight", dtype: "BF16", shape: [Self.hidden])
        plain(l0 + ".post_attention_layernorm.weight", dtype: "BF16", shape: [Self.hidden])
        plain(l0 + ".linear_attn.A_log", dtype: "BF16", shape: [32])
        plain(l0 + ".linear_attn.dt_bias", dtype: "F32", shape: [32])
        plain(l0 + ".linear_attn.conv1d.weight", dtype: "BF16", shape: [128])
        plain(l0 + ".linear_attn.norm.weight", dtype: "BF16", shape: [128])
        quantized(l0 + ".linear_attn.in_proj_qkv", outerShape: [192], innerLogical: Self.hidden, bits: 4)
        quantized(l0 + ".linear_attn.in_proj_z", outerShape: [128], innerLogical: Self.hidden, bits: 4)
        quantized(l0 + ".linear_attn.in_proj_a", outerShape: [32], innerLogical: Self.hidden, bits: 4)
        quantized(l0 + ".linear_attn.in_proj_b", outerShape: [32], innerLogical: Self.hidden, bits: 4)
        quantized(l0 + ".linear_attn.out_proj", outerShape: [Self.hidden], innerLogical: 128, bits: 4)

        // Layer 1: full attention block.
        let l1 = "language_model.model.layers.1"
        plain(l1 + ".input_layernorm.weight", dtype: "BF16", shape: [Self.hidden])
        plain(l1 + ".post_attention_layernorm.weight", dtype: "BF16", shape: [Self.hidden])
        plain(l1 + ".self_attn.q_norm.weight", dtype: "BF16", shape: [32])
        plain(l1 + ".self_attn.k_norm.weight", dtype: "BF16", shape: [32])
        quantized(l1 + ".self_attn.q_proj", outerShape: [Self.hidden], innerLogical: Self.hidden, bits: 4)
        quantized(l1 + ".self_attn.k_proj", outerShape: [Self.hidden], innerLogical: Self.hidden, bits: 4)
        quantized(l1 + ".self_attn.v_proj", outerShape: [Self.hidden], innerLogical: Self.hidden, bits: 4)
        quantized(l1 + ".self_attn.o_proj", outerShape: [Self.hidden], innerLogical: Self.hidden, bits: 4)

        // Both layers: router + shared expert gate (8-bit overrides, like the
        // real config), shared expert (base bits), routed experts (rank-3).
        for layer in [l0, l1] {
            quantized(layer + ".mlp.gate", outerShape: [Self.numExperts],
                      innerLogical: Self.hidden, bits: 8)
            quantized(layer + ".mlp.shared_expert_gate", outerShape: [1],
                      innerLogical: Self.hidden, bits: 8)
            quantized(layer + ".mlp.shared_expert.gate_proj", outerShape: [Self.hidden],
                      innerLogical: Self.hidden, bits: 4)
            quantized(layer + ".mlp.shared_expert.up_proj", outerShape: [Self.hidden],
                      innerLogical: Self.hidden, bits: 4)
            quantized(layer + ".mlp.shared_expert.down_proj", outerShape: [Self.hidden],
                      innerLogical: Self.hidden, bits: 4)
            quantized(layer + ".mlp.switch_mlp.gate_proj",
                      outerShape: [Self.numExperts, Self.hidden], innerLogical: Self.hidden, bits: 4)
            quantized(layer + ".mlp.switch_mlp.up_proj",
                      outerShape: [Self.numExperts, Self.hidden], innerLogical: Self.hidden, bits: 4)
            quantized(layer + ".mlp.switch_mlp.down_proj",
                      outerShape: [Self.numExperts, Self.hidden], innerLogical: Self.hidden, bits: 4)
        }

        // Multimodal tensors: a plain one and a quantized trio, all excluded.
        plain("vision_tower.patch_embed.norm.weight", dtype: "BF16", shape: [Self.hidden])
        quantized("vision_tower.blocks.0.attn.q_proj", outerShape: [Self.hidden],
                  innerLogical: Self.hidden, bits: 4)

        tensors.append(contentsOf: extraTensors)

        // -- safetensors shard, MLX-style: 8-byte header length, space-padded
        // sorted-key JSON header, payloads back to back.
        var off: UInt64 = 0
        var headerDict: [String: Any] = [:]
        for t in tensors {
            let begin = off, end = begin + UInt64(t.bytes.count)
            headerDict[t.name] = ["dtype": t.dtype, "shape": t.shape,
                                  "data_offsets": [begin, end]]
            off = end
        }
        headerDict["__metadata__"] = ["format": "mlx"]
        var headerData = try JSONSerialization.data(withJSONObject: headerDict,
                                                    options: [.sortedKeys])
        while headerData.count % 8 != 0 { headerData.append(0x20) }
        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        let fd = open(shardPath, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        precondition(fd >= 0, "open failed for \(shardPath)")
        defer { close(fd) }
        var headerLenLE = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { _ = write(fd, $0.baseAddress, 8) }
        headerData.withUnsafeBytes { _ = write(fd, $0.baseAddress, headerData.count) }
        for t in tensors {
            t.bytes.withUnsafeBufferPointer { _ = write(fd, $0.baseAddress, $0.count) }
        }

        // -- config.json: Qwen model_type, base 4-bit affine + 8-bit router
        // overrides (the real config's shape).
        var quant: [String: Any] = ["bits": 4, "group_size": Self.groupSize, "mode": "affine"]
        for li in 0..<Self.numLayers {
            for base in ["mlp.gate", "mlp.shared_expert_gate"] {
                quant["language_model.model.layers.\(li).\(base)"] =
                    ["bits": 8, "group_size": Self.groupSize]
            }
        }
        let config: [String: Any] = ["model_type": "qwen3_5_moe", "quantization": quant]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))

        // -- model.safetensors.index.json
        var weightMap: [String: String] = [:]
        for t in tensors { weightMap[t.name] = shardName }
        let indexObj: [String: Any] = ["metadata": ["format": "mlx"], "weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: indexObj, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString)
                .appendingPathComponent("model.safetensors.index.json")))

        return Snapshot(dir: dir, shardPath: shardPath)
    }

    /// The pinned reader refuses synthetic dimensions (S5 fail-closed), so the
    /// arch is constructed directly with the family + C4 block set.
    private func syntheticArch() -> ArchInfo {
        ArchInfo(
            hiddenSize: Self.hidden, intermediateSize: Self.hidden,
            moeIntermediateSize: Self.hidden,
            numHeads: 2, numKVHeads: 2, numFullKVHeads: 2,
            headDim: 32, fullHeadDim: 32,
            vocabSize: Self.vocab, slidingWindow: 0, finalLogitSoftcap: 0,
            ropeTheta: 10_000_000, fullRopeTheta: 10_000_000, partialRotaryFactor: 0.25,
            numLayers: Self.numLayers, numExperts: Self.numExperts, topKExperts: 2,
            tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: [0, 1], hiddenActivation: "silu",
            modelFamily: GTurboModelFamily.qwen35Moe.rawValue,
            qwen35: GTurboManifestQwen35V1(
                linearNumKeyHeads: 16, linearNumValueHeads: 32,
                linearKeyHeadDim: 128, linearValueHeadDim: 128,
                linearConvKernel: 4, rmsNormEps: 1e-6,
                normWeightsShifted: true,
                attentionOutputGate: true, sharedExpertGate: true))
    }

    private struct Planned {
        let snapshot: Snapshot
        let meta: IndexLoader.SourceMetadata
        let plan: RepackPlan
        let snapshotDir: String
        let outputDir: String
    }

    private func planFixture(tag: String,
                             extraTensors: [Tensor] = []) throws -> Planned {
        let snapshotDir = temporaryDirectory("\(tag)-snapshot")
        let outputDir = temporaryDirectory("\(tag)-output")
        let snapshot = try buildSnapshot(at: snapshotDir, extraTensors: extraTensors)
        let meta = try IndexLoader.load(snapshotDir: snapshotDir)
        // Parsed with the ABSOLUTE shard path: ResidentWriter mmaps tensors
        // through `sourceTensor.shardPath`.
        let shard = try Data(contentsOf: URL(fileURLWithPath: snapshot.shardPath))
        let headerLen = shard.prefix(8).withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let header = try Safetensors.parseHeaderBytes(
            path: snapshot.shardPath,
            fileSize: UInt64(shard.count),
            headerBytes: shard.subdata(in: 8..<(8 + Int(headerLen))))
        let plan = try RepackPlanner.plan(meta: meta, arch: syntheticArch(),
                                          shardHeaders: [header], outputDir: outputDir)
        return Planned(snapshot: snapshot, meta: meta, plan: plan,
                       snapshotDir: snapshotDir, outputDir: outputDir)
    }

    // MARK: - Plan: resident map + vision exclusion

    @Test func planMapsEveryQwenTensorAndDropsVision() throws {
        let fixture = try planFixture(tag: "plan")
        defer {
            try? FileManager.default.removeItem(atPath: fixture.snapshotDir)
            try? FileManager.default.removeItem(atPath: fixture.outputDir)
        }
        let plan = fixture.plan

        // 3.3 top level + both layer groups: exactly the 43 base weights.
        var expected: Set<String> = [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "language_model.model.norm.weight",
        ]
        for li in 0..<Self.numLayers {
            let l = "language_model.model.layers.\(li)"
            expected.insert(l + ".input_layernorm.weight")
            expected.insert(l + ".post_attention_layernorm.weight")
            if li == 0 {
                // Linear-attention layer: DeltaNet block.
                for p in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] {
                    expected.insert(l + ".linear_attn.\(p).weight")
                }
                expected.formUnion([
                    l + ".linear_attn.A_log", l + ".linear_attn.dt_bias",
                    l + ".linear_attn.conv1d.weight", l + ".linear_attn.norm.weight",
                ])
            } else {
                // Full-attention layer: gated GQA block.
                for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    expected.insert(l + ".self_attn.\(p).weight")
                }
                expected.formUnion([
                    l + ".self_attn.q_norm.weight", l + ".self_attn.k_norm.weight",
                ])
            }
            expected.formUnion([
                l + ".mlp.gate.weight", l + ".mlp.shared_expert_gate.weight",
                l + ".mlp.shared_expert.gate_proj.weight",
                l + ".mlp.shared_expert.up_proj.weight",
                l + ".mlp.shared_expert.down_proj.weight",
            ])
        }
        let names = Set(plan.resident.entries.map(\.name))
        #expect(names == expected)
        // The planner template puts the embedding first and the final norm last.
        #expect(plan.resident.entries.first?.name == "language_model.model.embed_tokens.weight")
        #expect(plan.resident.entries.last?.name == "language_model.model.norm.weight")

        // Vision is excluded, quantized tensors included.
        #expect(plan.excludedMultimodalTensorNames.sorted() == [
            "vision_tower.blocks.0.attn.q_proj.biases",
            "vision_tower.blocks.0.attn.q_proj.scales",
            "vision_tower.blocks.0.attn.q_proj.weight",
            "vision_tower.patch_embed.norm.weight",
        ].sorted())
        #expect(names.isDisjoint(with: plan.excludedMultimodalTensorNames))

        // The 4 router overrides (2 layers × gate + shared_expert_gate).
        #expect(plan.bitsOverrideCount == 4)
        #expect(plan.modelFamily == GTurboModelFamily.qwen35Moe.rawValue)
    }

    // MARK: - Plan: C5 expert packing

    @Test func expertBlobsFollowC5OrderAndStride() throws {
        let fixture = try planFixture(tag: "experts")
        defer {
            try? FileManager.default.removeItem(atPath: fixture.snapshotDir)
            try? FileManager.default.removeItem(atPath: fixture.outputDir)
        }
        let layers = fixture.plan.layers
        #expect(layers.count == Self.numLayers)
        for layer in layers {
            #expect(layer.layerIndex < Self.numLayers)
            #expect(layer.path.hasSuffix("packed_experts/layer_0\(layer.layerIndex).bin"))
            #expect(layer.expertsPerLayer == Self.numExperts)
            #expect(layer.subTensors.count == 9)
            var cursor: UInt64 = 0
            var roles: [String] = []
            var components: [String] = []
            for slice in layer.subTensors {
                roles.append(slice.role)
                components.append(slice.component)
                #expect(slice.offsetInExpertBlob == cursor, "\(slice.role).\(slice.component)")
                cursor += slice.sizeInExpertBlob
                if slice.component == "weights" {
                    #expect(slice.bitsForWeights == 4)
                    #expect(slice.dtype == GTurboFormatV1.DType.u32.rawValue)
                    // rank-3 source [experts, rows, packed] → packed = 64/8
                    #expect(slice.sourceTensor.shape ==
                            [UInt64(Self.numExperts), UInt64(Self.hidden), 8])
                    #expect(slice.sizeInExpertBlob == UInt64(Self.hidden * 8 * 4))
                } else {
                    #expect(slice.bitsForWeights == nil)
                    #expect(slice.dtype == GTurboFormatV1.DType.bf16.rawValue)
                    #expect(slice.sizeInExpertBlob == UInt64(Self.hidden) * 2)
                }
            }
            #expect(roles == ["gate", "gate", "gate", "up", "up", "up", "down", "down", "down"])
            #expect(components == Array(repeating: ["weights", "scales", "biases"], count: 3).flatMap { $0 })
            // 3 roles × (2048 weight + 128 scales + 128 biases) = 6912 →
            // stride is the 16 KiB page above it.
            #expect(cursor == 6912)
            #expect(layer.expertStride == 16_384)
            #expect(layer.expertStride % GTurboFormatV1.alignmentBytes == 0)
        }
    }

    // MARK: - Quant slots (manifest bit widths)

    @Test func manifestQuantSlotsMatchTheQwenLayout() throws {
        let fixture = try planFixture(tag: "slots")
        defer {
            try? FileManager.default.removeItem(atPath: fixture.snapshotDir)
            try? FileManager.default.removeItem(atPath: fixture.outputDir)
        }
        let bits = try fixture.plan.manifestQuantBitWidths()
        #expect(bits.embedding == 4)
        #expect(bits.attention == 4)      // linear_attn + self_attn agree
        #expect(bits.router == 8)         // mlp.gate + mlp.shared_expert_gate
        #expect(bits.sharedExpert == 4)
        #expect(bits.routedExpert == 4)
    }

    // MARK: - Manifest

    @Test func manifestCarriesFamilyStrideAndQuantSlots() throws {
        let fixture = try planFixture(tag: "manifest")
        defer {
            try? FileManager.default.removeItem(atPath: fixture.snapshotDir)
            try? FileManager.default.removeItem(atPath: fixture.outputDir)
        }
        let plan = fixture.plan
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "fixture/qwen36-synthetic",
            sourceSnapshotHash: fixture.meta.indexSha256Hex,
            files: [],
            expertsPerLayer: plan.arch.numExperts,
            numLayers: plan.arch.numLayers,
            expertStride: plan.layers[0].expertStride,
            bitWidths: try plan.manifestQuantBitWidths())
        let manifest = try GTurboManifestCodec.decode(data)
        #expect(manifest.modelFamily == GTurboModelFamily.qwen35Moe.rawValue)
        #expect(manifest.qwen35 == plan.qwen35)
        #expect(manifest.expertsPerLayer == Self.numExperts)
        #expect(manifest.expertStride == 16_384)
        let quant = try #require(manifest.quant)
        #expect(quant.embedding.weightBits == 4)
        #expect(quant.attention.weightBits == 4)
        #expect(quant.router.weightBits == 8)
        #expect(quant.sharedExpert.weightBits == 4)
        #expect(quant.routedExpert.weightBits == 4)
        #expect(quant.router.scheme == "affine")
        #expect(quant.router.groupSize == Self.groupSize)
    }

    // MARK: - Writing: BF16 and FP32 tensors intact

    @Test func residentWriterCopiesEveryTensorByteForByte() throws {
        let fixture = try planFixture(tag: "write")
        defer {
            try? FileManager.default.removeItem(atPath: fixture.snapshotDir)
            try? FileManager.default.removeItem(atPath: fixture.outputDir)
        }
        let audit = RepackAudit()
        var shards: [String: MmapHandle] = [:]
        let outFile = try ResidentWriter.write(plan: fixture.plan.resident,
                                               shardsByPath: &shards,
                                               audit: audit)
        #expect(outFile.size == fixture.plan.resident.totalSize)
        #expect(outFile.sha256.count == 64)

        let shard = try Data(contentsOf: URL(fileURLWithPath: fixture.snapshot.shardPath))
        let written = try Data(contentsOf: URL(fileURLWithPath: fixture.plan.resident.path))
        func sourceBytes(_ tensor: SourceTensor) -> Data {
            shard.subdata(in: Int(tensor.absoluteOffset)
                            ..< Int(tensor.absoluteOffset + tensor.sizeBytes))
        }
        var bf16Checks = 0, fp32Checks = 0
        for entry in fixture.plan.resident.entries {
            let weight = entry.sourceWeight
            #expect(written.subdata(in: Int(entry.fileOffset)
                                        ..< Int(entry.fileOffset + entry.sizeBytes))
                    == sourceBytes(weight), "\(entry.name) weight")
            if let scales = entry.sourceScales {
                #expect(written.subdata(in: Int(entry.scaleOffset)
                                            ..< Int(entry.scaleOffset + entry.scaleSize))
                        == sourceBytes(scales), "\(entry.name) scales")
            }
            if let biases = entry.sourceBiases {
                #expect(written.subdata(in: Int(entry.biasOffset)
                                            ..< Int(entry.biasOffset + entry.biasSize))
                        == sourceBytes(biases), "\(entry.name) biases")
            }
            switch weight.dtype {
            case .bf16: bf16Checks += 1
            case .fp32: fp32Checks += 1
            default: break
            }
        }
        // BF16 and FP32 payloads survive the copy untouched.
        #expect(bf16Checks > 0)
        #expect(fp32Checks == 1)   // the linear layer's F32 dt_bias
        let dtBias = try #require(
            fixture.plan.resident.entries
                .first { $0.name == "language_model.model.layers.0.linear_attn.dt_bias" })
        #expect(dtBias.sourceWeight.dtype == .fp32)
        #expect(dtBias.quantSpec == nil)
        // And BF16 unquantized norms carry no quantization spec either.
        let norm = try #require(
            fixture.plan.resident.entries
                .first { $0.name == "language_model.model.layers.0.input_layernorm.weight" })
        #expect(norm.sourceWeight.dtype == .bf16)
        #expect(norm.quantSpec == nil)
    }

    // MARK: - Fail-closed: off-allowlist names abort the plan

    @Test func namesOutsideTheAllowlistFailThePlan() throws {
        // An unknown expert role, a layer index past the end and an
        // unwhitelisted top-level tensor all classify as `.unknown`.
        let probes: [Tensor] = [
            Tensor(name: "language_model.model.layers.0.mlp.switch_mlp.leaky_proj.weight",
                   dtype: "U32", shape: [4, 64, 8], bytes: [UInt8](repeating: 0, count: 4 * 64 * 8 * 4)),
            Tensor(name: "language_model.model.layers.5.input_layernorm.weight",
                   dtype: "BF16", shape: [64], bytes: [UInt8](repeating: 0, count: 64 * 2)),
            Tensor(name: "language_model.model.rotary_emb.inv_freq",
                   dtype: "BF16", shape: [32], bytes: [UInt8](repeating: 0, count: 32 * 2)),
        ]
        var threw = false
        do {
            let fixture = try planFixture(tag: "unknown", extraTensors: probes)
            // If it ever planned, clean up and fail.
            try? FileManager.default.removeItem(atPath: fixture.snapshotDir)
            try? FileManager.default.removeItem(atPath: fixture.outputDir)
        } catch let error as RepackError {
            threw = true
            if case .unknownTensorPrefix = error {} else {
                Issue.record("wrong RepackError case: \(error)")
            }
        }
        #expect(threw)
    }
}
