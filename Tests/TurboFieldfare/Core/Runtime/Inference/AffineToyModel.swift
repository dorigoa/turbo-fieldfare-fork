import Foundation
import Metal
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore
import TurboFieldfareValidationSupport

/// A small Gemma 4 `.gturbo` written as either quantization profile from the
/// same 4-bit codes.
///
/// Every affine weight is drawn from a seeded uniform distribution and
/// quantized once with `Quantization.quantizeInt4Affine`. The `affine4`
/// install stores those nibbles; the `affine8` install stores each nibble as
/// a full byte (`q8 = q4`) with the same BF16 scales and biases. The router
/// is 8-bit in both, from the same `quantizeInt8Affine` rows. The two installs
/// therefore dequantize to the same weights, and the `affine4` runner, whose
/// kernels the production model already exercises, is a reference for the
/// `affine8` runner: any difference comes from summation order in the
/// kernels, not from the weights.
///
/// The architecture keeps the production structure at toy sizes: one
/// sliding-window and one full-attention layer (K = V, no `v_proj`), partial
/// RoPE on the full layer, 16 routed experts with top-8 routing (the routed
/// kernels require top-k 8), and a hidden size of 256 so the INT8 kernels
/// cover both a full 4-group block and remainder groups.
enum AffineToyModel {
    static let config = ArchConfig(
        hiddenSize: 256,
        intermediateSize: 256,
        moeIntermediateSize: 128,
        numHeads: 4,
        numKVHeads: 2,
        numFullKVHeads: 1,
        headDim: 32,
        fullHeadDim: 64,
        vocabSize: 1024,
        slidingWindow: 64,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 2,
        numExperts: 16,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: [0, 1],
        hiddenActivation: "gelu_pytorch_tanh")

    static let defaultSeed: UInt64 = 0x5EED_8B17

    /// Alignment the writer gives each quantized resident weight when the
    /// caller does not override it: the 4-byte grid of the older toy writer
    /// for `affine4`, the installer's 16-byte rule (D11) for `affine8`.
    static func defaultResidentWeightAlignment(_ profile: QuantProfile) -> UInt64 {
        switch profile {
        case .affine4: 4
        case .affine8: Model.affine8ResidentWeightAlignment
        }
    }

    /// Writes the toy install into a new temporary directory.
    ///
    /// - Parameter shiftedResidentWeight: the resident tensor whose weights
    ///   are moved 4 bytes past the aligned position, to build an install
    ///   that breaks the `affine8` alignment rule on purpose.
    static func write(profile: QuantProfile,
                      seed: UInt64 = defaultSeed,
                      shiftedResidentWeight: String? = nil) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-affine-toy-\(profile.rawValue)-\(UUID().uuidString)")
        let experts = directory.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

        let weights = Weights(seed: seed)
        let bits = profile.denseWeightBits
        let resident = try writeResident(profile: profile,
                                         weights: weights,
                                         shiftedResidentWeight: shiftedResidentWeight,
                                         to: directory.appendingPathComponent("model_weights.bin"))

        let blob = expertBlobLayout(bits: bits)
        let stride = blob.stride
        let layerBytes = Int(stride) * config.numExperts
        var layerFiles: [(name: String, size: Int, sha: String)] = []
        for layer in 0..<config.numLayers {
            var payload = Data(count: layerBytes)
            for expert in 0..<config.numExperts {
                let bytes = expertBlobBytes(layer: layer, expert: expert, bits: bits,
                                            layout: blob, weights: weights)
                payload.replaceSubrange(
                    (expert * Int(stride))..<(expert * Int(stride) + bytes.count),
                    with: bytes)
            }
            let name = String(format: "layer_%02d.bin", layer)
            let url = experts.appendingPathComponent(name)
            try payload.write(to: url)
            layerFiles.append((name, layerBytes, try Sha256Verifier.hashFile(at: url)))
        }

        var layoutLayers: [[String: Any]] = []
        for layer in 0..<config.numLayers {
            let entries: [[String: Any]] = (0..<config.numExperts).map { expert in
                [
                    "expert": expert,
                    "offset": UInt64(expert) * stride,
                    "size": stride,
                    "tensors": blob.tensors,
                ]
            }
            layoutLayers.append([
                "layer": layer,
                "file": String(format: "layer_%02d.bin", layer),
                "experts": entries,
            ])
        }
        let layoutData = try JSONSerialization.data(withJSONObject: [
            "expertStride": stride,
            "numLayers": config.numLayers,
            "expertsPerLayer": config.numExperts,
            "layers": layoutLayers,
        ] as [String: Any], options: [.sortedKeys])
        let layoutURL = experts.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)

        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": resident.size, "sha256": resident.sha],
            "packed_experts/layout.json": [
                "size": layoutData.count,
                "sha256": try Sha256Verifier.hashFile(at: layoutURL),
            ],
        ]
        for file in layerFiles {
            files["packed_experts/\(file.name)"] = ["size": file.size, "sha256": file.sha]
        }
        func slot(_ weightBits: Int) -> [String: Any] {
            [
                "weightBits": weightBits,
                "scheme": "affine",
                "scaleType": "BF16",
                "biasType": "BF16",
                "groupSize": Quantization.groupSize,
            ]
        }
        let manifest: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "toy-\(profile.rawValue)",
            "arch": [
                "hiddenSize": config.hiddenSize, "ffnIntermediate": config.intermediateSize,
                "moeIntermediateSize": config.moeIntermediateSize,
                "numHeads": config.numHeads, "numKVHeads": config.numKVHeads,
                "numFullKVHeads": config.numFullKVHeads,
                "headDim": config.headDim, "fullHeadDim": config.fullHeadDim,
                "vocabSize": config.vocabSize, "slidingWindow": config.slidingWindow,
                "finalLogitSoftcap": config.finalLogitSoftcap,
                "ropeTheta": config.ropeTheta, "fullRopeTheta": config.fullRopeTheta,
                "partialRotaryFactor": config.partialRotaryFactor,
                "numLayers": config.numLayers, "numExperts": config.numExperts,
                "topKExperts": config.topKExperts,
                "tieWordEmbeddings": config.tieWordEmbeddings,
                "attentionKEqV": config.attentionKEqV,
                "hiddenActivation": config.hiddenActivation,
                "fullAttentionLayerMask": config.fullAttentionLayerMask.map { Int($0) },
            ] as [String: Any],
            "quant": [
                "embedding": slot(bits),
                "attention": slot(bits),
                "router": slot(8),
                "sharedExpert": slot(bits),
                "routedExpert": slot(bits),
            ],
            "files": files,
            "expertsPerLayer": config.numExperts,
            "numLayers": config.numLayers,
            "expertStride": stride,
        ]
        let manifestData = try JSONSerialization.data(
            withJSONObject: manifest, options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    // MARK: - Weights

    /// Seeded source weights. Each tensor draws from its own `SeedTree`
    /// stream, so the bytes do not depend on the order tensors are written.
    struct Weights {
        let tree: SeedTree

        init(seed: UInt64) {
            self.tree = SeedTree(seed)
        }

        /// Uniform rows scaled for unit-variance outputs on unit-variance
        /// inputs, then quantized to 4-bit.
        func int4Rows(_ name: String, rows: Int, columns: Int,
                      amplitude: Float? = nil) -> [Quantization.Int4AffineRow] {
            var rng = tree.key(name)
            let a = amplitude ?? (3.0 / Float(columns)).squareRoot()
            return (0..<rows).map { _ in
                Quantization.quantizeInt4Affine((0..<columns).map { _ in rng.uniform(-a, a) })
            }
        }

        func int8Rows(_ name: String, rows: Int, columns: Int,
                      amplitude: Float) -> [Quantization.Int8AffineRow] {
            var rng = tree.key(name)
            return (0..<rows).map { _ in
                Quantization.quantizeInt8Affine((0..<columns).map { _ in rng.uniform(-amplitude, amplitude) })
            }
        }

        func bf16(_ name: String, count: Int, range: ClosedRange<Float>) -> [UInt16] {
            var rng = tree.key(name)
            return (0..<count).map { _ in
                Quantization.bf16Bits(rng.uniform(range.lowerBound, range.upperBound))
            }
        }
    }

    /// The bytes of one row of 4-bit codes at the requested width. At 8 bits
    /// each nibble becomes one byte with the same value.
    static func codes(_ row: Quantization.Int4AffineRow, columns: Int, bits: Int) -> [UInt8] {
        switch bits {
        case 4:
            return row.packed
        case 8:
            return (0..<columns).map { k in
                let byte = row.packed[k / 2]
                return k.isMultiple(of: 2) ? byte & 0x0F : byte >> 4
            }
        default:
            preconditionFailure("unsupported toy width \(bits)")
        }
    }

    private static func littleEndian(_ values: [UInt16]) -> [UInt8] {
        values.flatMap { [UInt8(truncatingIfNeeded: $0), UInt8(truncatingIfNeeded: $0 >> 8)] }
    }

    // MARK: - Resident region

    private struct ResidentTensor {
        let name: String
        let shape: [UInt32]
        let weights: [UInt8]
        let scales: [UInt8]
        let biases: [UInt8]
        var isAffine: Bool { !scales.isEmpty }
    }

    private static func residentTensors(bits: Int, weights: Weights) -> [ResidentTensor] {
        let d = config.hiddenSize
        func affine(_ name: String, rows: Int, columns: Int, amplitude: Float? = nil) -> ResidentTensor {
            let quantized = weights.int4Rows(name, rows: rows, columns: columns, amplitude: amplitude)
            return ResidentTensor(
                name: name,
                shape: [UInt32(rows), UInt32(columns), 0, 0],
                weights: quantized.flatMap { codes($0, columns: columns, bits: bits) },
                scales: littleEndian(quantized.flatMap(\.scales)),
                biases: littleEndian(quantized.flatMap(\.biases)))
        }
        func router(_ name: String) -> ResidentTensor {
            let quantized = weights.int8Rows(name, rows: config.numExperts, columns: d, amplitude: 0.1)
            return ResidentTensor(
                name: name,
                shape: [UInt32(config.numExperts), UInt32(d), 0, 0],
                weights: quantized.flatMap(\.packed),
                scales: littleEndian(quantized.flatMap(\.scales)),
                biases: littleEndian(quantized.flatMap(\.biases)))
        }
        func bf16(_ name: String, count: Int, range: ClosedRange<Float>) -> ResidentTensor {
            ResidentTensor(name: name,
                           shape: [UInt32(count), 0, 0, 0],
                           weights: littleEndian(weights.bf16(name, count: count, range: range)),
                           scales: [],
                           biases: [])
        }

        var tensors = [
            affine("language_model.model.embed_tokens.weight",
                   rows: config.vocabSize, columns: d, amplitude: 1.0),
            bf16("language_model.model.norm.weight", count: d, range: 0.75...1.25),
        ]
        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let headDim = isFull ? config.fullHeadDim : config.headDim
            let kvHeads = isFull ? config.numFullKVHeads : config.numKVHeads
            let qDim = config.numHeads * headDim
            let kvDim = kvHeads * headDim
            for norm in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm.weight",
                "router.scale",
            ] {
                tensors.append(bf16("\(prefix).\(norm)", count: d, range: 0.75...1.25))
            }
            tensors.append(bf16("\(prefix).self_attn.q_norm.weight", count: headDim, range: 0.75...1.25))
            tensors.append(bf16("\(prefix).self_attn.k_norm.weight", count: headDim, range: 0.75...1.25))
            tensors.append(bf16("\(prefix).router.per_expert_scale",
                                count: config.numExperts, range: 0.5...1.5))
            tensors.append(bf16("\(prefix).layer_scalar", count: 1, range: 0.9...1.1))
            tensors.append(affine("\(prefix).self_attn.q_proj.weight", rows: qDim, columns: d))
            tensors.append(affine("\(prefix).self_attn.k_proj.weight", rows: kvDim, columns: d))
            if !isFull {
                tensors.append(affine("\(prefix).self_attn.v_proj.weight", rows: kvDim, columns: d))
            }
            tensors.append(affine("\(prefix).self_attn.o_proj.weight", rows: d, columns: qDim))
            tensors.append(affine("\(prefix).mlp.gate_proj.weight",
                                  rows: config.intermediateSize, columns: d))
            tensors.append(affine("\(prefix).mlp.up_proj.weight",
                                  rows: config.intermediateSize, columns: d))
            tensors.append(affine("\(prefix).mlp.down_proj.weight",
                                  rows: d, columns: config.intermediateSize))
            tensors.append(router("\(prefix).router.proj.weight"))
        }
        return tensors
    }

    private static func writeResident(profile: QuantProfile,
                                      weights: Weights,
                                      shiftedResidentWeight: String?,
                                      to url: URL) throws -> (size: Int, sha: String) {
        let tensors = residentTensors(bits: profile.denseWeightBits, weights: weights)
        if let shiftedResidentWeight {
            precondition(tensors.contains { $0.name == shiftedResidentWeight && $0.isAffine },
                         "no quantized resident weight named \(shiftedResidentWeight)")
        }
        let weightAlignment = defaultResidentWeightAlignment(profile)
        let names = tensors.map(\.name)
        let stringTable = Array(names.joined().utf8)
        let entriesBase = GTurboBinary.indexHeaderBytes
        let stringTableBase = entriesBase + names.count * GTurboBinary.indexEntryBytes
        var nameOffsets: [UInt32] = []
        var cursor = 0
        for name in names {
            nameOffsets.append(UInt32(stringTableBase + cursor))
            cursor += name.utf8.count
        }
        let page = GTurboFormatV1.alignmentBytes
        let indexBytes = ((UInt64(stringTableBase + stringTable.count) + page - 1) / page) * page

        func alignUp(_ value: UInt64, _ alignment: UInt64) -> UInt64 {
            ((value + alignment - 1) / alignment) * alignment
        }
        var entries: [ResidentEntry] = []
        var payloadCursor = indexBytes
        for tensor in tensors {
            var weightOffset = alignUp(payloadCursor, tensor.isAffine ? weightAlignment : 2)
            if tensor.name == shiftedResidentWeight {
                weightOffset += 4
            }
            let weightBytes = UInt64(tensor.weights.count)
            let scaleBytes = UInt64(tensor.scales.count)
            let biasBytes = UInt64(tensor.biases.count)
            let scaleOffset = tensor.isAffine ? weightOffset + weightBytes : 0
            let biasOffset = tensor.isAffine ? scaleOffset + scaleBytes : 0
            entries.append(ResidentEntry(
                name: tensor.name,
                dtype: tensor.isAffine ? 0 : 1,
                logicalShape4: tensor.shape,
                fileOffset: weightOffset,
                sizeBytes: weightBytes,
                scaleOffset: scaleOffset,
                scaleSize: scaleBytes,
                biasOffset: biasOffset,
                biasSize: biasBytes,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(tensor.name),
                sourceScales: nil,
                sourceBiases: nil))
            payloadCursor = weightOffset + weightBytes + scaleBytes + biasBytes
        }
        let residentSize = payloadCursor - indexBytes

        var file = [UInt8](repeating: 0, count: Int(indexBytes + residentSize))
        file.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            for (index, entry) in entries.enumerated() {
                GTurboBinary.writeIndexEntry(
                    into: base.advanced(by: entriesBase + index * GTurboBinary.indexEntryBytes),
                    entry: entry,
                    nameOffset: nameOffsets[index])
            }
        }
        file.replaceSubrange(stringTableBase..<(stringTableBase + stringTable.count), with: stringTable)
        for (tensor, entry) in zip(tensors, entries) {
            let w = Int(entry.fileOffset)
            file.replaceSubrange(w..<(w + tensor.weights.count), with: tensor.weights)
            if tensor.isAffine {
                let s = Int(entry.scaleOffset)
                file.replaceSubrange(s..<(s + tensor.scales.count), with: tensor.scales)
                let b = Int(entry.biasOffset)
                file.replaceSubrange(b..<(b + tensor.biases.count), with: tensor.biases)
            }
        }
        try Data(file).write(to: url)
        return (file.count, try Sha256Verifier.hashFile(at: url))
    }

    // MARK: - Routed experts

    struct ExpertBlobLayout {
        /// Byte offset of each role's weights, scales and biases inside a blob,
        /// in the planner's order (gate, up, down; weights, scales, biases).
        let offsets: [String: Int]
        let stride: UInt64
        let tensors: [String: [String: Any]]
    }

    static func expertBlobLayout(bits: Int) -> ExpertBlobLayout {
        let d = config.hiddenSize
        let f = config.moeIntermediateSize
        var offsets: [String: Int] = [:]
        var tensors: [String: [String: Any]] = [:]
        var cursor = 0
        for (role, rows, columns) in [("gate", f, d), ("up", f, d), ("down", d, f)] {
            let weightBytes = rows * columns * bits / 8
            let auxBytes = rows * (columns / Quantization.groupSize) * MemoryLayout<UInt16>.size
            offsets[role] = cursor
            tensors[role] = ["offset": cursor, "size": weightBytes, "dtype": "U32",
                             "shape": [rows, columns], "bits": bits]
            cursor += weightBytes
            for suffix in ["scales", "biases"] {
                offsets["\(role)_\(suffix)"] = cursor
                tensors["\(role)_\(suffix)"] = ["offset": cursor, "size": auxBytes, "dtype": "BF16",
                                                "shape": [rows, columns / Quantization.groupSize]]
                cursor += auxBytes
            }
        }
        let page = Int(GTurboFormatV1.alignmentBytes)
        let stride = UInt64(((cursor + page - 1) / page) * page)
        return ExpertBlobLayout(offsets: offsets, stride: stride, tensors: tensors)
    }

    private static func expertBlobBytes(layer: Int, expert: Int, bits: Int,
                                        layout: ExpertBlobLayout,
                                        weights: Weights) -> [UInt8] {
        let d = config.hiddenSize
        let f = config.moeIntermediateSize
        var bytes: [UInt8] = []
        for (role, rows, columns) in [("gate", f, d), ("up", f, d), ("down", d, f)] {
            precondition(bytes.count == layout.offsets[role])
            let quantized = weights.int4Rows("routed.\(layer).\(expert).\(role)",
                                             rows: rows, columns: columns)
            bytes += quantized.flatMap { codes($0, columns: columns, bits: bits) }
            bytes += littleEndian(quantized.flatMap(\.scales))
            bytes += littleEndian(quantized.flatMap(\.biases))
        }
        return bytes
    }
}
