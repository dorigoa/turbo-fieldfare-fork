import Foundation

/// What an install plan writes, computed from source metadata alone. A dry run
/// prints it; a real install reports the same figures for the files it wrote.
public struct RemoteRepackPlanSummary: Sendable, Equatable {
    public struct QuantBits: Sendable, Equatable {
        public let embedding: Int
        public let attention: Int
        public let router: Int
        public let sharedExpert: Int
        public let routedExpert: Int
    }

    public let quantBits: QuantBits
    /// `model_weights.bin`: index page plus resident payload.
    public let residentFileBytes: UInt64
    /// Byte alignment of quantized resident weights: 1 (packed) or 16.
    public let residentQuantizedWeightAlignment: UInt64
    /// Number of `packed_experts/layer_XX.bin` files with experts.
    public let layerFileCount: Int
    /// Size of the largest layer file (they are all equal for Gemma 4).
    public let layerFileBytes: UInt64
    /// All layer files together.
    public let packedExpertBytes: UInt64
    /// Bytes of one expert's nine sub-tensors before page padding.
    public let expertPayloadBytes: UInt64
    /// Expert blob stride, padded to the 16 KiB `.gturbo` page.
    public let expertStride: UInt64
    /// `packed_experts/layout.json`, encoded exactly as the install writes it.
    public let layoutBytes: UInt64
    /// Vision and audio tensors the text pack leaves out.
    public let excludedMultimodalTensorCount: Int
    /// SHA-256 of the resident index page the install writes.
    public let residentIndexSHA256: String
    /// Canonical copy-plan fingerprint, as stored in a resume checkpoint.
    public let planFingerprint: String

    /// Model payload the install writes: resident file plus layer files.
    public var payloadOutputBytes: UInt64 { residentFileBytes + packedExpertBytes }
}

/// Sizes of the small files around the payload. Only a dry run resolves them,
/// with header-only requests, so the catalog can pin download and install sizes
/// without fetching a byte of the checkpoint itself.
public struct RemoteRepackMetadataEstimate: Sendable, Equatable {
    /// Index, `config.json` and every shard's safetensors header, which the
    /// dry run itself already fetched.
    public let sourceMetadataBytes: UInt64
    /// Files an install stores under `tokenizer/`, keyed by file name,
    /// including the copy of `config.json`.
    public let tokenizerFiles: [String: UInt64]

    /// Tokenizer files fetched after the payload; `config.json` is copied from
    /// the metadata already on disk, not fetched again.
    public var tokenizerDownloadBytes: UInt64 {
        tokenizerFiles.filter { $0.key != "config.json" }.values.reduce(0, +)
    }

    /// Bytes the `tokenizer/` directory occupies after an install.
    public var tokenizerInstalledBytes: UInt64 {
        tokenizerFiles.values.reduce(0, +)
    }
}

extension RemoteRepackPlanSummary {
    init(plan: RepackPlan, rangePlan: RangeCopyPlan) throws {
        let bits = try plan.manifestQuantBitWidths()
        let routedLayers = plan.layers.filter { $0.expertsPerLayer > 0 }
        let stride = routedLayers.first?.expertStride ?? 0
        let layout = try GTurboJSON.encodeLayout(plan: plan, expertStride: stride)
        self.init(
            quantBits: QuantBits(embedding: bits.embedding,
                                 attention: bits.attention,
                                 router: bits.router,
                                 sharedExpert: bits.sharedExpert,
                                 routedExpert: bits.routedExpert),
            residentFileBytes: plan.resident.totalSize,
            residentQuantizedWeightAlignment: plan.residentQuantizedWeightAlignment,
            layerFileCount: routedLayers.count,
            layerFileBytes: routedLayers.map(\.fileSize).max() ?? 0,
            packedExpertBytes: routedLayers.reduce(UInt64(0)) { $0 + $1.fileSize },
            expertPayloadBytes: routedLayers.first?.subTensors
                .reduce(UInt64(0)) { $0 + $1.sizeInExpertBlob } ?? 0,
            expertStride: stride,
            layoutBytes: UInt64(layout.count),
            excludedMultimodalTensorCount: plan.excludedMultimodalTensorNames.count,
            residentIndexSHA256: rangePlan.residentIndexSha256,
            planFingerprint: rangePlan.canonicalFingerprint)
    }
}
