import Foundation
import TurboFieldfareFormat

/// The pinned checkpoints TurboFieldfare can install, one per quantization.
///
/// Both Gemma variants are MLX affine group-64 uploads of the same Gemma 4
/// 26B-A4B instruction checkpoint, with the same 1,697 tensor names. They
/// differ only in weight bit width: `int4` packs embedding, attention, shared
/// and routed experts at 4 bits with an 8-bit router; `int8` packs every
/// quantized tensor at 8 bits. `qwen36` is the pinned MLX upload of Qwen
/// 3.6 35B-A3B at 4 bits; the catalog knows it, the repacker does not install
/// it until the Qwen repack lands. Each variant is pinned by revision and by
/// the SHA-256 of its `model.safetensors.index.json`, so an upstream
/// re-upload is refused instead of silently installed.
public enum SupportedModelVariant: String, CaseIterable, Codable, Sendable {
    case int4
    case int8
    case qwen36

    public var displayName: String {
        switch self {
        case .int4: "Gemma 4 26B-A4B IT 4-bit"
        case .int8: "Gemma 4 26B-A4B IT 8-bit"
        case .qwen36: "Qwen3.6 35B-A3B 4-bit"
        }
    }

    public var shortLabel: String {
        switch self {
        case .int4: "4-bit"
        case .int8: "8-bit"
        case .qwen36: "Qwen3.6"
        }
    }

    public var repoID: String {
        switch self {
        case .int4: "mlx-community/gemma-4-26b-a4b-it-4bit"
        case .int8: "mlx-community/gemma-4-26b-a4b-it-8bit"
        case .qwen36: "mlx-community/Qwen3.6-35B-A3B-4bit"
        }
    }

    public var revision: String {
        switch self {
        case .int4: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
        case .int8: "33c6d23798a0af159529890f79329206dbfbd73c"
        case .qwen36: "38740b847e4cb78f352aba30aa41c76e08e6eb46"
        }
    }

    public var sourceIndexSHA256: String {
        switch self {
        case .int4: "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13"
        case .int8: "4b96ec862d7ae3f8d150b22b295cefcc9b5ef5e83f172f530dc1f2d2c11339cc"
        case .qwen36: "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea"
        }
    }

    public var approximateDownloadBytes: UInt64 {
        switch self {
        case .int4: 14_620_479_420
        // From `TurboFieldfareRepack --dry-run --variant int8` at the pinned
        // revision with the default 64 MiB ranges: 26,939,947,964 payload
        // bytes in 418 ranges, plus 413,177 bytes of index, config and
        // safetensors headers, plus 32,189,832 bytes of tokenizer files.
        // (The int4 figure above is the payload ranges alone.)
        case .int8: 26_972_550_973
        // From `TurboFieldfareRepack --dry-run --variant qwen36` at the
        // pinned revision with the default 64 MiB ranges: 20,146,183,200
        // payload bytes in 308 ranges, plus 513,665 bytes of index, config
        // and safetensors headers, plus 19,998,246 bytes of tokenizer files.
        case .qwen36: 20_166_695_111
        }
    }

    public var installedBytes: UInt64 {
        switch self {
        case .int4: 14_291_921_884
        // From the same dry run: model_weights.bin 2,545,592,786 bytes (with
        // the 16-byte resident weight alignment), 30 expert files of
        // 809,500,672 bytes, layout.json 8,465,996 bytes and 32,202,012 bytes
        // of tokenizer files make 26,871,280,954 bytes. manifest.json and
        // verified-install.json add about 14 KB; the total is rounded up to
        // the next 100,000 bytes.
        case .int8: 26_871_300_000
        // From the same dry run: model_weights.bin 1,389,476,096 bytes, 40
        // expert files of 452,984,832 bytes (expert payload and stride
        // 1,769,472), layout.json 22,493,846 bytes and 20,021,837 bytes of
        // tokenizer files make 19,551,385,059 bytes. manifest.json and
        // verified-install.json add a few KB; the total is rounded up to
        // the next 100,000 bytes.
        case .qwen36: 19_551_400_000
        }
    }

    public var reserveBytes: UInt64 { 1_073_741_824 }

    /// Directory name a fresh install of this variant uses.
    public var defaultDirectoryName: String {
        switch self {
        case .int4: "gemma4.gturbo"
        case .int8: "gemma4-8bit.gturbo"
        case .qwen36: "qwen36.gturbo"
        }
    }

    /// The smallest Mac this variant is meant for. Load admission itself is
    /// decided by `ContextAdmission`; this figure is for install-time copy.
    public var recommendedMinimumHostMemoryBytes: UInt64 {
        switch self {
        case .int4: 8 << 30
        case .int8: 16 << 30
        // The S6 dry run measures disk sizes, not memory; the figure stays
        // at the 16 GB Mac class the 8-bit variant already targets.
        case .qwen36: 16 << 30
        }
    }

    /// The catalog entries whose `family` is Gemma 4 (`nil` model family,
    /// the one v1 shipped with). The app's model selectors filter through
    /// this until Qwen3.6 becomes installable there (S24).
    public static var gemma4Variants: [SupportedModelVariant] {
        allCases.filter { $0.family == nil }
    }

    /// The manifest model family this variant installs (C1). `nil` is Gemma 4,
    /// the family v1 shipped with: a manifest without `modelFamily` is Gemma,
    /// so the Gemma variants keep declaring nothing.
    package var family: GTurboModelFamily? {
        switch self {
        case .int4, .int8: nil
        case .qwen36: .qwen35Moe
        }
    }

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false) -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }

    /// The variant whose source index hashes to `hex` (64 hex digits).
    public static func variant(forSourceIndexSHA256 hex: String) -> SupportedModelVariant? {
        let normalized = hex.lowercased()
        return allCases.first { $0.sourceIndexSHA256 == normalized }
    }

    /// The variant a manifest's `sourceSnapshotHash` (`"sha256:<hex>"`) names.
    public static func variant(forSourceSnapshotHash value: String) -> SupportedModelVariant? {
        let prefix = "sha256:"
        guard value.hasPrefix(prefix) else { return nil }
        return variant(forSourceIndexSHA256: String(value.dropFirst(prefix.count)))
    }

    /// The variant installed from `id`, which is also the manifest `modelID`.
    public static func variant(forRepoID id: String) -> SupportedModelVariant? {
        allCases.first { $0.repoID == id }
    }
}
