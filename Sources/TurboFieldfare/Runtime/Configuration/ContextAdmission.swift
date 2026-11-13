import Foundation

/// A model's memory profile for admission; see `ContextAdmission.ModelFootprint`.
public typealias ModelFootprint = ContextAdmission.ModelFootprint

/// The one host-memory admission rule for a requested context length.
///
/// Every surface that can pick a context — the app menu, the CLI, the server,
/// the decode service — asks this type whether the host can back it, so a
/// refusal, a hidden menu row and a warning all quote the same arithmetic. The
/// numbers are FP16 KV geometry plus two measured constants; nothing here
/// allocates or touches the model.
public enum ContextAdmission {

    /// What a model occupies beyond its KV cache, and the headroom it needs.
    ///
    /// The figures differ by quantization profile: an 8-bit pack maps about
    /// twice the resident weights and fills slots twice the size, and each
    /// expert miss reads twice the bytes, so it also needs more page cache.
    public struct ModelFootprint: Sendable, Equatable {
        /// Resident weights, the 16 baseline expert-cache slots, prefill
        /// scratch, the tokenizer and the Metal library.
        public let nonKVRuntimeBytes: UInt64
        /// Bytes per routed-expert slot before host-page rounding.
        public let expertStrideBytes: Int
        /// Headroom left to the OS and to a minimum routed-expert page cache.
        public let hostReserveBytes: UInt64
        /// Recurrent state the family carries outside the KV cache, charged
        /// once per loaded model rather than per token. Zero for Gemma, whose
        /// only per-position state is the KV cache; for Qwen3.6 it is the
        /// DeltaNet recurrent state plus the short-conv state (3.4).
        public let linearStateBytes: UInt64

        public init(nonKVRuntimeBytes: UInt64, expertStrideBytes: Int, hostReserveBytes: UInt64,
                    linearStateBytes: UInt64 = 0) {
            self.nonKVRuntimeBytes = nonKVRuntimeBytes
            self.expertStrideBytes = expertStrideBytes
            self.hostReserveBytes = hostReserveBytes
            self.linearStateBytes = linearStateBytes
        }

        /// The 4-bit pack. The non-KV figure was measured at 1.75-1.94 GB
        /// across PR 156's five rungs and the M2-8 8K row at 16 cache slots;
        /// the pinned IT pack uses 3,358,720 bytes per routed expert.
        public static let affine4Measured = ModelFootprint(
            nonKVRuntimeBytes: 2 << 30,
            expertStrideBytes: 3_358_720,
            hostReserveBytes: 3 << 30)

        /// The 8-bit pack. The non-KV figure was measured on a 32 GB M5 at
        /// 3.07-3.09 GiB of peak physical footprint, net of the FP16 KV cache
        /// and of the slots above 16, at 4K and 8K with 16 and 32 slots; the
        /// 4-bit pack measured 1.73-1.75 GiB on the same runs. 3.5 GiB keeps
        /// about the margin the 4-bit figure keeps over its measurements. The
        /// host reserve is not measured: it stays scaled from the 4-bit one
        /// for the larger expert reads, so an 8 GB Mac is refused and a
        /// 16 GB Mac is admitted. The pinned 8-bit pack uses 6,324,224 bytes
        /// per routed expert.
        public static let affine8Measured = ModelFootprint(
            nonKVRuntimeBytes: 7 << 29,
            expertStrideBytes: 6_324_224,
            hostReserveBytes: 5 << 30)

        /// The name the 8-bit figures had before they were measured.
        public static let affine8Provisional = affine8Measured

        /// Qwen3.6, provisional (S24). The non-KV figure is an estimate, not a
        /// measurement: the 4-bit Gemma pack measures 1.75-1.94 GB resident
        /// around a 1.39 GB weight estimate (3.4), and Qwen's resident set is
        /// somewhat larger, so 2.5 GiB carries it with margin until S27
        /// measures it here. The DeltaNet (60 MiB) and conv (2.8 MiB) state is
        /// constant at 64 MiB (3.4, D5). The reserve is 3 GiB: the routed
        /// experts stream from 40 layer files with a cold-cache read of about
        /// 566 MB per token (3.4), between the 4-bit and 8-bit reserves. The
        /// pinned pack streams each routed expert at 1,769,472 bytes (S6).
        public static let qwen36Provisional = ModelFootprint(
            nonKVRuntimeBytes: 5 << 29,
            expertStrideBytes: 1_769_472,
            hostReserveBytes: 3 << 30,
            linearStateBytes: 64 << 20)

        public static func forProfile(_ profile: QuantProfile) -> ModelFootprint {
            switch profile {
            case .affine4: affine4Measured
            case .affine8: affine8Measured
            }
        }

        /// The footprint of an installed model, from its manifest alone. Reads
        /// a few kilobytes and never touches the weights. The family decides
        /// the profile first: Qwen3.6 shares the 4-bit quantization shape with
        /// the Gemma pack but not its memory profile (S24), so its manifest
        /// answers with the family's provisional figures.
        public static func forModelDirectory(_ directoryURL: URL,
                                             config: ArchConfig = .gemma4_26B_A4B) throws
            -> ModelFootprint {
            let manifest = try ManifestReader.load(directoryURL: directoryURL, expecting: config)
            return try ContextAdmission.footprint(of: manifest)
        }
    }

    /// The architecture whose geometry the KV term of admission applies to,
    /// per model family: the pinned constants of that family's checkpoint.
    public static func archConfig(forFamily family: ModelFamily) -> ArchConfig {
        switch family {
        case .gemma4: .gemma4_26B_A4B
        case .qwen35Moe: .qwen36_35B_A3B
        }
    }

    /// The architecture and footprint of an installed model together, from
    /// its manifest alone — the two answers every load-time admission needs
    /// so that a Qwen manifest is not measured with Gemma's geometry (S24).
    public static func admissionProfile(forModelDirectory directoryURL: URL,
                                        config: ArchConfig = .gemma4_26B_A4B) throws
        -> (config: ArchConfig, footprint: ModelFootprint) {
        let manifest = try ManifestReader.load(directoryURL: directoryURL, expecting: config)
        let family: ModelFamily
        if let name = manifest.modelFamily, ModelFamily(rawValue: name) == .qwen35Moe {
            family = .qwen35Moe
        } else {
            family = .gemma4
        }
        return (archConfig(forFamily: family), try footprint(of: manifest))
    }

    private static func footprint(of manifest: Manifest) throws -> ModelFootprint {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required to size the model's memory")
        }
        if manifest.modelFamily != nil {
            return .qwen36Provisional
        }
        return ModelFootprint.forProfile(try QuantProfile(manifestQuant: quant))
    }

    /// The 4-bit pack's non-KV footprint; see `ModelFootprint.affine4Measured`.
    public static let nonKVRuntimeFootprintBytes: UInt64 = ModelFootprint.affine4Measured.nonKVRuntimeBytes

    private static func additionalExpertCacheBytes(config: ArchConfig, slots: Int,
                                                   footprint: ModelFootprint) -> UInt64 {
        let page = Int(getpagesize())
        let slotBytes = ((footprint.expertStrideBytes + page - 1) / page) * page
        // Do not discount the measured baseline for smaller caches: the rest
        // of that allowance has not been measured separately.
        return UInt64(max(0, slots - 16)) * UInt64(config.numLayers) * UInt64(slotBytes)
    }

    /// The 4-bit pack's headroom; see `ModelFootprint.affine4Measured`. A
    /// host that satisfied only the projected footprint would run with every
    /// expert read faulting against a cold cache.
    public static let hostReserveBytes: UInt64 = ModelFootprint.affine4Measured.hostReserveBytes

    public enum Availability: Equatable, Sendable {
        case available
        case needsMemory(minimumHostBytes: UInt64)
    }

    /// Rows the production runtime allocates per sliding-window layer.
    ///
    /// `RealForwardRunner` floors its `maxPrefillChunkTokens` at the pooled
    /// image-token count, and `KVCacheManager` sizes the ring at
    /// `slidingWindow + maxPrefillChunkTokens`, so the widest chunk the runtime
    /// may see — not the default text chunk — is what the ring costs. 1,304
    /// today.
    public static func productionSlidingRingRows(config: ArchConfig, maxContext: Int) -> Int {
        let chunkRows = max(PrefillRuntimeConfig.defaultChunked.chunkTokens,
                            VisionConfig().maximumPooledTokens)
        return min(maxContext, config.slidingWindow + chunkRows)
    }

    /// FP16 K and V bytes the runtime allocates for `maxContext`.
    ///
    /// Sliding-window layers are ring-capped; only the full-attention layers
    /// grow with the context, at 20,480 B per token across the five of them.
    /// A `slidingWindow` of zero is the Qwen family's signature (D1): its
    /// linear layers hold DeltaNet state instead of KV (D5), so only the
    /// attention layers allocate — also 20,480 B per token, but across ten.
    public static func fp16KVBytes(config: ArchConfig, maxContext: Int) -> UInt64 {
        let fullLayers = config.fullAttentionLayerMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        let keyAndValue = 2
        let fp16Bytes = 2
        let fullBytesPerRow = config.numFullKVHeads * config.fullHeadDim * keyAndValue * fp16Bytes
        let fullBytes = UInt64(fullLayers * maxContext * fullBytesPerRow)
        guard config.slidingWindow > 0 else { return fullBytes }
        let slidingLayers = config.numLayers - fullLayers
        let slidingRows = productionSlidingRingRows(config: config, maxContext: maxContext)
        let slidingBytesPerRow = config.numKVHeads * config.headDim * keyAndValue * fp16Bytes
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow) + fullBytes
    }

    /// What the process is expected to occupy at this context.
    public static func projectedFootprintBytes(config: ArchConfig, maxContext: Int, expertCacheSlots: Int = 16,
                                               footprint: ModelFootprint = .affine4Measured) -> UInt64 {
        fp16KVBytes(config: config, maxContext: maxContext) + footprint.nonKVRuntimeBytes
            + footprint.linearStateBytes
            + additionalExpertCacheBytes(config: config, slots: expertCacheSlots, footprint: footprint)
    }

    /// The smallest host memory that admits this context.
    public static func minimumHostMemoryBytes(config: ArchConfig, maxContext: Int, expertCacheSlots: Int = 16,
                                              footprint: ModelFootprint = .affine4Measured) -> UInt64 {
        projectedFootprintBytes(config: config, maxContext: maxContext, expertCacheSlots: expertCacheSlots,
                                footprint: footprint) + footprint.hostReserveBytes
    }

    public static func availability(config: ArchConfig,
                                    maxContext: Int,
                                    hostMemoryBytes: UInt64, expertCacheSlots: Int = 16,
                                    footprint: ModelFootprint = .affine4Measured) -> Availability {
        let minimum = minimumHostMemoryBytes(config: config, maxContext: maxContext, expertCacheSlots: expertCacheSlots,
                                             footprint: footprint)
        return hostMemoryBytes >= minimum ? .available : .needsMemory(minimumHostBytes: minimum)
    }

    public static var hostMemoryBytes: UInt64 { ProcessInfo.processInfo.physicalMemory }

    /// One sentence naming what the context needs and what this Mac has, for a
    /// refusal, a hidden menu row's caption or a CLI warning.
    public static func needDescription(config: ArchConfig,
                                       maxContext: Int,
                                       hostMemoryBytes: UInt64, expertCacheSlots: Int = 16,
                                       footprint: ModelFootprint = .affine4Measured) -> String {
        let need = marketingGigabytesRoundedUp(
            minimumHostMemoryBytes(config: config, maxContext: maxContext, expertCacheSlots: expertCacheSlots,
                                   footprint: footprint))
        let have = marketingGigabytesNearest(hostMemoryBytes)
        return "A \(grouped(maxContext))-token context needs \(need) GB of memory; this Mac has \(have) GB."
    }

    /// Memory sizes Apple actually ships, so the sentence names a machine the
    /// reader can recognize instead of "10.25 GiB".
    private static let machineSizesGB = [8, 16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]

    /// The requirement is rounded in GiB because that is the unit
    /// `physicalMemory` reports: an "8 GB Mac" answers 8,589,934,592, so a need
    /// of 8,320,122,880 is met by it and must print as 8 GB, not 16.
    private static func marketingGigabytesRoundedUp(_ bytes: UInt64) -> Int {
        if let size = machineSizesGB.first(where: { UInt64($0) << 30 >= bytes }) { return size }
        return Int((bytes + (1 << 30) - 1) >> 30)
    }

    /// The host is reported in decimal GB, which is how the same machine is
    /// labeled in About This Mac.
    private static func marketingGigabytesNearest(_ bytes: UInt64) -> Int {
        let decimal = Double(bytes) / 1e9
        guard let nearest = machineSizesGB.min(by: {
            abs(Double($0) - decimal) < abs(Double($1) - decimal)
        }) else { return Int(decimal.rounded()) }
        return nearest
    }

    /// Grouped by threes without a locale, so the message reads the same in
    /// every region and in test assertions.
    private static func grouped(_ value: Int) -> String {
        let digits = Array(String(value))
        var out = ""
        for (index, digit) in digits.enumerated() {
            if index > 0, (digits.count - index) % 3 == 0 { out.append(",") }
            out.append(digit)
        }
        return out
    }
}
