import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

/// The checkpoints the app can install and load, named for its UI modules.
///
/// The catalog lives in the installer core, which the Mac targets do not
/// import. The alias lets a picker, a label or an accessibility identifier name
/// the type through `TurboFieldfareAppCore` alone.
public typealias AppModelVariant = SupportedModelVariant

/// The sampling values the app runs a model with: what the Inspector edits and
/// every generation request carries.
public struct AppSamplingValues: Codable, Equatable, Sendable {
    public var temperature: Double
    public var topKEnabled: Bool
    public var topK: Int
    public var topPEnabled: Bool
    public var topP: Double

    public init(temperature: Double, topKEnabled: Bool, topK: Int,
                topPEnabled: Bool, topP: Double) {
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
    }
}

extension SupportedModelVariant {
    /// The quantization profile a complete install of this variant carries.
    ///
    /// `qwen36` shares the 4-bit profile: embedding, attention and routed
    /// experts at 4 bits, router and shared-expert gate at 8, affine groups
    /// of 64, the same slot shape `QuantProfile.affine4` describes.
    public var quantProfile: QuantProfile {
        switch self {
        case .int4: .affine4
        case .int8: .affine8
        case .qwen36: .affine4
        }
    }

    /// What admission charges this variant beyond its KV cache.
    ///
    /// Taken from the variant rather than from an installed manifest, so the
    /// context menu, the clamp and the load refusal can answer before anything
    /// has been downloaded. The decode service, which loads a real directory,
    /// reads the manifest instead.
    public var modelFootprint: ModelFootprint {
        switch self {
        case .int4, .int8: .forProfile(quantProfile)
        case .qwen36: .qwen36Provisional
        }
    }

    /// The architecture whose geometry admission uses for this variant: the
    /// pinned constants of its checkpoint, so the KV term counts the family's
    /// own full-attention layers (S24).
    public var admissionArchConfig: ArchConfig {
        switch self {
        case .int4, .int8: .gemma4_26B_A4B
        case .qwen36: .qwen36_35B_A3B
        }
    }

    /// The sampling the family ships with, used until the user changes any of
    /// it (U4, S24): Qwen3.6 from its checkpoint's `generation_config.json` —
    /// temperature 1.0, top-k 20, top-p 0.95 — where Gemma has always run
    /// 0.2 / 64 / 0.95. Both families ship with top-k and top-p filtering on,
    /// as the Inspector's toggles have always defaulted to.
    public var defaultSampling: AppSamplingValues {
        switch self {
        case .int4, .int8:
            AppSamplingValues(temperature: 0.2, topKEnabled: true, topK: 64,
                              topPEnabled: true, topP: 0.95)
        case .qwen36:
            AppSamplingValues(temperature: 1.0, topKEnabled: true, topK: 20,
                              topPEnabled: true, topP: 0.95)
        }
    }

    /// Whether this family can take image input (D12, S24).
    ///
    /// Qwen3.6 has no image tower: the app disables attachments and says so
    /// with the same sentence everywhere, rather than accepting an image and
    /// ignoring it later.
    public var supportsImages: Bool {
        switch self {
        case .int4, .int8: true
        case .qwen36: false
        }
    }

    /// The caveat shown before this variant is downloaded, or nil when the
    /// default download needs none.
    ///
    /// Derived from the catalog, so pinning the 8-bit figures moves the copy
    /// with them: about 27 GB to download and a 16 GB Mac to run it.
    public var installNote: String? {
        switch self {
        case .int4:
            return nil
        case .qwen36, .int8:
            let download = Int((Double(approximateDownloadBytes) / 1e9).rounded())
            let memory = recommendedMinimumHostMemoryBytes >> 30
            return "~\(download) GB download; \(memory) GB memory recommended"
        }
    }
}
