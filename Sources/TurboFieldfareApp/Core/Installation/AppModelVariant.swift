import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

/// The checkpoints the app can install and load, named for its UI modules.
///
/// The catalog lives in the installer core, which the Mac targets do not
/// import. The alias lets a picker, a label or an accessibility identifier name
/// the type through `TurboFieldfareAppCore` alone.
public typealias AppModelVariant = SupportedModelVariant

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
