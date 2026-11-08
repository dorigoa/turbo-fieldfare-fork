import Foundation

/// The weight bit widths the runtime knows how to execute, as one value.
///
/// `manifest.quant` records bits per slot, which admits combinations no
/// kernel set was written for. The runtime accepts exactly the two pinned
/// checkpoints' layouts and refuses everything else at load, so a manifest
/// cannot pair, say, 8-bit embeddings with 4-bit routed experts and reach a
/// kernel that would read them with the wrong unpack.
///
/// - `affine4`: embedding 4, attention 4, router 8, shared expert 4 or 8
///   (8 is the historical shared-expert pack), routed experts 4.
/// - `affine8`: every slot 8.
public enum QuantProfile: String, Sendable, Equatable, CaseIterable {
    case affine4
    case affine8

    public init(manifestQuant quant: ManifestQuant) throws {
        let embedding = quant.embedding.weightBits
        let attention = quant.attention.weightBits
        let router = quant.router.weightBits
        let shared = quant.sharedExpert.weightBits
        let routed = quant.routedExpert.weightBits
        if embedding == 4, attention == 4, router == 8,
           shared == 4 || shared == 8, routed == 4 {
            self = .affine4
        } else if embedding == 8, attention == 8, router == 8,
                  shared == 8, routed == 8 {
            self = .affine8
        } else {
            throw ModelError.indexCorrupt(
                detail: "unsupported quantization profile: embedding \(embedding), "
                    + "attention \(attention), router \(router), sharedExpert \(shared), "
                    + "routedExpert \(routed) bits")
        }
    }

    /// Bits of the embedding, the tied LM head and the attention projections.
    public var denseWeightBits: Int {
        switch self {
        case .affine4: 4
        case .affine8: 8
        }
    }

    /// Bits of the streamed routed experts.
    public var routedWeightBits: Int {
        switch self {
        case .affine4: 4
        case .affine8: 8
        }
    }

    /// The router is 8-bit in both pinned checkpoints.
    public var routerWeightBits: Int { 8 }
}

/// Raised when a model's profile is recognized but this build cannot run it.
public enum QuantProfileError: Error, Equatable, CustomStringConvertible {
    case runtimeKernelsUnavailable(QuantProfile)

    public var description: String {
        switch self {
        case .runtimeKernelsUnavailable(let profile):
            "this build recognizes \(profile.rawValue) models but has no runtime kernels for them; "
                + "update TurboFieldfare to run this model"
        }
    }
}
