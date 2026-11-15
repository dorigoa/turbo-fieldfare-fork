import Foundation
import TurboFieldfareRepackCore

public struct AppModelInstallDescriptor: Equatable, Sendable {
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let rangeStagingBytes: UInt64
    public let reserveBytes: UInt64

    public init(displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
    }

    public var requiredFreeBytes: UInt64 {
        installedBytes + rangeStagingBytes + reserveBytes
    }

    /// The download of one catalog variant. Every figure comes from
    /// `SupportedModelVariant`, so the installer, the disk check and the
    /// Inspector quote the same numbers the repack command does.
    public static func descriptor(for variant: SupportedModelVariant) -> AppModelInstallDescriptor {
        AppModelInstallDescriptor(
            displayName: variant.displayName,
            repoID: variant.repoID,
            revision: variant.revision,
            sourceIndexSHA256: variant.sourceIndexSHA256,
            approximateDownloadBytes: variant.approximateDownloadBytes,
            installedBytes: variant.installedBytes,
            rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
            reserveBytes: variant.reserveBytes)
    }

    /// The 4-bit download, which is what the app installs unless the user
    /// chooses another variant.
    public static let `default` = descriptor(for: .int4)

    /// The image companion for a text variant. Its source is that variant's
    /// own (`SupportedModelVariant.visionCompanionSource`): the projector
    /// belongs with one checkpoint.
    public static func visionCompanion(for source: VisionCompanionSource)
        -> AppModelInstallDescriptor {
        AppModelInstallDescriptor(
            displayName: "Gemma 4 Image Support",
            repoID: source.repoID,
            revision: source.revision,
            sourceIndexSHA256: source.sourceIndexSHA256,
            approximateDownloadBytes: source.approximateDownloadBytes,
            installedBytes: source.installedBytes,
            rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
            reserveBytes: 1_073_741_824)
    }

    /// The default variant's image companion.
    public static let visionCompanion = visionCompanion(for: .gemma4QAT4bit)
}

public struct AppModelInstallRequirement: Equatable, Sendable {
    public let probePath: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(probePath: String = "", requiredBytes: UInt64, availableBytes: UInt64) {
        self.probePath = probePath
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum AppModelInstallReadiness: Equatable, Sendable {
    case installed
    case checking
    case ready(AppModelInstallRequirement)
    case insufficientSpace(AppModelInstallRequirement)
    case failed(String)

    public var requirement: AppModelInstallRequirement? {
        switch self {
        case .ready(let requirement), .insufficientSpace(let requirement):
            return requirement
        case .installed, .checking, .failed:
            return nil
        }
    }
}
