import Foundation

/// The default install source: the 4-bit variant. Kept as its own type so
/// every existing caller and the public API stay unchanged; new code that has
/// to tell the variants apart uses `SupportedModelVariant`.
public enum SupportedModelSource {
    public static let variant: SupportedModelVariant = .int4
    public static let displayName = variant.displayName
    public static let repoID = variant.repoID
    public static let revision = variant.revision
    public static let sourceIndexSHA256 = variant.sourceIndexSHA256
    public static let approximateDownloadBytes: UInt64 = variant.approximateDownloadBytes
    public static let installedBytes: UInt64 = variant.installedBytes
    public static let reserveBytes: UInt64 = variant.reserveBytes

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        variant.installOptions(outputDirectory: outputDirectory,
                               overwrite: overwrite,
                               token: token,
                               resume: resume)
    }
}
