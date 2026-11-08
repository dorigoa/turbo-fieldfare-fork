import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
}

public enum AppModelInstallationProbe {
    public static func status(
        at directory: URL,
        descriptor: AppModelInstallDescriptor = .default
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        do {
            let manifest = try ManifestReader.load(directoryURL: directory, expecting: .gemma4_26B_A4B)
            let expectedSource = "sha256:" + descriptor.sourceIndexSHA256
            guard manifest.sourceSnapshotHash == expectedSource else {
                return .partial("installed checkpoint does not match \(descriptor.displayName)")
            }
            let layout = directory.appendingPathComponent("packed_experts/layout.json")
            guard FileManager.default.fileExists(atPath: layout.path) else {
                return .partial("packed_experts/layout.json is missing")
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }

    /// The catalog variant an installed manifest was built from.
    ///
    /// Read from `sourceSnapshotHash`, the SHA-256 of the source index the
    /// repacker pinned, so a folder's name never decides it. Nil when there is
    /// no manifest, it cannot be read, or it names a checkpoint outside the
    /// catalog. Says nothing about whether the install is complete: that is
    /// `status`.
    public static func detectVariant(at directory: URL) -> SupportedModelVariant? {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path),
              let manifest = try? ManifestReader.load(directoryURL: directory,
                                                       expecting: .gemma4_26B_A4B),
              let hash = manifest.sourceSnapshotHash else {
            return nil
        }
        return SupportedModelVariant.variant(forSourceSnapshotHash: hash)
    }

    /// The status of an install of whichever catalog variant its manifest
    /// names, for choosing an existing model: any known variant is usable, and
    /// picking one switches the app to it.
    public static func status(anyVariantAt directory: URL) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }
        let manifest: Manifest
        do {
            manifest = try ManifestReader.load(directoryURL: directory, expecting: .gemma4_26B_A4B)
        } catch {
            return .partial("\(error)")
        }
        guard let hash = manifest.sourceSnapshotHash,
              let variant = SupportedModelVariant.variant(forSourceSnapshotHash: hash) else {
            let known = SupportedModelVariant.allCases.map(\.shortLabel).joined(separator: " or ")
            return .partial("installed checkpoint is not a supported variant (\(known))")
        }
        return status(at: directory, descriptor: .descriptor(for: variant))
    }
}
