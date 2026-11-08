import Foundation
import TurboFieldfareRepackCore

enum AppModelLocation {
    static func defaultURL(environment: [String: String] = ProcessInfo.processInfo.environment,
                           persistedPath: String? = nil,
                           variant: SupportedModelVariant = .int4) -> URL {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        return resolve(
            explicitURL: nil,
            environment: environment,
            persistedPath: persistedPath,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:),
            variant: variant)
    }

    /// Where `variant` lives when the user has not chosen a place for it: beside
    /// the 4-bit model, under the variant's own directory name. Someone who keeps
    /// the 4-bit model in `~/LLM/TurboFieldfare` finds the 8-bit one there too,
    /// and the two never share a directory.
    static func defaultURL(for variant: SupportedModelVariant, besideFourBit fourBit: URL) -> URL {
        let fourBit = fourBit.standardizedFileURL
        guard variant != .int4 else { return fourBit }
        return fourBit.deletingLastPathComponent()
            .appendingPathComponent(variant.defaultDirectoryName, isDirectory: true)
            .standardizedFileURL
    }

    static func resolve(explicitURL: URL?,
                        environment: [String: String] = [:],
                        persistedPath: String? = nil,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool,
                        variant: SupportedModelVariant = .int4) -> URL {
        let directoryName = "scratch/" + variant.defaultDirectoryName
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        if let path = environment["TURBO_FIELDFARE_MODEL_PATH"], !path.isEmpty {
            if path.hasPrefix("/") {
                return URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
            }
            return currentDirectoryURL.appendingPathComponent(path, isDirectory: true)
                .standardizedFileURL
        }
        if let persistedPath {
            return URL(fileURLWithPath: persistedPath, isDirectory: true).standardizedFileURL
        }
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return root.appendingPathComponent(directoryName, isDirectory: true)
                .standardizedFileURL
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return root.appendingPathComponent(directoryName, isDirectory: true)
                .standardizedFileURL
        }
        return applicationSupportURL
            .appendingPathComponent("TurboFieldfare", isDirectory: true)
            .appendingPathComponent(variant.defaultDirectoryName, isDirectory: true)
            .standardizedFileURL
    }

    static func canonicalURL(_ url: URL) -> URL {
        let url = url.standardizedFileURL
        // The final .gturbo child may not exist yet. Resolve its existing parent
        // first, since Foundation otherwise leaves parent aliases unresolved.
        return url.deletingLastPathComponent().resolvingSymlinksInPath()
            .appendingPathComponent(url.lastPathComponent, isDirectory: true)
            .resolvingSymlinksInPath()
    }

    /// `variant` is the one a download installs. An existing model may be any
    /// catalog variant: choosing one is how the app switches to it.
    static func validate(_ candidate: URL, intent: AppModel.ModelLocationIntent,
                         variant: SupportedModelVariant = .int4) throws {
        let candidate = canonicalURL(candidate)
        guard isVolumeAvailable(for: candidate) else {
            throw LocationError("The selected volume is unavailable. Reconnect it and retry, or choose another location.")
        }
        switch intent {
        case .existing:
            guard candidate.pathExtension == "gturbo" else {
                for name in SupportedModelVariant.allCases.map(\.defaultDirectoryName) {
                    let child = candidate.appendingPathComponent(name, isDirectory: true)
                    if AppModelInstallationProbe.status(anyVariantAt: child) == .complete {
                        throw LocationError("Choose the \(name) folder inside this folder. The vision pack is found beside it automatically.")
                    }
                }
                throw LocationError("Choose a completed text model folder ending in .gturbo (for example, gemma4.gturbo). Select the model folder itself, not its parent or the vision pack.")
            }
            switch AppModelInstallationProbe.status(anyVariantAt: candidate) {
            case .complete: break
            case .missing: throw LocationError("The selected directory has no completed manifest.json.")
            case .partial(let reason): throw LocationError("The selected installation is incompatible or incomplete: \(reason)")
            }
        case .download:
            let descriptor = AppModelInstallDescriptor.descriptor(for: variant)
            let parent = candidate.deletingLastPathComponent()
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: parent.path, isDirectory: &isDirectory), isDirectory.boolValue,
                  FileManager.default.isWritableFile(atPath: parent.path) else {
                throw LocationError("The download parent directory must exist and be writable.")
            }
            if FileManager.default.fileExists(atPath: candidate.path) {
                guard AppModelInstallationProbe.status(at: candidate, descriptor: descriptor) == .complete else {
                    throw LocationError("The destination already exists and is not a supported completed model. Choose another parent directory.")
                }
            }
            _ = try RemoteStreamingRepacker.inspectPersistentInstall(
                outputDirectory: candidate.path, repoID: descriptor.repoID,
                requestedRevision: descriptor.revision)
        }
    }

    static func isVolumeAvailable(for url: URL) -> Bool {
        let components = url.standardizedFileURL.pathComponents
        if components.count > 2, components[1] == "Volumes" {
            var isDirectory: ObjCBool = false
            let volume = "/Volumes/" + components[2]
            return FileManager.default.fileExists(atPath: volume, isDirectory: &isDirectory) && isDirectory.boolValue
        }
        return true
    }

    struct LocationError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            let appSources = candidate.appendingPathComponent(
                "Sources/TurboFieldfareApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
