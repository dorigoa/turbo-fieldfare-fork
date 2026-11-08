import Foundation
import TurboFieldfareRepackCore

private let variantList = SupportedModelVariant.allCases.map(\.rawValue).joined(separator: "|")

private let variantLines = SupportedModelVariant.allCases.map { variant in
    let gigabytes = Double(variant.approximateDownloadBytes) / 1_000_000_000
    let memory = variant.recommendedMinimumHostMemoryBytes >> 30
    let tag = variant == .int4 ? ", default" : ""
    return String(format: "  %@  %@ (about %.1f GB download, %llu GB Mac or larger%@)",
                  variant.rawValue, variant.displayName, gigabytes, memory, tag)
}.joined(separator: "\n")

private let usage = """
Usage:
  TurboFieldfareRepack --output <model.gturbo> [--variant \(variantList)]
                       [--overwrite] [--resume]
  TurboFieldfareRepack --dry-run --output <model.gturbo>
                       [--variant \(variantList)] [--overwrite]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
                       [--variant \(variantList)]
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --verify-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TurboFieldfareRepack --activate-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TurboFieldfareRepack --remove-vision-install
                       --vision-output <model.vision.gturbo>
  TurboFieldfareRepack --discard-partial
                       --vision-output <model.vision.gturbo>
  TurboFieldfareRepack --help

The installer streams a supported Gemma 4 checkpoint from Hugging Face and
repackages it without materializing the source checkpoint on disk. Set HF_TOKEN
only if Hugging Face requests authentication. A cancelled or interrupted
download can be continued with --resume, which needs the --variant the
download was started with, or removed with --discard-partial. Given a
--variant, a discard removes only a download of that variant.

--variant selects the pinned checkpoint:
\(variantLines)
Install each variant in its own directory, for example scratch/gemma4.gturbo
and scratch/gemma4-8bit.gturbo.

--dry-run reads only the source metadata and shard headers, then prints the
plan: download and output sizes, bits per quantization slot and expert stride.
It downloads no payload and leaves nothing behind but the install lock file.

--verify-install reads the variant from the installed manifest and takes no
--variant.

The optional image companion pack installs beside an existing text model and
is bound to it. Without the pack the text runtime is unchanged; image input is
simply unavailable.
"""

private struct Arguments {
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var dryRun = false
    /// nil when `--variant` was not given; installs then use `.int4`.
    var variant: SupportedModelVariant?
    var inputGTurbo: String?
    var visionOutput: String?
    var textModel: String?
    var verifyVisionInstall = false
    var activateVisionInstall = false
    var removeVisionInstall = false

    var installVariant: SupportedModelVariant { variant ?? .int4 }

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--dry-run":
                parsed.dryRun = true
                index += 1
            case "--verify-vision-install":
                parsed.verifyVisionInstall = true
                index += 1
            case "--activate-vision-install":
                parsed.activateVisionInstall = true
                index += 1
            case "--remove-vision-install":
                parsed.removeVisionInstall = true
                index += 1
            case "--variant":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                let value = values[index + 1]
                guard let variant = SupportedModelVariant(rawValue: value) else {
                    throw ParseError.invalidMode(
                        "unknown variant \(value); expected one of \(variantList)")
                }
                guard parsed.variant == nil || parsed.variant == variant else {
                    throw ParseError.invalidMode("--variant was given twice")
                }
                parsed.variant = variant
                index += 2
            case "--vision-output":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.visionOutput = values[index + 1]
                index += 2
            case "--text-model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.textModel = values[index + 1]
                index += 2
            case "--output", "--input-gturbo":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else {
                    parsed.inputGTurbo = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        let visionModes = [parsed.verifyVisionInstall,
                           parsed.activateVisionInstall,
                           parsed.removeVisionInstall].filter { $0 }.count
        guard visionModes <= 1 else {
            throw ParseError.invalidMode("vision install modes are mutually exclusive")
        }
        if visionModes == 1 || parsed.visionOutput != nil {
            guard parsed.visionOutput != nil else {
                throw ParseError.missingRequired("--vision-output")
            }
            // The image pack follows its text model, not a --variant, and has
            // no dry run.
            guard parsed.output == nil, parsed.inputGTurbo == nil,
                  !parsed.verifyInstall, parsed.variant == nil, !parsed.dryRun else {
                throw ParseError.invalidMode(
                    "vision install operations do not accept text install arguments")
            }
            // Discard runs first below, so accepting it alongside another mode
            // would silently perform the discard and exit 0 without ever doing
            // what was asked.
            guard !(parsed.discardPartial && visionModes == 1) else {
                throw ParseError.invalidMode(
                    "--discard-partial is mutually exclusive with the other vision "
                        + "install operations")
            }
            if parsed.removeVisionInstall || parsed.discardPartial {
                guard parsed.textModel == nil, !parsed.overwrite, !parsed.resume else {
                    throw ParseError.invalidMode(
                        "this vision operation only accepts --vision-output")
                }
            } else if parsed.verifyVisionInstall || parsed.activateVisionInstall {
                guard parsed.textModel != nil else {
                    throw ParseError.missingRequired("--text-model")
                }
                // Neither reads a download, so a transfer flag here is a
                // request this mode cannot honour rather than a no-op.
                guard !parsed.overwrite, !parsed.resume else {
                    throw ParseError.invalidMode(
                        "this vision operation only accepts --vision-output and "
                            + "--text-model")
                }
            } else {
                guard parsed.textModel != nil else {
                    throw ParseError.missingRequired("--text-model")
                }
            }
            return parsed
        }
        guard parsed.textModel == nil else {
            throw ParseError.invalidMode("--text-model requires --vision-output")
        }
        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall,
                  !parsed.dryRun else {
                throw ParseError.invalidMode(
                    "--discard-partial only accepts --output and --variant")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume,
                  parsed.variant == nil, !parsed.dryRun else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
            // A dry run must not touch a saved download, and resuming one
            // rewrites its checkpoint.
            guard !(parsed.dryRun && parsed.resume) else {
                throw ParseError.invalidMode("--dry-run and --resume are mutually exclusive")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func runVisionInstall(_ arguments: Arguments) async -> Int32? {
    guard let visionOutput = arguments.visionOutput else { return nil }

    if arguments.discardPartial {
        do {
            try RemoteVisionPackInstaller.discardPartial(outputDirectory: visionOutput)
            print("Discarded saved image-pack download for \(visionOutput)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.removeVisionInstall {
        do {
            try RemoteVisionPackInstaller.removeInstalled(outputDirectory: visionOutput)
            print("Removed image pack \(visionOutput)")
            return 0
        } catch {
            printError("remove-vision-install failed: \(error)")
            return 1
        }
    }

    guard let textModel = arguments.textModel else { return 2 }

    if arguments.verifyVisionInstall {
        do {
            let verification = try VisionPackVerifier.verify(
                directory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                installedDirectory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                textModelDirectory: URL(fileURLWithPath: textModel, isDirectory: true),
                verifyWeights: true)
            print("Verified image pack \(visionOutput)")
            print("Bound to text model \(textModel)")
            print("Text manifest sha256 \(verification.compatibleTextManifestSha256)")
            return 0
        } catch {
            printError("verify-vision-install failed: \(error)")
            return 1
        }
    }

    if arguments.activateVisionInstall {
        do {
            try RemoteVisionPackInstaller.activatePrepared(
                outputDirectory: visionOutput,
                textModelDirectory: textModel,
                repoID: SupportedModelSource.repoID,
                requestedRevision: SupportedModelSource.revision)
            print("Activated image pack \(visionOutput)")
            return 0
        } catch {
            printError("activate-vision-install failed: \(error)")
            return 1
        }
    }

    let options = RemoteVisionPackInstallOptions(
        repoID: SupportedModelSource.repoID,
        revision: SupportedModelSource.revision,
        textModelDirectory: textModel,
        outputDirectory: visionOutput,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        overwrite: arguments.overwrite,
        resume: arguments.resume)
    do {
        let progress = InstallProgressReporter()
        try await RemoteVisionPackInstaller(options: options).run(
            progress: { progress($0) })
        print("Installed image pack \(visionOutput)")
        print("Text model: \(textModel)")
        return 0
    } catch {
        printError("vision install failed: \(error)")
        return 1
    }
}

/// Names the variant a saved download belongs to when a resume or discard was
/// asked for under the wrong `--variant`.
private func savedDownloadHint(output: String,
                               requested: SupportedModelVariant) -> String? {
    guard let paths = try? RemoteInstallPaths(outputDirectory: output),
          let saved = try? RemoteInstallCheckpoint.load(from: paths.checkpointFile),
          let savedVariant = SupportedModelVariant.variant(forRepoID: saved.repoID),
          savedVariant != requested else {
        return nil
    }
    return "the saved download is \(savedVariant.displayName); "
        + "pass --variant \(savedVariant.rawValue)"
}

private func printDryRun(_ result: RemoteStreamingRepackResult,
                         variant: SupportedModelVariant) {
    let summary = result.summary
    let bits = summary.quantBits
    print("Dry run: \(variant.displayName) (no payload downloaded)")
    print("Source: \(variant.repoID) at \(result.resolvedCommit)")
    print("Source index sha256: \(variant.sourceIndexSHA256)")
    print("Quantization bits: embedding \(bits.embedding), attention \(bits.attention), "
        + "router \(bits.router), sharedExpert \(bits.sharedExpert), "
        + "routedExpert \(bits.routedExpert)")
    print("Resident quantized weight alignment: "
        + "\(summary.residentQuantizedWeightAlignment) byte(s)")
    print("Payload to download: \(result.remoteBytesToDownload) bytes in "
        + "\(result.rangeRequestCount) range requests "
        + "(\(result.remoteGapBytesDownloaded) gap bytes)")
    var estimatedDownload = result.remoteBytesToDownload
    if let estimate = result.metadataEstimate {
        estimatedDownload += estimate.sourceMetadataBytes + estimate.tokenizerDownloadBytes
        print("Source metadata: \(estimate.sourceMetadataBytes) bytes "
            + "(index, config, safetensors headers)")
        let names = estimate.tokenizerFiles.keys.sorted().joined(separator: ", ")
        print("Tokenizer files: \(estimate.tokenizerInstalledBytes) bytes installed, "
            + "\(estimate.tokenizerDownloadBytes) bytes to download (\(names))")
        print("Estimated download: \(estimatedDownload) bytes")
    }
    print("Resident file: model_weights.bin \(summary.residentFileBytes) bytes")
    print("Expert files: \(summary.layerFileCount) x \(summary.layerFileBytes) bytes "
        + "= \(summary.packedExpertBytes) bytes")
    print("Expert payload: \(summary.expertPayloadBytes) bytes, "
        + "expertStride: \(summary.expertStride) bytes")
    print("Layout file: packed_experts/layout.json \(summary.layoutBytes) bytes")
    print("Model output: \(summary.payloadOutputBytes) bytes (resident + expert files)")
    if let estimate = result.metadataEstimate {
        let installed = summary.payloadOutputBytes + summary.layoutBytes
            + estimate.tokenizerInstalledBytes
        print("Estimated installed size: \(installed) bytes "
            + "plus manifest.json and verified-install.json (a few KB)")
    }
    print("Excluded multimodal tensors: \(summary.excludedMultimodalTensorCount)")
    print("Resident index sha256: \(summary.residentIndexSHA256)")
    print("Plan fingerprint: \(summary.planFingerprint)")
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if let code = await runVisionInstall(arguments) {
        return code
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(
                outputDirectory: output,
                expectedRepoID: arguments.variant?.repoID)
            if let variant = arguments.variant {
                print("Discarded saved \(variant.shortLabel) download for \(output)")
            } else {
                print("Discarded saved download for \(output)")
            }
            return 0
        } catch {
            printError("discard failed: \(error)")
            if let variant = arguments.variant,
               let hint = savedDownloadHint(output: output, requested: variant) {
                printError(hint)
            }
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    let variant = arguments.installVariant
    let installOptions = variant.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume)
    let options = arguments.dryRun ? installOptions.asDryRun() : installOptions
    do {
        let progress = InstallProgressReporter()
        let result = try await RemoteStreamingRepacker(options: options).run(
            progress: { progress($0) })
        if result.dryRun {
            printDryRun(result, variant: variant)
            return 0
        }
        print("Installed \(variant.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("\(arguments.dryRun ? "dry run" : "install") failed: \(error)")
        if arguments.resume,
           let hint = savedDownloadHint(output: output, requested: variant) {
            printError(hint)
        }
        return 1
    }
}

exit(await run(CommandLine.arguments))
