import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite(.serialized)
struct RepackCLITests {
    @Test func helpNamesBothVariantsAndTheDryRun() throws {
        let result = try run(["--help"])

        #expect(result.status == 0)
        #expect(result.stdout.contains("--variant int4|int8"))
        #expect(result.stdout.contains("--dry-run"))
        for variant in SupportedModelVariant.allCases {
            #expect(result.stdout.contains(variant.displayName))
        }
    }

    @Test func unknownVariantIsRejected() throws {
        let output = temporaryOutput("unknown-variant")
        defer { clean(output) }
        let result = try run(["--output", output, "--variant", "int6"])

        #expect(result.status == 2)
        #expect(result.stderr.contains("unknown variant int6"))
    }

    /// S6 lifted the S4 refusal: `--variant qwen36` now runs the real repack
    /// path. Pinning the CLI here without touching the network: a resume
    /// without saved state fails before any request, like the 8-bit probe.
    @Test func qwen36ResumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("qwen36-missing-resume")
        defer { clean(output) }
        let result = try run(["--output", output, "--resume", "--variant", "qwen36"])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
        #expect(!FileManager.default.fileExists(atPath: output))
    }

    /// The help lists the new pin and still marks only 4-bit as the default.
    @Test func helpListsQwen36AndKeepsFourBitAsTheDefault() throws {
        let result = try run(["--help"])
        #expect(result.status == 0)
        #expect(result.stdout.contains("--variant int4|int8|qwen36"))
        let lines = result.stdout.split(separator: "\n")
        let int4Line = try #require(lines.first { $0.hasPrefix("  int4 ") })
        #expect(int4Line.contains(", default)"))
        let qwenLine = try #require(lines.first { $0.hasPrefix("  qwen36 ") })
        #expect(qwenLine.contains("Qwen3.6 35B-A3B 4-bit"))
        #expect(!qwenLine.contains("default"))
    }

    @Test func conflictingVariantsAreRejected() throws {
        let output = temporaryOutput("variant-twice")
        defer { clean(output) }
        let result = try run(["--output", output, "--variant", "int4", "--variant", "int8"])

        #expect(result.status == 2)
        #expect(result.stderr.contains("--variant was given twice"))
    }

    @Test func dryRunDoesNotResume() throws {
        let output = temporaryOutput("dry-run-resume")
        defer { clean(output) }
        let result = try run(["--dry-run", "--output", output, "--resume"])

        #expect(result.status == 2)
        #expect(result.stderr.contains("mutually exclusive"))
    }

    @Test func discardHasNoDryRun() throws {
        let output = temporaryOutput("dry-run-discard")
        defer { clean(output) }
        let result = try run(["--discard-partial", "--output", output, "--dry-run"])

        #expect(result.status == 2)
        #expect(result.stderr.contains("--discard-partial only accepts"))
    }

    /// Verification reads the variant from the manifest it checks.
    @Test func verificationTakesNoVariantOrDryRun() throws {
        for extra in [["--variant", "int8"], ["--dry-run"]] {
            let result = try run(["--verify-install", "--input-gturbo", "/tmp/missing.gturbo"]
                                 + extra)
            #expect(result.status == 2, "\(extra) was accepted")
            #expect(result.stderr.contains("verification accepts only --input-gturbo"))
        }
    }

    /// The image pack follows its text model; vision commands are unchanged.
    @Test func visionCommandsTakeNoVariantOrDryRun() throws {
        let output = temporaryOutput("vision-variant")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        for extra in [["--variant", "int8"], ["--dry-run"]] {
            let result = try run(["--vision-output", output,
                                  "--text-model", "/tmp/missing-text.gturbo"] + extra)
            #expect(result.status == 2, "\(extra) was accepted")
            #expect(result.stderr.contains("do not accept text install arguments"))
        }
    }

    @Test func eightBitResumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("missing-resume-8bit")
        defer { clean(output) }
        let result = try run(["--output", output, "--resume", "--variant", "int8"])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func discardForOneVariantKeepsTheOthersDownload() throws {
        let output = temporaryOutput("discard-variant")
        defer { clean(output) }
        try saveDownload(of: .int4, at: output)

        let refused = try run(["--discard-partial", "--output", output, "--variant", "int8"])
        #expect(refused.status == 1)
        #expect(refused.stderr.contains("pass --variant int4"))
        #expect(FileManager.default.fileExists(atPath: output + ".resume.json"))
        #expect(FileManager.default.fileExists(atPath: output + ".partial"))

        let discarded = try run(["--discard-partial", "--output", output, "--variant", "int4"])
        #expect(discarded.status == 0)
        #expect(discarded.stdout.contains("Discarded saved 4-bit download"))
        #expect(!FileManager.default.fileExists(atPath: output + ".resume.json"))
        #expect(!FileManager.default.fileExists(atPath: output + ".partial"))
    }

    /// A resume under the wrong variant fails on the saved checkpoint, before
    /// any request, and says which variant the download belongs to.
    @Test func resumeUnderTheWrongVariantNamesTheSavedOne() throws {
        let output = temporaryOutput("resume-wrong-variant")
        defer { clean(output) }
        try saveDownload(of: .int8, at: output)

        let result = try run(["--output", output, "--resume"])
        #expect(result.status == 1)
        #expect(result.stderr.contains(SupportedModelVariant.int8.repoID))
        #expect(result.stderr.contains("pass --variant int8"))
        #expect(FileManager.default.fileExists(atPath: output + ".resume.json"))
    }

    private func saveDownload(of variant: SupportedModelVariant, at output: String) throws {
        let paths = try RemoteInstallPaths(outputDirectory: output)
        try FileManager.default.createDirectory(atPath: paths.partialDirectory,
                                                withIntermediateDirectories: true)
        try RemoteInstallCheckpoint(
            repoID: variant.repoID, requestedRevision: variant.revision,
            resolvedCommit: variant.revision,
            sourceIndexSHA256: variant.sourceIndexSHA256,
            planFingerprint: String(repeating: "0", count: 64),
            totalSourceBytes: 1
        ).write(to: paths.checkpointFile, parentDirectory: paths.parentDirectory)
    }

    @Test func resumeAndDiscardAreMutuallyExclusive() throws {
        let output = temporaryOutput("exclusive")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--resume",
            "--discard-partial",
        ])

        #expect(result.status == 2)
        #expect(result.stderr.contains("mutually exclusive"))
    }

    @Test func resumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("missing-resume")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--resume",
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func discardWithoutStateReportsAnError() throws {
        let output = temporaryOutput("missing-discard")
        defer { clean(output) }
        let result = try run([
            "--discard-partial",
            "--output", output,
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    private func run(_ arguments: [String]) throws
        -> (status: Int32, stdout: String, stderr: String) {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/TurboFieldfareRepack")
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = stderr.fileHandleForReading.readDataToEndOfFile()
        return (
            process.terminationStatus,
            String(decoding: out, as: UTF8.self),
            String(decoding: err, as: UTF8.self))
    }

    private func temporaryOutput(_ tag: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-cli-\(tag)-\(UUID().uuidString).gturbo")
    }

    private func clean(_ output: String) {
        for path in [
            output,
            output + ".partial",
            output + ".install-state",
            output + ".install-state.cleanup",
            output + ".install.lock",
            output + ".resume.json",
        ] {
            try? FileManager.default.removeItem(atPath: path)
        }
    }
    @Test func visionActivationWithoutPreparedStateReportsAnError() throws {
        let output = temporaryOutput("vision-missing-activation")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        let result = try run([
            "--activate-vision-install",
            "--vision-output", output,
            "--text-model", "/tmp/missing-text.gturbo",
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func visionDiscardWithoutStateReportsAnError() throws {
        let output = temporaryOutput("vision-missing-discard")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        let result = try run([
            "--discard-partial",
            "--vision-output", output,
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func visionRemovalWithoutInstalledPackReportsAnError() throws {
        let output = temporaryOutput("vision-missing-remove")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        let result = try run([
            "--remove-vision-install",
            "--vision-output", output,
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    /// Discard is handled before the other vision modes, so accepting the
    /// combination silently performed the discard and exited 0 without ever
    /// removing, verifying or activating anything.
    @Test func visionModesAreMutuallyExclusive() throws {
        let output = temporaryOutput("vision-mode-clash")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        for mode in ["--remove-vision-install", "--verify-vision-install",
                     "--activate-vision-install"] {
            let result = try run(["--discard-partial", mode, "--vision-output", output])
            #expect(result.status == 2, "\(mode) with --discard-partial was accepted")
            #expect(result.stderr.contains("mutually exclusive"))
        }
    }

    /// Verify and activate read an installed or prepared pack; a transfer flag
    /// is a request they cannot honour, not a no-op.
    @Test func verifyAndActivateRejectTransferFlags() throws {
        let output = temporaryOutput("vision-flag-clash")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        for mode in ["--verify-vision-install", "--activate-vision-install"] {
            for flag in ["--overwrite", "--resume"] {
                let result = try run([mode, "--vision-output", output,
                                      "--text-model", "scratch/gemma4.gturbo", flag])
                #expect(result.status == 2, "\(mode) accepted \(flag)")
            }
        }
    }

    @Test func remoteVisionResumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("vision-missing-resume")
            .replacingOccurrences(of: ".gturbo", with: ".vision.gturbo")
        defer { clean(output) }
        let result = try run([
            "--vision-output", output,
            "--text-model", "/tmp/missing-text.gturbo",
            "--resume",
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

}
