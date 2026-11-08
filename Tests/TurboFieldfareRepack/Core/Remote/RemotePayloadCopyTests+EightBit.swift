import Foundation
import Synchronization
import Testing
import TurboFieldfareFormat

@testable import TurboFieldfareRepackCore

extension RemotePayloadCopyTests {
  @Test func eightBitInstallWritesAlignedResidentWeightsAndAnEightBitManifest() async throws {
    let snapshotDir = tmpDirForRemote("snap-8bit")
    let output = tmpPathForRemote("remote-8bit")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.build(
      at: snapshotDir, seed: 0x8B17_0000_0001, bits: 8)
    try serveSnapshot(snapshotDir, snapshot)

    let result = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: output, session: fakeHFSession())
    ).run()

    #expect(!result.dryRun)
    #expect(result.metadataEstimate == nil)
    #expect(result.summary.residentQuantizedWeightAlignment == 16)
    let bits = result.summary.quantBits
    #expect([bits.embedding, bits.attention, bits.router, bits.sharedExpert,
             bits.routedExpert] as [Int] == [8, 8, 8, 8, 8])

    // Manifest: every slot at 8 bits, read back through the v1 codec.
    let manifestData = try Data(contentsOf: URL(fileURLWithPath:
      (output as NSString).appendingPathComponent("manifest.json")))
    let manifest = try GTurboManifestCodec.decode(manifestData)
    let quant = try #require(manifest.quant)
    for slot in [quant.embedding, quant.attention, quant.router,
                 quant.sharedExpert, quant.routedExpert] {
      #expect(slot.weightBits == 8)
      #expect(slot.groupSize == 64)
    }
    #expect(manifest.expertStride == result.summary.expertStride)

    // Layout: 8-bit routed weights, logical columns four per packed word.
    let layoutData = try Data(contentsOf: URL(fileURLWithPath:
      (output as NSString).appendingPathComponent("packed_experts/layout.json")))
    #expect(UInt64(layoutData.count) == result.summary.layoutBytes)
    let layout = try GTurboPackedExpertsLayoutCodec.decode(layoutData)
    #expect(layout.expertStride == result.summary.expertStride)
    for layer in layout.layers {
      for expert in layer.experts {
        for role in ["gate", "up", "down"] {
          let weights = try #require(expert.tensors[role])
          #expect(weights.bits == 8)
          #expect(weights.dtype == "U32")
          #expect(weights.offset % 16 == 0)
          // [rows, cols] with one byte per column.
          #expect(UInt64(weights.shape[0]) * UInt64(weights.shape[1]) == weights.size)
        }
      }
    }

    // Resident file: aligned weights, zero padding, source bytes in place.
    let residentPath = (output as NSString).appendingPathComponent("model_weights.bin")
    let resident = try Data(contentsOf: URL(fileURLWithPath: residentPath))
    let shard = try Data(contentsOf: URL(fileURLWithPath: snapshot.shardPath))
    let sourceTensors = try Safetensors.parseHeader(path: snapshot.shardPath).tensors
    let sourceByName = Dictionary(uniqueKeysWithValues: sourceTensors.map { ($0.name, $0) })
    let entries = try resident.withUnsafeBytes { raw in
      let header = try GTurboResidentIndexCodec.decodeHeader(raw)
      return try GTurboResidentIndexCodec.decodeRegion(raw, header: header)
    }
    var cursor: UInt64 = 0
    var padded = 0
    for entry in entries.sorted(by: { $0.fileOffset < $1.fileOffset }) {
      if cursor == 0 { cursor = entry.fileOffset }
      if entry.dtype == GTurboFormatV1.DType.u32.rawValue {
        #expect(entry.fileOffset % 16 == 0, "\(entry.name)")
      }
      if entry.fileOffset > cursor {
        padded += 1
        #expect(resident[Int(cursor)..<Int(entry.fileOffset)].allSatisfy { $0 == 0 })
      }
      let source = try #require(sourceByName[entry.name])
      #expect(resident[Int(entry.fileOffset)..<Int(entry.fileOffset + entry.sizeBytes)]
              == shard[Int(source.absoluteOffset)..<Int(source.absoluteOffset + source.sizeBytes)])
      cursor = max(cursor, entry.fileOffset + entry.sizeBytes,
                   entry.scaleOffset + entry.scaleSize, entry.biasOffset + entry.biasSize)
    }
    #expect(padded > 0)

    // The verifier is variant-agnostic and accepts the 8-bit install.
    let verified = try VerifiedInstallTool.run(
      options: VerifyInstallOptions(inputGTurbo: output))
    // Every manifest entry plus the manifest itself.
    #expect(verified.fileCount == manifest.files.count + 1)
  }

  @Test func eightBitResumeMatchesAnUninterruptedInstall() async throws {
    let snapshotDir = tmpDirForRemote("snap-8bit-resume")
    let reference = tmpPathForRemote("remote-8bit-reference")
    let resumed = tmpPathForRemote("remote-8bit-resumed")
    defer { cleanUpRemote([snapshotDir, reference, resumed]) }
    let snapshot = try SyntheticSnapshot.build(
      at: snapshotDir, seed: 0x8B17_0000_0002, bits: 8)
    try serveSnapshot(snapshotDir, snapshot)

    _ = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: reference, session: fakeHFSession())
    ).run()

    let seen = Mutex<Int>(0)
    let task = Task {
      try await RemoteStreamingRepacker(
        options: remoteOptions(outputDir: resumed, session: fakeHFSession())
      ).run { progress in
        guard case .copyingPayload(_, let downloaded, _) = progress,
              downloaded > 0 else { return }
        let count = seen.withLock { $0 += 1; return $0 }
        if count == 6 {
          withUnsafeCurrentTask { $0?.cancel() }
        }
      }
    }
    await #expect(throws: CancellationError.self) {
      _ = try await task.value
    }
    let checkpoint = try RemoteInstallCheckpoint.load(from: resumed + ".resume.json")
    #expect(!checkpoint.completedRanges.isEmpty)
    #expect(checkpoint.sourceIndexSHA256 == (try IndexLoader.load(snapshotDir: snapshotDir))
      .indexSha256Hex)

    let result = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: resumed, session: fakeHFSession(), resume: true)
    ).run()
    #expect(result.reusedBytes > 0)
    #expect(result.downloadedThisRunBytes < result.remoteBytesToDownload)
    #expect(checkpoint.planFingerprint == result.summary.planFingerprint)

    // Same bytes as the uninterrupted install, padding included.
    let referenceFiles = try manifestFiles(reference)
    let resumedFiles = try manifestFiles(resumed)
    #expect(referenceFiles == resumedFiles)
    #expect(referenceFiles["model_weights.bin"] != nil)
    #expect(referenceFiles["packed_experts/layer_01.bin"] != nil)
  }

  @Test func eightBitDryRunPlansWithoutFetchingPayload() async throws {
    let snapshotDir = tmpDirForRemote("snap-8bit-dry")
    let output = tmpPathForRemote("remote-8bit-dry")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.build(
      at: snapshotDir, seed: 0x8B17_0000_0003, bits: 8)
    try serveSnapshot(snapshotDir, snapshot, optionalTokenizer: true)

    let result = try await RemoteStreamingRepacker(
      options: remoteOptions(outputDir: output, session: fakeHFSession()).asDryRun()
    ).run()

    #expect(result.dryRun)
    #expect(result.downloadedThisRunBytes == 0)
    #expect(result.remoteBytesToDownload > 0)
    let summary = result.summary
    #expect(summary.quantBits.routedExpert == 8)
    #expect(summary.quantBits.embedding == 8)
    #expect(summary.residentQuantizedWeightAlignment == 16)
    #expect(summary.layerFileCount == 2)
    #expect(summary.packedExpertBytes == 2 * summary.layerFileBytes)
    #expect(summary.layerFileBytes == 2 * summary.expertStride)
    #expect(summary.expertStride % GTurboFormatV1.alignmentBytes == 0)
    #expect(summary.expertPayloadBytes <= summary.expertStride)
    #expect(summary.layoutBytes > 0)
    // One vision norm plus two quantized vision weights with companions.
    #expect(summary.excludedMultimodalTensorCount == 7)

    let estimate = try #require(result.metadataEstimate)
    #expect(estimate.tokenizerFiles["tokenizer.json"] == UInt64(remoteTokenizerJSON.count))
    #expect(estimate.tokenizerFiles["tokenizer_config.json"]
            == UInt64(remoteTokenizerConfigJSON.count))
    #expect(estimate.tokenizerFiles["special_tokens_map.json"]
            == UInt64(remoteSpecialTokensMapJSON.count))
    #expect(estimate.tokenizerFiles["chat_template.jinja"] == UInt64(remoteChatTemplateJinja.count))
    #expect(estimate.tokenizerFiles["chat_template.json"] == nil)
    let config = try #require(FakeHFURLProtocol.files["config.json"])
    #expect(estimate.tokenizerFiles["config.json"] == UInt64(config.count))
    #expect(estimate.tokenizerDownloadBytes
            == estimate.tokenizerInstalledBytes - UInt64(config.count))
    #expect(estimate.sourceMetadataBytes > UInt64(config.count))

    // Only the 8-byte length prefix and the header were read from the shard,
    // and no tokenizer body was fetched.
    #expect(FakeHFURLProtocol.requestedRanges["model-00001-of-00001.safetensors"]?.count == 2)
    #expect(FakeHFURLProtocol.requestCounts["GET:tokenizer.json"] == nil)
    // Nothing is left behind but the lock file.
    #expect(!FileManager.default.fileExists(atPath: output))
    #expect(!FileManager.default.fileExists(atPath: output + ".partial"))
    #expect(!FileManager.default.fileExists(atPath: output + ".resume.json"))
  }

  @Test func eightBitSourceWithAnUnknownIndexIsRejected() async throws {
    let snapshotDir = tmpDirForRemote("snap-8bit-unknown")
    let output = tmpPathForRemote("remote-8bit-unknown")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.build(
      at: snapshotDir, seed: 0x8B17_0000_0004, bits: 8)
    try serveSnapshot(snapshotDir, snapshot)
    let pinned = SupportedModelVariant.int8
    let options = RemoteStreamingRepackOptions(
      repoID: pinned.repoID,
      revision: pinned.revision,
      outputDir: output,
      requireKnownSource: true,
      rangeChunkBytes: 4096,
      minFreeReserveBytes: 0,
      downloadSession: fakeHFSession(),
      baseURL: URL(string: "https://hf.test")!,
      retryBaseDelayNs: 0)

    do {
      _ = try await RemoteStreamingRepacker(options: options).run()
      Issue.record("an 8-bit source with an unpinned index was installed")
    } catch RepackError.sourceFingerprintRejected(_, let sha) {
      #expect(sha == (try IndexLoader.load(snapshotDir: snapshotDir)).indexSha256Hex)
    }
    #expect(FakeHFURLProtocol.requestedRanges["model-00001-of-00001.safetensors"] == nil)
    #expect(!FileManager.default.fileExists(atPath: output))
    #expect(!FileManager.default.fileExists(atPath: output + ".partial"))
    #expect(!FileManager.default.fileExists(atPath: output + ".resume.json"))
  }

  @Test func resumeUnderAnotherSourceNamesTheSavedOne() async throws {
    let snapshotDir = tmpDirForRemote("snap-8bit-wrong-resume")
    let output = tmpPathForRemote("remote-8bit-wrong-resume")
    defer { cleanUpRemote([snapshotDir, output]) }
    let snapshot = try SyntheticSnapshot.build(
      at: snapshotDir, seed: 0x8B17_0000_0005, bits: 8)
    try serveSnapshot(snapshotDir, snapshot)
    let paths = try RemoteInstallPaths(outputDirectory: output)
    try Posix.mkdirP(paths.partialDirectory)
    try RemoteInstallCheckpoint(
      repoID: "owner/model-8bit", requestedRevision: "main",
      resolvedCommit: FakeHFURLProtocol.commit,
      sourceIndexSHA256: String(repeating: "0", count: 64),
      planFingerprint: String(repeating: "0", count: 64),
      totalSourceBytes: 1
    ).write(to: paths.checkpointFile, parentDirectory: paths.parentDirectory)

    do {
      _ = try await RemoteStreamingRepacker(
        options: remoteOptions(outputDir: output, session: fakeHFSession(), resume: true)
      ).run()
      Issue.record("a download saved for another source was resumed")
    } catch RepackError.installStateIncompatible(let detail) {
      #expect(detail.contains("owner/model-8bit"))
    }
    // The saved download is left for the right variant.
    #expect(FileManager.default.fileExists(atPath: paths.checkpointFile))
    #expect(FileManager.default.fileExists(atPath: paths.partialDirectory))
  }

  @Test func discardForAVariantKeepsAnotherVariantsDownload() throws {
    let root = tmpDirForRemote("discard-variant")
    let output = (root as NSString).appendingPathComponent("model.gturbo")
    defer { cleanUpRemote([root]) }
    let paths = try RemoteInstallPaths(outputDirectory: output)
    func save(_ repoID: String) throws {
      try Posix.mkdirP(paths.partialDirectory)
      try RemoteInstallCheckpoint(
        repoID: repoID, requestedRevision: "main",
        resolvedCommit: String(repeating: "c", count: 40),
        sourceIndexSHA256: String(repeating: "0", count: 64),
        planFingerprint: String(repeating: "0", count: 64),
        totalSourceBytes: 1
      ).write(to: paths.checkpointFile, parentDirectory: paths.parentDirectory)
    }

    try save(SupportedModelVariant.int8.repoID)
    #expect(throws: RepackError.self) {
      try RemoteStreamingRepacker.discardPartial(
        outputDirectory: output, expectedRepoID: SupportedModelVariant.int4.repoID)
    }
    #expect(FileManager.default.fileExists(atPath: paths.checkpointFile))
    #expect(FileManager.default.fileExists(atPath: paths.partialDirectory))

    try RemoteStreamingRepacker.discardPartial(
      outputDirectory: output, expectedRepoID: SupportedModelVariant.int8.repoID)
    #expect(!FileManager.default.fileExists(atPath: paths.checkpointFile))
    #expect(!FileManager.default.fileExists(atPath: paths.partialDirectory))

    // Without an expected source the historical discard removes anything,
    // and an unreadable checkpoint never blocks recovery.
    try save(SupportedModelVariant.int8.repoID)
    try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
    #expect(!FileManager.default.fileExists(atPath: paths.checkpointFile))
    try Posix.mkdirP(paths.partialDirectory)
    try Data("not json".utf8).write(to: URL(fileURLWithPath: paths.checkpointFile))
    try RemoteStreamingRepacker.discardPartial(
      outputDirectory: output, expectedRepoID: SupportedModelVariant.int4.repoID)
    #expect(!FileManager.default.fileExists(atPath: paths.checkpointFile))
  }

  // MARK: - Helpers

  private func serveSnapshot(_ snapshotDir: String,
                             _ snapshot: SyntheticSnapshot.Snapshot,
                             optionalTokenizer: Bool = false) throws {
    resetFakeHF()
    FakeHFURLProtocol.files = try remoteFiles(
      snapshotDir: snapshotDir,
      snap: snapshot,
      includeRequiredTokenizer: true,
      includeOptionalTokenizer: optionalTokenizer)
  }

  /// Size and SHA-256 of every payload file the manifest lists.
  private func manifestFiles(_ output: String) throws -> [String: String] {
    let data = try Data(contentsOf: URL(fileURLWithPath:
      (output as NSString).appendingPathComponent("manifest.json")))
    let manifest = try GTurboManifestCodec.decode(data)
    var files: [String: String] = [:]
    for (path, entry) in manifest.files where !path.hasPrefix("tokenizer/") {
      files[path] = "\(entry.size):\(entry.sha256)"
    }
    return files
  }
}
