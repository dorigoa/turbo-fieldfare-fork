import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// The 8-bit plan and the guarantee that the 4-bit plan did not move.
@Suite struct RepackPlannerProfileTests {
    private struct Planned {
        let meta: IndexLoader.SourceMetadata
        let plan: RepackPlan
        let rangePlan: RangeCopyPlan
    }

    private static let goldenSeed: UInt64 = 0x4B17_0000_0000_0004

    private func planned(bits: Int, seed: UInt64 = goldenSeed) throws -> Planned {
        let snapshot = temporaryDirectory("snapshot")
        let output = temporaryDirectory("output")
        defer {
            try? FileManager.default.removeItem(atPath: snapshot)
            try? FileManager.default.removeItem(atPath: output)
        }
        let built = try SyntheticSnapshot.build(at: snapshot, seed: seed, bits: bits)
        let meta = try IndexLoader.load(snapshotDir: snapshot)
        let arch = try ArchInfo.load(
            configPath: (snapshot as NSString).appendingPathComponent("config.json"))
        // Parsed under the bare shard name, as the remote installer does, so
        // the copy-plan fingerprint does not depend on the temporary path.
        let shard = try Data(contentsOf: URL(fileURLWithPath: built.shardPath))
        let headerBytes = shard.prefix(8).withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let header = try Safetensors.parseHeaderBytes(
            path: (built.shardPath as NSString).lastPathComponent,
            fileSize: UInt64(shard.count),
            headerBytes: shard.subdata(in: 8..<(8 + Int(headerBytes))))
        let plan = try RepackPlanner.plan(meta: meta, arch: arch,
                                          shardHeaders: [header], outputDir: output)
        let rangePlan = try RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: 4096)
        return Planned(meta: meta, plan: plan, rangePlan: rangePlan)
    }

    // MARK: - 4-bit stays byte-identical

    /// Values captured from this fixture on `main` before the 8-bit work
    /// (11e505f). Any change to the 4-bit resident layout, the expert blobs,
    /// the layout file or the copy plan moves at least one of them.
    @Test func fourBitPlanMatchesThePreEightBitPlanner() throws {
        let planned = try planned(bits: 4)
        #expect(planned.meta.indexSha256Hex
                == "151db8b3aeb4e47fe28d4641886eee0d9cc8b844a4d8c45f5d54d05117bcf49a")
        #expect(planned.rangePlan.canonicalFingerprint
                == "3c9aa53735878e50b166eacc2a0d2748eabfd5f9666a1f4dacc003c305cd4fa2")
        #expect(planned.rangePlan.coalescedCopies.count == 86)
        #expect(planned.rangePlan.residentIndexSha256
                == "d9765a92bc632e878ae6202047184a777a8bcbf05baf2273a4725d8ccc924a70")
        let layout = try GTurboJSON.encodeLayout(
            plan: planned.plan, expertStride: planned.plan.layers[0].expertStride)
        #expect(sha256(layout)
                == "f44d81590cbc88fac5444687d975ba0182792a0357e1bde3dfd344a166fac49b")
        #expect(planned.plan.resident.totalSize == 261_292)
        #expect(planned.plan.layers[0].expertStride == 16_384)
        #expect(planned.rangePlan.remoteBytesToDownload == 300_204)
        #expect(planned.plan.residentQuantizedWeightAlignment == 1)
    }

    /// The 4-bit resident payload stays tightly packed even though its router
    /// and shared expert are 8-bit: no entry starts after a gap.
    @Test func fourBitResidentPayloadHasNoGaps() throws {
        let resident = try planned(bits: 4).plan.resident
        var cursor = resident.indexSize
        for entry in resident.entries {
            #expect(entry.fileOffset == cursor, "\(entry.name)")
            cursor = entry.fileOffset + entry.sizeBytes + entry.scaleSize + entry.biasSize
        }
        #expect(cursor == resident.totalSize)
    }

    // MARK: - 8-bit plan

    @Test func eightBitResidentWeightsStartOnSixteenBytes() throws {
        let plan = try planned(bits: 8).plan
        #expect(plan.residentQuantizedWeightAlignment == 16)
        var cursor = plan.resident.indexSize
        var paddedEntries = 0
        for entry in plan.resident.entries {
            #expect(entry.fileOffset >= cursor, "\(entry.name) overlaps its predecessor")
            if entry.quantSpec != nil {
                #expect(entry.fileOffset % 16 == 0, "\(entry.name)")
                #expect(entry.fileOffset - cursor < 16, "\(entry.name) is padded too far")
                // Scales and biases follow their weight without a gap.
                #expect(entry.scaleOffset == entry.fileOffset + entry.sizeBytes)
                #expect(entry.biasOffset == entry.scaleOffset + entry.scaleSize)
                if entry.fileOffset > cursor { paddedEntries += 1 }
            } else {
                #expect(entry.fileOffset == cursor, "\(entry.name) is unquantized")
            }
            cursor = entry.fileOffset + entry.sizeBytes + entry.scaleSize + entry.biasSize
        }
        #expect(cursor == plan.resident.totalSize)
        // The fixture's 2-byte layer_scalar and 8-byte router companions put
        // following weights off a 16-byte boundary, so the rule is exercised.
        #expect(paddedEntries > 0)
    }

    @Test func eightBitShapesAreFourValuesPerPackedWord() throws {
        let plan = try planned(bits: 8).plan
        for entry in plan.resident.entries where entry.quantSpec != nil {
            #expect(entry.quantSpec?.bits == 8)
            let packed = entry.sourceWeight.shape.last!
            let logicalColumns = entry.logicalShape4[entry.sourceWeight.shape.count - 1]
            #expect(UInt64(logicalColumns) == packed * 4, "\(entry.name)")
        }
        let routed = plan.layers.filter { $0.expertsPerLayer > 0 }
        #expect(routed.count == 2)
        for layer in routed {
            var blobEnd: UInt64 = 0
            for slice in layer.subTensors {
                if slice.component == "weights" {
                    #expect(slice.bitsForWeights == 8)
                    #expect(slice.logicalShape.last! == slice.sourceTensor.shape.last! * 4)
                    #expect(slice.offsetInExpertBlob % 16 == 0)
                }
                blobEnd = slice.offsetInExpertBlob + slice.sizeInExpertBlob
            }
            let page = GTurboFormatV1.alignmentBytes
            #expect(layer.expertStride == (blobEnd + page - 1) / page * page)
        }
    }

    @Test func alignmentNeedsEveryLanguageModelWeightAtEightBits() throws {
        #expect(try planned(bits: 4).plan.residentQuantizedWeightAlignment == 1)
        #expect(try planned(bits: 8).plan.residentQuantizedWeightAlignment == 16)
    }

    @Test func eightBitPlanHasItsOwnFingerprint() throws {
        let four = try planned(bits: 4)
        let eight = try planned(bits: 8)
        // The fixture's index lists the same names for both profiles, so only
        // the plan, not the index, tells them apart here.
        #expect(four.rangePlan.canonicalFingerprint != eight.rangePlan.canonicalFingerprint)
        #expect(four.rangePlan.residentIndexSha256 != eight.rangePlan.residentIndexSha256)
        // Excluded multimodal tensors do not depend on the profile.
        #expect(four.plan.excludedMultimodalTensorNames
                == eight.plan.excludedMultimodalTensorNames)
    }

    // MARK: - Manifest bits

    @Test func manifestBitsComeFromThePlannedTensors() throws {
        let four = try planned(bits: 4).plan.manifestQuantBitWidths()
        #expect([four.embedding, four.attention, four.router, four.sharedExpert,
                 four.routedExpert] as [Int] == [4, 4, 8, 8, 4])
        let eight = try planned(bits: 8).plan.manifestQuantBitWidths()
        #expect([eight.embedding, eight.attention, eight.router, eight.sharedExpert,
                 eight.routedExpert] as [Int] == [8, 8, 8, 8, 8])
    }

    @Test func manifestBitsRejectAPlanWithoutRoutedExperts() throws {
        let plan = try planned(bits: 8).plan
        let empty = plan.layers.map {
            LayerFilePlan(layerIndex: $0.layerIndex, path: $0.path,
                          expertsPerLayer: 0, expertStride: 0, subTensors: [])
        }
        #expect(throws: RepackError.self) {
            _ = try replacing(plan, layers: empty).manifestQuantBitWidths()
        }
    }

    @Test func manifestBitsRejectAPlanWithoutARouter() throws {
        let plan = try planned(bits: 8).plan
        let entries = plan.resident.entries.filter { !$0.name.hasSuffix(".router.proj.weight") }
        #expect(throws: RepackError.self) {
            _ = try replacing(plan, entries: entries).manifestQuantBitWidths()
        }
    }

    @Test func manifestBitsRejectAMixedSlot() throws {
        let plan = try planned(bits: 8).plan
        var flipped = false
        let entries = plan.resident.entries.map { entry -> ResidentEntry in
            guard !flipped, entry.name.hasSuffix(".self_attn.o_proj.weight") else { return entry }
            flipped = true
            return entry.with(quantSpec: QuantSpec(bits: 4))
        }
        #expect(flipped)
        #expect(throws: RepackError.self) {
            _ = try replacing(plan, entries: entries).manifestQuantBitWidths()
        }
    }

    @Test func manifestBitsRejectAQuantizedTensorWithoutASlot() throws {
        let plan = try planned(bits: 8).plan
        let template = plan.resident.entries.first { $0.quantSpec != nil }!
        let stray = ResidentEntry(
            name: "language_model.model.per_layer_projection.weight",
            dtype: template.dtype, logicalShape4: template.logicalShape4,
            fileOffset: template.fileOffset, sizeBytes: template.sizeBytes,
            scaleOffset: template.scaleOffset, scaleSize: template.scaleSize,
            biasOffset: template.biasOffset, biasSize: template.biasSize,
            quantSpec: QuantSpec(bits: 8),
            sourceWeight: template.sourceWeight,
            sourceScales: template.sourceScales,
            sourceBiases: template.sourceBiases)
        #expect(throws: RepackError.self) {
            _ = try replacing(plan, entries: plan.resident.entries + [stray])
                .manifestQuantBitWidths()
        }
    }

    // MARK: - Helpers

    private func replacing(_ plan: RepackPlan,
                           entries: [ResidentEntry]? = nil,
                           layers: [LayerFilePlan]? = nil) -> RepackPlan {
        let resident = ResidentFilePlan(path: plan.resident.path,
                                        entries: entries ?? plan.resident.entries,
                                        stringTable: plan.resident.stringTable,
                                        stringTableOffsets: plan.resident.stringTableOffsets,
                                        indexSize: plan.resident.indexSize,
                                        residentSize: plan.resident.residentSize)
        return RepackPlan(arch: plan.arch, baseMode: plan.baseMode,
                          baseGroupSize: plan.baseGroupSize,
                          bitsOverrideCount: plan.bitsOverrideCount,
                          resident: resident, layers: layers ?? plan.layers,
                          matchedModelID: plan.matchedModelID,
                          excludedMultimodalTensorNames: plan.excludedMultimodalTensorNames,
                          residentQuantizedWeightAlignment: plan.residentQuantizedWeightAlignment)
    }

    private func sha256(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private func temporaryDirectory(_ tag: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-profile-\(tag)-\(UUID().uuidString)")
    }
}

private extension ResidentEntry {
    func with(quantSpec: QuantSpec?) -> ResidentEntry {
        ResidentEntry(name: name, dtype: dtype, logicalShape4: logicalShape4,
                      fileOffset: fileOffset, sizeBytes: sizeBytes,
                      scaleOffset: scaleOffset, scaleSize: scaleSize,
                      biasOffset: biasOffset, biasSize: biasSize,
                      quantSpec: quantSpec,
                      sourceWeight: sourceWeight,
                      sourceScales: sourceScales,
                      sourceBiases: sourceBiases)
    }
}
