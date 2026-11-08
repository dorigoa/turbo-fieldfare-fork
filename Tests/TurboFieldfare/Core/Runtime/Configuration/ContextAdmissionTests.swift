import Testing
import Foundation
@testable import TurboFieldfare

/// Pins the admission arithmetic every context-picking surface shares. The
/// byte figures are the FP16 KV allocation the runtime actually makes — 25
/// ring-capped sliding layers plus 5 linear full layers — so a change to the
/// ring geometry or the layer mask fails here before it silently moves a menu
/// label, a server refusal and a CLI warning apart.
@Suite struct ContextAdmissionTests {

    @Test(arguments: [24, 32])
    func extraCacheSlotsAreChargedAtTheAdmissionBoundary(slots: Int) {
        let config = ArchConfig.gemma4_26B_A4B
        let baseline = ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 131_072)
        let page = Int(getpagesize())
        let slotBytes = ((3_358_720 + page - 1) / page) * page
        let expected = baseline + UInt64((slots - 16) * 30 * slotBytes)
        #expect(ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 131_072,
                                                        expertCacheSlots: slots) == expected)
        #expect(ContextAdmission.availability(config: config, maxContext: 131_072,
                                              hostMemoryBytes: expected - 1, expertCacheSlots: slots)
                == .needsMemory(minimumHostBytes: expected))
        #expect(ContextAdmission.availability(config: config, maxContext: 131_072,
                                              hostMemoryBytes: expected, expertCacheSlots: slots) == .available)
        #expect(ContextAdmission.availability(config: config, maxContext: 131_072,
                                              hostMemoryBytes: 8 << 30, expertCacheSlots: slots)
                == .needsMemory(minimumHostBytes: expected))
    }

    @Test func smallerCacheDoesNotDiscountTheMeasuredRuntimeAllowance() {
        #expect(ContextAdmission.projectedFootprintBytes(config: config, maxContext: 131_072,
                                                         expertCacheSlots: 8)
                == ContextAdmission.projectedFootprintBytes(config: config, maxContext: 131_072))
    }


    private let config = ArchConfig.gemma4_26B_A4B
    private let eightGigabyteHost: UInt64 = 8_589_934_592
    private let sixteenGigabyteHost: UInt64 = 17_179_869_184

    @Test func fp16KVBytesMatchesTheRuntimeAllocationAtEverySize() {
        let expected: [(context: Int, bytes: UInt64)] = [
            (4_096, 350_945_280),
            (8_192, 434_831_360),
            (16_384, 602_603_520),
            (32_768, 938_147_840),
            (65_536, 1_609_236_480),
            (131_072, 2_951_413_760),
            (262_144, 5_635_768_320),
        ]
        for row in expected {
            #expect(ContextAdmission.fp16KVBytes(config: config, maxContext: row.context)
                    == row.bytes,
                    "context \(row.context)")
        }
    }

    /// Only the five full-attention layers grow with the context; the sliding
    /// ring is fixed. 20,480 B per token across those five is the whole slope.
    @Test func onlyFullAttentionLayersGrowWithContext() {
        let small = ContextAdmission.fp16KVBytes(config: config, maxContext: 8_192)
        let large = ContextAdmission.fp16KVBytes(config: config, maxContext: 16_384)
        #expect(large - small == UInt64((16_384 - 8_192) * 20_480))
    }

    @Test func productionSlidingRingRowsMatchTheRunnersChunkFloor() {
        #expect(ContextAdmission.productionSlidingRingRows(config: config, maxContext: 8_192)
                == 1_304)
        // A context below the ring size caps the ring at the context.
        #expect(ContextAdmission.productionSlidingRingRows(config: config, maxContext: 1_024)
                == 1_024)
    }

    @Test func eightGigabyteHostBacks128KButNot256K() {
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 131_072,
                                              hostMemoryBytes: eightGigabyteHost)
                == .available)
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 262_144,
                                              hostMemoryBytes: eightGigabyteHost)
                == .needsMemory(minimumHostBytes: 11_004_477_440))
    }

    @Test func sixteenGigabyteHostBacksEverySize() {
        for context in [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144] {
            #expect(ContextAdmission.availability(config: config,
                                                  maxContext: context,
                                                  hostMemoryBytes: sixteenGigabyteHost)
                    == .available,
                    "context \(context)")
        }
    }

    /// The rule admits at exactly the minimum, not one byte above it: an
    /// off-by-one here would hide 128K from every 8 GB Mac, whose 270 MB of
    /// margin is the whole question step 9 of the plan measures.
    @Test func admissionBoundaryIsInclusive() {
        let minimum = ContextAdmission.minimumHostMemoryBytes(config: config,
                                                              maxContext: 131_072)
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 131_072,
                                              hostMemoryBytes: minimum - 1)
                == .needsMemory(minimumHostBytes: minimum))
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 131_072,
                                              hostMemoryBytes: minimum)
                == .available)
    }

    @Test func projectedFootprintIsKVPlusTheMeasuredNonKVRuntime() {
        let projected = ContextAdmission.projectedFootprintBytes(config: config,
                                                                 maxContext: 262_144)
        #expect(projected == 5_635_768_320 + ContextAdmission.nonKVRuntimeFootprintBytes)
        #expect(ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 262_144)
                == projected + ContextAdmission.hostReserveBytes)
    }

    @Test func needDescriptionNamesBothTheRequirementAndTheHost() {
        let message = ContextAdmission.needDescription(config: config,
                                                       maxContext: 262_144,
                                                       hostMemoryBytes: eightGigabyteHost)
        #expect(message.contains("16 GB"))
        #expect(message.contains("8 GB"))
        #expect(message.contains("262,144"))
    }

    /// 128K needs 8,320,122,880 bytes, which an 8 GB Mac has. Reporting that
    /// as "needs 16 GB" while `availability` calls it available would put the
    /// caption and the menu in direct contradiction.
    @Test func needDescriptionRoundsToASizeThatActuallyBacksTheContext() {
        let message = ContextAdmission.needDescription(config: config,
                                                       maxContext: 131_072,
                                                       hostMemoryBytes: eightGigabyteHost)
        #expect(message.contains("needs 8 GB"))
    }

    @Test func contextCeilingComesFromTheCheckpoint() {
        #expect(config.maxPositionEmbeddings == 262_144)
    }

    // MARK: - Footprint per quantization profile

    /// Every figure above is the 4-bit pack's; naming its footprint must not
    /// move any of them.
    @Test func explicitAffine4FootprintMatchesTheDefaults() {
        for context in [4_096, 8_192, 131_072, 262_144] {
            for slots in [8, 16, 32] {
                #expect(ContextAdmission.minimumHostMemoryBytes(
                            config: config, maxContext: context, expertCacheSlots: slots,
                            footprint: .affine4Measured)
                        == ContextAdmission.minimumHostMemoryBytes(
                            config: config, maxContext: context, expertCacheSlots: slots))
            }
        }
        #expect(ModelFootprint.affine4Measured.nonKVRuntimeBytes
                == ContextAdmission.nonKVRuntimeFootprintBytes)
        #expect(ModelFootprint.affine4Measured.hostReserveBytes
                == ContextAdmission.hostReserveBytes)
        #expect(ModelFootprint.forProfile(.affine4) == .affine4Measured)
        #expect(ModelFootprint.forProfile(.affine8) == .affine8Measured)
        #expect(ModelFootprint.affine8Provisional == .affine8Measured)
    }

    /// S9's measurements on a 32 GB M5 (`/usr/bin/time -l`, long-synthesis
    /// prompt, 256 new tokens): the projection must cover every peak physical
    /// footprint the 8-bit pack reached, at each context and slot count.
    @Test func affine8ProjectionCoversTheMeasuredFootprints() {
        let measured: [(context: Int, slots: Int, peakFootprint: UInt64)] = [
            (4_096, 16, 3_653_817_216),
            (4_096, 32, 6_694_919_872),
            (8_192, 16, 3_732_067_200),
            (8_192, 32, 6_783_786_712),
        ]
        for run in measured {
            let projected = ContextAdmission.projectedFootprintBytes(
                config: config, maxContext: run.context, expertCacheSlots: run.slots,
                footprint: .affine8Measured)
            #expect(projected >= run.peakFootprint, "context \(run.context), \(run.slots) slots")
            #expect(projected - run.peakFootprint < 1 << 29,
                    "context \(run.context), \(run.slots) slots: margin above half a GiB")
        }
    }

    /// The 8-bit pack is refused on an 8 GB Mac at every context and admitted
    /// on a 16 GB Mac at the app's default sizes.
    @Test func affine8IsRefusedOnEightGigabytesAndAdmittedOnSixteen() {
        let footprint = ModelFootprint.affine8Measured
        for context in [4_096, 8_192, 32_768] {
            let eight = ContextAdmission.availability(config: config, maxContext: context,
                                                      hostMemoryBytes: eightGigabyteHost,
                                                      footprint: footprint)
            #expect(eight != .available, "context \(context)")
            #expect(ContextAdmission.availability(config: config, maxContext: context,
                                                  hostMemoryBytes: sixteenGigabyteHost,
                                                  footprint: footprint) == .available,
                    "context \(context)")
        }
        let message = ContextAdmission.needDescription(config: config, maxContext: 8_192,
                                                       hostMemoryBytes: eightGigabyteHost,
                                                       footprint: footprint)
        #expect(message.contains("needs 16 GB"))
    }

    @Test func affine8ChargesExtraSlotsAtItsOwnStride() {
        let footprint = ModelFootprint.affine8Measured
        let page = Int(getpagesize())
        let slotBytes = ((6_324_224 + page - 1) / page) * page
        let baseline = ContextAdmission.minimumHostMemoryBytes(
            config: config, maxContext: 8_192, footprint: footprint)
        #expect(ContextAdmission.minimumHostMemoryBytes(
                    config: config, maxContext: 8_192, expertCacheSlots: 32,
                    footprint: footprint)
                == baseline + UInt64(16 * 30 * slotBytes))
    }

    @Test func footprintIsReadFromTheInstalledManifest() throws {
        let (affine8Directory, _) = try ManifestReaderTests.writeToyManifest(
            ["quant": ManifestReaderTests.quantAffine8()],
            config: config)
        defer { try? FileManager.default.removeItem(at: affine8Directory) }
        #expect(try ModelFootprint.forModelDirectory(affine8Directory) == .affine8Measured)

        let (affine4Directory, _) = try ManifestReaderTests.writeToyManifest(
            ["quant": ManifestReaderTests.quant()],
            config: config)
        defer { try? FileManager.default.removeItem(at: affine4Directory) }
        #expect(try ModelFootprint.forModelDirectory(affine4Directory) == .affine4Measured)
    }
}
