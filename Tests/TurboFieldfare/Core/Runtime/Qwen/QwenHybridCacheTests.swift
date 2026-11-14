import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// S18: `QwenHybridCache` byte geometry must equal plan section 3.4, and the
/// D6 rules (continue always, snapshot/restore, rewind only onto the state
/// position) must be enforced.
@Suite("QwenHybridCache")
struct QwenHybridCacheTests {

    private func makeCache(maxContext: Int) throws -> QwenHybridCache {
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try QwenHybridCache(device: device, maxContext: maxContext)
    }

    private func fill(_ buffer: MTLBuffer, byte: Int32, count: Int) {
        memset(buffer.contents(), byte, count)
    }

    private func firstDiff(_ buffer: MTLBuffer, byte: UInt8, count: Int) -> Int? {
        let raw = UnsafeRawBufferPointer(
            start: buffer.contents(), count: count)
        return raw.firstIndex(of: byte)
    }

    // MARK: - Sizes (3.4)

    @Test("byte geometry equals plan 3.4 at 4K and 32K")
    func byteGeometryMatchesPlan34() throws {
        let cache = try makeCache(maxContext: 4096)
        // 10 layers x 2 (K,V) x 2 heads x 256 x 2 B = 20.480 B/token.
        #expect(cache.kvBytesPerToken == 20_480)
        #expect(cache.kvBytesPerPositionPerLayer == 1_024)
        #expect(cache.attentionLayers.count == 10)
        #expect(cache.linearLayers.count == 30)
        // 4K: 20.480 x 4096 = 83.886.080 B = 80 MiB of KV.
        #expect(cache.kvBytesPerToken * 4_096 == 83_886_080)
        // 32K: 671.088.640 B = 640 MiB of KV.
        #expect(cache.kvBytesPerToken * 32_768 == 671_088_640)
        // DeltaNet 30 x 32 x 128 x 128 x 4 B; conv 30 x 3 x 8192 x 4 B.
        #expect(cache.deltaStateBytes == 62_914_560)
        #expect(cache.convStateBytes == 2_949_120)
        // Snapshot = DeltaNet + conv only (D6, ~63 MiB).
        #expect(cache.snapshotBytes == 65_863_680)
    }

    @Test("packed buffers carry the 3.4 byte counts for maxContext")
    func bufferLengthsMatchGeometry() throws {
        let cache = try makeCache(maxContext: 4096)
        #expect(cache.kBuffer.length == 10 * 4_096 * 1_024)
        #expect(cache.vBuffer.length == cache.kBuffer.length)
        #expect(cache.deltaStateBuffer.length == 62_914_560)
        #expect(cache.convStateBuffer.length == 2_949_120)
        #expect(cache.kvLayerStrideBytes == 4_096 * 1_024)
        #expect(cache.deltaStateLayerStrideBytes == 32 * 128 * 128 * 4)
        #expect(cache.convStateLayerStrideBytes == 3 * 8_192 * 4)
    }

    @Test("a 32K cache reports the 640 MiB KV footprint")
    func thirtyTwoKFootprint() throws {
        let cache = try makeCache(maxContext: 32_768)
        #expect(cache.kBuffer.length == 10 * 32_768 * 1_024)
        #expect(cache.kBuffer.length * 2 == 671_088_640)
    }

    @Test("maxContext must be positive")
    func invalidMaxContextThrows() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: QwenHybridCacheError.invalidMaxContext(0)) {
            try QwenHybridCache(device: device, maxContext: 0)
        }
    }

    // MARK: - Layout addressing (D5)

    @Test("buffer offsets follow the D5 packed layout")
    func packedOffsets() throws {
        let cache = try makeCache(maxContext: 256)
        // Attention layers sit at 3, 7, ..., 39.
        #expect(cache.attentionLayers == Array(stride(from: 3, to: 40, by: 4)))
        #expect(cache.linearLayers.first == 0)
        #expect(cache.kvByteOffset(layer: 3, position: 0) == 0)
        #expect(cache.kvByteOffset(layer: 7, position: 0) == 256 * 1_024)
        #expect(cache.kvByteOffset(layer: 3, position: 5) == 5 * 1_024)
        // A linear layer has no KV; an attention layer has no DeltaNet state.
        #expect(cache.kvByteOffset(layer: 0, position: 0) == nil)
        #expect(cache.kvByteOffset(layer: 3, position: 256) == nil)
        #expect(cache.deltaStateByteOffset(layer: 0) == 0)
        #expect(cache.deltaStateByteOffset(layer: 1) == 32 * 128 * 128 * 4)
        #expect(cache.deltaStateByteOffset(layer: 3) == nil)
        #expect(cache.convStateByteOffset(layer: 1) == 3 * 8_192 * 4)
        #expect(cache.convStateByteOffset(layer: 3) == nil)
    }

    // MARK: - Snapshot / restore / truncate (D6)

    @Test("snapshot and restore are byte-identical")
    func snapshotRestoreByteIdentical() throws {
        let cache = try makeCache(maxContext: 64)
        fill(cache.deltaStateBuffer, byte: 0xA5, count: cache.deltaStateBytes)
        fill(cache.convStateBuffer, byte: 0x3C, count: cache.convStateBytes)
        cache.advance(by: 10)

        let snapshot = cache.snapshot()
        #expect(snapshot.position == 10)

        // Trash both states after the snapshot.
        fill(cache.deltaStateBuffer, byte: 0x00, count: cache.deltaStateBytes)
        fill(cache.convStateBuffer, byte: 0xFF, count: cache.convStateBytes)
        try cache.restore(snapshot)
        #expect(cache.statePosition == 10)
        #expect(cache.position == 10)

        // Both restored regions must hold the pre-fill pattern byte for byte.
        #expect(firstDiff(cache.deltaStateBuffer, byte: 0x00,
                          count: cache.deltaStateBytes) == nil)
        #expect(firstDiff(cache.convStateBuffer, byte: 0xFF,
                          count: cache.convStateBytes) == nil)
    }

    @Test("rewind is legal only onto the DeltaNet state position")
    func rewindRules() throws {
        let cache = try makeCache(maxContext: 64)
        cache.advance(by: 10)
        let snapshot = cache.snapshot()

        // Continue at the tail always works.
        cache.advance(by: 4)
        // Going back to the snapshot position is a delta rewind: illegal
        // before the state is restored (the state now describes token 14).
        #expect(throws: QwenHybridCacheError.deltaStateCannotRewind(
            target: 10, statePosition: 14)) {
            try cache.truncate(to: 10)
        }
        // Discarding tokens written after the state position is legal.
        try cache.truncate(to: 14)
        #expect(cache.position == 14)

        // Extension path (D6): restore, then join the KV tail onto the state.
        try cache.restore(snapshot)
        #expect(cache.statePosition == 10)
        #expect(throws: QwenHybridCacheError.truncationTargetOutOfRange(
            target: 15, position: 14)) {
            try cache.truncate(to: 15)
        }
        #expect(throws: QwenHybridCacheError.truncationTargetOutOfRange(
            target: -1, position: 14)) {
            try cache.truncate(to: -1)
        }
        try cache.truncate(to: 10)
        #expect(cache.position == 10)
        #expect(cache.statePosition == 10)
    }

    @Test("restore from a foreign cache layout fails closed")
    func snapshotLayoutMismatchThrows() throws {
        let mine = try makeCache(maxContext: 64)
        let other = try QwenHybridCache(
            device: mine.device,
            config: .qwen36_35B_A3B,
            hybrid: QwenHybridConfig(
                linearNumKeyHeads: 16,
                linearNumValueHeads: 16,
                linearKeyHeadDim: 128,
                linearValueHeadDim: 128,
                linearConvKernel: 4,
                rmsNormEps: 1e-6,
                normWeightsShifted: true,
                attentionOutputGate: true,
                sharedExpertGate: true),
            maxContext: 64)
        let foreign = other.snapshot()
        #expect(throws: QwenHybridCacheError.snapshotLayoutMismatch) {
            try mine.restore(foreign)
        }
    }

    @Test("reset zeroes the recurrent state and rewinds positions")
    func resetClearsState() throws {
        let cache = try makeCache(maxContext: 64)
        fill(cache.deltaStateBuffer, byte: 0xA5, count: cache.deltaStateBytes)
        fill(cache.convStateBuffer, byte: 0xA5, count: cache.convStateBytes)
        cache.advance(by: 7)
        cache.reset()
        #expect(cache.position == 0)
        #expect(cache.statePosition == 0)
        #expect(firstDiff(cache.deltaStateBuffer, byte: 0xA5,
                          count: cache.deltaStateBytes) == nil)
        #expect(firstDiff(cache.convStateBuffer, byte: 0xA5,
                          count: cache.convStateBytes) == nil)
    }
}
