import Foundation
import Metal

/// Errors for `QwenHybridCache` (D6): every rewind that cannot restore a
/// consistent DeltaNet state fails closed instead of corrupting it.
public enum QwenHybridCacheError: Error, Equatable, Sendable {
    case invalidMaxContext(Int)
    case invalidGeometry(detail: String)
    case truncationTargetOutOfRange(target: Int, position: Int)
    case deltaStateCannotRewind(target: Int, statePosition: Int)
    case snapshotLayoutMismatch
}

/// Qwen 3.5-MoE hybrid cache (S18, D5/D6).
///
/// Layout (D5):
/// - KV FP16, only the full-attention layers, no sliding ring:
///   `[attention layer][position][KV head][fullHeadDim]` in one packed K and
///   one packed V buffer.
/// - DeltaNet state FP32, one packed buffer for all linear layers:
///   `[linear layer][value head][column c][row i]` (a column is
///   `linearValueHeadDim` contiguous floats).
/// - Conv state FP32: `[linear layer][convKernel - 1][qkv rows]`.
///
/// Prefix reuse (D6): the DeltaNet state cannot be rewound. Continuing at the
/// tail of the current position always works. `snapshot()` copies the DeltaNet
/// and conv state (~63 MiB) at the current position; `restore(_:)` puts that
/// state back and records the snapshot position; `truncate(to:)` then moves
/// the KV tail to that position. Truncating anywhere else throws
/// `deltaStateCannotRewind` and the caller re-prefills.
public final class QwenHybridCache {

    public let device: MTLDevice
    public let config: ArchConfig
    public let hybrid: QwenHybridConfig
    public let maxContext: Int

    /// Model layer indices with full attention, in model order.
    public let attentionLayers: [Int]
    /// Model layer indices with linear (DeltaNet) attention, in model order.
    public let linearLayers: [Int]
    /// Model layer → slot in the packed KV buffers.
    public let attentionSlot: [Int: Int]
    /// Model layer → slot in the packed DeltaNet/conv buffers.
    public let linearSlot: [Int: Int]

    // MARK: - Byte geometry (3.4)

    /// Bytes per position inside one packed K (or V) buffer, one layer.
    public let kvBytesPerPositionPerLayer: Int
    /// Bytes per token across all attention layers, K and V together.
    public let kvBytesPerToken: Int
    /// Total bytes of the packed DeltaNet state buffer (constant).
    public let deltaStateBytes: Int
    /// Total bytes of the packed conv state buffer (constant).
    public let convStateBytes: Int
    /// Bytes copied by `snapshot()`: DeltaNet + conv state (D6, ≈ 63 MiB).
    public let snapshotBytes: Int

    /// Byte stride between positions inside one layer's KV slice.
    public var kvPositionStrideBytes: Int { kvBytesPerPositionPerLayer }
    /// Byte stride between layers inside one packed K/V buffer.
    public var kvLayerStrideBytes: Int {
        kvBytesPerPositionPerLayer * maxContext
    }
    /// Byte stride between linear layers inside the packed state buffers.
    public var deltaStateLayerStrideBytes: Int {
        deltaStateBytes / max(linearLayers.count, 1)
    }
    public var convStateLayerStrideBytes: Int {
        convStateBytes / max(linearLayers.count, 1)
    }

    // MARK: - Buffers

    /// Packed K buffer: `[attention slot][position][KV head][fullHeadDim]`.
    public let kBuffer: MTLBuffer
    /// Packed V buffer, same layout as `kBuffer`.
    public let vBuffer: MTLBuffer
    /// Packed DeltaNet state: `[linear slot][value head][column][row]`.
    public let deltaStateBuffer: MTLBuffer
    /// Packed conv state: `[linear slot][tap][qkv row]`.
    public let convStateBuffer: MTLBuffer

    /// Token position at the KV tail.
    public private(set) var position: Int = 0
    /// Position the DeltaNet/conv state currently describes. After
    /// `restore(_:)` this is the snapshot position while `position` may still
    /// be ahead of it; `truncate(to:)` is the only legal join.
    public private(set) var statePosition: Int = 0

    public struct Snapshot {
        fileprivate let layoutKey: CacheLayoutKey
        fileprivate let deltaBytes: Data
        fileprivate let convBytes: Data
        /// Position the restored state describes; the only truncation target
        /// legal after this snapshot is restored.
        public let position: Int
    }

    fileprivate struct CacheLayoutKey: Equatable {
        let attentionLayerCount: Int
        let linearLayerCount: Int
        let deltaStateBytes: Int
        let convStateBytes: Int
    }
    private let layoutKey: CacheLayoutKey

    // MARK: - Init

    public init(device: MTLDevice,
                config: ArchConfig = .qwen36_35B_A3B,
                hybrid: QwenHybridConfig = .qwen36_35B_A3B,
                maxContext: Int) throws {
        guard maxContext > 0 else {
            throw QwenHybridCacheError.invalidMaxContext(maxContext)
        }
        guard config.fullAttentionLayerMask.count == config.numLayers,
              config.numLayers > 0 else {
            throw QwenHybridCacheError.invalidGeometry(
                detail: "fullAttentionLayerMask does not cover every layer")
        }
        let attentionLayers = (0..<config.numLayers).filter {
            config.fullAttentionLayerMask[$0] != 0
        }
        let linearLayers = (0..<config.numLayers).filter {
            config.fullAttentionLayerMask[$0] == 0
        }
        guard !attentionLayers.isEmpty, !linearLayers.isEmpty else {
            throw QwenHybridCacheError.invalidGeometry(
                detail: "a hybrid cache needs both attention and linear layers")
        }
        // The Qwen kernels read every attention layer with the full geometry
        // and every linear layer with the hybrid geometry; fail closed rather
        // than mis-stride a per-layer head dimension.
        guard config.headDim == config.fullHeadDim,
              config.numKVHeads == config.numFullKVHeads,
              config.slidingWindow == 0,
              hybrid.linearConvKernel >= 2,
              hybrid.linearNumKeyHeads > 0,
              hybrid.linearNumValueHeads > 0,
              hybrid.linearKeyHeadDim > 0,
              hybrid.linearValueHeadDim > 0 else {
            throw QwenHybridCacheError.invalidGeometry(
                detail: "config does not use the single Qwen attention geometry")
        }

        let valueDim = hybrid.linearNumValueHeads * hybrid.linearValueHeadDim
        let qkvRows = 2 * hybrid.linearNumKeyHeads * hybrid.linearKeyHeadDim + valueDim
        let attentionCount = attentionLayers.count
        let linearCount = linearLayers.count

        func checked(_ lhs: Int, _ rhs: Int, _ what: String) throws -> Int {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow, value >= 0 else {
                throw QwenHybridCacheError.invalidGeometry(
                    detail: "\(what) byte count overflows Int")
            }
            return value
        }

        let kvBytesPerPositionPerLayer = try checked(
            config.numFullKVHeads * config.fullHeadDim, 2, "KV per position")
        let kvBytesPerToken = try checked(
            attentionCount, 2 * kvBytesPerPositionPerLayer, "KV per token")
        let kvBufferBytes = try checked(
            attentionCount * maxContext, kvBytesPerPositionPerLayer, "KV buffer")
        let deltaStateBytes = try checked(
            linearCount * hybrid.linearNumValueHeads * hybrid.linearValueHeadDim,
            hybrid.linearKeyHeadDim * 4, "DeltaNet state")
        let convStateBytes = try checked(
            linearCount * (hybrid.linearConvKernel - 1) * qkvRows, 4, "conv state")

        self.device = device
        self.config = config
        self.hybrid = hybrid
        self.maxContext = maxContext
        self.attentionLayers = attentionLayers
        self.linearLayers = linearLayers
        self.attentionSlot = Dictionary(
            uniqueKeysWithValues: attentionLayers.enumerated().map { ($0.element, $0.offset) })
        self.linearSlot = Dictionary(
            uniqueKeysWithValues: linearLayers.enumerated().map { ($0.element, $0.offset) })
        self.kvBytesPerPositionPerLayer = kvBytesPerPositionPerLayer
        self.kvBytesPerToken = kvBytesPerToken
        self.deltaStateBytes = deltaStateBytes
        self.convStateBytes = convStateBytes
        self.snapshotBytes = deltaStateBytes + convStateBytes
        self.layoutKey = CacheLayoutKey(
            attentionLayerCount: attentionCount,
            linearLayerCount: linearCount,
            deltaStateBytes: deltaStateBytes,
            convStateBytes: convStateBytes)

        guard let kBuffer = device.makeBuffer(length: kvBufferBytes,
                                              options: .storageModeShared),
              let vBuffer = device.makeBuffer(length: kvBufferBytes,
                                              options: .storageModeShared),
              let deltaStateBuffer = device.makeBuffer(length: deltaStateBytes,
                                                       options: .storageModeShared),
              let convStateBuffer = device.makeBuffer(length: convStateBytes,
                                                      options: .storageModeShared) else {
            throw QwenHybridCacheError.invalidGeometry(
                detail: "could not allocate hybrid cache buffers")
        }
        self.kBuffer = kBuffer
        self.vBuffer = vBuffer
        self.deltaStateBuffer = deltaStateBuffer
        self.convStateBuffer = convStateBuffer
        // The DeltaNet recurrence reads the state before writing it, so the
        // state must start zeroed; fresh Metal buffers make no such promise.
        memset(deltaStateBuffer.contents(), 0, deltaStateBytes)
        memset(convStateBuffer.contents(), 0, convStateBytes)
    }

    // MARK: - Buffer addressing

    /// Byte offset of `(layer, position)` inside `kBuffer`/`vBuffer`.
    public func kvByteOffset(layer: Int, position: Int) -> Int? {
        guard let slot = attentionSlot[layer],
              position >= 0, position < maxContext else { return nil }
        return slot * kvLayerStrideBytes + position * kvBytesPerPositionPerLayer
    }

    /// Byte offset of one linear layer inside the packed DeltaNet state.
    public func deltaStateByteOffset(layer: Int) -> Int? {
        guard let slot = linearSlot[layer] else { return nil }
        return slot * deltaStateLayerStrideBytes
    }

    /// Byte offset of one linear layer inside the packed conv state.
    public func convStateByteOffset(layer: Int) -> Int? {
        guard let slot = linearSlot[layer] else { return nil }
        return slot * convStateLayerStrideBytes
    }

    // MARK: - Position (D6)

    /// Advances the KV tail after tokens were written at the old tail. The
    /// DeltaNet/conv state is always advanced in lockstep by the decode step
    /// that writes these tokens, so `statePosition` follows.
    public func advance(by tokens: Int) {
        precondition(tokens >= 0, "advance cannot move backwards")
        precondition(statePosition == position,
                     "the DeltaNet state is out of sync; truncate or restore first")
        precondition(position + tokens <= maxContext,
                     "advance past maxContext (\(maxContext))")
        position += tokens
        statePosition = position
    }

    /// Drops the KV tail down to `newPosition`. Only a target that keeps the
    /// DeltaNet state consistent is legal: the current state position (D6:
    /// continuing at the tail, or discarding tokens written after it).
    /// Any other rewind throws `deltaStateCannotRewind` and the caller
    /// re-prefills.
    public func truncate(to newPosition: Int) throws {
        guard newPosition >= 0, newPosition <= position else {
            throw QwenHybridCacheError.truncationTargetOutOfRange(
                target: newPosition, position: position)
        }
        guard newPosition == statePosition else {
            throw QwenHybridCacheError.deltaStateCannotRewind(
                target: newPosition, statePosition: statePosition)
        }
        position = newPosition
    }

    // MARK: - Snapshot / restore (D6)

    /// Copies the DeltaNet + conv state (~63 MiB) at the current position.
    /// The KV itself is never snapshotted; the caller keeps at most two
    /// snapshots (D6).
    public func snapshot() -> Snapshot {
        precondition(statePosition == position,
                     "snapshot needs a consistent DeltaNet state")
        var delta = Data(count: deltaStateBytes)
        var conv = Data(count: convStateBytes)
        delta.withUnsafeMutableBytes { (target: UnsafeMutableRawBufferPointer) in
            memcpy(target.baseAddress!, deltaStateBuffer.contents(), deltaStateBytes)
        }
        conv.withUnsafeMutableBytes { (target: UnsafeMutableRawBufferPointer) in
            memcpy(target.baseAddress!, convStateBuffer.contents(), convStateBytes)
        }
        return Snapshot(layoutKey: layoutKey,
                        deltaBytes: delta,
                        convBytes: conv,
                        position: statePosition)
    }

    /// Puts a snapshot's DeltaNet + conv state back. The KV tail stays where
    /// it is; `truncate(to: snapshot.position)` is then the legal join.
    public func restore(_ snapshot: Snapshot) throws {
        guard snapshot.layoutKey == layoutKey else {
            throw QwenHybridCacheError.snapshotLayoutMismatch
        }
        snapshot.deltaBytes.withUnsafeBytes {
            (deltaSource: UnsafeRawBufferPointer) in
            memcpy(deltaStateBuffer.contents(), deltaSource.baseAddress!, deltaStateBytes)
        }
        snapshot.convBytes.withUnsafeBytes { (convSource: UnsafeRawBufferPointer) in
            memcpy(convStateBuffer.contents(), convSource.baseAddress!, convStateBytes)
        }
        statePosition = snapshot.position
    }

    /// Clears everything: zeroed DeltaNet/conv state (the recurrence reads
    /// before it writes) and position 0. The KV needs no zeroing because
    /// every reader stops at `position`.
    public func reset() {
        memset(deltaStateBuffer.contents(), 0, deltaStateBytes)
        memset(convStateBuffer.contents(), 0, convStateBytes)
        position = 0
        statePosition = 0
    }
}
