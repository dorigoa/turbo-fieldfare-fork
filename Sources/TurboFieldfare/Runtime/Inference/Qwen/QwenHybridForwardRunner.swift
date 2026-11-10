import Foundation
import Metal

/// Qwen 3.5-MoE token-at-a-time forward runner (S19).
///
/// Step-2 skeleton: `produce` runs embedding → final RMSNorm → lm_head with
/// no layers in between, so the logits probe the load path, the embedding
/// table and the 4-bit GEMV rather than the model. Steps 3–6 insert the 40
/// layers of plan 3.5 between the embedding and the final norm. Prefill v1
/// feeds one token at a time: the same `produce` serves prompt and decode,
/// and there is deliberately no `ChunkedPrefillRunner` conformance (the
/// generator then prefills token by token, RawCompletion.swift:172).
///
/// `@unchecked Sendable` like `RealForwardRunner`: the Metal objects are
/// thread-safe for encoding and the debug taps are test-only single-thread
/// state.
public final class QwenHybridForwardRunner: LogitProducer,
                                            ContinuableLogitProducer,
                                            ContextWindowReporting, @unchecked Sendable {

    // MARK: - State

    private let model: Model
    private let ctx: MetalContext
    private let cache: QwenHybridCache
    private let cfg: ArchConfig
    private let hybrid: QwenHybridConfig
    public let maxContext: Int

    // Derived geometry (T = 1 work buffers below).
    private let hiddenSize: Int
    private let vocabSize: Int
    private let valueDim: Int
    private let qkvRows: Int
    private let queryDim: Int
    private let qgRows: Int
    private let kvDim: Int

    // Kernels (D4 module list).
    private let embedding: EmbedLookupInt4
    private let gemv4: DequantInt4GEMV
    private let gemv8: DequantInt8GEMV
    private let norms: QwenNorms
    private let deltaPrep: QwenDeltaNetPrep
    private let deltaScan: QwenDeltaNetScan
    private let attention: QwenAttention
    private let router: QwenRouter
    private let moe: QwenMoE

    // MARK: - Work buffers (one token, T = 1; map S19)

    /// Residual stream row; holds the embedding output in step 2.
    private let hiddenBuffer: MTLBuffer      // FP16 [hiddenSize]
    private let normedBuffer: MTLBuffer      // FP16 [hiddenSize] final RMSNorm out
    private let xNormBuffer: MTLBuffer       // FP16 [hiddenSize] per-layer input norm out
    private let qkvBuffer: MTLBuffer         // FP16 [qkvRows] in_proj_qkv out (DeltaNet)
    private let zBuffer: MTLBuffer           // FP16 [valueDim] in_proj_z out (DeltaNet)
    private let aBuffer: MTLBuffer           // FP16 [Hv] in_proj_a out (DeltaNet)
    private let bBuffer: MTLBuffer           // FP16 [Hv] in_proj_b out (DeltaNet)
    private let convOutBuffer: MTLBuffer     // FP16 [qkvRows] conv+SiLU out; q/k/v split in place
    private let betaBuffer: MTLBuffer        // FP32 [Hv] gates out
    private let decayBuffer: MTLBuffer       // FP32 [Hv] gates out
    private let coreOutBuffer: MTLBuffer     // FP16 [valueDim] DeltaNet scan out
    private let gatedOutBuffer: MTLBuffer    // FP16 [valueDim] DeltaNet gated norm out
    private let mixerBuffer: MTLBuffer       // FP16 [hiddenSize] out_proj/o_proj out (pre-residual)
    private let moeXBuffer: MTLBuffer        // FP16 [hiddenSize] post-attention norm out
    private let routerLogitsBuffer: MTLBuffer // FP16 [numExperts] router GEMV out
    private let routerTopIdsBuffer: MTLBuffer // UInt32 [topK] top-k ids
    private let routerTopWeightsBuffer: MTLBuffer // FP32 [topK] recalibrated top-k weights
    private let expertActsBuffer: MTLBuffer  // FP16 [topK·moeInter] routed gate·up·SiLU
    private let routedOutBuffer: MTLBuffer   // FP16 [topK·hiddenSize] routed down out
    /// C5 staging for the three resident shared-expert tensors (step 4).
    private let sharedBlobBuffer: MTLBuffer
    private let sharedActsBuffer: MTLBuffer  // FP16 [sharedInter] shared gate·up·SiLU
    private let sharedOutBuffer: MTLBuffer   // FP16 [hiddenSize] shared down out
    private let sharedGateBuffer: MTLBuffer  // FP16 [1] shared-expert gate scalar
    private let qgBuffer: MTLBuffer          // FP16 [qgRows] q_proj out (queries + gate)
    private let qBuffer: MTLBuffer           // FP16 [queryDim] query (split, norm, RoPE in place)
    private let gateBuffer: MTLBuffer        // FP16 [queryDim] attention output gate (split)
    private let attnKBuffer: MTLBuffer       // FP16 [kvDim] key (split, norm, RoPE in place)
    private let attnVBuffer: MTLBuffer       // FP16 [kvDim] value
    private let attnOutBuffer: MTLBuffer     // FP16 [queryDim] attention out (pre-gate)
    private let gatedBuffer: MTLBuffer       // FP16 [queryDim] attention out · gate

    // MARK: - Debug taps (test-only, S19 steps 2–7)

    /// Point of the forward pass whose hidden state a `HiddenTap` records.
    internal enum HiddenTapPoint: Equatable {
        /// Right after the embedding lookup, before any layer.
        case afterEmbedding
        /// After layer `layer` has applied its residual to `hiddenBuffer`.
        case afterLayer(Int)
        /// After the final RMSNorm (`normedBuffer`).
        case afterFinalNorm
    }

    /// One FP16 row of `hiddenSize` per produced token, recorded after each
    /// point, in the order of `points`: row for point `i`, token `t` is
    /// `i * rows + t`.
    internal struct HiddenTap: Equatable {
        internal let points: [HiddenTapPoint]
        internal let rows: Int
    }

    /// The tap set by a test before the run; production leaves it `nil`.
    internal var debugHiddenTap: HiddenTap?

    /// The layer whose internal quantities (C7 golden keys) the layer code
    /// records while it runs; `nil` in production.
    internal var debugInternalsLayer: Int?

    /// Recorded hidden rows of the current generation, layout as above.
    internal private(set) var tappedHidden: MTLBuffer?
    /// Tokens recorded so far (per point).
    internal private(set) var tappedHiddenCount = 0
    /// Recorded internals of `debugInternalsLayer`, keyed by C7 golden name.
    /// The values are the runner's own work buffers, so the tapped layer
    /// must be the last layer of the run or later layers overwrite them.
    internal private(set) var tappedInternals: [String: MTLBuffer] = [:]

    /// Test-only (S19 steps 3–5): the layer code records its work buffers
    /// under their C7 golden keys when it runs `debugInternalsLayer`.
    internal func recordInternals(layer: Int, _ entries: [String: MTLBuffer]) {
        guard layer == debugInternalsLayer else { return }
        tappedInternals = entries
    }

    // MARK: - Init

    public init(model: Model, context: MetalContext, maxContext: Int) throws {
        guard model.family == .qwen35Moe, let hybrid = model.qwenHybridConfig else {
            throw ModelError.archMismatch(
                field: "modelFamily",
                expected: ModelFamily.qwen35Moe.rawValue,
                actual: model.family.rawValue)
        }
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.hybrid = hybrid
        self.maxContext = maxContext
        self.hiddenSize = cfg.hiddenSize
        self.vocabSize = cfg.vocabSize
        self.valueDim = hybrid.linearNumValueHeads * hybrid.linearValueHeadDim
        self.qkvRows = 2 * hybrid.linearNumKeyHeads * hybrid.linearKeyHeadDim + valueDim
        self.queryDim = cfg.numHeads * cfg.fullHeadDim
        self.qgRows = hybrid.attentionOutputGate ? queryDim * 2 : queryDim
        self.kvDim = cfg.numKVHeads * cfg.fullHeadDim

        self.cache = try QwenHybridCache(device: context.device,
                                         config: cfg, hybrid: hybrid,
                                         maxContext: maxContext)
        self.embedding = try EmbedLookupInt4(context: context)
        self.gemv4 = try DequantInt4GEMV(context: context)
        self.gemv8 = try DequantInt8GEMV(context: context)
        self.norms = try QwenNorms(context: context)
        self.deltaPrep = try QwenDeltaNetPrep(context: context)
        self.deltaScan = try QwenDeltaNetScan(context: context)
        self.attention = try QwenAttention(context: context)
        self.router = try QwenRouter(context: context)
        self.moe = try QwenMoE(context: context)

        self.hiddenBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize, element: Float16.self, name: "hidden")
        self.normedBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize, element: Float16.self, name: "normed")
        self.xNormBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize, element: Float16.self, name: "xNorm")
        self.qkvBuffer = try makeWorkBuffer(device: context.device, elementCount: qkvRows, element: Float16.self, name: "qkv")
        self.zBuffer = try makeWorkBuffer(device: context.device, elementCount: valueDim, element: Float16.self, name: "z")
        self.aBuffer = try makeWorkBuffer(device: context.device, elementCount: hybrid.linearNumValueHeads, element: Float16.self, name: "a")
        self.bBuffer = try makeWorkBuffer(device: context.device, elementCount: hybrid.linearNumValueHeads, element: Float16.self, name: "b")
        self.convOutBuffer = try makeWorkBuffer(device: context.device, elementCount: qkvRows, element: Float16.self, name: "convOut")
        self.betaBuffer = try makeWorkBuffer(device: context.device, elementCount: hybrid.linearNumValueHeads, element: Float32.self, name: "beta")
        self.decayBuffer = try makeWorkBuffer(device: context.device, elementCount: hybrid.linearNumValueHeads, element: Float32.self, name: "decay")
        self.coreOutBuffer = try makeWorkBuffer(device: context.device, elementCount: valueDim, element: Float16.self, name: "coreOut")
        self.gatedOutBuffer = try makeWorkBuffer(device: context.device, elementCount: valueDim, element: Float16.self, name: "gatedOut")
        self.mixerBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize, element: Float16.self, name: "mixer")
        self.moeXBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize, element: Float16.self, name: "moeX")
        self.routerLogitsBuffer = try makeWorkBuffer(device: context.device, elementCount: cfg.numExperts, element: Float16.self, name: "routerLogits")
        self.routerTopIdsBuffer = try makeWorkBuffer(device: context.device, elementCount: cfg.topKExperts,
            element: UInt32.self, name: "routerTopIds")
        self.routerTopWeightsBuffer = try makeWorkBuffer(device: context.device, elementCount: cfg.topKExperts, element: Float32.self, name: "routerTopWeights")
        self.expertActsBuffer = try makeWorkBuffer(
            device: context.device, elementCount: cfg.topKExperts * cfg.moeIntermediateSize,
            element: Float16.self, name: "expertActs")
        self.routedOutBuffer = try makeWorkBuffer(
            device: context.device, elementCount: cfg.topKExperts * hiddenSize,
            element: Float16.self, name: "routedOut")
        self.sharedBlobBuffer = try makeWorkBuffer(
            device: context.device,
            elementCount: QwenMoE.expertBlobBytes(
                hidden: hiddenSize, inter: cfg.intermediateSize),
            element: UInt8.self, name: "sharedBlob")
        self.sharedActsBuffer = try makeWorkBuffer(device: context.device, elementCount: cfg.intermediateSize, element: Float16.self, name: "sharedActs")
        self.sharedOutBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize, element: Float16.self, name: "sharedOut")
        self.sharedGateBuffer = try makeWorkBuffer(device: context.device, elementCount: 1, element: Float16.self, name: "sharedGate")
        self.qgBuffer = try makeWorkBuffer(device: context.device, elementCount: qgRows, element: Float16.self, name: "qg")
        self.qBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim, element: Float16.self, name: "q")
        self.gateBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim, element: Float16.self, name: "gate")
        self.attnKBuffer = try makeWorkBuffer(device: context.device, elementCount: kvDim, element: Float16.self, name: "attnK")
        self.attnVBuffer = try makeWorkBuffer(device: context.device, elementCount: kvDim, element: Float16.self, name: "attnV")
        self.attnOutBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim, element: Float16.self, name: "attnOut")
        self.gatedBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim, element: Float16.self, name: "gated")
    }

    // MARK: - LogitProducer

    public func reset() {
        cache.reset()
        tappedHidden = nil
        tappedHiddenCount = 0
        tappedInternals = [:]
    }

    /// Runs one token at `position`: embedding → (steps 3–6: layers 0..39
    /// per `fullAttentionLayerMask`) → final RMSNorm → lm_head GEMV, leaving
    /// FP16 logits in `logits`. One token per call is also prefill v1.
    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        guard position >= 0 else {
            throw PrefillError.prefillCursorMismatch(
                "Qwen produce at negative position \(position)")
        }
        guard position < maxContext else {
            throw ModelError.contextExceedsModel(requested: position, maximum: maxContext)
        }
        guard position == cache.position else {
            throw PrefillError.prefillCursorMismatch(
                "Qwen produce at \(position) but the cache is at \(cache.position)")
        }
        precondition(token >= 0, "token ids are non-negative, got \(token)")
        precondition(logits.length >= vocabSize * MemoryLayout<Float16>.size,
                     "logits buffer holds \(logits.length) bytes, need \(vocabSize * MemoryLayout<Float16>.size)")
        guard let commandBuffer = ctx.queue.makeCommandBuffer() else {
            throw QwenRunnerError.commandBufferFailed
        }
        let hidden = hiddenBuffer

        // Embedding: 4-bit affine lookup of row `token` (3.5). Qwen applies
        // no sqrt(hidden) post-embedding scale, so the raw dequant row.
        embedding.encode(
            commandBuffer: commandBuffer,
            table: model.embedding.buffer, tableOffset: Int(model.embedding.offset),
            scales: model.embedding.buffer, scalesOffset: Int(model.embedding.scaleOffset),
            biases: model.embedding.buffer, biasesOffset: Int(model.embedding.biasOffset),
            out: hidden,
            tokenId: UInt32(token), d: UInt32(hiddenSize), outScale: 1.0)
        recordHiddenTap(.afterEmbedding, in: commandBuffer)

        // Steps 3–6: layers 0..<cfg.numLayers, linear (DeltaNet) or full
        // attention per cfg.fullAttentionLayerMask, each followed by the
        // MoE block; the residual stream stays in `hiddenBuffer`.

        // Final RMSNorm; the checkpoint weights already carry the +1
        // (normWeightsShifted), so the plain `x̂ · w` mode (3.5).
        norms.encodeRow(
            commandBuffer: commandBuffer,
            x: hidden,
            weight: model.finalNorm.buffer, weightOffset: Int(model.finalNorm.offset),
            out: normedBuffer,
            d: UInt32(hiddenSize), rows: 1,
            eps: Float(hybrid.rmsNormEps), mode: .plain)
        recordHiddenTap(.afterFinalNorm, in: commandBuffer)

        // lm_head GEMV: FP16 logits into the caller's buffer, no softcap.
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: model.qwenLMHead.buffer, weightsOffset: Int(model.qwenLMHead.offset),
            scales: model.qwenLMHead.buffer, scalesOffset: Int(model.qwenLMHead.scaleOffset),
            biases: model.qwenLMHead.buffer, biasesOffset: Int(model.qwenLMHead.biasOffset),
            x: normedBuffer, y: logits,
            m: UInt32(vocabSize), n: UInt32(hiddenSize))

        commandBuffer.commit()
        await commandBuffer.completed()
        cache.advance(by: 1)
    }

    // MARK: - ContinuableLogitProducer

    /// D6: the Qwen cache can continue exactly where the state is, so the
    /// only legal continuation is onto `cache.position`.
    public var continuationPosition: Int { cache.position }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition == cache.position else {
            throw PrefillError.prefillCursorMismatch(
                "Qwen continuation expects position \(expectedPosition), the cache is at \(cache.position)")
        }
    }

    // MARK: - Taps

    /// GPU-side blit of the current row into the tap buffer so the recorded
    /// state is the one at `point`, whatever the later layers do to the work
    /// buffers. Runs while encoding; the test reads `tappedHidden` after the
    /// command buffer completes.
    private func recordHiddenTap(_ point: HiddenTapPoint, in commandBuffer: MTLCommandBuffer) {
        guard let tap = debugHiddenTap,
              let pointIndex = tap.points.firstIndex(of: point),
              tappedHiddenCount < tap.rows else { return }
        let rowBytes = hiddenSize * MemoryLayout<Float16>.size
        let source: MTLBuffer
        switch point {
        case .afterFinalNorm: source = normedBuffer
        default: source = hiddenBuffer
        }
        let destination: MTLBuffer
        if let existing = tappedHidden {
            destination = existing
        } else {
            let bytes = tap.points.count * tap.rows * rowBytes
            guard let buffer = ctx.device.makeBuffer(length: bytes, options: .storageModeShared)
            else { preconditionFailure("tapped hidden allocation failed (\(bytes) B)") }
            tappedHidden = buffer
            destination = buffer
        }
        let offset = (pointIndex * tap.rows + tappedHiddenCount) * rowBytes
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: source, sourceOffset: 0,
                  to: destination, destinationOffset: offset,
                  size: rowBytes)
        blit.endEncoding()
        tappedHiddenCount += 1
    }
}

/// Allocate one shared-storage work buffer.
private func makeWorkBuffer<E>(device: MTLDevice, elementCount: Int,
                               element: E.Type, name: String) throws -> MTLBuffer {
    let bytes = elementCount * MemoryLayout<E>.size
    guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
    else { throw QwenRunnerError.bufferAllocationFailed(name: name, bytes: bytes) }
    return buffer
}

/// Internal failures of the runner that the public error enums do not
/// express.
private enum QwenRunnerError: Error, CustomStringConvertible {
    case bufferAllocationFailed(name: String, bytes: Int)
    case commandBufferFailed

    var description: String {
        switch self {
        case .bufferAllocationFailed(let name, let bytes):
            return "Qwen runner work buffer \(name) allocation failed (\(bytes) B)"
        case .commandBufferFailed:
            return "Qwen runner command buffer allocation failed"
        }
    }
}
