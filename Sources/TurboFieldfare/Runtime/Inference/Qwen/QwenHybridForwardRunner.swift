import Foundation
import Metal

/// Errors for the Qwen prefix reuse (D6). `PrefillError` is Gemma's and is
/// not modified, so the runner throws its own type instead.
public enum QwenRewindError: Error, Equatable, Sendable {
    /// No prefill snapshot at the requested position (D6): the DeltaNet
    /// state cannot be rewound anywhere else, so the caller re-prefills
    /// from position 0.
    case rewindUnsupported(position: Int)
}

/// Qwen 3.5-MoE forward runner: token-at-a-time `produce` (S19) and
/// chunked prefill (S20).
///
/// `produce` runs the embedding, then each of the 40 layers of plan 3.5 —
/// a Gated DeltaNet mixer where the `fullAttentionLayerMask` bit is 0, a
/// gated full-attention mixer where it is 1, each followed by its MoE block
/// (routed and shared expert) — then the final RMSNorm and the lm_head.
/// `prefillChunked` (S20, QwenHybridForwardRunner+Prefill.swift) runs the
/// same passes over `T` rows at a time, in the `PrefillRuntimeConfig`
/// chunk size, and the work buffers below are row-major for that.
///
/// `@unchecked Sendable` like `RealForwardRunner`: the Metal objects are
/// thread-safe for encoding and the debug taps are test-only single-thread
/// state.
public final class QwenHybridForwardRunner: LogitProducer,
                                            ContinuableLogitProducer,
                                            ContextWindowReporting, @unchecked Sendable {

    // MARK: - State

    /// Internal (not public) so the S20 prefill extension and the S19 tests
    /// can drive the runner.
    let model: Model
    let ctx: MetalContext
    /// Internal (not public) so the S19 tests can resolve the DeltaNet and
    /// conv state byte offsets of the tapped layer.
    internal let cache: QwenHybridCache
    let cfg: ArchConfig
    let hybrid: QwenHybridConfig
    public let maxContext: Int

    // Derived geometry (per row of the work buffers below).
    let hiddenSize: Int
    let vocabSize: Int
    let valueDim: Int
    let qkvRows: Int
    let queryDim: Int
    let qgRows: Int
    let kvDim: Int

    // Kernels (D4 module list).
    let embedding: EmbedLookupInt4
    let gemv4: DequantInt4GEMV
    let gemv8: DequantInt8GEMV
    let norms: QwenNorms
    let deltaPrep: QwenDeltaNetPrep
    let deltaScan: QwenDeltaNetScan
    let attention: QwenAttention
    let router: QwenRouter
    let moe: QwenMoE
    /// S20 chunked prefill: the 4-bit MPP GEMM over the chunk's rows (map
    /// S20 "GEMM prefill"); the per-row GEMV is its fallback.
    let mpp4: MPPPrefillInt4QMM
    /// S20 chunked prefill: the 8-bit MPP GEMM for the router over the
    /// chunk's rows (map S20 "GEMM prefill"); the per-row GEMV is its
    /// fallback (e.g. when the weight offset breaks the 4-byte alignment).
    let mpp8: MPPPrefillInt8QMM

    /// S20 chunked prefill: dirty while a chunked prefill has advanced the
    /// cache without committing; `prefillChunked` rejects with the same
    /// message as Gemma's `RealForwardRunner` until `reset()`.
    var chunkCommitState = PrefillChunkCommitState()

    /// D6 prefix reuse: the DeltaNet + conv state snapshots taken at the
    /// end of chunked prefills, oldest first, at most two (the oldest is
    /// evicted). A snapshot only addresses KV that stays intact until a
    /// prefill overwrites it, so `reset()` — which restarts the KV at
    /// position 0 — clears the list.
    var prefixSnapshots: [QwenHybridCache.Snapshot] = []

    // MARK: - Work buffers (row-major, maxT = PrefillRuntimeConfig.maxChunkTokens
    // rows; `produce` uses row 0, chunked prefill the first T; map S19/S20)

    /// Residual stream row; holds the embedding output in step 2.
    let hiddenBuffer: MTLBuffer      // FP16 [hiddenSize]
    let normedBuffer: MTLBuffer      // FP16 [hiddenSize] final RMSNorm out
    let xNormBuffer: MTLBuffer       // FP16 [hiddenSize] per-layer input norm out
    let qkvBuffer: MTLBuffer         // FP16 [qkvRows] in_proj_qkv out (DeltaNet)
    let zBuffer: MTLBuffer           // FP16 [valueDim] in_proj_z out (DeltaNet)
    let aBuffer: MTLBuffer           // FP16 [Hv] in_proj_a out (DeltaNet)
    let bBuffer: MTLBuffer           // FP16 [Hv] in_proj_b out (DeltaNet)
    let convOutBuffer: MTLBuffer     // FP16 [qkvRows] conv+SiLU out (raw, pre-L2)
    let dnQBuffer: MTLBuffer         // FP16 [Hk·dk] L2'd, scaled q (DeltaNet)
    let dnKBuffer: MTLBuffer         // FP16 [Hk·dk] L2'd k (DeltaNet)
    let betaBuffer: MTLBuffer        // FP32 [Hv] gates out
    let decayBuffer: MTLBuffer       // FP32 [Hv] gates out
    let coreOutBuffer: MTLBuffer     // FP16 [valueDim] DeltaNet scan out
    let gatedOutBuffer: MTLBuffer    // FP16 [valueDim] DeltaNet gated norm out
    let mixerBuffer: MTLBuffer       // FP16 [hiddenSize] out_proj/o_proj out (pre-residual)
    let moeXBuffer: MTLBuffer        // FP16 [hiddenSize] post-attention norm out
    let routerLogitsBuffer: MTLBuffer // FP16 [numExperts] router GEMV out
    let routerTopIdsBuffer: MTLBuffer // UInt32 [topK] top-k ids
    let routerTopWeightsBuffer: MTLBuffer // FP32 [topK] recalibrated top-k weights
    let expertActsBuffer: MTLBuffer  // FP16 [topK·moeInter] routed gate·up·SiLU
    let routedOutBuffer: MTLBuffer   // FP16 [topK·hiddenSize] routed down out
    /// Weighted sum of the routed expert outputs, Σ w · down (the golden
    /// `routed_out`; step 4).
    let routedSumBuffer: MTLBuffer   // FP16 [hiddenSize]
    /// Zero-filled inputs so `encodeResidual` can compute a bare weighted
    /// sum (step 4).
    let zeroHiddenBuffer: MTLBuffer  // FP16 [hiddenSize]
    let zeroScalarBuffer: MTLBuffer  // FP16 [1]
    /// C5 staging for the three resident shared-expert tensors (step 4).
    let sharedBlobBuffer: MTLBuffer
    let sharedActsBuffer: MTLBuffer  // FP16 [sharedInter] shared gate·up·SiLU
    let sharedOutBuffer: MTLBuffer   // FP16 [hiddenSize] shared down out
    let sharedGateBuffer: MTLBuffer  // FP16 [maxT] shared-expert gate scalar (one per row)
    let qgBuffer: MTLBuffer          // FP16 [qgRows] q_proj out (queries + gate)
    let qBuffer: MTLBuffer           // FP16 [queryDim] query (split, norm, RoPE in place)
    let gateBuffer: MTLBuffer        // FP16 [queryDim] attention output gate (split)
    let attnKBuffer: MTLBuffer       // FP16 [kvDim] key (split, norm, RoPE in place)
    let attnVBuffer: MTLBuffer       // FP16 [kvDim] value
    /// q and k after the per-head RMSNorm, then RoPE in place (step 5), so
    /// `qBuffer` / `attnKBuffer` keep the pre-norm values for the taps.
    let qNormedBuffer: MTLBuffer     // FP16 [queryDim]
    let kNormedBuffer: MTLBuffer     // FP16 [kvDim]
    /// Test-only copies of q and k after the per-head norm and before RoPE:
    /// the golden keys `q_pre_norm` / `k_pre_norm` hold exactly these
    /// (make_model_golden.py records `q_norm(q)` / `k_norm(k)`).
    let qPreRopeTapBuffer: MTLBuffer // FP16 [queryDim]
    let kPreRopeTapBuffer: MTLBuffer // FP16 [kvDim]
    let attnOutBuffer: MTLBuffer     // FP16 [queryDim] attention out (pre-gate)
    let gatedBuffer: MTLBuffer       // FP16 [queryDim] attention out · gate
    /// The layer whose shared expert currently fills `sharedBlobBuffer`
    /// (step 4); the blob is re-staged whenever the layer changes.
    var sharedBlobLayer: Int?

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

    /// Test-only (S19 steps 3–5): when set, the layer loop stops right after
    /// this layer, before the remaining layers, the final norm and the
    /// lm_head, so the tapped layer's work buffers survive for
    /// `tappedInternals`. `produce` still advances the cache.
    internal var debugStopAfterLayer: Int?

    /// Recorded hidden rows of the current generation, layout as above.
    internal private(set) var tappedHidden: MTLBuffer?
    /// Tokens recorded so far (one row per point each).
    internal private(set) var tappedHiddenCount = 0
    /// Recorded internals of `debugInternalsLayer`, keyed by C7 golden name.
    /// The values are the runner's own work buffers, so the tapped layer
    /// must be the last layer of the run or later layers overwrite them.
    internal private(set) var tappedInternals: [String: MTLBuffer] = [:]

    /// Test-only (S19 steps 3–5): the layer code records its work buffers
    /// under their C7 golden keys when it runs `debugInternalsLayer`. The
    /// entries merge, so the mixer and the MoE block of the same layer
    /// contribute their own keys.
    internal func recordInternals(layer: Int, _ entries: [String: MTLBuffer]) {
        guard layer == debugInternalsLayer else { return }
        for (key, buffer) in entries {
            tappedInternals[key] = buffer
        }
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
        // The `.control` variant serves the F16 GEMM path for every m (the
        // Apple10 variants are Gemma/Vision-only); map S20.
        self.mpp4 = MPPPrefillInt4QMM(context: context)
        // The 8-bit GEMM has a single F16 threadgroup path (no Apple10
        // variant); map S20 "GEMM prefill".
        self.mpp8 = MPPPrefillInt8QMM(context: context)

        // The work buffers are row-major with the largest prefill chunk
        // (S20): `produce` uses row 0, `prefillChunked` the first T rows.
        let maxT = PrefillRuntimeConfig.maxChunkTokens
        self.hiddenBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize * maxT, element: Float16.self, name: "hidden")
        self.normedBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize * maxT, element: Float16.self, name: "normed")
        self.xNormBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize * maxT, element: Float16.self, name: "xNorm")
        self.qkvBuffer = try makeWorkBuffer(device: context.device, elementCount: qkvRows * maxT, element: Float16.self, name: "qkv")
        self.zBuffer = try makeWorkBuffer(device: context.device, elementCount: valueDim * maxT, element: Float16.self, name: "z")
        self.aBuffer = try makeWorkBuffer(device: context.device, elementCount: hybrid.linearNumValueHeads * maxT, element: Float16.self, name: "a")
        self.bBuffer = try makeWorkBuffer(device: context.device, elementCount: hybrid.linearNumValueHeads * maxT, element: Float16.self, name: "b")
        self.convOutBuffer = try makeWorkBuffer(device: context.device, elementCount: qkvRows * maxT, element: Float16.self, name: "convOut")
        let dnQKDim = hybrid.linearNumKeyHeads * hybrid.linearKeyHeadDim
        self.dnQBuffer = try makeWorkBuffer(device: context.device, elementCount: dnQKDim * maxT, element: Float16.self, name: "dnQ")
        self.dnKBuffer = try makeWorkBuffer(device: context.device, elementCount: dnQKDim * maxT, element: Float16.self, name: "dnK")
        self.betaBuffer = try makeWorkBuffer(device: context.device, elementCount: hybrid.linearNumValueHeads * maxT, element: Float32.self, name: "beta")
        self.decayBuffer = try makeWorkBuffer(device: context.device, elementCount: hybrid.linearNumValueHeads * maxT, element: Float32.self, name: "decay")
        self.coreOutBuffer = try makeWorkBuffer(device: context.device, elementCount: valueDim * maxT, element: Float16.self, name: "coreOut")
        self.gatedOutBuffer = try makeWorkBuffer(device: context.device, elementCount: valueDim * maxT, element: Float16.self, name: "gatedOut")
        self.mixerBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize * maxT, element: Float16.self, name: "mixer")
        self.moeXBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize * maxT, element: Float16.self, name: "moeX")
        self.routerLogitsBuffer = try makeWorkBuffer(device: context.device, elementCount: cfg.numExperts * maxT, element: Float16.self, name: "routerLogits")
        self.routerTopIdsBuffer = try makeWorkBuffer(device: context.device, elementCount: cfg.topKExperts * maxT,
            element: UInt32.self, name: "routerTopIds")
        self.routerTopWeightsBuffer = try makeWorkBuffer(device: context.device, elementCount: cfg.topKExperts * maxT, element: Float32.self, name: "routerTopWeights")
        self.expertActsBuffer = try makeWorkBuffer(
            device: context.device, elementCount: cfg.topKExperts * cfg.moeIntermediateSize * maxT,
            element: Float16.self, name: "expertActs")
        self.routedOutBuffer = try makeWorkBuffer(
            device: context.device, elementCount: cfg.topKExperts * hiddenSize * maxT,
            element: Float16.self, name: "routedOut")
        self.routedSumBuffer = try makeWorkBuffer(
            device: context.device, elementCount: hiddenSize * maxT,
            element: Float16.self, name: "routedSum")
        self.zeroHiddenBuffer = try makeWorkBuffer(
            device: context.device, elementCount: hiddenSize * maxT,
            element: Float16.self, name: "zeroHidden")
        self.zeroScalarBuffer = try makeWorkBuffer(
            device: context.device, elementCount: 1,
            element: Float16.self, name: "zeroScalar")
        memset(self.zeroHiddenBuffer.contents(), 0, self.zeroHiddenBuffer.length)
        memset(self.zeroScalarBuffer.contents(), 0, self.zeroScalarBuffer.length)
        // The shared-expert weight blob is not per-token: one blob serves
        // every row of the chunk.
        self.sharedBlobBuffer = try makeWorkBuffer(
            device: context.device,
            elementCount: QwenMoE.expertBlobBytes(
                hidden: hiddenSize, inter: cfg.intermediateSize),
            element: UInt8.self, name: "sharedBlob")
        self.sharedActsBuffer = try makeWorkBuffer(device: context.device, elementCount: cfg.intermediateSize * maxT, element: Float16.self, name: "sharedActs")
        self.sharedOutBuffer = try makeWorkBuffer(device: context.device, elementCount: hiddenSize * maxT, element: Float16.self, name: "sharedOut")
        self.sharedGateBuffer = try makeWorkBuffer(device: context.device, elementCount: maxT, element: Float16.self, name: "sharedGate")
        self.qgBuffer = try makeWorkBuffer(device: context.device, elementCount: qgRows * maxT, element: Float16.self, name: "qg")
        self.qBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim * maxT, element: Float16.self, name: "q")
        self.gateBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim * maxT, element: Float16.self, name: "gate")
        self.attnKBuffer = try makeWorkBuffer(device: context.device, elementCount: kvDim * maxT, element: Float16.self, name: "attnK")
        self.attnVBuffer = try makeWorkBuffer(device: context.device, elementCount: kvDim * maxT, element: Float16.self, name: "attnV")
        self.qNormedBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim * maxT, element: Float16.self, name: "qNormed")
        self.kNormedBuffer = try makeWorkBuffer(device: context.device, elementCount: kvDim * maxT, element: Float16.self, name: "kNormed")
        self.qPreRopeTapBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim * maxT, element: Float16.self, name: "qPreRopeTap")
        self.kPreRopeTapBuffer = try makeWorkBuffer(device: context.device, elementCount: kvDim * maxT, element: Float16.self, name: "kPreRopeTap")
        self.attnOutBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim * maxT, element: Float16.self, name: "attnOut")
        self.gatedBuffer = try makeWorkBuffer(device: context.device, elementCount: queryDim * maxT, element: Float16.self, name: "gated")
    }

    // MARK: - LogitProducer

    public func reset() {
        cache.reset()
        chunkCommitState.reset()
        // The KV is back at position 0, so a prefill from 0 will overwrite
        // the ranges the snapshots address (D6).
        prefixSnapshots = []
        tappedHidden = nil
        tappedHiddenCount = 0
        tappedInternals = [:]
    }

    /// Runs one token at `position`: embedding → layers 0..39 per
    /// `fullAttentionLayerMask` → final RMSNorm → lm_head GEMV, leaving
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
        var commandBuffer = try makeCommandBuffer()
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

        // Layers 0..<cfg.numLayers per cfg.fullAttentionLayerMask (3.5):
        // a linear (Gated DeltaNet) or full-attention mixer, then the MoE
        // block. The residual stream stays in `hiddenBuffer` throughout.
        for L in 0..<cfg.numLayers {
            if cfg.fullAttentionLayerMask[L] == 0 {
                try runLinearLayer(L, in: commandBuffer)
            } else {
                try runAttentionLayer(L, position: position, in: commandBuffer)
            }
            try await runMoELayer(L, in: &commandBuffer)
            recordHiddenTap(.afterLayer(L), in: commandBuffer)
            if let stop = debugStopAfterLayer, L == stop {
                // Test-only: the tapped layer is the last executed, so skip
                // the remaining layers, the final norm and the lm_head.
                commandBuffer.commit()
                await commandBuffer.completed()
                finishHiddenTapRow()
                cache.advance(by: 1)
                return
            }
        }

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
        finishHiddenTapRow()
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

    /// D6 prefix reuse: restore the runner to the end of a previously
    /// prefilled prompt. Each finished `prefillChunked` saves one snapshot
    /// of the DeltaNet + conv state at its end (at most two, the oldest
    /// evicted). Rewinding to one of those positions puts the state back and
    /// truncates the KV tail there, so the caller can continue from it; any
    /// other target throws `QwenRewindError.rewindUnsupported` and the
    /// caller re-prefills from position 0.
    public func rewind(to position: Int) throws {
        guard let snapshot = prefixSnapshots.first(where: { $0.position == position }) else {
            throw QwenRewindError.rewindUnsupported(position: position)
        }
        try cache.restore(snapshot)
        // Legal after `restore`: the state position equals the target.
        try cache.truncate(to: position)
        // Snapshots ahead of the new tail address KV the truncation dropped,
        // so they are unusable until prefilled again.
        prefixSnapshots.removeAll { $0.position > position }
    }

    /// D6: copy the DeltaNet + conv state (~63 MiB) at the end of a chunked
    /// prefill. The KV is never copied; the snapshot addresses it in place
    /// and stays valid until that range is overwritten.
    func savePrefixSnapshot() {
        prefixSnapshots.append(cache.snapshot())
        if prefixSnapshots.count > 2 {
            prefixSnapshots.removeFirst(prefixSnapshots.count - 2)
        }
    }

    // MARK: - Layers

    /// One Gated DeltaNet (linear) layer L of plan 3.5: input RMSNorm, the
    /// 4-bit qkv/z/a/b projections, causal conv + SiLU with the 3-tap FP32
    /// state in the hybrid cache, per-head L2 on q and k, the 1/sqrt(keyDim)
    /// scale on q, the per-value-head beta/decay gates, the in-place FP32
    /// state scan, the gated per-head RMSNorm with silu(z), the 4-bit out
    /// projection and the residual add. One token per call (T = 1 decode;
    /// prefill v1 reuses the same path).
    private func runLinearLayer(_ L: Int, in commandBuffer: MTLCommandBuffer) throws {
        let h = hiddenBuffer
        let keyHeads = hybrid.linearNumKeyHeads
        let keyDim = hybrid.linearKeyHeadDim
        let valueHeads = hybrid.linearNumValueHeads
        let valueDim = valueHeads * hybrid.linearValueHeadDim
        let qkvChannels = 2 * keyHeads * keyDim + valueDim
        // Byte offsets of the q / k / v slices of the conv output (2 B each).
        let qkBytes = keyHeads * keyDim * MemoryLayout<Float16>.size
        let epsilon = Float(hybrid.rmsNormEps)

        // x = RMSNorm(h; input_layernorm); the checkpoint weights already
        // carry the +1 (normWeightsShifted), so the plain mode (3.5).
        let inputNorm = try model.inputNorm(layer: L)
        norms.encodeRow(
            commandBuffer: commandBuffer,
            x: h,
            weight: inputNorm.buffer, weightOffset: Int(inputNorm.offset),
            out: xNormBuffer,
            d: UInt32(hiddenSize), rows: 1, eps: epsilon, mode: .plain)

        // qkv / z / a / b projections (4-bit GEMV over the normalized row).
        let inProjQKV = try model.qwenInProjQKV(layer: L)
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: inProjQKV.buffer, weightsOffset: Int(inProjQKV.offset),
            scales: inProjQKV.buffer, scalesOffset: Int(inProjQKV.scaleOffset),
            biases: inProjQKV.buffer, biasesOffset: Int(inProjQKV.biasOffset),
            x: xNormBuffer, y: qkvBuffer,
            m: UInt32(qkvChannels), n: UInt32(hiddenSize))
        let inProjZ = try model.qwenInProjZ(layer: L)
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: inProjZ.buffer, weightsOffset: Int(inProjZ.offset),
            scales: inProjZ.buffer, scalesOffset: Int(inProjZ.scaleOffset),
            biases: inProjZ.buffer, biasesOffset: Int(inProjZ.biasOffset),
            x: xNormBuffer, y: zBuffer,
            m: UInt32(valueDim), n: UInt32(hiddenSize))
        let inProjA = try model.qwenInProjA(layer: L)
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: inProjA.buffer, weightsOffset: Int(inProjA.offset),
            scales: inProjA.buffer, scalesOffset: Int(inProjA.scaleOffset),
            biases: inProjA.buffer, biasesOffset: Int(inProjA.biasOffset),
            x: xNormBuffer, y: aBuffer,
            m: UInt32(valueHeads), n: UInt32(hiddenSize))
        let inProjB = try model.qwenInProjB(layer: L)
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: inProjB.buffer, weightsOffset: Int(inProjB.offset),
            scales: inProjB.buffer, scalesOffset: Int(inProjB.scaleOffset),
            biases: inProjB.buffer, biasesOffset: Int(inProjB.biasOffset),
            x: xNormBuffer, y: bBuffer,
            m: UInt32(valueHeads), n: UInt32(hiddenSize))

        // Causal depthwise conv (width 4) + SiLU over the qkv channels; the
        // 3-tap FP32 state lives in the hybrid cache and updates in place.
        let conv = try model.qwenConv1D(layer: L)
        guard let convOffset = cache.convStateByteOffset(layer: L) else {
            preconditionFailure("linear layer \(L) has no conv state slot")
        }
        deltaPrep.encodeConvSilu(
            commandBuffer: commandBuffer,
            qkv: qkvBuffer,
            convWeight: conv.buffer, weightOffset: Int(conv.offset),
            stateIn: cache.convStateBuffer, stateInOffset: convOffset,
            y: convOutBuffer,
            stateOut: cache.convStateBuffer, stateOutOffset: convOffset,
            channels: UInt32(qkvChannels), tokens: 1)

        // Per-head L2 on q and k (into dedicated buffers, so `convOutBuffer`
        // stays the raw conv+SiLU output for the `conv_out` tap), then the
        // 1/sqrt(keyDim) scale folded into q only (3.5).
        norms.encodeL2PerHead(
            commandBuffer: commandBuffer,
            x: convOutBuffer, xOffset: 0,
            out: dnQBuffer, outOffset: 0,
            headDim: UInt32(keyDim), numHeads: keyHeads, rows: 1, eps: epsilon)
        norms.encodeL2PerHead(
            commandBuffer: commandBuffer,
            x: convOutBuffer, xOffset: qkBytes,
            out: dnKBuffer, outOffset: 0,
            headDim: UInt32(keyDim), numHeads: keyHeads, rows: 1, eps: epsilon)
        deltaPrep.encodeScale(
            commandBuffer: commandBuffer,
            x: dnQBuffer, xOffset: 0, out: dnQBuffer, outOffset: 0,
            scale: 1.0 / sqrt(Float(keyDim)),
            n: keyHeads * keyDim)

        // Per-value-head gates: beta = sigmoid(b), decay = exp(g) with
        // g = -exp(A_log) * softplus(a + dt_bias).
        let aLog = try model.qwenALog(layer: L)
        let dtBias = try model.qwenDtBias(layer: L)
        deltaPrep.encodeGates(
            commandBuffer: commandBuffer,
            a: aBuffer, b: bBuffer,
            aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
            dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
            betaOut: betaBuffer, decayOut: decayBuffer,
            heads: UInt32(valueHeads), tokens: 1)

        // The recurrence: the FP32 [valueHeads][valueDim][keyDim] state of
        // this layer updates in place; value head j reads key head j / 2.
        guard let deltaOffset = cache.deltaStateByteOffset(layer: L) else {
            preconditionFailure("linear layer \(L) has no DeltaNet state slot")
        }
        deltaScan.encodeScan(
            commandBuffer: commandBuffer,
            q: dnQBuffer, qOffset: 0,
            k: dnKBuffer, kOffset: 0,
            v: convOutBuffer, vOffset: 2 * qkBytes,
            beta: betaBuffer, decay: decayBuffer,
            state: cache.deltaStateBuffer, stateOffset: deltaOffset,
            o: coreOutBuffer,
            tokens: 1,
            valueHeads: valueHeads, keyHeads: keyHeads,
            keyDim: keyDim, valueDim: hybrid.linearValueHeadDim,
            vRowStride: qkvChannels)

        // Gated per-head RMSNorm: RMSNorm(o; linear_attn.norm, plain w) *
        // silu(z), the shared weight across value heads (3.5).
        let deltaNorm = try model.qwenDeltaNetNorm(layer: L)
        norms.encodeGatedPerHead(
            commandBuffer: commandBuffer,
            o: coreOutBuffer, z: zBuffer,
            weight: deltaNorm.buffer, weightOffset: Int(deltaNorm.offset),
            out: gatedOutBuffer,
            headDim: UInt32(hybrid.linearValueHeadDim),
            numHeads: valueHeads, rows: 1, eps: epsilon)

        // Out projection + residual.
        let outProj = try model.qwenOutProj(layer: L)
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: outProj.buffer, weightsOffset: Int(outProj.offset),
            scales: outProj.buffer, scalesOffset: Int(outProj.scaleOffset),
            biases: outProj.buffer, biasesOffset: Int(outProj.biasOffset),
            x: gatedOutBuffer, y: mixerBuffer,
            m: UInt32(hiddenSize), n: UInt32(valueDim))
        deltaScan.encodeAdd(
            commandBuffer: commandBuffer,
            a: h, b: mixerBuffer, out: h, n: hiddenSize)

        // Test-only internals of this layer under the C7 golden keys. `v` is
        // a slice of the raw `convOutBuffer` (2 B/element) at 2*qkBytes; the
        // test applies that offset itself. `q` and `k` are the L2'd
        // (and, for q, scaled) values in their own buffers. `g` is not
        // produced by the gates kernel: the test recomputes it from the
        // tapped `a` and the A_log / dt_bias tensors (3.5). The state entry
        // is the whole cache buffer; the test applies the layer's byte
        // offset and transposes the [head][column][row] layout.
        if L == debugInternalsLayer {
            recordInternals(layer: L, [
                "x_norm": xNormBuffer,
                "qkv_raw": qkvBuffer,
                "z": zBuffer,
                "a": aBuffer,
                "b": bBuffer,
                "conv_out": convOutBuffer,
                "q": dnQBuffer,
                "k": dnKBuffer,
                "v": convOutBuffer,
                "beta": betaBuffer,
                "core_out": coreOutBuffer,
                "gated_out": gatedOutBuffer,
                "mixer_out": mixerBuffer,
                "state_heads0to3": cache.deltaStateBuffer,
            ])
        }
    }

    /// One full-attention layer L of plan 3.5, token at `position`: input
    /// RMSNorm, the 4-bit q (queries and output gate interleaved per head),
    /// k and v projections, the split of queries and gate, the per-head
    /// q/k RMSNorm, partial RoPE (first `partialRotaryFactor` of each head),
    /// the K/V write at `position` in the hybrid cache, causal GQA attention
    /// over positions [0, position], the sigmoid output gate, the 4-bit
    /// o_proj and the residual add.
    private func runAttentionLayer(_ L: Int, position: Int,
                                   in commandBuffer: MTLCommandBuffer) throws {
        let h = hiddenBuffer
        let queryHeads = cfg.numHeads
        let kvHeads = cfg.numKVHeads
        let headDim = cfg.fullHeadDim
        let epsilon = Float(hybrid.rmsNormEps)
        let rotatedPairs = Int((Double(headDim) * cfg.partialRotaryFactor).rounded()) / 2

        // x = RMSNorm(h; input_layernorm), plain mode (normWeightsShifted).
        let inputNorm = try model.inputNorm(layer: L)
        norms.encodeRow(
            commandBuffer: commandBuffer,
            x: h,
            weight: inputNorm.buffer, weightOffset: Int(inputNorm.offset),
            out: xNormBuffer,
            d: UInt32(hiddenSize), rows: 1, eps: epsilon, mode: .plain)

        // q_proj (queries + gate), k_proj, v_proj.
        let qProj = try model.qProj(layer: L)
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: qProj.buffer, weightsOffset: Int(qProj.offset),
            scales: qProj.buffer, scalesOffset: Int(qProj.scaleOffset),
            biases: qProj.buffer, biasesOffset: Int(qProj.biasOffset),
            x: xNormBuffer, y: qgBuffer,
            m: UInt32(qgRows), n: UInt32(hiddenSize))
        let kProj = try model.kProj(layer: L)
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: kProj.buffer, weightsOffset: Int(kProj.offset),
            scales: kProj.buffer, scalesOffset: Int(kProj.scaleOffset),
            biases: kProj.buffer, biasesOffset: Int(kProj.biasOffset),
            x: xNormBuffer, y: attnKBuffer,
            m: UInt32(kvDim), n: UInt32(hiddenSize))
        let vProj = try model.vProj(layer: L)
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: vProj.buffer, weightsOffset: Int(vProj.offset),
            scales: vProj.buffer, scalesOffset: Int(vProj.scaleOffset),
            biases: vProj.buffer, biasesOffset: Int(vProj.biasOffset),
            x: xNormBuffer, y: attnVBuffer,
            m: UInt32(kvDim), n: UInt32(hiddenSize))

        // Per head, q_proj rows are [query 256][gate 256]: split them.
        attention.encodeSplitQG(
            commandBuffer: commandBuffer,
            qg: qgBuffer, q: qBuffer, gate: gateBuffer,
            queryHeads: queryHeads, headDim: headDim, rows: 1)

        // Per-head RMSNorm of q and k (plain mode: the weights carry the +1).
        let qNorm = try model.qNorm(layer: L)
        norms.encodePerHead(
            commandBuffer: commandBuffer,
            x: qBuffer,
            weight: qNorm.buffer, weightOffset: Int(qNorm.offset),
            out: qNormedBuffer,
            headDim: UInt32(headDim), numHeads: queryHeads, rows: 1,
            eps: epsilon, mode: .plain)
        let kNorm = try model.kNorm(layer: L)
        norms.encodePerHead(
            commandBuffer: commandBuffer,
            x: attnKBuffer,
            weight: kNorm.buffer, weightOffset: Int(kNorm.offset),
            out: kNormedBuffer,
            headDim: UInt32(headDim), numHeads: kvHeads, rows: 1,
            eps: epsilon, mode: .plain)

        // Test-only: keep the normalized, not yet rotated q and k (golden
        // `q_pre_norm` / `k_pre_norm`) before RoPE rewrites them in place.
        if L == debugInternalsLayer, let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: qNormedBuffer, sourceOffset: 0,
                      to: qPreRopeTapBuffer, destinationOffset: 0,
                      size: qPreRopeTapBuffer.length)
            blit.copy(from: kNormedBuffer, sourceOffset: 0,
                      to: kPreRopeTapBuffer, destinationOffset: 0,
                      size: kPreRopeTapBuffer.length)
            blit.endEncoding()
        }

        // Partial RoPE in place, token at `position`.
        attention.encodeRopePartial(
            commandBuffer: commandBuffer,
            data: qNormedBuffer, positionBase: UInt32(position),
            headDim: headDim, numHeads: queryHeads, rows: 1,
            rotatedPairs: rotatedPairs, theta: Float(cfg.ropeTheta))
        attention.encodeRopePartial(
            commandBuffer: commandBuffer,
            data: kNormedBuffer, positionBase: UInt32(position),
            headDim: headDim, numHeads: kvHeads, rows: 1,
            rotatedPairs: rotatedPairs, theta: Float(cfg.ropeTheta))

        // K/V into this layer's cache slice at `position`, then causal GQA
        // attention over [0, position]. Strides in elements (D5 layout
        // [position][kv head][headDim]).
        guard let layerKV = cache.kvByteOffset(layer: L, position: 0) else {
            preconditionFailure("attention layer \(L) has no KV slot")
        }
        let headStride = headDim
        let posStride = kvHeads * headDim
        attention.encodeKVWrite(
            commandBuffer: commandBuffer,
            k: kNormedBuffer, v: attnVBuffer,
            kvK: cache.kBuffer, kvKOffset: layerKV,
            kvV: cache.vBuffer, kvVOffset: layerKV,
            rows: 1, kvHeads: kvHeads, headDim: headDim,
            kvStart: UInt32(position),
            posStride: posStride, headStride: headStride)
        attention.encodeAttention(
            commandBuffer: commandBuffer,
            q: qNormedBuffer,
            kvK: cache.kBuffer, kvKOffset: layerKV,
            kvV: cache.vBuffer, kvVOffset: layerKV,
            out: attnOutBuffer,
            queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim,
            queryTokens: 1, kvStart: UInt32(position),
            posStride: posStride, headStride: headStride,
            scale: 1.0 / sqrt(Float(headDim)))

        // Output gate: attn · sigmoid(gate), then o_proj and the residual.
        attention.encodeGate(
            commandBuffer: commandBuffer,
            attn: attnOutBuffer, gate: gateBuffer, out: gatedBuffer,
            count: queryDim)
        let oProj = try model.oProj(layer: L)
        gemv4.encode(
            commandBuffer: commandBuffer,
            weights: oProj.buffer, weightsOffset: Int(oProj.offset),
            scales: oProj.buffer, scalesOffset: Int(oProj.scaleOffset),
            biases: oProj.buffer, biasesOffset: Int(oProj.biasOffset),
            x: gatedBuffer, y: mixerBuffer,
            m: UInt32(hiddenSize), n: UInt32(queryDim))
        deltaScan.encodeAdd(
            commandBuffer: commandBuffer,
            a: h, b: mixerBuffer, out: h, n: hiddenSize)

        // Test-only internals of this layer under the C7 golden keys
        // (layer3_internals.json); `layer_out` comes from the MoE block.
        // `q_pre_norm` / `k_pre_norm` are the golden's names for q and k
        // after the per-head norm and before RoPE.
        if L == debugInternalsLayer {
            recordInternals(layer: L, [
                "x_norm": xNormBuffer,
                "q_pre_norm": qPreRopeTapBuffer,
                "gate": gateBuffer,
                "k_pre_norm": kPreRopeTapBuffer,
                "v": attnVBuffer,
                "q_rope": qNormedBuffer,
                "k_rope": kNormedBuffer,
                "attn_out": attnOutBuffer,
                "gated": gatedBuffer,
                "mixer_out": mixerBuffer,
            ])
        }
    }

    // MARK: - MoE block

    /// One MoE block (plan 3.5 "Blocco MoE") following the mixer of layer
    /// L, one token: the post-attention RMSNorm, the 8-bit router GEMV and
    /// the top-8 selection, the routed experts streamed from the layer file
    /// (C5), the shared expert with its scalar gate, and the residual sum.
    ///
    /// The top-8 ids live on the GPU after the router kernel, so the pass
    /// commits and waits once to read them back, streams the expert blobs,
    /// and continues on a fresh command buffer (`inout`).
    private func runMoELayer(_ L: Int, in commandBuffer: inout MTLCommandBuffer) async throws {
        let h = hiddenBuffer
        let hidden = hiddenSize
        let inter = cfg.moeIntermediateSize
        let slots = cfg.topKExperts
        let epsilon = Float(hybrid.rmsNormEps)

        // x = RMSNorm(h; post_attention_layernorm), plain mode
        // (normWeightsShifted); the golden `moe_x`.
        let postNorm = try model.postAttnNorm(layer: L)
        norms.encodeRow(
            commandBuffer: commandBuffer,
            x: h,
            weight: postNorm.buffer, weightOffset: Int(postNorm.offset),
            out: moeXBuffer,
            d: UInt32(hidden), rows: 1, eps: epsilon, mode: .plain)

        // Router: 8-bit GEMV over the 256 expert logits, then the kernel
        // top-8 with renormalized weights.
        let routerWeights = try model.qwenRouter(layer: L)
        gemv8.encode(
            commandBuffer: commandBuffer,
            weights: routerWeights.buffer, weightsOffset: Int(routerWeights.offset),
            scales: routerWeights.buffer, scalesOffset: Int(routerWeights.scaleOffset),
            biases: routerWeights.buffer, biasesOffset: Int(routerWeights.biasOffset),
            x: moeXBuffer, y: routerLogitsBuffer,
            m: UInt32(cfg.numExperts), n: UInt32(hidden))
        router.encodeTopK(
            commandBuffer: commandBuffer,
            logits: routerLogitsBuffer,
            indices: routerTopIdsBuffer,
            weights: routerTopWeightsBuffer,
            experts: cfg.numExperts, topK: slots)

        // The routed experts are read from the layer file (C5), so the ids
        // must be on the CPU first: commit, read them back, stream the
        // blobs. The golden (mlx-lm) lists the top-8 in ascending weight
        // order with ties by ascending expert id, while the router kernel
        // (S13) selects in descending order; the runner normalizes the
        // pairs to the golden order here. The MoE sum is order-independent,
        // so only the exposed ids/weights and the blob slot order change.
        commandBuffer.commit()
        await commandBuffer.completed()
        let idsRaw = UnsafeMutableRawBufferPointer(
            start: routerTopIdsBuffer.contents(),
            count: slots * MemoryLayout<UInt32>.size)
        let weightsRaw = UnsafeMutableRawBufferPointer(
            start: routerTopWeightsBuffer.contents(),
            count: slots * MemoryLayout<Float>.size)
        var pairs: [(id: Int, weight: Float)] = []
        for slot in 0..<slots {
            let id = idsRaw.load(fromByteOffset: slot * MemoryLayout<UInt32>.size,
                                 as: UInt32.self)
            let weight = weightsRaw.load(fromByteOffset: slot * MemoryLayout<Float>.size,
                                         as: Float.self)
            pairs.append((min(Int(id), cfg.numExperts - 1), weight))
        }
        pairs.sort { ($0.weight, $0.id) < ($1.weight, $1.id) }
        for (slot, pair) in pairs.enumerated() {
            idsRaw.storeBytes(of: UInt32(pair.id),
                              toByteOffset: slot * MemoryLayout<UInt32>.size,
                              as: UInt32.self)
            weightsRaw.storeBytes(of: pair.weight,
                                  toByteOffset: slot * MemoryLayout<Float>.size,
                                  as: Float.self)
        }
        let experts = pairs.map(\.id)
        let blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
        guard let routedArgBuffer = moe.makeBlobsArgumentBuffer(
            blobs: blobs.map(\.buffer), slots: slots)
        else {
            throw QwenRunnerError.argumentBufferFailed(name: "routedBlobs")
        }
        commandBuffer = try makeCommandBuffer()

        // C5 intra-blob offsets (routed and shared experts share the
        // layout); the routed slot buffers start at the blob beginning.
        let offsets = QwenMoE.expertOffsets(hidden: hidden, inter: inter)

        // Routed: silu(gate·x) ⊙ (up·x) per slot, then the down projection.
        moe.encodeGateUpSilu(
            commandBuffer: commandBuffer,
            blobsArgBuffer: routedArgBuffer, blobs: blobs.map(\.buffer),
            offsets: offsets,
            x: moeXBuffer, acts: expertActsBuffer,
            hidden: hidden, inter: inter, slots: slots)
        moe.encodeDown(
            commandBuffer: commandBuffer,
            blobsArgBuffer: routedArgBuffer, blobs: blobs.map(\.buffer),
            offsets: offsets,
            acts: expertActsBuffer, routedOut: routedOutBuffer,
            hidden: hidden, inter: inter, slots: slots)

        // Shared expert: the same two kernels as a one-slot pass over the
        // resident tensors staged into `sharedBlobBuffer` (per layer).
        if sharedBlobLayer != L {
            try stageSharedExpert(layer: L, in: commandBuffer)
            sharedBlobLayer = L
        }
        guard let sharedArgBuffer = moe.makeBlobsArgumentBuffer(
            blobs: [sharedBlobBuffer], slots: 1)
        else {
            throw QwenRunnerError.argumentBufferFailed(name: "sharedBlob")
        }
        moe.encodeGateUpSilu(
            commandBuffer: commandBuffer,
            blobsArgBuffer: sharedArgBuffer, blobs: [sharedBlobBuffer],
            offsets: offsets,
            x: moeXBuffer, acts: sharedActsBuffer,
            hidden: hidden, inter: inter, slots: 1)
        moe.encodeDown(
            commandBuffer: commandBuffer,
            blobsArgBuffer: sharedArgBuffer, blobs: [sharedBlobBuffer],
            offsets: offsets,
            acts: sharedActsBuffer, routedOut: sharedOutBuffer,
            hidden: hidden, inter: inter, slots: 1)

        // sg = sigmoid(W_sg · x): 8-bit GEMV (1×hidden) and the elementwise
        // sigmoid, in place on the FP16 scalar.
        let gateScalar = try model.qwenSharedExpertGateScalar(layer: L)
        gemv8.encode(
            commandBuffer: commandBuffer,
            weights: gateScalar.buffer, weightsOffset: Int(gateScalar.offset),
            scales: gateScalar.buffer, scalesOffset: Int(gateScalar.scaleOffset),
            biases: gateScalar.buffer, biasesOffset: Int(gateScalar.biasOffset),
            x: moeXBuffer, y: sharedGateBuffer,
            m: 1, n: UInt32(hidden))
        router.encodeSigmoid(
            commandBuffer: commandBuffer,
            x: sharedGateBuffer, out: sharedGateBuffer, count: 1)

        // Test-only: the routed contribution alone (the golden
        // `routed_out`): the residual kernel with zero residual and shared
        // inputs reduces to the weighted sum Σ w · down.
        if L == debugInternalsLayer {
            moe.encodeResidual(
                commandBuffer: commandBuffer,
                residual: zeroHiddenBuffer,
                routedOut: routedOutBuffer,
                sharedOut: zeroHiddenBuffer,
                routingW: routerTopWeightsBuffer,
                sg: zeroScalarBuffer,
                y: routedSumBuffer,
                hidden: hidden, slots: slots)
        }

        // h += Σ w · routed + sg · shared (the golden `layer_out`).
        moe.encodeResidual(
            commandBuffer: commandBuffer,
            residual: h,
            routedOut: routedOutBuffer,
            sharedOut: sharedOutBuffer,
            routingW: routerTopWeightsBuffer,
            sg: sharedGateBuffer,
            y: h,
            hidden: hidden, slots: slots)

        // Test-only MoE internals of the tapped layer (C7 golden keys).
        // `routed_out` is the weighted sum, not the per-slot projections;
        // `shared_out` is the pre-gate shared projection: the golden row
        // already carries the sg factor (a positive scalar, so the cosine
        // is unaffected). `layer_out` is `hiddenBuffer` after the residual.
        if L == debugInternalsLayer {
            recordInternals(layer: L, [
                "moe_x": moeXBuffer,
                "router_top_ids": routerTopIdsBuffer,
                "router_top_weights": routerTopWeightsBuffer,
                "routed_out": routedSumBuffer,
                "shared_out": sharedOutBuffer,
                "layer_out": h,
            ])
        }
    }

    /// C5 staging of layer L's three resident shared-expert tensors (gate,
    /// up, down; 4-bit affine) into `sharedBlobBuffer` at the blob's intra
    /// offsets. The blits are encoded on the command buffer, so they are
    /// ordered before the kernels that read the blob.
    /// Internal (not private) so the S20 chunked-prefill extension in
    /// `QwenHybridForwardRunner+Prefill.swift` reuses the C5 staging.
    internal func stageSharedExpert(layer L: Int, in commandBuffer: MTLCommandBuffer) throws {
        let offsets = QwenMoE.expertOffsets(
            hidden: hiddenSize, inter: cfg.moeIntermediateSize)
        let gate = try model.qwenSharedExpertGate(layer: L)
        let up = try model.qwenSharedExpertUp(layer: L)
        let down = try model.qwenSharedExpertDown(layer: L)
        let sections: [(view: TensorView, weight: UInt32,
                        scales: UInt32, biases: UInt32)] = [
            (gate, offsets.gateWOff, offsets.gateSOff, offsets.gateBOff),
            (up, offsets.upWOff, offsets.upSOff, offsets.upBOff),
            (down, offsets.downWOff, offsets.downSOff, offsets.downBOff),
        ]
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        for section in sections {
            blit.copy(from: section.view.buffer,
                      sourceOffset: Int(section.view.offset),
                      to: sharedBlobBuffer, destinationOffset: Int(section.weight),
                      size: Int(section.view.length))
            blit.copy(from: section.view.buffer,
                      sourceOffset: Int(section.view.scaleOffset),
                      to: sharedBlobBuffer, destinationOffset: Int(section.scales),
                      size: Int(section.view.scaleLength))
            blit.copy(from: section.view.buffer,
                      sourceOffset: Int(section.view.biasOffset),
                      to: sharedBlobBuffer, destinationOffset: Int(section.biases),
                      size: Int(section.view.biasLength))
        }
        blit.endEncoding()
    }

    /// A fresh command buffer on the shared queue.
    func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = ctx.queue.makeCommandBuffer() else {
            throw QwenRunnerError.commandBufferFailed
        }
        return commandBuffer
    }

    // MARK: - Taps

    /// GPU-side blit of the current row into the tap buffer so the recorded
    /// state is the one at `point`, whatever the later layers do to the work
    /// buffers. Runs while encoding; the test reads `tappedHidden` after the
    /// command buffer completes.
    /// `rows` consecutive rows (a `produce` token or a prefill chunk) are
    /// blit from the head of the source buffer, whose first `rows` rows hold
    /// them row-major.
    func recordHiddenTap(_ point: HiddenTapPoint, rows: Int = 1,
                         in commandBuffer: MTLCommandBuffer) {
        guard let tap = debugHiddenTap,
              let pointIndex = tap.points.firstIndex(of: point),
              tappedHiddenCount + rows <= tap.rows else { return }
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
                  size: rows * rowBytes)
        blit.endEncoding()
    }

    /// Closes the current rows' tap rows: every point wrote rows from
    /// `tappedHiddenCount`, the next call writes the following rows. Called
    /// once per `produce` (one row) and once per prefill span (`rows`), so
    /// taps with several points stay aligned.
    func finishHiddenTapRow(count: Int = 1) {
        guard let tap = debugHiddenTap,
              tappedHiddenCount + count <= tap.rows else { return }
        tappedHiddenCount += count
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
/// express. Internal (not private) so the S20 chunked-prefill extension in
/// `QwenHybridForwardRunner+Prefill.swift` can throw the same cases.
enum QwenRunnerError: Error, CustomStringConvertible {
    case bufferAllocationFailed(name: String, bytes: Int)
    case commandBufferFailed
    case argumentBufferFailed(name: String)

    var description: String {
        switch self {
        case .bufferAllocationFailed(let name, let bytes):
            return "Qwen runner work buffer \(name) allocation failed (\(bytes) B)"
        case .commandBufferFailed:
            return "Qwen runner command buffer allocation failed"
        case .argumentBufferFailed(let name):
            return "Qwen runner \(name) argument buffer allocation failed"
        }
    }
}
