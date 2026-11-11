import Foundation
import Metal

/// S20: chunked prefill for the Qwen hybrid runner.
///
/// `prefillChunked` runs the forward pass over `T` rows at a time, in the
/// `PrefillRuntimeConfig` chunk size: the embedding of the chunk's tokens
/// (this step) and, per `fullAttentionLayerMask`, the Gated DeltaNet mixer
/// or the gated full-attention mixer, each followed by its MoE block, then
/// the final RMSNorm and the lm_head of the last row (later steps). The
/// work buffers are row-major with `PrefillRuntimeConfig.maxChunkTokens`
/// rows; the first `T` rows carry the chunk. `produce` keeps using row 0,
/// so prompt and decode share the buffers.
extension QwenHybridForwardRunner: ChunkedPrefillRunner {

    /// Prefills `tokens` starting at `startPosition`, in chunks of
    /// `config.chunkTokens` rows. The KV tail must already be at
    /// `startPosition` (a fresh prefill at 0, or a continuation at the tail,
    /// D6). Each span commits, waits for the GPU and advances the cache
    /// before `onProgress(span.completedCount)`; the result seed is always
    /// `.logitsWritten` (Qwen never fuses greedy into prefill, S19),
    /// whatever `outputMode` asks.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        try chunkCommitState.requireClean(operation: "chunked prefill")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "Qwen chunked prefill requires a chunked PrefillRuntimeConfig")
        }
        guard startPosition >= 0 else {
            throw PrefillError.prefillCursorMismatch(
                "Qwen chunked prefill at negative startPosition \(startPosition)")
        }
        guard cache.position == startPosition else {
            throw PrefillError.prefillCursorMismatch(
                "Qwen chunked prefill at \(startPosition) but the cache is at \(cache.position)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw ModelError.contextExceedsModel(
                requested: startPosition + tokens.count, maximum: maxContext)
        }
        if tokens.isEmpty {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        precondition(tokens.allSatisfy { $0 >= 0 },
                     "token ids are non-negative")
        precondition(logits.length >= vocabSize * MemoryLayout<Float16>.size,
                     "logits buffer holds \(logits.length) bytes, need \(vocabSize * MemoryLayout<Float16>.size)")

        chunkCommitState.markDirty(startPosition: startPosition, tokenCount: tokens.count)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        try await PrefillSpanIteration.forEachSpan(spans) { index, span in
            try await runPrefillChunk(tokens, span: span,
                                      writeFinalHead: index == spans.count - 1,
                                      into: logits)
            onProgress(span.completedCount)
        }
        chunkCommitState.markCommitted()
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    /// Runs one chunk. The embedding of the chunk's `T` tokens into the
    /// first `T` rows of `hiddenBuffer`, then, per `fullAttentionLayerMask`,
    /// the Gated DeltaNet mixer on those rows (step 3) or the gated
    /// full-attention mixer (step 5), each followed by the MoE block
    /// (step 4), and the final RMSNorm with the lm_head of the last row
    /// into `logits` when `writeFinalHead` (step 6).
    private func runPrefillChunk(_ tokens: ArraySlice<Int32>,
                                 span: PrefillChunkSpan,
                                 writeFinalHead: Bool,
                                 into logits: MTLBuffer) async throws {
        let rows = span.tokenCount
        // Step 4 turns this back into a `var`: the MoE block closes and
        // reopens the command buffer around its CPU wait.
        let commandBuffer = try makeCommandBuffer()
        let rowBytes = hiddenSize * MemoryLayout<Float16>.size
        for t in 0..<rows {
            // Embedding: 4-bit affine lookup of the row, no post-embedding
            // scale (3.5), one pass per row.
            embedding.encode(
                commandBuffer: commandBuffer,
                table: model.embedding.buffer, tableOffset: Int(model.embedding.offset),
                scales: model.embedding.buffer, scalesOffset: Int(model.embedding.scaleOffset),
                biases: model.embedding.buffer, biasesOffset: Int(model.embedding.biasOffset),
                out: hiddenBuffer, outOffset: t * rowBytes,
                tokenId: UInt32(tokens[tokens.startIndex + span.tokenOffset + t]),
                d: UInt32(hiddenSize), outScale: 1.0)
        }
        recordHiddenTap(.afterEmbedding, rows: rows, in: commandBuffer)

        // Layers 0..<cfg.numLayers on the first `rows` rows (3.5): a linear
        // (Gated DeltaNet) or full-attention mixer, then the MoE block. The
        // residual stream stays in `hiddenBuffer`.
        for L in 0..<cfg.numLayers {
            if cfg.fullAttentionLayerMask[L] == 0 {
                try runPrefillLinearLayer(L, in: commandBuffer, rows: rows)
            } else {
                preconditionFailure("Qwen chunked prefill attention layers land in S20 step 5")
            }
            if let stop = debugStopAfterLayer, L == stop {
                // Test-only stop (step 3): the mixer of `stop` is the last
                // executed block, so skip the MoE block (step 4 inserts it
                // before this check, tap and all, like `produce`), the
                // remaining layers, the final norm and the lm_head.
                recordHiddenTap(.afterLayer(L), rows: rows, in: commandBuffer)
                commandBuffer.commit()
                await commandBuffer.completed()
                finishHiddenTapRow(count: rows)
                cache.advance(by: rows)
                return
            }
            preconditionFailure("Qwen chunked prefill MoE blocks land in S20 step 4")
        }
        preconditionFailure("Qwen chunked prefill final norm and lm_head land in S20 step 6")
    }

    /// One Gated DeltaNet (linear) layer L over the first `rows` rows of
    /// the work buffers (3.5): input RMSNorm, the 4-bit qkv/z/a/b
    /// projections as one MPP GEMM over the rows (map S20; per-row GEMV
    /// only if the GEMM reports no path), causal conv + SiLU with the
    /// 3-tap FP32 state in the hybrid cache, per-head L2 on q and k, the
    /// 1/sqrt(keyDim) scale on q, the per-value-head beta/decay gates, the
    /// in-place FP32 state scan over the rows, the gated per-head RMSNorm
    /// with silu(z), the 4-bit out projection and the residual add. Like
    /// the S19 layer it has no CPU wait, so it stays in the chunk's command
    /// buffer.
    private func runPrefillLinearLayer(_ L: Int, in commandBuffer: MTLCommandBuffer,
                                       rows: Int) throws {
        let keyHeads = hybrid.linearNumKeyHeads
        let keyDim = hybrid.linearKeyHeadDim
        let valueHeads = hybrid.linearNumValueHeads
        let valueDim = valueHeads * hybrid.linearValueHeadDim
        let qkvChannels = 2 * keyHeads * keyDim + valueDim
        // Byte offsets of the q / k / v slices of the conv output (2 B each).
        let qkBytes = keyHeads * keyDim * MemoryLayout<Float16>.size
        let epsilon = Float(hybrid.rmsNormEps)

        // x = RMSNorm(h; input_layernorm) over the rows; the checkpoint
        // weights already carry the +1 (normWeightsShifted), so the plain
        // mode (3.5).
        let inputNorm = try model.inputNorm(layer: L)
        norms.encodeRow(
            commandBuffer: commandBuffer,
            x: hiddenBuffer,
            weight: inputNorm.buffer, weightOffset: Int(inputNorm.offset),
            out: xNormBuffer,
            d: UInt32(hiddenSize), rows: rows, eps: epsilon, mode: .plain)

        // qkv / z / a / b projections over the rows.
        let inProjQKV = try model.qwenInProjQKV(layer: L)
        encodePrefillProjection(commandBuffer, view: inProjQKV,
                                x: xNormBuffer, y: qkvBuffer,
                                rows: rows, n: qkvChannels, k: hiddenSize)
        let inProjZ = try model.qwenInProjZ(layer: L)
        encodePrefillProjection(commandBuffer, view: inProjZ,
                                x: xNormBuffer, y: zBuffer,
                                rows: rows, n: valueDim, k: hiddenSize)
        let inProjA = try model.qwenInProjA(layer: L)
        encodePrefillProjection(commandBuffer, view: inProjA,
                                x: xNormBuffer, y: aBuffer,
                                rows: rows, n: valueHeads, k: hiddenSize)
        let inProjB = try model.qwenInProjB(layer: L)
        encodePrefillProjection(commandBuffer, view: inProjB,
                                x: xNormBuffer, y: bBuffer,
                                rows: rows, n: valueHeads, k: hiddenSize)

        // Causal depthwise conv (width 4) + SiLU over the rows; the 3-tap
        // FP32 state lives in the hybrid cache and updates in place.
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
            channels: UInt32(qkvChannels), tokens: rows)

        // Per-head L2 on q and k (into dedicated buffers, so `convOutBuffer`
        // stays the raw conv+SiLU output for the `conv_out` tap), then the
        // 1/sqrt(keyDim) scale folded into q only (3.5). One rows: 1 call
        // per row: the per-head kernel's xOffset is a base offset, so with
        // rows > 1 it would stride the wider conv rows by numHeads*headDim
        // instead of the full qkvChannels (map S20, step 3).
        let convRowBytes = qkvChannels * MemoryLayout<Float16>.size
        for t in 0..<rows {
            norms.encodeL2PerHead(
                commandBuffer: commandBuffer,
                x: convOutBuffer, xOffset: t * convRowBytes,
                out: dnQBuffer, outOffset: t * qkBytes,
                headDim: UInt32(keyDim), numHeads: keyHeads, rows: 1, eps: epsilon)
            norms.encodeL2PerHead(
                commandBuffer: commandBuffer,
                x: convOutBuffer, xOffset: t * convRowBytes + qkBytes,
                out: dnKBuffer, outOffset: t * qkBytes,
                headDim: UInt32(keyDim), numHeads: keyHeads, rows: 1, eps: epsilon)
        }
        deltaPrep.encodeScale(
            commandBuffer: commandBuffer,
            x: dnQBuffer, xOffset: 0, out: dnQBuffer, outOffset: 0,
            scale: 1.0 / sqrt(Float(keyDim)),
            n: keyHeads * keyDim * rows)

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
            heads: UInt32(valueHeads), tokens: rows)

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
            tokens: rows,
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
            numHeads: valueHeads, rows: rows, eps: epsilon)

        // Out projection over the rows + residual add.
        let outProj = try model.qwenOutProj(layer: L)
        encodePrefillProjection(commandBuffer, view: outProj,
                                x: gatedOutBuffer, y: mixerBuffer,
                                rows: rows, n: hiddenSize, k: valueDim)
        deltaScan.encodeAdd(
            commandBuffer: commandBuffer,
            a: hiddenBuffer, b: mixerBuffer, out: hiddenBuffer,
            n: hiddenSize * rows)

        // Test-only internals of this layer under the C7 golden keys, as in
        // the S19 layer: `v` is a slice of the raw `convOutBuffer` at
        // 2*qkBytes (per row); `g` and the MoE keys are not recorded here
        // (the gates kernel writes only beta/decay, and the MoE block is
        // step 4). The state entry is the whole cache buffer; the test
        // applies the layer's byte offset.
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

    /// One 4-bit projection over the chunk's `rows` (map S20, "GEMM
    /// prefill"): the MPP GEMM with `m = rows`, `n` outputs and `k` inputs
    /// on the row-major `x` / `y`. If the GEMM selects no path
    /// (`PathMetadata.path == .fallback`, nothing was encoded) the rows go
    /// through the per-row 4-bit GEMV instead.
    private func encodePrefillProjection(_ commandBuffer: MTLCommandBuffer,
                                         view: TensorView,
                                         x: MTLBuffer,
                                         y: MTLBuffer,
                                         rows: Int,
                                         n: Int,
                                         k: Int) {
        let metadata = mpp4.encode(
            commandBuffer: commandBuffer,
            weights: view.buffer, weightsOffset: Int(view.offset),
            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
            biases: view.buffer, biasesOffset: Int(view.biasOffset),
            x: x, xOffset: 0,
            y: y, yOffset: 0,
            m: rows, n: n, k: k)
        if metadata.path == .fallback {
            // The GEMM encoded nothing: the rows go through the per-row
            // 4-bit GEMV instead (the guard of the map; not expected).
            let inRowBytes = k * MemoryLayout<Float16>.size
            let outRowBytes = n * MemoryLayout<Float16>.size
            for t in 0..<rows {
                gemv4.encode(
                    commandBuffer: commandBuffer,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, xOffset: t * inRowBytes,
                    y: y, yOffset: t * outRowBytes,
                    m: UInt32(n), n: UInt32(k))
            }
        }
    }
}
