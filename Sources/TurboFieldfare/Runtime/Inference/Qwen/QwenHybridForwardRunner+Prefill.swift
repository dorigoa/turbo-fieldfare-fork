import Foundation
import Metal

/// S20: chunked prefill for the Qwen hybrid runner.
///
/// `prefillChunked` runs the forward pass over `T` rows at a time, in the
/// `PrefillRuntimeConfig` chunk size: the embedding of the chunk's tokens,
/// then, per `fullAttentionLayerMask`, the Gated DeltaNet mixer or the
/// gated full-attention mixer, each followed by its MoE block, then the
/// final RMSNorm and the lm_head of the last row. The
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
    /// into `logits` when `writeFinalHead`.
    private func runPrefillChunk(_ tokens: ArraySlice<Int32>,
                                 span: PrefillChunkSpan,
                                 writeFinalHead: Bool,
                                 into logits: MTLBuffer) async throws {
        let rows = span.tokenCount
        // `var`: the MoE block closes and reopens the command buffer
        // around its CPU wait (the ids must be read back before the routed
        // experts are streamed).
        var commandBuffer = try makeCommandBuffer()
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
                try runPrefillAttentionLayer(L, position: span.startPosition,
                                             in: commandBuffer, rows: rows)
            }
            try await runPrefillMoELayer(L, in: &commandBuffer, rows: rows)
            recordHiddenTap(.afterLayer(L), rows: rows, in: commandBuffer)
            if let stop = debugStopAfterLayer, L == stop {
                // Test-only stop (step 4, as in `produce`): this layer's
                // mixer and MoE are the last executed blocks, so skip the
                // remaining layers, the final norm and the lm_head.
                commandBuffer.commit()
                await commandBuffer.completed()
                finishHiddenTapRow(count: rows)
                cache.advance(by: rows)
                return
            }
        }
        // Final RMSNorm over the rows; the checkpoint weights already
        // carry the +1 (normWeightsShifted), so the plain `x̂ · w` mode
        // (3.5).
        norms.encodeRow(
            commandBuffer: commandBuffer,
            x: hiddenBuffer,
            weight: model.finalNorm.buffer, weightOffset: Int(model.finalNorm.offset),
            out: normedBuffer,
            d: UInt32(hiddenSize), rows: rows,
            eps: Float(hybrid.rmsNormEps), mode: .plain)
        recordHiddenTap(.afterFinalNorm, rows: rows, in: commandBuffer)

        // lm_head on the last row only, in the final span (as in `produce`
        // and the Gemma runner): FP16 logits into the caller's buffer, no
        // softcap. Qwen never fuses greedy into prefill, so the result
        // seed stays `.logitsWritten` whatever `outputMode` asked.
        if writeFinalHead {
            gemv4.encode(
                commandBuffer: commandBuffer,
                weights: model.qwenLMHead.buffer, weightsOffset: Int(model.qwenLMHead.offset),
                scales: model.qwenLMHead.buffer, scalesOffset: Int(model.qwenLMHead.scaleOffset),
                biases: model.qwenLMHead.buffer, biasesOffset: Int(model.qwenLMHead.biasOffset),
                x: normedBuffer, xOffset: (rows - 1) * rowBytes, y: logits,
                m: UInt32(vocabSize), n: UInt32(hiddenSize))
        }
        commandBuffer.commit()
        await commandBuffer.completed()
        finishHiddenTapRow(count: rows)
        cache.advance(by: rows)
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

    /// One gated full-attention layer L (3.5) over the first `rows` rows of
    /// the work buffers, the chunk's tokens at positions `position ..<
    /// position + rows`: input RMSNorm, the 4-bit q (queries and output
    /// gate interleaved per head), k and v projections as MPP GEMMs over
    /// the rows, the split of queries and gate, the per-head q/k RMSNorm,
    /// the partial RoPE with the chunk's positions, the K/V write of the
    /// rows at `position` in this layer's cache slice, the causal GQA
    /// attention (row t over [0, position + t]), the sigmoid output gate,
    /// the 4-bit o projection and the residual add. Like the S19 layer it
    /// has no CPU wait, so it stays in the chunk's command buffer.
    private func runPrefillAttentionLayer(_ L: Int, position: Int,
                                          in commandBuffer: MTLCommandBuffer,
                                          rows: Int) throws {
        let queryHeads = cfg.numHeads
        let kvHeads = cfg.numKVHeads
        let headDim = cfg.fullHeadDim
        let epsilon = Float(hybrid.rmsNormEps)
        let rotatedPairs = Int((Double(headDim) * cfg.partialRotaryFactor).rounded()) / 2

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

        // q_proj (queries + gate), k_proj, v_proj over the rows.
        let qProj = try model.qProj(layer: L)
        encodePrefillProjection(commandBuffer, view: qProj,
                                x: xNormBuffer, y: qgBuffer,
                                rows: rows, n: qgRows, k: hiddenSize)
        let kProj = try model.kProj(layer: L)
        encodePrefillProjection(commandBuffer, view: kProj,
                                x: xNormBuffer, y: attnKBuffer,
                                rows: rows, n: kvDim, k: hiddenSize)
        let vProj = try model.vProj(layer: L)
        encodePrefillProjection(commandBuffer, view: vProj,
                                x: xNormBuffer, y: attnVBuffer,
                                rows: rows, n: kvDim, k: hiddenSize)

        // Per head, q_proj rows are [query headDim][gate headDim]: split
        // them. `rows: rows` is safe (map S20 step 5): qwen_split_qg
        // strides its source rows by exactly queryHeads * 2 * headDim, the
        // full width of `qgBuffer`'s rows — the step-3 danger only hit the
        // wider convOut slices.
        attention.encodeSplitQG(
            commandBuffer: commandBuffer,
            qg: qgBuffer, q: qBuffer, gate: gateBuffer,
            queryHeads: queryHeads, headDim: headDim, rows: rows)

        // Per-head RMSNorm of q and k over the rows (plain mode: the
        // weights carry the +1). The buffers' rows are exactly
        // numHeads * headDim wide, so the base-offset kernel strides them
        // correctly.
        let qNorm = try model.qNorm(layer: L)
        norms.encodePerHead(
            commandBuffer: commandBuffer,
            x: qBuffer,
            weight: qNorm.buffer, weightOffset: Int(qNorm.offset),
            out: qNormedBuffer,
            headDim: UInt32(headDim), numHeads: queryHeads, rows: rows,
            eps: epsilon, mode: .plain)
        let kNorm = try model.kNorm(layer: L)
        norms.encodePerHead(
            commandBuffer: commandBuffer,
            x: attnKBuffer,
            weight: kNorm.buffer, weightOffset: Int(kNorm.offset),
            out: kNormedBuffer,
            headDim: UInt32(headDim), numHeads: kvHeads, rows: rows,
            eps: epsilon, mode: .plain)

        // Test-only: the normalized, not yet rotated q and k of the
        // chunk's rows (golden `q_pre_norm` / `k_pre_norm`) before RoPE
        // rewrites them in place.
        if L == debugInternalsLayer, let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: qNormedBuffer, sourceOffset: 0,
                      to: qPreRopeTapBuffer, destinationOffset: 0,
                      size: rows * queryDim * MemoryLayout<Float16>.size)
            blit.copy(from: kNormedBuffer, sourceOffset: 0,
                      to: kPreRopeTapBuffer, destinationOffset: 0,
                      size: rows * kvDim * MemoryLayout<Float16>.size)
            blit.endEncoding()
        }

        // Partial RoPE in place: row t sits at position + t.
        attention.encodeRopePartial(
            commandBuffer: commandBuffer,
            data: qNormedBuffer, positionBase: UInt32(position),
            headDim: headDim, numHeads: queryHeads, rows: rows,
            rotatedPairs: rotatedPairs, theta: Float(cfg.ropeTheta))
        attention.encodeRopePartial(
            commandBuffer: commandBuffer,
            data: kNormedBuffer, positionBase: UInt32(position),
            headDim: headDim, numHeads: kvHeads, rows: rows,
            rotatedPairs: rotatedPairs, theta: Float(cfg.ropeTheta))

        // The chunk's K/V into this layer's cache slice at `position`,
        // then causal GQA attention: row t attends [0, position + t].
        // Strides in elements (D5 layout [position][kv head][headDim]).
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
            rows: rows, kvHeads: kvHeads, headDim: headDim,
            kvStart: UInt32(position),
            posStride: posStride, headStride: headStride)
        attention.encodeAttention(
            commandBuffer: commandBuffer,
            q: qNormedBuffer,
            kvK: cache.kBuffer, kvKOffset: layerKV,
            kvV: cache.vBuffer, kvVOffset: layerKV,
            out: attnOutBuffer,
            queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim,
            queryTokens: rows, kvStart: UInt32(position),
            posStride: posStride, headStride: headStride,
            scale: 1.0 / sqrt(Float(headDim)))

        // Output gate over the rows, then o_proj and the residual add.
        attention.encodeGate(
            commandBuffer: commandBuffer,
            attn: attnOutBuffer, gate: gateBuffer, out: gatedBuffer,
            count: queryDim * rows)
        let oProj = try model.oProj(layer: L)
        encodePrefillProjection(commandBuffer, view: oProj,
                                x: gatedBuffer, y: mixerBuffer,
                                rows: rows, n: hiddenSize, k: queryDim)
        deltaScan.encodeAdd(
            commandBuffer: commandBuffer,
            a: hiddenBuffer, b: mixerBuffer, out: hiddenBuffer,
            n: hiddenSize * rows)

        // Test-only internals of this layer under the C7 golden keys
        // (layer3_internals.json), one row per chunk token, as in the S19
        // layer; the MoE block of the same layer adds its own keys.
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

    /// One MoE block (3.5 "Blocco MoE") over the first `rows` rows of the
    /// work buffers: post-attention RMSNorm, the 8-bit router GEMM over
    /// the rows (per-row GEMV if the GEMM reports no path), one top-K per
    /// row with a single CPU read of the rows' ids and weights, the routed
    /// experts of the chunk's id union fetched in groups of at most 16
    /// (the layer's pread slot count; map S20 step 4) and run per
    /// (row, slot) pair, then the shared expert and the residual per row.
    /// Closes and reopens the command buffer around the CPU wait and the
    /// expert streaming, like the S19 `runMoELayer`.
    private func runPrefillMoELayer(_ L: Int, in commandBuffer: inout MTLCommandBuffer,
                                    rows: Int) async throws {
        let hidden = hiddenSize
        let inter = cfg.moeIntermediateSize
        let slots = cfg.topKExperts
        let epsilon = Float(hybrid.rmsNormEps)
        let rowBytes = hidden * MemoryLayout<Float16>.size
        let interBytes = inter * MemoryLayout<Float16>.size

        // x = RMSNorm(h; post_attention_layernorm) over the rows (plain
        // mode, normWeightsShifted); the golden `moe_x`.
        let postNorm = try model.postAttnNorm(layer: L)
        norms.encodeRow(
            commandBuffer: commandBuffer,
            x: hiddenBuffer,
            weight: postNorm.buffer, weightOffset: Int(postNorm.offset),
            out: moeXBuffer,
            d: UInt32(hidden), rows: rows, eps: epsilon, mode: .plain)

        // Router: 8-bit GEMM over the rows (m = rows), the per-row GEMV if
        // the GEMM reports no path (e.g. weight offset not a multiple of
        // 4 B; map S20 "GEMM prefill").
        let routerWeights = try model.qwenRouter(layer: L)
        let routerMetadata = mpp8.encode(
            commandBuffer: commandBuffer,
            weights: routerWeights.buffer, weightsOffset: Int(routerWeights.offset),
            scales: routerWeights.buffer, scalesOffset: Int(routerWeights.scaleOffset),
            biases: routerWeights.buffer, biasesOffset: Int(routerWeights.biasOffset),
            x: moeXBuffer, xOffset: 0,
            y: routerLogitsBuffer, yOffset: 0,
            m: rows, n: cfg.numExperts, k: hidden)
        if routerMetadata.path == .fallback {
            let inRowBytes = hidden * MemoryLayout<Float16>.size
            let outRowBytes = cfg.numExperts * MemoryLayout<Float16>.size
            for t in 0..<rows {
                gemv8.encode(
                    commandBuffer: commandBuffer,
                    weights: routerWeights.buffer, weightsOffset: Int(routerWeights.offset),
                    scales: routerWeights.buffer, scalesOffset: Int(routerWeights.scaleOffset),
                    biases: routerWeights.buffer, biasesOffset: Int(routerWeights.biasOffset),
                    x: moeXBuffer, xOffset: t * inRowBytes,
                    y: routerLogitsBuffer, yOffset: t * outRowBytes,
                    m: UInt32(cfg.numExperts), n: UInt32(hidden))
            }
        }

        // One top-K per row: the ids and the recalibrated weights of row
        // t at t*slots in the [rows][slots] buffers.
        for t in 0..<rows {
            router.encodeTopK(
                commandBuffer: commandBuffer,
                logits: routerLogitsBuffer,
                logitsOffset: t * cfg.numExperts * MemoryLayout<Float16>.size,
                indices: routerTopIdsBuffer,
                indicesOffset: t * slots * MemoryLayout<UInt32>.size,
                weights: routerTopWeightsBuffer,
                weightsOffset: t * slots * MemoryLayout<Float>.size,
                experts: cfg.numExperts, topK: slots)
        }

        // The routed experts are read from the layer file (C5), so the ids
        // must be on the CPU first: commit, read the rows' ids and weights
        // back in one go. The golden (mlx-lm) lists the top-8 in ascending
        // weight order with ties by ascending expert id, while the router
        // kernel (S13) selects in descending order; the S19 runner
        // normalizes the pairs to the golden order per token. The MoE sum
        // is order-independent, so only the exposed ids/weights and the
        // slot-to-expert order change.
        commandBuffer.commit()
        await commandBuffer.completed()
        let idBytes = MemoryLayout<UInt32>.size
        let weightBytes = MemoryLayout<Float>.size
        let idsRaw = UnsafeMutableRawBufferPointer(
            start: routerTopIdsBuffer.contents(),
            count: rows * slots * idBytes)
        let weightsRaw = UnsafeMutableRawBufferPointer(
            start: routerTopWeightsBuffer.contents(),
            count: rows * slots * weightBytes)
        var union: [Int] = []
        var unionSeen = Set<Int>()
        for t in 0..<rows {
            var pairs: [(id: Int, weight: Float)] = []
            for slot in 0..<slots {
                let id = idsRaw.load(fromByteOffset: (t * slots + slot) * idBytes,
                                     as: UInt32.self)
                let weight = weightsRaw.load(fromByteOffset: (t * slots + slot) * weightBytes,
                                             as: Float.self)
                pairs.append((min(Int(id), cfg.numExperts - 1), weight))
            }
            pairs.sort { ($0.weight, $0.id) < ($1.weight, $1.id) }
            for (slot, pair) in pairs.enumerated() {
                idsRaw.storeBytes(of: UInt32(pair.id),
                                  toByteOffset: (t * slots + slot) * idBytes,
                                  as: UInt32.self)
                weightsRaw.storeBytes(of: pair.weight,
                                      toByteOffset: (t * slots + slot) * weightBytes,
                                      as: Float.self)
                if unionSeen.insert(pair.id).inserted {
                    union.append(pair.id)
                }
            }
        }
        union.sort()

        // C5 intra-blob offsets (routed and shared experts share the
        // layout); the routed slot buffers start at the blob beginning.
        let offsets = QwenMoE.expertOffsets(hidden: hidden, inter: inter)

        // Routed experts of the chunk's id union, in groups of at most 16:
        // the layer's pread slot count, beyond which the streamer
        // precondition-fails and a later fetch can overwrite the earlier
        // slots while the GPU still uses them (map S20 step 4). Each group
        // is fetched, run and awaited before the next one.
        var groupStart = 0
        while groupStart < union.count {
            let groupEnd = min(groupStart + 16, union.count)
            let group = Array(union[groupStart..<groupEnd])
            let blobs = try await model.fetchRoutedExperts(layer: L, experts: group)
            let blobByExpert = Dictionary(
                uniqueKeysWithValues: zip(group, blobs.map(\.buffer)))
            commandBuffer = try makeCommandBuffer()
            // Every (row, slot) pair whose expert is in this group: the
            // one-slot pass writes row t, slot j's activations and
            // projection at (t*slots+j).
            for t in 0..<rows {
                for slot in 0..<slots {
                    let id = idsRaw.load(fromByteOffset: (t * slots + slot) * idBytes,
                                         as: UInt32.self)
                    guard let blob = blobByExpert[Int(id)] else { continue }
                    guard let argBuffer = moe.makeBlobsArgumentBuffer(
                        blobs: [blob], slots: 1)
                    else {
                        throw QwenRunnerError.argumentBufferFailed(name: "routedBlob")
                    }
                    let actsOffset = (t * slots + slot) * interBytes
                    let routedOffset = (t * slots + slot) * rowBytes
                    moe.encodeGateUpSilu(
                        commandBuffer: commandBuffer,
                        blobsArgBuffer: argBuffer, blobs: [blob],
                        offsets: offsets,
                        x: moeXBuffer, xOffset: t * rowBytes,
                        acts: expertActsBuffer, actsOffset: actsOffset,
                        hidden: hidden, inter: inter, slots: 1)
                    moe.encodeDown(
                        commandBuffer: commandBuffer,
                        blobsArgBuffer: argBuffer, blobs: [blob],
                        offsets: offsets,
                        acts: expertActsBuffer, actsOffset: actsOffset,
                        routedOut: routedOutBuffer, routedOutOffset: routedOffset,
                        hidden: hidden, inter: inter, slots: 1)
                }
            }
            commandBuffer.commit()
            await commandBuffer.completed()
            groupStart = groupEnd
        }

        // Shared expert and residual, per row, on a fresh command buffer.
        commandBuffer = try makeCommandBuffer()
        // The shared expert runs as a one-slot pass over the resident
        // tensors staged into `sharedBlobBuffer` (per layer, like S19).
        if sharedBlobLayer != L {
            try stageSharedExpert(layer: L, in: commandBuffer)
            sharedBlobLayer = L
        }
        guard let sharedArgBuffer = moe.makeBlobsArgumentBuffer(
            blobs: [sharedBlobBuffer], slots: 1)
        else {
            throw QwenRunnerError.argumentBufferFailed(name: "sharedBlob")
        }
        let gateScalar = try model.qwenSharedExpertGateScalar(layer: L)
        let gateRowBytes = MemoryLayout<Float16>.size
        for t in 0..<rows {
            moe.encodeGateUpSilu(
                commandBuffer: commandBuffer,
                blobsArgBuffer: sharedArgBuffer, blobs: [sharedBlobBuffer],
                offsets: offsets,
                x: moeXBuffer, xOffset: t * rowBytes,
                acts: sharedActsBuffer, actsOffset: t * interBytes,
                hidden: hidden, inter: inter, slots: 1)
            moe.encodeDown(
                commandBuffer: commandBuffer,
                blobsArgBuffer: sharedArgBuffer, blobs: [sharedBlobBuffer],
                offsets: offsets,
                acts: sharedActsBuffer, actsOffset: t * interBytes,
                routedOut: sharedOutBuffer, routedOutOffset: t * rowBytes,
                hidden: hidden, inter: inter, slots: 1)
            // sg = sigmoid(W_sg · x): 8-bit GEMV (1×hidden) and the
            // elementwise sigmoid, in place on the row's FP16 scalar.
            gemv8.encode(
                commandBuffer: commandBuffer,
                weights: gateScalar.buffer, weightsOffset: Int(gateScalar.offset),
                scales: gateScalar.buffer, scalesOffset: Int(gateScalar.scaleOffset),
                biases: gateScalar.buffer, biasesOffset: Int(gateScalar.biasOffset),
                x: moeXBuffer, xOffset: t * rowBytes,
                y: sharedGateBuffer, yOffset: t * gateRowBytes,
                m: 1, n: UInt32(hidden))
            router.encodeSigmoid(
                commandBuffer: commandBuffer,
                x: sharedGateBuffer, xOffset: t * gateRowBytes,
                out: sharedGateBuffer, outOffset: t * gateRowBytes,
                count: 1)
            // h += Σ w · routed + sg · shared (the golden `layer_out`).
            moe.encodeResidual(
                commandBuffer: commandBuffer,
                residual: hiddenBuffer, residualOffset: t * rowBytes,
                routedOut: routedOutBuffer, routedOutOffset: t * slots * rowBytes,
                sharedOut: sharedOutBuffer, sharedOutOffset: t * rowBytes,
                routingW: routerTopWeightsBuffer,
                routingWOffset: t * slots * weightBytes,
                sg: sharedGateBuffer, sgOffset: t * gateRowBytes,
                y: hiddenBuffer, yOffset: t * rowBytes,
                hidden: hidden, slots: slots)
        }

        // Test-only: the routed contribution alone per row (the golden
        // `routed_out`): the residual kernel with zero residual and shared
        // inputs reduces to the weighted sum Σ w · down.
        if L == debugInternalsLayer {
            for t in 0..<rows {
                moe.encodeResidual(
                    commandBuffer: commandBuffer,
                    residual: zeroHiddenBuffer,
                    routedOut: routedOutBuffer, routedOutOffset: t * slots * rowBytes,
                    sharedOut: zeroHiddenBuffer,
                    routingW: routerTopWeightsBuffer,
                    routingWOffset: t * slots * weightBytes,
                    sg: zeroScalarBuffer,
                    y: routedSumBuffer, yOffset: t * rowBytes,
                    hidden: hidden, slots: slots)
            }
        }

        // Test-only MoE internals of the tapped layer (C7 golden keys), as
        // in the S19 layer; the buffers are row-major with `rows` rows
        // (row t at the t-th row of each). `routed_out` is the weighted
        // sum, not the per-slot projections; `shared_out` is the pre-gate
        // shared projection; `layer_out` is `hiddenBuffer` after the
        // residual.
        if L == debugInternalsLayer {
            recordInternals(layer: L, [
                "moe_x": moeXBuffer,
                "router_top_ids": routerTopIdsBuffer,
                "router_top_weights": routerTopWeightsBuffer,
                "routed_out": routedSumBuffer,
                "shared_out": sharedOutBuffer,
                "layer_out": hiddenBuffer,
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
