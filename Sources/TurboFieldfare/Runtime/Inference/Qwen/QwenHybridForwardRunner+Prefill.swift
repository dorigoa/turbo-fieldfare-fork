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

    /// Runs one chunk. This step is the skeleton: the embedding of the
    /// chunk's `T` tokens into the first `T` rows of `hiddenBuffer`, then
    /// the CPU wait. The later steps insert the mixers and MoE per
    /// `fullAttentionLayerMask` on the same rows, and the final RMSNorm and
    /// the lm_head of the last row when `writeFinalHead`, writing the
    /// logits.
    private func runPrefillChunk(_ tokens: ArraySlice<Int32>,
                                 span: PrefillChunkSpan,
                                 writeFinalHead: Bool,
                                 into logits: MTLBuffer) async throws {
        let rows = span.tokenCount
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
        // Later steps: the layers on the first `rows` rows (the tapped
        // layer stops the chunk when `debugStopAfterLayer` is set), the
        // final RMSNorm, and the lm_head into `logits` when `writeFinalHead`.
        commandBuffer.commit()
        await commandBuffer.completed()
        finishHiddenTapRow(count: rows)
        cache.advance(by: rows)
    }
}
