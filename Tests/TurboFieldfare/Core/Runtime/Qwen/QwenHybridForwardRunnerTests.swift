import CryptoKit
import Foundation
import Metal
import Testing
import TurboFieldfareValidationSupport

@testable import TurboFieldfare

/// S19: the Qwen token-at-a-time runner against the mlx-lm goldens of S8,
/// one test per step (2 embedding, 3 DeltaNet mixer, 4 MoE, 5 attention,
/// 6 full model, 7 greedy), with the C8 bounds.
@Suite("QwenHybridForwardRunner", .enabled(if: QwenTestResources.hasModel))
struct QwenHybridForwardRunnerTests {

    private struct Environment {
        let runner: QwenHybridForwardRunner
        let model: Model
        let device: MTLDevice
    }

    private func makeEnvironment(maxContext: Int = 2048) throws -> Environment {
        let directory = try #require(QwenTestResources.modelURL)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: directory, device: device)
        let context = try MetalContext()
        let runner = try QwenHybridForwardRunner(
            model: model, context: context, maxContext: maxContext)
        return Environment(runner: runner, model: model, device: device)
    }

    /// The prompt token ids from `prompts.json` (C7).
    private func promptIDs(_ name: String) throws -> [Int32] {
        let data = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/prompts.json"))
        let prompts = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
        let entry = try #require(prompts.first { ($0["name"] as? String) == name })
        let ids = try #require(entry["ids"] as? [Int])
        return ids.map(Int32.init)
    }

    /// Decode one C7 golden entry (`{"shape": [...], "f32b64": base64}`).
    /// The top level is cast loosely (`[String: Any]`) because the internals
    /// files mix C7 objects with raw arrays (e.g. `router_top_ids`), so a
    /// whole-object `as! [String: [String: Any]]` aborts on those.
    private func golden(_ file: String, _ key: String) throws -> (shape: [Int], f32: [Float]) {
        let data = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/\(file).json"))
        let object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let entry = try #require(object[key] as? [String: Any])
        let shape = try #require(entry["shape"] as? [Int])
        let base64 = try #require(entry["f32b64"] as? String)
        let bytes = try #require(Data(base64Encoded: base64))
        let count = bytes.count / MemoryLayout<Float>.size
        let floats = bytes.withUnsafeBytes { raw -> [Float] in
            Array(raw.bindMemory(to: Float.self)[0..<count])
        }
        return (shape, floats)
    }

    private func cosine(_ lhs: [Float], _ rhs: [Float]) -> Double {
        var dot = 0.0, normL = 0.0, normR = 0.0
        for index in 0..<lhs.count {
            dot += Double(lhs[index]) * Double(rhs[index])
            normL += Double(lhs[index]) * Double(lhs[index])
            normR += Double(rhs[index]) * Double(rhs[index])
        }
        return dot / (sqrt(normL) * sqrt(normR))
    }

    @Test("the p0 embeddings match the mlx-lm golden")
    func p0EmbeddingMatchesGolden() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let hiddenSize = env.model.config.hiddenSize
        let vocabSize = env.model.config.vocabSize
        let ids = try promptIDs("p0")
        #expect(ids.count == 5)

        runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
            points: [.afterEmbedding], rows: ids.count)
        #expect(runner.maxContext == 2048)

        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))
        for (position, id) in ids.enumerated() {
            try await runner.produce(token: id, position: position, into: logits)
        }
        #expect(runner.continuationPosition == ids.count)

        let tapped = try #require(runner.tappedHidden)
        #expect(runner.tappedHiddenCount == ids.count)
        let rowBytes = hiddenSize * MemoryLayout<Float16>.size
        let raw = UnsafeRawBufferPointer(start: tapped.contents(),
                                         count: ids.count * rowBytes)
        let (shape, goldenRows) = try golden("hidden_p0", "embedding")
        #expect(shape == [ids.count, hiddenSize])

        var minimumCosine = 2.0
        for row in 0..<ids.count {
            let actual = (0..<hiddenSize).map {
                Float(raw.load(fromByteOffset: row * rowBytes + $0 * MemoryLayout<Float16>.size,
                               as: Float16.self))
            }
            let expected = Array(
                goldenRows[(row * hiddenSize)..<(row + 1) * hiddenSize])
            minimumCosine = min(minimumCosine, cosine(actual, expected))
        }
        print("S19 step 2 p0 embedding minimum cosine: \(minimumCosine)")
        #expect(minimumCosine >= 0.9999)

        // reset() clears the KV position and the taps.
        runner.reset()
        #expect(runner.continuationPosition == 0)
        #expect(runner.tappedHidden == nil)
        #expect(runner.tappedHiddenCount == 0)
    }

    /// S20 step 2: the chunked prefill skeleton embeds all positions of p0
    /// into the first rows of `hiddenBuffer`; the tapped embeddings must
    /// match `hidden_p0.json → embedding` (cosine ≥ 0.9999) both in chunks
    /// of 2 (2+2+1 spans, the last partial) and in a single chunk, with
    /// `newPosition = 5` and the progress per completed span.
    @Test("the chunked p0 prefill embeddings match the golden")
    func p0ChunkedPrefillEmbeddingsMatchGolden() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let hiddenSize = env.model.config.hiddenSize
        let vocabSize = env.model.config.vocabSize
        let ids = try promptIDs("p0")
        #expect(ids.count == 5)
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))

        for chunkTokens in [2, 5] {
            runner.reset()
            runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
                points: [.afterEmbedding], rows: ids.count)
            // Step 3: each span now runs the layer-0 mixer before
            // stopping, so stop it there (the embedding tap is recorded
            // before the layer and is unaffected).
            runner.debugStopAfterLayer = 0
            var progress: [Int] = []
            let config = PrefillRuntimeConfig.defaultChunked
                .replacingChunkTokens(chunkTokens)
            let result = try await runner.prefillChunked(
                tokens: ids[...],
                startPosition: 0,
                outputMode: .logits,
                config: config,
                into: logits,
                onProgress: { progress.append($0) })
            #expect(result.newPosition == ids.count)
            #expect(result.seed == .logitsWritten)
            #expect(progress == (chunkTokens == 2 ? [2, 4, 5] : [5]))
            #expect(runner.continuationPosition == ids.count)
            #expect(!runner.chunkCommitState.isDirty)

            let tapped = try #require(runner.tappedHidden)
            #expect(runner.tappedHiddenCount == ids.count)
            let rowBytes = hiddenSize * MemoryLayout<Float16>.size
            let raw = UnsafeRawBufferPointer(start: tapped.contents(),
                                             count: ids.count * rowBytes)
            let (shape, goldenRows) = try golden("hidden_p0", "embedding")
            #expect(shape == [ids.count, hiddenSize])

            var minimumCosine = 2.0
            for row in 0..<ids.count {
                let actual = (0..<hiddenSize).map {
                    Float(raw.load(fromByteOffset: row * rowBytes + $0 * MemoryLayout<Float16>.size,
                                   as: Float16.self))
                }
                let expected = Array(
                    goldenRows[(row * hiddenSize)..<(row + 1) * hiddenSize])
                minimumCosine = min(minimumCosine, cosine(actual, expected))
            }
            print("S20 step 2 p0 chunked embedding (chunk \(chunkTokens)) minimum cosine: \(minimumCosine)")
            #expect(minimumCosine >= 0.9999)
        }

        // An empty slice returns the start position without touching the
        // cache.
        runner.reset()
        let empty = try await runner.prefillChunked(
            tokens: ids[..<0],
            startPosition: 0,
            outputMode: .logits,
            config: PrefillRuntimeConfig.defaultChunked,
            into: logits) { _ in }
        #expect(empty.newPosition == 0)
        #expect(empty.seed == .logitsWritten)
        #expect(runner.continuationPosition == 0)
    }

    /// S20 step 3: the chunked prefill runs the layer-0 Gated DeltaNet
    /// mixer over the chunk's rows. The `mixer_out` of every position of
    /// p0, in chunks of 2 (spans 2+2+1, one call per span so the mixer
    /// rows of each span can be read before the next span overwrites
    /// them), must match `layer0_internals.json → mixer_out` (cosine
    /// ≥ 0.995), and the DeltaNet and conv states at the end of the
    /// prompt must match the token-at-a-time `produce` run of S19
    /// (cosine ≥ 0.9999; the relative errors go to the report).
    @Test("the chunked layer-0 DeltaNet mixer matches the golden and the decode states")
    func chunkedLayer0DeltaNetMatchesGoldenAndDecodeStates() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let model = env.model
        let hybrid = try #require(model.qwenHybridConfig)
        let hiddenSize = model.config.hiddenSize
        let vocabSize = model.config.vocabSize
        let keyHeads = hybrid.linearNumKeyHeads
        let keyDim = hybrid.linearKeyHeadDim
        let valueHeads = hybrid.linearNumValueHeads
        let valueDim = valueHeads * hybrid.linearValueHeadDim
        let qkvChannels = 2 * keyHeads * keyDim + valueDim
        let ids = try promptIDs("p0")
        #expect(ids.count == 5)
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))

        // Reference: the S19 token-at-a-time `produce`, stopped after the
        // layer-0 mixer (the layer-0 states are the same in the full run).
        runner.reset()
        runner.debugStopAfterLayer = 0
        for (position, id) in ids.enumerated() {
            try await runner.produce(token: id, position: position, into: logits)
        }
        #expect(runner.continuationPosition == ids.count)
        guard let deltaOffset = runner.cache.deltaStateByteOffset(layer: 0),
              let convOffset = runner.cache.convStateByteOffset(layer: 0) else {
            Issue.record("no layer-0 state offsets"); return
        }
        // Per layer: [value head][column = value head dim][row = key dim].
        let deltaStateCount = valueHeads * hybrid.linearValueHeadDim * keyDim
        let decodeDelta = Self.readFP32(runner.cache.deltaStateBuffer, byteOffset: deltaOffset,
                                        count: deltaStateCount)
        let decodeConv = Self.readFP32(runner.cache.convStateBuffer, byteOffset: convOffset,
                                       count: 3 * qkvChannels)

        // Chunked: the same prompt in chunks of 2, one call per span
        // (2, 2, 1); the layer-0 mixer is the last executed block of each
        // call, so its rows survive in the work buffers to be read.
        runner.reset()
        runner.debugInternalsLayer = 0
        runner.debugStopAfterLayer = 0
        var progress: [Int] = []
        let config = PrefillRuntimeConfig.defaultChunked.replacingChunkTokens(2)
        let (shape, goldenMixer) = try golden("layer0_internals", "mixer_out")
        #expect(shape == [ids.count, hiddenSize])
        let rowBytes = hiddenSize * MemoryLayout<Float16>.size
        var minMixerCosine = 2.0
        var tappedRows = 0
        var chunkStart = 0
        while chunkStart < ids.count {
            let chunkEnd = min(chunkStart + 2, ids.count)
            let result = try await runner.prefillChunked(
                tokens: ids[chunkStart..<chunkEnd],
                startPosition: chunkStart,
                outputMode: .logits,
                config: config,
                into: logits,
                onProgress: { progress.append($0) })
            #expect(result.newPosition == chunkEnd)
            #expect(result.seed == .logitsWritten)
            #expect(runner.continuationPosition == chunkEnd)
            guard let mixerBuffer = runner.tappedInternals["mixer_out"] else {
                Issue.record("no mixer_out tapped after span \(chunkStart)..<\(chunkEnd)"); return
            }
            for t in 0..<(chunkEnd - chunkStart) {
                let row = Self.readFP16(mixerBuffer, byteOffset: t * rowBytes,
                                        count: hiddenSize)
                let position = tappedRows + t
                let expected = Array(
                    goldenMixer[(position * hiddenSize)..<(position + 1) * hiddenSize])
                minMixerCosine = min(minMixerCosine, cosine(row, expected))
            }
            tappedRows += chunkEnd - chunkStart
            chunkStart = chunkEnd
        }
        #expect(tappedRows == ids.count)
        // completedCount is per call (tokens done inside that call), so
        // the three one-span calls report 2, 2, 1.
        #expect(progress == [2, 2, 1])
        #expect(!runner.chunkCommitState.isDirty)
        print("S20 step 3 p0 chunked mixer_out minimum cosine: \(minMixerCosine)")
        #expect(minMixerCosine >= 0.995,
                "chunked layer 0 mixer_out minimum cosine \(minMixerCosine) < 0.995 (C8)")

        // The layer-0 states at the end of the prompt (cache layout
        // [head][column][row] for the DeltaNet, [tap][channel] for the
        // conv) against the token-at-a-time decode.
        let chunkedDelta = Self.readFP32(runner.cache.deltaStateBuffer, byteOffset: deltaOffset,
                                         count: deltaStateCount)
        let chunkedConv = Self.readFP32(runner.cache.convStateBuffer, byteOffset: convOffset,
                                        count: 3 * qkvChannels)
        let deltaCosine = cosine(decodeDelta, chunkedDelta)
        let deltaRel = RelError.compute(actual: chunkedDelta, reference: decodeDelta)
        let convCosine = cosine(decodeConv, chunkedConv)
        let convRel = RelError.compute(actual: chunkedConv, reference: decodeConv)
        print("S20 step 3 layer 0 DeltaNet state cosine \(deltaCosine), relative error \(deltaRel)")
        print("S20 step 3 layer 0 conv state cosine \(convCosine), relative error \(convRel)")
        #expect(deltaCosine >= 0.9999,
                "layer 0 DeltaNet state cosine \(deltaCosine) < 0.9999")
        #expect(convCosine >= 0.9999,
                "layer 0 conv state cosine \(convCosine) < 0.9999")
    }

    /// S20 step 4: the chunked prefill runs the MoE block over the
    /// chunk's rows. The hidden state after the full layer 0 (mixer +
    /// MoE + residual) of every position of p0, in chunks of 2 (one call
    /// per span, 2+2+1, so each span's rows survive in the work buffers)
    /// and in a single chunk, must match `hidden_p0.json → layer0`
    /// (cosine ≥ 0.995, C8); the chunks-of-2 rows must also match the
    /// token-at-a-time `produce` run (cosine ≥ 0.9999) and the tapped MoE
    /// internals must match `layer0_internals.json` per position (cosine
    /// ≥ 0.995).
    @Test("the chunked layer-0 MoE block matches the golden and the decode")
    func chunkedLayer0MoEMatchesGoldenAndDecode() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let hiddenSize = env.model.config.hiddenSize
        let vocabSize = env.model.config.vocabSize
        let slots = env.model.config.topKExperts
        let ids = try promptIDs("p0")
        #expect(ids.count == 5)
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))
        let rowBytes = hiddenSize * MemoryLayout<Float16>.size

        // Reference: the S19 token-at-a-time `produce`, stopped after the
        // full layer 0 (in `produce` the stop is after the MoE and the tap).
        runner.reset()
        runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
            points: [.afterLayer(0)], rows: ids.count)
        runner.debugStopAfterLayer = 0
        for (position, id) in ids.enumerated() {
            try await runner.produce(token: id, position: position, into: logits)
        }
        #expect(runner.continuationPosition == ids.count)
        guard let decodeTapped = runner.tappedHidden else {
            Issue.record("no decode hidden tap"); return
        }
        let decodeRows = (0..<ids.count).map {
            Self.readFP16(decodeTapped, byteOffset: $0 * rowBytes, count: hiddenSize)
        }

        let (layer0Shape, layer0Golden) = try golden("hidden_p0", "layer0")
        #expect(layer0Shape == [ids.count, hiddenSize])
        let (moeXShape, moeXGolden) = try golden("layer0_internals", "moe_x")
        let (routedOutShape, routedOutGolden) = try golden("layer0_internals", "routed_out")
        let (sharedOutShape, sharedOutGolden) = try golden("layer0_internals", "shared_out")
        let (weightsShape, weightsGolden) = try golden("layer0_internals", "router_top_weights")
        #expect(moeXShape == [ids.count, hiddenSize])
        #expect(routedOutShape == [ids.count, hiddenSize])
        #expect(sharedOutShape == [ids.count, hiddenSize])
        #expect(weightsShape == [ids.count, slots])

        for chunkTokens in [2, 5] {
            runner.reset()
            runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
                points: [.afterLayer(0)], rows: ids.count)
            runner.debugInternalsLayer = 0
            runner.debugStopAfterLayer = 0
            var progress: [Int] = []
            let config = PrefillRuntimeConfig.defaultChunked
                .replacingChunkTokens(chunkTokens)
            var minHiddenCosine = 2.0
            var minDecodeCosine = 2.0
            var minInternalCosine = 2.0
            var tappedRows = 0
            var chunkStart = 0
            while chunkStart < ids.count {
                let chunkEnd = min(chunkStart + chunkTokens, ids.count)
                let result = try await runner.prefillChunked(
                    tokens: ids[chunkStart..<chunkEnd],
                    startPosition: chunkStart,
                    outputMode: .logits,
                    config: config,
                    into: logits,
                    onProgress: { progress.append($0) })
                #expect(result.newPosition == chunkEnd)
                #expect(result.seed == .logitsWritten)
                #expect(runner.continuationPosition == chunkEnd)
                guard let tapped = runner.tappedHidden else {
                    Issue.record("no hidden tap after span \(chunkStart)..<\(chunkEnd)"); return
                }
                for t in 0..<(chunkEnd - chunkStart) {
                    let position = tappedRows + t
                    let row = Self.readFP16(tapped, byteOffset: position * rowBytes,
                                            count: hiddenSize)
                    let expected = Array(
                        layer0Golden[(position * hiddenSize)..<(position + 1) * hiddenSize])
                    minHiddenCosine = min(minHiddenCosine, cosine(row, expected))
                    if chunkTokens == 2 {
                        minDecodeCosine = min(minDecodeCosine,
                                              cosine(row, decodeRows[position]))
                        let internals: [(String, [Int], [Float])] = [
                            ("moe_x", moeXShape, moeXGolden),
                            ("routed_out", routedOutShape, routedOutGolden),
                            ("shared_out", sharedOutShape, sharedOutGolden),
                        ]
                        for (key, shape, goldenRows) in internals {
                            guard let buffer = runner.tappedInternals[key] else {
                                Issue.record("no \(key) tapped after span \(chunkStart)..<\(chunkEnd)"); return
                            }
                            let rowElements = goldenRows.count / shape[0]
                            let internalRow = Self.readFP16(buffer, byteOffset: t * rowBytes,
                                                            count: hiddenSize)
                            let expectedInternal = Array(goldenRows[
                                (position * rowElements)..<(position + 1) * rowElements])
                            minInternalCosine = min(minInternalCosine,
                                                    cosine(internalRow, expectedInternal))
                        }
                        guard let weightsBuffer = runner.tappedInternals["router_top_weights"] else {
                            Issue.record("no router_top_weights tapped after span \(chunkStart)..<\(chunkEnd)"); return
                        }
                        let weights = Self.readFP32(weightsBuffer,
                                                    byteOffset: t * slots * MemoryLayout<Float>.size,
                                                    count: slots)
                        let expectedWeights = Array(
                            weightsGolden[(position * slots)..<(position + 1) * slots])
                        minInternalCosine = min(minInternalCosine,
                                                cosine(weights, expectedWeights))
                    }
                }
                tappedRows += chunkEnd - chunkStart
                chunkStart = chunkEnd
            }
            #expect(tappedRows == ids.count)
            // completedCount is per call (tokens done inside that call),
            // so the one-span-per-call runs report 2, 2, 1.
            #expect(progress == (chunkTokens == 2 ? [2, 2, 1] : [5]))
            #expect(!runner.chunkCommitState.isDirty)
            print("S20 step 4 p0 chunked layer 0 hidden (chunk \(chunkTokens)) minimum cosine: \(minHiddenCosine)")
            #expect(minHiddenCosine >= 0.995,
                    "chunked layer 0 hidden minimum cosine \(minHiddenCosine) < 0.995 (C8)")
            if chunkTokens == 2 {
                print("S20 step 4 p0 chunked layer 0 hidden vs decode minimum cosine: \(minDecodeCosine)")
                #expect(minDecodeCosine >= 0.9999,
                        "chunked layer 0 hidden vs decode minimum cosine \(minDecodeCosine) < 0.9999")
                print("S20 step 4 p0 chunked MoE internals minimum cosine: \(minInternalCosine)")
                #expect(minInternalCosine >= 0.995,
                        "chunked MoE internals minimum cosine \(minInternalCosine) < 0.995 (C8)")
            }
        }
    }

    /// S20 step 5: the chunked prefill runs the gated full-attention layer
    /// (layer 3) over the chunk's rows: layers 0–3 of p0 in chunks of 2
    /// (one call per span, 2+2+1, so each span's rows survive in the work
    /// buffers) and in a single chunk. The hidden state after the full
    /// layer 3 (attention + MoE + residual) of every position must match
    /// `hidden_p0.json → layer3` (cosine ≥ 0.995, C8); the chunks-of-2
    /// rows must also match the token-at-a-time `produce` run (cosine
    /// ≥ 0.9999) and the tapped layer-3 internals must match
    /// `layer3_internals.json` per position (cosine ≥ 0.995).
    @Test("the chunked layer-3 attention block matches the golden and the decode")
    func chunkedLayer3AttentionMatchesGoldenAndDecode() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let config = env.model.config
        let hiddenSize = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numKVHeads * config.fullHeadDim
        let ids = try promptIDs("p0")
        #expect(ids.count == 5)
        #expect(config.fullAttentionLayerMask[3] == 1)
        let logits = try #require(env.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        let rowBytes = hiddenSize * MemoryLayout<Float16>.size

        // Reference: the S19 token-at-a-time `produce`, stopped after the
        // full layer 3 (in `produce` the stop is after the MoE and the
        // tap).
        runner.reset()
        runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
            points: [.afterLayer(3)], rows: ids.count)
        runner.debugStopAfterLayer = 3
        for (position, id) in ids.enumerated() {
            try await runner.produce(token: id, position: position, into: logits)
        }
        #expect(runner.continuationPosition == ids.count)
        guard let decodeTapped = runner.tappedHidden else {
            Issue.record("no decode hidden tap"); return
        }
        let decodeRows = (0..<ids.count).map {
            Self.readFP16(decodeTapped, byteOffset: $0 * rowBytes, count: hiddenSize)
        }

        let (layer3Shape, layer3Golden) = try golden("hidden_p0", "layer3")
        #expect(layer3Shape == [ids.count, hiddenSize])

        // FP16 row lengths of the tapped layer-3 work buffers, by golden
        // key (the same counts as the S19 step-5 test).
        let counts: [String: Int] = [
            "x_norm": hiddenSize,
            "q_pre_norm": queryDim,
            "gate": queryDim,
            "k_pre_norm": kvDim,
            "v": kvDim,
            "q_rope": queryDim,
            "k_rope": kvDim,
            "attn_out": queryDim,
            "gated": queryDim,
            "mixer_out": hiddenSize,
            "layer_out": hiddenSize,
        ]

        for chunkTokens in [2, 5] {
            runner.reset()
            runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
                points: [.afterLayer(3)], rows: ids.count)
            runner.debugInternalsLayer = 3
            runner.debugStopAfterLayer = 3
            var progress: [Int] = []
            let prefillConfig = PrefillRuntimeConfig.defaultChunked
                .replacingChunkTokens(chunkTokens)
            var minHiddenCosine = 2.0
            var minDecodeCosine = 2.0
            var minInternalCosine = 2.0
            var tappedRows = 0
            var chunkStart = 0
            while chunkStart < ids.count {
                let chunkEnd = min(chunkStart + chunkTokens, ids.count)
                let result = try await runner.prefillChunked(
                    tokens: ids[chunkStart..<chunkEnd],
                    startPosition: chunkStart,
                    outputMode: .logits,
                    config: prefillConfig,
                    into: logits,
                    onProgress: { progress.append($0) })
                #expect(result.newPosition == chunkEnd)
                #expect(result.seed == .logitsWritten)
                #expect(runner.continuationPosition == chunkEnd)
                guard let tapped = runner.tappedHidden else {
                    Issue.record("no hidden tap after span \(chunkStart)..<\(chunkEnd)"); return
                }
                for t in 0..<(chunkEnd - chunkStart) {
                    let position = tappedRows + t
                    let row = Self.readFP16(tapped, byteOffset: position * rowBytes,
                                            count: hiddenSize)
                    let expected = Array(
                        layer3Golden[(position * hiddenSize)..<(position + 1) * hiddenSize])
                    minHiddenCosine = min(minHiddenCosine, cosine(row, expected))
                    if chunkTokens == 2 {
                        minDecodeCosine = min(minDecodeCosine,
                                              cosine(row, decodeRows[position]))
                        for (key, count) in counts {
                            guard let buffer = runner.tappedInternals[key] else {
                                Issue.record("no \(key) tapped after span \(chunkStart)..<\(chunkEnd)"); return
                            }
                            let rowBytes = count * MemoryLayout<Float16>.size
                            let internalRow = Self.readFP16(
                                buffer, byteOffset: t * rowBytes, count: count)
                            let (shape, goldenRows) = try golden("layer3_internals", key)
                            let rowElements = goldenRows.count / shape[0]
                            #expect(rowElements == count,
                                    "\(key): tapped \(count) values per row, golden \(rowElements)")
                            let expectedInternal = Array(goldenRows[
                                (position * rowElements)..<(position + 1) * rowElements])
                            minInternalCosine = min(minInternalCosine,
                                                    cosine(internalRow, expectedInternal))
                        }
                    }
                }
                tappedRows += chunkEnd - chunkStart
                chunkStart = chunkEnd
            }
            #expect(tappedRows == ids.count)
            // completedCount is per call (tokens done inside that call),
            // so the one-span-per-call runs report 2, 2, 1.
            #expect(progress == (chunkTokens == 2 ? [2, 2, 1] : [5]))
            #expect(!runner.chunkCommitState.isDirty)
            print("S20 step 5 p0 chunked layer 3 hidden (chunk \(chunkTokens)) minimum cosine: \(minHiddenCosine)")
            #expect(minHiddenCosine >= 0.995,
                    "chunked layer 3 hidden minimum cosine \(minHiddenCosine) < 0.995 (C8)")
            if chunkTokens == 2 {
                print("S20 step 5 p0 chunked layer 3 hidden vs decode minimum cosine: \(minDecodeCosine)")
                #expect(minDecodeCosine >= 0.9999,
                        "chunked layer 3 hidden vs decode minimum cosine \(minDecodeCosine) < 0.9999")
                print("S20 step 5 p0 chunked layer 3 internals minimum cosine: \(minInternalCosine)")
                #expect(minInternalCosine >= 0.995,
                        "chunked layer 3 internals minimum cosine \(minInternalCosine) < 0.995 (C8)")
            }
        }
    }

    /// S20 step 6: the whole model through the chunked prefill. All six
    /// prompts, in chunks of 4 (so every prompt spans several chunks) and
    /// in a single chunk: the hidden state after the final norm of every
    /// position must match the S19 token-at-a-time `produce` run (cosine
    /// ≥ 0.98, minimum over the positions) and the last-position logits
    /// must match the same `produce` run (cosine ≥ 0.998, top-1 equal):
    /// the sheet's ≥ 0.9999 against the decode is unreachable with the
    /// GEMM prefill path the map chose (the GEMM/GEMV difference of ~1e-6
    /// flips the 8th/9th MoE expert ties), so the decision recorded in the
    /// S20 report keeps the GEMM and approves these bounds. For p0 and p1
    /// the logits also match the `logits_last.json` golden (cosine ≥ 0.998,
    /// the deviation approved in S19). The single-chunk run also asks for
    /// `outputMode = .greedyIfAvailable`: Qwen never fuses greedy into
    /// prefill, so the seed stays `.logitsWritten`.
    @Test("the chunked six-prompt prefill matches the decode states and logits")
    func chunkedSixPromptPrefillMatchesDecodeAndGolden() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let config = env.model.config
        let hiddenSize = config.hiddenSize
        let vocabSize = config.vocabSize
        let rowBytes = hiddenSize * MemoryLayout<Float16>.size
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))

        let lastData = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/logits_last.json"))
        let lastGolden = try JSONSerialization.jsonObject(with: lastData) as! [String: Any]

        for name in ["p0", "p1", "p2", "p3", "p4", "p5"] {
            let ids = try promptIDs(name)

            // Reference: the S19 token-at-a-time `produce` over the whole
            // prompt, tapped after the final norm at every position.
            runner.reset()
            runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
                points: [.afterFinalNorm], rows: ids.count)
            for (position, id) in ids.enumerated() {
                try await runner.produce(token: id, position: position, into: logits)
            }
            guard let decodeTapped = runner.tappedHidden else {
                Issue.record("no decode hidden tap for \(name)"); return
            }
            let decodeFinalNorm = (0..<ids.count).map {
                Self.readFP16(decodeTapped, byteOffset: $0 * rowBytes, count: hiddenSize)
            }
            let decodeLogits = Self.readFP16(logits, byteOffset: 0, count: vocabSize)

            for chunkTokens in [4, ids.count] {
                runner.reset()
                runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
                    points: [.afterFinalNorm], rows: ids.count)
                let prefillConfig = PrefillRuntimeConfig.defaultChunked
                    .replacingChunkTokens(chunkTokens)
                // The single-chunk run asks for the fused greedy head,
                // which Qwen does not have (S19): the seed must still be
                // `.logitsWritten`.
                let outputMode: PrefillOutputMode = (chunkTokens == ids.count
                                                     ? .greedyIfAvailable : .logits)
                let result = try await runner.prefillChunked(
                    tokens: ids[...],
                    startPosition: 0,
                    outputMode: outputMode,
                    config: prefillConfig,
                    into: logits,
                    onProgress: { _ in })
                #expect(result.newPosition == ids.count)
                #expect(result.seed == .logitsWritten)
                #expect(runner.continuationPosition == ids.count)
                #expect(!runner.chunkCommitState.isDirty)

                guard let tapped = runner.tappedHidden else {
                    Issue.record("no hidden tap for \(name) chunk \(chunkTokens)"); return
                }
                #expect(runner.tappedHiddenCount == ids.count)
                var minNormCosine = 2.0
                for position in 0..<ids.count {
                    let row = Self.readFP16(tapped, byteOffset: position * rowBytes,
                                            count: hiddenSize)
                    minNormCosine = min(minNormCosine,
                                        cosine(row, decodeFinalNorm[position]))
                }
                let logitsRow = Self.readFP16(logits, byteOffset: 0, count: vocabSize)
                let logitsCosine = cosine(logitsRow, decodeLogits)
                let top1Equal = Self.topIndices(logitsRow, count: 1)[0]
                    == Self.topIndices(decodeLogits, count: 1)[0]
                var goldenCosine: Double?
                if let lastEntry = lastGolden[name] as? [String: Any],
                   let logitsEntry = lastEntry["logits"] as? [String: Any] {
                    let expected = try Self.decodeF16(logitsEntry)
                    goldenCosine = cosine(logitsRow, expected)
                }
                print("S20 step 6 \(name) chunked (chunk \(chunkTokens)) final-norm min cosine: \(minNormCosine)")
                print("S20 step 6 \(name) chunked (chunk \(chunkTokens)) last logits vs decode cosine: \(logitsCosine), top-1 equal: \(top1Equal)")
                if let goldenCosine {
                    print("S20 step 6 \(name) chunked (chunk \(chunkTokens)) last logits vs golden cosine: \(goldenCosine)")
                }
                #expect(minNormCosine >= 0.98,
                        "\(name) chunk \(chunkTokens) final-norm min cosine \(minNormCosine) < 0.98 (approved deviation: GEMM prefill vs GEMV decode)")
                #expect(logitsCosine >= 0.998,
                        "\(name) chunk \(chunkTokens) last logits cosine \(logitsCosine) < 0.998 (approved deviation: GEMM prefill vs GEMV decode)")
                #expect(top1Equal,
                        "\(name) chunk \(chunkTokens) last logits top-1 differs from the decode")
                if let goldenCosine {
                    #expect(goldenCosine >= 0.998,
                            "\(name) chunk \(chunkTokens) last logits golden cosine \(goldenCosine) < 0.998 (C8 0.999, approved deviation)")
                }
            }
        }
    }

    /// S20 step 7 (D6): a chunked prefill can continue at its tail.
    /// Prefilling the first two tokens of p0 and then the rest must give
    /// the same last-position logits as the full chunked prefill of p0
    /// (chunks of 4, so the full prefill spans two chunks; cosine ≥ 0.9999,
    /// chunked-vs-chunked, where the GEMM/GEMV gap of step 6 does not
    /// apply).
    @Test("the chunked p0 prefill continuation matches the full prefill")
    func chunkedPrefillContinuationMatchesFullPrefill() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let vocabSize = env.model.config.vocabSize
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        let ids = try promptIDs("p0")
        let config = PrefillRuntimeConfig.defaultChunked.replacingChunkTokens(4)
        func prefill(_ tokens: ArraySlice<Int32>, _ startPosition: Int) async throws -> PrefillResult {
            try await runner.prefillChunked(
                tokens: tokens,
                startPosition: startPosition,
                outputMode: .logits,
                config: config,
                into: logits,
                onProgress: { _ in })
        }

        // Reference: the full chunked prefill of p0.
        runner.reset()
        let full = try await prefill(ids[...], 0)
        let fullLogits = Self.readFP16(logits, byteOffset: 0, count: vocabSize)
        #expect(full.newPosition == ids.count)

        // Head prefill, then the tail continuation at its end (D6).
        runner.reset()
        let head = try await prefill(ids[0..<2], 0)
        #expect(head.newPosition == 2)
        #expect(runner.continuationPosition == 2)
        try runner.prepareForContinuation(expectedPosition: 2)
        let tail = try await prefill(ids[2...], 2)
        #expect(tail.newPosition == ids.count)
        #expect(!runner.chunkCommitState.isDirty)
        let tailLogits = Self.readFP16(logits, byteOffset: 0, count: vocabSize)
        let cosine = cosine(tailLogits, fullLogits)
        print("S20 step 7 p0 continuation vs full prefill last-logits cosine: \(cosine)")
        #expect(cosine >= 0.9999,
                "p0 continuation last-logits cosine \(cosine) < 0.9999")
    }

    /// S20 step 7 (D6): prefix reuse through a snapshot. Prefill prompt A
    /// (p0), then prompt B's tokens (p1) at the tail: the runner now holds
    /// a snapshot at the end of each. Rewinding to the end of A and
    /// prefilling B again at the tail must give the same last-position
    /// logits as the full chunked prefill of A+B from a clean cache
    /// (chunks of 4; cosine ≥ 0.9999, chunked-vs-chunked).
    @Test("the chunked prefill rewinds to a prompt snapshot and reuses the prefix")
    func chunkedPrefillRewindReusesPrefix() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let vocabSize = env.model.config.vocabSize
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        let a = try promptIDs("p0")
        let b = try promptIDs("p1")
        let config = PrefillRuntimeConfig.defaultChunked.replacingChunkTokens(4)
        func prefill(_ tokens: ArraySlice<Int32>, _ startPosition: Int) async throws -> PrefillResult {
            try await runner.prefillChunked(
                tokens: tokens,
                startPosition: startPosition,
                outputMode: .logits,
                config: config,
                into: logits,
                onProgress: { _ in })
        }

        // Prefill A, then B at the tail: the snapshot list holds the end
        // of each (D6, at most two).
        runner.reset()
        let endA = try await prefill(a[...], 0)
        #expect(endA.newPosition == a.count)
        let endB = try await prefill(b[...], a.count)
        #expect(endB.newPosition == a.count + b.count)

        // Rewind to the end of A and re-prefill B at the tail.
        try runner.rewind(to: a.count)
        #expect(runner.continuationPosition == a.count)
        try runner.prepareForContinuation(expectedPosition: a.count)
        let reused = try await prefill(b[...], a.count)
        #expect(reused.newPosition == a.count + b.count)
        let reusedLogits = Self.readFP16(logits, byteOffset: 0, count: vocabSize)

        // Reference: the full chunked prefill of A+B from a clean cache.
        runner.reset()
        let full = try await prefill((a + b)[...], 0)
        #expect(full.newPosition == a.count + b.count)
        let fullLogits = Self.readFP16(logits, byteOffset: 0, count: vocabSize)
        let cosine = cosine(reusedLogits, fullLogits)
        print("S20 step 7 A+B rewind reuse vs full A+B prefill last-logits cosine: \(cosine)")
        #expect(cosine >= 0.9999,
                "A+B rewind reuse last-logits cosine \(cosine) < 0.9999")
    }

    /// S20 step 7 (D6): the DeltaNet state cannot be rewound to a position
    /// without a prefill snapshot, so `rewind` throws `rewindUnsupported`.
    /// The saved snapshot position is still supported, and after `reset`
    /// the snapshot is gone and the runner works from position 0 again.
    @Test("the chunked prefill rejects an unsupported rewind and resets")
    func chunkedPrefillRewindUnsupportedAndReset() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let vocabSize = env.model.config.vocabSize
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        let ids = try promptIDs("p0")
        let config = PrefillRuntimeConfig.defaultChunked.replacingChunkTokens(4)
        func prefill(_ tokens: ArraySlice<Int32>, _ startPosition: Int) async throws -> PrefillResult {
            try await runner.prefillChunked(
                tokens: tokens,
                startPosition: startPosition,
                outputMode: .logits,
                config: config,
                into: logits,
                onProgress: { _ in })
        }

        runner.reset()
        let end = try await prefill(ids[...], 0)
        #expect(end.newPosition == ids.count)

        // Only the end of the prompt has a snapshot: position 2 does not.
        do {
            try runner.rewind(to: 2)
            Issue.record("rewind(to: 2) must throw rewindUnsupported")
        } catch let error as QwenRewindError {
            #expect(error == .rewindUnsupported(position: 2))
        }

        // The saved snapshot position is supported; `reset` clears it and
        // the runner works from position 0 again.
        try runner.rewind(to: ids.count)
        #expect(runner.continuationPosition == ids.count)
        runner.reset()
        #expect(runner.continuationPosition == 0)
        let again = try await prefill(ids[...], 0)
        #expect(again.newPosition == ids.count)
        #expect(!runner.chunkCommitState.isDirty)
    }

    /// S20 step 8, extended by S26 step 1: prefill throughput on 512 and
    /// 4096 tokens at chunks 128 and 256, plus token-at-a-time decode after
    /// a default-chunk 4096-token prefill. With the S26 counters on, each
    /// run also splits its wall time into CPU encode, GPU busy and await
    /// seconds and counts the MoE work; one single-span run per chunk size
    /// prints the per-layer groups / experts / dispatches / argument
    /// buffers / expert-fetch of exactly one block. The runner uses
    /// `maxContext: 4224` so the 4096-token prefill fits. The rates go to
    /// the report (2.9); they are measurements, not golden bounds.
    @Test("the prefill and decode throughput are measured for the report")
    func prefillAndDecodeThroughputAreMeasured() async throws {
        let env = try makeEnvironment(maxContext: 4224)
        let runner = env.runner
        let vocabSize = env.model.config.vocabSize
        let numLayers = env.model.config.numLayers
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        func seconds(_ duration: Duration) -> Double {
            Double(duration.components.seconds)
                + Double(duration.components.attoseconds) * 1e-18
        }
        func prefill(_ count: Int, config: PrefillRuntimeConfig) async throws
            -> (tokens: Int, duration: Duration) {
            let tokens = Array(0..<count).map { Int32($0) }
            let clock = ContinuousClock()
            let start = clock.now
            let result = try await runner.prefillChunked(
                tokens: tokens[...],
                startPosition: 0,
                outputMode: .logits,
                config: config,
                into: logits,
                onProgress: { _ in })
            let duration = clock.now - start
            #expect(result.newPosition == count)
            return (count, duration)
        }
        typealias Counters = QwenHybridForwardRunner.S26PrefillCounters
        func printTotals(_ label: String, _ counters: Counters, spans: Int) {
            print("S26 \(label): cpuEncode=\(counters.cpuEncodeSeconds)s "
                + "gpu=\(counters.gpuSeconds)s await=\(counters.awaitSeconds)s "
                + "commandBuffers=\(counters.gpuIntervals.count)")
            var groups = 0, experts = 0, dispatches = 0, argBuffers = 0, fetch = 0.0
            for layer in 0..<numLayers {
                let stats = counters.layers[layer] ?? .init()
                groups += stats.groups
                experts += stats.expertsFetched
                dispatches += stats.dispatches
                argBuffers += stats.argumentBuffers
                fetch += stats.expertFetchSeconds
            }
            let per = Double(spans)
            print("S26 \(label) per span: groups=\(Double(groups) / per) "
                + "expertsRead=\(Double(experts) / per) "
                + "moeDispatches=\(Double(dispatches) / per) "
                + "argBuffers=\(Double(argBuffers) / per) "
                + "expertFetch=\(fetch / per)s")
            let head = counters.layers[-1] ?? .init()
            print("S26 \(label) embedding+final+head dispatches per span: "
                + "\(Double(head.dispatches) / per)")
        }
        func printLayerTable(_ label: String, _ counters: Counters) {
            print("S26 \(label) per-layer groups/experts/dispatches/argBuffers/fetch:")
            for layer in 0..<numLayers {
                let stats = counters.layers[layer] ?? .init()
                print("S26   L\(layer): \(stats.groups)/\(stats.expertsFetched)/"
                    + "\(stats.dispatches)/\(stats.argumentBuffers)/"
                    + "\(stats.expertFetchSeconds)s")
            }
        }

        // Warm-up (one default span) before the timed runs.
        runner.reset()
        let warmup = try await prefill(128, config: .defaultChunked)
        print("S20 step 8 warm-up prefill \(warmup.tokens) tokens: \(warmup.duration)")

        for chunk in [128, 256] {
            let config = PrefillRuntimeConfig.defaultChunked.replacingChunkTokens(chunk)
            #expect(config.chunkTokens == chunk)
            for count in [512, 4096] {
                runner.reset()
                let counters = Counters()
                runner.s26Counters = counters
                let run = try await prefill(count, config: config)
                #expect(counters.waitForDrain())
                runner.s26Counters = nil
                let secondsOfRun = seconds(run.duration)
                let rate = Double(run.tokens) / secondsOfRun
                let spans = (count + chunk - 1) / chunk
                print("S26 prefill \(run.tokens) tokens (chunk \(chunk), \(spans) spans): "
                    + "\(secondsOfRun)s = \(rate) tokens/s")
                printTotals("prefill \(count) chunk \(chunk)", counters, spans: spans)
                #expect(rate > 0)
            }
            // One block, exactly: a single-span prefill of `chunk` tokens
            // attributes one block's cost layer by layer.
            runner.reset()
            let counters = Counters()
            runner.s26Counters = counters
            let block = try await prefill(chunk, config: config)
            #expect(counters.waitForDrain())
            runner.s26Counters = nil
            print("S26 single block \(chunk) tokens: \(seconds(block.duration))s")
            printTotals("block \(chunk)", counters, spans: 1)
            printLayerTable("block \(chunk)", counters)
        }

        // Decode token per token after a fresh default-chunk 4096-token
        // prefill, without counters (the S20 baseline conditions).
        runner.reset()
        _ = try await prefill(4096, config: .defaultChunked)
        let decodeCount = 128
        let clock = ContinuousClock()
        let decodeStart = clock.now
        for offset in 0..<decodeCount {
            let id = Int32(4096 + offset)
            try await runner.produce(
                token: id, position: 4096 + offset, into: logits)
        }
        let decodeDuration = clock.now - decodeStart
        let decodeSeconds = seconds(decodeDuration)
        let decodeRate = Double(decodeCount) / decodeSeconds
        #expect(runner.continuationPosition == 4096 + decodeCount)
        print("S26 decode \(decodeCount) tokens after the 4096-token prefill: "
            + "\(decodeSeconds)s = \(decodeRate) tokens/s")
        #expect(decodeRate > 0)
    }

    /// S26 step 1: the reference hashes the optimization steps 2-4 must
    /// keep identical (the card's "regola dei risultati"). SHA-256 over the
    /// FP16 logits of the last position for the six prompts at chunks 4 and
    /// 128, plus the logits of eight greedy decode steps after p0. Saved to
    /// `scratch/S26/baseline.json` (outside git) and printed for the report.
    /// The chunk-4 and chunk-128 hashes must agree per prompt: the chunked
    /// path is bit-deterministic across span splits (S20 step 6).
    @Test("the S26 logit hash baseline is recorded for steps 2-5")
    func s26LogitHashBaselineIsRecorded() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let vocabSize = env.model.config.vocabSize
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))
        func logitHash() -> String {
            let raw = Data(bytes: logits.contents(),
                           count: vocabSize * MemoryLayout<Float16>.size)
            return SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
        }

        var promptHashes: [String: [String: String]] = [:]
        for name in ["p0", "p1", "p2", "p3", "p4", "p5"] {
            let ids = try promptIDs(name)
            var row: [String: String] = [:]
            for chunk in [4, 128] {
                runner.reset()
                let config = PrefillRuntimeConfig.defaultChunked.replacingChunkTokens(chunk)
                let result = try await runner.prefillChunked(
                    tokens: ids[...], startPosition: 0, outputMode: .logits,
                    config: config, into: logits, onProgress: { _ in })
                #expect(result.newPosition == ids.count)
                row["chunk\(chunk)"] = logitHash()
            }
            #expect(row["chunk4"] == row["chunk128"],
                    "\(name): chunked prefill is not bit-deterministic across span splits")
            promptHashes[name] = row
            print("S26 hash \(name): chunk4 \(row["chunk4"]!) chunk128 \(row["chunk128"]!)")
        }

        // Eight greedy decode steps after p0: the prefill's logits decide
        // token 1, each produce's logits decide the next one.
        let ids0 = try promptIDs("p0")
        runner.reset()
        let result = try await runner.prefillChunked(
            tokens: ids0[...], startPosition: 0, outputMode: .logits,
            config: .defaultChunked, into: logits, onProgress: { _ in })
        #expect(result.newPosition == ids0.count)
        var greedyHashes: [String] = []
        var greedyTokens: [Int] = []
        var row = Self.readFP16(logits, byteOffset: 0, count: vocabSize)
        for step in 0..<8 {
            greedyHashes.append(logitHash())
            let token = Self.topIndices(row, count: 1)[0]
            greedyTokens.append(token)
            if step < 7 {
                try await runner.produce(token: Int32(token),
                                         position: ids0.count + step, into: logits)
                row = Self.readFP16(logits, byteOffset: 0, count: vocabSize)
            }
        }
        print("S26 hash greedy p0 tokens \(greedyTokens)")
        for (step, hash) in greedyHashes.enumerated() {
            print("S26 hash greedy step \(step): \(hash)")
        }

        // Save the baseline (scratch, outside git) for the later steps.
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var modelBytes = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &modelBytes, &size, nil, 0)
        let modelText = String(decoding: modelBytes.prefix { $0 != 0 }.map(UInt8.init(bitPattern:)),
                               as: UTF8.self)
        let baseline: [String: Any] = [
            "mac": modelText,
            "variant": "Qwen3.6 4-bit (scratch/qwen36.gturbo)",
            "promptLogitsSha256": promptHashes,
            "greedyAfterP0": ["tokens": greedyTokens, "logitsSha256": greedyHashes],
        ]
        let directory = QwenTestResources.repoRoot.appendingPathComponent("scratch/S26")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: baseline,
                                              options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("baseline.json"))
        print("S26 baseline written to "
            + directory.appendingPathComponent("baseline.json").path)
    }

    /// S19 step 3: the Gated DeltaNet mixer of layer 0 must reproduce the
    /// mlx-lm golden `layer0_internals.json` for all five tokens of p0
    /// (C8: cosine ≥ 0.995 per internal quantity; the DeltaNet state uses
    /// the same cosine bound, its relative error goes to the report).
    @Test("the layer-0 DeltaNet mixer matches the mlx-lm golden")
    func layer0DeltaNetMatchesGolden() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let model = env.model
        let hybrid = try #require(model.qwenHybridConfig)
        let hiddenSize = model.config.hiddenSize
        let vocabSize = model.config.vocabSize
        let keyHeads = hybrid.linearNumKeyHeads
        let keyDim = hybrid.linearKeyHeadDim
        let valueHeads = hybrid.linearNumValueHeads
        let valueDim = valueHeads * hybrid.linearValueHeadDim
        let qkvChannels = 2 * keyHeads * keyDim + valueDim
        let qkBytes = keyHeads * keyDim * MemoryLayout<Float16>.size
        let ids = try promptIDs("p0")
        #expect(ids.count == 5)

        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))

        // The work buffers hold one token's row, so re-running each prefix
        // of p0 yields the tapped layer's quantities for the last token of
        // the prefix (deterministic state: the rows equal the full run's).
        struct RowSpec {
            let byteOffset: Int
            let count: Int
            let isFP32: Bool
            init(_ byteOffset: Int, _ count: Int, _ isFP32: Bool) {
                self.byteOffset = byteOffset
                self.count = count
                self.isFP32 = isFP32
            }
        }
        let specs: [String: RowSpec] = [
            "x_norm": RowSpec(0, hiddenSize, false),
            "qkv_raw": RowSpec(0, qkvChannels, false),
            "z": RowSpec(0, valueDim, false),
            "a": RowSpec(0, valueHeads, false),
            "b": RowSpec(0, valueHeads, false),
            "conv_out": RowSpec(0, qkvChannels, false),
            "q": RowSpec(0, keyHeads * keyDim, false),
            "k": RowSpec(0, keyHeads * keyDim, false),
            "v": RowSpec(2 * qkBytes, valueDim, false),
            "beta": RowSpec(0, valueHeads, true),
            "core_out": RowSpec(0, valueDim, false),
            "gated_out": RowSpec(0, valueDim, false),
            "mixer_out": RowSpec(0, hiddenSize, false),
        ]
        var actual: [String: [[Float]]] = [:]
        for end in 0..<ids.count {
            runner.reset()
            runner.debugInternalsLayer = 0
            runner.debugStopAfterLayer = 0
            for position in 0...end {
                try await runner.produce(
                    token: ids[position], position: position, into: logits)
            }
            let tapped = runner.tappedInternals
            guard !tapped.isEmpty else {
                Issue.record("no internals tapped after prefix \(end)"); return
            }
            for (key, spec) in specs {
                guard let buffer = tapped[key] else {
                    Issue.record("missing internal \(key)"); return
                }
                let row = spec.isFP32
                    ? Self.readFP32(buffer, byteOffset: spec.byteOffset, count: spec.count)
                    : Self.readFP16(buffer, byteOffset: spec.byteOffset, count: spec.count)
                actual[key, default: []].append(row)
            }
        }

        // g = -exp(A_log) * softplus(a + dt_bias): the gates kernel writes
        // only beta/decay, so recompute g on the CPU from the tapped `a`
        // rows and the BF16 A_log / dt_bias tensors (3.5). The gate tensors
        // live at their resident offsets inside the model buffer, so read
        // from the view's offset, not the buffer start.
        let aLog = try model.qwenALog(layer: 0)
        let dtBias = try model.qwenDtBias(layer: 0)
        let aLogRow = Self.readBF16(aLog.buffer, byteOffset: Int(aLog.offset),
                                    count: valueHeads)
        let dtBiasRow = Self.readBF16(dtBias.buffer, byteOffset: Int(dtBias.offset),
                                      count: valueHeads)
        let aRows = try #require(actual["a"])
        actual["g"] = aRows.map { row in
            (0..<valueHeads).map { j in
                let x = Double(row[j]) + Double(dtBiasRow[j])
                return Float(-exp(Double(aLogRow[j])) * Self.softplus(x))
            }
        }

        // C8: cosine ≥ 0.995 for every internal quantity, over all rows.
        for key in specs.keys.sorted() + ["g"] {
            let (shape, goldenRows) = try golden("layer0_internals", key)
            let rows = try #require(actual[key])
            #expect(rows.count == shape[0],
                    "\(key): tapped \(rows.count) rows, golden \(shape[0])")
            let rowElements = goldenRows.count / shape[0]
            var minCosine = 2.0
            for row in rows.indices {
                let actualRow = rows[row]
                #expect(actualRow.count == rowElements,
                        "\(key) row \(row): \(actualRow.count) values, golden \(rowElements)")
                let expected = Array(
                    goldenRows[(row * rowElements)..<(row + 1) * rowElements])
                minCosine = min(minCosine, cosine(actualRow, expected))
            }
            print("S19 step 3 layer 0 \(key): min cosine \(minCosine)")
            #expect(minCosine >= 0.995,
                    "layer 0 \(key) min cosine \(minCosine) < 0.995 (C8)")
        }

        // The DeltaNet state after the full p0 prompt (the last prefix run
        // left it in the cache). Cache layout [testa][colonna][riga] (D5)
        // vs the golden's [testa][riga][colonna]: transpose first (C8:
        // relative error ≤ 1e-4).
        guard let stateBuffer = runner.tappedInternals["state_heads0to3"],
              let stateOffset = runner.cache.deltaStateByteOffset(layer: 0) else {
            Issue.record("no DeltaNet state tapped"); return
        }
        let (stateShape, goldenState) = try golden("layer0_internals", "state_heads0to3")
        #expect(stateShape == [4, keyDim, keyDim])
        let stateHeads = stateShape[0]
        let stateRaw = UnsafeRawBufferPointer(start: stateBuffer.contents(),
                                              count: stateBuffer.length)
        var actualState = [Float](repeating: 0, count: stateHeads * keyDim * keyDim)
        for head in 0..<stateHeads {
            for row in 0..<keyDim {          // key dimension (riga i)
                for column in 0..<keyDim {   // value dimension (colonna c)
                    let offset = stateOffset
                        + ((head * keyDim + column) * keyDim + row) * MemoryLayout<Float>.size
                    actualState[(head * keyDim + row) * keyDim + column] =
                        stateRaw.load(fromByteOffset: offset, as: Float32.self)
                }
            }
        }
        // C8's 1e-4 relative-error bound is for the GPU kernel against its
        // CPU reference (S15); against mlx-lm the state uses the same
        // cosine bound as the other internal quantities, with the relative
        // error recorded in the report (S19 deviation of 2026-11-11).
        let stateCosine = cosine(actualState, goldenState)
        let stateRel = RelError.compute(actual: actualState, reference: goldenState)
        print("S19 step 3 layer 0 state_heads0to3 cosine \(stateCosine), relative error \(stateRel)")
        #expect(stateCosine >= 0.995,
                "layer 0 state_heads0to3 cosine \(stateCosine) < 0.995 (C8)")
    }

    /// S19 step 4: the MoE block after the layer-0 mixer must reproduce the
    /// mlx-lm golden `layer0_internals.json` for all five tokens of p0
    /// (C8: cosine ≥ 0.995 per quantity; the top-8 ids equal per position,
    /// a mismatch only as an annotated near-tie with its two weights) and
    /// the hidden state after layer 0, `hidden_p0.json → layer0`
    /// (cosine ≥ 0.995).
    @Test("the layer-0 MoE block matches the mlx-lm golden")
    func layer0MoEMatchesGolden() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let hiddenSize = env.model.config.hiddenSize
        let vocabSize = env.model.config.vocabSize
        let slots = env.model.config.topKExperts
        let ids = try promptIDs("p0")
        #expect(ids.count == 5)

        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))

        // The work buffers hold one token's row, so re-running each prefix
        // of p0 yields the tapped layer's quantities for the last token of
        // the prefix (deterministic state: the rows equal the full run's).
        struct RowSpec {
            let byteOffset: Int
            let count: Int
            let isFP32: Bool
            init(_ byteOffset: Int, _ count: Int, _ isFP32: Bool) {
                self.byteOffset = byteOffset
                self.count = count
                self.isFP32 = isFP32
            }
        }
        let specs: [String: RowSpec] = [
            "moe_x": RowSpec(0, hiddenSize, false),
            "router_top_weights": RowSpec(0, slots, true),
            "routed_out": RowSpec(0, hiddenSize, false),
            "shared_out": RowSpec(0, hiddenSize, false),
            "layer_out": RowSpec(0, hiddenSize, false),
        ]
        let rowBytes = hiddenSize * MemoryLayout<Float16>.size
        var actual: [String: [[Float]]] = [:]
        var actualTopIDs: [[Int]] = []
        var hiddenRows: [[Float]] = []
        for end in 0..<ids.count {
            runner.reset()
            runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
                points: [.afterLayer(0)], rows: ids.count)
            runner.debugInternalsLayer = 0
            runner.debugStopAfterLayer = 0
            for position in 0...end {
                try await runner.produce(
                    token: ids[position], position: position, into: logits)
            }
            let tapped = runner.tappedInternals
            guard !tapped.isEmpty else {
                Issue.record("no internals tapped after prefix \(end)"); return
            }
            for (key, spec) in specs {
                guard let buffer = tapped[key] else {
                    Issue.record("missing internal \(key)"); return
                }
                let row = spec.isFP32
                    ? Self.readFP32(buffer, byteOffset: spec.byteOffset, count: spec.count)
                    : Self.readFP16(buffer, byteOffset: spec.byteOffset, count: spec.count)
                actual[key, default: []].append(row)
            }
            guard let idsBuffer = tapped["router_top_ids"] else {
                Issue.record("missing router_top_ids"); return
            }
            let idsRaw = UnsafeRawBufferPointer(start: idsBuffer.contents(),
                                                count: slots * MemoryLayout<UInt32>.size)
            actualTopIDs.append((0..<slots).map {
                Int(idsRaw.load(fromByteOffset: $0 * MemoryLayout<UInt32>.size,
                                as: UInt32.self))
            })
            guard let tappedHidden = runner.tappedHidden else {
                Issue.record("no hidden tapped after prefix \(end)"); return
            }
            let hiddenRaw = UnsafeRawBufferPointer(start: tappedHidden.contents(),
                                                   count: (end + 1) * rowBytes)
            hiddenRows.append((0..<hiddenSize).map {
                Float(hiddenRaw.load(fromByteOffset: end * rowBytes
                    + $0 * MemoryLayout<Float16>.size, as: Float16.self))
            })
        }

        // C8: cosine ≥ 0.995 for every quantity, over all rows.
        for key in specs.keys.sorted() {
            let (shape, goldenRows) = try golden("layer0_internals", key)
            let rows = try #require(actual[key])
            #expect(rows.count == shape[0],
                    "\(key): tapped \(rows.count) rows, golden \(shape[0])")
            let rowElements = goldenRows.count / shape[0]
            var minCosine = 2.0
            for row in rows.indices {
                let actualRow = rows[row]
                #expect(actualRow.count == rowElements,
                        "\(key) row \(row): \(actualRow.count) values, golden \(rowElements)")
                let expected = Array(
                    goldenRows[(row * rowElements)..<(row + 1) * rowElements])
                minCosine = min(minCosine, cosine(actualRow, expected))
            }
            print("S19 step 4 layer 0 \(key): min cosine \(minCosine)")
            #expect(minCosine >= 0.995,
                    "layer 0 \(key) min cosine \(minCosine) < 0.995 (C8)")
        }

        // The top-8 ids: equal per position; a mismatch is a near-tie and
        // the test shows the two weights in play (C8). The golden ids are a
        // raw [5][8] int array (not a C7 object), in ascending weight order;
        // the same set in another order (tied weights) passes.
        let data = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/layer0_internals.json"))
        let internals = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let goldenIDs = try #require(internals["router_top_ids"] as? [[Int]])
        let (goldenWShape, goldenW) = try golden("layer0_internals", "router_top_weights")
        #expect(goldenWShape == [ids.count, slots])
        let actualW = try #require(actual["router_top_weights"])
        var equalPositions = 0
        for row in 0..<ids.count where actualTopIDs[row] == goldenIDs[row] {
            equalPositions += 1
        }
        print("S19 step 4 layer 0 router_top_ids: \(equalPositions)/\(ids.count) positions equal")
        for row in 0..<ids.count {
            let goldenRow = goldenIDs[row]
            let actualRow = actualTopIDs[row]
            guard actualRow != goldenRow else { continue }
            print("S19 step 4 layer 0 position \(row): golden top-8 \(goldenRow), actual \(actualRow)")
            let goldenSet = Set(goldenRow)
            let actualSet = Set(actualRow)
            let dropped = goldenRow.filter { !actualSet.contains($0) }
            let gained = actualRow.filter { !goldenSet.contains($0) }
            #expect(dropped.count == gained.count,
                    "p0 position \(row): top-8 differs without 1:1 swaps (golden \(goldenRow), actual \(actualRow))")
            var nearTie = true
            for (index, droppedExpert) in dropped.enumerated() {
                let gainedExpert = gained[index]
                let wGolden = goldenW[row * slots + goldenRow.firstIndex(of: droppedExpert)!]
                let wActual = actualW[row][actualRow.firstIndex(of: gainedExpert)!]
                print("S19 step 4 layer 0 position \(row): near-tie? golden expert \(droppedExpert) weight \(wGolden) vs actual expert \(gainedExpert) weight \(wActual)")
                if abs(wGolden - wActual) > 0.001 { nearTie = false }
            }
            #expect(nearTie,
                    "p0 position \(row): top-8 mismatch is not a near-tie (golden \(goldenRow), actual \(actualRow))")
        }

        // The hidden state after layer 0 (mixer + MoE + residual).
        let (layer0Shape, layer0Golden) = try golden("hidden_p0", "layer0")
        #expect(layer0Shape == [ids.count, hiddenSize])
        var minHiddenCosine = 2.0
        for row in 0..<ids.count {
            let expected = Array(
                layer0Golden[(row * hiddenSize)..<(row + 1) * hiddenSize])
            minHiddenCosine = min(minHiddenCosine, cosine(hiddenRows[row], expected))
        }
        print("S19 step 4 layer 0 hidden state: min cosine \(minHiddenCosine)")
        #expect(minHiddenCosine >= 0.995,
                "layer 0 hidden state min cosine \(minHiddenCosine) < 0.995 (C8)")
    }

    /// S19 step 5: the full-attention mixer of layer 3 (after the linear
    /// layers 0–2) must reproduce `layer3_internals.json` for all five
    /// tokens of p0, and the hidden state after layer 3 must match
    /// `hidden_p0.json → layer3` (C8: cosine ≥ 0.995 for each). The golden
    /// `q_pre_norm` / `k_pre_norm` are q and k after the per-head norm and
    /// before RoPE (make_model_golden.py), not before the norm.
    @Test("the layer-3 attention matches the mlx-lm golden")
    func layer3AttentionMatchesGolden() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let config = env.model.config
        let hiddenSize = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numKVHeads * config.fullHeadDim
        let ids = try promptIDs("p0")
        #expect(ids.count == 5)
        #expect(config.fullAttentionLayerMask[3] == 1)

        let logits = try #require(env.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))

        // FP16 row lengths of the tapped work buffers, by golden key.
        let counts: [String: Int] = [
            "x_norm": hiddenSize,
            "q_pre_norm": queryDim,
            "gate": queryDim,
            "k_pre_norm": kvDim,
            "v": kvDim,
            "q_rope": queryDim,
            "k_rope": kvDim,
            "attn_out": queryDim,
            "gated": queryDim,
            "mixer_out": hiddenSize,
            "layer_out": hiddenSize,
        ]
        let rowBytes = hiddenSize * MemoryLayout<Float16>.size
        var actual: [String: [[Float]]] = [:]
        var hiddenRows: [[Float]] = []
        // One prefix per token, as in steps 3–4: the work buffers hold the
        // last token's row, and layer 3's attention reads the cache rows
        // the earlier tokens of the prefix wrote.
        for end in 0..<ids.count {
            runner.reset()
            runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
                points: [.afterLayer(3)], rows: ids.count)
            runner.debugInternalsLayer = 3
            runner.debugStopAfterLayer = 3
            for position in 0...end {
                try await runner.produce(
                    token: ids[position], position: position, into: logits)
            }
            let tapped = runner.tappedInternals
            for (key, count) in counts {
                guard let buffer = tapped[key] else {
                    Issue.record("missing internal \(key) after prefix \(end)"); return
                }
                actual[key, default: []].append(
                    Self.readFP16(buffer, byteOffset: 0, count: count))
            }
            let tappedHidden = try #require(runner.tappedHidden)
            hiddenRows.append(Self.readFP16(tappedHidden, byteOffset: end * rowBytes,
                                            count: hiddenSize))
        }

        for key in counts.keys.sorted() {
            let (shape, goldenRows) = try golden("layer3_internals", key)
            let rows = try #require(actual[key])
            #expect(rows.count == shape[0], "\(key): tapped \(rows.count) rows, golden \(shape[0])")
            let rowElements = goldenRows.count / shape[0]
            var minCosine = 2.0
            for row in rows.indices {
                #expect(rows[row].count == rowElements,
                        "\(key) row \(row): \(rows[row].count) values, golden \(rowElements)")
                let expected = Array(goldenRows[(row * rowElements)..<(row + 1) * rowElements])
                minCosine = min(minCosine, cosine(rows[row], expected))
            }
            print("S19 step 5 layer 3 \(key): min cosine \(minCosine)")
            #expect(minCosine >= 0.995, "layer 3 \(key) min cosine \(minCosine) < 0.995 (C8)")
        }

        let (shape, layer3Golden) = try golden("hidden_p0", "layer3")
        #expect(shape == [ids.count, hiddenSize])
        var minHiddenCosine = 2.0
        for row in 0..<ids.count {
            let expected = Array(layer3Golden[(row * hiddenSize)..<(row + 1) * hiddenSize])
            minHiddenCosine = min(minHiddenCosine, cosine(hiddenRows[row], expected))
        }
        print("S19 step 5 layer 3 hidden state: min cosine \(minHiddenCosine)")
        #expect(minHiddenCosine >= 0.995,
                "layer 3 hidden state min cosine \(minHiddenCosine) < 0.995 (C8)")
    }

    /// S19 step 6: the whole model, teacher-forced on the six prompts.
    /// p0's hidden state after layer 39 (cosine ≥ 0.99) and after the final
    /// norm; top-1 equal to `logits_topk.json` on at least 98% of all
    /// positions; the full last-position logits of p0 and p1 against
    /// `logits_last.json` (cosine ≥ 0.999). All bounds from C8.
    @Test("the full model matches the mlx-lm golden states and logits")
    func fullModelMatchesGolden() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let config = env.model.config
        let hiddenSize = config.hiddenSize
        let vocabSize = config.vocabSize
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))

        let topData = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/logits_topk.json"))
        let topGolden = try JSONSerialization.jsonObject(with: topData) as! [String: Any]
        let lastData = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/logits_last.json"))
        let lastGolden = try JSONSerialization.jsonObject(with: lastData) as! [String: Any]

        var positions = 0
        var equalTop1 = 0
        var overlapSum = 0
        var tokens = 0
        var elapsed: Duration = .zero
        let clock = ContinuousClock()
        for name in ["p0", "p1", "p2", "p3", "p4", "p5"] {
            let ids = try promptIDs(name)
            let entry = try #require(topGolden[name] as? [String: Any])
            let goldenTopIDs = try #require(entry["top_ids"] as? [[Int]])
            #expect(goldenTopIDs.count == ids.count)
            runner.reset()
            if name == "p0" {
                runner.debugHiddenTap = QwenHybridForwardRunner.HiddenTap(
                    points: [.afterLayer(39), .afterFinalNorm], rows: ids.count)
            } else {
                runner.debugHiddenTap = nil
            }
            var lastRow: [Float] = []
            for (position, id) in ids.enumerated() {
                let start = clock.now
                try await runner.produce(token: id, position: position, into: logits)
                elapsed += clock.now - start
                tokens += 1
                let row = Self.readFP16(logits, byteOffset: 0, count: vocabSize)
                let top = Self.topIndices(row, count: 20)
                positions += 1
                if top.first == goldenTopIDs[position].first { equalTop1 += 1 }
                overlapSum += Set(top).intersection(goldenTopIDs[position]).count
                if position == ids.count - 1 { lastRow = row }
            }
            if let lastEntry = lastGolden[name] as? [String: Any],
               let logitsEntry = lastEntry["logits"] as? [String: Any] {
                let expected = try Self.decodeF16(logitsEntry)
                let lastCosine = cosine(lastRow, expected)
                print("S19 step 6 \(name) last-position logits cosine: \(lastCosine)")
                // C8 asks 0.999. The mlx-lm golden runs with bf16
                // activations (S8: its own layer-0 reconstruction differs by
                // 3e-3 relative from bf16 rounding alone), this runner with
                // FP16, and the gap grows over 40 layers: p0 measures
                // 0.9985, p1 0.9993. The bound against these bf16 goldens
                // is 0.998, a deviation the user approved on 2026-11-11
                // (S19 report).
                #expect(lastCosine >= 0.998,
                        "\(name) last-position logits cosine \(lastCosine) < 0.998 (C8 0.999, approved deviation)")
            }
            if name == "p0" {
                let tapped = try #require(runner.tappedHidden)
                let rowBytes = hiddenSize * MemoryLayout<Float16>.size
                for (pointIndex, key, bound) in [(0, "layer39", 0.99), (1, "final_norm", 0.99)] {
                    let (shape, goldenRows) = try golden("hidden_p0", key)
                    #expect(shape == [ids.count, hiddenSize])
                    var minCosine = 2.0
                    for row in 0..<ids.count {
                        let actual = Self.readFP16(
                            tapped, byteOffset: (pointIndex * ids.count + row) * rowBytes,
                            count: hiddenSize)
                        let expected = Array(goldenRows[(row * hiddenSize)..<(row + 1) * hiddenSize])
                        minCosine = min(minCosine, cosine(actual, expected))
                    }
                    print("S19 step 6 p0 \(key) hidden state: min cosine \(minCosine)")
                    #expect(minCosine >= bound, "p0 \(key) min cosine \(minCosine) < \(bound) (C8)")
                }
            }
        }
        let share = Double(equalTop1) / Double(positions)
        let meanOverlap = Double(overlapSum) / Double(positions)
        let seconds = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) * 1e-18
        print("S19 step 6 top-1 equal on \(equalTop1)/\(positions) positions (\(share)); mean top-20 overlap \(meanOverlap)/20")
        print("S19 step 6 teacher-forced: \(tokens) tokens in \(seconds) s, \(seconds / Double(tokens) * 1000) ms/token")
        #expect(share >= 0.98, "top-1 equal on \(equalTop1)/\(positions) positions < 98% (C8)")
    }

    /// S19 step 7: greedy decoding in the test (the CLI is S22): after
    /// each prompt, 32 argmax tokens must equal `greedy.json` for at least
    /// 4 of the 6 prompts (C8); a divergence prints its step and the two
    /// logits in play, to explain it as a near-tie. The continuations of
    /// p0 and p4 are printed as text with `QwenTokenizer` for the report.
    @Test("greedy decoding matches the mlx-lm golden continuations")
    func greedyMatchesGolden() async throws {
        let env = try makeEnvironment()
        let runner = env.runner
        let vocabSize = env.model.config.vocabSize
        let logits = try #require(env.device.makeBuffer(
            length: vocabSize * MemoryLayout<Float16>.size, options: .storageModeShared))
        let tokenizerFolder = try #require(QwenTestResources.tokenizerFolder)
        let tokenizer = try await QwenTokenizer.load(from: tokenizerFolder)
        let data = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/greedy.json"))
        let golden = try JSONSerialization.jsonObject(with: data) as! [String: [Int]]

        var identical = 0
        var decodeTokens = 0
        var decodeTime: Duration = .zero
        let clock = ContinuousClock()
        for name in ["p0", "p1", "p2", "p3", "p4", "p5"] {
            let prompt = try promptIDs(name)
            let expected = try #require(golden[name])
            runner.reset()
            for (position, id) in prompt.enumerated() {
                try await runner.produce(token: id, position: position, into: logits)
            }
            var generated: [Int] = []
            var divergence: String?
            for step in 0..<expected.count {
                let row = Self.readFP16(logits, byteOffset: 0, count: vocabSize)
                let next = Self.topIndices(row, count: 1)[0]
                if divergence == nil, next != expected[step] {
                    divergence = "step \(step): actual \(next) (logit \(row[next])) vs golden \(expected[step]) (logit \(row[expected[step]]))"
                }
                generated.append(next)
                if step < expected.count - 1 {
                    let start = clock.now
                    try await runner.produce(token: Int32(next),
                                             position: prompt.count + step, into: logits)
                    decodeTime += clock.now - start
                    decodeTokens += 1
                }
            }
            if generated == expected {
                identical += 1
                print("S19 step 7 \(name): greedy identical (32 tokens)")
            } else {
                print("S19 step 7 \(name): greedy diverges at \(divergence ?? "?")")
            }
            if name == "p0" || name == "p4" {
                let text = tokenizer.decode(generated.map(Int32.init))
                print("S19 step 7 \(name) continuation: \(text.debugDescription)")
            }
        }
        let seconds = Double(decodeTime.components.seconds)
            + Double(decodeTime.components.attoseconds) * 1e-18
        print("S19 step 7 greedy identical for \(identical)/6 prompts")
        print("S19 step 7 decode: \(decodeTokens) tokens in \(seconds) s, \(seconds / Double(decodeTokens) * 1000) ms/token")
        #expect(identical >= 4, "greedy identical for \(identical)/6 prompts < 4 (C8)")
    }

    /// Indices of the `count` largest values, largest first (ties: lower
    /// index first).
    private static func topIndices(_ values: [Float], count: Int) -> [Int] {
        var best: [(index: Int, value: Float)] = []
        best.reserveCapacity(count + 1)
        for (index, value) in values.enumerated() {
            if best.count == count, let last = best.last, value <= last.value { continue }
            let slot = best.firstIndex { value > $0.value } ?? best.count
            best.insert((index, value), at: slot)
            if best.count > count { best.removeLast() }
        }
        return best.map(\.index)
    }

    /// Decode a C7 `{"shape": [...], "f16b64": base64}` entry to floats.
    private static func decodeF16(_ entry: [String: Any]) throws -> [Float] {
        let base64 = try #require(entry["f16b64"] as? String)
        let bytes = try #require(Data(base64Encoded: base64))
        return bytes.withUnsafeBytes { raw in
            raw.bindMemory(to: Float16.self).map(Float.init)
        }
    }

    private static func readFP16(_ buffer: MTLBuffer, byteOffset: Int, count: Int) -> [Float] {
        let raw = UnsafeRawBufferPointer(start: buffer.contents(), count: buffer.length)
        return (0..<count).map { i in
            Float(raw.load(fromByteOffset: byteOffset + i * MemoryLayout<Float16>.size,
                           as: Float16.self))
        }
    }

    private static func readFP32(_ buffer: MTLBuffer, byteOffset: Int, count: Int) -> [Float] {
        let raw = UnsafeRawBufferPointer(start: buffer.contents(), count: buffer.length)
        return (0..<count).map { i in
            raw.load(fromByteOffset: byteOffset + i * MemoryLayout<Float32>.size,
                     as: Float32.self)
        }
    }

    /// BF16 → Float (the Qwen gate tensors are stored as BF16), reading from
    /// `byteOffset` inside the (whole-model) buffer.
    private static func readBF16(_ buffer: MTLBuffer, byteOffset: Int, count: Int) -> [Float] {
        let raw = UnsafeRawBufferPointer(start: buffer.contents(), count: buffer.length)
        return (0..<count).map { i in
            let bits = raw.load(fromByteOffset: byteOffset + i * MemoryLayout<UInt16>.size,
                                as: UInt16.self)
            return Float(bitPattern: UInt32(bits) << 16)
        }
    }

    /// 3.5's softplus, with the usual large-argument clamp.
    private static func softplus(_ x: Double) -> Double {
        x > 20 ? x : log(1 + exp(x))
    }
}
