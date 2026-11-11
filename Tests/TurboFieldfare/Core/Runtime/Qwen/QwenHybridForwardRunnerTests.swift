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
