import Foundation
import Metal
import Testing
import TurboFieldfareValidationSupport

@testable import TurboFieldfare

/// S19 step 2: the runner skeleton (embedding → final norm → lm_head, no
/// layers) must embed p0's five tokens like the mlx-lm golden
/// `hidden_p0.json → embedding` (C8: cosine ≥ 0.9999 per row).
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
