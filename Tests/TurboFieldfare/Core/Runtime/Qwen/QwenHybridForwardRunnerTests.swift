import Foundation
import Metal
import Testing

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
    private func golden(_ file: String, _ key: String) throws -> (shape: [Int], f32: [Float]) {
        let data = try Data(contentsOf: QwenTestResources.fixtures
            .appendingPathComponent("model/\(file).json"))
        let object = try JSONSerialization.jsonObject(with: data) as! [String: [String: Any]]
        let entry = try #require(object[key])
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
}
