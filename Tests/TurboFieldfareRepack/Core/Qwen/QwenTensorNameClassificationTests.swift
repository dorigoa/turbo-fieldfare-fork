import Foundation
import Testing
@testable import TurboFieldfareRepackCore

/// S6: `RepackPlanner.classify(qwen: true)` on every real tensor name of the
/// pinned checkpoint (`Tests/QwenFixtures/checkpoint/headers.json`, 2,090
/// names, saved by S1): nothing is unknown, the routed experts are exactly
/// 40 layers × 3 base names, and the 333 vision-tower tensors are excluded.
@Suite struct QwenTensorNameClassificationTests {

    /// `Tests/QwenFixtures/checkpoint/headers.json`.
    private static let fixtureHeadersPath = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Core/Qwen
        .deletingLastPathComponent()   // Core
        .deletingLastPathComponent()   // TurboFieldfareRepack
        .deletingLastPathComponent()   // Tests
        .appendingPathComponent("QwenFixtures/checkpoint/headers.json")
        .path

    private static let numLayers = 40

    private static func tensorNames() throws -> [String] {
        let data = try Data(contentsOf: URL(fileURLWithPath: fixtureHeadersPath))
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { Issue.record("headers.json is not a JSON object"); return [] }
        return object.keys.sorted()
    }

    @Test func everyRealCheckpointNameIsClassified() throws {
        let names = try Self.tensorNames()
        #expect(names.count == 2090)

        var unknown: [String] = []
        var excluded = 0
        var resident = 0
        var routed: [(role: String, layer: Int, name: String)] = []
        for name in names {
            switch RepackPlanner.classify(name, numLayers: Self.numLayers, qwen: true) {
            case .unknown:                       unknown.append(name)
            case .excludedMultimodal:            excluded += 1
            case .lmResident:                    resident += 1
            case .routedExpert(let role, let layer):
                routed.append((role, layer, name))
            }
        }

        // No name falls through to the fail-closed bucket.
        #expect(unknown.isEmpty, "unknown names: \(unknown.prefix(10))")
        // The vision tower is excluded whole.
        #expect(excluded == 333)
        // Routed experts: weight + scales + biases for 40 layers × 3 roles.
        #expect(routed.count == 360)
        let bases = Set(routed.map { $0.name }
            .filter { $0.hasSuffix(".weight") }
            .map { String($0.dropLast(".weight".count)) })
        #expect(bases.count == 40 * 3)
        // Every base is a switch_mlp projection; none is a scales/biases name.
        #expect(bases.allSatisfy { $0.contains(".mlp.switch_mlp.") })
        #expect(bases.allSatisfy { !$0.hasSuffix(".scales") && !$0.hasSuffix(".biases") })
        // Exactly one tensor per (layer, role, component): 40 × 3 × 3.
        let slots = Set(routed.map { "\($0.layer)/\($0.role)/"
            + $0.name.split(separator: ".").last.map(String.init)! })
        #expect(slots.count == 360)
        // Each layer carries gate, up and down.
        for layer in 0..<Self.numLayers {
            let roles = Set(routed.filter { $0.layer == layer }.map(\.role))
            #expect(roles == ["gate", "up", "down"], "layer \(layer) roles \(roles)")
        }
        // Everything else is a resident language-model tensor.
        #expect(resident == 2090 - 333 - 360)
    }

    @Test func theRouterNeverLeaksIntoTheExpertBucket() throws {
        // The trap from the sheet: `mlp.gate.weight` is the router, not an
        // expert; it and the shared expert stay resident.
        for suffix in ["mlp.gate.weight", "mlp.shared_expert_gate.weight",
                       "mlp.shared_expert.gate_proj.weight",
                       "mlp.shared_expert.up_proj.weight",
                       "mlp.shared_expert.down_proj.weight"] {
            let bucket = RepackPlanner.classify(
                "language_model.model.layers.7." + suffix,
                numLayers: Self.numLayers, qwen: true)
            #expect(bucket == .lmResident, "\(suffix)")
        }
        let expert = RepackPlanner.classify(
            "language_model.model.layers.7.mlp.switch_mlp.gate_proj.weight",
            numLayers: Self.numLayers, qwen: true)
        #expect(expert == .routedExpert(role: "gate", layer: 7))
    }

    @Test func theGemmaClassifierIsUnchangedForTheseNames() throws {
        // With the default `qwen: false` parameter the Gemma bucket is
        // unchanged: every language_model name is still resident and the
        // expert-role probe never fires for Qwen switch_mlp names.
        let names = try Self.tensorNames()
        for name in names where name.hasPrefix("language_model.") {
            #expect(RepackPlanner.classify(name, numLayers: Self.numLayers) == .lmResident,
                    "\(name)")
        }
    }
}
