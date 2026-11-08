import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The 8-bit routed decode phases selected by `MoE(routedWeightBits: 8)`:
/// gate/up/activation over all slots or a subset, then the down projection
/// with the weighted top-8 reduce, against `MoeInt8Ref`.
@Suite struct MoEInt8RoutedTests {
    typealias Fixtures = MoERoutedFixtures

    struct Shape: Sendable, CustomTestStringConvertible {
        let d: Int
        let f: Int
        let shifted: Bool

        var testDescription: String {
            "D=\(d) F=\(f) \(shifted ? "shifted" : "packed")"
        }
    }

    /// Small shapes run the unspecialized pipelines. D=128/F=64 use only the
    /// one-group remainder walk, D=256/F=256 one full 4-group block each, and
    /// D=320/F=192 a block plus a remainder group and a remainder-only F.
    static let genericShapes: [Shape] = [
        Shape(d: 128, f: 64, shifted: false),
        Shape(d: 128, f: 64, shifted: true),
        Shape(d: 256, f: 256, shifted: true),
        Shape(d: 320, f: 192, shifted: false),
        Shape(d: 320, f: 192, shifted: true),
    ]

    static let hits: [UInt32] = [0, 2, 5, 7]
    static let misses: [UInt32] = [1, 3, 4, 6]

    @Test(arguments: genericShapes)
    func genericPipelinesMatchReference(_ shape: Shape) throws {
        let experts = Fixtures.experts(bits: 8, d: shape.d, f: shape.f,
                                       seed: 0x5301, label: "int8-routed-\(shape.d)x\(shape.f)")
        let inputs = Fixtures.inputs(d: shape.d, seed: 0x5302, label: "int8-routed-inputs")
        let blobs = experts.map {
            Fixtures.blob($0, layout: shape.shifted ? .shifted : .packed)
        }
        if shape.shifted {
            #expect(blobs[0].offsets.gateWOff % 16 == 4)
            #expect(blobs[0].offsets.downWOff % 16 == 4)
            #expect(blobs[0].offsets.upSOff % 4 == 2)
        }
        let context = try MetalContext()
        let driver = try Fixtures.Driver(
            context: context, kernel: try MoE(context: context, routedWeightBits: 8),
            blobs: blobs, inputs: inputs, d: shape.d, f: shape.f)

        let full = try driver.runFull()
        let split = try driver.runSplit(hits: Self.hits, misses: Self.misses)
        let expected = Fixtures.reference(
            experts: experts, x: inputs.x, residual: inputs.residual,
            routingWeights: inputs.routingWeights, d: shape.d, f: shape.f)

        let error = RelError.compute(actual: full.y, reference: expected)
        #expect(error < Tolerance.fp16ChainedReduction, "rel=\(error)")
        #expect(split == full.y)
    }

    /// Plan §2.5 blobs at production D and F: the specialized pipelines
    /// (function constants 0-3), 16-byte-aligned rows inside a 6,324,224-byte
    /// slot, the full path and the runner's hit/miss split.
    @Test func productionBlobsMatchReference() throws {
        let d = Fixtures.Production8Bit.d
        let f = Fixtures.Production8Bit.f
        let experts = Fixtures.experts(bits: 8, d: d, f: f,
                                       seed: 0x5303, label: "int8-routed-production")
        let inputs = Fixtures.inputs(d: d, seed: 0x5304, label: "int8-routed-production-inputs")
        let blobs = experts.map { Fixtures.blob($0, layout: .production8Bit) }
        #expect(blobs.allSatisfy { $0.bytes.count == Fixtures.Production8Bit.expertStride })
        let offsets = blobs[0].offsets
        #expect([offsets.gateWOff, offsets.upWOff, offsets.downWOff]
            .allSatisfy { $0.isMultiple(of: 16) })

        let context = try MetalContext()
        let driver = try Fixtures.Driver(
            context: context, kernel: try MoE(context: context, routedWeightBits: 8),
            blobs: blobs, inputs: inputs, d: d, f: f)
        let full = try driver.runFull()
        let split = try driver.runSplit(hits: Self.hits, misses: Self.misses)
        let expected = Fixtures.reference(
            experts: experts, x: inputs.x, residual: inputs.residual,
            routingWeights: inputs.routingWeights, d: d, f: f)

        let error = RelError.compute(actual: full.y, reference: expected)
        #expect(error < Tolerance.fp16ChainedReduction, "rel=\(error)")
        #expect(split == full.y)
        let allFinite = full.y.allSatisfy { $0.isFinite }
        #expect(allFinite)
    }

    /// A subset dispatch writes the rows of its active slots and nothing
    /// else, with the same values the full dispatch produces.
    @Test func subsetWritesOnlyTheActiveSlots() throws {
        let d = 320
        let f = 192
        let experts = Fixtures.experts(bits: 8, d: d, f: f, seed: 0x5305, label: "int8-subset")
        let inputs = Fixtures.inputs(d: d, seed: 0x5306, label: "int8-subset-inputs")
        let context = try MetalContext()
        let driver = try Fixtures.Driver(
            context: context, kernel: try MoE(context: context, routedWeightBits: 8),
            blobs: experts.map { Fixtures.blob($0, layout: .shifted) },
            inputs: inputs, d: d, f: f)

        let full = try driver.runFull()
        let sentinel: Float16 = -7.5
        let acts = try driver.makeActs(filledWith: sentinel)
        let active: [UInt32] = [6, 1]
        let activeBuffer = try driver.slotsBuffer(active)
        try driver.run { commandBuffer in
            driver.encodePhase1Subset(commandBuffer, acts: acts,
                                      slots: active, slotsBuffer: activeBuffer)
        }
        let subset = Fp16Buffer.read(acts, count: Fixtures.topK * f)
        for slot in 0..<Fixtures.topK {
            let rows = (slot * f)..<((slot + 1) * f)
            if active.contains(UInt32(slot)) {
                #expect(Array(subset[rows]) == Array(full.acts[rows]), "slot \(slot)")
            } else {
                #expect(subset[rows].allSatisfy { $0 == Float(sentinel) }, "slot \(slot)")
            }
        }
    }

    /// With one-hot routing weights the reduce returns the residual plus
    /// exactly one expert's FFN, so every slot must read its own blob.
    @Test func eachSlotReadsItsOwnBlob() throws {
        let d = 256
        let f = 128
        let experts = Fixtures.experts(bits: 8, d: d, f: f, seed: 0x5307, label: "int8-one-hot")
        let inputs = Fixtures.inputs(d: d, seed: 0x5308, label: "int8-one-hot-inputs")
        let context = try MetalContext()
        let driver = try Fixtures.Driver(
            context: context, kernel: try MoE(context: context, routedWeightBits: 8),
            blobs: experts.map { Fixtures.blob($0, layout: .packed) },
            inputs: inputs, d: d, f: f)
        let acts = try driver.makeActs()
        try driver.run { driver.encodePhase1Full($0, acts: acts) }

        for slot in [0, 3, 7] {
            var weights = [Float](repeating: 0, count: Fixtures.topK)
            weights[slot] = 1
            guard let routing = Fp16Buffer.make(context.device, values: weights) else {
                throw MetalError.noDevice
            }
            let y = try driver.makeOutput()
            try driver.run { driver.encodePhase2($0, acts: acts, y: y, routing: routing) }
            let ffn = MoeInt8Ref.runFFN(
                gateRows: experts[slot].gate, upRows: experts[slot].up,
                downRows: experts[slot].down, x: inputs.x, d: d, f: f)
            let expected = zip(inputs.residual, ffn).map { $0 + $1 }
            let error = RelError.compute(actual: Fp16Buffer.read(y, count: d),
                                         reference: expected)
            #expect(error < Tolerance.fp16ChainedReduction, "slot \(slot) rel=\(error)")
        }
    }
}
