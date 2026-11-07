import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Synthetic routed-expert blobs and a small driver for the decode MoE
/// phases, shared by the 4-bit and 8-bit routed tests.
///
/// Experts come from `Quantization.quantizeInt4Affine` /
/// `quantizeInt8Affine` over seeded uniform rows. A blob lays the nine
/// sub-tensors out in the planner's order (gate, up, down × weights, scales,
/// biases) at offsets chosen by a `Layout`.
enum MoERoutedFixtures {
    static let topK = 8

    /// The 8-bit expert blob of plan §2.5 (D = 2816, F = 704).
    enum Production8Bit {
        static let d = 2816
        static let f = 704
        static let offsets = MoEExpertOffsets(
            gateWOff: 0, gateSOff: 1_982_464, gateBOff: 2_044_416,
            upWOff: 2_106_368, upSOff: 4_088_832, upBOff: 4_150_784,
            downWOff: 4_212_736, downSOff: 6_195_200, downBOff: 6_257_152)
        static let payloadBytes = 6_319_104
        static let expertStride = 6_324_224
    }

    enum Layout {
        /// Sub-tensors back to back from offset 0.
        case packed
        /// Weights at offsets ≡ 4 (mod 16) and scales and biases at offsets
        /// ≡ 2 (mod 4): the weakest alignment plan C5 lets INT8 kernels assume.
        case shifted
        /// Plan §2.5 offsets inside a 6,324,224-byte slot.
        case production8Bit
    }

    struct Expert {
        let bits: Int
        let gate: [Quantization.Int8AffineRow]
        let up: [Quantization.Int8AffineRow]
        let down: [Quantization.Int8AffineRow]
        let gate4: [Quantization.Int4AffineRow]
        let up4: [Quantization.Int4AffineRow]
        let down4: [Quantization.Int4AffineRow]
    }

    struct Blob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    static func experts(bits: Int, d: Int, f: Int, seed: UInt64, label: String) -> [Expert] {
        var rng = SeedTree(seed).key(label)
        func rows(_ count: Int, _ columns: Int) -> [[Float]] {
            (0..<count).map { _ in (0..<columns).map { _ in rng.uniform(-0.4, 0.4) } }
        }
        return (0..<topK).map { _ in
            switch bits {
            case 4:
                return Expert(
                    bits: 4, gate: [], up: [], down: [],
                    gate4: rows(f, d).map { Quantization.quantizeInt4Affine($0) },
                    up4: rows(f, d).map { Quantization.quantizeInt4Affine($0) },
                    down4: rows(d, f).map { Quantization.quantizeInt4Affine($0) })
            case 8:
                return Expert(
                    bits: 8,
                    gate: rows(f, d).map { Quantization.quantizeInt8Affine($0) },
                    up: rows(f, d).map { Quantization.quantizeInt8Affine($0) },
                    down: rows(d, f).map { Quantization.quantizeInt8Affine($0) },
                    gate4: [], up4: [], down4: [])
            default:
                preconditionFailure("unsupported bits \(bits)")
            }
        }
    }

    static func blob(_ expert: Expert, layout: Layout) -> Blob {
        typealias Parts = (weights: [UInt8], scales: [UInt16], biases: [UInt16])
        func parts8(_ rows: [Quantization.Int8AffineRow]) -> Parts {
            (rows.flatMap(\.packed), rows.flatMap(\.scales), rows.flatMap(\.biases))
        }
        func parts4(_ rows: [Quantization.Int4AffineRow]) -> Parts {
            (rows.flatMap(\.packed), rows.flatMap(\.scales), rows.flatMap(\.biases))
        }
        let projections: [Parts] = expert.bits == 8
            ? [parts8(expert.gate), parts8(expert.up), parts8(expert.down)]
            : [parts4(expert.gate4), parts4(expert.up4), parts4(expert.down4)]

        var bytes = [UInt8]()
        if case .production8Bit = layout {
            precondition(expert.bits == 8)
            bytes = [UInt8](repeating: 0, count: Production8Bit.expertStride)
        }
        func pad(modulus: Int, remainder: Int) {
            while bytes.count % modulus != remainder { bytes.append(0xA5) }
        }
        func place(_ values: [UInt8], at fixed: UInt32?) -> UInt32 {
            if let fixed {
                bytes.replaceSubrange(Int(fixed)..<(Int(fixed) + values.count), with: values)
                return fixed
            }
            let offset = UInt32(bytes.count)
            bytes.append(contentsOf: values)
            return offset
        }
        func littleEndian(_ values: [UInt16]) -> [UInt8] {
            values.flatMap { [UInt8(truncatingIfNeeded: $0), UInt8(truncatingIfNeeded: $0 >> 8)] }
        }

        let fixed = Production8Bit.offsets
        let fixedOffsets: [[UInt32]] = [
            [fixed.gateWOff, fixed.gateSOff, fixed.gateBOff],
            [fixed.upWOff, fixed.upSOff, fixed.upBOff],
            [fixed.downWOff, fixed.downSOff, fixed.downBOff],
        ]
        var offsets = [UInt32]()
        for (index, projection) in projections.enumerated() {
            let pieces = [projection.weights, littleEndian(projection.scales),
                          littleEndian(projection.biases)]
            for (role, piece) in pieces.enumerated() {
                switch layout {
                case .packed:
                    offsets.append(place(piece, at: nil))
                case .shifted:
                    if role == 0 {
                        pad(modulus: 16, remainder: 4)
                    } else {
                        pad(modulus: 4, remainder: 2)
                    }
                    offsets.append(place(piece, at: nil))
                case .production8Bit:
                    offsets.append(place(piece, at: fixedOffsets[index][role]))
                }
            }
        }
        if case .production8Bit = layout {
            precondition(Int(offsets[8]) + projections[2].biases.count * 2
                         == Production8Bit.payloadBytes)
        }
        return Blob(bytes: bytes, offsets: MoEExpertOffsets(
            gateWOff: offsets[0], gateSOff: offsets[1], gateBOff: offsets[2],
            upWOff: offsets[3], upSOff: offsets[4], upBOff: offsets[5],
            downWOff: offsets[6], downSOff: offsets[7], downBOff: offsets[8]))
    }

    /// FP32 reference of the routed half of the MoE block over all 8 slots.
    static func reference(experts: [Expert], x: [Float], residual: [Float],
                          routingWeights: [Float], d: Int, f: Int) -> [Float] {
        if experts[0].bits == 8 {
            return MoeInt8Ref.applyStreamedRouted(
                x: x, residual: residual,
                routedGate: experts.map(\.gate),
                routedUp: experts.map(\.up),
                routedDown: experts.map(\.down),
                indices: Array(0..<topK), routingWeights: routingWeights, d: d, f: f)
        }
        return MoeRef.applyStreamedRouted(
            x: x, residual: residual,
            routedGate: experts.map(\.gate4),
            routedUp: experts.map(\.up4),
            routedDown: experts.map(\.down4),
            indices: Array(0..<topK), routingWeights: routingWeights, d: d, f: f)
    }

    struct Inputs {
        let x: [Float]
        let residual: [Float]
        let routingWeights: [Float]
    }

    /// FP16-representable activations and residual, and eight distinct
    /// positive routing weights that sum to about one, as a top-8 softmax.
    static func inputs(d: Int, seed: UInt64, label: String) -> Inputs {
        var rng = SeedTree(seed).key(label)
        let x = (0..<d).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<d).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let raw = (0..<topK).map { _ in rng.uniform(0.2, 1.0) }
        let total = raw.reduce(0, +)
        return Inputs(x: x, residual: residual,
                      routingWeights: raw.map { Float(Float16($0 / total)) })
    }

    /// Encodes the routed phases of one decode step against eight blobs.
    final class Driver {
        let context: MetalContext
        let kernel: MoE
        let blobs: [Blob]
        let d: Int
        let f: Int
        let routedBuffers: [MTLBuffer]
        let argumentBuffer: MTLBuffer
        private let x: MTLBuffer
        private let residual: MTLBuffer
        private let routing: MTLBuffer

        init(context: MetalContext, kernel: MoE, blobs: [Blob],
             inputs: Inputs, d: Int, f: Int) throws {
            precondition(blobs.count == topK)
            self.context = context
            self.kernel = kernel
            self.blobs = blobs
            self.d = d
            self.f = f
            let buffers = blobs.compactMap { blob in
                blob.bytes.withUnsafeBytes {
                    context.device.makeBuffer(bytes: $0.baseAddress!, length: blob.bytes.count,
                                              options: .storageModeShared)
                }
            }
            guard buffers.count == topK,
                  let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                    routedBlobs: buffers, topK: UInt32(topK)),
                  let x = Fp16Buffer.make(context.device, values: inputs.x),
                  let residual = Fp16Buffer.make(context.device, values: inputs.residual),
                  let routing = Fp16Buffer.make(context.device, values: inputs.routingWeights)
            else { throw MetalError.noDevice }
            self.routedBuffers = buffers
            self.argumentBuffer = argumentBuffer
            self.x = x
            self.residual = residual
            self.routing = routing
        }

        var offsets: MoEExpertOffsets { blobs[0].offsets }

        func makeActs(filledWith value: Float16 = 0) throws -> MTLBuffer {
            guard let acts = Fp16Buffer.make(
                context.device,
                halves: [Float16](repeating: value, count: topK * f)) else {
                throw MetalError.noDevice
            }
            return acts
        }

        func slotsBuffer(_ slots: [UInt32]) throws -> MTLBuffer {
            guard let buffer = context.device.makeBuffer(
                bytes: slots, length: max(slots.count, 1) * MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else { throw MetalError.noDevice }
            return buffer
        }

        func encodePhase1Full(_ commandBuffer: MTLCommandBuffer, acts: MTLBuffer) {
            kernel.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: commandBuffer, routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers, routedOffsets: offsets,
                x: x, acts: acts, d: UInt32(d), f: UInt32(f), topK: UInt32(topK))
        }

        func encodePhase1Subset(_ commandBuffer: MTLCommandBuffer, acts: MTLBuffer,
                                slots: [UInt32], slotsBuffer: MTLBuffer) {
            kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: commandBuffer, routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers, routedOffsets: offsets,
                x: x, acts: acts, activeSlots: slotsBuffer,
                activeSlotIndices: slots, activeCount: UInt32(slots.count),
                d: UInt32(d), f: UInt32(f), topK: UInt32(topK))
        }

        func encodePhase2(_ commandBuffer: MTLCommandBuffer, acts: MTLBuffer,
                          y: MTLBuffer, routing: MTLBuffer? = nil) {
            kernel.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: commandBuffer, routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers, routedOffsets: offsets,
                acts: acts, routingWeights: routing ?? self.routing,
                residual: residual, y: y,
                d: UInt32(d), f: UInt32(f), topK: UInt32(topK))
        }

        func run(_ encode: (MTLCommandBuffer) throws -> Void) throws {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw MetalError.noQueue
            }
            try encode(commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            try checkCommandBufferError(commandBuffer)
        }

        func makeOutput() throws -> MTLBuffer {
            guard let y = Fp16Buffer.make(context.device, count: d) else {
                throw MetalError.noDevice
            }
            return y
        }

        /// Full phase 1 over all eight slots, then the weighted down-reduce.
        func runFull() throws -> (acts: [Float], y: [Float]) {
            let acts = try makeActs()
            let y = try makeOutput()
            try run { commandBuffer in
                encodePhase1Full(commandBuffer, acts: acts)
                encodePhase2(commandBuffer, acts: acts, y: y)
            }
            return (Fp16Buffer.read(acts, count: topK * f), Fp16Buffer.read(y, count: d))
        }

        /// The runner's hit/miss split: phase 1 over the cache hits in one
        /// command buffer, then over the misses and the down-reduce in a
        /// second one.
        func runSplit(hits: [UInt32], misses: [UInt32]) throws -> [Float] {
            let acts = try makeActs()
            let y = try makeOutput()
            let hitSlots = try slotsBuffer(hits)
            let missSlots = try slotsBuffer(misses)
            try run { commandBuffer in
                encodePhase1Subset(commandBuffer, acts: acts, slots: hits, slotsBuffer: hitSlots)
            }
            try run { commandBuffer in
                encodePhase1Subset(commandBuffer, acts: acts, slots: misses,
                                   slotsBuffer: missSlots)
                encodePhase2(commandBuffer, acts: acts, y: y)
            }
            return Fp16Buffer.read(y, count: d)
        }
    }
}
