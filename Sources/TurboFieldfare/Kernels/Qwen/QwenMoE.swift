import Foundation
import Metal

/// Swift wrapper for the Qwen 3.5 MoE kernels in `qwen_moe.metal`
/// (plan S17, math from 3.5 "Blocco MoE", blob layout from contract C5):
///
///   gateUpSilu  act[slot][i]  = silu(gate_i . x) * (up_i . x)     INT4
///   down        routed[slot][d] = down_d . act[slot]              INT4
///   residual    y[d] = residual[d] + sum_k w_k * routed[k][d]
///                              + sg * shared[d]
///
/// Expert weights arrive as one blob buffer per streamed expert slot (the
/// `PreadExpertStreamer` slot buffers) or, for the shared expert, one
/// resident buffer; `ExpertOffsets` carries the C5 intra-blob byte offsets
/// (gate, up, down x weight, scales, biases). The blob pointers are handed to
/// the kernels through an argument buffer, same mechanism as Gemma's
/// `MoE.swift` (`makeArgumentEncoder(bufferIndex: 0)`).
///
/// Shapes: routed experts are `hidden` = 2048 in, `inter` = 512 intermediate
/// (3.4); the shared expert has the same shape and runs as a one-slot pass.
/// `w` is the FP32 [slots] renormalized router weight vector and `sg` the
/// FP16 [1] scalar gate, both produced by the S13 router chain.
///
/// Prefill is a per-token loop over these three kernels (correct before
/// fast); each call handles exactly one token.
final class QwenMoE {

    /// Streamed routed experts per token (S13 router ships exactly 8).
    static let maxSlots = 8

    /// C5 intra-blob byte offsets for one expert. Mirrors the
    /// `QwenExpertOffsets` struct in `qwen_moe.metal` (nine UInt32).
    struct ExpertOffsets {
        var gateWOff: UInt32
        var gateSOff: UInt32
        var gateBOff: UInt32
        var upWOff: UInt32
        var upSOff: UInt32
        var upBOff: UInt32
        var downWOff: UInt32
        var downSOff: UInt32
        var downBOff: UInt32
    }

    private let psoGateUpSilu: MTLComputePipelineState
    private let psoDown: MTLComputePipelineState
    private let psoResidual: MTLComputePipelineState
    private let argEncoder: MTLArgumentEncoder

    init(context: MetalContext) throws {
        self.psoGateUpSilu = try context.pipeline("qwen_moe_gate_up_silu")
        self.psoDown = try context.pipeline("qwen_moe_down")
        self.psoResidual = try context.pipeline("qwen_moe_residual")
        guard let gateUpFunction = context.library.makeFunction(
            name: "qwen_moe_gate_up_silu") else {
            throw MetalError.noDevice
        }
        self.argEncoder = gateUpFunction.makeArgumentEncoder(bufferIndex: 0)
    }

    /// C5 offsets for an expert with `hidden` inputs and `inter`
    /// intermediates, INT4-affine with group size 64:
    /// weight = rows * cols / 2 bytes, scales/biases = rows * cols / 64 BF16.
    /// For Qwen 3.5 (2048, 512) each matrix is 589.824 B and the full blob
    /// stride is 1.769.472 B (3.4).
    static func expertOffsets(hidden: Int, inter: Int) -> ExpertOffsets {
        precondition(hidden % Quantization.groupSize == 0
                     && inter % Quantization.groupSize == 0,
                     "hidden and inter must be multiples of the group size")
        let gateW = inter * hidden / 2
        let gateS = inter * (hidden / Quantization.groupSize) * 2
        let gateTotal = gateW + 2 * gateS
        let downW = hidden * inter / 2
        let downS = hidden * (inter / Quantization.groupSize) * 2
        return ExpertOffsets(
            gateWOff: 0,
            gateSOff: UInt32(gateW),
            gateBOff: UInt32(gateW + gateS),
            upWOff: UInt32(gateTotal),
            upSOff: UInt32(gateTotal + downW),
            upBOff: UInt32(gateTotal + downW + downS),
            downWOff: UInt32(2 * gateTotal),
            downSOff: UInt32(2 * gateTotal + downW),
            downBOff: UInt32(2 * gateTotal + downW + downS))
    }

    /// Total bytes of one expert blob (== the `.gturbo` `expertStride`).
    static func expertBlobBytes(hidden: Int, inter: Int) -> Int {
        let gateW = inter * hidden / 2
        let gateS = inter * (hidden / Quantization.groupSize) * 2
        let downS = hidden * (inter / Quantization.groupSize) * 2
        return 3 * gateW + 4 * gateS + 2 * downS
    }

    /// Argument buffer carrying one blob pointer per slot (index 0...slots-1;
    /// unused slots get the first blob, the kernels never read them).
    func makeBlobsArgumentBuffer(blobs: [MTLBuffer], slots: Int) -> MTLBuffer? {
        precondition(slots >= 1 && slots <= Self.maxSlots,
                     "slots must be 1...\(Self.maxSlots), got \(slots)")
        precondition(blobs.count == slots,
                     "expected \(slots) blob buffers, got \(blobs.count)")
        guard let buffer = blobs.first?.device.makeBuffer(
            length: argEncoder.encodedLength, options: .storageModeShared)
        else { return nil }
        argEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in blobs.enumerated() {
            argEncoder.setBuffer(blob, offset: 0, index: index)
        }
        return buffer
    }

    /// Stage 1a: SiLU-multiply activations for `slots` experts, one token.
    /// `acts` holds `slots * inter` FP16 values.
    func encodeGateUpSilu(commandBuffer: MTLCommandBuffer,
                          blobsArgBuffer: MTLBuffer,
                          blobs: [MTLBuffer],
                          offsets: ExpertOffsets,
                          x: MTLBuffer, xOffset: Int = 0,
                          acts: MTLBuffer, actsOffset: Int = 0,
                          hidden: Int, inter: Int, slots: Int) {
        precondition(hidden % Quantization.groupSize == 0
                     && inter % Quantization.groupSize == 0)
        precondition(slots >= 1 && slots <= Self.maxSlots
                     && blobs.count == slots)
        var hiddenVar = UInt32(hidden)
        var interVar = UInt32(inter)
        var slotsVar = UInt32(slots)
        var offs = offsets
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGateUpSilu)
        enc.setBuffer(blobsArgBuffer, offset: 0, index: 0)
        for blob in blobs { enc.useResource(blob, usage: .read) }
        enc.setBytes(&offs, length: MemoryLayout<ExpertOffsets>.stride, index: 1)
        enc.setBuffer(x, offset: xOffset, index: 2)
        enc.setBuffer(acts, offset: actsOffset, index: 3)
        enc.setBytes(&hiddenVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&interVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&slotsVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: (slots * inter + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Stage 1b: down projection for `slots` experts, one token.
    /// `routedOut` holds `slots * hidden` FP16 values, `acts` comes from
    /// `encodeGateUpSilu` (same slots/offsets).
    func encodeDown(commandBuffer: MTLCommandBuffer,
                    blobsArgBuffer: MTLBuffer,
                    blobs: [MTLBuffer],
                    offsets: ExpertOffsets,
                    acts: MTLBuffer, actsOffset: Int = 0,
                    routedOut: MTLBuffer, routedOutOffset: Int = 0,
                    hidden: Int, inter: Int, slots: Int) {
        precondition(hidden % Quantization.groupSize == 0
                     && inter % Quantization.groupSize == 0)
        precondition(slots >= 1 && slots <= Self.maxSlots
                     && blobs.count == slots)
        var hiddenVar = UInt32(hidden)
        var interVar = UInt32(inter)
        var slotsVar = UInt32(slots)
        var offs = offsets
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoDown)
        enc.setBuffer(blobsArgBuffer, offset: 0, index: 0)
        for blob in blobs { enc.useResource(blob, usage: .read) }
        enc.setBytes(&offs, length: MemoryLayout<ExpertOffsets>.stride, index: 1)
        enc.setBuffer(acts, offset: actsOffset, index: 2)
        enc.setBuffer(routedOut, offset: routedOutOffset, index: 3)
        enc.setBytes(&hiddenVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&interVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&slotsVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: (slots * hidden + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Stage 2: MoE block output — weighted sum of the routed expert
    /// projections plus `sg` times the shared expert projection, summed onto
    /// the residual. One token; `routingW` is Float32 [slots], `sg` FP16 [1]
    /// (both from the S13 router chain), `sharedOut` holds `hidden` FP16.
    func encodeResidual(commandBuffer: MTLCommandBuffer,
                        residual: MTLBuffer, residualOffset: Int = 0,
                        routedOut: MTLBuffer, routedOutOffset: Int = 0,
                        sharedOut: MTLBuffer, sharedOutOffset: Int = 0,
                        routingW: MTLBuffer, routingWOffset: Int = 0,
                        sg: MTLBuffer, sgOffset: Int = 0,
                        y: MTLBuffer, yOffset: Int = 0,
                        hidden: Int, slots: Int) {
        precondition(slots >= 1 && slots <= Self.maxSlots)
        var hiddenVar = UInt32(hidden)
        var slotsVar = UInt32(slots)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoResidual)
        enc.setBuffer(residual, offset: residualOffset, index: 0)
        enc.setBuffer(routedOut, offset: routedOutOffset, index: 1)
        enc.setBuffer(sharedOut, offset: sharedOutOffset, index: 2)
        enc.setBuffer(routingW, offset: routingWOffset, index: 3)
        enc.setBuffer(sg, offset: sgOffset, index: 4)
        enc.setBuffer(y, offset: yOffset, index: 5)
        enc.setBytes(&hiddenVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&slotsVar, length: MemoryLayout<UInt32>.size, index: 7)
        enc.dispatchThreads(MTLSize(width: hidden, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256,
                                                           height: 1, depth: 1))
        enc.endEncoding()
    }
}
