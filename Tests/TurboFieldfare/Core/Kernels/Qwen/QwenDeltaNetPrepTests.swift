import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the S14 DeltaNet front-half kernels (conv+SiLU, beta/decay
/// gates, the 1/sqrt(128) q scale, and the q/k/v split feeding the S12 L2
/// norm) against the CPU references in `QwenDeltaNetPrepRef`.
///
/// Real shapes per plan 3.2/3.5: conv_dim = 8192 channels with a width-4
/// depthwise kernel, 16 key + 32 value heads of 128, and T = 1 / 7 / 300
/// tokens. The continuity test pins the contract the S18/S19 runners rely
/// on: prefill of T1 tokens then T2 single-token decodes must reproduce a
/// single prefill of T1 + T2 (identical outputs, identical conv state).
/// Tolerances per contract C8.
@Suite struct QwenDeltaNetPrepTests {

    private static let eps: Float = 1e-6
    private static let convDim = 8192          // channels (3.2)
    private static let keyHeads = 16
    private static let valueHeads = 32
    private static let headDim = 128

    // MARK: - Buffer helpers

    /// FP32 buffer from values.
    private static func makeFloat32Buffer(_ device: MTLDevice, _ values: [Float])
        throws -> MTLBuffer {
        guard let buf = device.makeBuffer(length: values.count * MemoryLayout<Float>.size,
                                          options: .storageModeShared) else {
            Issue.record("float32 alloc failed")
            throw BufferAllocFailure()
        }
        buf.contents().bindMemory(to: Float.self, capacity: values.count)
            .update(from: values, count: values.count)
        return buf
    }

    private static func readFloat32(_ buf: MTLBuffer, count: Int) -> [Float] {
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// BF16 buffer (norm weights, conv weight, A_log, dt_bias all share it),
    /// plus the Float values the kernel sees after the BF16 round-trip.
    private static func bf16Buffer(_ device: MTLDevice, _ values: [Float])
        throws -> (MTLBuffer, [Float]) {
        let bits = values.map { Quantization.bf16Bits($0) }
        guard let buf = device.makeBuffer(length: bits.count * 2,
                                          options: .storageModeShared) else {
            Issue.record("bf16 alloc failed")
            throw BufferAllocFailure()
        }
        let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: bits.count)
        for i in 0..<bits.count { ptr[i] = bits[i] }
        return (buf, bits.map { Quantization.bf16ToFloat($0) })
    }

    private struct BufferAllocFailure: Error {}

    // MARK: - Conv + SiLU

    /// One conv run: GPU vs reference, FP32 outputs.
    private static func runConv(channels: Int, tokens: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("qwen-conv-c\(channels)-t\(tokens)")
        let qkvFp16 = (0..<tokens * channels).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let qkvRef = qkvFp16.map { Float($0) }
        let wFp32 = (0..<channels * 4).map { _ in rng.uniform(-1.0, 1.0) }
        let stateIn = (0..<3 * channels).map { _ in rng.uniform(-2.0, 2.0) }

        let ctx = try MetalContext()
        let prep = try QwenDeltaNetPrep(context: ctx)
        let (wBuf, wRef) = try bf16Buffer(ctx.device, wFp32)
        guard let qkvBuf = Fp16Buffer.make(ctx.device, halves: qkvFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: tokens * channels) else {
            Issue.record("alloc failed"); return
        }
        let stateInBuf = try makeFloat32Buffer(ctx.device, stateIn)
        let stateOutBuf = try makeFloat32Buffer(ctx.device,
                                                [Float](repeating: 0, count: 3 * channels))

        let cb = ctx.queue.makeCommandBuffer()!
        prep.encodeConvSilu(commandBuffer: cb, qkv: qkvBuf,
                            convWeight: wBuf,
                            stateIn: stateInBuf,
                            y: yBuf,
                            stateOut: stateOutBuf,
                            channels: UInt32(channels), tokens: tokens)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenDeltaNetPrepRef.convSilu(qkv: qkvRef, tokens: tokens,
                                               channels: channels,
                                               weight: wRef, stateIn: stateIn)
        let actualY = Fp16Buffer.read(yBuf, count: tokens * channels)
        let relErr = RelError.compute(actual: actualY, reference: ref.y)
        let maxAbsY = RelError.maxAbsDiff(actualY, ref.y)
        #expect(relErr < Tolerance.fp16Reduction,
                "conv c=\(channels) t=\(tokens): relErr=\(relErr) maxAbs=\(maxAbsY)")

        // The state is a copy of FP32 inputs and FP16-rounded qkv: exact.
        let actualState = readFloat32(stateOutBuf, count: 3 * channels)
        let maxAbsS = RelError.maxAbsDiff(actualState, ref.stateOut)
        #expect(maxAbsS == 0,
                "conv state c=\(channels) t=\(tokens): maxAbs=\(maxAbsS)")
    }

    @Test func convSilu_small() throws {
        try Self.runConv(channels: 8, tokens: 1, seed: 0xE01)
        try Self.runConv(channels: 8, tokens: 7, seed: 0xE02)
    }

    @Test func convSilu_real() throws {
        // Decode (T = 1), a short prefill (T = 7) and a long one (T = 300)
        // at the full 8192-channel width (3.2).
        try Self.runConv(channels: Self.convDim, tokens: 1, seed: 0xE03)
        try Self.runConv(channels: Self.convDim, tokens: 7, seed: 0xE04)
        try Self.runConv(channels: Self.convDim, tokens: 300, seed: 0xE05)
    }

    /// Prefill of T1 tokens, then T2 single-token decode dispatches (state
    /// buffer reused in place each step), must equal a single prefill of
    /// T1 + T2 — outputs within tolerance, final state bit-identical.
    @Test func convSilu_continuity() throws {
        let channels = Self.convDim
        let t1 = 7, t2 = 11
        let total = t1 + t2
        var rng = SeedTree(0xE10).key("qwen-conv-continuity")
        let qkvFp16 = (0..<total * channels).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let wFp32 = (0..<channels * 4).map { _ in rng.uniform(-1.0, 1.0) }
        let zeroState = [Float](repeating: 0, count: 3 * channels)

        let ctx = try MetalContext()
        let prep = try QwenDeltaNetPrep(context: ctx)
        let (wBuf, wRef) = try Self.bf16Buffer(ctx.device, wFp32)
        guard let qkvBuf = Fp16Buffer.make(ctx.device, halves: qkvFp16) else {
            Issue.record("alloc failed"); return
        }
        let strideBytes = channels * MemoryLayout<Float16>.stride

        // Reference: one prefill of everything from the zero state.
        let ref = QwenDeltaNetPrepRef.convSilu(
            qkv: qkvFp16.map { Float($0) }, tokens: total, channels: channels,
            weight: wRef, stateIn: zeroState)

        // Path A: one prefill dispatch.
        guard let yABuf = Fp16Buffer.make(ctx.device, count: total * channels) else {
            Issue.record("alloc failed"); return
        }
        let stateABuf = try Self.makeFloat32Buffer(ctx.device, zeroState)
        var cb = ctx.queue.makeCommandBuffer()!
        prep.encodeConvSilu(commandBuffer: cb, qkv: qkvBuf, convWeight: wBuf,
                            stateIn: stateABuf, y: yABuf, stateOut: stateABuf,
                            channels: UInt32(channels), tokens: total)
        cb.commit(); cb.waitUntilCompleted()

        // Path B: prefill t1, then one decode dispatch per remaining token,
        // reading and writing the state buffer in place (same runtime
        // pattern the S18 cache will use).
        guard let yBBuf = Fp16Buffer.make(ctx.device, count: total * channels) else {
            Issue.record("alloc failed"); return
        }
        let stateBBuf = try Self.makeFloat32Buffer(ctx.device, zeroState)
        cb = ctx.queue.makeCommandBuffer()!
        prep.encodeConvSilu(commandBuffer: cb, qkv: qkvBuf, convWeight: wBuf,
                            stateIn: stateBBuf, y: yBBuf, stateOut: stateBBuf,
                            channels: UInt32(channels), tokens: t1)
        cb.commit(); cb.waitUntilCompleted()
        for step in 0..<t2 {
            let t = t1 + step
            cb = ctx.queue.makeCommandBuffer()!
            prep.encodeConvSilu(commandBuffer: cb,
                                qkv: qkvBuf, qkvOffset: t * strideBytes,
                                convWeight: wBuf,
                                stateIn: stateBBuf,
                                y: yBBuf, yOffset: t * strideBytes,
                                stateOut: stateBBuf,
                                channels: UInt32(channels), tokens: 1)
            cb.commit(); cb.waitUntilCompleted()
        }

        let yA = Fp16Buffer.read(yABuf, count: total * channels)
        let yB = Fp16Buffer.read(yBBuf, count: total * channels)
        // Both paths against the single-prefill reference...
        #expect(RelError.compute(actual: yA, reference: ref.y) < Tolerance.fp16Reduction,
                "continuity A: relErr=\(RelError.compute(actual: yA, reference: ref.y))")
        #expect(RelError.compute(actual: yB, reference: ref.y) < Tolerance.fp16Reduction,
                "continuity B: relErr=\(RelError.compute(actual: yB, reference: ref.y))")
        // ...and against each other bit for bit: identical arithmetic chain.
        #expect(RelError.maxAbsDiff(yA, yB) == 0,
                "A/B outputs differ, maxAbs=\(RelError.maxAbsDiff(yA, yB))")
        let sA = Self.readFloat32(stateABuf, count: 3 * channels)
        let sB = Self.readFloat32(stateBBuf, count: 3 * channels)
        #expect(RelError.maxAbsDiff(sA, sB) == 0, "final conv state differs")
        #expect(RelError.maxAbsDiff(sA, ref.stateOut) == 0, "conv state vs reference")
    }

    // MARK: - Gates

    private static func runGates(heads: Int, tokens: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("qwen-gates-h\(heads)-t\(tokens)")
        let aFp16 = (0..<tokens * heads).map { _ in Float16(rng.uniform(-6.0, 4.0)) }
        let bFp16 = (0..<tokens * heads).map { _ in Float16(rng.uniform(-6.0, 6.0)) }
        let aLogFp32 = (0..<heads).map { _ in rng.uniform(-8.0, 1.0) }
        let dtBiasFp32 = (0..<heads).map { _ in rng.uniform(-3.0, 1.0) }

        let ctx = try MetalContext()
        let prep = try QwenDeltaNetPrep(context: ctx)
        let (aLogBuf, aLogRef) = try bf16Buffer(ctx.device, aLogFp32)
        let (dtBuf, dtRef) = try bf16Buffer(ctx.device, dtBiasFp32)
        guard let aBuf = Fp16Buffer.make(ctx.device, halves: aFp16),
              let bBuf = Fp16Buffer.make(ctx.device, halves: bFp16) else {
            Issue.record("alloc failed"); return
        }
        let betaBuf = try makeFloat32Buffer(ctx.device,
                                            [Float](repeating: 0, count: tokens * heads))
        let decayBuf = try makeFloat32Buffer(ctx.device,
                                             [Float](repeating: 0, count: tokens * heads))

        let cb = ctx.queue.makeCommandBuffer()!
        prep.encodeGates(commandBuffer: cb, a: aBuf, b: bBuf,
                         aLog: aLogBuf, dtBias: dtBuf,
                         betaOut: betaBuf, decayOut: decayBuf,
                         heads: UInt32(heads), tokens: tokens)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenDeltaNetPrepRef.gates(a: aFp16.map { Float($0) },
                                            b: bFp16.map { Float($0) },
                                            aLog: aLogRef, dtBias: dtRef)
        let beta = readFloat32(betaBuf, count: tokens * heads)
        let decay = readFloat32(decayBuf, count: tokens * heads)
        let relBeta = RelError.compute(actual: beta, reference: ref.beta)
        let relDecay = RelError.compute(actual: decay, reference: ref.decay)
        #expect(relBeta < Tolerance.fp16Reduction,
                "gates h=\(heads) t=\(tokens) beta: relErr=\(relBeta)")
        #expect(relDecay < Tolerance.fp16Reduction,
                "gates h=\(heads) t=\(tokens) decay: relErr=\(relDecay)")
        // Sanity on the formulas: beta in (0,1), decay = exp(g) with g <= 0.
        #expect(beta.allSatisfy { $0 > 0 && $0 < 1 }, "beta range")
        #expect(decay.allSatisfy { $0 > 0 && $0 <= 1 }, "decay range")
    }

    @Test func gates_small() throws {
        try Self.runGates(heads: 4, tokens: 3, seed: 0xF01)
    }

    @Test func gates_real() throws {
        // 32 value heads, decode / short prefill / long prefill.
        try Self.runGates(heads: Self.valueHeads, tokens: 1, seed: 0xF02)
        try Self.runGates(heads: Self.valueHeads, tokens: 7, seed: 0xF03)
        try Self.runGates(heads: Self.valueHeads, tokens: 300, seed: 0xF04)
    }

    // MARK: - q/k/v split + L2 norm + 1/sqrt(128)

    /// The conv output slice [tokens][8192] (16384 bytes per row) feeds S15
    /// as: q = bytes 0..4096, k = bytes 4096..8192 (16 heads x 128 each),
    /// L2-normalized with the S12 kernel; q additionally scaled by
    /// 1/sqrt(128) with the S14 scale kernel; v = bytes 8192..16384
    /// untouched.
    @Test func prepSplitL2Scale() throws {
        let channels = Self.convDim
        let keyHeads = Self.keyHeads
        let headDim = Self.headDim
        let eps = Self.eps
        let scale = Float(1.0 / (128.0).squareRoot())
        let rowsList = [1, 7, 300]
        for (i, rows) in rowsList.enumerated() {
            var rng = SeedTree(UInt64(0xA10 + i)).key("qwen-prep-t\(rows)")
            let yFp16 = (0..<rows * channels).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
            let yRef = yFp16.map { Float($0) }

            let ctx = try MetalContext()
            let prep = try QwenDeltaNetPrep(context: ctx)
            let norms = try QwenNorms(context: ctx)
            guard let yBuf = Fp16Buffer.make(ctx.device, halves: yFp16),
                  let qBuf = Fp16Buffer.make(ctx.device, count: rows * keyHeads * headDim),
                  let kBuf = Fp16Buffer.make(ctx.device, count: rows * keyHeads * headDim) else {
                Issue.record("alloc failed"); return
            }

            let qBytes = keyHeads * headDim * MemoryLayout<Float16>.stride  // 4096
            let kByteOffset = qBytes                                        // after q
            // Kernels inside one encoder may overlap: q gets its L2 norm and
            // its scale in two command buffers, the same way the runner will.
            // QwenNorms.encodeL2PerHead assumes a compact [rows][heads*headDim]
            // x layout, so the token rows of the 8192-wide conv output are
            // dispatched one row at a time (row stride 16384 bytes). S18/S19
            // may batch these per-row dispatches.
            let rowBytes = channels * MemoryLayout<Float16>.stride          // 16384
            let cb = ctx.queue.makeCommandBuffer()!
            for r in 0..<rows {
                norms.encodeL2PerHead(commandBuffer: cb, x: yBuf,
                                      xOffset: r * rowBytes,
                                      out: qBuf, outOffset: r * qBytes,
                                      headDim: UInt32(headDim), numHeads: keyHeads,
                                      rows: 1, eps: eps)
                norms.encodeL2PerHead(commandBuffer: cb, x: yBuf,
                                      xOffset: r * rowBytes + kByteOffset,
                                      out: kBuf, outOffset: r * qBytes,
                                      headDim: UInt32(headDim), numHeads: keyHeads,
                                      rows: 1, eps: eps)
            }
            cb.commit(); cb.waitUntilCompleted()
            let cb2 = ctx.queue.makeCommandBuffer()!
            prep.encodeScale(commandBuffer: cb2, x: qBuf, out: qBuf,
                             scale: scale,
                             n: rows * keyHeads * headDim)
            cb2.commit(); cb2.waitUntilCompleted()

            // Reference: slice, L2 (sum, eps), scale q.
            var qHead = [Float](), kHead = [Float]()
            for r in 0..<rows {
                qHead += QwenNormRef.perHeadL2Norm(
                    x: Array(yRef[r * channels ..< r * channels + keyHeads * headDim]),
                    headDim: headDim, eps: eps)
                let kStart = r * channels + keyHeads * headDim
                kHead += QwenNormRef.perHeadL2Norm(
                    x: Array(yRef[kStart ..< kStart + keyHeads * headDim]),
                    headDim: headDim, eps: eps)
            }
            let qRef = qHead.map { $0 * scale }
            let qAct = Fp16Buffer.read(qBuf, count: rows * keyHeads * headDim)
            let kAct = Fp16Buffer.read(kBuf, count: rows * keyHeads * headDim)
            #expect(RelError.compute(actual: qAct, reference: qRef) < Tolerance.fp16Reduction,
                    "prep q t=\(rows): relErr=\(RelError.compute(actual: qAct, reference: qRef))")
            #expect(RelError.compute(actual: kAct, reference: kHead) < Tolerance.fp16Reduction,
                    "prep k t=\(rows): relErr=\(RelError.compute(actual: kAct, reference: kHead))")
        }
    }
}
