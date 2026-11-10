import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the S15 DeltaNet recurrence scan kernel (`qwen_deltanet_scan`)
/// against the CPU reference in `QwenDeltaNetScanRef`.
///
/// Real shapes per plan 3.2/3.5: 16 key + 32 value heads of 128 (Hv = 32,
/// each value head reads key head j / 2), tokens = 1 / 7 / 300, the state
/// in the FP32 D5 column layout, plus the C8 check on the state after 256
/// steps and the continuity contract (prefill T1 + T2 decodes = one prefill
/// of T1 + T2) exactly as S14 pinned for the conv.
/// Tolerances per contract C8.
@Suite struct QwenDeltaNetScanTests {

    private static let keyHeads = 16
    private static let valueHeads = 32
    private static let dim = 128               // keyDim = valueDim = 128 (3.2)

    // MARK: - Buffer helpers

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

    private struct BufferAllocFailure: Error {}

    /// Realistic prep distribution: the S14 prep feeds the scan L2-normalized
    /// heads (q additionally scaled by 1/sqrt(128)), not raw projections —
    /// without the norms the recurrence explodes and FP16 saturates.
    private static func realisticQK(rng: inout SplitMix64, tokens: Int,
                                   keyHeads: Int, dim: Int)
        -> (q: [Float16], k: [Float16]) {
        let scale = Float(1.0 / (Double(dim).squareRoot()))
        var q = [Float16](), k = [Float16]()
        for _ in 0..<tokens {
            for _ in 0..<keyHeads {
                let qh = (0..<dim).map { _ in rng.uniform(-2.0, 2.0) }
                let kh = (0..<dim).map { _ in rng.uniform(-2.0, 2.0) }
                q += QwenNormRef.perHeadL2Norm(x: qh, headDim: dim, eps: 1e-6)
                    .map { Float16($0 * scale) }
                k += QwenNormRef.perHeadL2Norm(x: kh, headDim: dim, eps: 1e-6)
                    .map { Float16($0) }
            }
        }
        return (q, k)
    }

    // MARK: - Scan vs CPU reference

    /// One scan run: GPU vs reference. `o` compared after FP16 rounding of
    /// the GPU result; the final state compared in FP32.
    private static func runScan(keyHeads: Int, valueHeads: Int, dim: Int,
                                tokens: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("qwen-scan-kh\(keyHeads)-vh\(valueHeads)-t\(tokens)")
        let vDim = valueHeads * dim
        let (qFp16, kFp16) = Self.realisticQK(rng: &rng, tokens: tokens,
                                              keyHeads: keyHeads, dim: dim)
        // v as a compact [tokens][vHeads*dim] slice (vRowStride = vDim): the
        // kernel reads it through the same row-stride it sees on the raw
        // 8192-wide conv output.
        let vFp16 = (0..<tokens * vDim).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        // decay in (0, 1) — a realistic per-token forget factor; beta in (0, 1).
        let beta = (0..<tokens * valueHeads).map { _ in rng.uniform(0.0, 1.0) }
        let decay = (0..<tokens * valueHeads).map { _ in rng.uniform(0.5, 1.0) }
        let state0 = (0..<valueHeads * dim * dim).map { _ in rng.uniform(-1.0, 1.0) }

        let ctx = try MetalContext()
        let scan = try QwenDeltaNetScan(context: ctx)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp16),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kFp16),
              let vBuf = Fp16Buffer.make(ctx.device, halves: vFp16),
              let oBuf = Fp16Buffer.make(ctx.device, count: tokens * vDim) else {
            Issue.record("alloc failed"); return
        }
        let betaBuf = try makeFloat32Buffer(ctx.device, beta)
        let decayBuf = try makeFloat32Buffer(ctx.device, decay)
        let stateBuf = try makeFloat32Buffer(ctx.device, state0)

        let cb = ctx.queue.makeCommandBuffer()!
        scan.encodeScan(commandBuffer: cb, q: qBuf, k: kBuf, v: vBuf,
                        beta: betaBuf, decay: decayBuf,
                        state: stateBuf, o: oBuf,
                        tokens: tokens, valueHeads: valueHeads, keyHeads: keyHeads,
                        keyDim: dim, valueDim: dim, vRowStride: vDim)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenDeltaNetScanRef.scan(
            q: qFp16.map { Float($0) }, k: kFp16.map { Float($0) },
            v: vFp16.map { Float($0) }, beta: beta, decay: decay,
            stateIn: state0, tokens: tokens, valueHeads: valueHeads,
            keyHeads: keyHeads, keyDim: dim, valueDim: dim, vRowStride: vDim)

        let oAct = Fp16Buffer.read(oBuf, count: tokens * vDim)
        let relO = RelError.compute(actual: oAct, reference: ref.o)
        #expect(relO < Tolerance.fp16Reduction,
                "scan o kh=\(keyHeads) vh=\(valueHeads) t=\(tokens): relErr=\(relO)")

        let stateAct = readFloat32(stateBuf, count: valueHeads * dim * dim)
        let relS = RelError.compute(actual: stateAct, reference: ref.stateOut)
        #expect(relS < Tolerance.fp16Reduction,
                "scan state kh=\(keyHeads) vh=\(valueHeads) t=\(tokens): relErr=\(relS)")
    }

    @Test func scan_small() throws {
        try Self.runScan(keyHeads: 2, valueHeads: 4, dim: 128, tokens: 1, seed: 0x1501)
        try Self.runScan(keyHeads: 2, valueHeads: 4, dim: 128, tokens: 7, seed: 0x1502)
    }

    @Test func scan_real() throws {
        // Decode, short prefill, long prefill at the real geometry.
        let kh = Self.keyHeads, vh = Self.valueHeads, d = Self.dim
        try Self.runScan(keyHeads: kh, valueHeads: vh, dim: d, tokens: 1, seed: 0x1503)
        try Self.runScan(keyHeads: kh, valueHeads: vh, dim: d, tokens: 7, seed: 0x1504)
        try Self.runScan(keyHeads: kh, valueHeads: vh, dim: d, tokens: 300, seed: 0x1505)
    }

    /// C8: the FP32 DeltaNet state drifts by relative error <= 1e-4 after
    /// 256 steps (kernel fma + simd_sum vs the CPU reference's sequential
    /// chain). Real geometry, prefill of 256 tokens.
    @Test func scan_stateError256Steps() throws {
        var rng = SeedTree(0x1510).key("qwen-scan-256")
        let kh = Self.keyHeads, vh = Self.valueHeads, d = Self.dim
        let tokens = 256
        let vDim = vh * d
        let (qFp16, kFp16) = Self.realisticQK(rng: &rng, tokens: tokens,
                                              keyHeads: kh, dim: d)
        let vFp16 = (0..<tokens * vDim).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let beta = (0..<tokens * vh).map { _ in rng.uniform(0.0, 1.0) }
        let decay = (0..<tokens * vh).map { _ in rng.uniform(0.5, 1.0) }
        let state0 = (0..<vh * d * d).map { _ in rng.uniform(-1.0, 1.0) }

        let ctx = try MetalContext()
        let scan = try QwenDeltaNetScan(context: ctx)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp16),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kFp16),
              let vBuf = Fp16Buffer.make(ctx.device, halves: vFp16),
              let oBuf = Fp16Buffer.make(ctx.device, count: tokens * vDim) else {
            Issue.record("alloc failed"); return
        }
        let betaBuf = try Self.makeFloat32Buffer(ctx.device, beta)
        let decayBuf = try Self.makeFloat32Buffer(ctx.device, decay)
        let stateBuf = try Self.makeFloat32Buffer(ctx.device, state0)

        let cb = ctx.queue.makeCommandBuffer()!
        scan.encodeScan(commandBuffer: cb, q: qBuf, k: kBuf, v: vBuf,
                        beta: betaBuf, decay: decayBuf,
                        state: stateBuf, o: oBuf,
                        tokens: tokens, valueHeads: vh, keyHeads: kh,
                        keyDim: d, valueDim: d, vRowStride: vDim)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenDeltaNetScanRef.scan(
            q: qFp16.map { Float($0) }, k: kFp16.map { Float($0) },
            v: vFp16.map { Float($0) }, beta: beta, decay: decay,
            stateIn: state0, tokens: tokens, valueHeads: vh,
            keyHeads: kh, keyDim: d, valueDim: d, vRowStride: vDim)
        let stateAct = Self.readFloat32(stateBuf, count: vh * d * d)
        let relS = RelError.compute(actual: stateAct, reference: ref.stateOut)
        #expect(relS <= 1e-4, "C8 state drift after 256 steps: relErr=\(relS)")
    }

    /// Prefill of T1 tokens, then T2 single-token decode dispatches (state
    /// buffer reused in place each step, the S18 cache pattern), must equal
    /// a single prefill of T1 + T2: identical outputs, bit-identical final
    /// state — the same contract S14 pinned for the conv.
    @Test func scan_continuity() throws {
        let kh = Self.keyHeads, vh = Self.valueHeads, d = Self.dim
        let t1 = 7, t2 = 11
        let total = t1 + t2
        var rng = SeedTree(0x1520).key("qwen-scan-continuity")
        let qk = kh * d
        let vDim = vh * d
        let (qFp16, kFp16) = Self.realisticQK(rng: &rng, tokens: total,
                                              keyHeads: kh, dim: d)
        let vFp16 = (0..<total * vDim).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let beta = (0..<total * vh).map { _ in rng.uniform(0.0, 1.0) }
        let decay = (0..<total * vh).map { _ in rng.uniform(0.5, 1.0) }
        let zeroState = [Float](repeating: 0, count: vh * d * d)

        let ctx = try MetalContext()
        let scan = try QwenDeltaNetScan(context: ctx)
        let betaBuf = try Self.makeFloat32Buffer(ctx.device, beta)
        let decayBuf = try Self.makeFloat32Buffer(ctx.device, decay)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp16),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kFp16),
              let vBuf = Fp16Buffer.make(ctx.device, halves: vFp16) else {
            Issue.record("alloc failed"); return
        }
        let qkBytes = qk * MemoryLayout<Float16>.stride
        let vBytes = vDim * MemoryLayout<Float16>.stride
        let oBytes = vDim * MemoryLayout<Float16>.stride
        let betaBytes = vh * MemoryLayout<Float>.stride
        let decayBytes = vh * MemoryLayout<Float>.stride

        // Reference: one prefill of everything from the zero state.
        let ref = QwenDeltaNetScanRef.scan(
            q: qFp16.map { Float($0) }, k: kFp16.map { Float($0) },
            v: vFp16.map { Float($0) }, beta: beta, decay: decay,
            stateIn: zeroState, tokens: total, valueHeads: vh,
            keyHeads: kh, keyDim: d, valueDim: d, vRowStride: vDim)

        func encodeScan(on cb: MTLCommandBuffer, o: MTLBuffer, state: MTLBuffer,
                        token0: Int, tokens: Int) {
            scan.encodeScan(commandBuffer: cb,
                            q: qBuf, qOffset: token0 * qkBytes,
                            k: kBuf, kOffset: token0 * qkBytes,
                            v: vBuf, vOffset: token0 * vBytes,
                            beta: betaBuf, betaOffset: token0 * betaBytes,
                            decay: decayBuf, decayOffset: token0 * decayBytes,
                            state: state,
                            o: o, oOffset: token0 * oBytes,
                            tokens: tokens, valueHeads: vh, keyHeads: kh,
                            keyDim: d, valueDim: d, vRowStride: vDim)
        }

        // Path A: one prefill dispatch.
        guard let oABuf = Fp16Buffer.make(ctx.device, count: total * vDim) else {
            Issue.record("alloc failed"); return
        }
        let stateABuf = try Self.makeFloat32Buffer(ctx.device, zeroState)
        var cb = ctx.queue.makeCommandBuffer()!
        encodeScan(on: cb, o: oABuf, state: stateABuf, token0: 0, tokens: total)
        cb.commit(); cb.waitUntilCompleted()

        // Path B: prefill t1, then one decode dispatch per remaining token,
        // reading and writing the state buffer in place.
        guard let oBBuf = Fp16Buffer.make(ctx.device, count: total * vDim) else {
            Issue.record("alloc failed"); return
        }
        let stateBBuf = try Self.makeFloat32Buffer(ctx.device, zeroState)
        cb = ctx.queue.makeCommandBuffer()!
        encodeScan(on: cb, o: oBBuf, state: stateBBuf, token0: 0, tokens: t1)
        cb.commit(); cb.waitUntilCompleted()
        for step in 0..<t2 {
            let t = t1 + step
            cb = ctx.queue.makeCommandBuffer()!
            encodeScan(on: cb, o: oBBuf, state: stateBBuf, token0: t, tokens: 1)
            cb.commit(); cb.waitUntilCompleted()
        }

        let oA = Fp16Buffer.read(oABuf, count: total * vDim)
        let oB = Fp16Buffer.read(oBBuf, count: total * vDim)
        #expect(RelError.compute(actual: oA, reference: ref.o) < Tolerance.fp16Reduction,
                "continuity A: relErr=\(RelError.compute(actual: oA, reference: ref.o))")
        #expect(RelError.compute(actual: oB, reference: ref.o) < Tolerance.fp16Reduction,
                "continuity B: relErr=\(RelError.compute(actual: oB, reference: ref.o))")
        // ...and against each other bit for bit: identical arithmetic chain.
        #expect(RelError.maxAbsDiff(oA, oB) == 0,
                "A/B outputs differ, maxAbs=\(RelError.maxAbsDiff(oA, oB))")
        let sA = Self.readFloat32(stateABuf, count: vh * d * d)
        let sB = Self.readFloat32(stateBBuf, count: vh * d * d)
        #expect(RelError.maxAbsDiff(sA, sB) == 0, "final DeltaNet state differs")
    }

    // MARK: - Output stage: gated RMSNorm (S12) + out_proj GEMV + residual

    /// The linear layer's output stage (plan 3.5): per-head gated RMSNorm of
    /// `o` with `silu(z)` (S12 kernel), `out_proj` as an INT4 `DequantInt4GEMV`
    /// (4096 -> 2048 at the real geometry), then `h += W_out * o` with the
    /// S15 add. GPU chain vs CPU chain
    /// (`QwenNormRef.gatedPerHeadRMSNorm` + `DequantInt4GemvRef` + add).
    private static func runOutProj(valueHeads: Int, dim: Int, hidden: Int,
                                   seed: UInt64) throws {
        let n = valueHeads * dim          // 4096 at the real geometry
        precondition(hidden % Quantization.groupSize == 0)
        var rng = SeedTree(seed).key("qwen-outproj-vh\(valueHeads)-m\(hidden)")
        let oFp16 = (0..<n).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let zFp16 = (0..<n).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let wNorm = (0..<dim).map { _ in rng.uniform(0.5, 1.5) }
        let hFp16 = (0..<hidden).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let weightRows = (0..<hidden).map { _ in
            Quantization.quantizeInt4Affine((0..<n).map { _ in rng.uniform(-1.0, 1.0) })
        }

        let ctx = try MetalContext()
        let norms = try QwenNorms(context: ctx)
        let scan = try QwenDeltaNetScan(context: ctx)
        let gemv = try DequantInt4GEMV(context: ctx)
        guard let oBuf = Fp16Buffer.make(ctx.device, halves: oFp16),
              let zBuf = Fp16Buffer.make(ctx.device, halves: zFp16),
              let gBuf = Fp16Buffer.make(ctx.device, count: n),
              let hBuf = Fp16Buffer.make(ctx.device, halves: hFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: hidden),
              let outBuf = Fp16Buffer.make(ctx.device, count: hidden) else {
            Issue.record("alloc failed"); return
        }
        let bits = wNorm.map { Quantization.bf16Bits($0) }
        guard let wBuf = ctx.device.makeBuffer(length: bits.count * 2,
                                               options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        wBuf.contents().bindMemory(to: UInt16.self, capacity: bits.count)
            .update(from: bits, count: bits.count)
        let wRef = bits.map { Quantization.bf16ToFloat($0) }

        let packed = weightRows.flatMap(\.packed)
        let scales = weightRows.flatMap(\.scales)
        let biases = weightRows.flatMap(\.biases)
        func alloc(_ bytes: Int) -> MTLBuffer? {
            ctx.device.makeBuffer(length: bytes, options: .storageModeShared)
        }
        guard let packBuf = alloc(packed.count),
              let scaleBuf = alloc(scales.count * 2),
              let biasBuf = alloc(biases.count * 2) else {
            Issue.record("alloc failed"); return
        }
        packBuf.contents().bindMemory(to: UInt8.self, capacity: packed.count)
            .update(from: packed, count: packed.count)
        scaleBuf.contents().bindMemory(to: UInt16.self, capacity: scales.count)
            .update(from: scales, count: scales.count)
        biasBuf.contents().bindMemory(to: UInt16.self, capacity: biases.count)
            .update(from: biases, count: biases.count)

        let cb = ctx.queue.makeCommandBuffer()!
        norms.encodeGatedPerHead(commandBuffer: cb, o: oBuf, z: zBuf,
                                 weight: wBuf, out: gBuf,
                                 headDim: UInt32(dim), numHeads: valueHeads,
                                 rows: 1, eps: 1e-6)
        cb.commit(); cb.waitUntilCompleted()

        let cb2 = ctx.queue.makeCommandBuffer()!
        gemv.encode(commandBuffer: cb2,
                    weights: packBuf, scales: scaleBuf, biases: biasBuf,
                    x: gBuf, y: yBuf,
                    m: UInt32(hidden), n: UInt32(n))
        cb2.commit(); cb2.waitUntilCompleted()

        let cb3 = ctx.queue.makeCommandBuffer()!
        scan.encodeAdd(commandBuffer: cb3, a: hBuf, b: yBuf, out: outBuf, n: hidden)
        cb3.commit(); cb3.waitUntilCompleted()

        // CPU chain: gated norm (FP32) -> dequantize rows -> dot -> add.
        let oRef = oFp16.map { Float($0) }
        let gRef = QwenNormRef.gatedPerHeadRMSNorm(o: oRef, z: zFp16.map { Float($0) },
                                                   weight: wRef, headDim: dim,
                                                   eps: 1e-6)
        let yRef = DequantInt4GemvRef.apply(weightRows: weightRows, x: gRef, n: n)
        let outRef = QwenDeltaNetScanRef.residualAdd(h: hFp16.map { Float($0) },
                                                     proj: yRef)
        let outAct = Fp16Buffer.read(outBuf, count: hidden)
        let rel = RelError.compute(actual: outAct, reference: outRef)
        #expect(rel < Tolerance.quantInt4,
                "out_proj vh=\(valueHeads) m=\(hidden): relErr=\(rel)")
    }

    @Test func outProj_small() throws {
        try Self.runOutProj(valueHeads: 4, dim: 128, hidden: 256, seed: 0x1530)
    }

    @Test func outProj_real() throws {
        // 32 heads x 128 = 4096 into the real hidden size 2048 (3.2/D1).
        try Self.runOutProj(valueHeads: Self.valueHeads, dim: Self.dim,
                            hidden: 2048, seed: 0x1531)
    }
}
