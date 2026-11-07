import Accelerate
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Synthetic MLX affine 8-bit routed experts for the INT8 prefill MoE tests.
///
/// Every expert is a blob laid out like an installed `layer_XX.bin` entry
/// (plan §2.5): gate, up and down, each as weights, BF16 scales and BF16
/// biases, back to back, with the stride rounded up to 16 KiB. All blobs live
/// in one shared `MTLBuffer` after `leadingBytes`, so a view's offset is
/// `leadingBytes + slot * stride`, like the slots of the pread streamer. The
/// bytes are written straight into the buffer; the CPU reference reads them
/// back from there, so no copy of the weights stays on the Swift heap.
///
/// `@unchecked Sendable` like `TensorView`: the pool is immutable after
/// `init`, and the reference reads its buffer from concurrent workers.
struct PrefillInt8ExpertPool: @unchecked Sendable {
    /// Hands a pointer to `concurrentPerform` workers that each write a
    /// disjoint range behind it.
    private struct DisjointWrites<Pointer>: @unchecked Sendable {
        let pointer: Pointer
    }

    struct Layout: Equatable, @unchecked Sendable {
        let d: Int
        let f: Int
        let offsets: MoEExpertOffsets
        let payloadBytes: Int
        let stride: Int

        static let strideAlignment = 16 * 1024

        init(d: Int, f: Int) {
            precondition(d % Quantization.groupSize == 0 && f % Quantization.groupSize == 0)
            self.d = d
            self.f = f
            let gateUpWeights = f * d
            let gateUpGroups = f * (d / Quantization.groupSize) * 2
            let downWeights = d * f
            let downGroups = d * (f / Quantization.groupSize) * 2
            var cursor = 0
            func take(_ bytes: Int) -> UInt32 {
                defer { cursor += bytes }
                return UInt32(cursor)
            }
            self.offsets = MoEExpertOffsets(gateWOff: take(gateUpWeights),
                                            gateSOff: take(gateUpGroups),
                                            gateBOff: take(gateUpGroups),
                                            upWOff: take(gateUpWeights),
                                            upSOff: take(gateUpGroups),
                                            upBOff: take(gateUpGroups),
                                            downWOff: take(downWeights),
                                            downSOff: take(downGroups),
                                            downBOff: take(downGroups))
            self.payloadBytes = cursor
            self.stride = (cursor + Self.strideAlignment - 1)
                / Self.strideAlignment * Self.strideAlignment
        }

        static func == (lhs: Layout, rhs: Layout) -> Bool {
            lhs.d == rhs.d && lhs.f == rhs.f
                && lhs.payloadBytes == rhs.payloadBytes && lhs.stride == rhs.stride
                && PrefillInt8ExpertPool.offsetList(lhs.offsets)
                    == PrefillInt8ExpertPool.offsetList(rhs.offsets)
        }
    }

    enum Values {
        /// Weights drawn from `uniform(-range, range)` with `SeedTree` and
        /// quantized row by row with `Quantization.quantizeInt8Affine`.
        case quantized(seed: UInt64, range: Float)
        /// Every byte value 0...255 across each row, with BF16 scales and
        /// biases that are negative, tiny and subnormal.
        case adversarial
    }

    let layout: Layout
    let expertCount: Int
    let leadingBytes: Int
    let buffer: MTLBuffer

    var d: Int { layout.d }
    var f: Int { layout.f }
    var offsets: MoEExpertOffsets { layout.offsets }

    init(device: MTLDevice,
         d: Int,
         f: Int,
         expertCount: Int,
         values: Values,
         leadingBytes: Int = 0) throws {
        let layout = Layout(d: d, f: f)
        let length = leadingBytes + expertCount * layout.stride
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("INT8 expert pool")
        }
        buffer.label = "test.prefill.int8ExpertPool"
        memset(buffer.contents(), 0xC3, leadingBytes)
        self.layout = layout
        self.expertCount = expertCount
        self.leadingBytes = leadingBytes
        self.buffer = buffer
        // Each (expert, role) has its own `SeedTree` stream and its own bytes,
        // so the projections can be written concurrently with the same result.
        let contents = DisjointWrites(pointer: buffer.contents())
        let blobOffsets = (0..<expertCount).map { leadingBytes + $0 * layout.stride }
        DispatchQueue.concurrentPerform(iterations: expertCount * 3) { index in
            let expert = index / 3
            let role = index % 3
            let base = contents.pointer.advanced(by: blobOffsets[expert])
            if role == 0 {
                memset(base.advanced(by: layout.payloadBytes), 0,
                       layout.stride - layout.payloadBytes)
            }
            Self.writeProjection(role: role, expert: expert, layout: layout,
                                 values: values, blob: base)
        }
    }

    func blobOffset(_ expert: Int) -> Int {
        leadingBytes + expert * layout.stride
    }

    /// A view of `expert`'s blob, shaped like the views the streamer returns.
    func view(expert: Int) -> TensorView {
        TensorView(buffer: buffer,
                   offset: UInt64(blobOffset(expert)),
                   length: UInt64(layout.stride),
                   scaleOffset: 0,
                   scaleLength: 0,
                   biasOffset: 0,
                   biasLength: 0,
                   shape: (0, UInt32(expert), 0, 0),
                   dtype: 0)
    }

    func binding(expertIDs: [Int]) throws -> PrefillStreamedTileBinding {
        try PrefillStreamedTileBinding(expertIDs: expertIDs,
                                       views: expertIDs.map { view(expert: $0) })
    }

    static func offsetList(_ o: MoEExpertOffsets) -> [UInt32] {
        [o.gateWOff, o.gateSOff, o.gateBOff, o.upWOff, o.upSOff, o.upBOff,
         o.downWOff, o.downSOff, o.downBOff]
    }

    // MARK: - Generation

    /// role 0 = gate (f x d), 1 = up (f x d), 2 = down (d x f).
    private static func writeProjection(role: Int,
                                        expert: Int,
                                        layout: Layout,
                                        values: Values,
                                        blob: UnsafeMutableRawPointer) {
        let (rows, columns) = role == 2 ? (layout.d, layout.f) : (layout.f, layout.d)
        let groups = columns / Quantization.groupSize
        let (wOff, sOff, bOff) = Self.roleOffsets(role, layout.offsets)
        let weights = blob.advanced(by: Int(wOff)).assumingMemoryBound(to: UInt8.self)
        let scales = blob.advanced(by: Int(sOff))
        let biases = blob.advanced(by: Int(bOff))
        switch values {
        case .quantized(let seed, let range):
            var rng = SeedTree(seed).key("prefill-moe-int8-e\(expert)-r\(role)-\(rows)x\(columns)")
            var raw = [Float](repeating: 0, count: columns)
            for row in 0..<rows {
                for index in 0..<columns {
                    raw[index] = rng.uniform(-range, range)
                }
                let quantized = Quantization.quantizeInt8Affine(raw)
                quantized.packed.withUnsafeBufferPointer { packed in
                    (weights + row * columns).update(from: packed.baseAddress!, count: columns)
                }
                for group in 0..<groups {
                    Self.storeU16(quantized.scales[group], scales, row * groups + group)
                    Self.storeU16(quantized.biases[group], biases, row * groups + group)
                }
            }
        case .adversarial:
            let scaleTable: [UInt16] = [
                Quantization.bf16Bits(0.0004),
                Quantization.bf16Bits(-0.0006),
                0x0001,
                0x8001,
            ]
            let biasTable: [UInt16] = [
                Quantization.bf16Bits(-0.05),
                Quantization.bf16Bits(0.03),
                0x0001,
                0x8001,
            ]
            for row in 0..<rows {
                for column in 0..<columns {
                    weights[row * columns + column] =
                        UInt8(truncatingIfNeeded: (row * columns + column) &* 37
                                &+ 0x29 &+ expert &* 11 &+ role &* 5)
                }
                for group in 0..<groups {
                    Self.storeU16(scaleTable[(row + group + expert) % scaleTable.count],
                                  scales, row * groups + group)
                    Self.storeU16(biasTable[(row * 3 + group + role) % biasTable.count],
                                  biases, row * groups + group)
                }
            }
        }
    }

    private static func roleOffsets(_ role: Int,
                                    _ o: MoEExpertOffsets) -> (UInt32, UInt32, UInt32) {
        switch role {
        case 0: (o.gateWOff, o.gateSOff, o.gateBOff)
        case 1: (o.upWOff, o.upSOff, o.upBOff)
        default: (o.downWOff, o.downSOff, o.downBOff)
        }
    }

    private static func storeU16(_ value: UInt16, _ base: UnsafeMutableRawPointer, _ index: Int) {
        base.storeBytes(of: value, toByteOffset: index * 2, as: UInt16.self)
    }

    // MARK: - CPU reference

    /// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` with the kernel's clamp of
    /// the tanh argument to ±20.
    static func gelu(_ x: Float) -> Float {
        let inner = min(max(0.7978845608028654 * (x + 0.044715 * x * x * x), -20), 20)
        return 0.5 * x * (1 + tanh(inner))
    }

    /// Dequantizes one projection of `expert` to FP32, row-major
    /// `[rows * columns]`.
    func dequantized(expert: Int, role: Int) -> [Float] {
        let (rows, columns) = role == 2 ? (d, f) : (f, d)
        let groups = columns / Quantization.groupSize
        let (wOff, sOff, bOff) = Self.roleOffsets(role, offsets)
        let blob = UnsafeRawPointer(buffer.contents()).advanced(by: blobOffset(expert))
        let weights = blob.advanced(by: Int(wOff)).assumingMemoryBound(to: UInt8.self)
        var out = [Float](repeating: 0, count: rows * columns)
        out.withUnsafeMutableBufferPointer { w in
            for row in 0..<rows {
                for group in 0..<groups {
                    let index = row * groups + group
                    var scale = Quantization.bf16ToFloat(
                        blob.loadUnaligned(fromByteOffset: Int(sOff) + index * 2, as: UInt16.self))
                    var bias = Quantization.bf16ToFloat(
                        blob.loadUnaligned(fromByteOffset: Int(bOff) + index * 2, as: UInt16.self))
                    let start = row * columns + group * Quantization.groupSize
                    vDSP_vfltu8(weights + start, 1, w.baseAddress! + start, 1,
                                vDSP_Length(Quantization.groupSize))
                    vDSP_vsmsa(w.baseAddress! + start, 1, &scale, &bias,
                               w.baseAddress! + start, 1,
                               vDSP_Length(Quantization.groupSize))
                }
            }
        }
        return out
    }

    /// FP32 reference of the two batched phases for `pairs`, laid out like
    /// `route_partials`: element `(token * topK + rank) * d + row`. Elements
    /// no pair covers are NaN.
    ///
    /// Gate and up are FP32 sums; the activation `gelu(gate) · up` is rounded
    /// to FP16 because the kernels hand it to the down phase through the FP16
    /// scratch; the down projection stays FP32. Each expert is dequantized
    /// once with vDSP and multiplied with `vDSP_mmul`, a different staging and
    /// summation order from the kernels.
    func cpuRoutePartials(pairs: [PrefillTokenExpertPair],
                          hidden: [Float16],
                          hiddenStride: Int,
                          queryCount: Int,
                          topK: Int) -> [Float] {
        let byExpert = Dictionary(grouping: pairs, by: { Int($0.expert) })
        let experts = byExpert.keys.sorted()
        // Experts are independent; compute them concurrently into their own
        // slots, then scatter serially.
        var downs = [[Float]](repeating: [], count: experts.count)
        downs.withUnsafeMutableBufferPointer { buffer in
            let slots = DisjointWrites(pointer: buffer)
            DispatchQueue.concurrentPerform(iterations: experts.count) { index in
                slots.pointer[index] = expertDown(expert: experts[index],
                                          pairs: byExpert[experts[index]]!,
                                          hidden: hidden,
                                          hiddenStride: hiddenStride)
            }
        }
        var out = [Float](repeating: .nan, count: queryCount * topK * d)
        for (index, expert) in experts.enumerated() {
            let expertPairs = byExpert[expert]!
            let n = expertPairs.count
            for (j, pair) in expertPairs.enumerated() {
                let outBase = (Int(pair.token) * topK + Int(pair.rank)) * d
                for row in 0..<d {
                    out[outBase + row] = downs[index][row * n + j]
                }
            }
        }
        return out
    }

    /// Down-projection outputs of `expert` for its `pairs`, `[d * n]` with
    /// pair `j` in column `j`.
    private func expertDown(expert: Int,
                            pairs: [PrefillTokenExpertPair],
                            hidden: [Float16],
                            hiddenStride: Int) -> [Float] {
        let n = pairs.count
        var xT = [Float](repeating: 0, count: d * n)
        for (j, pair) in pairs.enumerated() {
            let base = Int(pair.token) * hiddenStride
            for k in 0..<d {
                xT[k * n + j] = Float(hidden[base + k])
            }
        }
        let gateW = dequantized(expert: expert, role: 0)
        let upW = dequantized(expert: expert, role: 1)
        let downW = dequantized(expert: expert, role: 2)
        var gate = [Float](repeating: 0, count: f * n)
        var up = [Float](repeating: 0, count: f * n)
        vDSP_mmul(gateW, 1, xT, 1, &gate, 1, vDSP_Length(f), vDSP_Length(n), vDSP_Length(d))
        vDSP_mmul(upW, 1, xT, 1, &up, 1, vDSP_Length(f), vDSP_Length(n), vDSP_Length(d))
        var act = [Float](repeating: 0, count: f * n)
        for index in act.indices {
            act[index] = Float(Float16(Self.gelu(gate[index]) * up[index]))
        }
        var down = [Float](repeating: 0, count: d * n)
        vDSP_mmul(downW, 1, act, 1, &down, 1, vDSP_Length(d), vDSP_Length(n), vDSP_Length(f))
        return down
    }
}

/// Routes, activations and a runner that drives `PrefillGroupedRoutedMoE`
/// tile by tile the way `RealForwardRunner` does.
enum PrefillInt8MoEHarness {
    /// FP16 bit pattern that marks route-partial elements no kernel wrote.
    static let sentinel = Float16(bitPattern: 0xFBFF)  // -65504

    /// Each token picks `topK` distinct experts out of `numExperts` with
    /// `SeedTree`, then the pairs are grouped into tiles of `tileExperts`.
    static func randomRoutes(tokens: Int,
                             topK: Int,
                             numExperts: Int,
                             tileExperts: Int,
                             seed: UInt64) throws -> PrefillMoEGroupedRoutes {
        var rng = SeedTree(seed).key("prefill-moe-int8-routes-\(tokens)x\(topK)of\(numExperts)")
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(tokens * topK)
        for token in 0..<tokens {
            var experts = Array(0..<numExperts)
            for i in 0..<topK {
                let j = i + Int(rng.next() % UInt64(numExperts - i))
                experts.swapAt(i, j)
                pairs.append(PrefillTokenExpertPair(
                    token: UInt32(token),
                    expert: UInt32(experts[i]),
                    rank: UInt32(i),
                    weight: Float16(1.0 / Float(topK))))
            }
        }
        return try PrefillMoEGrouping.groupTokenExpertPairs(pairs,
                                                            queryCount: tokens,
                                                            topK: topK,
                                                            numExperts: numExperts,
                                                            tileExpertCount: tileExperts)
    }

    static func randomHidden(tokens: Int, stride: Int, seed: UInt64) -> [Float16] {
        var rng = SeedTree(seed).key("prefill-moe-int8-hidden-\(tokens)x\(stride)")
        return (0..<(tokens * stride)).map { _ in Float16(rng.uniform(-1, 1)) }
    }

    struct Run {
        let routePartials: [Float16]
        let microbatches: Int
        let tileExpertCounts: [Int]
    }

    /// Encodes every tile of `routes`, one command buffer per tile, with the
    /// tile's experts bound to their pool views, and reads back the route
    /// partials (`sentinel` where nothing was written).
    static func runTiles(grouped: PrefillGroupedRoutedMoE,
                         context: MetalContext,
                         pool: PrefillInt8ExpertPool,
                         routes: PrefillMoEGroupedRoutes,
                         hidden: [Float16],
                         hiddenStride: Int,
                         topK: Int,
                         pairMicrobatchRows: Int,
                         tiles: [Int]? = nil) throws -> Run {
        let device = context.device
        let partialCount = routes.queryCount * topK * pool.d
        let metadata = try grouped.makeStreamedMetadataBuffers(device: device, routes: routes)
        guard let hiddenBuffer = Fp16Buffer.make(device, halves: hidden),
              let partials = Fp16Buffer.make(
                device, halves: [Float16](repeating: sentinel, count: partialCount)),
              let gateUpActScratch = device.makeBuffer(
                length: 3 * pairMicrobatchRows * pool.f * MemoryLayout<Float16>.stride,
                options: .storageModePrivate),
              let downScratch = device.makeBuffer(
                length: pairMicrobatchRows * pool.d * MemoryLayout<Float16>.stride,
                options: .storageModePrivate) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("INT8 MoE harness buffers")
        }
        var microbatches = 0
        var tileExpertCounts: [Int] = []
        for tileIndex in tiles ?? Array(routes.tiles.indices) {
            let tile = routes.tiles[tileIndex]
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex,
                                                                     routes: routes)
            let binding = try pool.binding(expertIDs: expertIDs)
            try binding.validateCoversPairs(routes.sortedPairs,
                                            pairStart: Int(tile.pairStart),
                                            pairCount: Int(tile.pairCount))
            let argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: device,
                                                                        binding: binding)
            let params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(pool.d),
                routedIntermediate: UInt32(pool.f),
                topK: UInt32(topK),
                hiddenStrideElements: UInt32(hiddenStride),
                binding: binding,
                offsets: pool.offsets)
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw PrefillGroupedRoutedMoEError.allocationFailed("command buffer")
            }
            microbatches += grouped.encodeStreamedBatched(
                commandBuffer: commandBuffer,
                hidden: hiddenBuffer,
                sortedPairs: metadata.sortedPairs,
                routePartials: partials,
                gateUpActScratch: gateUpActScratch,
                downScratch: downScratch,
                argumentBuffer: argumentBuffer,
                binding: binding,
                params: params,
                pairMicrobatchRows: pairMicrobatchRows)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }
            tileExpertCounts.append(expertIDs.count)
        }
        return Run(routePartials: Fp16Buffer.readHalf(partials, count: partialCount),
                   microbatches: microbatches,
                   tileExpertCounts: tileExpertCounts)
    }

    /// Relative error (`RelError.compute`) over the elements the reference
    /// covers, plus the count of elements whose coverage differs: written
    /// where the reference has none, or left as the sentinel where it has one.
    static func compare(_ actual: [Float16],
                        _ reference: [Float]) -> (rel: Float, maxAbs: Float, coverageMismatches: Int) {
        var a: [Float] = []
        var r: [Float] = []
        var mismatches = 0
        for (value, expected) in zip(actual, reference) {
            let written = value.bitPattern != sentinel.bitPattern
            if expected.isNaN {
                if written { mismatches += 1 }
            } else if !written {
                mismatches += 1
            } else {
                a.append(Float(value))
                r.append(expected)
            }
        }
        guard !a.isEmpty else { return (0, 0, mismatches) }
        return (RelError.compute(actual: a, reference: r),
                RelError.maxAbsDiff(a, r),
                mismatches)
    }
}
