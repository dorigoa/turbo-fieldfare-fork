import Metal
import Testing
import TurboFieldfareValidationSupport

@testable import TurboFieldfare

extension PrefillGroupedRoutedMoEInt8Tests {
  typealias Pool = PrefillInt8ExpertPool
  typealias Harness = PrefillInt8MoEHarness

  /// Twin of `PrefillGroupedRoutedMoETests.streamedBatchedMatchesReferenceAcrossPartialMicrobatch`
  /// with INT8 blobs: the same six pairs, all 16 experts bound, views at
  /// nonzero offsets and a partial second microbatch.
  @Test func streamedBatchedMatchesReferenceAcrossPartialMicrobatch() throws {
    let d = 64
    let f = 64
    let rows = 3
    let topK = 2
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      [
        PrefillGroupedRoutedMoETests.pair(token: 0, expert: 2, rank: 0),
        PrefillGroupedRoutedMoETests.pair(token: 0, expert: 0, rank: 1),
        PrefillGroupedRoutedMoETests.pair(token: 1, expert: 1, rank: 0),
        PrefillGroupedRoutedMoETests.pair(token: 1, expert: 2, rank: 1),
        PrefillGroupedRoutedMoETests.pair(token: 2, expert: 0, rank: 0),
        PrefillGroupedRoutedMoETests.pair(token: 2, expert: 1, rank: 1),
      ],
      queryCount: rows,
      topK: topK,
      numExperts: 16,
      tileExpertCount: 16)
    let ctx = try MetalContext()
    let pool = try Pool(device: ctx.device, d: d, f: f, expertCount: 16,
                        values: .quantized(seed: 0x5501, range: 0.5),
                        leadingBytes: 64)
    let hidden = Harness.randomHidden(tokens: rows, stride: d, seed: 0x5502)
    let expected = pool.cpuRoutePartials(pairs: routes.sortedPairs, hidden: hidden,
                                         hiddenStride: d, queryCount: rows, topK: topK)

    let grouped = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)
    guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
      let pairBuffer = ctx.device.makeBuffer(
        bytes: routes.sortedPairs,
        length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let outputBuffer = Fp16Buffer.make(
        ctx.device,
        halves: [Float16](repeating: Harness.sentinel, count: rows * topK * d)),
      let activationScratch = ctx.device.makeBuffer(
        length: 3 * 4 * f * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let downScratch = ctx.device.makeBuffer(
        length: 4 * d * MemoryLayout<Float16>.stride,
        options: .storageModePrivate),
      let commandBuffer = ctx.queue.makeCommandBuffer()
    else {
      Issue.record("allocation failed")
      return
    }

    let binding = try pool.binding(expertIDs: Array(0..<16))
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(routes.sortedPairs.count),
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(
      device: ctx.device,
      binding: binding)
    let microbatches = grouped.encodeStreamedBatched(
      commandBuffer: commandBuffer,
      hidden: hiddenBuffer,
      sortedPairs: pairBuffer,
      routePartials: outputBuffer,
      gateUpActScratch: activationScratch,
      downScratch: downScratch,
      argumentBuffer: argumentBuffer,
      binding: binding,
      params: params,
      pairMicrobatchRows: 4)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }

    let actual = Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    let result = Harness.compare(actual, expected)
    #expect(microbatches == 2)
    #expect(result.coverageMismatches == 0)
    #expect(result.rel <= Tolerance.fp16Reduction, "rel=\(result.rel) maxAbs=\(result.maxAbs)")
    #expect(binding.views.allSatisfy { $0.offset > 0 })
  }

  /// Production shapes (D = 2816, F = 704) with the §2.5 blob layout, routed
  /// top-8 over 20 experts and run tile by tile like the runner, with tiles
  /// of 8 experts (the scheduler default) and of 16 (the maximum). Every
  /// pair is computed on its own, so both tilings give the same bits.
  @Test func productionShapeTilesOf8And16MatchReference() throws {
    let d = 2816
    let f = 704
    let topK = 8
    let tokens = 24
    let experts = 20
    let ctx = try MetalContext()
    let pool = try Pool(device: ctx.device, d: d, f: f, expertCount: experts,
                        values: .quantized(seed: 0x5510, range: 0.05))
    #expect(pool.layout == Pool.Layout(d: 2816, f: 704))
    #expect(pool.layout.stride == 6_324_224)
    let hidden = Harness.randomHidden(tokens: tokens, stride: d, seed: 0x5511)
    let grouped = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)

    var runs: [Int: Harness.Run] = [:]
    var reference: [Float] = []
    for tileExperts in [8, 16] {
      let routes = try Harness.randomRoutes(tokens: tokens, topK: topK, numExperts: experts,
                                            tileExperts: tileExperts, seed: 0x5512)
      if reference.isEmpty {
        reference = pool.cpuRoutePartials(pairs: routes.sortedPairs, hidden: hidden,
                                          hiddenStride: d, queryCount: tokens, topK: topK)
      }
      let run = try Harness.runTiles(grouped: grouped, context: ctx, pool: pool,
                                     routes: routes, hidden: hidden, hiddenStride: d,
                                     topK: topK, pairMicrobatchRows: 32)
      let result = Harness.compare(run.routePartials, reference)
      #expect(run.tileExpertCounts.allSatisfy { $0 <= tileExperts })
      #expect(run.tileExpertCounts.reduce(0, +) == routes.groups.count)
      #expect(run.tileExpertCounts.contains(tileExperts), "\(run.tileExpertCounts)")
      #expect(result.coverageMismatches == 0, "tile \(tileExperts)")
      #expect(result.rel <= Tolerance.fp16Reduction,
              "tile \(tileExperts): rel=\(result.rel) maxAbs=\(result.maxAbs)")
      runs[tileExperts] = run
    }
    #expect(runs[8]!.tileExpertCounts == [8, 8, 4])
    #expect(runs[16]!.tileExpertCounts == [16, 4])
    #expect(runs[8]!.routePartials.map(\.bitPattern) == runs[16]!.routePartials.map(\.bitPattern))
  }

  /// Microbatches of 1, 7, 32 and 128 pairs (the runner clamps to 1...128)
  /// split the tiles differently and give the same bits.
  @Test func microbatchSizesGiveTheSameBits() throws {
    let d = 256
    let f = 192
    let topK = 8
    let tokens = 40
    let ctx = try MetalContext()
    let pool = try Pool(device: ctx.device, d: d, f: f, expertCount: 16,
                        values: .quantized(seed: 0x5520, range: 0.2))
    let routes = try Harness.randomRoutes(tokens: tokens, topK: topK, numExperts: 16,
                                          tileExperts: 8, seed: 0x5521)
    let hidden = Harness.randomHidden(tokens: tokens, stride: d, seed: 0x5522)
    let reference = pool.cpuRoutePartials(pairs: routes.sortedPairs, hidden: hidden,
                                          hiddenStride: d, queryCount: tokens, topK: topK)
    let grouped = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)

    var baseline: [UInt16]?
    for rows in [1, 7, 32, 128] {
      let run = try Harness.runTiles(grouped: grouped, context: ctx, pool: pool,
                                     routes: routes, hidden: hidden, hiddenStride: d,
                                     topK: topK, pairMicrobatchRows: rows)
      let expectedMicrobatches = routes.tiles.reduce(0) {
        $0 + (Int($1.pairCount) + rows - 1) / rows
      }
      #expect(run.microbatches == expectedMicrobatches, "rows \(rows)")
      let result = Harness.compare(run.routePartials, reference)
      #expect(result.coverageMismatches == 0, "rows \(rows)")
      #expect(result.rel <= Tolerance.fp16Reduction, "rows \(rows): rel=\(result.rel)")
      let bits = run.routePartials.map(\.bitPattern)
      if let baseline {
        #expect(bits == baseline, "rows \(rows) changed the output bits")
      } else {
        baseline = bits
      }
    }
  }

  /// Every byte value with negative, tiny and subnormal BF16 scales and
  /// biases; F = 192 leaves the down rows at three groups.
  @Test func adversarialAffineValuesMatchReference() throws {
    let d = 128
    let f = 192
    let topK = 4
    let tokens = 9
    let ctx = try MetalContext()
    let pool = try Pool(device: ctx.device, d: d, f: f, expertCount: 6, values: .adversarial)
    let routes = try Harness.randomRoutes(tokens: tokens, topK: topK, numExperts: 6,
                                          tileExperts: 16, seed: 0x5530)
    let hidden = Harness.randomHidden(tokens: tokens, stride: d, seed: 0x5531)
    let reference = pool.cpuRoutePartials(pairs: routes.sortedPairs, hidden: hidden,
                                          hiddenStride: d, queryCount: tokens, topK: topK)
    let grouped = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)
    let run = try Harness.runTiles(grouped: grouped, context: ctx, pool: pool,
                                   routes: routes, hidden: hidden, hiddenStride: d,
                                   topK: topK, pairMicrobatchRows: 32)
    let result = Harness.compare(run.routePartials, reference)
    #expect(result.coverageMismatches == 0)
    #expect(result.rel <= Tolerance.fp16Reduction, "rel=\(result.rel) maxAbs=\(result.maxAbs)")
  }

  /// A hidden stride wider than D (rows of a larger activation buffer) and a
  /// pool whose blobs start 4 bytes past a 16-byte boundary: the kernels need
  /// only 4-byte weights (plan C5).
  @Test func hiddenStrideAndFourByteBlobOffsetsMatchReference() throws {
    let d = 128
    let f = 64
    let stride = d + 64
    let topK = 4
    let tokens = 6
    let ctx = try MetalContext()
    let pool = try Pool(device: ctx.device, d: d, f: f, expertCount: 8,
                        values: .quantized(seed: 0x5540, range: 0.3),
                        leadingBytes: 4)
    #expect((0..<8).allSatisfy { pool.blobOffset($0) % 16 == 4 })
    let routes = try Harness.randomRoutes(tokens: tokens, topK: topK, numExperts: 8,
                                          tileExperts: 8, seed: 0x5541)
    let hidden = Harness.randomHidden(tokens: tokens, stride: stride, seed: 0x5542)
    let reference = pool.cpuRoutePartials(pairs: routes.sortedPairs, hidden: hidden,
                                          hiddenStride: stride, queryCount: tokens, topK: topK)
    let grouped = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)
    let run = try Harness.runTiles(grouped: grouped, context: ctx, pool: pool,
                                   routes: routes, hidden: hidden, hiddenStride: stride,
                                   topK: topK, pairMicrobatchRows: 5)
    let result = Harness.compare(run.routePartials, reference)
    #expect(result.coverageMismatches == 0)
    #expect(result.rel <= Tolerance.fp16Reduction, "rel=\(result.rel) maxAbs=\(result.maxAbs)")
  }

  /// `weightBits: 4` (explicit) and the default give the same bits on the
  /// 4-bit synthetic pool of the original suite, within its reference bound.
  @Test func fourBitPathIsUnchanged() throws {
    let d = 64
    let f = 64
    let rows = 3
    let topK = 2
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      [
        PrefillGroupedRoutedMoETests.pair(token: 0, expert: 2, rank: 0),
        PrefillGroupedRoutedMoETests.pair(token: 0, expert: 0, rank: 1),
        PrefillGroupedRoutedMoETests.pair(token: 1, expert: 1, rank: 0),
        PrefillGroupedRoutedMoETests.pair(token: 1, expert: 2, rank: 1),
        PrefillGroupedRoutedMoETests.pair(token: 2, expert: 0, rank: 0),
        PrefillGroupedRoutedMoETests.pair(token: 2, expert: 1, rank: 1),
      ],
      queryCount: rows,
      topK: topK,
      numExperts: 16,
      tileExpertCount: 16)
    let pool = PrefillGroupedRoutedMoETests.makeSyntheticExpertPool(numExperts: 16, d: d, f: f)
    let hidden = (0..<(rows * d)).map { i in Float16(Float((i % 17) - 8) * 0.01) }
    let expected = PrefillGroupedRoutedMoETests.cpuSyntheticRoutePartials(
      routes: routes, hidden: hidden, hiddenStride: d, pool: pool, topK: topK, d: d, f: f)

    let ctx = try MetalContext()
    let expertIDs = Array(0..<16)
    let binding = try PrefillStreamedTileBinding(
      expertIDs: expertIDs,
      views: PrefillGroupedRoutedMoETests.streamedViewsWithNonzeroOffsets(
        device: ctx.device, pool: pool, expertIDs: expertIDs))
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(routes.sortedPairs.count),
      d: UInt32(d),
      routedIntermediate: UInt32(f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(d),
      binding: binding,
      offsets: pool.offsets)

    func run(_ grouped: PrefillGroupedRoutedMoE) throws -> [Float16] {
      guard let hiddenBuffer = Fp16Buffer.make(ctx.device, halves: hidden),
        let pairBuffer = ctx.device.makeBuffer(
          bytes: routes.sortedPairs,
          length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
          options: .storageModeShared),
        let outputBuffer = Fp16Buffer.make(
          ctx.device, halves: [Float16](repeating: -77, count: rows * topK * d)),
        let activationScratch = ctx.device.makeBuffer(
          length: 3 * 4 * f * MemoryLayout<Float16>.stride, options: .storageModePrivate),
        let downScratch = ctx.device.makeBuffer(
          length: 4 * d * MemoryLayout<Float16>.stride, options: .storageModePrivate),
        let commandBuffer = ctx.queue.makeCommandBuffer()
      else {
        throw PrefillGroupedRoutedMoEError.allocationFailed("4-bit run")
      }
      let argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: ctx.device,
                                                                  binding: binding)
      let microbatches = grouped.encodeStreamedBatched(
        commandBuffer: commandBuffer, hidden: hiddenBuffer, sortedPairs: pairBuffer,
        routePartials: outputBuffer, gateUpActScratch: activationScratch,
        downScratch: downScratch, argumentBuffer: argumentBuffer, binding: binding,
        params: params, pairMicrobatchRows: 4)
      #expect(microbatches == 2)
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      if let error = commandBuffer.error { throw error }
      return Fp16Buffer.readHalf(outputBuffer, count: rows * topK * d)
    }

    let byDefault = try run(PrefillGroupedRoutedMoE(context: ctx))
    let explicit = try run(PrefillGroupedRoutedMoE(context: ctx, weightBits: 4))
    #expect(byDefault.map(\.bitPattern) == explicit.map(\.bitPattern))
    let maxAbsoluteError = zip(explicit, expected).reduce(Float(0)) {
      max($0, abs(Float($1.0) - Float($1.1)))
    }
    #expect(maxAbsoluteError <= 0.0015, "maxAbsoluteError=\(maxAbsoluteError)")
  }
}
