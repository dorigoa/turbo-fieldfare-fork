import Metal
import Testing
import TurboFieldfareValidationSupport

@testable import TurboFieldfare

extension PrefillGroupedRoutedMoEInt8Tests {
  /// One tile encoded directly: `expertIDs` bound in the given order to their
  /// pool views, `pairs` (already in tile order) as the whole pair range.
  private static func runSingleTile(
    grouped: PrefillGroupedRoutedMoE,
    context: MetalContext,
    pool: PrefillInt8ExpertPool,
    expertIDs: [Int],
    pairs: [PrefillTokenExpertPair],
    hidden: [Float16],
    queryCount: Int,
    topK: Int
  ) throws -> [Float16] {
    let device = context.device
    let count = queryCount * topK * pool.d
    let binding = try pool.binding(expertIDs: expertIDs)
    guard let hiddenBuffer = Fp16Buffer.make(device, halves: hidden),
      let pairBuffer = device.makeBuffer(
        bytes: pairs,
        length: pairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
        options: .storageModeShared),
      let output = Fp16Buffer.make(
        device, halves: [Float16](repeating: PrefillInt8MoEHarness.sentinel, count: count)),
      let activationScratch = device.makeBuffer(
        length: 3 * 32 * pool.f * MemoryLayout<Float16>.stride, options: .storageModePrivate),
      let downScratch = device.makeBuffer(
        length: 32 * pool.d * MemoryLayout<Float16>.stride, options: .storageModePrivate),
      let commandBuffer = context.queue.makeCommandBuffer()
    else {
      throw PrefillGroupedRoutedMoEError.allocationFailed("single tile")
    }
    let argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: device, binding: binding)
    let params = PrefillGroupedRoutedMoEStreamedParams(
      pairStart: 0,
      pairCount: UInt32(pairs.count),
      d: UInt32(pool.d),
      routedIntermediate: UInt32(pool.f),
      topK: UInt32(topK),
      hiddenStrideElements: UInt32(pool.d),
      binding: binding,
      offsets: pool.offsets)
    _ = grouped.encodeStreamedBatched(
      commandBuffer: commandBuffer, hidden: hiddenBuffer, sortedPairs: pairBuffer,
      routePartials: output, gateUpActScratch: activationScratch, downScratch: downScratch,
      argumentBuffer: argumentBuffer, binding: binding, params: params)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error { throw error }
    return Fp16Buffer.readHalf(output, count: count)
  }

  /// The argument buffer has the same size and the same contents for both
  /// widths: the INT8 kernels take the 4-bit `PrefillStreamedRoutedBlobsMSL`.
  @Test func argumentBufferMatchesTheFourBitLayout() throws {
    let ctx = try MetalContext()
    let pool = try PrefillInt8ExpertPool(device: ctx.device, d: 64, f: 64, expertCount: 16,
                                         values: .adversarial, leadingBytes: 32)
    let fourBit = try PrefillGroupedRoutedMoE(context: ctx)
    let eightBit = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)
    for expertIDs in [[5], [3, 9, 1], Array(0..<16).reversed()] {
      let binding = try pool.binding(expertIDs: Array(expertIDs))
      let a = try fourBit.makeStreamedArgumentBuffer(device: ctx.device, binding: binding)
      let b = try eightBit.makeStreamedArgumentBuffer(device: ctx.device, binding: binding)
      #expect(a.buffer.length == b.buffer.length)
      #expect(a.buffer.storageMode == b.buffer.storageMode)
      let bytesA = Array(UnsafeRawBufferPointer(start: a.buffer.contents(), count: a.buffer.length))
      let bytesB = Array(UnsafeRawBufferPointer(start: b.buffer.contents(), count: b.buffer.length))
      #expect(bytesA == bytesB, "experts \(expertIDs)")
    }
  }

  /// Sixteen live experts bound in a shuffled order that differs from their
  /// IDs: each pair must reach its own expert's blob through the local-slot
  /// search, including the last slot (`local_expert_15`).
  @Test func localSlotSearchFollowsTheBindingOrder() throws {
    let ctx = try MetalContext()
    let topK = 4
    let tokens = 12
    let pool = try PrefillInt8ExpertPool(device: ctx.device, d: 128, f: 128, expertCount: 20,
                                         values: .quantized(seed: 0x5550, range: 0.3))
    let expertIDs = [17, 3, 9, 0, 19, 12, 5, 14, 1, 8, 16, 6, 11, 2, 18, 7]
    var rng = SeedTree(0x5551).key("prefill-moe-int8-shuffled-binding")
    var pairs: [PrefillTokenExpertPair] = []
    for token in 0..<tokens {
      var choices = expertIDs
      for rank in 0..<topK {
        let j = rank + Int(rng.next() % UInt64(choices.count - rank))
        choices.swapAt(rank, j)
        pairs.append(PrefillTokenExpertPair(token: UInt32(token), expert: UInt32(choices[rank]),
                                            rank: UInt32(rank), weight: 0.25))
      }
    }
    // Make sure slot 15 (expert 7) and slot 0 (expert 17) carry pairs. A
    // repeated expert at another rank of the same token is still its own row.
    pairs[0].expert = 7
    pairs[1].expert = 17
    let hidden = PrefillInt8MoEHarness.randomHidden(tokens: tokens, stride: 128, seed: 0x5552)
    let reference = pool.cpuRoutePartials(pairs: pairs, hidden: hidden, hiddenStride: 128,
                                          queryCount: tokens, topK: topK)
    let grouped = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)
    let actual = try Self.runSingleTile(grouped: grouped, context: ctx, pool: pool,
                                        expertIDs: expertIDs, pairs: pairs, hidden: hidden,
                                        queryCount: tokens, topK: topK)
    let result = PrefillInt8MoEHarness.compare(actual, reference)
    #expect(Set(pairs.map { Int($0.expert) }).isSubset(of: Set(expertIDs)))
    #expect(pairs.contains { $0.expert == 7 } && pairs.contains { $0.expert == 17 })
    #expect(result.coverageMismatches == 0)
    #expect(result.rel <= Tolerance.fp16Reduction, "rel=\(result.rel) maxAbs=\(result.maxAbs)")
  }

  /// A pair whose expert the tile does not bind is skipped by both phases, as
  /// in the 4-bit kernels: its route-partial row keeps its old contents.
  @Test func unboundRouteExpertsAreSkipped() throws {
    let ctx = try MetalContext()
    let pool = try PrefillInt8ExpertPool(device: ctx.device, d: 64, f: 128, expertCount: 6,
                                         values: .quantized(seed: 0x5560, range: 0.3))
    let pairs = [
      PrefillTokenExpertPair(token: 0, expert: 4, rank: 0, weight: 0.5),
      PrefillTokenExpertPair(token: 0, expert: 5, rank: 1, weight: 0.5),
      PrefillTokenExpertPair(token: 1, expert: 1, rank: 0, weight: 0.5),
      PrefillTokenExpertPair(token: 1, expert: 4, rank: 1, weight: 0.5),
    ]
    let hidden = PrefillInt8MoEHarness.randomHidden(tokens: 2, stride: 64, seed: 0x5561)
    let boundPairs = pairs.filter { $0.expert != 5 }
    let reference = pool.cpuRoutePartials(pairs: boundPairs, hidden: hidden, hiddenStride: 64,
                                          queryCount: 2, topK: 2)
    let grouped = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)
    let actual = try Self.runSingleTile(grouped: grouped, context: ctx, pool: pool,
                                        expertIDs: [4, 1], pairs: pairs, hidden: hidden,
                                        queryCount: 2, topK: 2)
    let result = PrefillInt8MoEHarness.compare(actual, reference)
    // Token 0, rank 1 (expert 5) is unbound: its 64 elements stay sentinel.
    #expect(actual[64..<128].allSatisfy { $0.bitPattern == PrefillInt8MoEHarness.sentinel.bitPattern })
    #expect(result.coverageMismatches == 0)
    #expect(result.rel <= Tolerance.fp16Reduction, "rel=\(result.rel)")
  }

  /// Encoding one tile of a multi-tile route writes exactly that tile's
  /// pair rows, as the runner relies on when it issues tiles one by one.
  @Test func eachTileWritesOnlyItsOwnPairs() throws {
    let ctx = try MetalContext()
    let topK = 8
    let tokens = 10
    let pool = try PrefillInt8ExpertPool(device: ctx.device, d: 128, f: 64, expertCount: 24,
                                         values: .quantized(seed: 0x5570, range: 0.3))
    let routes = try PrefillInt8MoEHarness.randomRoutes(tokens: tokens, topK: topK,
                                                        numExperts: 24, tileExperts: 8,
                                                        seed: 0x5571)
    #expect(routes.tiles.count >= 3)
    let hidden = PrefillInt8MoEHarness.randomHidden(tokens: tokens, stride: 128, seed: 0x5572)
    let grouped = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)
    for tileIndex in routes.tiles.indices {
      let tile = routes.tiles[tileIndex]
      let tilePairs = Array(routes.sortedPairs[Int(tile.pairStart)..<Int(tile.pairStart + tile.pairCount)])
      let reference = pool.cpuRoutePartials(pairs: tilePairs, hidden: hidden, hiddenStride: 128,
                                            queryCount: tokens, topK: topK)
      let run = try PrefillInt8MoEHarness.runTiles(
        grouped: grouped, context: ctx, pool: pool, routes: routes, hidden: hidden,
        hiddenStride: 128, topK: topK, pairMicrobatchRows: 16, tiles: [tileIndex])
      let result = PrefillInt8MoEHarness.compare(run.routePartials, reference)
      #expect(result.coverageMismatches == 0, "tile \(tileIndex)")
      #expect(result.rel <= Tolerance.fp16Reduction, "tile \(tileIndex): rel=\(result.rel)")
    }
  }

  /// INT8 views must start on 4-byte boundaries; the 4-bit path keeps
  /// accepting any offset.
  @Test func int8ViewsOffTheFourByteGridAreRejected() throws {
    let ctx = try MetalContext()
    guard let buffer = ctx.device.makeBuffer(length: 4096, options: .storageModeShared) else {
      Issue.record("allocation failed")
      return
    }
    func view(_ offset: UInt64) -> TensorView {
      TensorView(buffer: buffer, offset: offset, length: 64, scaleOffset: 0, scaleLength: 0,
                 biasOffset: 0, biasLength: 0, shape: (0, 0, 0, 0), dtype: 0)
    }
    let fourBit = try PrefillGroupedRoutedMoE(context: ctx)
    let eightBit = try PrefillGroupedRoutedMoE(context: ctx, weightBits: 8)
    for offset: UInt64 in [0, 4, 16, 64, 2052] {
      let binding = try PrefillStreamedTileBinding(expertIDs: [3, 9], views: [view(0), view(offset)])
      _ = try eightBit.makeStreamedArgumentBuffer(device: ctx.device, binding: binding)
    }
    for offset: UInt64 in [1, 2, 3, 6, 18] {
      let binding = try PrefillStreamedTileBinding(expertIDs: [3, 9], views: [view(0), view(offset)])
      #expect {
        _ = try eightBit.makeStreamedArgumentBuffer(device: ctx.device, binding: binding)
      } throws: { error in
        guard case PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(let reason) = error else {
          return false
        }
        return reason.contains("expert 9") && reason.contains("4-byte")
      }
      _ = try fourBit.makeStreamedArgumentBuffer(device: ctx.device, binding: binding)
    }
  }

  /// Views fetched through the pread streamer (toy model) start on slot
  /// boundaries, so they pass the INT8 view check unchanged.
  @Test func streamerFetchedViewsPassTheInt8ViewCheck() async throws {
    let dir = try ModelLoaderTests.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy(),
      streamingMode: .pread(slotCount: 16))
    let routes = try PrefillGroupedRoutedMoETests.tileFetchRoutes()
    let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
      model: model, layer: 1, tileIndex: 0, routes: routes)
    #expect(fetch.binding.views.allSatisfy { $0.offset.isMultiple(of: 4) })
    try PrefillGroupedRoutedMoE.validateInt8ExpertViews(fetch.binding)
  }
}
