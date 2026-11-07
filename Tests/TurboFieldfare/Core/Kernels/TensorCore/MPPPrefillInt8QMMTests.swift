import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

private let mppInt8TensorOpsAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPPrefillInt8QMM(context: context).isAvailable
}()

/// `MPPPrefillInt8QMM` against the FP32 CPU reference and against
/// `PrefillInt8QMM`, plus its availability and fallback contract.
@Suite struct MPPPrefillInt8QMMTests {
    typealias Projection = PrefillInt8Projection
    typealias Buffers = PrefillInt8ProjectionBuffers

    private static var moduleURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboFieldfare/Metal/TensorCore/tensorops_int8.metal")
    }

    private static func expectSelectedPathMetadata(_ metadata: MPPPrefillInt8QMM.PathMetadata,
                                                   weightsOffset: Int,
                                                   label: String) {
        #expect(metadata.path == .affineThreadgroupF16, "\(label) fell back")
        #expect(metadata.fallbackCount == 0, "\(label)")
        #expect(metadata.tileM == 64 && metadata.tileN == 32 && metadata.tileK == 64, "\(label)")
        #expect(metadata.threadgroupBytes == 4096, "\(label)")
        #expect(metadata.cooperativeValueBytesEstimate == 16_384, "\(label)")
        #expect(metadata.simultaneousWeightTiles == 1, "\(label)")
        #expect(metadata.modelDerivedCPUHeapScratchBytes == 0, "\(label)")
        #expect(metadata.expandedWeightBufferBytes == 0, "\(label)")
        #expect(metadata.weightBufferOffsetAlignment == 4, "\(label)")
        #expect(metadata.weightBufferOffsetRemainder128 == weightsOffset % 128, "\(label)")
    }

    /// Runs MPP and the plain QMM once per token count, each into its own
    /// sentinel-filled output, and checks every requested row against the CPU
    /// reference and the QMM, and that no row past `m` was written.
    private static func check(_ projection: Projection,
                              label: String,
                              tokenCounts: [Int],
                              weightsOffset: Int = 0,
                              scalesOffset: Int = 0,
                              biasesOffset: Int = 0,
                              xOffset: Int = 0) throws {
        let n = projection.rows
        let k = projection.columns
        let tokens = projection.tokens
        precondition(tokenCounts.allSatisfy { $0 <= tokens })
        let context = try MetalContext()
        let candidate = MPPPrefillInt8QMM(context: context)
        let baseline = try PrefillInt8QMM(context: context)
        #expect(candidate.isAvailable, "\(candidate.unavailableReason ?? "unknown error")")
        guard let buffers = Buffers(device: context.device,
                                    projection: projection,
                                    weightsOffset: weightsOffset,
                                    scalesOffset: scalesOffset,
                                    biasesOffset: biasesOffset,
                                    xOffset: xOffset),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed for \(label)")
            return
        }
        var runs: [(m: Int, mpp: MTLBuffer, qmm: MTLBuffer)] = []
        for m in tokenCounts {
            guard let mppOut = Buffers.makeSentinelOutput(device: context.device,
                                                          count: tokens * n),
                  let qmmOut = Fp16Buffer.make(context.device, count: m * n) else {
                Issue.record("output allocation failed for \(label)")
                return
            }
            let metadata = candidate.encode(commandBuffer: commandBuffer,
                                            weights: buffers.weights, weightsOffset: weightsOffset,
                                            scales: buffers.scales, scalesOffset: scalesOffset,
                                            biases: buffers.biases, biasesOffset: biasesOffset,
                                            x: buffers.x, xOffset: xOffset,
                                            y: mppOut,
                                            m: m,
                                            n: n,
                                            k: k)
            expectSelectedPathMetadata(metadata, weightsOffset: weightsOffset,
                                       label: "\(label) M=\(m)")
            baseline.encode(commandBuffer: commandBuffer,
                            weights: buffers.weights, weightsOffset: weightsOffset,
                            scales: buffers.scales, scalesOffset: scalesOffset,
                            biases: buffers.biases, biasesOffset: biasesOffset,
                            x: buffers.x, xOffset: xOffset,
                            y: qmmOut,
                            t: m,
                            n: n,
                            k: k)
            runs.append((m, mppOut, qmmOut))
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil, "\(label): \(String(describing: commandBuffer.error))")

        let reference = projection.cpuReference()
        for (m, mppOut, qmmOut) in runs {
            let count = m * n
            let actual = Fp16Buffer.read(mppOut, count: count)
            let expected = Array(reference[0..<count])
            let qmm = Fp16Buffer.read(qmmOut, count: count)
            let cpuRel = RelError.compute(actual: actual, reference: expected)
            let cpuMaxAbs = RelError.maxAbsDiff(actual, expected)
            let qmmRel = RelError.compute(actual: actual, reference: qmm)
            let qmmMaxAbs = RelError.maxAbsDiff(actual, qmm)
            #expect(cpuRel <= Tolerance.fp16Reduction,
                    "\(label) M=\(m): CPU rel=\(cpuRel) maxAbs=\(cpuMaxAbs)")
            #expect(qmmRel <= Tolerance.fp16Reduction,
                    "\(label) M=\(m): QMM rel=\(qmmRel) maxAbs=\(qmmMaxAbs)")
            let allFinite = actual.allSatisfy { $0.isFinite }
            #expect(allFinite, "\(label) M=\(m): non-finite output")
            let stray = Buffers.overwrittenCount(mppOut, from: count, count: tokens * n)
            #expect(stray == 0, "\(label) M=\(m): \(stray) elements written past row \(m)")
        }
    }

    /// Availability follows tensor support, exactly like the 4-bit twin, and
    /// an unavailable kernel says why.
    @Test func availabilityFollowsTheFourBitTwin() throws {
        let context = try MetalContext()
        let int8 = MPPPrefillInt8QMM(context: context)
        let int4 = MPPPrefillInt4QMM(context: context)
        #expect(int8.isAvailable == int4.isAvailable)
        if int8.isAvailable {
            #expect(int8.unavailableReason == nil)
        } else {
            #expect(int8.unavailableReason?.isEmpty == false)
        }
        let library = try MetalContext.privateLibrary(device: context.device,
                                                      module: "tensorops_int8")
        #expect(library.functionNames.contains(MPPPrefillInt8QMM.kernelName) == int8.isAvailable)

        let contract: any AffineMPPPrefillQMMEncoding = int8
        #expect(contract.isAvailable == int8.isAvailable)
    }

    /// The Q, K/V and O projections of plan §2.4, which the runner sends to
    /// MPP from 32 tokens, plus the shared-expert shapes, at every chunk
    /// length of the plan. Weights sit 4 mod 16 bytes into their buffer,
    /// scales and biases 2 mod 4 (plan C5).
    @Test(.enabled(if: mppInt8TensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func productionShapesMatchCPUAndQMM() throws {
        for (index, shape) in Projection.productionShapes.enumerated() {
            let projection = Projection.random(rows: shape.rows,
                                               columns: shape.columns,
                                               tokens: Projection.tokenCounts.max()!,
                                               seed: 0x5400 + UInt64(index))
            try Self.check(projection,
                           label: shape.name,
                           tokenCounts: Projection.tokenCounts,
                           weightsOffset: 20,
                           scalesOffset: 2,
                           biasesOffset: 6)
        }
    }

    /// Partial tiles in both M and N, the smallest K and an odd number of
    /// groups, every byte value with adversarial BF16 scales and biases, and
    /// activations at a 2-byte offset.
    @Test(.enabled(if: mppInt8TensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func partialTilesAndOffsetsMatchCPUAndQMM() throws {
        try Self.check(Projection.random(rows: 32, columns: 64, tokens: 64,
                                         seed: 0x5430, weightRange: 0.5),
                       label: "single tile", tokenCounts: [64])
        try Self.check(Projection.random(rows: 35, columns: 64, tokens: 17,
                                         seed: 0x5431, weightRange: 0.5),
                       label: "partial M and N", tokenCounts: [1, 17],
                       weightsOffset: 4, xOffset: 2)
        try Self.check(Projection.random(rows: 33, columns: 2112, tokens: 65,
                                         seed: 0x5432),
                       label: "odd groups", tokenCounts: [1, 65],
                       weightsOffset: 132, scalesOffset: 4, biasesOffset: 2)
        try Self.check(Projection.adversarial(rows: 35, columns: 128, tokens: 17),
                       label: "adversarial", tokenCounts: [17],
                       weightsOffset: 20, scalesOffset: 2, biasesOffset: 6)
    }

    /// The same dispatch gives the same bits every time on the largest
    /// production projection.
    @Test(.enabled(if: mppInt8TensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func fullProductionShapeIsByteStableAcross32Dispatches() throws {
        let m = 32
        let projection = Projection.random(rows: 2816, columns: 8192, tokens: m, seed: 0x5440)
        let outputElements = m * projection.rows
        let outputBytes = outputElements * MemoryLayout<Float16>.stride
        let context = try MetalContext()
        let candidate = MPPPrefillInt8QMM(context: context)
        guard let buffers = Buffers(device: context.device, projection: projection),
              let outputs = context.device.makeBuffer(length: outputBytes * 32,
                                                      options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        for run in 0..<32 {
            let metadata = candidate.encode(commandBuffer: commandBuffer,
                                            weights: buffers.weights,
                                            scales: buffers.scales,
                                            biases: buffers.biases,
                                            x: buffers.x,
                                            y: outputs,
                                            yOffset: run * outputBytes,
                                            m: m,
                                            n: projection.rows,
                                            k: projection.columns)
            #expect(metadata.path == .affineThreadgroupF16)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let first = outputs.contents().assumingMemoryBound(to: UInt16.self)
        for run in 1..<32 {
            let candidateOutput = outputs.contents()
                .advanced(by: run * outputBytes)
                .assumingMemoryBound(to: UInt16.self)
            var mismatch: Int?
            for index in 0..<outputElements where first[index] != candidateOutput[index] {
                mismatch = index
                break
            }
            #expect(mismatch == nil, "dispatch \(run) first mismatch=\(mismatch ?? -1)")
        }
    }

    /// Every input the kernel cannot take reports `.fallback` and encodes
    /// nothing, so the caller's output keeps its contents. A weight offset
    /// that is even but not a multiple of 4 is valid for the 4-bit twin and a
    /// fallback here. When MPP is unavailable every call falls back.
    @Test func unsupportedOrUnalignedInputsFallBackWithoutEncoding() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt8QMM(context: context)
        let projection = Projection.random(rows: 32, columns: 128, tokens: 32, seed: 0x5450)
        let elements = 32 * 32
        guard let buffers = Buffers(device: context.device, projection: projection,
                                    weightsOffset: 8),
              let output = Buffers.makeSentinelOutput(device: context.device,
                                                      count: elements + 1),
              let selectedOutput = Buffers.makeSentinelOutput(device: context.device,
                                                              count: elements),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        func encode(weightsOffset: Int = 8,
                    scalesOffset: Int = 0,
                    biasesOffset: Int = 0,
                    xOffset: Int = 0,
                    yOffset: Int = 0,
                    y: MTLBuffer? = nil,
                    m: Int = 32,
                    n: Int = 32,
                    k: Int = 128) -> MPPPrefillInt8QMM.PathMetadata {
            candidate.encode(commandBuffer: commandBuffer,
                             weights: buffers.weights, weightsOffset: weightsOffset,
                             scales: buffers.scales, scalesOffset: scalesOffset,
                             biases: buffers.biases, biasesOffset: biasesOffset,
                             x: buffers.x, xOffset: xOffset,
                             y: y ?? output, yOffset: yOffset,
                             m: m, n: n, k: k)
        }
        let rejected: [(String, MPPPrefillInt8QMM.PathMetadata)] = [
            ("K not a multiple of 64", encode(k: 96)),
            ("K = 65", encode(k: 65)),
            ("M = 0", encode(m: 0)),
            ("N = 0", encode(n: 0)),
            ("negative weight offset", encode(weightsOffset: -4)),
            ("weight offset 2 mod 4", encode(weightsOffset: 10)),
            ("weight offset 1 mod 4", encode(weightsOffset: 9)),
            ("odd scale offset", encode(scalesOffset: 1)),
            ("odd bias offset", encode(biasesOffset: 3)),
            ("odd activation offset", encode(xOffset: 1)),
            ("odd output offset", encode(yOffset: 1)),
        ]
        let selected = encode(y: selectedOutput)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        for (label, metadata) in rejected {
            #expect(metadata.path == .fallback, "\(label)")
            #expect(metadata.fallbackCount == 1, "\(label)")
            #expect(metadata.weightBufferOffsetAlignment == 4, "\(label)")
        }
        #expect(Buffers.overwrittenCount(output, from: 0, count: elements + 1) == 0,
                "a fallback wrote to the output")
        if candidate.isAvailable {
            Self.expectSelectedPathMetadata(selected, weightsOffset: 8, label: "aligned call")
            #expect(Buffers.overwrittenCount(selectedOutput, from: 0, count: elements) == elements)
        } else {
            #expect(selected.path == .fallback)
            #expect(selected.fallbackCount == 1)
            #expect(Buffers.overwrittenCount(selectedOutput, from: 0, count: elements) == 0)
        }
    }

    @Test func tensorOpsInt8ModuleDeclaresOnlyTheMPPKernel() throws {
        let source = try String(contentsOf: Self.moduleURL, encoding: .utf8)
        let regex = try NSRegularExpression(pattern: #"kernel\s+void\s+(\w+)\s*\("#)
        let range = NSRange(source.startIndex..., in: source)
        let kernels = regex.matches(in: source, range: range).compactMap { match in
            Range(match.range(at: 1), in: source).map { String(source[$0]) }
        }
        #expect(kernels == [MPPPrefillInt8QMM.kernelName])
        #expect(!source.contains("function_constant("))
        // Tensor code stays inside the guard so devices without tensor
        // support compile an empty library.
        if let guardStart = source.range(of: "#if defined(__HAVE_TENSOR__)"),
           let kernelStart = source.range(of: "kernel void"),
           let guardEnd = source.range(of: "#endif // __HAVE_TENSOR__") {
            #expect(guardStart.lowerBound < kernelStart.lowerBound)
            #expect(kernelStart.lowerBound < guardEnd.lowerBound)
        } else {
            Issue.record("tensorops_int8.metal lost its __HAVE_TENSOR__ guard")
        }
    }
}
