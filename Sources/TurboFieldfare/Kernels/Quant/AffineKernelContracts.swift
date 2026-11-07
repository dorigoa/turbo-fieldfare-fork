import Foundation
import Metal

// Kernel contracts shared by the 4-bit and 8-bit affine weight paths.
//
// Each protocol's single requirement is the existing 4-bit wrapper's encode
// method, label for label and type for type, so the 4-bit classes conform
// with empty extensions and their call sites do not change. An 8-bit wrapper
// conforms by offering the same method over 8-bit weights; the forward
// runner then picks one conformer per quantization profile.
//
// Swift protocol requirements cannot declare default arguments. A caller
// that goes through one of these protocols passes every argument, offsets
// included; a caller that holds the concrete class keeps its defaults.

/// One-row-per-output affine GEMV: `y[m] = W[m, :] · x`.
protocol AffineGEMVEncoding: AnyObject {
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int,
                scales: MTLBuffer,
                scalesOffset: Int,
                biases: MTLBuffer,
                biasesOffset: Int,
                x: MTLBuffer,
                xOffset: Int,
                y: MTLBuffer,
                yOffset: Int,
                m: UInt32,
                n: UInt32)
}

/// Decode embedding row lookup with an output scale.
protocol AffineEmbedLookupEncoding: AnyObject {
    func encode(commandBuffer: MTLCommandBuffer,
                table: MTLBuffer, tableOffset: Int,
                scales: MTLBuffer, scalesOffset: Int,
                biases: MTLBuffer, biasesOffset: Int,
                out: MTLBuffer, outOffset: Int,
                tokenId: UInt32,
                d: UInt32,
                outScale: Float)
}

/// Fused decode Q/K/V projection.
protocol AffineFusedQKVEncoding: AnyObject {
    func encode(commandBuffer: MTLCommandBuffer,
                qWeights: MTLBuffer, qWeightsOffset: Int,
                qScales: MTLBuffer, qScalesOffset: Int,
                qBiases: MTLBuffer, qBiasesOffset: Int,
                kWeights: MTLBuffer, kWeightsOffset: Int,
                kScales: MTLBuffer, kScalesOffset: Int,
                kBiases: MTLBuffer, kBiasesOffset: Int,
                vWeights: MTLBuffer, vWeightsOffset: Int,
                vScales: MTLBuffer, vScalesOffset: Int,
                vBiases: MTLBuffer, vBiasesOffset: Int,
                x: MTLBuffer,
                qOut: MTLBuffer, qOutOffset: Int,
                kOut: MTLBuffer, kOutOffset: Int,
                vOut: MTLBuffer, vOutOffset: Int,
                qRows: UInt32,
                kvRows: UInt32,
                n: UInt32)
}

/// Final norm, tied LM-head projection and greedy argmax in one chain.
protocol AffineGreedyHeadEncoding: AnyObject {
    func encodeGreedyDecode(commandBuffer: MTLCommandBuffer,
                            hidden: MTLBuffer,
                            hiddenOffset: Int,
                            normWeight: MTLBuffer,
                            normOffset: Int,
                            weights: MTLBuffer,
                            weightsOffset: Int,
                            scales: MTLBuffer,
                            scalesOffset: Int,
                            biases: MTLBuffer,
                            biasesOffset: Int,
                            outToken: MTLBuffer,
                            d: UInt32,
                            vocab: UInt32,
                            rmsEps: Float)
}

/// Final norm and full-vocabulary logits for the last prefill row.
protocol AffinePrefillFinalRowHeadEncoding: AnyObject {
    func encodeLogits(commandBuffer: MTLCommandBuffer,
                      hiddenBlock: MTLBuffer,
                      row: Int,
                      rowStrideElements: Int,
                      normWeight: MTLBuffer,
                      normWeightOffset: Int,
                      weights: MTLBuffer,
                      weightsOffset: Int,
                      scales: MTLBuffer,
                      scalesOffset: Int,
                      biases: MTLBuffer,
                      biasesOffset: Int,
                      logits: MTLBuffer,
                      logitsOffset: Int,
                      d: UInt32,
                      vocab: UInt32,
                      rmsEps: Float)
}

/// Prefill embedding lookup for a block of tokens.
protocol AffinePrefillEmbedEncoding: AnyObject {
    func encode(commandBuffer: MTLCommandBuffer,
                table: MTLBuffer, tableOffset: Int,
                scales: MTLBuffer, scalesOffset: Int,
                biases: MTLBuffer, biasesOffset: Int,
                tokens: MTLBuffer, tokensOffset: Int,
                out: MTLBuffer, outOffset: Int,
                t: UInt32,
                d: UInt32,
                outScale: Float)
}

/// Prefill projection `Y[t, n] = X[t, :] · W[n, :]` without matrix units.
protocol AffinePrefillQMMEncoding: AnyObject {
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int,
                scales: MTLBuffer, scalesOffset: Int,
                biases: MTLBuffer, biasesOffset: Int,
                x: MTLBuffer, xOffset: Int,
                y: MTLBuffer, yOffset: Int,
                t: Int,
                n: Int,
                k: Int)
}

/// Prefill projection on Metal Performance Primitives with staged
/// dequantization. Reports which path ran; `.fallback` means nothing was
/// encoded and the caller must use another kernel.
protocol AffineMPPPrefillQMMEncoding: AnyObject {
    var isAvailable: Bool { get }

    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int,
                scales: MTLBuffer, scalesOffset: Int,
                biases: MTLBuffer, biasesOffset: Int,
                x: MTLBuffer, xOffset: Int,
                y: MTLBuffer, yOffset: Int,
                m: Int,
                n: Int,
                k: Int) -> MPPPrefillInt4QMM.PathMetadata
}

extension DequantInt4GEMV: AffineGEMVEncoding {}
extension DequantInt8GEMV: AffineGEMVEncoding {}
extension EmbedLookupInt4: AffineEmbedLookupEncoding {}
extension FusedQKVGEMV: AffineFusedQKVEncoding {}
extension LMHeadChainInt4: AffineGreedyHeadEncoding {}
extension PrefillFinalRowHeadInt4: AffinePrefillFinalRowHeadEncoding {}
extension PrefillEmbedLookupInt4: AffinePrefillEmbedEncoding {}
extension PrefillInt4QMM: AffinePrefillQMMEncoding {}
extension MPPPrefillInt4QMM: AffineMPPPrefillQMMEncoding {}
