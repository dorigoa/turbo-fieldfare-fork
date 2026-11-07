#include <metal_stdlib>
using namespace metal;

// ============================================================================
// tensorops_int8 — INT8 affine prefill projections on Metal Performance
// Primitives, with dequantization staged through FP16 threadgroup memory.
//
// Twin of `mpp_prefill_affine_threadgroup_f16` in tensorops.metal for MLX
// affine 8-bit weights (group 64, BF16 scale and bias per group). Same
// bindings, same 64 x 32 x 64 tile and the same per-group accumulation; only
// the staging loop changes, reading one 4-byte word (four columns) per step
// instead of one nibble.
//
// Owner: S4 of docs/plans/8bit-model-variant-plan.md (function constants
// 120-129, none used). Compiled on its own through
// `MetalContext.privateLibrary(device:module:)`, like `tensorops`, so it does
// not share symbols with the runtime library. Keep tensor code inside the
// guard: devices without tensor support compile an empty library.
// ============================================================================

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr uint kMPPAffine8GroupSize = 64;

constant constexpr int kMPPAffine8TileM = 64;
constant constexpr int kMPPAffine8TileN = 32;
constant constexpr int kMPPAffine8TileK = 64;
constant constexpr uint kMPPAffine8WordsPerTileRow = uint(kMPPAffine8TileK) / 4u;

// output[m, n] = activations[m, :] · W[n, :]
//
// Alignment: the weight base must be a multiple of 4 bytes (rows are K bytes
// and K is a multiple of 64, so every staged word stays aligned); scales,
// biases, activations and output need 2 bytes. `MPPPrefillInt8QMM` reports a
// fallback instead of dispatching when an offset breaks these rules.
kernel void mpp_prefill_affine8_threadgroup_f16(
    device const uint8_t* weights       [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffine8TileM, kMPPAffine8TileN, kMPPAffine8TileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[kMPPAffine8TileN * kMPPAffine8TileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffine8TileK, kMPPAffine8TileN),
        array<int32_t, 2>({1, kMPPAffine8TileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffine8TileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0,
        int32_t(tgid.y) * kMPPAffine8TileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint groupsPerRow = K / kMPPAffine8GroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < groupsPerRow; ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffine8TileN) * kMPPAffine8WordsPerTileRow;
             linear += threads) {
            const uint localN = linear / kMPPAffine8WordsPerTileRow;
            const uint localWord = linear % kMPPAffine8WordsPerTileRow;
            const uint globalN = tgid.x * uint(kMPPAffine8TileN) + localN;
            threadgroup half* staged = weightTile
                + localN * uint(kMPPAffine8TileK) + localWord * 4u;
            if (globalN < N) {
                device const uint* groupWords = reinterpret_cast<device const uint*>(
                    weights + globalN * K + group * kMPPAffine8GroupSize);
                const uint word = groupWords[localWord];
                const float scale = float(scales[globalN * groupsPerRow + group]);
                const float bias = float(biases[globalN * groupsPerRow + group]);
                staged[0] = half(fma(float(word & 0xffu), scale, bias));
                staged[1] = half(fma(float((word >> 8) & 0xffu), scale, bias));
                staged[2] = half(fma(float((word >> 16) & 0xffu), scale, bias));
                staged[3] = half(fma(float(word >> 24), scale, bias));
            } else {
                staged[0] = half(0.0f);
                staged[1] = half(0.0f);
                staged[2] = half(0.0f);
                staged[3] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + group * uint(kMPPAffine8TileK),
            dextents<int32_t, 2>(kMPPAffine8TileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0,
            int32_t(tgid.y) * kMPPAffine8TileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        // As in the 4-bit twin: keep the next group from overwriting B while
        // another SIMD group can still be consuming the current tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffine8TileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffine8TileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

#endif // __HAVE_TENSOR__
