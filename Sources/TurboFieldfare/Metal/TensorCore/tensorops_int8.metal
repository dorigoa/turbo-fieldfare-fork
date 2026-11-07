#include <metal_stdlib>
using namespace metal;

// ============================================================================
// tensorops_int8 — INT8 affine prefill projections on Metal Performance
// Primitives, with dequantization staged through FP16 threadgroup memory.
//
// Placeholder registered by S0 of docs/plans/8bit-model-variant-plan.md.
// Owner: S4 (function constants 120-129). Compiled on its own through
// `MetalContext.privateLibrary(device:module:)`, like `tensorops`, so it does
// not share symbols with the runtime library. Keep tensor code inside the
// guard: devices without tensor support compile an empty library.
// ============================================================================

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

#endif // __HAVE_TENSOR__
