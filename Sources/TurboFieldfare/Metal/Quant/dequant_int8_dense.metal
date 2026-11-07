#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int8_dense — INT8 affine dense decode kernels: embedding, fused QKV, greedy LM head.
//
// Placeholder registered by S0 of docs/plans/8bit-model-variant-plan.md.
// Owner: S2 (function constants 100-109). This module is concatenated after the 4-bit modules, so it may
// reuse their structs, helpers and function constants. Every new symbol
// carries an int8 suffix or prefix; new function-constant indices stay in
// the range the plan assigns to the owner.
// ============================================================================
