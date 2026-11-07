#include <metal_stdlib>
using namespace metal;

// ============================================================================
// moe_int8 — INT8 affine routed-expert decode kernels.
//
// Placeholder registered by S0 of docs/plans/8bit-model-variant-plan.md.
// Owner: S3 (function constants 110-119). This module is concatenated after the 4-bit modules, so it may
// reuse their structs, helpers and function constants. Every new symbol
// carries an int8 suffix or prefix; new function-constant indices stay in
// the range the plan assigns to the owner.
// ============================================================================
