#include <metal_stdlib>
using namespace metal;

// ============================================================================
// prefill_int8 — INT8 affine dense prefill kernels: block embedding and QMM.
//
// Placeholder registered by S0 of docs/plans/8bit-model-variant-plan.md.
// Owner: S4 (function constants 120-129). This module is concatenated after the 4-bit modules, so it may
// reuse their structs, helpers and function constants. Every new symbol
// carries an int8 suffix or prefix; new function-constant indices stay in
// the range the plan assigns to the owner.
// ============================================================================
