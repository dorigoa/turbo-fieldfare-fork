#include <metal_stdlib>
using namespace metal;

// Qwen 3.5 MoE normalization kernels (row RMSNorm with w or 1+w, per-head
// RMSNorm for q/k, gated per-head RMSNorm with silu(z), per-head L2 norm).
// Owned by plan section S12.
