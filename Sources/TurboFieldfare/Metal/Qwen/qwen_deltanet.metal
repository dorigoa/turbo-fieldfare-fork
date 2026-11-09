#include <metal_stdlib>
using namespace metal;

// Qwen 3.5 MoE Gated DeltaNet kernels: causal depthwise conv, gates and the
// recurrent state update. Owned by plan sections S14 and S15.
