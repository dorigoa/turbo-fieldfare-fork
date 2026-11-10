#include <metal_stdlib>
using namespace metal;

// Qwen 3.5 MoE gated attention kernels: RoPE, causal GQA attention and the
// output gate. Owned by plan section S16.

// Split the q_proj output into the query and the gate of each head.
// q_proj writes [rows][queryHeads * 2 * headDim]; per head the first
// `headDim` values are the query and the next `headDim` are the gate
// (plan 3.5: "q = qg[0:256], gate = qg[256:512]"). Pure element copy, so
// q and gate are bit-exact slices of the projection output.
kernel void qwen_split_qg(
    device const half* qg [[buffer(0)]],
    device half* q [[buffer(1)]],
    device half* gate [[buffer(2)]],
    constant uint& queryHeads [[buffer(3)]],
    constant uint& headDim [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]   // (dim, head, row)
) {
    const uint d = gid.x, h = gid.y, r = gid.z;
    if (d >= headDim || h >= queryHeads || r >= rows) return;
    const size_t src = ((size_t)r * queryHeads + h) * (2u * (size_t)headDim);
    const size_t dst = ((size_t)r * queryHeads + h) * (size_t)headDim + d;
    q[dst] = qg[src + d];
    gate[dst] = qg[src + headDim + d];
}
