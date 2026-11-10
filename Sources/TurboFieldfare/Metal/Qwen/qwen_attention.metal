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

// Partial NeoX-style RoPE (plan 3.5): rotate only the FIRST 2*rotatedPairs
// dims of each head, pairing dimension i with i + rotatedPairs
// (Qwen "rotate_half" over the rotated block), with
// inv_freq_i = theta^(-2i / (2*rotatedPairs)). The attention layer rotates
// the first 64 dims of a 256-wide head with theta 1e7. Token r sits at
// position positionBase + r. In-place safe: each pair touches only its two
// dimensions. This is NOT Gemma's rope_proportional_neox, which pairs
// (i, headDim/2 + i) and divides the exponent by headDim.
kernel void qwen_rope_partial(
    device half* data [[buffer(0)]],           // [rows][heads*headDim] FP16
    constant uint& positionBase [[buffer(1)]], // position of token row 0
    constant uint& headDim [[buffer(2)]],
    constant uint& numHeads [[buffer(3)]],
    constant uint& rotatedPairs [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]      // (pair, head, row)
) {
    const uint pair = gid.x, h = gid.y, r = gid.z;
    if (pair >= rotatedPairs || h >= numHeads) return;
    device half* head0 = data + ((size_t)r * numHeads + h) * (size_t)headDim;
    const float rotDim = float(2u * rotatedPairs);
    const float invFreq = pow(theta, -float(2u * pair) / rotDim);
    const float angle = float(positionBase + r) * invFreq;
    const float c = cos(angle), s = sin(angle);
    const float x0 = float(head0[pair]);
    const float x1 = float(head0[rotatedPairs + pair]);
    head0[pair] = half(x0 * c - x1 * s);
    head0[rotatedPairs + pair] = half(x0 * s + x1 * c);
}

// Append rows of K and V to the attention KV cache (D5 layout inside the
// per-layer slice: [position][kvHead][headDim], so headStride = headDim and
// posStride = kvHeads * headDim; both are parameters so the runner can pass
// a capacity-strided buffer and write token t at cache position
// kvStart + t). Pure element copy: bit-exact.
kernel void qwen_kv_write(
    device const half* k [[buffer(0)]],      // [rows][kvHeads*headDim]
    device const half* v [[buffer(1)]],
    device half* kvK [[buffer(2)]],          // cache [positions][kvHeads][headDim]
    device half* kvV [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& kvHeads [[buffer(5)]],
    constant uint& headDim [[buffer(6)]],
    constant uint& kvStart [[buffer(7)]],
    constant uint& posStride [[buffer(8)]],  // elements between cached positions
    constant uint& headStride [[buffer(9)]], // elements between KV heads
    uint3 gid [[thread_position_in_grid]]    // (dim, head, row)
) {
    const uint d = gid.x, h = gid.y, r = gid.z;
    if (d >= headDim || h >= kvHeads || r >= rows) return;
    const size_t src = ((size_t)r * kvHeads + h) * (size_t)headDim + d;
    const size_t dst = ((size_t)(kvStart + r) * (size_t)posStride)
                     + (size_t)h * (size_t)headStride + d;
    kvK[dst] = k[src];
    kvV[dst] = v[src];
}

// Causal GQA attention over the D5 KV cache (plan 3.5): query head h reads
// KV head h / (queryHeads / kvHeads); scores carry `scale` (1/sqrt(headDim)
// = 1/16 at the real head_dim 256) and the softmax runs in FP32 with an
// online max/rescale. One threadgroup per (query head, query token): thread
// tid owns dimension tid (headDim <= 256), the per-position dot is a
// simd_sum tree plus an 8-way threadgroup reduce, the output accumulator
// lives in a register. Query token t attends cache positions
// [0, kvStart + t] — causal; decode (queryTokens = 1) and prefill (T)
// share the code path. K/V use the same cache strides as qwen_kv_write.
kernel void qwen_attention(
    device const half* q [[buffer(0)]],      // [T][queryHeads*headDim]
    device const half* kvK [[buffer(1)]],    // cache, D5 strides
    device const half* kvV [[buffer(2)]],
    device half* out [[buffer(3)]],          // [T][queryHeads*headDim]
    constant uint& queryHeads [[buffer(4)]],
    constant uint& kvHeads [[buffer(5)]],
    constant uint& headDim [[buffer(6)]],
    constant uint& queryTokens [[buffer(7)]],
    constant uint& kvStart [[buffer(8)]],
    constant uint& posStride [[buffer(9)]],
    constant uint& headStride [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],   // (head, token)
    uint tid [[thread_index_in_threadgroup]]     // 0..255
) {
    const uint h = tg.x, t = tg.y;
    if (h >= queryHeads || t >= queryTokens) return;
    const uint kvh = h / (queryHeads / kvHeads);
    const uint kvLen = kvStart + t + 1u;
    const uint lane = tid & 31u, sg = tid >> 5;
    const bool active = tid < headDim;

    threadgroup float qv[256];
    threadgroup float parts[8];
    threadgroup float mRun, sumRun, alphaBC, eBC;

    device const half* qrow = q + ((size_t)t * queryHeads + h) * (size_t)headDim;
    if (active) qv[tid] = float(qrow[tid]);
    if (tid == 0u) { mRun = -1e30f; sumRun = 0.f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half* kbase = kvK + (size_t)kvh * (size_t)headStride;
    device const half* vbase = kvV + (size_t)kvh * (size_t)headStride;
    const float qd = active ? qv[tid] : 0.f;
    float acc = 0.f;

    for (uint p = 0u; p < kvLen; ++p) {
        const size_t pbase = (size_t)p * (size_t)posStride;
        float part = 0.f;
        if (active) part = qd * float(kbase[pbase + tid]);
        part = simd_sum(part);
        if (lane == 0u) parts[sg] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float s = 0.f;
            for (uint g = 0u; g < (headDim + 31u) / 32u; ++g) s += parts[g];
            s *= scale;
            const float mOld = mRun;
            const float mNew = max(mOld, s);
            const float e = exp(s - mNew);
            const float alpha = exp(mOld - mNew);
            sumRun = sumRun * alpha + e;
            mRun = mNew;
            alphaBC = alpha;
            eBC = e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float vp = 0.f;
        if (active) vp = float(vbase[pbase + tid]);
        acc = acc * alphaBC + eBC * vp;
    }

    if (active) {
        out[((size_t)t * queryHeads + h) * (size_t)headDim + tid]
            = half(acc / sumRun);
    }
}

// Output gate (plan 3.5): out = attn * sigmoid(gate), elementwise over the
// flattened [T][queryHeads*headDim] layout; `gate` is the second half of the
// q_proj output recovered by qwen_split_qg. Sigmoid in FP32 with a single
// FP16 rounding at the end. S15's qwen_sigmoid_f16 computes sigmoid(x)
// alone, so it cannot be reused unmodified for the fused multiply.
kernel void qwen_attn_gate(
    device const half* attn [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const float g = float(gate[gid]);
    out[gid] = half(float(attn[gid]) * (1.0f / (1.0f + exp(-g))));
}
