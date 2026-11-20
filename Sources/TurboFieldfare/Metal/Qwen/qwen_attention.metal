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

// Grouped split-KV attention, the production path for every shape it covers
// (headDim a multiple of 128 up to 256, at most 8 query heads per KV head).
//
// `qwen_attention` above gives each query head its own threadgroup and walks
// the whole cache one position at a time with two threadgroup barriers per
// position: in decode that is 16 threadgroups on the whole GPU, measured at
// ~1.1 µs per position per layer, so its cost grew with the context and
// dominated the token time well before the context was full. Here:
//
//   - one threadgroup per (KV head, chunk of positions, query token);
//   - SIMD group g serves query head kvHead * qPerKV + g, so the 8 query heads
//     of a GQA group share every K/V row the threadgroup reads;
//   - lane l owns the half4 columns l and l + 32 of a 256-wide head, so one
//     row is one coalesced load per column block and the per-position dot is a
//     single simd_sum — no threadgroup barrier anywhere in the loop;
//   - with numChunks > 1 (decode) each threadgroup writes the un-normalized
//     online-softmax state of its chunk and qwen_attention_combine merges the
//     chunks; with numChunks == 1 (prefill) the output is written directly.
//
// Same algorithm and FP32 accumulation as qwen_attention: scores carry
// `scale`, softmax is the online max/rescale recurrence. Query token t of a
// call attends cache positions [0, kvStart + t].
constant constexpr uint kQwenAttnMaxVecs = 2;   // half4 columns per lane: headDim / 128

[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_attention_grouped(
    device const half*  q          [[buffer(0)]],   // [T][queryHeads*headDim]
    device const half*  kvK        [[buffer(1)]],   // cache, D5 strides
    device const half*  kvV        [[buffer(2)]],
    device       half*  out        [[buffer(3)]],   // [T][queryHeads*headDim] (numChunks == 1)
    device       float* mOut       [[buffer(4)]],   // [T*queryHeads][numChunks] (numChunks > 1)
    device       float* dOut       [[buffer(5)]],
    device       float* oOut       [[buffer(6)]],   // [T*queryHeads][numChunks][headDim]
    constant     uint&  queryHeads [[buffer(7)]],
    constant     uint&  kvHeads    [[buffer(8)]],
    constant     uint&  headDim    [[buffer(9)]],
    constant     uint&  queryTokens [[buffer(10)]],
    constant     uint&  kvStart    [[buffer(11)]],
    constant     uint&  posStride  [[buffer(12)]],
    constant     uint&  headStride [[buffer(13)]],
    constant     float& scale      [[buffer(14)]],
    constant     uint&  chunkLen   [[buffer(15)]],
    constant     uint&  numChunks  [[buffer(16)]],
    uint3 tg   [[threadgroup_position_in_grid]],    // (kvHead, chunk, token)
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]]
) {
    const uint kvh = tg.x, chunk = tg.y, t = tg.z;
    const uint qPerKV = queryHeads / kvHeads;
    // No threadgroup barriers below, so idle SIMD groups may leave early.
    if (kvh >= kvHeads || t >= queryTokens || sg >= qPerKV) return;
    const uint h = kvh * qPerKV + sg;
    const uint vecs = headDim / 128u;
    const uint kvLen = kvStart + t + 1u;
    const uint pStart = chunk * chunkLen;
    const uint pEnd = min(pStart + chunkLen, kvLen);

    device const half4* qrow =
        (device const half4*)(q + ((size_t)t * queryHeads + h) * (size_t)headDim);
    float4 qv[kQwenAttnMaxVecs];
    float4 acc[kQwenAttnMaxVecs];
    for (uint j = 0u; j < kQwenAttnMaxVecs; ++j) {
        qv[j] = j < vecs ? float4(qrow[lane + 32u * j]) : float4(0.f);
        acc[j] = float4(0.f);
    }

    float m = -1e30f;
    float d = 0.f;
    const size_t headOffset = (size_t)kvh * (size_t)headStride;
    // Four positions per step: the four K rows (and then the four V rows) are
    // independent loads, so each lane keeps several in flight instead of
    // waiting out one memory latency per position. One rescale per step.
    uint p = pStart;
    for (; p + 4u <= pEnd; p += 4u) {
        float s[4];
        for (uint r = 0u; r < 4u; ++r) {
            const size_t base = (size_t)(p + r) * (size_t)posStride + headOffset;
            device const half4* krow = (device const half4*)(kvK + base);
            float part = 0.f;
            for (uint j = 0u; j < kQwenAttnMaxVecs; ++j) {
                if (j < vecs) part += dot(qv[j], float4(krow[lane + 32u * j]));
            }
            s[r] = simd_sum(part) * scale;
        }
        const float mNew = max(max(m, max(s[0], s[1])), max(s[2], s[3]));
        const float alpha = exp(m - mNew);
        float e[4];
        for (uint r = 0u; r < 4u; ++r) e[r] = exp(s[r] - mNew);
        d = d * alpha + ((e[0] + e[1]) + (e[2] + e[3]));
        for (uint j = 0u; j < kQwenAttnMaxVecs; ++j) {
            if (j >= vecs) continue;
            float4 add = float4(0.f);
            for (uint r = 0u; r < 4u; ++r) {
                const size_t base = (size_t)(p + r) * (size_t)posStride + headOffset;
                add += e[r] * float4(((device const half4*)(kvV + base))[lane + 32u * j]);
            }
            acc[j] = acc[j] * alpha + add;
        }
        m = mNew;
    }
    for (; p < pEnd; ++p) {
        const size_t base = (size_t)p * (size_t)posStride + headOffset;
        device const half4* krow = (device const half4*)(kvK + base);
        device const half4* vrow = (device const half4*)(kvV + base);
        float part = 0.f;
        for (uint j = 0u; j < kQwenAttnMaxVecs; ++j) {
            if (j < vecs) part += dot(qv[j], float4(krow[lane + 32u * j]));
        }
        const float s = simd_sum(part) * scale;
        const float mNew = max(m, s);
        const float alpha = exp(m - mNew);
        const float e = exp(s - mNew);
        d = d * alpha + e;
        for (uint j = 0u; j < kQwenAttnMaxVecs; ++j) {
            if (j < vecs) acc[j] = acc[j] * alpha + e * float4(vrow[lane + 32u * j]);
        }
        m = mNew;
    }

    if (numChunks == 1u) {
        device half4* orow =
            (device half4*)(out + ((size_t)t * queryHeads + h) * (size_t)headDim);
        const float inv = 1.f / d;
        for (uint j = 0u; j < kQwenAttnMaxVecs; ++j) {
            if (j < vecs) orow[lane + 32u * j] = half4(acc[j] * inv);
        }
        return;
    }
    // An empty chunk leaves (m, d, acc) = (-1e30, 0, 0); the combine weighs
    // it by exp(-1e30 - mGlobal) = 0.
    const size_t pi = ((size_t)t * queryHeads + h) * (size_t)numChunks + chunk;
    if (lane == 0u) { mOut[pi] = m; dOut[pi] = d; }
    device float4* orow = (device float4*)(oOut + pi * (size_t)headDim);
    for (uint j = 0u; j < kQwenAttnMaxVecs; ++j) {
        if (j < vecs) orow[lane + 32u * j] = acc[j];
    }
}

// Merge the per-chunk states of qwen_attention_grouped: one threadgroup per
// (query token, query head) row,
//   mG = max_c m_c,  w_c = exp(m_c - mG),  out = sum_c w_c o_c / sum_c w_c d_c,
// with a single FP16 rounding at the end.
constant constexpr uint kQwenAttnMaxChunks = 128;

[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_attention_combine(
    device const float* mIn       [[buffer(0)]],
    device const float* dIn       [[buffer(1)]],
    device const float* oIn       [[buffer(2)]],
    device       half*  out       [[buffer(3)]],   // [rows][headDim]
    constant     uint&  headDim   [[buffer(4)]],
    constant     uint&  numChunks [[buffer(5)]],
    constant     uint&  rows      [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    if (row >= rows) return;
    threadgroup float w[kQwenAttnMaxChunks];
    device const float* mRow = mIn + (size_t)row * numChunks;
    device const float* dRow = dIn + (size_t)row * numChunks;
    float mG = -1e30f;
    for (uint c = 0u; c < numChunks; ++c) mG = max(mG, mRow[c]);
    for (uint c = tid; c < numChunks; c += tsize) w[c] = exp(mRow[c] - mG);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denom = 0.f;
    for (uint c = 0u; c < numChunks; ++c) denom += dRow[c] * w[c];
    const float inv = 1.f / denom;
    device const float* oRow = oIn + (size_t)row * numChunks * (size_t)headDim;
    device half* dst = out + (size_t)row * (size_t)headDim;
    for (uint i = tid; i < headDim; i += tsize) {
        float acc = 0.f;
        for (uint c = 0u; c < numChunks; ++c) acc += oRow[(size_t)c * headDim + i] * w[c];
        dst[i] = half(acc * inv);
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
