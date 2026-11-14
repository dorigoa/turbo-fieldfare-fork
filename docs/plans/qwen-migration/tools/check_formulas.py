# usage: scratch/qwen36-venv/bin/python -I docs/plans/qwen-migration/tools/check_formulas.py <model_dir> <fixtures_dir>
# Implements the 3.5 formulas in NumPy (float32) with dequantized weights and
# compares them against the S8 internal-quantity golden (Appendix D).
# Prints one line per formula: "OK|FAIL name rel=<max abs diff / max abs ref>".
import base64
import json
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load

TOL = 1e-2
RESULTS = []


def dec(o):
    return np.frombuffer(base64.b64decode(o["f32b64"]), dtype="<f4").reshape(o["shape"])


def rms(x, w, eps=1e-6):
    # 3.5: x_hat * w' with x_hat = x * rsqrt(mean(x^2) + eps), w' = w (S1:
    # normWeightsShifted = true; also verified on mlx.nn.RMSNorm).
    return x * (1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def check(name, mine, ref):
    d = float(np.max(np.abs(mine - ref)))
    s = float(np.max(np.abs(ref))) + 1e-12
    ok = d / s <= TOL
    RESULTS.append((ok, name, d / s))
    print(f"{'OK  ' if ok else 'FAIL'} {name} rel={d / s:.3e}", flush=True)


def deq(module):
    """Dequantize an mlx quantized Linear/Embedding to float32 numpy [out, in]."""
    q, s, b = module.weight, module.scales, module.biases
    cols = s.shape[-1] * 64
    bits = (q.shape[-1] * 32) // cols
    return np.array(mx.dequantize(q, s, b, 64, bits).astype(mx.float32))


def deq_row(module, row):
    q, s, b = module.weight, module.scales, module.biases
    cols = s.shape[-1] * 64
    bits = (q.shape[-1] * 32) // cols
    return np.array(
        mx.dequantize(
            q[row : row + 1], s[row : row + 1], b[row : row + 1], 64, bits
        ).astype(mx.float32)
    )[0]


def main():
    t0 = time.time()
    model, tok = load(sys.argv[1])
    fx = sys.argv[2]
    lm = model.language_model
    tm = lm.model
    G = lambda n: json.load(open(f"{fx}/{n}", encoding="utf-8"))
    h = G("hidden_p0.json")
    l0 = G("layer0_internals.json")
    l3 = G("layer3_internals.json")
    topk = G("logits_topk.json")
    hid_emb = dec(h["embedding"])  # residual input of layer 0
    T = hid_emb.shape[0]

    # --- Embedding: h = dequant(embed)[token], no sqrt(hidden) scaling. ---
    ids = None
    for p in G("prompts.json"):
        if p["name"] == "p0":
            ids = p["ids"]
    mine = np.stack([deq_row(tm.embed_tokens, t) for t in ids])
    check("embed_lookup", mine, hid_emb)

    # --- Layer 0 (Gated DeltaNet + MoE) ---
    layer = lm.layers[0]
    la = layer.linear_attn
    x0 = dec(l0["x_norm"])
    check(
        "l0.input_layernorm(x_hat*w)",
        rms(hid_emb, np.array(layer.input_layernorm.weight.astype(mx.float32))),
        x0,
    )
    Wq = deq(la.in_proj_qkv)
    Wz = deq(la.in_proj_z)
    Wa = deq(la.in_proj_a)
    Wb = deq(la.in_proj_b)
    qkv_ref = dec(l0["qkv_raw"])
    check("l0.qkv = W_qkv x", x0 @ Wq.T, qkv_ref)
    check("l0.z = W_z x", x0 @ Wz.T, dec(l0["z"]))
    check("l0.a = W_a x", x0 @ Wa.T, dec(l0["a"]))
    check("l0.b = W_b x", x0 @ Wb.T, dec(l0["b"]))

    # causal depthwise conv, kernel 4, taps j=0..3 oldest->newest, then SiLU
    wtap = np.array(la.conv1d.weight.astype(mx.float32))[:, :, 0]  # [8192, 4]
    pad = np.zeros((3, qkv_ref.shape[1]), dtype=np.float32)
    upad = np.concatenate([pad, qkv_ref], axis=0)  # u[t-3+j] -> upad[t + j]
    conv = np.zeros_like(qkv_ref)
    for j in range(4):
        conv += upad[j : j + T] * wtap[:, j]
    check("l0.conv_taps+silu", silu(conv), dec(l0["conv_out"]))

    conv_out = dec(l0["conv_out"])
    qh = conv_out[:, :2048].reshape(T, 16, 128)
    kh = conv_out[:, 2048:4096].reshape(T, 16, 128)
    vh = conv_out[:, 4096:8192].reshape(T, 32, 128)
    # q_i = q_i * rsqrt(sum q_i^2 + 1e-6) (per head), then q /= sqrt(128)
    def l2(t, eps=1e-6):
        return t * (1.0 / np.sqrt((t * t).sum(axis=-1, keepdims=True) + eps))

    check("l0.q_l2norm+scale", l2(qh) * (128.0**-0.5), dec(l0["q"]))
    check("l0.k_l2norm", l2(kh), dec(l0["k"]))
    a_g = dec(l0["a"])
    b_g = dec(l0["b"])
    A_log = np.array(la.A_log.astype(mx.float32))
    dt_bias = np.array(la.dt_bias.astype(mx.float32))
    sp = np.log1p(np.exp(np.clip(a_g + dt_bias, -30, 30)))
    check("l0.g = -exp(A_log)*softplus(a+dt_bias)", -np.exp(A_log) * sp, dec(l0["g"]))
    check("l0.beta = sigmoid(b)", 1.0 / (1.0 + np.exp(-b_g)), dec(l0["beta"]))

    # DeltaNet recurrence, S[head][key i][value c] (3.5 notation), FP32
    beta = 1.0 / (1.0 + np.exp(-b_g))  # [T, 32]
    decay = np.exp(dec(l0["g"]))  # [T, 32]
    qq = dec(l0["q"])  # [T, 16, 128] (scaled)
    kk = dec(l0["k"])  # [T, 16, 128]
    S = np.zeros((32, 128, 128), dtype=np.float32)  # [Hv][key i][value c]
    core = np.zeros((T, 32, 128), dtype=np.float32)
    for t in range(T):
        for j in range(32):
            kj = kk[t, j // 2]
            qj = qq[t, j // 2]
            S[j] *= decay[t, j]
            kv_mem = S[j].T @ kj  # sum_i S[i][c] k[i]
            delta = (vh[t, j] - kv_mem) * beta[t, j]
            S[j] += np.outer(kj, delta)
            core[t, j] = S[j].T @ qj
    check("l0.deltanet_core", core, dec(l0["core_out"]))
    check(
        "l0.deltanet_state_h0..3",
        np.ascontiguousarray(S[:4]),
        dec(l0["state_heads0to3"]),
    )

    wnorm = np.array(la.norm.weight.astype(mx.float32))
    gated = rms(dec(l0["core_out"]), wnorm) * silu(dec(l0["z"]).reshape(T, 32, 128))
    check("l0.gated_norm*rmsnorm_w*silu(z)", gated, dec(l0["gated_out"]))
    Wo = deq(la.out_proj)
    check("l0.out_proj", gated.reshape(T, 4096) @ Wo.T, dec(l0["mixer_out"]))

    moe_x = dec(l0["moe_x"])
    check(
        "l0.post_attention_layernorm",
        rms(hid_emb + dec(l0["mixer_out"]),
            np.array(layer.post_attention_layernorm.weight.astype(mx.float32))),
        moe_x,
    )
    moe = layer.mlp
    p = softmax(moe_x @ deq(moe.gate).T)  # [T, 256]
    top_ids_ref = l0["router_top_ids"]
    top_w_ref = dec(l0["router_top_weights"])
    ids_ok, wsum, wcnt = True, 0.0, 0
    for t in range(T):
        mine_top = np.argsort(-p[t], kind="stable")[:8]
        if set(mine_top.tolist()) != set(top_ids_ref[t]):
            ids_ok = False
            continue
        order_ref = np.argsort(top_ids_ref[t])
        order_mine = np.argsort(mine_top)
        w = p[t, mine_top]
        w = w / w.sum()
        wsum += float(np.max(np.abs(w[order_mine] - top_w_ref[t][order_ref])))
        wcnt += 1
    rel = wsum / wcnt / float(np.max(np.abs(top_w_ref)) + 1e-12)
    ok = ids_ok and rel <= TOL
    RESULTS.append((ok, "l0.router_topk+renorm", max(rel, 0.0 if ids_ok else 1.0)))
    print(
        f"{'OK  ' if ok else 'FAIL'} l0.router_topk+renorm ids={'yes' if ids_ok else 'NO'} rel={rel:.3e}",
        flush=True,
    )

    # routed experts (SiLU), shared expert with sigmoid scalar gate
    gate_ref = dec(l0["router_top_weights"])
    up_w = moe.switch_mlp.up_proj
    gated_w = moe.switch_mlp.gate_proj
    down_w = moe.switch_mlp.down_proj
    routed = np.zeros((T, 2048), dtype=np.float32)
    experts = sorted({e for row in top_ids_ref for e in row})
    cache = {e: (deq_row(gated_w, e), deq_row(up_w, e), deq_row(down_w, e)) for e in experts}
    for t in range(T):
        for i, e in enumerate(top_ids_ref[t]):
            wg, wu, wd = cache[e]
            u = moe_x[t] @ wg.T
            check_u = moe_x[t] @ wu.T
            routed[t] += gate_ref[t, i] * (silu(u) * check_u) @ wd.T
    check("l0.routed_experts_silu", routed, dec(l0["routed_out"]))
    se = moe.shared_expert
    u = moe_x @ deq(se.gate_proj).T
    c = moe_x @ deq(se.up_proj).T
    shared = (silu(u) * c) @ deq(se.down_proj).T
    sg = 1.0 / (1.0 + np.exp(-(moe_x @ deq(moe.shared_expert_gate).T[:, 0])))
    check("l0.shared_expert_x_sigmgate", sg[:, None] * shared, dec(l0["shared_out"]))
    check(
        "l0.layer_out_residual",
        (hid_emb + dec(l0["mixer_out"])) + dec(l0["routed_out"]) + dec(l0["shared_out"]),
        dec(l0["layer_out"]),
    )

    # --- Layer 3 (gated attention) ---
    layer = lm.layers[3]
    sa = layer.self_attn
    x3 = dec(l3["x_norm"])
    qg = x3 @ deq(sa.q_proj).T
    qh3 = qg.reshape(T, 16, 2, 256)[:, :, 0, :]
    gate3 = qg.reshape(T, 16, 2, 256)[:, :, 1, :]
    check("l3.q_gate_split", gate3, dec(l3["gate"]))
    wqn = np.array(sa.q_norm.weight.astype(mx.float32))
    check("l3.q_rmsnorm(q_norm w)", rms(qh3, wqn), dec(l3["q_pre_norm"]))
    kh3 = x3 @ deq(sa.k_proj).T
    vh3 = x3 @ deq(sa.v_proj).T
    wkn = np.array(sa.k_norm.weight.astype(mx.float32))
    check("l3.k_rmsnorm(k_norm w)", rms(kh3.reshape(T, 2, 256), wkn), dec(l3["k_pre_norm"]))
    check("l3.v_proj", vh3.reshape(T, 2, 256), dec(l3["v"]))

    # RoPE on first 64 dims, rotate_half pairs (i, i+32), inv_freq = 1e7^(-2i/64)
    def rope(x, head_dim_rot=64, theta=1e7):
        out = x.copy()
        D = x.shape[-1]
        half = head_dim_rot // 2
        inv = theta ** (-2.0 * np.arange(half) / head_dim_rot)
        for t in range(T):
            ang = t * inv
            c, s = np.cos(ang), np.sin(ang)
            a = x[t, ..., :half]
            bpart = x[t, ..., half:head_dim_rot]
            out[t, ..., :half] = a * c - bpart * s
            out[t, ..., half:head_dim_rot] = bpart * c + a * s
        return out

    check("l3.rope_q", rope(dec(l3["q_pre_norm"])), dec(l3["q_rope"]))
    check("l3.rope_k", rope(dec(l3["k_pre_norm"])), dec(l3["k_rope"]))

    qf = dec(l3["q_rope"])  # [T,16,256]
    kf = dec(l3["k_rope"])  # [T,2,256]
    vf = dec(l3["v"])
    attn = np.zeros_like(qf)
    scale = 256.0**-0.5
    for t in range(T):
        for hq in range(16):
            hk = hq // 8
            logits = (kf[: t + 1, hk] @ qf[t, hq]) * scale  # [t+1]
            pr = softmax(logits)
            attn[t, hq] = pr @ vf[: t + 1, hk]
    check("l3.attention_gqa_h/8", attn, dec(l3["attn_out"]))
    gatef = dec(l3["gate"])
    gated3 = (attn * (1.0 / (1.0 + np.exp(-gatef)))).reshape(T, 4096)
    check("l3.attn_out*sigmoid(gate)", gated3, dec(l3["gated"]))
    Wo3 = deq(sa.o_proj)
    check("l3.o_proj", gated3 @ Wo3.T, dec(l3["mixer_out"]))

    # --- Final norm and head (on p0 last position) ---
    wfin = np.array(tm.norm.weight.astype(mx.float32))
    check(
        "final_norm(x_hat*w)",
        rms(dec(h["layer39"]), wfin),
        dec(h["final_norm"]),
    )
    yf = dec(h["final_norm"])[-1]
    top_ids = topk["p0"]["top_ids"][-1]
    tl = topk["p0"]["top_logits"]
    top_logits = np.frombuffer(
        base64.b64decode(tl["f32b64"]), dtype="<f4"
    ).reshape(tl["shape"])[-1]
    mine = np.array([yf @ deq_row(lm.lm_head, i) for i in top_ids])
    d = float(np.max(np.abs(mine - top_logits)))
    s = float(np.max(np.abs(top_logits)))
    ok = d / s <= TOL
    RESULTS.append((ok, "lm_head_top20_p0_last", d / s))
    print(f"{'OK  ' if ok else 'FAIL'} lm_head_top20_p0_last rel={d / s:.3e}", flush=True)

    nfail = sum(1 for ok, _, _ in RESULTS if not ok)
    print(f"[check_formulas] {len(RESULTS)} checks, {nfail} failed, {time.time() - t0:.1f}s", flush=True)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
