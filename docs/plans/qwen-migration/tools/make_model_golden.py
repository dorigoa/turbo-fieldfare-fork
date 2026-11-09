# usage: scratch/qwen36-venv/bin/python -I docs/plans/qwen-migration/tools/make_model_golden.py <model_dir> <prompts.json> <out_dir>
# Produces the S8 model golden fixtures (C7, Appendix D) with mlx-lm.
import base64
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk


def enc(a, f16=False):
    a = np.asarray(a, dtype=np.float32)
    if f16:
        return {"shape": list(a.shape), "f16b64": base64.b64encode(a.astype("<f2").tobytes()).decode()}
    return {"shape": list(a.shape), "f32b64": base64.b64encode(a.astype("<f4").tobytes()).decode()}


def f32(a):
    """mlx array [1, T, ...] -> numpy float32 [T, ...] (first batch row)."""
    return np.array(a[0].astype(mx.float32))


def npy(a):
    return np.array(a.astype(mx.float32))


def full_logits(model, ids):
    out = model(mx.array([ids]))  # [1, T, V], teacher forcing without cache
    mx.eval(out)
    return np.array(out[0].astype(mx.float32))


def greedy(model, ids, n=32):
    ids = list(ids)
    new = []
    for _ in range(n):  # simple: recompute everything at every step
        nxt = int(np.argmax(full_logits(model, ids)[-1]))
        ids.append(nxt)
        new.append(nxt)
    return new


def relerr(recon, ref):
    d = np.max(np.abs(recon - ref))
    s = np.max(np.abs(ref))
    return float(d), float(d / (s + 1e-12))


def main():
    t_start = time.time()
    model, tok = load(sys.argv[1])
    prompts = json.load(open(sys.argv[2], encoding="utf-8"))
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    def encode_prompt(p):
        if p["kind"] == "text":
            return tok.encode(p["text"], add_special_tokens=False)
        messages = [{"role": "user", "content": p["user"]}]
        rendered = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=p["enable_thinking"],
            tokenize=False,
        )
        return tok.encode(rendered, add_special_tokens=False)

    ids_by_name = {}
    prompt_records = []
    for p in prompts:
        ids = encode_prompt(p)
        ids_by_name[p["name"]] = ids
        rec = dict(p)
        rec["ids"] = ids
        prompt_records.append(rec)
        print(f"[ids] {p['name']}: {len(ids)} tokens", flush=True)
    json.dump(prompt_records, open(f"{out_dir}/prompts.json", "w", encoding="utf-8"), ensure_ascii=False)

    # 2. logits_topk.json: top-20 ids and values per prompt and position.
    topk = {}
    logits_last = {}
    for p in prompts:
        name = p["name"]
        t0 = time.time()
        L = full_logits(model, ids_by_name[name])  # [T, V]
        part = np.argpartition(L, -20, axis=-1)[:, -20:]
        rows = np.arange(L.shape[0])[:, None]
        vals = L[rows, part]
        order = np.argsort(-vals, axis=-1)
        top_ids = part[rows, order]
        top_vals = vals[rows, order]
        topk[name] = {"top_ids": top_ids.tolist(), "top_logits": enc(top_vals)}
        print(f"[topk] {name}: T={L.shape[0]} in {time.time() - t0:.1f}s", flush=True)
        if name in ("p0", "p1"):
            logits_last[name] = {"logits": enc(L[-1], f16=True)}
    json.dump(topk, open(f"{out_dir}/logits_topk.json", "w", encoding="utf-8"))
    json.dump(logits_last, open(f"{out_dir}/logits_last.json", "w", encoding="utf-8"))

    # 3. greedy.json
    greedy_out = {}
    for p in prompts:
        name = p["name"]
        t0 = time.time()
        toks = greedy(model, ids_by_name[name], n=32)
        greedy_out[name] = toks
        print(f"[greedy] {name}: {toks} in {time.time() - t0:.1f}s", flush=True)
    json.dump(greedy_out, open(f"{out_dir}/greedy.json", "w", encoding="utf-8"))

    # 4. hidden_p0 + layer0/layer3 internals via a manual layer loop on p0.
    tm = model.language_model.model  # Qwen3_5TextModel
    layers = tm.layers
    ids0 = ids_by_name["p0"]
    T0 = len(ids0)
    T = min(8, T0)
    fa_mask = "causal" if T0 > 1 else None

    h = tm.embed_tokens(mx.array([ids0]))  # [1, T0, 2048]
    hidden = {"embedding": enc(f32(h))}
    internals0, internals3 = None, None

    for i, layer in enumerate(layers):
        mask = None if layer.is_linear else fa_mask
        if i == 0:
            internals0 = layer0_internals(layer, h, T)
        if i == 3:
            internals3 = layer3_internals(layer, h, T, fa_mask)
        h = layer(h, mask=mask, cache=None)
        if i in (0, 3, 39):
            hidden[f"layer{i}"] = enc(f32(h))
    hidden["final_norm"] = enc(f32(tm.norm(h)))
    json.dump(hidden, open(f"{out_dir}/hidden_p0.json", "w", encoding="utf-8"))

    internals0["state_heads0to3"] = internals0.pop("state")
    json.dump(internals0, open(f"{out_dir}/layer0_internals.json", "w", encoding="utf-8"))
    json.dump(internals3, open(f"{out_dir}/layer3_internals.json", "w", encoding="utf-8"))
    print(f"[done] total {time.time() - t_start:.1f}s", flush=True)


def layer0_internals(layer, hin, T):
    la = layer.linear_attn
    x = layer.input_layernorm(hin)
    qkv = la.in_proj_qkv(x)
    z = la.in_proj_z(x).reshape(1, -1, la.num_v_heads, la.head_v_dim)
    b = la.in_proj_b(x)
    a = la.in_proj_a(x)

    conv_state = mx.zeros((1, la.conv_kernel_size - 1, la.conv_dim), dtype=x.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = nn.silu(la.conv1d(conv_input))
    q, k, v = [
        t.reshape(1, -1, hh, dd)
        for t, hh, dd in zip(
            mx.split(conv_out, [la.key_dim, 2 * la.key_dim], -1),
            [la.num_k_heads, la.num_k_heads, la.num_v_heads],
            [la.head_k_dim, la.head_k_dim, la.head_v_dim],
        )
    ]
    qn, kn = normalize_qk(q, k, inv_scale=la.head_k_dim**-0.5, eps=1e-6)
    g_pre = -mx.exp(la.A_log.astype(mx.float32)) * nn.softplus(
        a.astype(mx.float32) + la.dt_bias.astype(mx.float32)
    )
    beta = mx.sigmoid(b)
    core_out, state = gated_delta_update(
        qn, kn, v, a, b, la.A_log, la.dt_bias, None, None, use_kernel=True
    )
    gated = la.norm(core_out, z)
    mixer = la.out_proj(gated.reshape(1, -1, la.value_dim))
    h2 = hin + mixer
    moe_x = layer.post_attention_layernorm(h2)

    moe = layer.mlp
    gates = mx.softmax(moe.gate(moe_x), axis=-1, precise=True)
    k8 = moe.top_k
    inds = mx.argpartition(gates, kth=-k8, axis=-1)[..., -k8:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    y = moe.switch_mlp(moe_x, inds)
    routed = (y * scores[..., None]).sum(axis=-2)
    shared_raw = moe.shared_expert(moe_x)
    sg = mx.sigmoid(moe.shared_expert_gate(moe_x))
    shared_out = sg * shared_raw
    layer_out = h2 + routed + shared_out

    ref = layer(hin, mask=None, cache=None)
    mx.eval(ref)
    d, r = relerr(npy(layer_out), npy(ref))
    print(f"[recon] layer0 max abs diff {d:.3e} rel {r:.3e}", flush=True)

    # MLX state is [B, Hv, Dv, Dk] (row=value dim); store [head][key i][value c].
    st = np.ascontiguousarray(npy(state)[0, :4].transpose(0, 2, 1))
    sl = lambda a: enc(f32(a[:, :T]))
    return {
        "x_norm": sl(x),
        "qkv_raw": sl(qkv),
        "z": sl(z.reshape(1, -1, la.value_dim)),
        "a": sl(a),
        "b": sl(b),
        "conv_out": sl(conv_out),
        "q": sl(qn),
        "k": sl(kn),
        "v": sl(v),
        "beta": sl(beta),
        "g": {"shape": [T, 32], "f32b64": base64.b64encode(npy(g_pre)[0, :T].astype("<f4")).decode()},
        "core_out": sl(core_out),
        "gated_out": sl(gated),
        "mixer_out": sl(mixer),
        "moe_x": sl(moe_x),
        "router_top_ids": npy(inds)[0, :T].astype(int).tolist(),
        "router_top_weights": sl(scores),
        "routed_out": sl(routed),
        "shared_out": sl(shared_out),
        "layer_out": sl(layer_out),
        "state": {"shape": [4, 128, 128], "f32b64": base64.b64encode(st.astype("<f4")).decode()},
    }


def layer3_internals(layer, hin, T, fa_mask):
    sa = layer.self_attn
    x = layer.input_layernorm(hin)
    qo = sa.q_proj(x).reshape(1, -1, sa.num_attention_heads, 2 * sa.head_dim)
    q, gate = mx.split(qo, 2, axis=-1)  # [1,T,16,256] each
    keys = sa.k_proj(x).reshape(1, -1, sa.num_key_value_heads, sa.head_dim)
    values = sa.v_proj(x).reshape(1, -1, sa.num_key_value_heads, sa.head_dim)

    q_normed = sa.q_norm(q)
    k_normed = sa.k_norm(keys)
    q_rope = sa.rope(q_normed.transpose(0, 2, 1, 3))
    k_rope = sa.rope(k_normed.transpose(0, 2, 1, 3))
    attn = scaled_dot_product_attention(
        q_rope, k_rope, values.transpose(0, 2, 1, 3),
        cache=None, scale=sa.scale, mask=fa_mask,
    ).transpose(0, 2, 1, 3)  # [1,T,16,256]
    gated_full = attn * mx.sigmoid(gate)
    gated_flat = gated_full.reshape(1, -1, sa.num_attention_heads * sa.head_dim)
    mixer = sa.o_proj(gated_flat)
    h2 = hin + mixer
    layer_out = h2 + layer.mlp(layer.post_attention_layernorm(h2))

    ref = layer(hin, mask=fa_mask, cache=None)
    mx.eval(ref)
    d, r = relerr(npy(layer_out), npy(ref))
    print(f"[recon] layer3 max abs diff {d:.3e} rel {r:.3e}", flush=True)

    sl = lambda a: enc(f32(a[:, :T]))
    return {
        "x_norm": sl(x),
        "q_pre_norm": sl(q_normed),
        "gate": sl(gate),
        "k_pre_norm": sl(k_normed),
        "v": sl(values),
        "q_rope": enc(np.ascontiguousarray(npy(q_rope).transpose(0, 2, 1, 3)[0, :T])),
        "k_rope": enc(np.ascontiguousarray(npy(k_rope).transpose(0, 2, 1, 3)[0, :T])),
        "attn_out": sl(attn),
        "gated": sl(gated_flat),
        "mixer_out": sl(mixer),
        "layer_out": sl(layer_out),
    }


if __name__ == "__main__":
    main()
