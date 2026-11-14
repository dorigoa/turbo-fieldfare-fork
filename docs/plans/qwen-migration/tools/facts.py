# usage: python3 -I facts.py <facts_dir>
# facts_dir must contain headers.json (Appendix A), config.json,
# generation_config.json and index.json from the pinned revision.
# Verifies plan sections 3.1-3.4 against the real checkpoint metadata.
import json, sys

failures = 0

def check(name, ok, detail=""):
    global failures
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failures += 1

def main(facts_dir):
    headers = json.load(open(f"{facts_dir}/headers.json"))
    config = json.load(open(f"{facts_dir}/config.json"))
    gen = json.load(open(f"{facts_dir}/generation_config.json"))
    index = json.load(open(f"{facts_dir}/index.json"))
    t = config["text_config"]

    # 3.2 values straight from config.json / generation_config.json.
    check("3.2 layers", t["num_hidden_layers"] == 40 and t["full_attention_interval"] == 4)
    check("3.2 hidden_size", t["hidden_size"] == 2048)
    check("3.2 DeltaNet", (t["linear_num_key_heads"], t["linear_key_head_dim"],
          t["linear_num_value_heads"], t["linear_value_head_dim"],
          t["linear_conv_kernel_dim"], t["mamba_ssm_dtype"]) == (16, 128, 32, 128, 4, "float32"))
    check("3.2 attention", (t["num_attention_heads"], t["num_key_value_heads"],
          t["head_dim"], t["attn_output_gate"], t["num_hidden_layers"]) == (16, 2, 256, True, 40))
    rp = t["rope_parameters"]
    check("3.2 rope", rp["rope_theta"] == 10000000 and rp["partial_rotary_factor"] == 0.25
          and rp["mrope_section"] == [11, 11, 10])
    check("3.2 MoE", (t["num_experts"], t["num_experts_per_tok"],
          t["moe_intermediate_size"], t["hidden_act"]) == (256, 8, 512, "silu"))
    check("3.2 shared expert", t["shared_expert_intermediate_size"] == 512)
    check("3.2 vocab", t["vocab_size"] == 248320)
    check("3.2 untied embeddings", t["tie_word_embeddings"] is False)
    check("3.2 rms_norm_eps", t["rms_norm_eps"] == 1e-6)
    check("3.2 max_position_embeddings", t["max_position_embeddings"] == 262144)
    check("3.2 tokens", gen["bos_token_id"] == 248044 and gen["pad_token_id"] == 248044
          and sorted(gen["eos_token_id"]) == [248044, 248046])
    check("3.2 sampling", gen["temperature"] == 1.0 and gen["top_k"] == 20 and gen["top_p"] == 0.95)

    # 3.1 index and shard facts.
    shards = sorted(set(index["weight_map"].values()))
    check("3.1 shards", len(index["weight_map"]) == len(headers)
          and shards == [f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)])
    check("3.1 total_size", index["metadata"]["total_size"] == 20401929952,
          str(index["metadata"]["total_size"]))
    prefixes = {}
    for key in headers:
        prefixes[key.split(".")[0]] = prefixes.get(key.split(".")[0], 0) + 1
    check("3.1 tensor counts", prefixes == {"language_model": 1757, "vision_tower": 333},
          str(prefixes))
    check("3.1 no mtp tensors", not [k for k in headers if ".mtp." in k or k.startswith("mtp.")])

    # 3.3 expected language_model names for all 40 layers (attention mask: N % 4 == 3).
    expected = {"language_model.model.embed_tokens.weight", "language_model.model.embed_tokens.scales",
                "language_model.model.embed_tokens.biases", "language_model.lm_head.weight",
                "language_model.lm_head.scales", "language_model.lm_head.biases",
                "language_model.model.norm.weight"}
    linear = ([f"model.layers.N.linear_attn.in_proj_{p}.{s}"
               for p in ("qkv", "z", "a", "b") for s in ("weight", "scales", "biases")]
              + ["model.layers.N.linear_attn.conv1d.weight",
                 "model.layers.N.linear_attn.A_log",
                 "model.layers.N.linear_attn.dt_bias",
                 "model.layers.N.linear_attn.norm.weight"]
              + [f"model.layers.N.linear_attn.out_proj.{s}"
                 for s in ("weight", "scales", "biases")])
    attn = ([f"model.layers.N.self_attn.{p}.{s}"
             for p in ("q_proj", "k_proj", "v_proj", "o_proj") for s in ("weight", "scales", "biases")]
            + ["model.layers.N.self_attn.q_norm.weight",
               "model.layers.N.self_attn.k_norm.weight"])
    moe = ([f"model.layers.N.mlp.gate.{s}" for s in ("weight", "scales", "biases")]
           + [f"model.layers.N.mlp.switch_mlp.{p}.{s}"
              for p in ("gate_proj", "up_proj", "down_proj") for s in ("weight", "scales", "biases")]
           + [f"model.layers.N.mlp.shared_expert.{p}.{s}"
              for p in ("gate_proj", "up_proj", "down_proj") for s in ("weight", "scales", "biases")]
           + [f"model.layers.N.mlp.shared_expert_gate.{s}"
              for s in ("weight", "scales", "biases")])
    norms_layer = ["model.layers.N.input_layernorm.weight",
                   "model.layers.N.post_attention_layernorm.weight"]
    for n in range(40):
        group = norms_layer + moe + (attn if n % 4 == 3 else linear)
        expected |= {("language_model." + x).replace(".N.", f".{n}.") for x in group}
    lm = {k for k in headers if k.startswith("language_model.")}
    check("3.3 names all 40 layers", lm == expected,
          f"missing={sorted(expected - lm)[:6]} extra={sorted(lm - expected)[:6]}")

    # 3.3/3.4 shapes and dtypes. Quantized weight is U32 with last dimension
    # columns * bits / 32; scales and biases are BF16 with columns / 64 (group 64).
    def quant_map(n):
        m = {"model.layers.N.mlp.gate": (256, 2048, 8),
             "model.layers.N.mlp.shared_expert_gate": (1, 2048, 8)}
        if n % 4 != 3:
            m |= {"model.layers.N.linear_attn.in_proj_qkv": (8192, 2048, 4),
                  "model.layers.N.linear_attn.in_proj_z": (4096, 2048, 4),
                  "model.layers.N.linear_attn.in_proj_a": (32, 2048, 4),
                  "model.layers.N.linear_attn.in_proj_b": (32, 2048, 4),
                  "model.layers.N.linear_attn.out_proj": (2048, 4096, 4)}
        if n % 4 == 3:
            m |= {"model.layers.N.self_attn.q_proj": (8192, 2048, 4),
                  "model.layers.N.self_attn.k_proj": (512, 2048, 4),
                  "model.layers.N.self_attn.v_proj": (512, 2048, 4),
                  "model.layers.N.self_attn.o_proj": (2048, 4096, 4)}
        for p in ("switch_mlp", "shared_expert"):
            pre = f"model.layers.N.mlp.{p}"
            if p == "switch_mlp":
                m |= {f"{pre}.gate_proj": (256, 512, 2048, 4), f"{pre}.up_proj": (256, 512, 2048, 4),
                      f"{pre}.down_proj": (256, 2048, 512, 4)}
            else:
                m |= {f"{pre}.gate_proj": (512, 2048, 4), f"{pre}.up_proj": (512, 2048, 4),
                      f"{pre}.down_proj": (2048, 512, 4)}
        return {k.replace(".N.", f".{n}."): v for k, v in m.items()}

    shapes = {"model.embed_tokens": (248320, 2048, 4),
              "lm_head": (248320, 2048, 4)}
    for n in range(40):
        shapes |= quant_map(n)
    bad = []
    for name, spec in shapes.items():
        dims, columns, bits = spec[:-2], spec[-2], spec[-1]
        prefix = f"language_model.{name}."
        w = headers[prefix + "weight"]
        want_w = list(dims) + [columns * bits // 32]
        want_sb = list(dims) + [columns // 64]
        if w["dtype"] != "U32" or w["shape"] != want_w:
            bad.append(f"{prefix}weight {w['dtype']} {w['shape']} != U32 {want_w}")
        for s in ("scales", "biases"):
            v = headers[prefix + s]
            if v["dtype"] != "BF16" or v["shape"] != want_sb:
                bad.append(f"{prefix}{s} {v['dtype']} {v['shape']} != BF16 {want_sb}")
    check("3.3/3.4 quantized shapes, dtypes, group 64 and 8-bit gates", not bad, "; ".join(bad[:5]))

    norms = {"language_model.model.norm.weight": 2048,
             "language_model.model.layers.0.input_layernorm.weight": 2048,
             "language_model.model.layers.0.post_attention_layernorm.weight": 2048,
             "language_model.model.layers.3.self_attn.q_norm.weight": 256,
             "language_model.model.layers.3.self_attn.k_norm.weight": 256,
             "language_model.model.layers.0.linear_attn.norm.weight": 128}
    check("3.3 norms shape/dtype", all(
        headers[k]["dtype"] == "BF16" and headers[k]["shape"] == [d] for k, d in norms.items()))
    free = ["language_model.model.layers.0.linear_attn.conv1d.weight",
            "language_model.model.layers.0.linear_attn.A_log",
            "language_model.model.layers.0.linear_attn.dt_bias"]
    for k in free:
        v = headers[k]
        print(f"[info] {k.split('language_model.')[1]}: dtype={v['dtype']} shape={v['shape']}")
    check("3.1 conv1d/A_log/dt_bias not quantized", all(
        headers[k]["dtype"] == "BF16" for k in free))

    # 3.4 bytes per expert and expert totals (all three switch_mlp tensors together).
    per_proj = [headers[f"language_model.model.layers.{n}.mlp.switch_mlp.{p}.{s}"]["bytes"]
                for n in range(40) for p in ("gate_proj", "up_proj", "down_proj")
                for s in ("weight", "scales", "biases")]
    per_expert = sum(per_proj) / (256 * 40)
    check("3.4 bytes per expert", per_expert == 1769472 and per_expert / 3 == 589824,
          f"per_expert={per_expert}")
    total = sum(per_proj)
    check("3.4 expert total bytes", total == 18119393280, str(total))

    print(f"{'OK' if failures == 0 else str(failures) + ' FAILURES'}")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
