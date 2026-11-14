# usage: scratch/qwen36-venv/bin/python -I chat_golden.py <tokenizer_dir> <cases.json> <out.json>
# cases.json: list of {"name", "messages", "tools"?, "enable_thinking"?, "add_generation_prompt"}
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
out = []
for case in json.load(open(sys.argv[2], encoding="utf-8")):
    kwargs = {"tokenize": False,
              "add_generation_prompt": case.get("add_generation_prompt", True)}
    if "tools" in case:
        kwargs["tools"] = case["tools"]
    if "enable_thinking" in case:
        kwargs["enable_thinking"] = case["enable_thinking"]
    text = tok.apply_chat_template(case["messages"], **kwargs)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out.append(dict(case, rendered=text, ids=ids))
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out))
