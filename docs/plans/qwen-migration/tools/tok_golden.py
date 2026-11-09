# usage: scratch/qwen36-venv/bin/python -I tok_golden.py <tokenizer_dir> <cases.json> <out.json>
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{sys.argv[1]}/tokenizer.json")
out = []
for text in json.load(open(sys.argv[2], encoding="utf-8")):
    ids = tok.encode(text, add_special_tokens=False).ids
    out.append({"text": text, "ids": ids,
                "decoded": tok.decode(ids, skip_special_tokens=False)})
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out))
