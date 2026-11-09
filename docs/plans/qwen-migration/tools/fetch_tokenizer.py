# usage: python3 -I fetch_tokenizer.py <out_dir>
# Downloads the tokenizer files of the pinned revision into <out_dir> and
# prints the SHA-256 of every file. <out_dir> must be new and empty.
import hashlib
import json
import os
import sys
import urllib.request

REPO = "mlx-community/Qwen3.6-35B-A3B-4bit"
REV = "38740b847e4cb78f352aba30aa41c76e08e6eb46"
FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "generation_config.json"]


def url(name):
    return f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"


def main(out_dir):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        sys.exit(f"{out_dir} must be new and empty")
    os.makedirs(out_dir, exist_ok=True)
    for name in FILES:
        req = urllib.request.Request(url(name),
                                     headers={"User-Agent": "turbo-fieldfare-repack"})
        data = urllib.request.urlopen(req).read()
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"{name}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
    # generation_config.json must still parse, as a lightweight sanity check.
    json.load(open(os.path.join(out_dir, "generation_config.json"), encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1])
