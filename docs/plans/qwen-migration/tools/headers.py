# usage: python3 -I headers.py <out_dir>
import json, struct, sys, urllib.request, os

REPO = "mlx-community/Qwen3.6-35B-A3B-4bit"
REV = "38740b847e4cb78f352aba30aa41c76e08e6eb46"
SHARDS = 4

def url(name):
    return f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"

def fetch(name, start, end):
    req = urllib.request.Request(url(name), headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req).read()

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for i in range(1, SHARDS + 1):
        name = f"model-{i:05d}-of-{SHARDS:05d}.safetensors"
        header_len = struct.unpack("<Q", fetch(name, 0, 7))[0]
        header = json.loads(fetch(name, 8, 8 + header_len - 1))
        header.pop("__metadata__", None)
        for key, value in header.items():
            start, end = value["data_offsets"]
            tensors[key] = {"shard": name, "dtype": value["dtype"], "shape": value["shape"],
                            "start": 8 + header_len + start, "bytes": end - start}
    with open(os.path.join(out_dir, "headers.json"), "w") as f:
        json.dump(tensors, f)
    print(len(tensors))

if __name__ == "__main__":
    main(sys.argv[1])
