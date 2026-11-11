"""Download only the newly authorized CPU embedding model, pinned to an immutable revision."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-Embedding-0.6B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
    args = parser.parse_args()
    target = ROOT / "models/Qwen3-Embedding-0.6B"
    manifest = target / "agent-manifest.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        for name, digest in record["files"].items():
            if hashlib.sha256((target / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Existing embedding snapshot changed: " + name)
        print(json.dumps(record), flush=True)
        return
    if args.source == "huggingface":
        revision = HfApi().model_info(MODEL, timeout=20).sha
        snapshot_download(
            MODEL,
            revision=revision,
            local_dir=target,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
    else:
        # Official publisher mirror; resolve once, then pin every request and verify source hashes.
        endpoint = "https://modelscope.cn/api/v1/models/" + MODEL + "/repo"
        data = json.load(urllib.request.urlopen(endpoint + "/files?Revision=master", timeout=30))[
            "Data"
        ]
        revision = max(data["Files"], key=lambda f: f["CommittedDate"])["Revision"]
        pinned = json.load(
            urllib.request.urlopen(
                endpoint + "/files?" + urllib.parse.urlencode({"Revision": revision}), timeout=30
            )
        )["Data"]["Files"]
        target.mkdir(parents=True, exist_ok=True)
        for item in pinned:
            name = item["Path"]
            if item["Type"] != "blob" or Path(name).suffix not in {".json", ".txt", ".safetensors"}:
                continue
            if Path(name).name != name:
                raise ValueError("Unexpected model file path")
            destination = target / name
            if not destination.exists():
                url = (
                    endpoint
                    + "?"
                    + urllib.parse.urlencode({"Revision": revision, "FilePath": name})
                )
                temporary = destination.with_suffix(destination.suffix + ".partial")
                with (
                    urllib.request.urlopen(url, timeout=90) as response,
                    temporary.open("wb") as out,
                ):
                    while chunk := response.read(2**20):
                        out.write(chunk)
                temporary.replace(destination)
            if (
                destination.stat().st_size != item["Size"]
                or hashlib.sha256(destination.read_bytes()).hexdigest() != item["Sha256"]
            ):
                raise RuntimeError("Publisher checksum mismatch: " + name)
            print("VERIFIED", name, item["Size"], flush=True)
    files = {
        str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file() and ".cache" not in p.parts
    }
    record = {"model": MODEL, "revision": revision, "source": args.source, "files": files}
    manifest.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
