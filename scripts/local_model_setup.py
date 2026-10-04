"""Download official public weights to the experiment workspace; no credentials required."""
import argparse
import json
import platform
import subprocess
from pathlib import Path

import psutil
import torch
import transformers
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    root = Path("reports/local_20261004")
    root.mkdir(parents=True, exist_ok=True)
    hardware = {"platform": platform.platform(), "python": platform.python_version(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                "ram_total": psutil.virtual_memory().total, "ram_available": psutil.virtual_memory().available,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_free_total": torch.cuda.mem_get_info() if torch.cuda.is_available() else None}
    (root / "hardware.json").write_text(json.dumps(hardware, indent=2))
    print(json.dumps(hardware), flush=True)
    api = HfApi(token=False)
    info = api.model_info(args.model, revision=args.revision)
    destination = Path("models") / args.model.split("/")[-1]
    meta = {"model_id": args.model, "revision": info.sha, "source": f"https://huggingface.co/{args.model}",
            "license": info.card_data.get("license") if info.card_data else None,
            "local_dir": str(destination), "download_status": "started"}
    meta_path = root / (args.model.split("/")[-1] + "_source.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    snapshot_download(args.model, revision=info.sha, local_dir=destination, token=False,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE*"],
                      max_workers=3)
    meta["download_status"] = "complete"
    meta_path.write_text(json.dumps(meta, indent=2))
    (root / "requirements-lock.txt").write_text(subprocess.check_output(
        [__import__("sys").executable, "-m", "pip", "freeze"], text=True))
    print("MODEL_DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
