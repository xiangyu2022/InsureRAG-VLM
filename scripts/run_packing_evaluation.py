"""Paired frozen-model diagnostics; never trains or selects checkpoints."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from scripts import run_local_lora_experiment as runner
from src.insurerag_vlm.local_answer_metrics import prompt_ids

ROOT = Path("reports/packing_20261004")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model-kind", choices=["base", "adapter"], required=True)
    args = parser.parse_args()
    if args.split == "test":
        plan = json.loads((ROOT / "frozen_protocol.json").read_text())
        for name, expected in plan["source_hashes"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    runner.ROOT = ROOT
    runner.seed_all()
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3.5-2B", local_files_only=True)
    adapter = "models/local_20261004_lora/step-300" if args.model_kind == "adapter" else None
    model = runner.load_model("models/Qwen3.5-2B", adapter)
    limit = model.config.text_config.max_position_embeddings
    invocation = {"utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                  "model": "Qwen/Qwen3.5-2B", "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                  "adapter": adapter, "max_prompt_tokens": 2048, "max_new_tokens": 128,
                  "no_tokenizer_truncation": True, "dataset_hashes": {}}
    for packing in ("legacy", "balanced"):
        path = ROOT / f"{args.split}_{packing}.jsonl"
        rows = runner.read(path)
        invocation["dataset_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in rows:
            actual = len(prompt_ids(tokenizer, row))
            assert actual == row["prompt_tokens"] <= 2048
            assert actual + 128 <= limit
        (ROOT / f"{args.model_kind}_{args.split}_invocation.json").write_text(json.dumps(invocation, indent=2))
        runner.evaluate(model, tokenizer, rows, f"{args.model_kind}_{args.split}_{packing}", 128)


if __name__ == "__main__":
    main()
