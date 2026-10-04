"""Freeze exploratory regression rules after dev and before exposed-test rerun."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("reports/packing_20261004")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = ROOT / "frozen_protocol.json"
    if path.exists():
        raise RuntimeError("Protocol already frozen; do not overwrite it")
    dev = {}
    for model in ("base", "adapter"):
        for packing in ("legacy", "balanced"):
            name = f"{model}_dev_{packing}_metrics.json"
            dev[name] = json.loads((ROOT / name).read_text())
    tests = (ROOT / "unit_tests.log").read_text()
    assert "Ran 53 tests" in tests and "\nOK" in tests
    sources = ["src/insurerag_vlm/context_packing.py", "src/insurerag_vlm/hybrid_pipeline.py",
               "src/insurerag_vlm/local_answer_metrics.py", "scripts/prepare_packing_experiment.py",
               "scripts/run_packing_evaluation.py", "scripts/run_local_lora_experiment.py"]
    plan = {"utc": datetime.now(timezone.utc).isoformat(),
            "scope": "EXPLORATORY regression on previously observed test rows; no new independent holdout",
            "decision": "Freeze rank-preserving, per-source bounded packing with query-only fragment selection after dev engineering checks. Do not change model checkpoints or tune on exposed test.",
            "source_hashes": {p: sha(p) for p in sources}, "dev_metrics_at_freeze": dev,
            "test_input_sha256": sha("reports/local_20261004/test.jsonl"),
            "dev_context_hashes": {p: sha(ROOT / p) for p in ("dev_legacy.jsonl", "dev_balanced.jsonl")},
            "checkpoint_sha256": sha("models/local_20261004_lora/step-300/adapter_model.safetensors"),
            "config": {"retrieval_top_k": 5, "answer_pages": 3, "max_context_chars": 3200,
                       "max_page_chars": 900, "max_prompt_tokens": 2048, "max_new_tokens": 128,
                       "decode": "greedy, thinking disabled", "model_selection": "none", "training": "none"},
            "query_scope": "Same document-scoped queries as the first experiment; no gold page/answer/span supplied to packing.",
            "measurements": ["source retention and exact reference-span retention (proxies)",
                "content F1 on all original 62 positives, no topic-based filtering", "exact gold-source citation",
                "citation resolves to packed source", "lexical support proxy", "no negative context labels inferred from reference absence"],
            "not_measured": ["semantic entailment", "human correctness", "context-level abstention precision/recall"],
            "holdout_audit": json.loads((ROOT / "holdout_feasibility.json").read_text())}
    path.write_text(json.dumps(plan, indent=2))
    print(f"Frozen {path} at {plan['utc']}")


if __name__ == "__main__":
    main()
