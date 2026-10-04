"""Re-score saved outputs without inference; verify paired evaluation integrity."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.insurerag_vlm.local_answer_metrics import score, aggregate
from src.insurerag_vlm.retrieval_metrics import score_ranking

ROOT = Path("reports/local_20261004")


def read(name):
    return [json.loads(x) for x in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def main():
    checks = {}
    for split in ("dev", "test", "retrieved_context_test"):
        source = {r["id"]: r for r in read(f"{split}.jsonl")}
        for model in ("base", "adapter"):
            name = f"{model}_{split}"
            path = ROOT / f"{name}_metrics.json"
            if not path.exists():
                continue
            saved = read(f"{name}_predictions.jsonl")
            assert len(saved) == len(source)
            assert {r["id"] for r in saved} == source.keys()
            for row in saved:
                expected = score(source[row["id"]], row["prediction"])
                for key, value in expected.items():
                    assert row[key] == value, (name, row["id"], key)
            assert aggregate(saved) == json.loads(path.read_text())
            checks[name] = {"saved_rows": len(saved), "rescore_matches": True,
                            "at_output_token_cap": sum(r["generated_tokens"] >= 128 for r in saved)}
    context_rows = read("retrieved_context_test.jsonl")
    context_checks = []
    for row in context_rows:
        sources = re.findall(r"^SOURCE:\s*(.+)$", row["evidence"], re.M)
        gold = {"question": row["question"], "evidence_sources": [row["source"]]}
        present = any(score_ranking(gold, [{"source": source}])["hit_at_5"] for source in sources)
        context_checks.append({"id": row["id"], "gold_in_top5": row["gold_in_top5"],
                               "gold_source_in_packed_context": present,
                               "packed_sources": sources})
    result = {"status": "passed", "checks": checks,
              "retrieved_context": {"rows": len(context_checks),
                  "gold_in_top5": sum(r["gold_in_top5"] for r in context_checks),
                  "gold_source_in_packed_context": sum(r["gold_source_in_packed_context"] for r in context_checks),
                  "note": "Source presence alone does not prove the reference span survives truncation."},
              "note": "Confirms saved scores match final pure metric implementation; does not certify semantic label correctness."}
    (ROOT / "retrieved_context_source_audit.json").write_text(json.dumps(context_checks, indent=2))
    (ROOT / "prediction_integrity_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
