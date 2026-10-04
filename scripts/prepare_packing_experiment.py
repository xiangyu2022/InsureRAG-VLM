"""Prepare dev or exposed-test packing diagnostics. Gold never enters packing."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.context_packing import normalized, pack_evidence
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.local_answer_metrics import prompt_ids
from src.insurerag_vlm.query_understanding import understand_query
from src.insurerag_vlm.retrieval_metrics import score_ranking

ROOT = Path("reports/packing_20261004")
OLD = Path("reports/local_20261004")


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    args = parser.parse_args()
    if args.split == "test" and not (ROOT / "frozen_protocol.json").exists():
        raise RuntimeError("Freeze the packing change before preparing the exposed-test regression")
    if args.split == "test":
        plan = json.loads((ROOT / "frozen_protocol.json").read_text())
        assert hashlib.sha256((OLD / "test.jsonl").read_bytes()).hexdigest() == plan["test_input_sha256"]
        for name, expected in plan["source_hashes"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3.5-2B", local_files_only=True)
    def count(context, question):
        return len(prompt_ids(tokenizer, {"question": question, "evidence": context,
                                         "source": "", "input_source": "retrieved"}))
    config = ModelConfig(retrieval_model="local-hashing", vlm_model="local-extractive", use_hf_api=False,
                         corpus_source="curated", curated_dataset_dir=Path("data/04_curated"),
                         enable_image_signal=False, index_dir=OLD / "retrieval_index")
    pipeline = DocumentRetrievalPipeline(config)
    originals = read(OLD / f"{args.split}.jsonl")
    outputs = {"legacy": [], "balanced": []}
    audits = []
    old_test = {r["id"]: r for r in read(OLD / "retrieved_context_test.jsonl")} if args.split == "test" else {}
    for row in originals:
        if not row["answerable"]:
            continue
        question = f"In the document {row['doc_id']}, {row['question']}"
        ranked = pipeline.rank_pages(question, config.curated_dataset_dir, top_k=5)
        legacy = pipeline.pack_long_context_legacy(ranked, 3)
        # The packer only receives query-derived metadata and retrieved pages.
        packed = pipeline.pack_context_with_audit(ranked, 3, understand_query(question),
                                                  lambda context: count(context, question), 2048, question=question)
        page_by_source = {page["source"]: page for page in ranked}
        assert packed["sources"][0] == ranked[0]["source"]
        for segment in packed["segments"]:
            page = page_by_source[segment["source"]]
            if segment["kind"] == "text":
                raw = [page.get("text_snippet", ""), *(page.get("snippet_support") or [])]
                assert any(segment["text"] in normalized(text) for text in raw), "Evidence/source mapping changed"
        if old_test:
            assert [p["source"] for p in ranked] == old_test[row["id"]]["retrieved_sources"]
            assert legacy == old_test[row["id"]]["evidence"], "Legacy control changed"
        # Gold-based measurements begin only AFTER both contexts are frozen.
        gold = {"question": question, "evidence_sources": [row["source"]]}
        matches = [p for p in ranked if score_ranking(gold, [p])["hit_at_5"]]
        expected = matches[0]["source"] if matches else row["source"]
        for name, context in (("legacy", legacy), ("balanced", packed["context"])):
            prompt_tokens = count(context, question)
            assert prompt_tokens <= 2048
            sources = re.findall(r"^SOURCE:\s*(.+)$", context, re.M)
            gold_present = any(score_ranking(gold, [{"source": source}])["hit_at_5"] for source in sources)
            reference_present = norm(row["reference_content"]) in norm(context)
            outputs[name].append({**row, "question": question, "evidence": context, "source": expected,
                                  "original_source": row["source"], "input_source": "retrieved",
                                  "original_evidence": row["evidence"], "prompt_tokens": prompt_tokens,
                                  "packed_sources": sources, "gold_in_top5": bool(matches),
                                  "gold_source_in_context": gold_present,
                                  "reference_span_in_context": reference_present,
                                  "evaluation_scope": "dev" if args.split == "dev" else "exposed_test_exploratory_regression"})
        audits.append({"id": row["id"], "ranked_sources": [p["source"] for p in ranked], **packed})
    summary = {}
    for name, rows in outputs.items():
        path = ROOT / f"{args.split}_{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
        summary[name] = {"rows": len(rows), "gold_in_top5": sum(r["gold_in_top5"] for r in rows),
                         "gold_source_in_context": sum(r["gold_source_in_context"] for r in rows),
                         "reference_span_in_context": sum(r["reference_span_in_context"] for r in rows),
                         "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (ROOT / f"{args.split}_packing_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    # Actual tokenizer stress test: byte-heavy Unicode plus long intact source IDs.
    fixtures = [{"source": f"packet/{i}/policy.pdf#page=17", "text_snippet": "保障条款αβ🙂 "*300} for i in range(3)]
    fixture = pack_evidence(fixtures, 3, prompt_token_counter=lambda ctx: count(ctx, "Explain this evidence."),
                            max_prompt_tokens=400)
    assert fixture["prompt_tokens"] <= 400 and len(fixture["sources"]) == 3
    summary["actual_tokenizer_fixture"] = {"prompt_tokens": fixture["prompt_tokens"], "sources": fixture["sources"]}
    (ROOT / f"{args.split}_packing_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
