"""Windows-compatible public-PDF smoke; original corpus is read-only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.data import load_documents
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.preprocess import PageImagePreprocessConfig, preprocess_page_images
from src.insurerag_vlm.qa import generate_policy_qa_pairs, compute_retrieval_metrics
from src.insurerag_vlm.retrieval_manifest import scope_and_merge
from src.insurerag_vlm.retrieval_metrics import score_ranking
from src.insurerag_vlm.visual import build_visual_index, compute_visual_retrieval_metrics, visual_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    root = Path("reports/local_20261004/artifacts/public_pdf_smoke")
    root.mkdir(parents=True, exist_ok=True)
    docs = load_documents(args.corpus)
    pdfs = sorted(args.corpus.rglob("*.pdf"))
    preprocessing = preprocess_page_images(PageImagePreprocessConfig(args.corpus, root / "prep", render_dpi=50))
    manifest = [json.loads(x) for x in preprocessing.page_manifest_path.read_text(encoding="utf-8").splitlines()]
    # All generated visual sources must identify a real page in the text path.
    text_sources = {d.metadata["source"] for d in docs}
    assert all(p["source"] in text_sources for p in manifest)
    qa = generate_policy_qa_pairs(args.corpus, root / "qa", target_count=20, unsupported_count=10)
    raw = [json.loads(x) for x in qa.qa_path.read_text(encoding="utf-8").splitlines()]
    examples = scope_and_merge([r for r in raw if r.get("answerable", True)])
    eval_path = root / "scoped_qa.jsonl"
    eval_path.write_text("".join(json.dumps(r) + "\n" for r in examples), encoding="utf-8")
    config = ModelConfig(index_dir=root / "text", corpus_source="documents", enable_image_signal=False,
                         retrieval_model="local-hashing", vlm_model="local-extractive", use_hf_api=False)
    pipeline = DocumentRetrievalPipeline(config)
    pipeline.build_index(args.corpus)
    metrics = {"text": vars(compute_retrieval_metrics(pipeline, args.corpus, eval_path))}
    for backend in ("visual_stub", "local_image"):
        index = root / backend
        build_visual_index(preprocessing.page_manifest_path, index, backend)
        metrics[backend] = compute_visual_retrieval_metrics(eval_path, index, backend)
        for row in examples:
            assert visual_search(row["question"], index, backend, 5) == visual_search(row["question"], index, backend, 10)[:5]
    first = manifest[0]
    identity_score = score_ranking({"question": "identity probe", "evidence_sources": [first["source"]]}, [first])
    assert identity_score["recall_at_5"] == 1
    prediction = pipeline.query("What coverage limits or deductibles are described in the documents?", args.corpus)
    report = {"status": "passed", "pdf_count": len(pdfs), "page_count": len(manifest),
              "source_identity_check": "all_visual_pages_match_text_sources", "prefix_check": "passed",
              "identity_probe": identity_score, "metrics": metrics, "query_output": prediction,
              "scope": "functional smoke, rule labels, public source corpus; not held-out adaptation evaluation",
              "pdf_hashes": {p.relative_to(args.corpus).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in pdfs}}
    Path("reports/local_20261004/public_pdf_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "pdf_count", "page_count", "source_identity_check", "prefix_check")}))


if __name__ == "__main__":
    main()
