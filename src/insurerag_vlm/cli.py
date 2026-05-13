import argparse
import json
from pathlib import Path
from typing import List

from .ablation import run_ablation
from .benchmark import RunGpuBenchmarkConfig, run_gpu_benchmark
from .app import run_demo_server
from .calibration import run_calibration
from .config import ModelConfig
from .diff import compare_clause_diff, render_clause_diff, summarize_clause_diff, write_policy_diff
from .dense_training import DEFAULT_DENSE_BASE_MODEL, DenseRetrieverTrainConfig, run_dense_retriever_training
from .evaluation import generate_evaluation_examples
from .pdf import extract_layout_by_page, extract_text_by_page, render_pdf_pages
from .pipeline import DocumentRetrievalPipeline
from .preprocess import PageImagePreprocessConfig, preprocess_page_images
from .qa import (
    compute_retrieval_metrics,
    download_public_datasets,
    generate_policy_qa_pairs,
    import_acord_qa,
    import_cuad_qa,
    merge_qa_files,
)
from .sft import DEFAULT_QWEN_7B_MODEL, QwenLoraSFTConfig, run_lora_sft, run_lora_smoke_test
from .training_data import TrainingCorpusBuildConfig, build_training_corpora
from .validation import CuratedValidationConfig, validate_curated_data
from .visual import SUPPORTED_VISUAL_BACKENDS, build_visual_index, compute_visual_retrieval_metrics, visual_search


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _load_source_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        pages = extract_text_by_page(path)
        return "\n\n---\n\n".join(pages)
    return _read_text_file(path)


def _add_retrieval_mode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default=ModelConfig().retrieval_mode,
        choices=["hybrid_multimodal", "hybrid_text", "dense_only", "sparse_only", "visual"],
    )
    parser.add_argument(
        "--corpus-source",
        type=str,
        default=ModelConfig().corpus_source,
        choices=["auto", "curated", "documents"],
    )
    parser.add_argument(
        "--disable-image-signal",
        action="store_true",
        help="Disable the lightweight page-image auxiliary signal and run text-only retrieval.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="InsureRAG-VLM retrieval and inference pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index", help="Build a page index from a data folder")
    build_parser.add_argument("data_folder", type=Path, help="Path to the folder containing text, image, or PDF files")
    build_parser.add_argument("--retrieval-model", default=ModelConfig().retrieval_model)
    build_parser.add_argument("--index-dir", type=Path, default=ModelConfig().index_dir)
    build_parser.add_argument("--render-pdf-pages", action="store_true", help="Render PDF pages as images for page-image indexing")
    build_parser.add_argument("--pdf-render-dir", type=Path, default=ModelConfig().pdf_render_dir)
    _add_retrieval_mode_args(build_parser)

    preprocess_parser = subparsers.add_parser(
        "preprocess-pages",
        help="Render PDFs into page images and create ColQwen2-compatible page manifests",
    )
    preprocess_parser.add_argument("input_dir", type=Path, help="Folder containing insurance policy PDFs")
    preprocess_parser.add_argument("--output-root", type=Path, default=Path("data"), help="Root for page images, processed metadata, and index manifests")
    preprocess_parser.add_argument("--render-dpi", type=int, default=200, help="DPI for page-image rendering")
    preprocess_parser.add_argument("--run-ocr", action="store_true", help="Run lightweight OCR as auxiliary metadata, not as the primary retrieval signal")
    preprocess_parser.add_argument("--source-type", type=str, default="public_policy_pdf", help="Source label stored in page manifests")
    preprocess_parser.add_argument("--train-ratio", type=float, default=0.8)
    preprocess_parser.add_argument("--valid-ratio", type=float, default=0.1)
    preprocess_parser.add_argument("--seed", type=int, default=42)

    query_parser = subparsers.add_parser("query", help="Ask a question against the built index")
    query_parser.add_argument("data_folder", type=Path, help="Path to the data folder")
    query_parser.add_argument("question", type=str, help="Question to ask the system")
    query_parser.add_argument("--top-k", type=int, default=ModelConfig().max_retrievals)
    query_parser.add_argument("--use-openai", action="store_true")
    query_parser.add_argument("--hf-token", type=str, default=None)
    query_parser.add_argument("--openai-key", type=str, default=None)
    query_parser.add_argument("--vlm-model", type=str, default=ModelConfig().vlm_model)
    query_parser.add_argument("--index-dir", type=Path, default=ModelConfig().index_dir)
    query_parser.add_argument("--render-pdf-pages", action="store_true", help="Render PDF pages as images for page-image retrieval")
    query_parser.add_argument("--pdf-render-dir", type=Path, default=ModelConfig().pdf_render_dir)
    query_parser.add_argument("--json", action="store_true", help="Emit structured grounded answer JSON")
    _add_retrieval_mode_args(query_parser)

    demo_parser = subparsers.add_parser("demo", help="Run a quick interactive demo")
    demo_parser.add_argument("data_folder", type=Path, help="Path to the folder containing text, image, or PDF files")
    demo_parser.add_argument("--hf-token", type=str, default=None)
    demo_parser.add_argument("--vlm-model", type=str, default=ModelConfig().vlm_model)
    demo_parser.add_argument("--index-dir", type=Path, default=ModelConfig().index_dir)
    demo_parser.add_argument("--render-pdf-pages", action="store_true", help="Render PDF pages as images for page-image retrieval")
    demo_parser.add_argument("--pdf-render-dir", type=Path, default=ModelConfig().pdf_render_dir)
    _add_retrieval_mode_args(demo_parser)

    web_demo_parser = subparsers.add_parser("demo-web", help="Run the animated local browser demo")
    web_demo_parser.add_argument("--host", type=str, default="127.0.0.1")
    web_demo_parser.add_argument("--port", type=int, default=7860)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate QA and citation predictions")
    eval_parser.add_argument("data_folder", type=Path, help="Path to the folder containing documents")
    eval_parser.add_argument("examples_path", type=Path, help="Path to the JSON evaluation examples file")
    eval_parser.add_argument("--top-k", type=int, default=ModelConfig().max_retrievals)
    eval_parser.add_argument("--use-openai", action="store_true")
    eval_parser.add_argument("--hf-token", type=str, default=None)
    eval_parser.add_argument("--openai-key", type=str, default=None)
    eval_parser.add_argument("--vlm-model", type=str, default=ModelConfig().vlm_model)
    eval_parser.add_argument("--retrieval-model", type=str, default=ModelConfig().retrieval_model)
    eval_parser.add_argument("--index-dir", type=Path, default=ModelConfig().index_dir)
    eval_parser.add_argument("--render-pdf-pages", action="store_true", help="Render PDF pages as images for page-image retrieval")
    eval_parser.add_argument("--pdf-render-dir", type=Path, default=ModelConfig().pdf_render_dir)
    _add_retrieval_mode_args(eval_parser)

    generate_parser = subparsers.add_parser("generate-eval", help="Generate evaluation examples from QA input")
    generate_parser.add_argument("input_path", type=Path, help="Path to the QA input file (csv, json, jsonl)")
    generate_parser.add_argument("output_path", type=Path, help="Path to the output JSON evaluation file")
    generate_parser.add_argument("--format", type=str, default="auto", choices=["auto", "csv", "json", "jsonl"])
    generate_parser.add_argument("--question-key", type=str, default="question")
    generate_parser.add_argument("--answer-key", type=str, default="answer")
    generate_parser.add_argument("--citations-key", type=str, default="citations")

    import_parser = subparsers.add_parser("import-data", help="Download or register public CUAD/ACORD/InsuranceQA/public PDF/real web data sources")
    import_parser.add_argument("--output-root", type=Path, default=Path("data"))
    import_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cuad", "acord", "insuranceqa"],
        choices=["cuad", "acord", "insuranceqa", "public_docs", "real_domain_mix"],
    )

    qa_parser = subparsers.add_parser("generate-qa", help="Generate QA/evidence pairs and hard negatives")
    qa_parser.add_argument("data_folder", type=Path, help="Folder containing policy PDFs/text files")
    qa_parser.add_argument("--output-dir", type=Path, default=Path("data/02_processed"))
    qa_parser.add_argument("--cuad-master", type=Path, default=None, help="Optional CUAD master_clauses.csv path")
    qa_parser.add_argument("--acord-root", type=Path, default=None, help="Optional ACORD extracted folder or zip extract root")
    qa_parser.add_argument("--max-cuad", type=int, default=2000)
    qa_parser.add_argument("--max-acord", type=int, default=3000)
    qa_parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Expand rule-based PDF QA/evidence generation until roughly this many answerable examples exist",
    )
    qa_parser.add_argument("--unsupported-count", type=int, default=50, help="Number of unsupported questions for abstention evaluation")

    metrics_parser = subparsers.add_parser("retrieval-metrics", help="Compute retrieval metrics against generated QA evidence pages")
    metrics_parser.add_argument("data_folder", type=Path)
    metrics_parser.add_argument("qa_path", type=Path)
    metrics_parser.add_argument("--top-k", type=int, default=10)
    metrics_parser.add_argument("--retrieval-model", type=str, default=ModelConfig().retrieval_model)
    metrics_parser.add_argument("--index-dir", type=Path, default=ModelConfig().index_dir)
    _add_retrieval_mode_args(metrics_parser)

    visual_build_parser = subparsers.add_parser("build-visual-index", help="Build a page-image retrieval index from page_manifest.jsonl")
    visual_build_parser.add_argument("page_manifest", type=Path)
    visual_build_parser.add_argument("--index-dir", type=Path, default=Path("data/03_index/colqwen2"))
    visual_build_parser.add_argument("--backend", type=str, default="visual_stub", choices=sorted(SUPPORTED_VISUAL_BACKENDS))

    visual_metrics_parser = subparsers.add_parser("visual-retrieval-metrics", help="Evaluate page-image retrieval backend")
    visual_metrics_parser.add_argument("qa_path", type=Path)
    visual_metrics_parser.add_argument("--index-dir", type=Path, default=Path("data/03_index/colqwen2"))
    visual_metrics_parser.add_argument("--backend", type=str, default="visual_stub", choices=sorted(SUPPORTED_VISUAL_BACKENDS))
    visual_metrics_parser.add_argument("--top-k", type=int, default=10)

    ablation_parser = subparsers.add_parser("run-ablation", help="Run local/OpenAI/visual retrieval and answer ablations")
    ablation_parser.add_argument("--data-folder", type=Path, required=True)
    ablation_parser.add_argument("--qa-path", type=Path, required=True)
    ablation_parser.add_argument("--output-dir", type=Path, default=Path("reports/ablation"))
    ablation_parser.add_argument("--index-dir", type=Path, default=Path("data"))
    ablation_parser.add_argument("--visual-index-dir", type=Path, default=Path("data/03_index/colqwen2"))
    ablation_parser.add_argument("--top-k", type=int, default=5)

    calibration_parser = subparsers.add_parser("run-calibration", help="Run selective prediction and abstention calibration report")
    calibration_parser.add_argument("--data-folder", type=Path, required=True)
    calibration_parser.add_argument("--qa-path", type=Path, required=True)
    calibration_parser.add_argument("--output-dir", type=Path, default=Path("reports/calibration"))
    calibration_parser.add_argument("--index-dir", type=Path, default=Path("data"))
    calibration_parser.add_argument("--top-k", type=int, default=3)

    gpu_benchmark_parser = subparsers.add_parser(
        "run-gpu-benchmark",
        help="Run reproducible text/local-image/GPU visual benchmark and write a compact report",
    )
    gpu_benchmark_parser.add_argument("--data-folder", type=Path, required=True)
    gpu_benchmark_parser.add_argument("--output-dir", type=Path, default=Path("reports/research_proof"))
    gpu_benchmark_parser.add_argument("--backend", type=str, default="colqwen2_local", choices=sorted(SUPPORTED_VISUAL_BACKENDS))
    gpu_benchmark_parser.add_argument("--target-count", type=int, default=300)
    gpu_benchmark_parser.add_argument("--unsupported-count", type=int, default=50)
    gpu_benchmark_parser.add_argument("--top-k", type=int, default=10)
    gpu_benchmark_parser.add_argument("--render-dpi", type=int, default=200)
    gpu_benchmark_parser.add_argument("--run-ocr", action="store_true")
    gpu_benchmark_parser.add_argument(
        "--allow-backend-failures",
        action="store_true",
        help="Write partial metrics if the requested GPU backend cannot be loaded on the current machine",
    )

    training_corpora_parser = subparsers.add_parser(
        "build-training-corpora",
        help="Build doc-disjoint retrieval triples, retrieval-conditioned SFT, and calibration manifests",
    )
    training_corpora_parser.add_argument("--data-folder", type=Path, required=True)
    training_corpora_parser.add_argument("--output-dir", type=Path, default=Path("reports/training_data"))
    training_corpora_parser.add_argument("--qa-path", type=Path, default=None)
    training_corpora_parser.add_argument("--hard-negatives-path", type=Path, default=None)
    training_corpora_parser.add_argument("--sft-dataset-path", type=Path, default=Path("data/04_curated/sft_dataset.jsonl"))
    training_corpora_parser.add_argument("--index-dir", type=Path, default=Path("data/train_index"))
    training_corpora_parser.add_argument("--retrieval-model", type=str, default="local-hashing")
    training_corpora_parser.add_argument("--target-qa-count", type=int, default=300)
    training_corpora_parser.add_argument("--unsupported-count", type=int, default=50)
    training_corpora_parser.add_argument("--top-k", type=int, default=5)
    training_corpora_parser.add_argument("--max-negatives", type=int, default=4)
    training_corpora_parser.add_argument("--max-sft-pages", type=int, default=3)
    _add_retrieval_mode_args(training_corpora_parser)

    dense_train_parser = subparsers.add_parser(
        "train-dense-retriever",
        help="Train a local dense retriever from retrieval_train.jsonl triples",
    )
    dense_train_parser.add_argument("--dataset-path", type=Path, required=True)
    dense_train_parser.add_argument("--output-dir", type=Path, default=Path("models/retrieval/bge-base-insurerag"))
    dense_train_parser.add_argument("--model-name", type=str, default=DEFAULT_DENSE_BASE_MODEL)
    dense_train_parser.add_argument("--max-samples", type=int, default=None)
    dense_train_parser.add_argument("--max-length", type=int, default=384)
    dense_train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    dense_train_parser.add_argument("--num-train-epochs", type=float, default=2.0)
    dense_train_parser.add_argument("--max-steps", type=int, default=-1)
    dense_train_parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    dense_train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    dense_train_parser.add_argument("--logging-steps", type=int, default=10)
    dense_train_parser.add_argument("--save-steps", type=int, default=200)
    dense_train_parser.add_argument("--save-total-limit", type=int, default=2)
    dense_train_parser.add_argument("--seed", type=int, default=42)
    dense_train_parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16")
    dense_train_parser.add_argument("--margin", type=float, default=0.2)
    dense_train_parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    dense_train_parser.add_argument("--auto-resume", action="store_true")

    sft_parser = subparsers.add_parser("sft-lora-qwen", help="SFT Qwen 7B with LoRA/QLoRA on the curated SFT dataset")
    sft_parser.add_argument("--dataset-path", type=Path, default=Path("data/04_curated/sft_dataset.jsonl"))
    sft_parser.add_argument("--output-dir", type=Path, default=Path("models/qwen7b-insurerag-lora"))
    sft_parser.add_argument("--model-name", type=str, default=DEFAULT_QWEN_7B_MODEL)
    sft_parser.add_argument("--adapter-path", type=Path, default=None, help="Optional existing LoRA adapter to continue training from")
    sft_parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick GPU tests")
    sft_parser.add_argument("--max-length", type=int, default=2048)
    sft_parser.add_argument("--lora-r", type=int, default=16)
    sft_parser.add_argument("--lora-alpha", type=int, default=32)
    sft_parser.add_argument("--lora-dropout", type=float, default=0.05)
    sft_parser.add_argument("--learning-rate", type=float, default=2e-4)
    sft_parser.add_argument("--num-train-epochs", type=float, default=1.0)
    sft_parser.add_argument("--max-steps", type=int, default=-1, help="Use 1 for a tiny end-to-end GPU demo")
    sft_parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    sft_parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    sft_parser.add_argument("--logging-steps", type=int, default=10)
    sft_parser.add_argument("--save-steps", type=int, default=100)
    sft_parser.add_argument("--save-total-limit", type=int, default=2)
    sft_parser.add_argument("--seed", type=int, default=42)
    sft_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA loading")
    sft_parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16")
    sft_parser.add_argument("--no-gradient-checkpointing", action="store_true")
    sft_parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    sft_parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from the latest checkpoint-* directory under --output-dir when present",
    )

    sft_smoke_parser = subparsers.add_parser(
        "sft-lora-smoke-test",
        help="Validate curated SFT records, Qwen tokenization, and CUDA visibility",
    )
    sft_smoke_parser.add_argument("--dataset-path", type=Path, default=Path("data/04_curated/sft_dataset.jsonl"))
    sft_smoke_parser.add_argument("--model-name", type=str, default=DEFAULT_QWEN_7B_MODEL)
    sft_smoke_parser.add_argument("--max-length", type=int, default=1024)
    sft_smoke_parser.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Only for local CPU syntax checks; real SFT still requires CUDA",
    )
    sft_smoke_parser.add_argument(
        "--format-only",
        action="store_true",
        help="Validate SFT prompt/label formatting without torch, CUDA, or Hugging Face downloads",
    )

    validate_parser = subparsers.add_parser(
        "validate-curated-data",
        help="Validate curated RAG/SFT JSONL datasets and write research-proof data quality reports",
    )
    validate_parser.add_argument("--dataset-dir", type=Path, default=Path("data/04_curated"))
    validate_parser.add_argument("--output-dir", type=Path, default=Path("reports/research_proof"))
    validate_parser.add_argument("--min-unsupported", type=int, default=50)
    validate_parser.add_argument("--min-sft-records", type=int, default=1)
    validate_parser.add_argument("--min-rag-records", type=int, default=1)
    validate_parser.add_argument("--no-update-summary", action="store_true")

    diff_parser = subparsers.add_parser("diff", help="Compare two text or PDF sources for clause changes")
    diff_parser.add_argument("original", type=Path, help="Original text or PDF file")
    diff_parser.add_argument("revised", type=Path, help="Revised text or PDF file")

    policy_diff_parser = subparsers.add_parser("policy-diff", help="Summarize policy-specific coverage/deductible/endorsement drift")
    policy_diff_parser.add_argument("original", type=Path)
    policy_diff_parser.add_argument("revised", type=Path)
    policy_diff_parser.add_argument("--output", type=Path, default=Path("reports/diff/diff_summary.json"))

    extract_parser = subparsers.add_parser("extract-pdf", help="Extract text or layout from a PDF file")
    extract_parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    extract_parser.add_argument("--output-dir", type=Path, default=None)
    extract_parser.add_argument("--layout", action="store_true", help="Print page layout blocks")
    extract_parser.add_argument("--render", action="store_true", help="Render page images to the output directory")

    args = parser.parse_args()
    use_hf_api = True
    if args.command in {"query", "evaluate"}:
        use_hf_api = not args.use_openai
    elif args.command == "demo":
        use_hf_api = True

    config = ModelConfig(
        retrieval_model=getattr(args, "retrieval_model", ModelConfig().retrieval_model),
        vlm_model=getattr(args, "vlm_model", ModelConfig().vlm_model),
        use_hf_api=use_hf_api,
        hf_api_token=getattr(args, "hf_token", None),
        openai_api_key=getattr(args, "openai_key", None),
        retrieval_mode=getattr(args, "retrieval_mode", ModelConfig().retrieval_mode),
        corpus_source=getattr(args, "corpus_source", ModelConfig().corpus_source),
        enable_image_signal=not getattr(args, "disable_image_signal", False),
        index_dir=getattr(args, "index_dir", ModelConfig().index_dir),
        render_pdf_pages=getattr(args, "render_pdf_pages", False),
        pdf_render_dir=getattr(args, "pdf_render_dir", ModelConfig().pdf_render_dir),
    )

    if args.command == "preprocess-pages":
        result = preprocess_page_images(
            PageImagePreprocessConfig(
                input_dir=args.input_dir,
                output_root=args.output_root,
                render_dpi=args.render_dpi,
                run_ocr=args.run_ocr,
                source_type=args.source_type,
                train_ratio=args.train_ratio,
                valid_ratio=args.valid_ratio,
                seed=args.seed,
            )
        )
        print("Page-image preprocessing complete.")
        print(f"Documents: {result.document_count}")
        print(f"Pages: {result.page_count}")
        print(f"Documents table: {result.documents_path}")
        print(f"Pages table: {result.pages_path}")
        print(f"OCR/text auxiliary metadata: {result.ocr_aux_path}")
        print(f"ColQwen2 page manifest: {result.page_manifest_path}")
        print(f"Weak retrieval pairs: {result.retrieval_pairs_path}")
        print(f"Document splits: {result.splits_path}")
        return

    if args.command == "demo-web":
        run_demo_server(host=args.host, port=args.port)
        return

    if args.command == "sft-lora-smoke-test":
        result = run_lora_smoke_test(
            dataset_path=args.dataset_path,
            model_name=args.model_name,
            max_length=args.max_length,
            skip_cuda_check=args.skip_cuda_check,
            format_only=args.format_only,
        )
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.__dict__.items()}, indent=2))
        return

    if args.command == "sft-lora-qwen":
        result = run_lora_sft(
            QwenLoraSFTConfig(
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                model_name=args.model_name,
                adapter_path=args.adapter_path,
                max_samples=args.max_samples,
                max_length=args.max_length,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                learning_rate=args.learning_rate,
                num_train_epochs=args.num_train_epochs,
                max_steps=args.max_steps,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                logging_steps=args.logging_steps,
                save_steps=args.save_steps,
                save_total_limit=args.save_total_limit,
                seed=args.seed,
                load_in_4bit=not args.no_4bit,
                bf16=not args.fp16,
                gradient_checkpointing=not args.no_gradient_checkpointing,
                resume_from_checkpoint=args.resume_from_checkpoint,
                auto_resume=args.auto_resume,
            )
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "validate-curated-data":
        result = validate_curated_data(
            CuratedValidationConfig(
                dataset_dir=args.dataset_dir,
                output_dir=args.output_dir,
                min_unsupported=args.min_unsupported,
                min_sft_records=args.min_sft_records,
                min_rag_records=args.min_rag_records,
                update_summary=not args.no_update_summary,
            )
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["passed"]:
            raise SystemExit(1)
        return

    if args.command == "build-training-corpora":
        result = build_training_corpora(
            TrainingCorpusBuildConfig(
                data_folder=args.data_folder,
                output_dir=args.output_dir,
                qa_path=args.qa_path,
                hard_negatives_path=args.hard_negatives_path,
                sft_dataset_path=args.sft_dataset_path,
                index_dir=args.index_dir,
                retrieval_model=args.retrieval_model,
                retrieval_mode=args.retrieval_mode,
                corpus_source=args.corpus_source,
                enable_image_signal=not args.disable_image_signal,
                target_qa_count=args.target_qa_count,
                unsupported_count=args.unsupported_count,
                top_k=args.top_k,
                max_negatives=args.max_negatives,
                max_sft_pages=args.max_sft_pages,
            )
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "train-dense-retriever":
        result = run_dense_retriever_training(
            DenseRetrieverTrainConfig(
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                model_name=args.model_name,
                max_samples=args.max_samples,
                max_length=args.max_length,
                learning_rate=args.learning_rate,
                num_train_epochs=args.num_train_epochs,
                max_steps=args.max_steps,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                logging_steps=args.logging_steps,
                save_steps=args.save_steps,
                save_total_limit=args.save_total_limit,
                seed=args.seed,
                bf16=not args.fp16,
                margin=args.margin,
                resume_from_checkpoint=args.resume_from_checkpoint,
                auto_resume=args.auto_resume,
            )
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    pipeline = DocumentRetrievalPipeline(config)

    if args.command == "build-index":
        pipeline.build_index(args.data_folder)
        print("Index build complete.")
    elif args.command == "query":
        if args.json:
            result = pipeline.query_structured(args.question, args.data_folder, top_k=args.top_k)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            answer = pipeline.query(args.question, args.data_folder, top_k=args.top_k)
            print("\n=== ANSWER ===\n", answer)
    elif args.command == "demo":
        pipeline.quick_demo(args.data_folder)
    elif args.command == "evaluate":
        metrics = pipeline.evaluate(args.data_folder, args.examples_path, top_k=args.top_k)
        print("\n=== EVALUATION ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
    elif args.command == "generate-eval":
        generate_evaluation_examples(
            args.input_path,
            args.output_path,
            input_format=args.format,
            question_key=args.question_key,
            answer_key=args.answer_key,
            citations_key=args.citations_key,
        )
        print(f"Generated evaluation examples at {args.output_path}")
    elif args.command == "import-data":
        result = download_public_datasets(output_root=args.output_root, datasets=args.datasets)
        print(f"External data manifest: {result['manifest_path']}")
        for record in result["records"]:
            print(f"- {record['dataset']}: {record['status']} -> {record['local_path']}")
    elif args.command == "generate-qa":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        policy_result = generate_policy_qa_pairs(
            args.data_folder,
            args.output_dir,
            target_count=args.target_count,
            unsupported_count=args.unsupported_count,
        )
        qa_files = [policy_result.qa_path]
        print(f"Policy QA: {policy_result.qa_count} examples -> {policy_result.qa_path}")
        print(f"Policy hard negatives: {policy_result.hard_negative_count} -> {policy_result.hard_negatives_path}")
        if policy_result.splits_path:
            print(f"Policy QA document splits: {policy_result.splits_path}")

        if args.cuad_master:
            cuad_path = args.output_dir / "cuad_qa_pairs.jsonl"
            count = import_cuad_qa(args.cuad_master, cuad_path, max_examples=args.max_cuad)
            qa_files.append(cuad_path)
            print(f"CUAD QA: {count} examples -> {cuad_path}")

        if args.acord_root:
            acord_qa_path = args.output_dir / "acord_qa_pairs.jsonl"
            acord_hard_path = args.output_dir / "acord_hard_negatives.jsonl"
            acord_result = import_acord_qa(
                args.acord_root,
                acord_qa_path,
                acord_hard_path,
                max_examples=args.max_acord,
            )
            qa_files.append(acord_qa_path)
            print(f"ACORD QA: {acord_result.qa_count} examples -> {acord_result.qa_path}")
            print(f"ACORD hard negatives: {acord_result.hard_negative_count} -> {acord_result.hard_negatives_path}")

        merged_path = args.output_dir / "qa_pairs_merged.jsonl"
        merged_count = merge_qa_files(qa_files, merged_path)
        print(f"Merged QA: {merged_count} examples -> {merged_path}")
    elif args.command == "retrieval-metrics":
        metrics_config = ModelConfig(
            retrieval_model=args.retrieval_model,
            index_dir=args.index_dir,
            retrieval_mode=args.retrieval_mode,
            corpus_source=args.corpus_source,
            enable_image_signal=not args.disable_image_signal,
        )
        metrics_pipeline = DocumentRetrievalPipeline(metrics_config)
        metrics_pipeline.build_index(args.data_folder)
        metrics = compute_retrieval_metrics(metrics_pipeline, args.data_folder, args.qa_path, top_k=args.top_k)
        print("\n=== RETRIEVAL METRICS ===")
        print(f"evaluated_count: {metrics.evaluated_count}")
        print(f"recall_at_1: {metrics.recall_at_1:.4f}")
        print(f"recall_at_5: {metrics.recall_at_5:.4f}")
        print(f"mrr_at_10: {metrics.mrr_at_10:.4f}")
        print(f"ndcg_at_10: {metrics.ndcg_at_10:.4f}")
    elif args.command == "build-visual-index":
        result = build_visual_index(args.page_manifest, args.index_dir, backend=args.backend)
        print("Visual index build complete.")
        print(f"backend: {result.backend}")
        print(f"pages: {result.page_count}")
        print(f"index: {result.index_path}")
        print(f"metadata: {result.metadata_path}")
    elif args.command == "visual-retrieval-metrics":
        metrics = compute_visual_retrieval_metrics(args.qa_path, args.index_dir, backend=args.backend, top_k=args.top_k)
        print("\n=== VISUAL RETRIEVAL METRICS ===")
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"{name}: {value:.4f}")
            else:
                print(f"{name}: {value}")
    elif args.command == "run-ablation":
        outputs = run_ablation(
            data_folder=args.data_folder,
            qa_path=args.qa_path,
            output_dir=args.output_dir,
            index_dir=args.index_dir,
            visual_index_dir=args.visual_index_dir,
            top_k=args.top_k,
        )
        print("Ablation complete.")
        for name, path in outputs.items():
            print(f"{name}: {path}")
    elif args.command == "run-calibration":
        outputs = run_calibration(
            data_folder=args.data_folder,
            qa_path=args.qa_path,
            output_dir=args.output_dir,
            index_dir=args.index_dir,
            top_k=args.top_k,
        )
        print("Calibration complete.")
        for name, path in outputs.items():
            print(f"{name}: {path}")
    elif args.command == "run-gpu-benchmark":
        outputs = run_gpu_benchmark(
            RunGpuBenchmarkConfig(
                data_folder=args.data_folder,
                output_dir=args.output_dir,
                backend=args.backend,
                target_count=args.target_count,
                unsupported_count=args.unsupported_count,
                top_k=args.top_k,
                render_dpi=args.render_dpi,
                run_ocr=args.run_ocr,
                allow_backend_failures=args.allow_backend_failures,
            )
        )
        print("GPU benchmark complete.")
        for name, path in outputs.items():
            print(f"{name}: {path}")
    elif args.command == "diff":
        old_text = _load_source_text(args.original)
        new_text = _load_source_text(args.revised)
        changes = compare_clause_diff(old_text, new_text)
        print(render_clause_diff(changes))
        print("\n=== DIFF SUMMARY ===")
        from .diff import summarize_clause_diff

        print(summarize_clause_diff(old_text, new_text))
    elif args.command == "policy-diff":
        old_text = _load_source_text(args.original)
        new_text = _load_source_text(args.revised)
        result = write_policy_diff(old_text, new_text, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nWrote policy diff summary to {args.output}")
    elif args.command == "extract-pdf":
        if args.layout:
            layouts = extract_layout_by_page(args.pdf_path)
            for page_layout in layouts:
                print(f"Page {page_layout.page_number}: {len(page_layout.blocks)} blocks")
                for block in page_layout.blocks:
                    print(f"  - bbox={block['bbox']} text={block['text'][:100]!r}")
        if args.render and args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            images = render_pdf_pages(args.pdf_path)
            for idx, image_bytes in enumerate(images, start=1):
                output_path = args.output_dir / f"page_{idx:03}.png"
                output_path.write_bytes(image_bytes)
                print(f"Rendered page {idx} to {output_path}")
        if not args.layout and not args.render:
            pages = extract_text_by_page(args.pdf_path)
            for idx, text in enumerate(pages, start=1):
                print(f"\n--- PAGE {idx} ---\n")
                print(text[:200])
    else:
        parser.print_help()
