import argparse
from pathlib import Path
from typing import List

from .config import ModelConfig
from .diff import compare_clause_diff, render_clause_diff, summarize_clause_diff
from .evaluation import generate_evaluation_examples
from .pdf import extract_layout_by_page, extract_text_by_page, render_pdf_pages
from .pipeline import DocumentRetrievalPipeline
from .preprocess import PageImagePreprocessConfig, preprocess_page_images


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _load_source_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        pages = extract_text_by_page(path)
        return "\n\n---\n\n".join(pages)
    return _read_text_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="InsureRAG-VLM retrieval and inference pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index", help="Build a page index from a data folder")
    build_parser.add_argument("data_folder", type=Path, help="Path to the folder containing text, image, or PDF files")
    build_parser.add_argument("--retrieval-model", default=ModelConfig().retrieval_model)
    build_parser.add_argument("--index-dir", type=Path, default=ModelConfig().index_dir)
    build_parser.add_argument("--render-pdf-pages", action="store_true", help="Render PDF pages as images for page-image indexing")
    build_parser.add_argument("--pdf-render-dir", type=Path, default=ModelConfig().pdf_render_dir)

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

    demo_parser = subparsers.add_parser("demo", help="Run a quick interactive demo")
    demo_parser.add_argument("data_folder", type=Path, help="Path to the folder containing text, image, or PDF files")
    demo_parser.add_argument("--hf-token", type=str, default=None)
    demo_parser.add_argument("--vlm-model", type=str, default=ModelConfig().vlm_model)
    demo_parser.add_argument("--index-dir", type=Path, default=ModelConfig().index_dir)
    demo_parser.add_argument("--render-pdf-pages", action="store_true", help="Render PDF pages as images for page-image retrieval")
    demo_parser.add_argument("--pdf-render-dir", type=Path, default=ModelConfig().pdf_render_dir)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate QA and citation predictions")
    eval_parser.add_argument("data_folder", type=Path, help="Path to the folder containing documents")
    eval_parser.add_argument("examples_path", type=Path, help="Path to the JSON evaluation examples file")
    eval_parser.add_argument("--top-k", type=int, default=ModelConfig().max_retrievals)
    eval_parser.add_argument("--use-openai", action="store_true")
    eval_parser.add_argument("--hf-token", type=str, default=None)
    eval_parser.add_argument("--openai-key", type=str, default=None)
    eval_parser.add_argument("--vlm-model", type=str, default=ModelConfig().vlm_model)
    eval_parser.add_argument("--index-dir", type=Path, default=ModelConfig().index_dir)
    eval_parser.add_argument("--render-pdf-pages", action="store_true", help="Render PDF pages as images for page-image retrieval")
    eval_parser.add_argument("--pdf-render-dir", type=Path, default=ModelConfig().pdf_render_dir)

    generate_parser = subparsers.add_parser("generate-eval", help="Generate evaluation examples from QA input")
    generate_parser.add_argument("input_path", type=Path, help="Path to the QA input file (csv, json, jsonl)")
    generate_parser.add_argument("output_path", type=Path, help="Path to the output JSON evaluation file")
    generate_parser.add_argument("--format", type=str, default="auto", choices=["auto", "csv", "json", "jsonl"])
    generate_parser.add_argument("--question-key", type=str, default="question")
    generate_parser.add_argument("--answer-key", type=str, default="answer")
    generate_parser.add_argument("--citations-key", type=str, default="citations")

    diff_parser = subparsers.add_parser("diff", help="Compare two text or PDF sources for clause changes")
    diff_parser.add_argument("original", type=Path, help="Original text or PDF file")
    diff_parser.add_argument("revised", type=Path, help="Revised text or PDF file")

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

    pipeline = DocumentRetrievalPipeline(config)

    if args.command == "build-index":
        pipeline.build_index(args.data_folder)
        print("Index build complete.")
    elif args.command == "query":
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
    elif args.command == "diff":
        old_text = _load_source_text(args.original)
        new_text = _load_source_text(args.revised)
        changes = compare_clause_diff(old_text, new_text)
        print(render_clause_diff(changes))
        print("\n=== DIFF SUMMARY ===")
        from .diff import summarize_clause_diff

        print(summarize_clause_diff(old_text, new_text))
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
