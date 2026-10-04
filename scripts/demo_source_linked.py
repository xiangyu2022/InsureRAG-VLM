"""Start the existing local demo on the expanded source-linked research corpus."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.insurerag_vlm import app
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, default=ROOT/'data/research_corpus/scaled_public_v2')
    p.add_argument('--index-dir', type=Path, default=ROOT/'reports/scaled_public_demo/index')
    p.add_argument('--retrieval-model', default='local-hashing', help='Explicit CPU baseline or local BGE checkpoint directory')
    p.add_argument('--vlm-model', default='local-extractive', help='Explicit CPU baseline or ollama:qwen3.5:4b')
    p.add_argument('--endpoint', default='http://127.0.0.1:11435')
    p.add_argument('--expected-digest')
    p.add_argument('--graph-mode', choices=['off', 'explicit', 'all'], default='explicit')
    p.add_argument('--retrieval-mode', choices=['sparse_only', 'dense_only', 'hybrid_text'], default='hybrid_text')
    p.add_argument('--port', type=int, default=7860)
    p.add_argument('--smoke-question', help='Run one real application-path request and exit without starting a server')
    args = p.parse_args()
    corpus = args.corpus.resolve()
    config = ModelConfig(retrieval_model=args.retrieval_model, vlm_model=args.vlm_model,
        ollama_base_url=args.endpoint, vlm_expected_digest=args.expected_digest,
        corpus_source='curated', curated_dataset_dir=corpus, index_dir=args.index_dir,
        retrieval_mode=args.retrieval_mode, enable_image_signal=False, graph_mode=args.graph_mode)
    pipeline = DocumentRetrievalPipeline(config)
    pipeline.build_index(corpus)
    # Keep the application's existing public-reference intent guard for this corpus.
    app.DATA_FOLDER = corpus
    app.DemoHandler._data_folder = corpus
    app.DemoHandler._index_dir = args.index_dir
    app.DemoHandler._pipeline = pipeline
    if args.smoke_question:
        print(json.dumps(app.build_chat_response(args.smoke_question, pipeline, corpus), ensure_ascii=True))
    else:
        print('Expanded source-linked public guides; research prototype, no personal policy loaded.', flush=True)
        app.run_demo_server(host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
