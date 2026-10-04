"""Search the research FAQ corpus with the validation-selected retrieve/rerank pipeline."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha
from scripts.eval_insuranceqa_scale import SparseBM25, ranked
from scripts.eval_insuranceqa_reranker import rank_row
from src.insurerag_vlm.retriever import EmbeddingRetriever
from src.insurerag_vlm.reranker import LocalCrossEncoder


def run(args):
    start = time.perf_counter()
    selection = json.loads(args.selection.read_text(encoding='utf8'))
    protocol = json.loads((args.scoring_run/'protocol.json').read_text(encoding='utf8'))
    if selection['scoring_protocol_sha256'] != sha(args.scoring_run/'protocol.json'):
        raise ValueError('Selection and scoring protocol do not match')
    for relative, expected in selection['code_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Inference implementation changed since validation: {relative}')
    if selection['fixture_lock_sha256'] != sha(args.fixture/'manifest.lock.json'):
        raise ValueError('Selection does not describe this answer corpus')
    # Load answers only: an interactive query never reads question labels.
    manifest = json.loads((args.fixture/'manifest.lock.json').read_text(encoding='utf8'))
    if sha(args.fixture/'answers.jsonl') != manifest['files']['answers.jsonl']:
        raise ValueError('Answer corpus hash mismatch')
    if sha(args.baseline/'answer_embeddings.npy') != protocol['embedding_cache_sha256']['answer_embeddings.npy']:
        raise ValueError('Answer embedding cache hash mismatch')
    answers = sorted(read_jsonl(args.fixture/'answers.jsonl'), key=lambda a: int(a['id']))
    embeddings = np.load(args.baseline/'answer_embeddings.npy')
    retriever = EmbeddingRetriever(str(args.embedding_model), use_hf_api=False, pooling='cls', max_length=512,
                                  query_instruction=protocol['embedding']['query_instruction'])
    # Acquisition timestamps/paths can differ after downloading the same pinned
    # weights. Verify all inference files, excluding acquisition-only metadata.
    inference_files = lambda files: {k: v for k, v in files.items() if k != 'download_provenance.json'}
    if inference_files(retriever.index_fingerprint()['checkpoint_files_sha256']) != inference_files(protocol['embedding']['checkpoint_files_sha256']):
        raise ValueError('BGE checkpoint differs from the benchmark checkpoint')
    import torch
    torch.set_num_threads(4)
    _, model = retriever._ensure_local_transformer(); model.to(args.device)
    query = retriever.embed_texts([retriever.query_instruction+args.question])[0]
    d = embeddings @ query; sparse = SparseBM25([a['text'] for a in answers]); b = sparse.scores(args.question)
    bo, do = ranked(b, positive_only=True), ranked(d); union = sorted(set(bo)|set(do))
    ce = LocalCrossEncoder(args.reranker_model, args.device)
    if inference_files(ce.fingerprint()['files_sha256']) != inference_files(protocol['reranker']['files_sha256']):
        raise ValueError('Reranker checkpoint differs from the validation checkpoint')
    cross = ce.score(args.question, [answers[j]['text'] for j in union])
    row = {'candidate_ids': [answers[j]['id'] for j in union], 'dense': d[union].tolist(),
           'sparse': b[union].tolist(), 'cross': cross.tolist(), 'bge_ids': [answers[j]['id'] for j in do]}
    order = rank_row(row, selection['config']); lookup = {a['id']: a for a in answers}
    result = {'question': args.question, 'retrieval_only': True, 'generation_calls': 0,
              'corpus': 'Historical InsuranceQA FAQ answers; not a personal policy or current benefits authority',
              'config': selection['config'], 'candidates_scored': len(union),
              'seconds_including_model_and_index_loading': time.perf_counter()-start,
              'results': [{'rank': i+1, 'answer_id': id, 'text': lookup[id]['text']} for i, id in enumerate(order[:args.top_k])]}
    text = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        args.output.write_text(text, encoding='utf8')
    print(text)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--question', required=True)
    p.add_argument('--embedding-model', type=Path, required=True)
    p.add_argument('--reranker-model', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--scoring-run', type=Path, required=True)
    p.add_argument('--fixture', type=Path, default=ROOT/'data/benchmarks/insuranceqa_v2')
    p.add_argument('--baseline', type=Path, default=ROOT/'reports/insuranceqa_v2/retrieval_frozen')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--output', type=Path)
    run(p.parse_args())
