"""Pin the unchanged BGE document index and original training-query anchors."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json
from src.insurerag_vlm.retriever import EmbeddingRetriever


def main():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    threadpool_limits(4)
    start = time.perf_counter()
    data = ROOT / 'data/training/query_adaptation_v1'
    manifest = json.loads((data / 'manifest.lock.json').read_text(encoding='utf8'))
    for name, digest in manifest['files'].items():
        assert sha(data / name) == digest
    answers = read_jsonl(data / 'answers.jsonl')
    groups = read_jsonl(data / 'train_groups.jsonl')
    base = ROOT / 'reports/condition_v1/index_cache'
    base_manifest = json.loads((base / 'manifest.json').read_text(encoding='utf8'))
    assert sha(base / 'answer_embeddings.npy') == base_manifest['answer_embeddings_sha256']
    prioranswers = read_jsonl(ROOT / 'data/training/condition_listwise_v1/answers.jsonl')
    assert answers[:len(prioranswers)] == prioranswers
    out = ROOT / 'reports/query_adaptation_v1/index_cache'
    out.mkdir(parents=True, exist_ok=False)
    encoder = EmbeddingRetriever(str(ROOT / '../models/bge-small-en-v1.5'), use_hf_api=False,
                                 pooling='cls', max_length=512,
                                 query_instruction='Represent this sentence for searching relevant passages: ')
    _, model = encoder._ensure_local_transformer()
    model.to('cuda')
    assert encoder.index_fingerprint()['checkpoint_files_sha256'] == base_manifest['embedding']['checkpoint_files_sha256']
    vectors = [np.load(base / 'answer_embeddings.npy')]
    additional = answers[len(prioranswers):]
    for offset in range(0, len(additional), 256):
        vectors.append(encoder.embed_texts([a['text'] for a in additional[offset:offset+256]]))
        if offset % 2048 == 0:
            print(json.dumps({'fiqa_documents_encoded': min(offset+256, len(additional)), 'total': len(additional),
                              'seconds': round(time.perf_counter()-start, 1)}), flush=True)
    dense = np.vstack(vectors)
    np.save(out / 'answer_embeddings.npy', dense)
    priorgroups = read_jsonl(ROOT / 'data/training/condition_v1/train_groups.jsonl')
    priorquery = np.load(base / 'training_query_embeddings.npy')
    assert len(priorgroups) == len(priorquery)
    previous = {g['id']: (g['question'], i) for i, g in enumerate(priorgroups)}
    queries = np.empty((len(groups), 384), dtype=np.float32)
    new = []
    for i, g in enumerate(groups):
        if g['id'] in previous:
            text, j = previous[g['id']]
            assert text == g['question']
            queries[i] = priorquery[j]
        else:
            new.append(i)
    for offset in range(0, len(new), 256):
        indices = new[offset:offset+256]
        queries[indices] = encoder.embed_texts([encoder.query_instruction + groups[i]['question'] for i in indices])
    np.save(out / 'training_query_embeddings.npy', queries)
    assert np.isfinite(dense).all() and np.isfinite(queries).all()
    assert np.allclose(np.linalg.norm(dense, axis=1), 1., atol=1e-5)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1., atol=1e-5)
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'answer_count': len(answers),
                'training_questions': len(groups), 'answer_file_sha256': sha(data / 'answers.jsonl'),
                'training_groups_sha256': sha(data / 'train_groups.jsonl'),
                'embedding': encoder.index_fingerprint(), 'document_encoder_frozen': True,
                'answer_embeddings_sha256': sha(out / 'answer_embeddings.npy'),
                'training_query_embeddings_sha256': sha(out / 'training_query_embeddings.npy'),
                'parent_index_manifest_sha256': sha(base / 'manifest.json'),
                'reused_document_embeddings': len(prioranswers), 'new_document_embeddings': len(additional),
                'new_query_embeddings': len(new), 'encoding_precision': 'FP32',
                'seconds': time.perf_counter()-start, 'code_sha256': sha(Path(__file__))}, out / 'manifest.json')
    print(json.dumps({'completed': True, 'seconds': round(time.perf_counter()-start, 1)}), flush=True)


if __name__ == '__main__':
    main()
