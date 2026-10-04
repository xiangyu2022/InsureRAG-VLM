"""Build a reproducible local BGE index for a curated research snippet corpus."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, write_json, sha
from src.insurerag_vlm.retriever import EmbeddingRetriever


def run(args):
    import torch
    torch.set_num_threads(4)
    records_path = args.corpus/'rag_snippets.jsonl'
    manifest = json.loads((args.corpus/'manifest.json').read_text(encoding='utf8'))
    if sha(records_path) != manifest['files']['rag_snippets.jsonl']:
        raise ValueError('Corpus snippets changed')
    rows = read_jsonl(records_path)
    if not rows or len({r['record_id'] for r in rows}) != len(rows) or any(not r['text'].strip() for r in rows):
        raise ValueError('Empty text or colliding snippet IDs')
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = EmbeddingRetriever(str(args.model), use_hf_api=False, pooling='cls', max_length=512,
                                query_instruction='Represent this sentence for searching relevant passages: ')
    protocol = {'created_utc':datetime.now(timezone.utc).isoformat(), 'snippets':len(rows),
        'corpus_manifest_sha256':sha(args.corpus/'manifest.json'), 'records_sha256':sha(records_path),
        'embedding':encoder.index_fingerprint(), 'code_sha256':sha(Path(__file__)),
        'retriever_code_sha256':sha(ROOT/'src/insurerag_vlm/retriever.py'),
        'algorithm':'Exact cosine; no approximate HNSW error', 'device':args.device,
        'logical_records':manifest['unique_normalized_text_records'], 'generation_calls':0}
    write_json(protocol, args.output/'protocol.json')
    tokenizer, model = encoder._ensure_local_transformer(); model.to(args.device)
    vectors = []; truncated = 0; start = time.perf_counter()
    for i in range(0,len(rows),256):
        texts = [r['text'] for r in rows[i:i+256]]
        lengths = tokenizer(texts, truncation=False, add_special_tokens=True, verbose=False)['input_ids']
        truncated += sum(len(t)>512 for t in lengths)
        vectors.append(encoder.embed_texts(texts))
        if i%4096 == 0 or i+256 >= len(rows):
            print(json.dumps({'embedded':min(i+256,len(rows)),'total':len(rows),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    dense = np.vstack(vectors); np.save(args.output/'answer_embeddings.npy', dense)
    if sha(records_path) != protocol['records_sha256'] or sha(Path(__file__)) != protocol['code_sha256']:
        raise ValueError('Index source changed during build')
    write_json({'status':'completed','snippets':len(rows),'dimension':int(dense.shape[1]),
        'seconds':time.perf_counter()-start,'truncated_at_512_tokens':truncated,
        'answer_embeddings_sha256':sha(args.output/'answer_embeddings.npy'),
        'protocol_sha256':sha(args.output/'protocol.json')},args.output/'completion.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(p.parse_args())
