"""Use the validation-selected trained reranker on FAQ answers or HICRIC snippets."""
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
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder


def verify_files(actual, expected):
    strip = lambda d: {k:v for k,v in d.items() if k != 'download_provenance.json'}
    if strip(actual) != strip(expected):
        raise ValueError('Model inference files differ from the evaluated checkpoint')


def run(args):
    if not args.question.strip() or not 1 <= args.top_k <= 100:
        raise ValueError('Provide a nonempty question and top-k between 1 and 100')
    import torch
    torch.set_num_threads(4); start = time.perf_counter()
    selection_path = args.selection if args.profile == 'trained' else args.general_selection
    selection = json.loads(selection_path.read_text(encoding='utf8'))
    if selection['selected_on'] != 'valid': raise ValueError('Profile must use a validation selection')
    if args.profile == 'trained' and sha(args.reranker_model/'model.safetensors') != selection['selected_weights_sha256']:
        raise ValueError('Model is not the validation-selected trained checkpoint')
    for name, expected in selection['code_sha256'].items():
        if sha(ROOT/name) != expected: raise ValueError('Frozen ranking implementation changed')
    if args.profile == 'trained':
        scoring = selection_path.parent/f"valid_epoch_{selection['epoch']}"/'protocol.json'
        expected_scoring_sha = selection['validation_artifacts'][str(selection['epoch'])]['protocol_sha256']
        model_key = 'model'
    else:
        scoring = ROOT/'reports/insuranceqa_v2/rerank_valid_v2/protocol.json'
        expected_scoring_sha = selection['scoring_protocol_sha256']; model_key = 'reranker'
    if sha(scoring) != expected_scoring_sha: raise ValueError('Checkpoint evaluation provenance changed')
    scored_model = json.loads(scoring.read_text(encoding='utf8'))[model_key]
    if args.corpus == 'insuranceqa':
        fixture = ROOT/'data/benchmarks/insuranceqa_v2'
        manifest = json.loads((fixture/'manifest.lock.json').read_text(encoding='utf8'))
        if sha(fixture/'answers.jsonl') != manifest['files']['answers.jsonl'] or sha(fixture/'manifest.lock.json') != selection['fixture_lock_sha256']:
            raise ValueError('FAQ answer corpus changed')
        answers = sorted(read_jsonl(fixture/'answers.jsonl'), key=lambda a:int(a['id']))
        original = json.loads((ROOT/'reports/insuranceqa_v2/rerank_valid_v2/protocol.json').read_text(encoding='utf8'))
        embedding_protocol = original['embedding']
        vectors_path = ROOT/'reports/insuranceqa_v2/retrieval_frozen/answer_embeddings.npy'
        expected_vectors_sha = original['embedding_cache_sha256']['answer_embeddings.npy']
    else:
        folder = ROOT/'data/research_corpus/hicric_public_v1'
        protocol_path = args.index/'protocol.json'
        protocol = json.loads(protocol_path.read_text(encoding='utf8'))
        completion = json.loads((args.index/'completion.json').read_text(encoding='utf8'))
        if completion['status'] != 'completed' or sha(protocol_path) != completion['protocol_sha256']:
            raise ValueError('Incomplete index')
        if sha(folder/'rag_snippets.jsonl') != protocol['records_sha256']:
            raise ValueError('Snippet index does not describe this corpus')
        answers = [{**r, 'id':r['record_id']} for r in read_jsonl(folder/'rag_snippets.jsonl')]
        embedding_protocol = protocol['embedding']; vectors_path = args.index/'answer_embeddings.npy'
        expected_vectors_sha = completion['answer_embeddings_sha256']
    if sha(vectors_path) != expected_vectors_sha: raise ValueError('Index vectors changed')
    dense = np.load(vectors_path)
    if dense.shape[0] != len(answers): raise ValueError('Index record count mismatch')
    embedding = EmbeddingRetriever(str(args.embedding_model),use_hf_api=False,pooling='cls',max_length=512,
                                  query_instruction=embedding_protocol['query_instruction'])
    verify_files(embedding.index_fingerprint()['checkpoint_files_sha256'],embedding_protocol['checkpoint_files_sha256'])
    _, model = embedding._ensure_local_transformer(); model.to(args.device)
    query = embedding.embed_texts([embedding.query_instruction+args.question])[0]
    d = dense @ query; sparse = SparseBM25([a['text'] for a in answers]); b = sparse.scores(args.question)
    do = ranked(d); bo = ranked(b,positive_only=True); union = sorted(set(do)|set(bo))
    ce = DomainCrossEncoder(args.reranker_model,args.device)
    verify_files(ce.fingerprint()['files_sha256'],scored_model['files_sha256'])
    cross = ce.score(args.question,[answers[j]['text'] for j in union])
    row = {'candidate_ids':[answers[j]['id'] for j in union], 'dense':d[union].tolist(),
           'sparse':b[union].tolist(),'cross':cross.tolist(),'bge_ids':[answers[j]['id'] for j in do]}
    order = rank_row(row,selection['config']); lookup = {a['id']:a for a in answers}
    result = {'question':args.question,'corpus':args.corpus,'answer_candidates':len(answers),
        'retrieval_only':True,'generation_calls':0,'fine_tuned_in_this_project':args.profile=='trained',
        'profile':args.profile,'selected_epoch':selection.get('epoch'),'config':selection['config'],'candidates_scored':len(union),
        'seconds_including_loading':time.perf_counter()-start,
        'scope':'Archived public research evidence; snippet search is not evaluated policy reasoning',
        'results':[{'rank':i+1,'answer_id':aid,'text':lookup[aid]['text'],
                    **{k:lookup[aid][k] for k in ['source_url','doc_id','start_word','end_word','page_unit'] if k in lookup[aid]}}
                   for i,aid in enumerate(order[:args.top_k])]}
    text = json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output: args.output.write_text(text,encoding='utf8')
    print(text)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--question',required=True);p.add_argument('--embedding-model',type=Path,required=True)
    p.add_argument('--reranker-model',type=Path,required=True)
    p.add_argument('--selection',type=Path,default=ROOT/'reports/domain_training_v2/selection.lock.json')
    p.add_argument('--profile',choices=['trained','general'],default='trained',
                   help='Explicit model choice; no automatic query classifier or claim of validated routing')
    p.add_argument('--general-selection',type=Path,default=ROOT/'reports/insuranceqa_v2/rerank_selection_v2/selection.lock.json')
    p.add_argument('--corpus',choices=['insuranceqa','hicric'],default='insuranceqa')
    p.add_argument('--index',type=Path,default=ROOT/'reports/hicric_public_v1/bge_index')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--top-k',type=int,default=5)
    p.add_argument('--output',type=Path)
    run(p.parse_args())
