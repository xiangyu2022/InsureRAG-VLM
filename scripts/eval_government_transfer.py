"""One locked evaluation on publisher-written government Q/A, never used for tuning."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, write_json, sha, verify_fixture
from scripts.prepare_hicric_extension import clean
from scripts.eval_insuranceqa_scale import SparseBM25, ranked, metrics, aggregate
from scripts.eval_insuranceqa_reranker import rank_row, paired_summary
from src.insurerag_vlm.retriever import EmbeddingRetriever
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder


def audit_sources(cases, answers, records, train):
    """Verify labels by exact publisher spans, without model predictions."""
    lookup = {r['id']: r for r in records}; answer_map = {r['id']: r for r in answers}
    for case in cases:
        raw = lookup[case['source_record_id']]['text']
        if clean(raw[slice(*case['question_span'])]) != case['question']:
            raise ValueError('Question does not match original publisher span')
        if clean(raw[slice(*case['answer_span'])]) != answer_map[case['gold_answer_ids'][0]]['text']:
            raise ValueError('Answer label does not match original publisher span')
        if case['source_url'] not in {p['source_url'] for p in lookup[case['source_record_id']]['provenance']}:
            raise ValueError('Incorrect publisher attribution')
    train_norm = {clean(r['question']).lower() for r in train}
    overlaps = [c['id'] for c in cases if clean(c['question']).lower() in train_norm]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2)
    matrix = vectorizer.fit_transform([r['question'] for r in train]+[c['question'] for c in cases])
    nn = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=4).fit(matrix[:len(train)])
    distances, indices = nn.kneighbors(matrix[len(train):])
    near = [{'id': c['id'], 'train_id': train[int(j[0])]['id'], 'cosine': float(1-d[0])}
            for c,d,j in zip(cases, distances, indices) if 1-d[0] >= .92]
    return {'verified_source_spans': len(cases), 'exact_training_question_overlaps': overlaps,
            'near_training_question_overlaps': near, 'near_rule': 'char_wb TF-IDF 3-5 grams, cosine >= 0.92',
            'maximum_train_question_similarity': float(1-distances.min()),
            'expert_relevance_adjudication': False, 'post_evaluation_case_removal': False}


def run(args):
    import torch
    torch.set_num_threads(4)
    benchmark = ROOT/'data/benchmarks/hicric_government_qa_v1'
    corpus = ROOT/'data/research_corpus/hicric_public_v1'
    lock = json.loads((benchmark/'manifest.lock.json').read_text(encoding='utf8'))
    for name, expected in lock['files'].items():
        if sha(benchmark/name) != expected: raise ValueError('Government benchmark changed')
    manifest = json.loads((corpus/'manifest.json').read_text(encoding='utf8'))
    if sha(corpus/'manifest.json') != lock['corpus_manifest_sha256'] or sha(corpus/'records.jsonl') != manifest['files']['records.jsonl']:
        raise ValueError('Original source records changed')
    selection = json.loads(args.selection.read_text(encoding='utf8'))
    if selection['selected_on'] != 'valid' or sha(args.trained_model/'model.safetensors') != selection['selected_weights_sha256']:
        raise ValueError('Transfer evaluation requires frozen validation-selected weights')
    if sha(args.training_data/'manifest.lock.json') != selection['training_manifest_sha256']:
        raise ValueError('Training audit fixture does not describe the selected checkpoint')
    training_lock = json.loads((args.training_data/'manifest.lock.json').read_text(encoding='utf8'))
    for name, expected in training_lock['files'].items():
        if sha(args.training_data/name) != expected: raise ValueError('Training audit fixture changed')
    verify_fixture(ROOT/'data/benchmarks/insuranceqa_v2')
    cases = read_jsonl(benchmark/'questions.jsonl'); government = read_jsonl(benchmark/'answers.jsonl')
    insurance = sorted(read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/answers.jsonl'), key=lambda a:int(a['id']))
    answers = insurance+government; ids = [a['id'] for a in answers]
    if len(ids) != len(set(ids)): raise ValueError('Colliding answer IDs')
    args.output.mkdir(parents=True, exist_ok=False)
    audit = audit_sources(cases, government, read_jsonl(corpus/'records.jsonl'),
                          read_jsonl(args.training_data/'train_groups.jsonl'))
    write_json(audit, args.output/'source_audit.json')
    original = json.loads((ROOT/'reports/insuranceqa_v2/rerank_valid_v2/protocol.json').read_text(encoding='utf8'))
    embeddings_path = ROOT/'reports/insuranceqa_v2/retrieval_frozen/answer_embeddings.npy'
    if sha(embeddings_path) != original['embedding_cache_sha256']['answer_embeddings.npy']:
        raise ValueError('InsuranceQA embedding cache changed')
    embedding = EmbeddingRetriever(str(args.embedding_model), use_hf_api=False, pooling='cls', max_length=512,
                                  query_instruction=original['embedding']['query_instruction'])
    strip = lambda d: {k:v for k,v in d.items() if k != 'download_provenance.json'}
    if strip(embedding.index_fingerprint()['checkpoint_files_sha256']) != strip(original['embedding']['checkpoint_files_sha256']):
        raise ValueError('BGE checkpoint mismatch')
    base = DomainCrossEncoder(args.base_model, 'cuda', 64, 512)
    trained = DomainCrossEncoder(args.trained_model, 'cuda', 64, 512)
    if strip(base.fingerprint()['files_sha256']) != strip(original['reranker']['files_sha256']):
        raise ValueError('Original reranker checkpoint mismatch')
    files = [Path(__file__), ROOT/'scripts/eval_insuranceqa_reranker.py', ROOT/'scripts/eval_insuranceqa_scale.py',
             ROOT/'src/insurerag_vlm/reranker.py', ROOT/'src/insurerag_vlm/domain_reranker.py', ROOT/'src/insurerag_vlm/retriever.py']
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    previous = json.loads((ROOT/'reports/insuranceqa_v2/rerank_selection_v2/selection.lock.json').read_text(encoding='utf8'))['config']
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'questions': len(cases),
        'answer_candidates': len(answers), 'government_answers': len(government), 'insuranceqa_distractors': len(insurance),
        'selection_lock_sha256': sha(args.selection), 'benchmark_lock_sha256': sha(benchmark/'manifest.lock.json'),
        'source_audit_sha256': sha(args.output/'source_audit.json'), 'code_sha256': code,
        'training_manifest_sha256': sha(args.training_data/'manifest.lock.json'),
        'trained_config': selection['config'], 'previous_config': previous,
        'trained_model': trained.fingerprint(), 'base_model': base.fingerprint(), 'embedding': embedding.index_fingerprint(),
        'candidate_policy': 'Shared union of BGE top100 and positive BM25 top100 over all 27,829 answers; no label injection',
        'primary_metric': 'Hit@10', 'trained_on_these_cases': False, 'tuning_on_these_cases': False,
        'generation_calls': 0, 'cluster_unit': 'publisher source URL',
        'limitations': ['Publisher Q/A association is mechanically extracted, not expert relevance adjudication.',
                       'Archived historical guidance; candidate answers may omit context or contain source extraction artifacts.',
                       'Public pretrained models may have encountered government source texts.',
                       'FAQ answer retrieval, not multi-page policy or legal reasoning.']}
    write_json(protocol, args.output/'protocol.json'); start = time.perf_counter()
    _, encoder = embedding._ensure_local_transformer(); encoder.to('cuda')
    def encode(texts):
        return np.vstack([embedding.embed_texts(texts[i:i+128]) for i in range(0,len(texts),128)])
    dense = np.vstack([np.load(embeddings_path), encode([a['text'] for a in government])])
    queries = encode([embedding.query_instruction+c['question'] for c in cases])
    np.save(args.output/'answer_embeddings.npy', dense); np.save(args.output/'query_embeddings.npy', queries)
    encoder.to('cpu'); sparse = SparseBM25([a['text'] for a in answers]); scored = []
    with (args.output/'scores.jsonl').open('w', encoding='utf8') as handle:
        for i, case in enumerate(cases):
            d = dense @ queries[i]; b = sparse.scores(case['question'])
            do = ranked(d); bo = ranked(b, positive_only=True); union = sorted(set(do)|set(bo))
            texts = [answers[j]['text'] for j in union]
            row = {'id':case['id'], 'candidate_ids':[ids[j] for j in union], 'dense':d[union].tolist(),
                   'sparse':b[union].tolist(), 'bge_ids':[ids[j] for j in do], 'bm25_ids':[ids[j] for j in bo],
                   'cross_base':base.score(case['question'], texts).tolist(),
                   'cross_trained':trained.score(case['question'], texts).tolist()}
            scored.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
            if (i+1)%100 == 0: print(json.dumps({'government_questions':i+1,'seconds':round(time.perf_counter()-start,1)}), flush=True)
    arms = {k:[] for k in ['bge','bm25','previous_untrained','untrained_same_fusion','trained_selected','trained_bge_candidates']}
    for row, case in zip(scored,cases):
        orders = {'bge':row['bge_ids'], 'bm25':row['bm25_ids'],
            'previous_untrained':rank_row({**row,'cross':row['cross_base']}, previous),
            'untrained_same_fusion':rank_row({**row,'cross':row['cross_base']}, selection['config']),
            'trained_selected':rank_row({**row,'cross':row['cross_trained']}, selection['config']),
            'trained_bge_candidates':rank_row({**row,'cross':row['cross_trained']},
                                            {**selection['config'],'pool':'bge100','lexical_weight':0.})}
        for arm, order in orders.items():
            arms[arm].append({'id':case['id'],'arm':arm,'source_url':case['source_url'],
                              'top_answer_ids':order, **metrics(order,set(case['gold_answer_ids']))})
    cluster_cases = [{**c, 'gold_answer_ids':[c['source_url']]} for c in cases]
    paired = {}
    for arm in ['bge','previous_untrained','untrained_same_fusion','trained_bge_candidates']:
        result = paired_summary(arms['trained_selected'], arms[arm], cluster_cases)
        result['source_cluster_bootstrap_95'] = result.pop('label_cluster_bootstrap_95')
        result['question_level_exact_binomial_p'] = result.pop('paired_exact_binomial_p')
        result['interpretation'] = 'Source-URL clusters account for correlated questions; 37 public sources, no expert alternate-relevance labels.'
        paired[arm] = result
    with (args.output/'predictions.jsonl').open('w', encoding='utf8') as handle:
        for values in arms.values():
            for row in values: handle.write(json.dumps(row)+'\n')
    if code != {p.relative_to(ROOT).as_posix():sha(p) for p in files}: raise ValueError('Evaluation code changed')
    summary = {'status':'completed', 'seconds':time.perf_counter()-start,
        'summaries':{arm:aggregate(values) for arm,values in arms.items()}, 'trained_minus_comparator':paired,
        'protocol_sha256':sha(args.output/'protocol.json'), 'scores_sha256':sha(args.output/'scores.jsonl'),
        'predictions_sha256':sha(args.output/'predictions.jsonl')}
    write_json(summary,args.output/'summary.json'); print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embedding-model',type=Path,required=True); p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--trained-model',type=Path,required=True); p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--training-data',type=Path,default=ROOT/'data/training/insuranceqa_hardneg_v2')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
