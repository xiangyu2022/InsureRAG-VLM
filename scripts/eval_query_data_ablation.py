"""Same-budget retrieval evaluation; select queries on validation, then freeze test."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json
from scripts.eval_insuranceqa_scale import SparseBM25, ranked, rrf, metrics, aggregate
from scripts.eval_insuranceqa_reranker import rank_row
from src.insurerag_vlm.query_adaptation import QueryEncoder, blend_queries
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder

RUN = ROOT / 'reports/query_adaptation_v1/data_ablation'
FIXED_RERANKER = ROOT / '../models/insurerag-condition-listwise-v1-seed-123/epoch-1'
CFG = {'pool': 'union200', 'lexical_weight': .2, 'cross_weight': .5}


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def measure(order, gold, candidates):
    result = metrics(order, gold)
    gains = [1/np.log2(i+2) for i, a in enumerate(order[:10]) if a in gold]
    ideal = sum(1/np.log2(i+2) for i in range(min(10, len(gold))))
    return {**result, 'ndcg_at_10': float(sum(gains)/ideal) if ideal else 0.,
            'candidate_hit': float(bool(set(candidates) & gold))}


def summarize(rows):
    return {**aggregate(rows), **{k: float(np.mean([r[k] for r in rows])) for k in ['ndcg_at_10', 'candidate_hit']}}


def configurations(plan):
    configs = [{'name': 'original', 'model': 'original', 'alpha': 0.}]
    for seed in plan['seeds']:
        for epoch in plan['epochs_considered']:
            for alpha in plan['query_blend_alphas']:
                configs.append({'name': f'seed_{seed}_epoch_{epoch}_a{round(alpha*100)}',
                                'model': f'seed_{seed}_epoch_{epoch}', 'seed': seed, 'epoch': epoch, 'alpha': alpha})
    return configs


def modelpath(config):
    if config['model'] == 'original': return ROOT / '../models/bge-small-en-v1.5'
    return ROOT / f'../models/insurerag-query-ablation-v1-seed-{config["seed"]}/epoch-{config["epoch"]}'


def specs(split):
    common = [
        ('insuranceqa', f'data/benchmarks/insuranceqa_v2/{split}.jsonl', 'data/benchmarks/insuranceqa_v2/answers.jsonl',
         'valid_legacy_seed_123' if split == 'valid' else 'test_legacy_selected'),
        ('multidomain', f'data/benchmarks/multidomain_v1/{split}.jsonl', 'data/training/retention_v2/answers.jsonl',
         'valid_mixed_seed_123' if split == 'valid' else 'test_mixed_selected'),
        ('fiqa', f'data/benchmarks/fiqa_v1/{split}.jsonl', 'data/benchmarks/fiqa_v1/answers.jsonl', None),
    ]
    if split == 'test':
        common += [
            ('historical_government', 'data/benchmarks/hicric_government_qa_v1/questions.jsonl',
             'reports/condition_listwise_v1/historical_government_answers.jsonl', 'test_oldgov_selected'),
            ('condition_government', 'data/benchmarks/condition_v1/test.jsonl', 'data/benchmarks/condition_v1/answers.jsonl', 'test_fresh_selected'),
        ]
    return common


def evaluation_hashes():
    names = ['scripts/eval_query_data_ablation.py', 'scripts/analyze_condition_secondary_metrics.py',
             'scripts/prepare_insuranceqa.py', 'src/insurerag_vlm/retriever.py']
    return {name: sha(ROOT / name) for name in names}


def check_contract():
    plan = load(RUN / 'selection_protocol.json')
    for name, digest in plan['frozen_code_sha256'].items(): assert sha(ROOT / name) == digest, name
    implementation = load(RUN / 'evaluation_implementation.lock.json')
    assert implementation['code_sha256'] == evaluation_hashes()
    assert implementation['selection_protocol_sha256'] == sha(RUN / 'selection_protocol.json')
    for name, digest in implementation['fixture_files_sha256'].items(): assert sha(ROOT / name) == digest, name
    assert sha(FIXED_RERANKER / 'model.safetensors') == plan['fixed_reranker_weights_sha256']
    return plan


def freeze():
    path = RUN / 'evaluation_implementation.lock.json'
    if path.exists() or (RUN / 'valid').exists(): raise ValueError('Evaluation already frozen/started')
    files = {p for split in ['valid', 'test'] for _, q, a, _ in specs(split) for p in [q, a]}
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(),
                'selection_protocol_sha256': sha(RUN / 'selection_protocol.json'),
                'code_sha256': evaluation_hashes(), 'fixture_files_sha256': {p: sha(ROOT / p) for p in sorted(files)},
                'before_first_validation_or_new_test_scoring': True}, path)


def prior_scores(folder, questions_path, answers_path, plan, cohort):
    primary = ROOT / 'reports/query_adaptation_v1'
    scores = primary / 'valid' / f'{cohort}_scores.jsonl'
    summary = load(primary / 'valid/summary.json')
    assert sha(scores) == summary['score_files_sha256'][scores.name]
    protocol = load(primary / 'valid/protocol.json')
    assert protocol['fixed_reranker_weights_sha256'] == plan['fixed_reranker_weights_sha256']
    fixtures = load(primary / 'evaluation_implementation.lock.json')['fixture_files_sha256']
    assert fixtures[questions_path.relative_to(ROOT).as_posix()] == sha(questions_path)
    assert fixtures[answers_path.relative_to(ROOT).as_posix()] == sha(answers_path)
    rows = read_jsonl(scores)
    assert [r['id'] for r in rows] == [c['id'] for c in read_jsonl(questions_path)]
    return {r['id']: dict(zip(r['candidate_ids'], r['cross'])) for r in rows}


def partitions(cohort, cases):
    if cohort == 'multidomain':
        return {domain: [i for i, c in enumerate(cases) if c['domain'] == domain] for domain in ['government', 'general']}
    return {'all': list(range(len(cases)))}


def paired(new, old, cases, cohort):
    # Source clusters where available; otherwise connected shared-gold components.
    from scripts.analyze_condition_secondary_metrics import clusters
    buckets = clusters(cases, shared_labels=cohort in {'insuranceqa', 'fiqa'})
    sizes = np.array([len(b) for b in buckets])
    draws = np.random.default_rng(20261002).integers(0, len(buckets), (5000, len(buckets)))
    result = {'n': len(cases), 'clusters': len(buckets), 'bootstrap_repetitions': 5000,
              'cluster_unit': 'shared-gold connected components' if cohort in {'insuranceqa', 'fiqa'} else 'source URL/title',
              'multiple_comparisons_adjusted': False, 'metrics': {}}
    for metric in ['hit_at_1', 'hit_at_10', 'mrr_at_100', 'ndcg_at_10', 'candidate_hit']:
        delta = np.array([a[metric]-b[metric] for a, b in zip(new, old)])
        sums = np.array([delta[b].sum() for b in buckets])
        boot = sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
        result['metrics'][metric] = {'difference': float(delta.mean()), 'cluster_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(),
                                     'wins': int((delta>0).sum()), 'losses': int((delta<0).sum())}
    return result


def score(split):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = check_contract()
    configs = configurations(plan)
    selection = None
    if split == 'test':
        selection = load(RUN / 'selection.lock.json')
        assert selection['selected_on'] == 'valid'
        assert selection['selection_protocol_sha256'] == sha(RUN / 'selection_protocol.json')
        configs = [configs[0]] + ([selection['config']] if selection['config']['name'] != 'original' else [])
        assert sha(modelpath(selection['config']) / 'model.safetensors') == selection['query_weights_sha256']
    out = RUN / split
    out.mkdir(exist_ok=False)
    data = ROOT / 'data/training/query_adaptation_v1'
    allanswers = read_jsonl(data / 'answers.jsonl')
    globalmap = {a['id']: i for i, a in enumerate(allanswers)}
    cache = ROOT / 'reports/query_adaptation_v1/index_cache'
    index_manifest = load(cache / 'manifest.json')
    assert sha(cache / 'answer_embeddings.npy') == index_manifest['answer_embeddings_sha256']
    alldense = np.load(cache / 'answer_embeddings.npy')
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'split': split, 'configs': configs,
                'selection_protocol_sha256': sha(RUN / 'selection_protocol.json'),
                'evaluation_implementation_sha256': sha(RUN / 'evaluation_implementation.lock.json'),
                'selection_lock_sha256': sha(RUN / 'selection.lock.json') if selection else None,
                'fixed_reranker_weights_sha256': plan['fixed_reranker_weights_sha256'],
                'query_models_sha256': {c['model']: sha(modelpath(c) / 'model.safetensors') for c in configs},
                'code_sha256': evaluation_hashes(), 'candidate_budget': 'dense top100 union positive-BM25 top100',
                'gold_injection': False, 'answer_encoder_changed': False}
    write_json(protocol, out / 'protocol.json')
    predictions = defaultdict(lambda: defaultdict(list))
    summaries, paired_results, case_sets = {}, {}, {}
    start = time.perf_counter()
    new_cross = reused_cross = 0
    for cohort, question_file, answer_file, previous_folder in specs(split):
        cases = read_jsonl(ROOT / question_file)
        answers = read_jsonl(ROOT / answer_file)
        case_sets[cohort] = cases
        ids = [a['id'] for a in answers]
        idset = set(ids)
        position = {a: j for j, a in enumerate(ids)}
        dense = np.empty((len(answers), 384), dtype=np.float32)
        missing = []
        for i, a in enumerate(answers):
            j = globalmap.get(a['id'])
            if j is not None:
                assert allanswers[j]['text'] == a['text']
                dense[i] = alldense[j]
            else: missing.append(i)
        if missing:
            from src.insurerag_vlm.retriever import EmbeddingRetriever
            encoder = EmbeddingRetriever(str(ROOT / '../models/bge-small-en-v1.5'), use_hf_api=False,
                                         pooling='cls', max_length=512)
            for offset in range(0, len(missing), 128):
                ix = missing[offset:offset+128]
                dense[ix] = encoder.embed_texts([answers[i]['text'] for i in ix])
            np.save(out / f'{cohort}_missing_document_vectors.npy', dense[missing])
            write_json({'answer_ids': [ids[i] for i in missing], 'encoder': encoder.index_fingerprint()}, out / f'{cohort}_missing_documents.json')
            del encoder
            torch.cuda.empty_cache()
        vectors = {}
        for config in configs:
            if config['model'] in vectors: continue
            if config['model'] != 'original':
                training = load(modelpath(config) / 'query_adapter_config.json')
                assert training['weights_sha256'] == protocol['query_models_sha256'][config['model']]
                assert training['document_encoder_weights_sha256'] == plan['initial_query_weights_sha256']
            encoder = QueryEncoder(modelpath(config), 'cuda')
            vectors[config['model']] = encoder.encode([c['question'] for c in cases])
            np.save(out / f'{cohort}_{config["model"]}_queries.npy', vectors[config['model']])
            del encoder
            torch.cuda.empty_cache()
        blended = {c['name']: vectors['original'] if c['name'] == 'original'
                   else blend_queries(vectors['original'], vectors[c['model']], c['alpha']) for c in configs}
        sparse = SparseBM25([a['text'] for a in answers])
        csc = sparse.matrix.tocsc()
        old = prior_scores(previous_folder, ROOT / question_file, ROOT / answer_file, plan, cohort)
        reranker = DomainCrossEncoder(FIXED_RERANKER, 'cuda', 64, 512)
        with (out / f'{cohort}_scores.jsonl').open('w', encoding='utf8') as handle:
            for offset in range(0, len(cases), 8):
                batch = cases[offset:offset+8]
                products = {name: dense @ vec[offset:offset+8].T for name, vec in blended.items()}
                rows, score_pairs, pair_keys = [], [], []
                for k, case in enumerate(batch):
                    query = sparse.vectorizer.transform([case['question']]).tocsr()
                    b = np.asarray(csc[:, query.indices] @ query.data).ravel()
                    bo = ranked(b, positive_only=True)
                    row = {'id': case['id'], 'bm25_ids': [ids[j] for j in bo], 'arms': {}}
                    union_all = set()
                    for name, product in products.items():
                        d = product[:, k]
                        do = ranked(d)
                        union = sorted(set(do) | set(bo))
                        row['arms'][name] = {'candidate_indices': union, 'bge_ids': [ids[j] for j in do],
                                             'dense': d[union].tolist(), 'sparse': b[union].tolist()}
                        union_all.update(union)
                    row['candidate_indices'] = sorted(union_all)
                    row['cross_cache'] = {j: old[case['id']][ids[j]] for j in union_all if ids[j] in old.get(case['id'], {})}
                    reused_cross += len(row['cross_cache'])
                    for j in row['candidate_indices']:
                        if j not in row['cross_cache']:
                            pair_keys.append((len(rows), j))
                            score_pairs.append((case['question'], answers[j]['text']))
                    rows.append(row)
                scores = reranker.score_pairs(score_pairs)
                new_cross += len(scores)
                for (r, j), value in zip(pair_keys, scores): rows[r]['cross_cache'][j] = float(value)
                for case, row in zip(batch, rows):
                    gold = set(case['gold_answer_ids'])
                    assert gold <= idset
                    serialized = {'id': case['id'], 'candidate_ids': [ids[j] for j in row['candidate_indices']],
                                  'cross': [row['cross_cache'][j] for j in row['candidate_indices']], 'arms': {}}
                    for name, arm in row['arms'].items():
                        candidate_ids = [ids[j] for j in arm['candidate_indices']]
                        score_row = {'candidate_ids': candidate_ids, 'bge_ids': arm['bge_ids'],
                                     'dense': arm['dense'], 'sparse': arm['sparse'],
                                     'cross': [row['cross_cache'][j] for j in arm['candidate_indices']]}
                        order = rank_row(score_row, CFG)
                        predictions[cohort][name].append({'id': case['id'], 'arm': name,
                            'top_answer_ids': order, **measure(order, gold, candidate_ids)})
                        serialized['arms'][name] = {k: v for k, v in score_row.items() if k != 'cross'}
                        if name == 'original' or split == 'test':
                            dense_name = 'bge' if name == 'original' else 'adapted_dense'
                            predictions[cohort][dense_name].append({'id': case['id'], 'arm': dense_name,
                                'top_answer_ids': arm['bge_ids'], **measure(arm['bge_ids'], gold, arm['bge_ids'])})
                            # Reciprocal-rank fusion over the same fixed retrieval depths.
                            rank_scores = defaultdict(float)
                            for order0 in [arm['bge_ids'], row['bm25_ids']]:
                                for rank, aid in enumerate(order0, 1): rank_scores[aid] += 1/(60+rank)
                            ro = sorted(rank_scores, key=lambda a: (-rank_scores[a], position[a]))[:100]
                            rn = 'rrf' if name == 'original' else 'adapted_rrf'
                            predictions[cohort][rn].append({'id': case['id'], 'arm': rn,
                                'top_answer_ids': ro, **measure(ro, gold, candidate_ids)})
                    predictions[cohort]['bm25'].append({'id': case['id'], 'arm': 'bm25', 'top_answer_ids': row['bm25_ids'],
                                                         **measure(row['bm25_ids'], gold, row['bm25_ids'])})
                    handle.write(json.dumps(serialized, ensure_ascii=True)+'\n')
                handle.flush()
                if (offset+len(batch)) % 200 == 0 or offset+len(batch) == len(cases):
                    print(json.dumps({'split': split, 'cohort': cohort, 'questions': offset+len(batch),
                                      'total': len(cases), 'new_cross_pairs': new_cross,
                                      'seconds': round(time.perf_counter()-start, 1)}), flush=True)
        del reranker
        torch.cuda.empty_cache()
        summaries[cohort] = {}
        paired_results[cohort] = {}
        for part, ix in partitions(cohort, cases).items():
            summaries[cohort][part] = {name: summarize([rows[i] for i in ix]) for name, rows in predictions[cohort].items()}
            if split == 'test':
                chosen = selection['config']['name']
                subset = [cases[i] for i in ix]
                new = [predictions[cohort][chosen][i] for i in ix]
                paired_results[cohort][part] = {name: paired(new, [predictions[cohort][name][i] for i in ix], subset, cohort)
                                               for name in ['original', 'bge', 'rrf']}
    with (out / 'predictions.jsonl').open('w', encoding='utf8') as handle:
        for cohort, arms in predictions.items():
            for rows in arms.values():
                for row in rows: handle.write(json.dumps({'cohort': cohort, **row})+'\n')
    check_contract()
    report = {'status': 'completed', 'split': split, 'summaries': summaries,
              'paired_selected_minus_controls': paired_results, 'new_cross_pairs': new_cross, 'reused_cross_pairs': reused_cross,
              'seconds': time.perf_counter()-start, 'protocol_sha256': sha(out / 'protocol.json'),
              'predictions_sha256': sha(out / 'predictions.jsonl'),
              'score_files_sha256': {p.name: sha(p) for p in out.glob('*_scores.jsonl')},
              'query_vector_files_sha256': {p.name: sha(p) for p in out.glob('*_queries.npy')}}
    write_json(report, out / 'summary.json')
    assert split == 'valid', 'Data ablation must not score tests or select a model'
    print(json.dumps({'completed': split, 'seconds': round(report['seconds'], 1), 'new_cross_pairs': new_cross}), flush=True)


def select(*args):
    raise ValueError('Data ablation cannot select or promote a model')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['freeze', 'valid'])
    args = p.parse_args()
    freeze() if args.phase == 'freeze' else score(args.phase)
