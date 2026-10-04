"""Decontaminate author train questions and mine BGE/BM25 negatives without test fitting."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, write_json, sha, verify_fixture
from scripts.eval_insuranceqa_scale import SparseBM25, ranked
from src.insurerag_vlm.retriever import EmbeddingRetriever


def normalize(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def run(args):
    verify_fixture(args.fixture)
    args.output.mkdir(parents=True, exist_ok=False)
    train = read_jsonl(args.fixture/'train.jsonl')
    heldout = read_jsonl(args.fixture/'valid.jsonl') + read_jsonl(args.fixture/'test.jsonl')
    answers = sorted(read_jsonl(args.fixture/'answers.jsonl'), key=lambda a: int(a['id']))
    answer_map = {a['id']: a for a in answers}
    heldout_labels = {a for c in heldout for a in c['gold_answer_ids']}
    # Held-out content is used solely for exclusion, never as a fitting example.
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2)
    vectors = vectorizer.fit_transform([c['question'] for c in heldout+train])
    nearest = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute', n_jobs=2).fit(vectors[:len(heldout)])
    distance, neighbor = nearest.kneighbors(vectors[len(heldout):])
    clean = []; excluded = []; seen = set()
    for i, c in enumerate(train):
        reasons = []
        if 1-distance[i,0] >= .92: reasons.append('heldout_question_char_similarity_ge_0.92')
        if set(c['gold_answer_ids']) & heldout_labels: reasons.append('heldout_positive_answer_id_overlap')
        key = normalize(c['question'])
        if key in seen: reasons.append('duplicate_training_question')
        seen.add(key)
        if reasons:
            excluded.append({'id': c['id'], 'reasons': reasons, 'nearest_heldout_id': heldout[int(neighbor[i,0])]['id'],
                             'similarity': float(1-distance[i,0])})
        else:
            clean.append(c)
    write_json({'original_train': len(train), 'retained_train': len(clean), 'excluded_train': len(excluded),
                'exclusion_reasons': dict(Counter(r for c in excluded for r in c['reasons'])),
                'rule': 'Question char_wb 3-5gram cosine >= .92, any shared heldout positive answer ID, or repeated normalized train question',
                'heldout_used_only_for_exclusion': True, 'exclusions': excluded}, args.output/'decontamination.json')
    print(json.dumps({'retained_train_questions': len(clean), 'excluded': len(excluded)}), flush=True)
    retriever = EmbeddingRetriever(str(args.model), use_hf_api=False, pooling='cls', max_length=512,
                                  query_instruction='Represent this sentence for searching relevant passages: ')
    import torch
    torch.set_num_threads(4); threadpool_limits(4)
    _, model = retriever._ensure_local_transformer(); model.to(args.device)
    queries = []
    for start in range(0, len(clean), 256):
        queries.append(retriever.embed_texts([retriever.query_instruction+c['question'] for c in clean[start:start+256]]))
    queries = np.vstack(queries); dense = np.load(args.baseline/'answer_embeddings.npy')
    np.save(args.output/'train_query_embeddings.npy', queries)
    sparse = SparseBM25([a['text'] for a in answers]); groups = []; rng = np.random.default_rng(20261001)
    answer_norm = {a['id']: normalize(a['text']) for a in answers}
    start = time.perf_counter()
    for i, c in enumerate(clean):
        d = dense @ queries[i]; b = sparse.scores(c['question'])
        dr = ranked(d); br = ranked(b, positive_only=True)
        positives = set(c['gold_answer_ids']); positive_texts = {answer_norm[x] for x in positives}
        negatives = []; source = {}; negative_texts = set()
        # Equal mining budgets add lexical distractors, not extra fusion weight.
        for channel, order in [('dense_hard', dr), ('lexical_hard', br)]:
            taken = 0
            for index in order:
                aid = answers[index]['id']; normalized = answer_norm[aid]
                if aid in positives or normalized in positive_texts or normalized in negative_texts:
                    continue
                negatives.append(aid); negative_texts.add(normalized); source[aid] = channel; taken += 1
                if taken == 4: break
        for index in rng.permutation(len(answers)):
            aid = answers[index]['id']; normalized = answer_norm[aid]
            if aid not in positives and normalized not in positive_texts and normalized not in negative_texts:
                negatives.append(aid); source[aid] = 'random'; break
        groups.append({'id': c['id'], 'question': c['question'], 'domain': c['domain'],
                       'positive_ids': sorted(positives, key=int), 'negative_ids': negatives,
                       'negative_sources': source})
        if (i+1) % 1000 == 0: print(json.dumps({'mined_questions': i+1, 'total': len(clean), 'seconds': time.perf_counter()-start}), flush=True)
    with (args.output/'train_groups.jsonl').open('w', encoding='utf8') as handle:
        for g in groups: handle.write(json.dumps(g)+'\n')
    file_names = ['train_groups.jsonl', 'decontamination.json']
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'version': 'insuranceqa_hardneg_v1',
        'fixture_lock_sha256': sha(args.fixture/'manifest.lock.json'), 'mining_code_sha256': sha(Path(__file__)),
        'embedding': retriever.index_fingerprint(), 'answer_embeddings_sha256': sha(args.baseline/'answer_embeddings.npy'),
        'original_train_questions': len(train), 'training_questions': len(groups),
        'labeled_positive_pairs': sum(len(g['positive_ids']) for g in groups),
        'mined_negative_pairs': sum(len(g['negative_ids']) for g in groups),
        'negative_types': dict(Counter(t for g in groups for t in g['negative_sources'].values())),
        'domains': dict(Counter(g['domain'] for g in groups)), 'seed': 20261001,
        'test_used_for_hyperparameters': False, 'heldout_used_for_split_hygiene_only': True,
        'positive_annotation': 'Original InsuranceQA authors, unchanged answer IDs',
        'negative_annotation': 'Unlabeled retrieved distractors, not human-verified incorrect answers',
        'limitations': ['Plausible unlabeled answers can be false negatives.',
                       'Lexical decontamination is not complete semantic decontamination.',
                       'Pair count does not equal independent question count.'],
        'use_terms': 'InsuranceQA original research-only use terms retained',
        'files': {name: sha(args.output/name) for name in file_names}}
    write_json(protocol, args.output/'manifest.lock.json')
    print(json.dumps({k: v for k, v in protocol.items() if k not in ['embedding', 'files']}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, default=ROOT/'data/benchmarks/insuranceqa_v2')
    p.add_argument('--baseline', type=Path, default=ROOT/'reports/insuranceqa_v2/retrieval_frozen')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
