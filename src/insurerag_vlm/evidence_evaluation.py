"""Strict retrieval/packing metrics with fixed question and publisher denominators.

These measure released relevance labels and literal retention, not semantic
support or policy adjudication. A single hit is deliberately not called recall.
"""
import math
import re
from collections import Counter, defaultdict


def normalize_space(text):
    return ' '.join(str(text).split())


def retrieval_metrics(order, gold, cutoffs=(1, 5, 10)):
    if not gold or len(set(gold)) != len(gold):
        raise ValueError('Gold evidence must be nonempty and unique')
    if len(set(order)) != len(order):
        raise ValueError('Ranking must contain unique evidence IDs')
    if not cutoffs or min(cutoffs) < 1 or len(order) < max(cutoffs):
        raise ValueError('Ranking depth must cover every requested cutoff')
    gold = set(gold)
    output = {}
    for k in cutoffs:
        found = gold.intersection(order[:k])
        output[f'recall_at_{k}'] = len(found) / len(gold)
        output[f'hit_at_{k}'] = float(bool(found))
        output[f'complete_at_{k}'] = float(found == gold)
        dcg = sum(1 / math.log2(i + 2) for i, aid in enumerate(order[:k]) if aid in gold)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        output[f'ndcg_at_{k}'] = dcg / ideal
    return output


def packing_metrics(order, gold_texts, context, top_k=5):
    if top_k < 1 or len(order) < top_k or not gold_texts:
        raise ValueError('Nonempty gold and sufficient retrieval depth required')
    selected = set(order[:top_k])
    retained = {aid for aid, text in gold_texts.items()
                if aid in selected and normalize_space(text) in normalize_space(context)}
    before = selected.intersection(gold_texts)
    complete_before = before == set(gold_texts)
    complete_after = retained == set(gold_texts)
    return {
        'gold_chunks': len(gold_texts),
        'gold_chunks_before_packing': len(before),
        'gold_chunks_after_packing': len(retained),
        'chunk_recall_before_packing': len(before) / len(gold_texts),
        'chunk_recall_after_packing': len(retained) / len(gold_texts),
        'complete_before_packing': float(complete_before),
        'complete_after_packing': float(complete_after),
        'retrieval_incomplete': float(not complete_before),
        'packing_lost_complete_evidence': float(complete_before and not complete_after),
        'packing_lost_chunks': len(before - retained),
    }


def aggregate_by_publisher(rows, expected_ids, metric_keys):
    expected_ids = list(expected_ids)
    if len(set(expected_ids)) != len(expected_ids) or not expected_ids:
        raise ValueError('Expected question denominator must be unique and nonempty')
    if len(rows) != len(expected_ids) or {r['id'] for r in rows} != set(expected_ids):
        raise ValueError('Missing, duplicate or extra evaluation questions')
    groups = defaultdict(list)
    for row in rows:
        if not row.get('publisher'): raise ValueError('Publisher missing')
        groups[row['publisher']].append(row)
    per_group = {g: {'questions': len(items), **{
        k: sum(float(r[k]) for r in items) / len(items) for k in metric_keys}}
        for g, items in sorted(groups.items())}
    return {'questions': len(rows), 'publishers': len(groups), 'by_publisher': per_group,
            'question_macro': {k: sum(float(r[k]) for r in rows) / len(rows) for k in metric_keys},
            'publisher_macro': {k: sum(r[k] for r in per_group.values()) / len(groups) for k in metric_keys}}


def token_f1(answer, reference):
    a = Counter(re.findall(r'\w+', answer.casefold()))
    b = Counter(re.findall(r'\w+', reference.casefold()))
    match = sum((a & b).values())
    return 2 * match / (sum(a.values()) + sum(b.values())) if a and b else 0.0


def citation_markers(answer):
    """Parse emitted SOURCE fields, preserving unknown IDs for evaluation.

    A no-source sentinel is recorded separately, never credited as a valid ID
    or as proof that the surrounding answer actually abstained.
    """
    values = []
    for match in re.finditer(r'(?im)\bsources?\s*:\s*([^\n]+)', answer):
        for value in match.group(1).split(','):
            value = re.sub(r'(?i)^\s*sources?\s*:\s*', '', value).strip(' .`*[]')
            if value:
                values.append(value)
    sentinels = {'none', 'n/a', 'na', 'insufficient_evidence', 'no sources', 'no source'}
    return {'ids': [v for v in values if v.casefold() not in sentinels],
            'sentinels': [v for v in values if v.casefold() in sentinels]}


def abstention_metrics(rows):
    """Positive class is evidence-absent; all-refuse exposes zero coverage."""
    if not rows: raise ValueError('Empty abstention denominator')
    absent = [r for r in rows if not r['answerable']]
    present = [r for r in rows if r['answerable']]
    refused = [r for r in rows if r['abstained']]
    tp = sum(r['abstained'] for r in absent)
    return {'n': len(rows), 'answerable_n': len(present), 'unanswerable_n': len(absent),
            'abstention_precision': tp / len(refused) if refused else None,
            'abstention_recall': tp / len(absent) if absent else None,
            'answerable_coverage': sum(not r['abstained'] for r in present) / len(present) if present else None,
            'abstentions': len(refused)}


def publisher_cluster_delta(baseline, candidate, metric, draws=2000, seed=42):
    """Paired equal-publisher contrast; few clusters cannot support broad claims."""
    import numpy as np
    if len({r['id'] for r in baseline})!=len(baseline) or len({r['id'] for r in candidate})!=len(candidate):
        raise ValueError('Duplicate paired question')
    b={r['id']:r for r in baseline};c={r['id']:r for r in candidate}
    if not b or b.keys()!=c.keys():raise ValueError('Different paired denominators')
    groups=defaultdict(list)
    for key in b:
        if b[key]['publisher']!=c[key]['publisher']:raise ValueError('Paired publisher changed')
        groups[b[key]['publisher']].append(float(c[key][metric])-float(b[key][metric]))
    deltas={g:float(np.mean(v)) for g,v in sorted(groups.items())}
    values=np.asarray(list(deltas.values()));rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),size=(draws,len(values)))].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975])
    return {'metric':metric,'questions':len(b),'publisher_groups':len(groups),
            'publisher_deltas':deltas,'publisher_macro_delta':float(values.mean()),
            'paired_publisher_cluster_bootstrap_95ci':[float(lo),float(hi)],'draws':draws,'seed':seed,
            'limitation':'Exploratory, unstable interval with few publisher clusters; no independent question bootstrap claim.'}
