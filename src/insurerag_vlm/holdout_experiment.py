"""Prospective independent-holdout metrics; never loads a model or gold into inference."""
import math
import random
from collections import defaultdict



def score_retrieval(order, gold, *, answerable, failed=False):
    """Annotated chunk recall, not hit rate; refusal evidence is diagnostic only.

    Rank failures remain zero in the answerable denominator. A short successful
    ranking is valid: missing ranks contribute no relevance, never fake IDs.
    """
    if type(answerable) is not bool or type(failed) is not bool:
        raise ValueError('Explicit boolean answerability/failure required')
    if isinstance(order, (str, bytes)) or isinstance(gold, (str, bytes)):
        raise ValueError('Use sequences of canonical evidence IDs')
    order, gold = list(order), list(gold)
    for values in (order, gold):
        if any(not isinstance(v, str) or not v or v != v.strip() for v in values):
            raise ValueError('Nonempty canonical evidence IDs required')
        if len(set(values)) != len(values):
            raise ValueError('Duplicate evidence ID')
    names = ('recall_at_5', 'hit_at_5', 'complete_at_5', 'ndcg_at_10')
    if not answerable:
        return dict(answerable=False, failed=failed, **dict.fromkeys(names))
    if not gold:
        raise ValueError('Answerable case requires preannotated relevance labels')
    if failed:
        return dict(answerable=True, failed=True, **dict.fromkeys(names, 0.0))
    # Use explicit rank positions; no padding IDs that could collide with gold.
    relevant = set(gold)
    found = relevant.intersection(order[:5])
    dcg = sum(1 / math.log2(i + 2) for i, v in enumerate(order[:10]) if v in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return dict(answerable=True, failed=False, recall_at_5=len(found)/len(relevant),
                hit_at_5=float(bool(found)), complete_at_5=float(found == relevant),
                ndcg_at_10=dcg/ideal)


def paired_family_delta(baseline, candidate, expected_ids, metric, *, draws=2000, seed=42):
    """Resample whole document families, paired across arms, preserving all rows.

    The primary estimand is question-macro difference; each bootstrap draw
    resamples families with replacement and divides by the sampled row count.
    It is deliberately not an independent-question bootstrap or an equal-family
    mean mislabeled as a question mean. Failed queries must already be scored.
    """
    if type(draws) is not int or draws < 2 or type(seed) is not int:
        raise ValueError('Integer seed and at least two bootstrap draws required')
    if isinstance(expected_ids, (str, bytes)):
        raise ValueError('Expected question IDs must be a sequence')
    expected = list(expected_ids)
    if not expected or any(not isinstance(x, str) or not x or x != x.strip() for x in expected):
        raise ValueError('Nonempty canonical expected question IDs required')
    if len(set(expected)) != len(expected):
        raise ValueError('Duplicate expected question ID')
    def index(rows):
        result = {}
        for row in rows:
            if row['id'] in result:
                raise ValueError('Duplicate prediction ID')
            result[row['id']] = row
        if set(result) != set(expected):
            raise ValueError('Missing or extra questions; do not shrink denominators')
        return result
    b, c = index(baseline), index(candidate)
    groups = defaultdict(list)
    for ident in expected:
        family = b[ident].get('document_family')
        if not isinstance(family, str) or not family or family != family.strip():
            raise ValueError('Canonical document family required')
        if c[ident].get('document_family') != family:
            raise ValueError('Paired document family changed')
        values = [b[ident][metric], c[ident][metric]]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Metrics must be finite proportions in [0,1]')
        groups[family].append(values[1] - values[0])
    summaries = [(sum(v), len(v)) for _, v in sorted(groups.items())]
    result = dict(questions=len(expected), document_families=len(groups), metric=metric,
                  question_macro_delta=sum(s for s,n in summaries)/len(expected),
                  family_macro_delta=sum(s/n for s,n in summaries)/len(groups),
                  draws=draws, seed=seed, ci95=None,
                  limitation='Family-cluster uncertainty; not a guarantee of representative sampling or unknown-pretraining absence.')
    if len(groups) < 2:
        result['ci_unavailable_reason'] = 'Fewer than two independent document families'
        return result
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [summaries[rng.randrange(len(summaries))] for _ in summaries]
        samples.append(sum(s for s,n in selected)/sum(n for s,n in selected))
    samples.sort()
    def quantile(p):
        pos = p*(len(samples)-1)
        lo = int(pos)
        return samples[lo] + (pos-lo)*(samples[min(lo+1,len(samples)-1)]-samples[lo])
    result['ci95'] = [quantile(.025), quantile(.975)]
    return result


def evidence_unit_metrics(order, units, *, k=5, failed=False):
    """Complete support uses necessary units, not the number of alias chunks.

    Each unit has an ID and alternative_chunk_sets. Any one complete set proves
    retrieval of that unit; a set can require multiple chunks when a source span
    crosses a chunk boundary. This is a frozen-annotation scorer, not semantic
    judgment, and is not called on unanswerable cases.
    """
    if type(k) is not int or k < 1 or type(failed) is not bool:
        raise ValueError('Positive integer cutoff and boolean failure required')
    if isinstance(order, (str, bytes)):
        raise ValueError('Ranking must be a sequence')
    order = list(order)
    if any(not isinstance(v, str) or not v or v != v.strip() for v in order) or len(set(order)) != len(order):
        raise ValueError('Ranking requires unique canonical chunk IDs')
    units = list(units)
    if not units:
        raise ValueError('Necessary evidence units cannot be empty')
    seen, matches = set(), []
    selected = set() if failed else set(order[:k])
    for unit in units:
        ident = unit.get('id')
        if not isinstance(ident, str) or not ident or ident != ident.strip() or ident in seen:
            raise ValueError('Necessary units require unique canonical IDs')
        seen.add(ident)
        alternatives = unit.get('alternative_chunk_sets')
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError('Every unit requires explicit nonempty alternatives')
        sets = set()
        for alternative in alternatives:
            if not isinstance(alternative, list) or not alternative:
                raise ValueError('Empty or scalar alternative is not evidence')
            if any(not isinstance(v, str) or not v or v != v.strip() for v in alternative) or len(set(alternative)) != len(alternative):
                raise ValueError('Alternatives require unique canonical chunk IDs')
            key = frozenset(alternative)
            if key in sets:
                raise ValueError('Duplicate alternative support set')
            sets.add(key)
        matches.append(any(s <= selected for s in sets))
    return {'necessary_units':len(units), 'retrieved_units':sum(matches),
            'unit_recall':sum(matches)/len(units),
            'complete_evidence':float(all(matches)), 'k':k, 'failed':failed}
