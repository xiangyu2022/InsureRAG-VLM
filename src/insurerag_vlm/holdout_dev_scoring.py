"""Reviewed DEV scoring and immutable selection; never imports a model."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from .holdout_dev_runner import fingerprint, validate_requests
from .holdout_experiment import evidence_unit_metrics, score_retrieval
from .holdout_retrieval import ARMS

METRICS = ('complete_evidence_at_5', 'recall_at_5', 'ndcg_at_10',
           'hit_at_5', 'chunk_complete_at_5', 'unit_recall_at_5')


def _canonical(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def _finite(value, *, proportion=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and value >= 0 and (not proportion or value <= 1))


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def percentile(values, p):
    """Prespecified linear interpolation at (n-1)*p, including failed requests."""
    values = sorted(values)
    if not values or any(not _finite(x) for x in values) or not 0 <= p <= 1:
        raise ValueError('Finite nonnegative observations and valid quantile required')
    position = (len(values)-1)*p
    lower = int(position)
    return values[lower]+(position-lower)*(values[min(lower+1,len(values)-1)]-values[lower])


def score_run(requests, chunk_ids, mapping, run, records):
    """Score all intended per-arm requests against reviewed frozen annotations.

    Binary chunk recall/nDCG use the union of reviewed relevant alternatives.
    Primary completeness instead requires every necessary unit, with each unit's
    alternative sets interpreted disjunctively. Refusal context is diagnostic.
    """
    expected = validate_requests(requests)
    if (run.get('execution') != 'standalone_per_arm' or run.get('split') != 'DEV'
            or run.get('arms') != list(ARMS)
            or run.get('requests_sha256') != fingerprint(requests)
            or run.get('chunk_ids_sha256') != fingerprint(chunk_ids)):
        raise ValueError('Bound standalone DEV run required; shared timings cannot select an arm')
    if (mapping.get('whole_dev_clearance') is not True
            or mapping.get('global_equivalent_alias_review_complete') is not True):
        raise ValueError('Reviewed complete DEV support mapping required')
    if not isinstance(chunk_ids, list) or not chunk_ids or any(not _canonical(x) for x in chunk_ids):
        raise ValueError('Canonical unique global corpus IDs required')
    chunks = set(chunk_ids)
    if len(chunks) != len(chunk_ids):
        raise ValueError('Canonical unique global corpus IDs required')
    questions = {r['id']:r['question'] for r in requests}
    annotations = {}
    golds = {}
    gaps = {}
    for item in mapping['items']:
        ident = item['id']
        if ident in annotations or ident not in expected:
            raise ValueError('Duplicate or unexpected DEV annotation')
        if (item.get('question_sha256') != hashlib.sha256(questions[ident].encode('utf8')).hexdigest()
                or not _canonical(item.get('document_family'))
                or item.get('source_necessity_reviewed') is not True
                or item.get('global_semantic_alias_review_complete') is not True):
            raise ValueError('Unreviewed or mismatched DEV annotation')
        if item.get('task') not in {'ordinary_qa','numerical_calculation','multi_evidence','insufficient_evidence'}:
            raise ValueError('Unknown registered task')
        answerable = item['task'] != 'insufficient_evidence'
        if item.get('answer_relevance_gold_applicable') is not answerable:
            raise ValueError('Explicit task-consistent answer relevance required')
        units = item.get('units')
        if not isinstance(units, list) or (not answerable and units):
            raise ValueError('Refusals cannot have answer-relevance units')
        if answerable:
            evidence_unit_metrics([], units)  # Validate every necessary/alternative set.
        gold = {cid for unit in units for alt in unit['alternative_chunk_sets'] for cid in alt}
        gap = {cid for span in item.get('gap_context_only', []) for cid in span['chunk_ids']}
        if not (gold | gap) <= chunks:
            raise ValueError('Annotation references a non-global chunk')
        annotations[ident], golds[ident], gaps[ident] = item, sorted(gold), gap
    if set(annotations) != expected:
        raise ValueError('Missing DEV annotations; do not shrink denominators')
    indexed = {}
    for record in records:
        key = (record.get('id'), record.get('arm'))
        if key in indexed or key[0] not in expected or key[1] not in ARMS:
            raise ValueError('Duplicate or unexpected standalone prediction')
        request = dict(id=key[0], question=questions[key[0]])
        content = {k:v for k,v in record.items() if k != 'record_sha256'}
        if (record.get('record_sha256') != fingerprint(content)
                or record.get('run_sha256') != fingerprint(run)
                or record.get('request_sha256') != fingerprint(request)):
            raise ValueError('Standalone prediction fingerprint mismatch')
        order = record.get('order')
        if (not isinstance(order, list) or any(not _canonical(x) for x in order)
                or len(order) > 10 or len(set(order)) != len(order) or not set(order) <= chunks):
            raise ValueError('Invalid standalone ranking')
        if (record.get('status') not in ('success','failed')
                or not _finite(record.get('elapsed_seconds'))
                or (record['status']=='failed' and order)):
            raise ValueError('Invalid standalone failure or elapsed time')
        indexed[key] = record
    if len(indexed) != len(expected)*len(ARMS):
        raise ValueError('Missing intended arm requests; do not shrink denominators')
    cases, summaries = [], {}
    for arm in ARMS:
        rows = []
        for ident in sorted(expected):
            item, record = annotations[ident], indexed[ident,arm]
            answerable = item['answer_relevance_gold_applicable']
            failed = record['status']=='failed'
            score = score_retrieval(record['order'], golds[ident], answerable=answerable, failed=failed)
            unit = evidence_unit_metrics(record['order'], item['units'], failed=failed) if answerable else None
            diagnostic = (float(bool(set(record['order'][:5]) & gaps[ident])) if not failed else 0.0) if gaps[ident] else None
            row = dict(id=ident, arm=arm, document_family=item['document_family'], answerable=answerable,
                       failed=failed, elapsed_seconds=record['elapsed_seconds'],
                       complete_evidence_at_5=unit['complete_evidence'] if unit else None,
                       unit_recall_at_5=unit['unit_recall'] if unit else None,
                       recall_at_5=score['recall_at_5'], ndcg_at_10=score['ndcg_at_10'],
                       hit_at_5=score['hit_at_5'], chunk_complete_at_5=score['complete_at_5'],
                       gap_context_hit_at_5=diagnostic if not answerable else None)
            rows.append(row)
        answer_rows = [r for r in rows if r['answerable']]
        family = defaultdict(list)
        for row in answer_rows:
            family[row['document_family']].append(row)
        summaries[arm] = dict(intended=len(rows), successful=sum(not r['failed'] for r in rows),
                             failed=sum(r['failed'] for r in rows), answerable_intended=len(answer_rows),
                             answerable_failed=sum(r['failed'] for r in answer_rows),
                             refusal_intended=len(rows)-len(answer_rows),
                             answerable_families=len(family),
                             question_macro={m:_mean([r[m] for r in answer_rows]) for m in METRICS},
                             family_macro={m:_mean([_mean([r[m] for r in family[f]]) for f in sorted(family)]) for m in METRICS},
                             latency_seconds=dict(n=len(rows), p50=percentile([r['elapsed_seconds'] for r in rows],.5),
                                                  p95=percentile([r['elapsed_seconds'] for r in rows],.95)))
        cases.extend(rows)
    return dict(version=1, status='REVIEWED_DEV_STANDALONE_RETRIEVAL_SCORED', split='DEV',
                execution='standalone_per_arm', run_sha256=fingerprint(run),
                mapping_fingerprint=fingerprint(mapping), bindings=run['bindings'],
                requests_sha256=run['requests_sha256'], intended_requests=len(expected),
                arm_order=list(ARMS), arms=summaries, cases=cases,
                latency_method='Linear interpolation at (n-1)*p over all intended requests, including failures; setup/warm-up excluded.',
                dev_selection_locked=False, test_inference_authorized=False)


def select_arm(report):
    """Apply registered family-macro metric, latency and arm-order tie breaks."""
    if (report.get('status') != 'REVIEWED_DEV_STANDALONE_RETRIEVAL_SCORED'
            or report.get('split') != 'DEV' or report.get('execution') != 'standalone_per_arm'
            or report.get('arm_order') != list(ARMS) or set(report.get('arms', {})) != set(ARMS)):
        raise ValueError('Complete standalone DEV scores required')
    keys = {}
    denominators = set()
    for index, arm in enumerate(ARMS):
        row = report['arms'][arm]
        metrics = [row['family_macro'][m] for m in METRICS[:3]]
        latency = row['latency_seconds']
        if (not row['answerable_intended'] or not row['answerable_families']
                or row['intended'] != report['intended_requests']
                or latency['n'] != report['intended_requests']
                or any(not _finite(x, proportion=True) for x in metrics)
                or not _finite(latency['p95'])):
            raise ValueError('Missing denominators, quality or standalone latency')
        denominators.add((row['answerable_intended'], row['answerable_families'], row['refusal_intended']))
        keys[arm] = tuple(-x for x in metrics)+(latency['p95'],index)
    if len(denominators) != 1:
        raise ValueError('All arms must retain the same intended denominators')
    chosen = min(ARMS, key=keys.__getitem__)
    return dict(arm=chosen, priority=['family_macro_complete_evidence_at_5',
                                    'family_macro_recall_at_5','family_macro_ndcg_at_10',
                                    'lower_standalone_p95','registered_arm_order'],
                comparison_keys={k:list(v) for k,v in keys.items()})


def write_selection_lock(path, report, artifact_hashes):
    """Write once; caller hashes actual reviewed freeze artifacts, not labels.

    This locks a DEV decision/configuration. TEST still needs its separate
    >=1000-item global acceptance and immutable dataset freeze.
    """
    required = {'dev_data','corpus','models','tokenizers','code','prompt','packing','rubric','protocol'}
    if set(artifact_hashes) != required or any(
        not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x)
        for x in artifact_hashes.values()
    ):
        raise ValueError('Complete freeze artifact SHA256 bindings required')
    selection = select_arm(report)
    lock = dict(version=1,status='DEV_SELECTION_LOCK',split='DEV',selection=selection,
                score_report_sha256=fingerprint(report),artifacts=artifact_hashes,
                test_inference_authorized=False)
    lock['lock_sha256'] = fingerprint(lock)
    with Path(path).open('x',encoding='utf8') as handle:
        json.dump(lock,handle,indent=2,allow_nan=False)
        handle.write('\n')
    return lock
