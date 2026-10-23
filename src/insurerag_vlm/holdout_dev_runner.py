"""Label-free, resumable DEV retrieval orchestration with explicit readiness gates."""
import hashlib
import json
from pathlib import Path
import time

from .holdout_corpus import validate_inclusion_manifest
from .holdout_retrieval import ARMS


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf8')).hexdigest()


def validate_requests(requests):
    """The inferencer receives only IDs and questions, never annotations."""
    if not isinstance(requests, list) or not requests:
        raise ValueError('Nonempty DEV request list required')
    seen = set()
    for row in requests:
        if (not isinstance(row, dict) or set(row) != {'id', 'question'}
                or any(not isinstance(v, str) or not v.strip() for v in row.values())
                or row['id'] in seen):
            raise ValueError('Requests require unique IDs and question text only')
        seen.add(row['id'])
    return seen


def readiness(documents, manifest, requests, clearance):
    """Collect blockers without loading models, tokenizers or answer labels.

    A clearance is a review workflow assertion bound to artifacts, not an
    independent proof of content quality or semantic support equivalence.
    """
    blockers = []
    try:
        validate_requests(requests)
    except (ValueError, TypeError) as exc:
        blockers.append(str(exc))
    try:
        validate_inclusion_manifest(documents, manifest)
    except (ValueError, KeyError, TypeError) as exc:
        blockers.append('Corpus: ' + str(exc))
    if clearance.get('status') != 'READY_FOR_DEV_RETRIEVAL' or clearance.get('split') != 'DEV':
        blockers.append('DEV clearance must explicitly authorize reviewed DEV retrieval')
    for name, value in [('requests_sha256', fingerprint(requests)),
                        ('source_manifest_sha256', fingerprint(manifest))]:
        if clearance.get(name) != value:
            blockers.append('Clearance binding mismatch: ' + name)
    for name in ['dev_quality_gate_passed', 'source_family_isolation_reviewed',
                 'historical_contamination_reviewed', 'support_necessity_reviewed',
                 'equivalent_support_aliases_reviewed']:
        if clearance.get(name) is not True:
            blockers.append('Unfinished review: ' + name)
    for name in ['support_mapping_sha256', 'retained_pool_sha256', 'review_record_sha256']:
        value = clearance.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            blockers.append('Missing artifact binding: ' + name)
    return dict(ready=not blockers, blockers=blockers, request_count=len(requests),
                source_count=len(documents), generation_required=False)


def run_requests(requests, chunk_ids, infer, output, bindings):
    """Save each intended request, including failures; resume only exact bindings.

    Call readiness before constructing the backend. This function deliberately
    has no labels or evaluator. Any later scorer must retain failed requests in
    its denominator. Saved failures are not automatically retried on resume.
    """
    validate_requests(requests)
    chunks = set(chunk_ids)
    if not chunks or len(chunks) != len(chunk_ids):
        raise ValueError('Nonempty unique global corpus chunk IDs required')
    output = Path(output)
    run = dict(version=1, bindings=bindings, requests_sha256=fingerprint(requests),
               chunk_ids_sha256=fingerprint(list(chunk_ids)), arms=list(ARMS))
    path = output/'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != run:
            raise ValueError('Resume artifacts differ; preserve this run and use a new directory')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output directory has unbound artifacts')
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run, indent=2)+'\n', encoding='utf8')
    run_hash = fingerprint(run)
    def valid_orders(orders):
        if not isinstance(orders, dict) or set(orders) != set(ARMS):
            raise ValueError('Every registered arm needs an output')
        for ids in orders.values():
            if (not isinstance(ids, list) or len(ids) > 10
                    or len(set(ids)) != len(ids) or not set(ids) <= chunks):
                raise ValueError('Invalid ranks, duplicates or non-global chunk ID')
    records = []
    reused = 0
    for request in requests:
        file = output/(fingerprint(request)+'.json')
        if file.exists():
            record = json.loads(file.read_text(encoding='utf8'))
            if (record.get('run_sha256') != run_hash or record.get('id') != request['id']
                    or record.get('request_sha256') != fingerprint(request)):
                raise ValueError('Saved request binding mismatch')
            receipt = record.pop('record_sha256', None)
            if receipt != fingerprint(record):
                raise ValueError('Saved request content fingerprint mismatch')
            record['record_sha256'] = receipt
            valid_orders(record.get('orders'))
            reused += 1
        else:
            start = time.perf_counter()
            try:
                result = infer(dict(request))
                orders = result['orders']
                valid_orders(orders)
                record = dict(status='success', orders=orders, trace=result.get('trace', {}))
            except Exception as exc:
                # Preserve a failed intended query, not an invented empty success.
                record = dict(status='failed', orders={arm: [] for arm in ARMS},
                              error=dict(type=type(exc).__name__, message=str(exc)))
            record.update(id=request['id'], request_sha256=fingerprint(request),
                          run_sha256=run_hash, elapsed_seconds=time.perf_counter()-start)
            record['record_sha256'] = fingerprint(record)
            temporary = file.with_suffix('.pending')
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf8')
            temporary.replace(file)
        if record.get('status') not in ('success', 'failed'):
            raise ValueError('Invalid saved request status')
        records.append(record)
    summary = dict(intended_requests=len(requests), successful_requests=sum(r['status']=='success' for r in records),
                   failed_requests=sum(r['status']=='failed' for r in records), reused_requests=reused,
                   run_sha256=run_hash, arms=list(ARMS), generation_calls=0,
                   quality_scores_computed=False, dev_selection_locked=False,
                   latency_note='Shared six-arm execution traces; not standalone per-arm p95 for selection.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    return summary
