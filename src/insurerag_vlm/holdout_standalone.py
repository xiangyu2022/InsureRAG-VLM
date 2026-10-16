"""Resumable independent-arm DEV execution; no answer labels enter inference."""
import json
import math
from pathlib import Path
import time

from .holdout_dev_runner import fingerprint, validate_requests
from .holdout_retrieval import ARMS


def _manifest(requests, chunk_ids, bindings):
    validate_requests(requests)
    if (not isinstance(chunk_ids, list) or not chunk_ids
            or any(not isinstance(x, str) or not x or x != x.strip() for x in chunk_ids)
            or len(set(chunk_ids)) != len(chunk_ids)):
        raise ValueError('Unique nonempty global chunk IDs required')
    return dict(version=1, execution='standalone_per_arm', split='DEV',
                arms=list(ARMS), bindings=bindings,
                requests_sha256=fingerprint(requests), chunk_ids_sha256=fingerprint(chunk_ids))


def _validate_order(order, chunks):
    if (not isinstance(order, list) or any(not isinstance(x, str) for x in order)
            or len(order) > 10 or len(set(order)) != len(order) or not set(order) <= chunks):
        raise ValueError('Invalid standalone ranking or non-global chunk ID')


def _read_record(path, request, arm, run_hash, chunks):
    record = json.loads(path.read_text(encoding='utf8'))
    if not isinstance(record, dict):
        raise ValueError('Standalone record must be an object')
    receipt = record.pop('record_sha256', None)
    if receipt != fingerprint(record):
        raise ValueError('Saved standalone content fingerprint mismatch')
    record['record_sha256'] = receipt
    if (record.get('id') != request['id'] or record.get('arm') != arm
            or record.get('request_sha256') != fingerprint(request)
            or record.get('run_sha256') != run_hash):
        raise ValueError('Saved standalone request binding mismatch')
    _validate_order(record.get('order'), chunks)
    seconds = record.get('elapsed_seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Finite nonnegative standalone elapsed time required')
    if record.get('status') not in ('success', 'failed'):
        raise ValueError('Invalid standalone status')
    if record['status'] == 'failed' and record['order']:
        raise ValueError('Failed standalone query cannot have a successful ranking')
    return record


def read_records(requests, chunk_ids, output, bindings):
    """Load a complete bound run; missing intended requests are never dropped."""
    run = _manifest(requests, chunk_ids, bindings)
    output = Path(output)
    if json.loads((output/'run_manifest.json').read_text(encoding='utf8')) != run:
        raise ValueError('Standalone run bindings differ')
    records = []
    for request in requests:
        for arm in ARMS:
            path = output/arm/(fingerprint(request)+'.json')
            if not path.is_file():
                raise ValueError('Incomplete intended standalone requests')
            records.append(_read_record(path, request, arm, fingerprint(run), set(chunk_ids)))
    return run, records


def run_requests(requests, chunk_ids, infer_arm, output, bindings, *, synchronize=lambda: None):
    """Time each arm end-to-end, preserve failures, and resume exact records.

    The backend must compute only this arm's channels. Setup and warm-up happen
    before this call. Synchronization is inside the observed request time.
    Six-arm shared timings and per-component sums are not accepted as substitutes.
    """
    run = _manifest(requests, chunk_ids, bindings)
    output = Path(output)
    path = output/'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != run:
            raise ValueError('Standalone resume bindings differ')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Standalone output has unbound artifacts')
        output.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf8') as handle:
            json.dump(run, handle, indent=2, allow_nan=False)
    run_hash, chunks = fingerprint(run), set(chunk_ids)
    reused = 0
    records = []
    for request in requests:
        for arm in ARMS:
            path = output/arm/(fingerprint(request)+'.json')
            if path.exists():
                record = _read_record(path, request, arm, run_hash, chunks)
                reused += 1
            else:
                start = time.perf_counter()
                try:
                    synchronize()
                    result = infer_arm(arm, dict(request))
                    synchronize()
                    _validate_order(result['order'], chunks)
                    record = dict(status='success', order=result['order'], trace=result.get('trace', {}))
                    json.dumps(record, allow_nan=False)
                except Exception as exc:
                    record = dict(status='failed', order=[],
                                  error=dict(type=type(exc).__name__, message=str(exc)))
                record.update(id=request['id'], arm=arm, request_sha256=fingerprint(request),
                              run_sha256=run_hash, elapsed_seconds=time.perf_counter()-start)
                record['record_sha256'] = fingerprint(record)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.pending')
                temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf8')
                temporary.replace(path)
            records.append(record)
    summary = dict(execution='standalone_per_arm', intended_requests=len(requests),
                   intended_arm_requests=len(requests)*len(ARMS), reused_arm_requests=reused,
                   arms={arm:dict(successful=sum(r['status']=='success' for r in records if r['arm']==arm),
                                  failed=sum(r['status']=='failed' for r in records if r['arm']==arm)) for arm in ARMS},
                   run_sha256=run_hash, generation_calls=0, quality_scores_computed=False,
                   dev_selection_locked=False,
                   latency_scope='Per-arm elapsed retrieval including synchronization and validation; setup/warm-up separate.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    return summary
