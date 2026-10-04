#!/usr/bin/env python3
"""Run one explicitly pinned local Ollama model on frozen v2 prompts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_research_benchmark as v1
from scripts import research_v2 as v2


def validate_prediction_prefix(rows, inputs):
    if len(rows) > len(inputs):
        raise ValueError('Prediction prefix exceeds frozen schedule')
    keys = ('id', 'mode', 'question', 'answerable', 'document_id', 'document_family_id',
            'category', 'fact_id', 'document_scope', 'prompt', 'prompt_sha256', 'annotation_sha256', 'ranked_sources')
    for row, prepared in zip(rows, inputs):
        if any(row.get(key) != prepared.get(key) for key in keys):
            raise ValueError('Prediction prefix differs from frozen input identity/prompt')
        if row.get('scores', {}).get('score_version') != v2.SCORE_VERSION:
            raise ValueError('Prediction prefix has a different score version')


def run(args):
    if args.timeout <= 0 or not args.model or not args.expected_digest:
        raise ValueError('Positive timeout and explicit model/digest are required')
    protocol, inputs, pages = v2.load_prepared(args.prepared, root=ROOT)
    options = protocol['generation_options']
    # The borrowed transport performs exact tag/digest checks on every request.
    # Credentials and other installed tags cannot activate alternate providers.
    client = v1.LocalOllamaClient(SimpleNamespace(endpoint=args.endpoint, model=args.model,
        timeout=args.timeout, num_ctx=options['num_ctx'], num_predict=options['num_predict']))
    if client.model_info['digest'] != args.expected_digest:
        raise RuntimeError('Installed model digest differs from the explicitly requested artifact')
    if client.options != options or protocol.get('thinking') is not False:
        raise ValueError('Transport decoding options differ from the frozen protocol')
    protocol_sha = v1.sha(Path(args.prepared) / 'protocol.json')
    output = Path(args.output)
    prediction_path = output / 'predictions.jsonl'
    if args.resume:
        metadata = json.loads((output / 'run_metadata.json').read_text(encoding='utf-8'))
        if metadata.get('status') == 'complete':
            raise ValueError('A complete frozen run cannot be resumed or overwritten')
        if (metadata.get('prepared_protocol_sha256') != protocol_sha
                or metadata.get('model', {}).get('model_digest') != args.expected_digest
                or metadata.get('model', {}).get('resolved_model') != args.model):
            raise ValueError('Resume protocol/model identity differs')
        results = v1.read_jsonl(prediction_path) if prediction_path.exists() else []
        if results and v1.sha(prediction_path) != metadata.get('predictions_sha256'):
            raise ValueError('Partial predictions changed; resume rejected')
        validate_prediction_prefix(results, inputs)
        metadata.setdefault('resume_events', []).append(datetime.now(timezone.utc).isoformat())
    else:
        output.mkdir(parents=True, exist_ok=False)
        results = []
        metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
            'protocol_version': v2.PROTOCOL_VERSION, 'score_version': v2.SCORE_VERSION,
            'prepared_protocol_sha256': protocol_sha, 'prepared_protocol': protocol,
            'prepared_files': protocol['prepared_files'], 'benchmark_lock_sha256': protocol['benchmark_lock_sha256'],
            'code_sha256': protocol['code_sha256'], 'prompt_contract_sha256': protocol['prompt_contract_sha256'],
            'split': protocol['split'], 'partial_dev_smoke': protocol['partial_dev_smoke'],
            'request_schedule': protocol['request_schedule'], 'case_count': protocol['case_count'],
            'model': client.backend_metadata(), 'runtime_environment': v1.runtime_environment(),
            'evaluation_path': protocol['evaluation_path'], 'completed_requests': 0,
            'adapter_status': 'Unadapted explicit Ollama artifact; no project SFT adapter is loaded.'}
    metadata['status'] = 'running'
    by_source = {page['citation']: page for page in pages}
    v1.write_json(metadata, output / 'run_metadata.json')
    try:
        for prepared in inputs[len(results):]:
            row = dict(prepared)
            client.last_prompt = None
            client.last_generation_metadata = {}
            started = time.perf_counter()
            raw, error = '', None
            try:
                raw = client.generate(prepared['prompt'])
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
            generation = dict(client.last_generation_metadata)
            row.update({'raw_response': raw, 'generation': generation, 'error': error,
                        'wall_seconds': time.perf_counter() - started,
                        'generation_used': client.last_prompt is not None})
            row['scores'] = v2.score_response(raw, prepared['case'], by_source, prepared['ranked_sources'],
                prompt=prepared['prompt'], generation=generation, error=error)
            results.append(row)
            with prediction_path.open('a', encoding='utf-8') as handle:
                handle.write(v2.canonical(row) + '\n')
            metadata['completed_requests'] = len(results)
            metadata['request_errors'] = sum(bool(item.get('error')) for item in results)
            metadata['predictions_sha256'] = v1.sha(prediction_path)
            v1.write_json(metadata, output / 'run_metadata.json')
            print(json.dumps({'completed': len(results), 'scheduled': len(inputs), 'id': row['id'],
                              'mode': row['mode'], 'seconds': round(row['wall_seconds'], 3),
                              'strict_contract_pass': row['scores']['context_grounded_key_pass'],
                              'error': error}), flush=True)
        if v2.source_hashes(ROOT) != protocol['code_sha256']:
            raise RuntimeError('Evaluation source files changed during the run; result is not a frozen-code completion')
        statistics = protocol['statistics']
        summary = v2.summarize(results, bootstrap=statistics['bootstrap_draws'], seed=statistics['bootstrap_seed'])
        v1.write_json(summary, output / 'summary.json')
        (output / 'summary.md').write_text(v2.render_summary(summary), encoding='utf-8')
        metadata.update({'status': 'complete', 'finished_utc': datetime.now(timezone.utc).isoformat(),
                         'summary_sha256': v1.sha(output / 'summary.json')})
    except BaseException as exc:
        metadata.update({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                         'last_error': type(exc).__name__ + ': ' + str(exc)})
        raise
    finally:
        v1.write_json(metadata, output / 'run_metadata.json')
    print(json.dumps({'status': metadata['status'], 'requests': len(results), 'output': str(output)}))
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--expected-digest', required=True)
    p.add_argument('--endpoint', default='http://127.0.0.1:11435')
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Append only the missing suffix of an interrupted matching run; never rerun completed rows')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
