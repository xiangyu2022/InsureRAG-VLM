#!/usr/bin/env python3
"""Compare complete v2 arms only when their actual prompts and protocol match."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_research_benchmark as v1
from scripts import research_v2 as v2


def read_run(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'run_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('status') != 'complete' or not metadata.get('finished_utc'):
        raise ValueError('Comparison requires a complete run')
    if metadata.get('score_version') != v2.SCORE_VERSION or metadata.get('split') != 'test' or metadata.get('partial_dev_smoke'):
        raise ValueError('Comparison requires the complete v2 test split')
    if not metadata.get('model', {}).get('model_digest'):
        raise ValueError('Exact deployment digest is required')
    for filename, key in [('predictions.jsonl', 'predictions_sha256'), ('summary.json', 'summary_sha256')]:
        if v1.sha(folder / filename) != metadata.get(key):
            raise ValueError('Completed result checksum mismatch: ' + filename)
    rows = v1.read_jsonl(folder / 'predictions.jsonl')
    schedule = [{'id': row['id'], 'mode': row['mode']} for row in rows]
    if schedule != metadata['request_schedule'] or len(rows) != metadata['completed_requests'] or len(schedule) != len({(r['id'], r['mode']) for r in rows}):
        raise ValueError('Incomplete, duplicate, or reordered prediction schedule')
    protocol = metadata['prepared_protocol']
    if v2.digest_object(protocol['prompt_contract']) != protocol['prompt_contract_sha256']:
        raise ValueError('Prompt contract identity mismatch')
    for row in rows:
        if row['prompt_sha256'] != v1.sha(row['prompt']) or row['annotation_sha256'] != v2.digest_object(row['case']):
            raise ValueError('Recorded prompt/annotation hash mismatch')
        if row['scores'].get('score_version') != v2.SCORE_VERSION:
            raise ValueError('Prediction score versions differ')
    return metadata, rows


def validate_matched(left, right, left_rows, right_rows):
    for key in ('prepared_protocol_sha256', 'prepared_files', 'benchmark_lock_sha256', 'code_sha256',
                'prompt_contract_sha256', 'score_version', 'request_schedule'):
        if not left.get(key) or left.get(key) != right.get(key):
            raise ValueError('Unmatched or missing protocol field: ' + key)
    for key in ('generation_options', 'thinking'):
        if left['model'].get(key) != right['model'].get(key):
            raise ValueError('Unmatched model request option: ' + key)
    if left['prepared_protocol'] != right['prepared_protocol']:
        raise ValueError('Unmatched full prepared protocol')
    if len(left_rows) != len(right_rows):
        raise ValueError('Unmatched prediction counts')
    keys = ('id', 'mode', 'question', 'answerable', 'document_scope', 'document_id', 'document_family_id',
            'category', 'fact_id', 'annotation_sha256', 'case', 'prompt', 'prompt_sha256', 'ranked_sources')
    for a, b in zip(left_rows, right_rows, strict=True):
        # Failed calls also retain their intended frozen input and must match.
        if any(a.get(key) != b.get(key) for key in keys):
            raise ValueError('Unmatched actual question, label, evidence prompt, or retrieval order')


def compare(left_folder, right_folder):
    left, left_rows = read_run(left_folder)
    right, right_rows = read_run(right_folder)
    validate_matched(left, right, left_rows, right_rows)
    statistics = left['prepared_protocol']['statistics']
    kwargs = {'bootstrap': statistics['bootstrap_draws'], 'seed': statistics['bootstrap_seed']}
    results = {}
    for mode in sorted({row['mode'] for row in left_rows}):
        a = [row for row in left_rows if row['mode'] == mode]
        b = [row for row in right_rows if row['mode'] == mode]
        results[mode] = {'overall': {metric: v2.paired_metric(a, b, metric, **kwargs) for metric in v2.METRICS},
                        'family_cluster_sensitivity': {metric: v2.paired_metric(a, b, metric, cluster_key='document_family_id', **kwargs) for metric in v2.METRICS},
                        'by_category': {}, 'by_document': {}}
        for field, target in [('category', 'by_category'), ('document_id', 'by_document')]:
            for value in sorted({row[field] for row in a}):
                selected = [(x, y) for x, y in zip(a, b, strict=True) if x[field] == value]
                results[mode][target][value] = {metric: v2.paired_metric(
                    [x for x, _ in selected], [y for _, y in selected], metric, **kwargs) for metric in v2.METRICS}
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'role': 'matched_deployment_diagnostic_not_causal_model_version_effect',
            'score_version': v2.SCORE_VERSION, 'summary_version': v2.SUMMARY_VERSION,
            'prepared_protocol_sha256': left['prepared_protocol_sha256'],
            'left': left['model'], 'right': right['model'], 'modes': results,
            'limitations': ['Deterministic contracts are not expert semantic insurance accuracy.',
                            'Conditional-accuracy arms may answer different subsets; this is not accuracy at matched coverage.',
                            'Paired bootstrap resamples the same documents in both arms, retaining failures and undefined-denominator accounting.',
                            'Authored documents are not a random population sample; intervals are descriptive.',
                            'Model size, architecture and quantized artifact differ; the difference is not a causal version effect.']}


def render(report):
    lines = ['# Paired expanded diagnostic', '',
             f"Left: `{report['left']['resolved_model']}`. Right: `{report['right']['resolved_model']}`.", '',
             '| Mode | Metric | Left | Right | Difference (pp) | Paired document-cluster 95% interval (pp) |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for mode, groups in report['modes'].items():
        for name, value in groups['overall'].items():
            delta = 'n/a' if value['right_minus_left_percentage_points'] is None else f"{value['right_minus_left_percentage_points']:+.2f}"
            interval = 'not estimable' if value['ci95_low_pp'] is None else f"{value['ci95_low_pp']:+.2f} to {value['ci95_high_pp']:+.2f}"
            lines.append(f"| {mode} | {name} | {value['left_successes']}/{value['left_total']} | {value['right_successes']}/{value['right_total']} | {delta} | {interval} |")
    lines += ['', *['- ' + note for note in report['limitations']], '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.left, args.right)
    args.output.mkdir(parents=True, exist_ok=False)
    v1.write_json(report, args.output / 'comparison.json')
    (args.output / 'comparison.md').write_text(render(report), encoding='utf-8')
    print(render(report))
