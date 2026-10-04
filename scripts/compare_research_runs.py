"""Compare two complete frozen-test deployments without silently mixing protocols."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


MATCHED_OPTIONS = [
    'split', 'mode', 'retrieval_model', 'retrieval_mode', 'top_k',
    'max_context_chars', 'num_ctx', 'num_predict', 'limit',
]
METRICS = [
    'retrieval_hit_at_k', 'answer_key_match', 'citation_match',
    'context_grounded_key_pass', 'completed_strict_unsupported_abstention',
    'json_contract_success',
]
ROW_METRICS = {
    'retrieval_hit_at_k': ('retrieval_hit', True),
    'answer_key_match': ('answer_key_match', True),
    'citation_match': ('citation_match', True),
    'context_grounded_key_pass': ('context_grounded_key_pass', True),
    'completed_strict_unsupported_abstention': ('completed_strict_abstention', False),
}


def read_run(folder):
    folder = Path(folder)
    metadata = json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
    raw = (folder/'predictions.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata.get('predictions_sha256'):
        raise ValueError('Prediction checksum does not match completed run: '+str(folder))
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    ids = [(row['id'], row['mode']) for row in rows]
    schedule = [(row['id'], row['mode']) for row in metadata['request_schedule']]
    if ids != schedule or len(ids) != len(set(ids)):
        raise ValueError('Incomplete, reordered, or duplicate evaluation requests')
    if not metadata.get('finished_utc') or metadata.get('completed_requests') != len(rows):
        raise ValueError('Run is not complete')
    if metadata['split'] != 'test' or metadata.get('partial_dev_smoke'):
        raise ValueError('Final comparison requires the complete test split')
    if not metadata.get('model', {}).get('model_digest'):
        raise ValueError('Exact deployment digest is required')
    return metadata, rows, json.loads((folder/'summary.json').read_text(encoding='utf-8'))


def validate_matched(left, right, left_rows, right_rows):
    for key in ['benchmark_lock_sha256', 'prompt_contract_sha256', 'code_sha256',
                'score_version', 'primary_answer_metric', 'request_schedule']:
        if not left.get(key) or not right.get(key):
            raise ValueError('Unmatched or missing protocol field: '+key)
        if left.get(key) != right.get(key):
            raise ValueError('Unmatched protocol field: '+key)
    for key in MATCHED_OPTIONS:
        if left['run_config'].get(key) != right['run_config'].get(key):
            raise ValueError('Unmatched run option: '+key)
    for key in ['generation_options', 'thinking']:
        if left['model'].get(key) != right['model'].get(key):
            raise ValueError('Unmatched decoding option: '+key)
    for a, b in zip(left_rows, right_rows, strict=True):
        for key in ['id', 'mode', 'question', 'answerable', 'document_scope']:
            if a[key] != b[key]:
                raise ValueError('Unmatched evaluation record: '+key)
        if not a.get('error') and not b.get('error'):
            if a.get('prompt') != b.get('prompt') or a.get('ranked_sources') != b.get('ranked_sources'):
                raise ValueError('Unmatched actual evidence prompt or retrieval order')


def compare(left_folder, right_folder):
    left, left_rows, left_summary = read_run(left_folder)
    right, right_rows, right_summary = read_run(right_folder)
    validate_matched(left, right, left_rows, right_rows)
    pairs = {}
    for mode in left_summary:
        pairs[mode] = {}
        for metric, (row_key, answerable) in ROW_METRICS.items():
            selected = [(a,b) for a,b in zip(left_rows,right_rows,strict=True)
                        if a['mode'] == mode and a['answerable'] is answerable
                        and a['scores'].get(row_key) is not None]
            if not selected:
                continue
            counts = {'both_pass':0, 'left_only':0, 'right_only':0, 'neither_pass':0}
            for a,b in selected:
                x,y = bool(a['scores'].get(row_key)),bool(b['scores'].get(row_key))
                counts['both_pass' if x and y else 'left_only' if x else 'right_only' if y else 'neither_pass'] += 1
            pairs[mode][metric] = {'n':len(selected), **counts,
                'right_minus_left_percentage_points':100*(counts['right_only']-counts['left_only'])/len(selected)}
    return {
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'role':'matched_deployment_diagnostic_not_causal_model_version_effect',
        'left':{'folder':str(left_folder),'model':left['model'],'summary':left_summary},
        'right':{'folder':str(right_folder),'model':right['model'],'summary':right_summary},
        'protocol_checks':'complete request schedule, exact input labels, benchmark/prompt/code/decoding identities and prediction checksums matched',
        'benchmark_lock_sha256':left['benchmark_lock_sha256'],
        'score_version':left['score_version'], 'paired_counts':pairs,
        'limitations':[
            'Model family, parameter count, and quantized artifact differ; this does not isolate a causal version effect.',
            'Tiny AI-authored/agent-reviewed public-guide diagnostic; no independent human adjudication.',
            'Shared documents make per-question Wilson intervals descriptive; no significance or population-generalization claim.',
            'Strict answer/evidence contracts are not semantic correctness or a production error rate.',
            'Transport failures remain in denominators; they are not dropped from a comparison.',
        ],
    }


def render(report):
    lines = ['# Matched frozen-test deployment comparison', '',
        f"Left: `{report['left']['model']['resolved_model']}`. Right: `{report['right']['model']['resolved_model']}`.", '',
        report['protocol_checks']+'.', '',
        '| Mode / metric | Left successes / n | Right successes / n | Right minus left (pp) |',
        '| --- | ---: | ---: | ---: |']
    for mode, metrics in report['paired_counts'].items():
        for metric, item in metrics.items():
            lines.append(f"| {mode} / {metric} | {item['both_pass']+item['left_only']}/{item['n']} | {item['both_pass']+item['right_only']}/{item['n']} | {item['right_minus_left_percentage_points']:+.2f} |")
    lines += ['', *['- '+text for text in report['limitations']], '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left',type=Path,required=True)
    parser.add_argument('--right',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = compare(args.left,args.right)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (args.output/'comparison.md').write_text(render(report),encoding='utf-8')
    print(render(report))
