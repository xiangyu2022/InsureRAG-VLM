"""Small, explicitly exposed development diagnostic; never a fresh holdout.

Prepare label-free trained retrieval once, then run each generator separately.
FinQA reference answers are kept outside generation inputs. Negative controls
are synthetic interventions and are reported separately from public questions.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.answer_evidence_reranker import retrieve_batch, answer_retrieval


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def prepare(args):
    if args.output.exists(): raise ValueError('Output directory already exists')
    specs = [('finqa', 'finqa_evidence_v1/valid.jsonl', 8),
             ('insuranceqa', 'insuranceqa_v2/valid.jsonl', 4),
             ('condition', 'condition_v1/test.jsonl', 4), ('fiqa', 'fiqa_v1/valid.jsonl', 4)]
    original = {r['id']: r for r in json.loads(args.finqa_dev.read_text(encoding='utf8'))}
    cases, hashes = [], {}
    for corpus, relative, n in specs:
        path = ROOT / 'data/benchmarks' / relative
        hashes[relative] = digest(path)
        selected = sorted(read_rows(path), key=lambda r: hashlib.sha256(('qwen35-dev-20261004:' + r['id']).encode()).hexdigest())[:n]
        for r in selected:
            case = {'id': r['id'], 'question': r['question'], 'corpus': corpus,
                    'report': r.get('source_group') if corpus == 'finqa' else None,
                    'gold_answer_ids': r['gold_answer_ids'], 'cohort': corpus,
                    'annotation_origin': r['annotation_origin'], 'expected_abstain': False,
                    'split_use': 'previously exposed public development diagnostic'}
            if corpus == 'finqa':
                qa = original[r['upstream_id']]['qa']
                assert qa['question'] == r['question']
                case.update(reference_answer=qa['answer'], reference_numeric=qa['exe_ans'], upstream_id=r['upstream_id'])
            cases.append(case)
    normal_count = len(cases)
    for i in range(4):
        base = cases[i]
        cases.append({**base, 'id': base['id'] + ('__empty' if i < 2 else '__wrong_report'),
                      'cohort': 'synthetic_empty' if i < 2 else 'synthetic_wrong_report',
                      'expected_abstain': True, 'intervention_base_id': base['id'],
                      'annotation_origin': 'synthetic missing-evidence diagnostic, not real-world labels'})
    for key, question in [('personal_deductible', 'What is the collision deductible on my own policy ZX-14?'),
                          ('personal_limit', 'What is the dwelling coverage limit on my own policy ZX-14?')]:
        cases.append({'id': key, 'question': question, 'corpus': 'insuranceqa', 'report': None,
                      'cohort': 'synthetic_personal', 'gold_answer_ids': [], 'expected_abstain': True,
                      'annotation_origin': 'synthetic personal-policy question against public FAQ only'})
    requests = [{k: c[k] for k in ('question', 'corpus', 'report')} for c in cases[:normal_count] + cases[-2:]]
    results = retrieve_batch(requests, args.device)
    retrieved = {c['id']: r for c, r in zip(cases[:normal_count] + cases[-2:], results)}
    for i, c in enumerate(cases[normal_count:normal_count+4]):
        r = dict(retrieved[c['intervention_base_id']])
        if i < 2: r['results'] = []
        else:
            donor = next(x for x in cases[:8] if x['report'] != c['report'])
            r['results'] = retrieved[donor['id']]['results']
            r['intervention_donor_id'] = donor['id']
        r['synthetic_intervention'] = c['cohort']
        retrieved[c['id']] = r
    args.output.mkdir(parents=True)
    write(args.output / 'cases.json', cases)
    write(args.output / 'retrieval.json', retrieved)
    write(args.output / 'protocol.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
          'base_commit': '31caa1d51ca724dddf20524cbfbed780390cce66',
          'purpose': 'exposed development diagnosis, no fresh holdout claim',
          'selection': 'first N SHA256(qwen35-dev-20261004:ID), fixed before generation',
          'counts': {'finqa': 8, 'insuranceqa': 4, 'condition': 4, 'fiqa': 4, 'synthetic': 6},
          'fixture_sha256': hashes, 'finqa_upstream_dev_sha256': digest(args.finqa_dev),
          'retrieval_input_fields': ['question', 'corpus', 'report'], 'gold_injected': False,
          'retrieval_output_top_k': 5, 'candidate_budget': 'dense100 union positive BM25 top100',
          'context_chars': 8000, 'per_page_chars': 2400, 'max_answer_pages': 5,
          'thinking': False, 'num_ctx': 4096, 'num_predict': 384, 'temperature': 0, 'seed': 42,
          'cases_sha256': digest(args.output / 'cases.json'), 'retrieval_sha256': digest(args.output / 'retrieval.json')})
    print(json.dumps({'prepared': len(cases), 'retrieved': len(requests)}), flush=True)


def resources(base_url):
    import requests
    result = {'utc': datetime.now(timezone.utc).isoformat()}
    try: result['ollama_ps'] = requests.get(base_url + '/api/ps', timeout=10).json()
    except requests.RequestException as e: result['ollama_ps_error'] = type(e).__name__
    try:
        result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError): result['gpu'] = None
    return result


def run(args):
    import requests
    from src.insurerag_vlm.vlm import VLMClient
    protocol = json.loads((args.data / 'protocol.json').read_text(encoding='utf8'))
    for name in ('cases', 'retrieval'):
        if digest(args.data / (name + '.json')) != protocol[name + '_sha256']: raise ValueError('Frozen diagnostic inputs changed')
    cases = json.loads((args.data / 'cases.json').read_text(encoding='utf8'))
    retrieval = json.loads((args.data / 'retrieval.json').read_text(encoding='utf8'))
    if args.ids: cases = [c for c in cases if c['id'] in args.ids.split(',')]
    if not cases: raise ValueError('No selected cases')
    if args.output.exists(): raise ValueError('Output directory already exists')
    args.output.mkdir(parents=True)
    baseline = resources(args.base_url)
    # Unload only models on the isolated, explicitly supplied local service.
    for item in baseline.get('ollama_ps', {}).get('models', []):
        r = requests.post(args.base_url + '/api/generate', json={'model': item['name'], 'keep_alive': 0}, timeout=120)
        r.raise_for_status()
    client = VLMClient('ollama:' + args.model, ollama_base_url=args.base_url,
                       expected_model_digest=args.expected_digest, request_timeout=300,
                       generation_options={k: protocol[k] for k in ('num_ctx', 'num_predict', 'temperature', 'seed')})
    write(args.output / 'environment.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
          'model': client.backend_metadata(), 'before': baseline, 'diagnostic_protocol_sha256': digest(args.data / 'protocol.json'),
          'prompt_mode': args.prompt_mode,
          'json_input_encoding': 'utf-8',
          'script_sha256': digest(Path(__file__)), 'answer_integration_sha256': digest(ROOT / 'scripts/answer_evidence_reranker.py')})
    for i, case in enumerate(cases):
        started = time.perf_counter()
        try: result = answer_retrieval(retrieval[case['id']], client, prompt_mode=args.prompt_mode)
        except Exception as exc:
            result = {'error': type(exc).__name__, 'message': str(exc), 'generation': client.backend_metadata()}
        row = {'id': case['id'], 'cohort': case['cohort'], **result, 'observed_resources': resources(args.base_url),
               'wall_seconds_including_postprocessing': time.perf_counter() - started}
        with (args.output / 'predictions.jsonl').open('a', encoding='utf8') as f: f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'model': args.model, 'case': case['id'], 'completed': i+1, 'total': len(cases),
                          'seconds': round(row['wall_seconds_including_postprocessing'], 2), 'error': row.get('error'),
                          'abstain': row.get('served', {}).get('abstain')}), flush=True)
    write(args.output / 'completion.json', {'completed_utc': datetime.now(timezone.utc).isoformat(), 'count': len(cases),
                                          'predictions_sha256': digest(args.output / 'predictions.jsonl')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    a = subs.add_parser('prepare'); a.add_argument('--finqa-dev', type=Path, required=True)
    a.add_argument('--device', default='cpu', choices=['cpu','cuda']); a.add_argument('--output', type=Path, required=True)
    a = subs.add_parser('run'); a.add_argument('--data', type=Path, required=True); a.add_argument('--model', required=True)
    a.add_argument('--expected-digest', required=True); a.add_argument('--base-url', required=True)
    a.add_argument('--ids'); a.add_argument('--output', type=Path, required=True)
    a.add_argument('--prompt-mode', choices=['main', 'research'], default='main')
    args = p.parse_args(); (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__': main()
