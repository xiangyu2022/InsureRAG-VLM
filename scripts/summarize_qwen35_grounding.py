"""Verify comparison artifacts and summarize bounded diagnostic measurements."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def load(path): return json.loads(path.read_text(encoding='utf8'))
def rows(path): return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def reference_agreement(value, reference):
    if value is None: return False
    target = Decimal(reference.replace('%', '').replace(',', ''))
    # A descriptive dev metric, not FinQA execution accuracy: percentage
    # answers within 0.1 percentage point; decimal counts .01; integers exact.
    tolerance = Decimal('.1') if '%' in reference else Decimal('.01') if '.' in reference else Decimal('0')
    return abs(Decimal(str(value)) - target) <= tolerance


def runtime_summary(predictions):
    meta = [r['generation']['last_generation'] for r in predictions]
    warm = [m['wall_seconds'] for m in meta[1:]]
    rates = [m['eval_count'] / (m['eval_duration'] / 1e9) for m in meta if m.get('eval_duration')]
    ps = [r['observed_resources']['ollama_ps']['models'][0] for r in predictions]
    snapshots = [int(r['observed_resources']['gpu'].split(',')[1]) for r in predictions if r['observed_resources'].get('gpu')]
    free_snapshots = [int(r['observed_resources']['gpu'].split(',')[2]) for r in predictions if r['observed_resources'].get('gpu')]
    return {'n': len(meta), 'first_request_seconds': meta[0]['wall_seconds'],
            'warm_median_seconds': statistics.median(warm),
            'warm_p95_seconds': sorted(warm)[max(0, math.ceil(.95*len(warm))-1)],
            'median_decode_tokens_per_second': statistics.median(rates),
            'maximum_observed_gpu_used_mib': max(snapshots), 'continuous_peak_measured': False,
            'minimum_observed_gpu_free_mib': min(free_snapshots),
            'maximum_ollama_resident_bytes': max(p['size'] for p in ps),
            'all_postrequest_snapshots_fully_gpu_resident': all(p['size_vram'] == p['size'] for p in ps),
            'max_prompt_tokens': max(m['prompt_eval_count'] for m in meta),
            'truncated_outputs': sum(bool(m['truncated']) for m in meta),
            'single_pass_latency_not_a_production_SLA': True}


def summarize(root, arm_prefix='utf8_'):
    protocol = load(root / 'dev/protocol.json')
    for name in ('cases', 'retrieval'):
        assert sha(root / 'dev' / (name+'.json')) == protocol[name+'_sha256']
    cases = {c['id']: c for c in load(root / 'dev/cases.json')}
    retrieval = load(root / 'dev/retrieval.json')
    review = load(root / 'numeric_review.json')
    result = {'evaluation_status': 'exposed exploratory development diagnostic; no expert accuracy or fresh holdout claim',
              'reference_metric': review['metric'], 'retrieval': {}, 'answer_arms': {}, 'arithmetic_arms': {}}
    for cohort in ('finqa', 'insuranceqa', 'condition', 'fiqa'):
        selected = [c for c in cases.values() if c['cohort'] == cohort]
        samples = []
        for c in selected:
            gold = set(c['gold_answer_ids']); r = retrieval[c['id']]
            got = {a['answer_id'] for a in r['results']}; candidate = set(r['candidate_ids'])
            samples.append({'hit': bool(gold & got), 'full': gold <= got, 'recall': len(gold & got)/len(gold),
                            'candidate_hit': bool(gold & candidate), 'candidate_full': gold <= candidate})
        result['retrieval'][cohort] = {'n': len(samples), 'hit_at_5_count': sum(s['hit'] for s in samples),
              'all_evidence_at_5_count': sum(s['full'] for s in samples),
              'macro_gold_recall_at_5': statistics.mean(s['recall'] for s in samples),
              'candidate_hit_count': sum(s['candidate_hit'] for s in samples),
              'candidate_all_evidence_count': sum(s['candidate_full'] for s in samples)}
    for name in ('baseline_4b', 'baseline_9b', 'revised_4b', 'revised_9b'):
        path = root / (arm_prefix + name) / 'predictions.jsonl'; completion = load(root / (arm_prefix + name) / 'completion.json')
        assert sha(path) == completion['predictions_sha256']
        predictions = rows(path); assert {r['id'] for r in predictions} == set(cases)
        assert all('error' not in r for r in predictions)
        public = [r for r in predictions if not r['cohort'].startswith('synthetic')]
        synthetic = [r for r in predictions if r['cohort'].startswith('synthetic')]
        unknown = {}
        valid = 0
        for r in public:
            ids = set(re.findall(r'\b(?:finqa|insuranceqa|condition|fiqa):[A-Za-z0-9_]+', r['raw_answer']))
            known = {f"{retrieval[r['id']]['corpus']}:{item['answer_id']}" for item in retrieval[r['id']]['results']}
            if ids & known: valid += 1
            if ids-known: unknown[r['id']] = sorted(ids-known)
        annotations = review['arms'][name]
        assert {a['id'] for a in annotations} == {c['id'] for c in cases.values() if c['cohort'] == 'finqa'}
        for a in annotations: a['reference_agreement'] = reference_agreement(a['primary_numeric_answer'], cases[a['id']]['reference_answer'])
        result['answer_arms'][name] = {
            'runtime': runtime_summary(predictions), 'public_questions': len(public),
            'public_served_nonabstain_count': sum(not r['served']['abstain'] for r in public),
            'public_postprocessing_repair_count': sum(r['served']['answer_repaired'] for r in public),
            'public_rejection_reasons': dict(Counter(r['served']['citation_support_reason'] for r in public if r['served']['abstain'])),
            'synthetic_controls': len(synthetic), 'synthetic_served_abstain_count': sum(r['served']['abstain'] for r in synthetic),
            'synthetic_explicit_model_abstain_count': sum(r['served']['explicit_abstention'] for r in synthetic),
            'public_with_known_corpus_id_count': valid, 'unknown_corpus_ids': unknown,
            'citation_id_checks_do_not_establish_entailment': True,
            'finqa_primary_numeric_reference_agreement': sum(a['reference_agreement'] for a in annotations),
            'finqa_n': len(annotations), 'numeric_annotations': annotations,
        }
    for name in ('arithmetic_4b', 'arithmetic_9b', 'arithmetic_v2_4b', 'arithmetic_v2_9b'):
        # First-pass rejected schemas are historical audit evidence, optional
        # when reconstructing only the final diagnostic from this revision.
        if name in ('arithmetic_4b', 'arithmetic_9b') and not (root / name / 'completion.json').exists():
            continue
        path = root / name / 'predictions.jsonl'; completion = load(root / name / 'completion.json')
        assert sha(path) == completion['predictions_sha256']
        predictions = rows(path); assert len(predictions) == completion['count'] == 12
        public = [r for r in predictions if r['cohort'] == 'finqa']
        verified = [r for r in public if r['arithmetic_audit']['status'] == 'inputs_and_arithmetic_verified']
        result['arithmetic_arms'][name] = {'runtime': runtime_summary(predictions), 'public_questions': len(public),
            'statuses': dict(Counter(r['arithmetic_audit']['status'] for r in public)),
            'computed_reference_agreement': sum(reference_agreement(r['arithmetic_audit']['computed_result'], cases[r['id']]['reference_answer']) for r in verified),
            'verified_arithmetic_with_wrong_reference_count': sum(not reference_agreement(r['arithmetic_audit']['computed_result'], cases[r['id']]['reference_answer']) for r in verified),
            'synthetic_abstentions': sum(r['arithmetic_audit']['status'] == 'model_abstained' for r in predictions if r['cohort'].startswith('synthetic')),
            'n_synthetic': 4, 'semantics_verified': False,
            'details': [{'id': r['id'], **r['arithmetic_audit']} for r in public]}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm-prefix', default='utf8_')
    args = p.parse_args(); result = summarize(args.root, args.arm_prefix)
    (args.root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: {n: {key: val for key, val in v.items() if key in ('finqa_primary_numeric_reference_agreement', 'public_served_nonabstain_count', 'computed_reference_agreement', 'statuses')} for n,v in result[k].items()} for k in ('answer_arms','arithmetic_arms')}))
