"""Post-selection diagnosis, matched data ablation, and historical parity audit."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json
from scripts.eval_query_adaptation import specs, partitions, paired, summarize, CFG
from src.insurerag_vlm.reranker import blend_scores

RUN = ROOT / 'reports/query_adaptation_v1'
def load(p): return json.loads(p.read_text(encoding='utf8'))
def rows_by_key(path):
    rows = {}
    for r in read_jsonl(path):
        key = (r['cohort'], r['arm'], r['id'])
        assert key not in rows
        rows[key] = r
    return rows


def main():
    lock = load(RUN / 'selection.lock.json')
    chosen = lock['config']['name']
    test = load(RUN / 'test/summary.json')
    assert test['predictions_sha256'] == sha(RUN / 'test/predictions.jsonl')
    pred = rows_by_key(RUN / 'test/predictions.jsonl')
    prior = rows_by_key(ROOT / 'reports/condition_listwise_v1/test_evaluation/predictions.jsonl')
    mapping = {'insuranceqa': 'historical_insuranceqa', 'multidomain': 'historical_multidomain',
               'historical_government': 'historical_government', 'condition_government': 'fresh_government'}
    parity, stages, cards = {}, {}, []
    for cohort, qfile, afile, _ in specs('test'):
        cases = read_jsonl(ROOT / qfile)
        answers = {a['id']: a for a in read_jsonl(ROOT / afile)}
        raw = {r['id']: r for r in read_jsonl(RUN / 'test' / f'{cohort}_scores.jsonl')}
        if cohort in mapping:
            checks = Counter()
            for c in cases:
                new = pred[cohort, 'original', c['id']]
                old = prior[mapping[cohort], 'selected', c['id']]
                for k in ['hit_at_1', 'hit_at_5', 'hit_at_10', 'hit_at_100', 'mrr_at_100', 'label_recall_at_10']:
                    checks[k+'_changed_rows'] += int(abs(new[k]-old[k]) > 1e-12)
                checks['top100_order_changed_rows'] += int(new['top_answer_ids'] != old['top_answer_ids'])
            assert not any(checks[k+'_changed_rows'] for k in ['hit_at_1', 'hit_at_10', 'mrr_at_100']), checks
            parity[cohort] = {'n': len(cases), **dict(checks)}
        for part, ix in partitions(cohort, cases).items():
            for arm in ['original', chosen]:
                counts = Counter()
                for i in ix:
                    r = pred[cohort, arm, cases[i]['id']]
                    counts['hit10' if r['hit_at_10'] else 'candidate_missing' if not r['candidate_hit'] else 'reranking_miss'] += 1
                stages[f'{cohort}/{part}/{arm}'] = {'n': len(ix), **dict(counts)}
        for c in cases:
            old, new = [pred[cohort, arm, c['id']] for arm in ['original', chosen]]
            if new['hit_at_10'] and old['hit_at_10']: continue
            row = raw[c['id']]
            cross = dict(zip(row['candidate_ids'], row['cross']))
            detail = {}
            for arm, output in [('original', old), (chosen, new)]:
                a = row['arms'][arm]
                scores = blend_scores(np.array(a['dense']), np.array(a['sparse']),
                                      np.array([cross[x] for x in a['candidate_ids']]),
                                      CFG['lexical_weight'], CFG['cross_weight'])
                order = np.array(a['candidate_ids'])[np.argsort(-scores, kind='stable')].tolist()
                assert order[:100] == output['top_answer_ids']
                positions = [i+1 for i, aid in enumerate(order) if aid in c['gold_answer_ids']]
                detail[arm] = {'gold_rank_in_full_union': min(positions) if positions else None,
                               'candidate_hit': bool(output['candidate_hit']), 'hit_at_10': bool(output['hit_at_10']),
                               'top3': [{'id': aid, 'text': answers[aid]['text']} for aid in order[:3]]}
            cards.append({'cohort': cohort, 'id': c['id'], 'question': c['question'],
                          'domain': c.get('domain'), 'source_url': c.get('source_url'),
                          'transition': 'gain' if new['hit_at_10'] else 'loss' if old['hit_at_10'] else 'persistent_miss',
                          'gold_answers': [{'id': aid, 'text': answers[aid]['text']} for aid in c['gold_answer_ids']],
                          'arms': detail, 'expert_adjudicated': False, 'used_for_training_or_selection': False})
    missed = [c['id'] for c in load(RUN / 'recall_failure_diagnosis.json')['cases']]
    assert len(missed) == 194
    recall = {'old_missing_n': len(missed),
              'entered_candidates': sum(bool(pred['insuranceqa', chosen, i]['candidate_hit']) for i in missed),
              'recovered_hit10': sum(bool(pred['insuranceqa', chosen, i]['hit_at_10']) for i in missed),
              'new_candidate_losses': sum(bool(pred['insuranceqa', 'original', i]['candidate_hit']) and not pred['insuranceqa', chosen, i]['candidate_hit']
                                          for co, arm, i in pred if co == 'insuranceqa' and arm == 'original')}
    fixture = read_jsonl(ROOT / 'data/benchmarks/fiqa_v1/test.jsonl')
    audit = load(RUN / 'data_verification.json')
    keyword_ids = set(audit['insurance_keyword_slice']['test'])
    subsets = {'exclude_one_prior_reranker_near_question': [c for c in fixture if c['id'] != 'fiqa_q_3446'],
               'insurance_keywords_only_descriptive': [c for c in fixture if c['id'] in keyword_ids]}
    sensitivity = {}
    for name, cases in subsets.items():
        rows = {a: [pred['fiqa', a, c['id']] for c in cases] for a in ['bge', 'original', chosen]}
        sensitivity[name] = {'n': len(cases), 'summaries': {a: summarize(r) for a, r in rows.items()},
                             'paired_selected_minus_original': paired(rows[chosen], rows['original'], cases, 'fiqa')}
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'historical_control_parity': parity,
                'failure_stages': stages, 'historical_faq_recall_transition': recall, 'fiqa_sensitivity': sensitivity,
                'test_predictions_sha256': sha(RUN / 'test/predictions.jsonl'),
                'selection_unchanged_sha256': sha(RUN / 'selection.lock.json'),
                'post_test_diagnostic_only': True, 'code_sha256': sha(Path(__file__))}, RUN / 'failure_analysis.json')
    write_json({'expert_adjudicated': False, 'n': len(cards), 'cases': cards}, RUN / 'failure_cases.json')

    # Validation-only matched-budget contrast; it cannot change the frozen primary selection.
    mainvalid, abvalid = load(RUN / 'valid/summary.json'), load(RUN / 'data_ablation/valid/summary.json')
    mainpred, abpred = rows_by_key(RUN / 'valid/predictions.jsonl'), rows_by_key(RUN / 'data_ablation/valid/predictions.jsonl')
    comparisons, selected_pair = {}, {}
    for cohort, qfile, _, _ in specs('valid'):
        cases = read_jsonl(ROOT / qfile)
        for part, ix in partitions(cohort, cases).items():
            key = f'{cohort}/{part}'
            a, b = mainvalid['summaries'][cohort][part], abvalid['summaries'][cohort][part]
            configs = [n for n in a if n.startswith('seed_')]
            comparisons[key] = {n: {metric: a[n][metric]-b[n][metric]
                                   for metric in ['hit_at_1', 'hit_at_10', 'mrr_at_100', 'candidate_hit']} for n in configs}
            subset = [cases[i] for i in ix]
            selected_pair[key] = paired([mainpred[cohort, chosen, c['id']] for c in subset],
                                        [abpred[cohort, chosen, c['id']] for c in subset], subset, cohort)
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'selected_configuration': chosen,
                'registered_after_primary_validation_before_new_test': True, 'selection_changed': False,
                'all_matched_config_differences_added_finance_minus_old_replay': comparisons,
                'selected_configuration_paired': selected_pair,
                'unlabelled_finance_documents_in_both_arms': True,
                'interpretation': 'Validation-only labelled-data effect conditional on fixed document corpus and optimization budget; not a fresh-test data ablation.',
                'primary_summary_sha256': sha(RUN / 'valid/summary.json'),
                'ablation_summary_sha256': sha(RUN / 'data_ablation/valid/summary.json')}, RUN / 'data_ablation_results.json')
    print(json.dumps({'failure_cards': len(cards), 'faq_recall': recall, 'parity': parity}))


if __name__ == '__main__': main()
