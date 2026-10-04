"""Register the query-only intervention and selection rule before gradients."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import sha, write_json


def main():
    run = ROOT / 'reports/query_adaptation_v1'
    path = run / 'selection_protocol.json'
    if path.exists(): raise ValueError('Plan already exists')
    data = ROOT / 'data/training/query_adaptation_v1'
    cache = run / 'index_cache/manifest.json'
    assert cache.exists()
    fixed = ROOT / 'reports/condition_listwise_v1/selection.lock.json'
    reranker = json.loads(fixed.read_text(encoding='utf8'))
    code = ['scripts/train_query_adaptation.py', 'src/insurerag_vlm/query_adaptation.py',
            'scripts/eval_insuranceqa_reranker.py', 'scripts/eval_insuranceqa_scale.py',
            'src/insurerag_vlm/reranker.py', 'src/insurerag_vlm/domain_reranker.py']
    plan = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'human_local_date': '2026-10-02',
        'research_question': 'Can training only the query encoder improve same-budget candidate recall and downstream retrieval?',
        'seeds': [42, 123], 'epochs_considered': [1, 2], 'query_blend_alphas': [.5, 1.],
        'training': {'epochs': 2, 'learning_rate': 8e-6, 'micro_batch': 16, 'accumulation': 2,
                     'temperature': .05, 'repeat_factors': {'insuranceqa': 2, 'government': 8, 'general': 1, 'finance': 2},
                     'retention_weights': {'insuranceqa': .25, 'government': 1., 'general': 1., 'finance': .5},
                     'loss': 'full allowed-corpus multi-positive softmax + KL(original BGE query distribution || adapted)',
                     'pooling': 'cls', 'max_query_tokens': 512, 'normalization': 'l2',
                     'precision': 'FP32 parameters/logits with BF16 encoder autocast',
                     'document_encoder_updated': False, 'reranker_updated': False},
        'ranking': {'pool': 'union200', 'lexical_weight': .2, 'cross_weight': .5,
                    'dense_candidates': 100, 'bm25_candidates': 100},
        'selection': {'split': 'valid', 'objective': 'weighted pipeline Hit@10',
                      'domain_weights': {'insuranceqa': .6, 'finance': .2, 'government': .1, 'general': .1},
                      'guards_max_drop_from_original_pipeline': {'insuranceqa': .005, 'finance': .01, 'government': 2/127, 'general': .005},
                      'promotion': 'Strictly exceed eligible original-query pipeline weighted Hit@10, or retain original BGE query encoder.',
                      'tie_breaks': ['weighted MRR@100', 'weighted Hit@1', 'stable registered candidate order']},
        'validation_questions': {'insuranceqa': 2000, 'government': 127, 'general': 400, 'finance': 500},
        'test_questions': {'historical_insuranceqa': 2000, 'historical_government': 416,
                           'previous_government': 81, 'previous_general': 600, 'previous_condition_government': 157,
                           'new_finance_fiqa': 648},
        'test_controls': ['original BGE dense', 'BM25', 'original RRF', 'original-query fixed-reranker pipeline',
                          'selected adapted-query dense', 'selected adapted-query RRF', 'selected adapted-query same-budget pipeline'],
        'test_policy': 'Only validation-selected query checkpoint/alpha plus unchanged controls. No post-test checkpoint changes, label repairs, exclusions, or training.',
        'statistical_plan': 'Paired 5000-repeat bootstrap: shared-positive connected components for FAQ/FiQA, source URL/title for government/general. Report Hit@10, Hit@1, MRR@100, nDCG@10, and candidate coverage. Historical/near-duplicate sensitivities descriptive only.',
        'initial_query_weights_sha256': sha(ROOT / '../models/bge-small-en-v1.5/model.safetensors'),
        'fixed_reranker_weights_sha256': reranker['selected_weights_sha256'],
        'fixed_reranker_selection_sha256': sha(fixed),
        'training_manifest_sha256': sha(data / 'manifest.lock.json'),
        'document_index_manifest_sha256': sha(cache),
        'fiqa_fixture_sha256': sha(ROOT / 'data/benchmarks/fiqa_v1/manifest.lock.json'),
        'frozen_code_sha256': {name: sha(ROOT / name) for name in code},
        'evaluation_implementation': 'Will be separately hashed before the first validation evaluation, against this pre-training contract.',
        'limitations': ['Original historical FAQ label/source overlap remains disclosed.',
                        'FiQA is broader finance, not an insurance-only test.',
                        'Public pretraining exposure is unknown. The fixed insurance reranker can have prior near-question exposure; audit separately.',
                        'Repeated validation use and multiple checkpoint/configuration comparisons limit statistical interpretation.'],
    }
    write_json(plan, path)
    print(json.dumps({'registered': str(path), 'sha256': sha(path), 'query_checkpoints': 4, 'candidate_configurations': 9}))


if __name__ == '__main__':
    main()
