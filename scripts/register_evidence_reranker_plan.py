"""Freeze head/evidence ranking training, controls and selection before gradients."""
from datetime import datetime,timezone
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha,write_json
from scripts.eval_evidence_reranker import specs,PREVIOUS,QUERY

if __name__=='__main__':
    run=ROOT/'reports/evidence_reranker_v1';path=run/'selection_protocol.json';assert not path.exists()
    data=ROOT/'data/training/evidence_reranker_v1';assert (data/'manifest.lock.json').exists()
    code=['scripts/train_evidence_reranker.py','scripts/eval_evidence_reranker.py','src/insurerag_vlm/evidence_ranking.py',
          'src/insurerag_vlm/finqa_evidence.py','scripts/eval_query_adaptation.py','scripts/eval_insuranceqa_reranker.py',
          'scripts/eval_insuranceqa_scale.py','scripts/analyze_condition_secondary_metrics.py','scripts/prepare_insuranceqa.py',
          'scripts/prepare_query_adaptation_data.py','src/insurerag_vlm/query_adaptation.py','src/insurerag_vlm/reranker.py',
          'src/insurerag_vlm/domain_reranker.py','scripts/run_evidence_reranker_experiment.py']
    files={p for split in ['valid','test'] for _,q,a,_ in specs(split) for p in [q,a]}
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'human_local_date':'2026-10-02',
                'question':'Can refreshed reranker training recover head ranking and complete financial-report evidence while preserving FAQ recall?',
                'seeds':[42,123],'epochs':[1,2],'cross_weights':[.5,.75,1.],
                'fixed_candidate_policy':'prior query alpha .5, dense100 union positive BM25 top100, lexical weight .2',
                'finqa_scope':'known annual report (company/year), all released pages, no question/gold-dependent passage formatting',
                'initial_reranker_weights_sha256':sha(PREVIOUS/'model.safetensors'),
                'fixed_query_weights_sha256':sha(QUERY/'model.safetensors'),
                'prior_selection_sha256':sha(ROOT/'reports/query_adaptation_v1/selection.lock.json'),
                'training_manifest_sha256':sha(data/'manifest.lock.json'),
                'frozen_code_sha256':{p:sha(ROOT/p) for p in code},'fixture_files_sha256':{p:sha(ROOT/p) for p in sorted(files)},
                'training':{'epochs':2,'learning_rate':3e-6,'micro_groups':8,'accumulation':4,
                            'repeat_factors':{'insuranceqa':2,'government':8,'general':1,'finance':1,'financial_report':3},
                            'retention_weights':{'insuranceqa':.2,'government':.3,'general':.3,'finance':.1,'financial_report':.1},
                            'pair_weight':.5,'sampled_pairs_per_query':6,'head_rank_weight':'absolute reciprocal-rank swap difference, floor .05; detached',
                            'loss':'listwise CE(label smoothing .02) + .5 head-weighted softplus margin + frozen previous-teacher KL(T=2)',
                            'ambiguous_negative_weight':.5,'ambiguity_rule':'non-FinQA negative teacher logit >= sampled positive minus 1; never becomes a positive',
                            'precision':'FP32 parameters, BF16 encoder autocast, FP32 loss','max_pair_tokens':512},
                'selection':{'primary':'weighted MRR@100 across old domains plus complete-evidence@5 for financial reports',
                             'domain_weights':{'insuranceqa':.5,'finance':.15,'government':.1,'general':.1,'financial_report':.15},
                             'guards':[{'domain':'insuranceqa','metric':'hit_at_10','max_drop':.005},
                                       {'domain':'insuranceqa','metric':'hit_at_1','max_drop':0.},
                                       {'domain':'finance','metric':'hit_at_10','max_drop':.01},
                                       {'domain':'government','metric':'hit_at_10','max_drop':2/127},
                                       {'domain':'general','metric':'hit_at_10','max_drop':.005},
                                       {'domain':'financial_report','metric':'all_evidence_at_5','max_drop':.01}],
                             'guard_reference':'previous_cw50 on same frozen candidates',
                             'promotion':'strictly greater utility than validation-best eligible old-reranker weight; tie FAQ Hit@1, then stable enumeration',
                             'fallback':'validation-best eligible old-reranker configuration',
                             'comparisons':'old default, independently validation-tuned old model, old same selected fusion weight, public BGE'},
                'no_test_based_model_choice':True,'public_finqa_test_pretraining_exposure':'unknown',
                'no_numerical_execution_or_generated_answer_accuracy_claim':True},path)
    print(json.dumps({'registered':sha(path)}),flush=True)
