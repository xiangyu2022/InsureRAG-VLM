"""Verify training exclusion, actual weight updates and both recorded retrieval evaluations."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha,verify_fixture
from scripts.prepare_reranker_training import normalize
from scripts.eval_insuranceqa_scale import aggregate,metrics


def locked_files(folder):
    lock=json.loads((folder/'manifest.lock.json').read_text(encoding='utf8'))
    for name,expected in lock['files'].items():
        if sha(folder/name)!=expected:raise ValueError('Data artifact changed: '+name)
    return lock


def verify_predictions(folder,cases,selection_path):
    summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
    if sha(folder/'predictions.jsonl')!=summary['predictions_sha256']:raise ValueError('Predictions changed')
    rows=read_jsonl(folder/'predictions.jsonl');gold={c['id']:set(c['gold_answer_ids']) for c in cases}
    for row in rows:
        if len(row['top_answer_ids'])!=len(set(row['top_answer_ids'])):raise ValueError('Duplicate ranked answer')
        recalculated=metrics(row['top_answer_ids'],gold[row['id']])
        for key,value in recalculated.items():
            if abs(row[key]-value)>1e-12:raise ValueError('Per-question metric mismatch')
    for arm,reported in summary['summaries'].items():
        group=[r for r in rows if r['arm']==arm]
        if {r['id'] for r in group}!=set(gold) or len(group)!=len(cases):raise ValueError('Missing/duplicated test cases')
        recomputed=aggregate(group)
        for key in ['n','hit_at_1','hit_at_5','hit_at_10','hit_at_100','mrr_at_100','label_recall_at_10']:
            if abs(recomputed[key]-reported[key])>1e-12:raise ValueError('Aggregate metric mismatch')
    return {'questions':len(cases),'arms':len(summary['summaries']),'all_per_question_metrics_recomputed':True,
            'all_aggregate_metrics_recomputed':True,'selection_lock_sha256':sha(selection_path)}


def run(args):
    fixture=ROOT/'data/benchmarks/insuranceqa_v2';verify_fixture(fixture)
    training=ROOT/'data/training/insuranceqa_hardneg_v2';lock=locked_files(training)
    groups=read_jsonl(training/'train_groups.jsonl')
    original={r['id']:r for r in read_jsonl(fixture/'train.jsonl')}
    answers={r['id']:normalize(r['text']) for r in read_jsonl(fixture/'answers.jsonl')}
    heldout=read_jsonl(fixture/'valid.jsonl')+read_jsonl(fixture/'test.jsonl')
    heldout_ids={a for c in heldout for a in c['gold_answer_ids']};heldout_texts={answers[a] for a in heldout_ids}
    seen=set()
    for group in groups:
        if group['question']!=original[group['id']]['question'] or set(group['positive_ids'])!=set(original[group['id']]['gold_answer_ids']):
            raise ValueError('Training question or positive label no longer matches the author data')
        text=normalize(group['question'])
        if text in seen:raise ValueError('Duplicate training question')
        seen.add(text)
        if set(group['positive_ids']) & heldout_ids:raise ValueError('Held-out positive ID in training positives')
        if {answers[a] for a in group['positive_ids']} & heldout_texts:raise ValueError('Held-out positive text in training positives')
        negative_texts=[answers[a] for a in group['negative_ids']]
        if len(set(negative_texts))!=len(negative_texts):raise ValueError('Duplicate negative text')
        if set(negative_texts)&{answers[a] for a in group['positive_ids']}:raise ValueError('Own positive used as negative')
    folder=args.run;selection_path=folder/'selection.lock.json';selection=json.loads(selection_path.read_text(encoding='utf8'))
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf8'))
    complete=json.loads((folder/'completion.json').read_text(encoding='utf8'))
    if complete['status']!='completed' or sha(folder/'training_log.jsonl')!=complete['log_sha256']:raise ValueError('Training incomplete')
    if sha(ROOT/'scripts/train_domain_reranker.py')!=protocol['code_sha256']:raise ValueError('Training code drift')
    if sha(args.base/'model.safetensors')!=protocol['base_model_files_sha256']['model.safetensors']:
        raise ValueError('Weight comparison requires the recorded public base checkpoint')
    if sha(training/'manifest.lock.json')!=selection['training_manifest_sha256']:raise ValueError('Training source drift')
    if sha(args.model/'model.safetensors')!=selection['selected_weights_sha256']:raise ValueError('Wrong selected weights')
    provenance=json.loads((args.model/'training_provenance.json').read_text(encoding='utf8'))
    if provenance['training_protocol_sha256']!=sha(folder/'protocol.json'):raise ValueError('Weight provenance mismatch')
    for name,expected in selection['code_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Selection/evaluation code drift')
    for epoch,artifact in selection['validation_artifacts'].items():
        for filename,key in [('protocol.json','protocol_sha256'),('scores.jsonl','scores_sha256')]:
            if sha(folder/f'valid_epoch_{epoch}'/filename)!=artifact[key]:raise ValueError('Validation artifact drift')
    test_protocol=json.loads((folder/'test_selected/protocol.json').read_text(encoding='utf8'))
    if test_protocol['selection_lock_sha256']!=sha(selection_path):raise ValueError('Test selection mismatch')
    test_summary=json.loads((folder/'test_evaluation/summary.json').read_text(encoding='utf8'))
    if test_summary['test_scores_sha256']!=sha(folder/'test_selected/scores.jsonl'):raise ValueError('Test score drift')
    for relative,expected in test_protocol['code_sha256'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Neural scoring code drift')
    test_check=verify_predictions(folder/'test_evaluation',read_jsonl(fixture/'test.jsonl'),selection_path)
    benchmark=ROOT/'data/benchmarks/hicric_government_qa_v1';locked_files(benchmark)
    transfer=ROOT/'reports/hicric_government_qa_v1/transfer_v1'
    transfer_protocol=json.loads((transfer/'protocol.json').read_text(encoding='utf8'))
    transfer_summary=json.loads((transfer/'summary.json').read_text(encoding='utf8'))
    if transfer_protocol['selection_lock_sha256']!=sha(selection_path):raise ValueError('Transfer model was not validation-selected')
    if selection['created_utc']>=transfer_protocol['created_utc']:raise ValueError('Transfer evaluated before selection')
    if transfer_summary['protocol_sha256']!=sha(transfer/'protocol.json') or transfer_summary['scores_sha256']!=sha(transfer/'scores.jsonl'):
        raise ValueError('Transfer artifact drift')
    for relative,expected in transfer_protocol['code_sha256'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Transfer evaluation code drift')
    transfer_check=verify_predictions(transfer,read_jsonl(benchmark/'questions.jsonl'),selection_path)
    from safetensors.numpy import load_file
    base=load_file(str(args.base/'model.safetensors'));trained=load_file(str(args.model/'model.safetensors'))
    # Modern Transformers regenerates this deterministic buffer instead of saving it.
    ignored_buffers=[]
    for state in [base,trained]:
        positions=state.pop('bert.embeddings.position_ids',None)
        if positions is not None:
            if positions.ndim!=2 or not np.array_equal(positions,np.arange(positions.shape[1])[None,:]):
                raise ValueError('Unexpected nonstandard position-ID buffer')
            ignored_buffers.append('bert.embeddings.position_ids')
    if base.keys()!=trained.keys():raise ValueError('Unexpected checkpoint architecture change')
    changed=0;total=0;squared_delta=0.;changed_tensors=0
    for key in base:
        if base[key].shape!=trained[key].shape:raise ValueError('Parameter shape mismatch')
        delta=trained[key].astype(np.float64)-base[key].astype(np.float64)
        if not np.isfinite(delta).all():raise ValueError('Nonfinite learned parameters')
        count=int(np.count_nonzero(delta));changed+=count;total+=delta.size
        changed_tensors+=int(count>0);squared_delta+=float(np.sum(delta*delta))
    if changed==0:raise ValueError('No actual parameter update')
    output={'status':'verified','training_questions':len(groups),'author_positive_pairs':lock['labeled_positive_pairs'],
        'mined_negative_pairs':lock['mined_negative_pairs'],'original_question_and_positive_labels_preserved':True,
        'heldout_positive_id_collisions':0,'heldout_positive_text_collisions':0,'duplicate_training_questions':0,
        'optimizer_steps':complete['optimizer_steps'],'pair_exposures':complete['pairs_seen'],
        'selected_epoch':selection['epoch'],'actual_parameter_comparison':{'changed_elements':changed,'total_elements':total,
        'changed_tensors':changed_tensors,'total_tensors':len(base),'l2_delta':squared_delta**.5,
        'excluded_nonlearned_buffers':sorted(set(ignored_buffers))},
        'historical_test':test_check,'government_transfer':transfer_check,
        'selection_lock_sha256':sha(selection_path),'verification_code_sha256':sha(Path(__file__))}
    write_json(output,folder/'verification.json');print(json.dumps(output,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'reports/domain_training_v2')
    p.add_argument('--base',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    run(p.parse_args())
