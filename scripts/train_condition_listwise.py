"""Listwise ranking with ambiguous-negative weights and previous-specialist retention."""
import argparse,json,math,random,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha

def sample(group,rng,presentation):
    positive=group['positive_ids'][presentation % len(group['positive_ids'])]
    negatives=[]
    for source,count in [('student_hard',2),('lexical_hard',2),('random',1)]:
        pool=[a for a in group['negative_ids'] if group['negative_sources'][a]==source]
        if len(pool)<count:raise ValueError('Missing negative category')
        negatives.extend(rng.sample(pool,count))
    if positive in negatives or len(set(negatives))!=5:raise ValueError('Invalid training group')
    return [positive]+negatives

def confidence(teacher):
    """Ambiguous is lower-confidence unlabeled, never an invented positive label."""
    return [.25 if score>=teacher[0]-1 else 1. for score in teacher[1:]]

def supervised_loss(logits,negative_confidence):
    import torch
    if logits.ndim!=2 or negative_confidence.shape!=(logits.shape[0],logits.shape[1]-1):raise ValueError('Loss shape mismatch')
    if not torch.isfinite(logits).all() or not torch.isfinite(negative_confidence).all() or not ((negative_confidence>0)&(negative_confidence<=1)).all():raise ValueError('Invalid logits/confidence')
    adjusted=torch.cat([logits[:,:1],logits[:,1:]+torch.log(negative_confidence)],dim=1)
    return torch.nn.functional.cross_entropy(adjusted,torch.zeros(logits.shape[0],device=logits.device,dtype=torch.long),label_smoothing=.02,reduction='none')

def run(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    plan=ROOT/'reports/condition_listwise_v1/selection_protocol.json'
    frozen=json.loads(plan.read_text(encoding='utf8'))
    assert sha(Path(__file__))==frozen['training_code_sha256'] and args.seed in frozen['seeds']
    data=ROOT/'data/training/condition_listwise_v1';lock=json.loads((data/'manifest.lock.json').read_text(encoding='utf8'))
    for n,h in lock['files'].items():assert sha(data/n)==h
    groups=read_jsonl(data/'train_groups.jsonl');answers={a['id']:a['text'] for a in read_jsonl(data/'answers.jsonl')}
    modelpath=ROOT/'../models/insurerag-retention-v2/epoch-2'
    assert sha(modelpath/'model.safetensors')==frozen['initial_weights_sha256']
    assert lock['anchor_model']['files_sha256']['model.safetensors']==frozen['initial_weights_sha256']
    output=ROOT/f'reports/condition_listwise_v1/train_seed_{args.seed}';checkpoints=ROOT/f'../models/insurerag-condition-listwise-v1-seed-{args.seed}'
    output.mkdir(parents=True,exist_ok=False);checkpoints.mkdir(exist_ok=False)
    indices=[i for i,g in enumerate(groups) for _ in range({'insuranceqa':2,'government':8,'general':1}[g['source_domain']])]
    tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(modelpath,local_files_only=True,trust_remote_code=False,dtype=torch.float32).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-6,weight_decay=.01,fused=True)
    micro=4;accumulation=4;steps=math.ceil(math.ceil(len(indices)/micro)/accumulation);warmup=round(steps*.1)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:(s+1)/max(1,warmup) if s<warmup else max(0.,(steps-s)/max(1,steps-warmup)))
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':args.seed,'unique_training_questions':len(groups),
        'domain_counts':dict(Counter(g['source_domain'] for g in groups)),'presentations_per_epoch':len(indices),
        'repeat_factors':{'insuranceqa':2,'government':8,'general':1},'epochs':1,'optimizer_steps_planned':steps,
        'learning_rate':3e-6,'optimizer':'AdamW','weight_decay':.01,'warmup_steps':warmup,'micro_groups':micro,'accumulation':accumulation,
        'loss':'listwise cross entropy on confidence-adjusted negative logits + KL(previous specialist || student)',
        'ambiguity_rule':'teacher negative >= sampled positive - 1 gives weight 0.25; otherwise 1.0',
        'kl_temperature':2.,'kl_weights':{'insuranceqa':.1,'government':.2,'general':.2},'teacher':'previous retention specialist, not public baseline','label_smoothing':.02,'precision':'FP32 parameters, BF16 autocast',
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'initial_weights_sha256':sha(modelpath/'model.safetensors'),
        'training_manifest_sha256':sha(data/'manifest.lock.json'),'selection_protocol_sha256':sha(plan),'code_sha256':sha(Path(__file__)),
        'test_metrics_used':False,'positive_sampling':'Cycle author-positive IDs across repeat presentations; seed offsets start.'}
    write_json(protocol,output/'protocol.json');rng=random.Random(args.seed);rng.shuffle(indices);presentations=Counter()
    optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();step=0;pairs=0;n=0;ranktotal=0;kl_total=0
    model.train();batches=math.ceil(len(indices)/micro)
    with (output/'training_log.jsonl').open('w',encoding='utf8') as log:
        for bi,offset in enumerate(range(0,len(indices),micro)):
            batch=[groups[i] for i in indices[offset:offset+micro]];qs=[];ps=[];ts=[];weights=[];cs=[]
            for g in batch:
                chosen=sample(g,rng,presentations[g['id']]+args.seed);presentations[g['id']]+=1
                qs.extend([g['question']]*6);ps.extend(answers[a] for a in chosen)
                t=[g['teacher_scores'][a] for a in chosen];ts.append([g['anchor_scores'][a] for a in chosen]);cs.append(confidence(t));weights.append(.1 if g['source_domain']=='insuranceqa' else .2)
            features=tokenizer(qs,ps,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(**features).logits.reshape(len(batch),6).float()
                ranking=supervised_loss(logits,torch.tensor(cs,device='cuda'))
                teacher=torch.tensor(ts,device='cuda');kl=torch.nn.functional.kl_div(torch.log_softmax(logits/2,dim=1),torch.softmax(teacher/2,dim=1),reduction='none').sum(dim=1)*4
                loss=(ranking+torch.tensor(weights,device='cuda')*kl).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            windowsize=min(accumulation,batches-(bi//accumulation)*accumulation);(loss/windowsize).backward()
            pairs+=len(qs);n+=len(batch);ranktotal+=float(ranking.detach().sum());kl_total+=float(kl.detach().sum())
            if (bi+1)%accumulation==0 or bi+1==batches:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                if step%200==0 or bi+1==batches:
                    row={'seed':args.seed,'step':step,'mean_ranking_loss':ranktotal/n,'mean_kl':kl_total/n,'pairs_seen':pairs,'seconds':round(time.perf_counter()-start,1)}
                    log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    folder=checkpoints/'epoch-1';model.eval();model.save_pretrained(folder,safe_serialization=True);tokenizer.save_pretrained(folder)
    provenance={'fine_tuned_in_this_project':True,'weights_sha256':sha(folder/'model.safetensors'),'initial_weights_sha256':protocol['initial_weights_sha256'],
        'training_manifest_sha256':protocol['training_manifest_sha256'],'training_protocol_sha256':sha(output/'protocol.json'),
        'training_code_sha256':protocol['code_sha256'],'seed':args.seed,'epoch':1,'optimizer_steps':step,'unique_training_questions':len(groups),'pair_exposures':pairs,
        'recipe':'confidence-adjusted listwise ranking + previous specialist retention'}
    write_json(provenance,folder/'training_provenance.json')
    assert sha(Path(__file__))==protocol['code_sha256']
    write_json({'status':'completed','seconds':time.perf_counter()-start,'optimizer_steps':step,'pair_exposures':pairs,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint':provenance,'log_sha256':sha(output/'training_log.jsonl')},output/'completion.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);run(p.parse_args())
