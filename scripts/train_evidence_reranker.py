"""Fine-tune the current reranker using refreshed hard negatives and expert facts."""
from collections import Counter
from datetime import datetime,timezone
import argparse,json,math,random,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,sha,write_json
from src.insurerag_vlm.evidence_ranking import head_losses,sample_training_group


def main(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    assert torch.cuda.is_available()
    run=ROOT/'reports/evidence_reranker_v1';plan=json.loads((run/'selection_protocol.json').read_text(encoding='utf8'))
    assert args.seed in plan['seeds']
    for p,h in plan['frozen_code_sha256'].items():assert sha(ROOT/p)==h,p
    data=ROOT/'data/training/evidence_reranker_v1';manifest=json.loads((data/'manifest.lock.json').read_text(encoding='utf8'))
    assert sha(data/'manifest.lock.json')==plan['training_manifest_sha256']
    for p,h in manifest['files'].items():assert sha(data/p)==h,p
    groups=read_jsonl(data/'train_groups.jsonl');answers={a['id']:a['text'] for a in read_jsonl(data/'answers.jsonl')}
    modelpath=ROOT/'../models/insurerag-condition-listwise-v1-seed-123/epoch-1'
    assert sha(modelpath/'model.safetensors')==plan['initial_reranker_weights_sha256']
    tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(modelpath,local_files_only=True,trust_remote_code=False,dtype=torch.float32).to('cuda')
    out=run/f'train_seed_{args.seed}';models=ROOT/f'../models/insurerag-evidence-reranker-v1-seed-{args.seed}'
    out.mkdir(exist_ok=False);models.mkdir(exist_ok=False)
    cfg=plan['training'];indices=[i for i,g in enumerate(groups) for _ in range(cfg['repeat_factors'][g['source_domain']])]
    micro,accumulation=cfg['micro_groups'],cfg['accumulation'];batches=math.ceil(len(indices)/micro)
    per_epoch=math.ceil(batches/accumulation);total_steps=per_epoch*cfg['epochs'];warmup=round(total_steps*.1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01,fused=True)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:(s+1)/max(1,warmup) if s<warmup else max(0.,(total_steps-s)/max(1,total_steps-warmup)))
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':args.seed,'training':cfg,
              'unique_training_questions':len(groups),'domain_counts':dict(Counter(g['source_domain'] for g in groups)),
              'query_presentations_per_epoch':len(indices),'optimizer_steps_planned':total_steps,
              'trainable_parameters':sum(p.numel() for p in model.parameters()),
              'selection_protocol_sha256':sha(run/'selection_protocol.json'),'training_manifest_sha256':sha(data/'manifest.lock.json'),
              'initial_weights_sha256':sha(modelpath/'model.safetensors'),'code_sha256':plan['frozen_code_sha256'],
              'validation_or_test_used_in_gradients':False,'known_positive_ids_never_sampled_as_negatives':True}
    write_json(protocol,out/'protocol.json');rng=random.Random(args.seed);presentations=Counter();step=pairs=queries=0;epochs=[]
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    with (out/'training_log.jsonl').open('w',encoding='utf8') as log:
        for epoch in range(1,cfg['epochs']+1):
            rng.shuffle(indices);model.train();loss_sum=kl_sum=0.;n=0
            for bi,offset in enumerate(range(0,len(indices),micro)):
                batch=[groups[i] for i in indices[offset:offset+micro]];qs=[];ps=[];teacher=[];confidence=[];weights=[]
                for g in batch:
                    sampled=sample_training_group(g,rng,presentations[g['id']]+args.seed);presentations[g['id']]+=1
                    qs.extend([g['question']]*6);ps.extend(answers[a] for a in sampled)
                    scores=[g['anchor_scores'][a] for a in sampled];teacher.append(scores)
                    confidence.append([1. if g['source_domain']=='financial_report' or s<scores[0]-1 else .5 for s in scores[1:]])
                    weights.append(cfg['retention_weights'][g['source_domain']])
                features=tokenizer(qs,ps,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    logits=model(**features).logits.reshape(len(batch),6).float()
                    rank,kl=head_losses(logits,torch.tensor(teacher,device='cuda'),torch.tensor(confidence,device='cuda'),cfg['pair_weight'])
                    loss=(rank+torch.tensor(weights,device='cuda')*kl).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                window=min(accumulation,batches-(bi//accumulation)*accumulation);(loss/window).backward()
                pairs+=len(qs);queries+=len(batch);n+=len(batch);loss_sum+=float(rank.detach().sum());kl_sum+=float(kl.detach().sum())
                if (bi+1)%accumulation==0 or bi+1==batches:
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                    if step%200==0 or bi+1==batches:
                        r={'seed':args.seed,'epoch':epoch,'step':step,'mean_rank_loss':loss_sum/n,'mean_kl':kl_sum/n,'pair_presentations':pairs,
                           'query_presentations':queries,'seconds':round(time.perf_counter()-start,1)}
                        log.write(json.dumps(r)+'\n');log.flush();print(json.dumps(r),flush=True)
            folder=models/f'epoch-{epoch}';model.eval();model.save_pretrained(folder,safe_serialization=True);tokenizer.save_pretrained(folder)
            provenance={'fine_tuned_in_this_project':True,'seed':args.seed,'epoch':epoch,'weights_sha256':sha(folder/'model.safetensors'),
                        'initial_weights_sha256':protocol['initial_weights_sha256'],'training_protocol_sha256':sha(out/'protocol.json'),
                        'optimizer_steps':step,'pair_presentations':pairs,'query_presentations':queries,
                        'recipe':'head-weighted pairwise plus listwise ranking, refreshed negatives and frozen-teacher KL'}
            write_json(provenance,folder/'training_provenance.json');epochs.append(provenance)
    for p,h in plan['frozen_code_sha256'].items():assert sha(ROOT/p)==h,p
    write_json({'status':'completed','seconds':time.perf_counter()-start,'optimizer_steps':step,'pair_presentations':pairs,'query_presentations':queries,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'epochs':epochs,'training_log_sha256':sha(out/'training_log.jsonl')},out/'completion.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);main(p.parse_args())
