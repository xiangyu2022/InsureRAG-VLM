"""Multidomain rehearsal with frozen-teacher KL regularization, actual CUDA training."""
import argparse
from datetime import datetime,timezone
import json,math,random,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.train_domain_reranker import sample_group


def run(args):
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError('CUDA required by recorded protocol')
    lock=json.loads((args.data/'manifest.lock.json').read_text(encoding='utf8'))
    for name,expected in lock['files'].items():
        if sha(args.data/name)!=expected:raise ValueError('Training data changed')
    groups=read_jsonl(args.data/'train_groups.jsonl');answers={a['id']:a['text'] for a in read_jsonl(args.data/'answers.jsonl')}
    # Extra government presentations are explicitly counted, never new unique examples.
    indices=[i for i,g in enumerate(groups) for _ in range(4 if g['source_domain']=='government' else 1)]
    args.output.mkdir(parents=True,exist_ok=False);args.checkpoints.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,dtype=torch.float32).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01,fused=True)
    micro=4;accumulation=4;steps_per_epoch=math.ceil(math.ceil(len(indices)/micro)/accumulation)
    total_steps=steps_per_epoch*args.epochs;warmup=round(total_steps*.1)
    schedule=lambda step:(step+1)/max(1,warmup) if step<warmup else max(0.,(total_steps-step)/max(1,total_steps-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':args.seed,'recipe':args.recipe,
        'unique_training_questions':len(groups),'question_presentations_per_epoch':len(indices),'government_repeat_factor':4,
        'epochs':args.epochs,'optimizer':'AdamW','learning_rate':args.learning_rate,'weight_decay':.01,
        'warmup_steps':warmup,'optimizer_steps_planned':total_steps,'micro_batch_groups':micro,'gradient_accumulation':accumulation,
        'supervised_loss':'listwise cross entropy, label smoothing 0.05','teacher_loss':'KL(teacher || student) on same six candidate pairs',
        'temperature':2.,'teacher_weight_insurance':.15,'teacher_weight_other':.5,
        'precision':'FP32 parameters, BF16 autocast','trainable_parameters':sum(p.numel() for p in model.parameters()),
        'initial_weights_sha256':sha(args.model/'model.safetensors'),'training_manifest_sha256':sha(args.data/'manifest.lock.json'),
        'public_teacher':lock['teacher'],'code_sha256':sha(Path(__file__)),'sampler_code_sha256':sha(ROOT/'scripts/train_domain_reranker.py'),
        'test_metrics_used':False,'scope':'Domain adaptation plus rehearsal, not Qwen training'}
    write_json(protocol,args.output/'protocol.json');step=0;pairs_seen=0;start=time.perf_counter();checkpoints=[]
    optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    with (args.output/'training_log.jsonl').open('w',encoding='utf8') as log:
        for epoch in range(1,args.epochs+1):
            model.train();rng=random.Random(args.seed+epoch);order=list(indices);rng.shuffle(order)
            group_count=0;ce_total=0.;kl_total=0.;micro_batches=math.ceil(len(order)/micro)
            for bi,offset in enumerate(range(0,len(order),micro)):
                batch=[groups[i] for i in order[offset:offset+micro]];questions=[];passages=[];teacher=[];weights=[]
                for g in batch:
                    chosen=sample_group(g,rng,epoch)
                    questions.extend([g['question']]*6);passages.extend(answers[a] for a in chosen)
                    teacher.append([g['teacher_scores'][a] for a in chosen]);weights.append(.15 if g['source_domain']=='insuranceqa' else .5)
                features=tokenizer(questions,passages,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    logits=model(**features).logits.reshape(len(batch),6).float()
                    ce=torch.nn.functional.cross_entropy(logits,torch.zeros(len(batch),device='cuda',dtype=torch.long),label_smoothing=.05,reduction='none')
                    teacher_logits=torch.tensor(teacher,device='cuda',dtype=torch.float32)
                    kl=torch.nn.functional.kl_div(torch.log_softmax(logits/2.,dim=1),torch.softmax(teacher_logits/2.,dim=1),reduction='none').sum(dim=1)*4.
                    loss=(ce+torch.tensor(weights,device='cuda')*kl).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                window_start=(bi//accumulation)*accumulation;window_size=min(accumulation,micro_batches-window_start)
                (loss/window_size).backward();group_count+=len(batch);pairs_seen+=len(questions)
                ce_total+=float(ce.detach().sum());kl_total+=float(kl.detach().sum())
                if (bi+1)%accumulation==0 or bi+1==micro_batches:
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                    if step%200==0 or bi+1==micro_batches:
                        row={'epoch':epoch,'optimizer_step':step,'groups_seen_in_epoch':group_count,'mean_supervised_loss':ce_total/group_count,
                            'mean_teacher_kl':kl_total/group_count,'pairs_seen':pairs_seen,'seconds':round(time.perf_counter()-start,1)}
                        log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            folder=args.checkpoints/f'epoch-{epoch}';model.eval();model.save_pretrained(folder,safe_serialization=True);tokenizer.save_pretrained(folder)
            provenance={'epoch':epoch,'optimizer_steps':step,'fine_tuned_in_this_project':True,
                'weights_sha256':sha(folder/'model.safetensors'),'initial_weights_sha256':protocol['initial_weights_sha256'],
                'training_manifest_sha256':protocol['training_manifest_sha256'],'training_protocol_sha256':sha(args.output/'protocol.json'),
                'training_code_sha256':protocol['code_sha256'],'recipe':args.recipe,'unique_training_questions':len(groups),'pair_exposures':pairs_seen}
            write_json(provenance,folder/'training_provenance.json');checkpoints.append(provenance)
            write_json({'checkpoints':checkpoints},args.output/'checkpoints.json')
    if sha(Path(__file__))!=protocol['code_sha256']:raise ValueError('Trainer changed during execution')
    write_json({'status':'completed','seconds':time.perf_counter()-start,'epochs':args.epochs,'optimizer_steps':step,'pair_exposures':pairs_seen,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoints':checkpoints,'log_sha256':sha(args.output/'training_log.jsonl')},args.output/'completion.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--recipe',default='rehearsal_from_v2');p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=5e-6);p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
