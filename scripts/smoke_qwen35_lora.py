#!/usr/bin/env python3
"""Two synthetic text-only LoRA updates: compatibility evidence, never a quality benchmark."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata, util
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
TARGET_PATTERN = re.compile(
    r'model\.language_model\.layers\.\d+\.(?:'
    r'self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|'
    r'linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|'
    r'mlp\.(?:gate_proj|up_proj|down_proj))'
)
SYNTHETIC_ROWS = [
    {'record_id':'synthetic_lora_smoke_01','split':'train','source':'synthetic_policy_a.pdf#page=1',
     'evidence':'Fictional policy A lists a collision deductible of $750. This is synthetic training data.',
     'question':'What collision deductible is listed in fictional policy A?',
     'answer':'The collision deductible is $750. SOURCE: synthetic_policy_a.pdf#page=1'},
    {'record_id':'synthetic_lora_smoke_02','split':'train','source':'synthetic_policy_b.pdf#page=1',
     'evidence':'Fictional policy B lists a wind deductible of $1,250. No earthquake limit is stated. This is synthetic training data.',
     'question':'What earthquake limit is stated in fictional policy B?',
     'answer':'The provided excerpt does not state an earthquake limit. SOURCE: insufficient_evidence'},
]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def package_versions():
    result = {}
    for name in ['torch','transformers','peft','accelerate','safetensors','tokenizers','huggingface-hub']:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def validate_snapshot(path):
    """Use a pinned local HF snapshot; never download an alternate checkpoint."""
    path = Path(path).absolute()
    if path.name!=REVISION or path.parent.name!='snapshots' or path.parent.parent.name!='models--Qwen--Qwen3.5-0.8B':
        raise ValueError('Expected the pinned Hugging Face snapshot directory for '+MODEL_ID+' at '+REVISION)
    required = ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','model.safetensors.index.json']
    missing = [name for name in required if not (path/name).is_file()]
    if missing:
        raise ValueError('Incomplete local snapshot: '+', '.join(missing))
    config = json.loads((path/'config.json').read_text(encoding='utf-8'))
    text_config = config.get('text_config',{})
    expected = {'hidden_size':1024,'num_hidden_layers':24,'vocab_size':248320}
    if config.get('model_type')!='qwen3_5' or config.get('architectures')!=['Qwen3_5ForConditionalGeneration'] or any(text_config.get(key)!=value for key,value in expected.items()):
        raise ValueError('Local config is not the expected Qwen3.5-0.8B conditional-generation architecture')
    index = json.loads((path/'model.safetensors.index.json').read_text(encoding='utf-8'))
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(not (path/name).is_file() or Path(name).name!=name for name in shards):
        raise ValueError('Local model weight shard is missing or has an invalid path')
    files = sorted({*required,*shards,*[name for name in ['vocab.json','merges.txt','preprocessor_config.json','video_preprocessor_config.json'] if (path/name).is_file()]})
    return path,config,{name:{'sha256':sha_file(path/name),'bytes':(path/name).stat().st_size} for name in files}


def tokenize_rows(tokenizer,rows=SYNTHETIC_ROWS,max_length=256):
    if not 2<=max_length<=256:
        raise ValueError('This bounded compatibility smoke requires max_length between 2 and 256')
    if not getattr(tokenizer,'chat_template',None):
        raise ValueError('The checkpoint chat template is required')
    tokenized,audit = [],[]
    for row in rows:
        messages = [
            {'role':'system','content':'Answer only from the synthetic policy excerpt and cite its source. Abstain when unsupported.'},
            {'role':'user','content':'Evidence: '+row['evidence']+'\nSource: '+row['source']+'\nQuestion: '+row['question']},
            {'role':'assistant','content':row['answer']},
        ]
        prompt = tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        full = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
        prompt_ids = tokenizer(prompt,add_special_tokens=False)['input_ids']
        full_ids = tokenizer(full,add_special_tokens=False)['input_ids']
        if full_ids[:len(prompt_ids)]!=prompt_ids:
            raise ValueError('Prompt tokens are not an exact prefix of the complete conversation: '+row['record_id'])
        if len(full_ids)>max_length or len(full_ids)<=len(prompt_ids):
            raise ValueError('Complete assistant supervision must fit without truncation: '+row['record_id'])
        labels = [-100]*len(prompt_ids)+full_ids[len(prompt_ids):]
        if not any(label!=-100 for label in labels[1:]):
            raise ValueError('No causal next-token assistant targets: '+row['record_id'])
        tokenized.append({'input_ids':full_ids,'attention_mask':[1]*len(full_ids),'labels':labels})
        audit.append({'record_id':row['record_id'],'prompt_tokens':len(prompt_ids),'full_tokens':len(full_ids),
            'assistant_target_tokens':len(full_ids)-len(prompt_ids),'prompt_is_exact_token_prefix':True,
            'truncated':False,'input_modality':'text-only'})
    return tokenized,audit


def expected_target_names(names):
    return sorted(name for name in names if TARGET_PATTERN.fullmatch(name))


def set_portable_adapter_identity(model):
    """PEFT otherwise serializes this machine's snapshot path and no revision."""
    config = model.peft_config['default']
    config.base_model_name_or_path = MODEL_ID
    config.revision = REVISION


def verify_saved_tokenizer(original_examples,tokenizer,max_length):
    saved_examples,saved_audit = tokenize_rows(tokenizer,max_length=max_length)
    if saved_examples!=original_examples:
        raise RuntimeError('Saved tokenizer changed input IDs, assistant labels, or attention masks')
    return saved_examples,{'passed':True,'records_compared':len(saved_examples),
        'input_ids_labels_and_attention_masks_equal':True,'tokenization_audit':saved_audit}


def preflight(args):
    versions = package_versions()
    if versions.get('transformers')!='5.17.0':
        raise RuntimeError('This compatibility experiment pins transformers==5.17.0')
    path,config,files = validate_snapshot(args.model_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples,audit = tokenize_rows(tokenizer,max_length=args.max_length)
    dataset_json = ''.join(json.dumps(row,sort_keys=True)+'\n' for row in SYNTHETIC_ROWS)
    report = {'role':'qwen35_text_only_lora_compatibility_smoke_not_quality_evaluation',
        'smoke_version':'v2_portable_adapter_and_tokenizer_reload',
        'model_id':MODEL_ID,'model_revision':REVISION,'model_snapshot':str(path),
        'snapshot_revision_check':'pinned Hugging Face snapshot path; actual file hashes recorded',
        'model_files':files,'model_type':config['model_type'],'loader_class':'Qwen3_5ForConditionalGeneration',
        'auto_loader':'AutoModelForImageTextToText','library_versions':versions,
        'synthetic_dataset_sha256':hashlib.sha256(dataset_json.encode()).hexdigest(),
        'training_record_ids':[row['record_id'] for row in SYNTHETIC_ROWS],
        'tokenization_audit':audit,'input_modality':'text-only','vision_training':False,
        'lora':{'r':4,'alpha':8,'dropout':0.0,'language_only':True,'expected_target_modules':186},
        'steps_requested':2,'max_sequence_length':args.max_length,'seed':42,
        'source_code_sha256':sha_file(Path(__file__)),
        'limitations':['Exactly two synthetic records and two optimizer updates; no held-out evaluation.',
            'Finite losses, changed adapter weights, and reload parity establish compatibility only.',
            'No accuracy, loss-improvement, generalization, visual-training, or Qwen3.5-4B-training claim.']}
    return path,tokenizer,examples,report,dataset_json


def run(args):
    path,tokenizer,examples,report,dataset_json = preflight(args)
    if args.preflight:
        report['status'] = 'tokenizer_and_snapshot_preflight_passed_no_model_loaded'
        print(json.dumps(report,indent=2))
        return report
    if args.output is None:
        raise ValueError('--output is required for a GPU smoke run')
    output = args.output.absolute()
    output.mkdir(parents=True,exist_ok=False)
    report.update({'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'})
    write_json(output/'report.json',report)
    (output/'synthetic_train.jsonl').write_text(dataset_json,encoding='utf-8')
    started = time.perf_counter()
    try:
        import torch
        from peft import LoraConfig,PeftModel,get_peft_model
        from transformers import AutoModelForImageTextToText,AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('An available BF16-capable CUDA GPU is required; no CPU fallback')
        optional = {name:util.find_spec(name) is not None for name in ['fla','causal_conv1d','triton']}
        if any(optional.values()):
            raise RuntimeError('Use the isolated reference-kernel environment without optional FLA/causal-conv1d/Triton')
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(0)
        report.update({'cuda_device':torch.cuda.get_device_name(0),'cuda_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'torch_cuda_version':torch.version.cuda,'optional_kernels':optional,'delta_rule_backend':'reference PyTorch',
            'precision':'BF16 base weights; PEFT adapter dtype recorded below','quantization':None})
        report['current_stage'] = 'load_base_and_attach_language_adapters'
        load_kwargs = dict(local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',use_kernels=False)
        load_started = time.perf_counter()
        base = AutoModelForImageTextToText.from_pretrained(str(path),**load_kwargs)
        if type(base).__name__!='Qwen3_5ForConditionalGeneration':
            raise RuntimeError('Unexpected model loader class: '+type(base).__name__)
        base.config.use_cache = False
        base.config.text_config.use_cache = False
        targets = expected_target_names(name for name,module in base.named_modules() if isinstance(module,torch.nn.Linear))
        if len(targets)!=186:
            raise RuntimeError('Expected exactly 186 language-only LoRA modules, found '+str(len(targets)))
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model = get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,bias='none',
            task_type='CAUSAL_LM',target_modules=targets))
        trainable = [(name,param) for name,param in model.named_parameters() if param.requires_grad]
        if not trainable or any('lora_' not in name or '.language_model.' not in name or '.visual.' in name for name,_ in trainable):
            raise RuntimeError('Only language-model LoRA tensors may be trainable')
        report['lora'].update({'target_modules':targets,'trainable_parameters':sum(p.numel() for _,p in trainable),
            'trainable_tensors':len(trainable),'trainable_dtypes':sorted({str(p.dtype) for _,p in trainable})})
        report['base_parameters_including_adapter'] = sum(p.numel() for p in model.parameters())
        before = {name:param.detach().cpu().clone() for name,param in trainable}
        optimizer = torch.optim.AdamW([param for _,param in trainable],lr=1e-4,weight_decay=0.0,foreach=False)
        torch.cuda.synchronize()
        report['load_and_adapter_seconds'] = time.perf_counter()-load_started
        report['optimizer'] = {'name':'AdamW','learning_rate':1e-4,'weight_decay':0.0,'gradient_clip_norm':1.0}
        steps = []
        report['training_steps'] = steps
        report['current_stage'] = 'two_synthetic_optimizer_steps'
        model.train()
        for step,example in enumerate(examples,start=1):
            step_started = time.perf_counter()
            batch = {name:torch.tensor([values],dtype=torch.long,device='cuda:0') for name,values in example.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch,use_cache=False).loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            if any(param.grad is None for _,param in trainable):
                raise RuntimeError('At least one selected LoRA tensor did not receive a gradient')
            finite = bool(torch.stack([torch.isfinite(param.grad).all() for _,param in trainable]).all())
            if not finite:
                raise RuntimeError('Non-finite adapter gradient')
            grad_norm = float(torch.nn.utils.clip_grad_norm_([param for _,param in trainable],1.0).detach())
            if not math.isfinite(grad_norm):
                raise RuntimeError('Non-finite combined adapter gradient norm')
            optimizer.step()
            if not bool(torch.stack([torch.isfinite(param).all() for _,param in trainable]).all()):
                raise RuntimeError('Optimizer produced a non-finite adapter weight')
            torch.cuda.synchronize()
            steps.append({'step':step,'record_id':SYNTHETIC_ROWS[step-1]['record_id'],'loss':float(loss.detach()),
                'all_adapter_gradients_finite':finite,'gradient_norm_before_clipping':grad_norm,
                'wall_seconds':time.perf_counter()-step_started,'tokens':len(example['input_ids'])})
            print(json.dumps(steps[-1]),flush=True)
            del loss,batch
        del optimizer
        changed = sum(not torch.equal(before[name],param.detach().cpu()) for name,param in trainable)
        if not changed:
            raise RuntimeError('No adapter tensor changed after the optimizer updates')
        report['training_steps'] = steps
        report['changed_adapter_tensors'] = changed
        del before
        model.eval()
        eval_batch = {name:torch.tensor([values],dtype=torch.long,device='cuda:0') for name,values in examples[0].items() if name!='labels'}
        with torch.no_grad():
            reference = model(**eval_batch,use_cache=False).logits.detach().float().cpu()
        if not bool(torch.isfinite(reference).all()):
            raise RuntimeError('Non-finite pre-save evaluation logits')
        adapter_path = output/'adapter'
        report['current_stage'] = 'save_adapter_and_tokenizer'
        set_portable_adapter_identity(model)
        # Embeddings were frozen and never resized. Explicit false avoids PEFT
        # consulting the Hub to infer this after the canonical ID is restored.
        model.save_pretrained(str(adapter_path),safe_serialization=True,save_embedding_layers=False)
        tokenizer.save_pretrained(str(adapter_path))
        saved_config = json.loads((adapter_path/'adapter_config.json').read_text(encoding='utf-8'))
        if saved_config.get('base_model_name_or_path')!=MODEL_ID or saved_config.get('revision')!=REVISION:
            raise RuntimeError('Saved adapter lost its canonical base identity or pinned revision')
        report['portable_adapter_base'] = {'model_id':MODEL_ID,'revision':REVISION,
            'reload_route':'explicit AutoModelForImageTextToText base plus PeftModel.from_pretrained'}
        saved_tokenizer = AutoTokenizer.from_pretrained(str(adapter_path),local_files_only=True,trust_remote_code=False)
        reloaded_examples,report['tokenizer_reload_parity'] = verify_saved_tokenizer(examples,saved_tokenizer,args.max_length)
        provenance = {key:report[key] for key in ['model_id','model_revision','model_files','library_versions',
            'synthetic_dataset_sha256','training_record_ids','input_modality','vision_training','lora','source_code_sha256']}
        write_json(adapter_path/'smoke_provenance.json',provenance)
        report['training_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
        report['training_peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
        del trainable,model,base
        gc.collect()
        torch.cuda.empty_cache()
        report['current_stage'] = 'reload_fresh_base_and_compare_forward_logits'
        reload_started = time.perf_counter()
        reloaded_base = AutoModelForImageTextToText.from_pretrained(str(path),**load_kwargs)
        reloaded_base.config.use_cache = False
        reloaded_base.config.text_config.use_cache = False
        reloaded = PeftModel.from_pretrained(reloaded_base,str(adapter_path),is_trainable=False)
        reloaded.eval()
        eval_batch = {name:torch.tensor([values],dtype=torch.long,device='cuda:0') for name,values in reloaded_examples[0].items() if name!='labels'}
        with torch.no_grad():
            observed = reloaded(**eval_batch,use_cache=False).logits.detach().float().cpu()
        if not bool(torch.isfinite(observed).all()):
            raise RuntimeError('Non-finite evaluation logits after saved-adapter reload')
        max_difference = float((observed-reference).abs().max())
        # BF16 storage and implementation-specific reduction order permit a small
        # numerical tolerance; exact equality is recorded independently.
        matches = bool(torch.allclose(observed,reference,atol=0.02,rtol=0.002))
        report['reload_parity'] = {'passed':matches,'bitwise_equal_float32_logits':torch.equal(observed,reference),
            'maximum_absolute_difference':max_difference,'atol':0.02,'rtol':0.002,
            'compared_shape':list(reference.shape),'scope':'full forward logits for synthetic training row 1, eval mode',
            'wall_seconds':time.perf_counter()-reload_started}
        if not matches:
            raise RuntimeError('Saved adapter reload logits failed the declared parity tolerance')
        report['artifact_files'] = {str(file.relative_to(output)):{'sha256':sha_file(file),'bytes':file.stat().st_size}
            for file in sorted(adapter_path.rglob('*')) if file.is_file()}
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
        report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
        report['status'] = 'compatibility_smoke_passed'
        report['current_stage'] = 'complete'
        return report
    except Exception as exc:
        report.update({'status':'failed','error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if 'torch' in locals() and torch.cuda.is_available():
            report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
            report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['total_wall_seconds_after_preflight'] = time.perf_counter()-started
        write_json(output/'report.json',report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,required=True,help='Pinned local Hugging Face snapshot directory; no downloads')
    p.add_argument('--output',type=Path,help='New directory for adapter, two training rows, and compatibility report')
    p.add_argument('--max-length',type=int,default=256,help='Complete synthetic conversations must fit; at most 256')
    p.add_argument('--preflight',action='store_true',help='Validate snapshot and tokenizer only; never load model weights or use CUDA')
    return p


if __name__=='__main__':
    run(parser().parse_args())
