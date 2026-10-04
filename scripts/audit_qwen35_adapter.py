#!/usr/bin/env python3
"""CPU-only audit of every saved LoRA tensor and pinned Hub file provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.smoke_qwen35_lora import MODEL_ID,REVISION,package_versions,sha_file,write_json


def git_blob_sha1(path):
    path = Path(path)
    digest = hashlib.sha1(('blob '+str(path.stat().st_size)+'\0').encode())
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def authenticate_files(snapshot,model_files,info):
    if info.get('sha')!=REVISION:
        raise ValueError('Hub metadata returned a different revision')
    remote = {row['rfilename']:row for row in info['siblings']}
    checks = {}
    for name,local in model_files.items():
        if name not in remote:
            checks[name] = {'matched':False,'reason':'file absent from pinned Hub metadata'}
            continue
        item = remote[name]
        if item.get('lfs',{}).get('sha256'):
            method = 'sha256_of_LFS_content'
            expected,observed = item['lfs']['sha256'],local['sha256']
        else:
            method = 'git_blob_sha1'
            expected,observed = item.get('blobId'),git_blob_sha1(snapshot/name)
        checks[name] = {'method':method,'expected':expected,'observed':observed,
            'size_matches':item.get('size')==local['bytes'],
            'matched':bool(expected and expected==observed and item.get('size')==local['bytes'])}
    return checks


def audit(args):
    # Explicitly hide CUDA before importing the CUDA-capable Torch package.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    source = args.run.resolve()
    report_path = source/'report.json'
    original = json.loads(report_path.read_text(encoding='utf-8'))
    result = {'role':'independent_cpu_adapter_tensor_and_source_provenance_audit',
        'started_utc':datetime.now(timezone.utc).isoformat(),'source_run':str(source),
        'source_report_sha256':sha_file(report_path),'source_code_sha256':sha_file(Path(__file__)),
        'library_versions':package_versions(),'forward_passes':0,'optimizer_steps':0,'device':'cpu',
        'status':'running'}
    try:
        if original.get('status')!='compatibility_smoke_passed' or original.get('model_id')!=MODEL_ID or original.get('model_revision')!=REVISION:
            raise ValueError('Expected a completed smoke of the pinned Qwen3.5-0.8B base')
        snapshot = Path(original['model_snapshot']).resolve()
        adapter = source/'adapter'
        actual_files = {}
        for name,expected in original['model_files'].items():
            path = snapshot/name
            actual_files[name] = {'sha256':sha_file(path),'bytes':path.stat().st_size}
            if actual_files[name]!=expected:
                raise ValueError('Base file differs from the source run: '+name)
        result['source_model_files_unchanged'] = True
        for name,expected in original['artifact_files'].items():
            # Original Windows manifests use backslashes; accept them on any OS.
            path = source/Path(name.replace('\\','/'))
            if sha_file(path)!=expected['sha256'] or path.stat().st_size!=expected['bytes']:
                raise ValueError('Saved artifact differs from source run: '+name)
        result['source_artifacts_unchanged'] = True
        config = json.loads((adapter/'adapter_config.json').read_text(encoding='utf-8'))
        if config.get('base_model_name_or_path')!=MODEL_ID or config.get('revision')!=REVISION:
            raise ValueError('Adapter metadata is not canonical and revision-pinned')
        result['canonical_adapter_identity'] = {'model_id':MODEL_ID,'revision':REVISION}

        url = f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true'
        result['official_metadata_url'] = url
        try:
            with urllib.request.urlopen(url,timeout=30) as response:
                info = json.load(response)
            write_json(output/'official_hf_metadata.json',info)
            checks = authenticate_files(snapshot,actual_files,info)
            result['official_source_authentication'] = {'available':True,'all_files_match':all(row['matched'] for row in checks.values()),
                'files':checks,'metadata_sha256':sha_file(output/'official_hf_metadata.json')}
            if not result['official_source_authentication']['all_files_match']:
                raise ValueError('At least one local file differs from the pinned official Hub source')
        except (OSError,TimeoutError,json.JSONDecodeError) as exc:
            result['official_source_authentication'] = {'available':False,'error':type(exc).__name__+': '+str(exc)}

        import torch
        from peft import PeftModel,get_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForImageTextToText
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA was initialized in a CPU-only audit')
        base = AutoModelForImageTextToText.from_pretrained(str(snapshot),local_files_only=True,
            trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='sdpa',use_kernels=False)
        if torch.cuda.is_initialized():
            raise RuntimeError('Base loading initialized CUDA in a CPU-only audit')
        if type(base).__name__!='Qwen3_5ForConditionalGeneration':
            raise RuntimeError('Unexpected conditional model class')
        # PEFT otherwise infers an adapter-loading device independently of the
        # already CPU-resident base and can initialize a CUDA context.
        model = PeftModel.from_pretrained(base,str(adapter),is_trainable=False,local_files_only=True,torch_device='cpu')
        model.eval()
        if any(param.device.type!='cpu' for param in model.parameters()):
            raise RuntimeError('A parameter was loaded outside CPU memory')
        serialized = load_file(str(adapter/'adapter_model.safetensors'),device='cpu')
        loaded = get_peft_model_state_dict(model,adapter_name='default',save_embedding_layers=False)
        missing,extra = sorted(set(serialized)-set(loaded)),sorted(set(loaded)-set(serialized))
        result['tensor_count_serialized'] = len(serialized)
        result['tensor_count_loaded'] = len(loaded)
        result['missing_tensor_keys'],result['extra_tensor_keys'] = missing,extra
        tensors = {}
        for key in sorted(set(serialized)&set(loaded)):
            saved,current = serialized[key],loaded[key].detach().cpu()
            matches = saved.shape==current.shape and saved.dtype==current.dtype and torch.equal(saved,current)
            tensors[key] = {'serialized_shape':list(saved.shape),'loaded_shape':list(current.shape),
                'serialized_dtype':str(saved.dtype),'loaded_dtype':str(current.dtype),'exact_match':bool(matches),
                'serialized_sha256':hashlib.sha256(saved.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                'loaded_sha256':hashlib.sha256(current.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        write_json(output/'tensor_comparison.json',tensors)
        result['tensor_comparison_sha256'] = sha_file(output/'tensor_comparison.json')
        result['matching_tensor_count'] = sum(row['exact_match'] for row in tensors.values())
        result['serialized_adapter_parameter_count'] = sum(tensor.numel() for tensor in serialized.values())
        result['loaded_adapter_parameter_count'] = sum(tensor.numel() for tensor in loaded.values())
        result['all_tensors_exactly_equal'] = (not missing and not extra and bool(tensors)
            and result['matching_tensor_count']==len(serialized)==original['lora']['trainable_tensors'])
        result['cuda_initialized'] = torch.cuda.is_initialized()
        if not result['all_tensors_exactly_equal'] or result['cuda_initialized']:
            raise RuntimeError('Exact CPU adapter-state verification failed')
        result['status'] = 'passed'
        result['limitations'] = ['Checks serialization and exact CPU loading, not predictive quality or held-out generalization.',
            'No forward pass, training step, or GPU operation is performed.',
            'Official source verification is reported separately and is unavailable if the metadata service cannot be reached.']
        return result
    except Exception as exc:
        result.update({'status':'failed','error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        result['wall_seconds'] = time.perf_counter()-started
        write_json(output/'audit_report.json',result)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New audit directory; the source run is never changed')
    return p


if __name__=='__main__':
    result = audit(parser().parse_args())
    print(json.dumps({key:result[key] for key in ['status','matching_tensor_count','serialized_adapter_parameter_count',
        'all_tensors_exactly_equal','cuda_initialized','wall_seconds']},indent=2))
