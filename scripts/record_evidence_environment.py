"""Record the actual local training runtime without loading another GPU model."""
from datetime import datetime,timezone
from importlib.metadata import version,PackageNotFoundError
import json,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha,write_json


if __name__=='__main__':
    out=ROOT/'reports/evidence_reranker_v1/environment.json'
    assert not out.exists()
    packages={}
    for name in ['torch','transformers','tokenizers','safetensors','numpy','scipy','scikit-learn','threadpoolctl','huggingface-hub']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).strip()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'python_executable':sys.executable,
            'python_version':sys.version,'platform':platform.platform(),'machine':platform.machine(),
            'packages':packages,'gpu_name_memory_MiB_driver_csv':gpu,
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
            'workspace_includes_uncommitted_changes':True,'code_sha256':sha(Path(__file__)),
            'training_precision':'FP32 weights/optimizer with BF16 autocast forward',
            'evaluation_precision':'DomainCrossEncoder FP16 CUDA weights/inference; query encoding FP32',
            'tf32_matmul':False,'torch_cpu_threads':4,'seed_values':[42,123],
            'strict_bitwise_determinism_requested':False,
            'limits':'Pinned seeds, source and weights are reproducibility records; bitwise identical GPU retraining is not guaranteed.'}
    write_json(result,out);print(json.dumps(result))
