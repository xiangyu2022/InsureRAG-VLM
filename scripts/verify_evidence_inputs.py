"""Check fixed encoders against the inherited pre-experiment file inventories."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha


def verify_fixed_query_inputs():
    prior=ROOT/'reports/query_adaptation_v1'
    base=ROOT/'../models/bge-small-en-v1.5'
    manifest=json.loads((prior/'index_cache/manifest.json').read_text(encoding='utf8'))
    base_files=manifest['embedding']['checkpoint_files_sha256']
    for n,h in base_files.items():assert sha(base/n)==h,n
    lock=json.loads((prior/'checkpoint_files.lock.json').read_text(encoding='utf8'))
    prefix='models/insurerag-query-adaptation-v1-seed-42/epoch-2/'
    adapted={n:h for n,h in lock['files_sha256'].items() if n.startswith(prefix)}
    assert len(adapted)>=5 and prefix+'model.safetensors' in adapted
    for n,h in adapted.items():assert sha(ROOT.parent/n)==h,n
    actual={p.relative_to(ROOT.parent).as_posix():sha(p) for p in (ROOT.parent/prefix).iterdir()
            if p.is_file() and p.suffix in {'.json','.txt','.safetensors'}}
    assert actual==adapted, 'Query inference file set differs from the earlier delivery'
    return {'base_encoder_files_sha256':base_files,'adapted_query_files_sha256':adapted,
            'prior_index_manifest_sha256':sha(prior/'index_cache/manifest.json'),
            'prior_checkpoint_inventory_sha256':sha(prior/'checkpoint_files.lock.json')}


if __name__=='__main__':
    result=verify_fixed_query_inputs()
    print(json.dumps({'verified':True,'base_files':len(result['base_encoder_files_sha256']),
                      'adapted_query_files':len(result['adapted_query_files_sha256'])}))
