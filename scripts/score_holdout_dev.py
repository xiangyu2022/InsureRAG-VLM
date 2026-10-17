"""Score a complete reviewed standalone DEV run; optionally lock its selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.insurerag_vlm.holdout_dev_runner import fingerprint, readiness
from src.insurerag_vlm.holdout_dev_scoring import score_run, select_arm, write_selection_lock
from src.insurerag_vlm.holdout_standalone import read_records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['documents','manifest','requests','clearance','support-mapping','run','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--lock',type=Path)
    parser.add_argument('--freeze-artifacts',type=Path)
    args = parser.parse_args()
    read = lambda p:json.loads(p.read_text(encoding='utf8'))
    docs = [json.loads(line) for line in args.documents.read_text(encoding='utf8').splitlines()]
    manifest, requests, clearance = read(args.manifest), read(args.requests), read(args.clearance)
    check = readiness(docs, manifest, requests, clearance)
    if not check['ready']:
        print(json.dumps(check))
        return 2
    if sha(args.support_mapping) != clearance['support_mapping_sha256']:
        raise ValueError('Reviewed support mapping file binding mismatch')
    saved = read(args.run/'standalone/run_manifest.json')
    bindings = saved['bindings']
    wanted = dict(source_file_sha256=sha(args.documents), source_manifest_sha256=sha(args.manifest),
                  clearance_sha256=sha(args.clearance),
                  protocol_sha256=sha(ROOT/'reports/holdout1000_v1/evaluation_protocol_v1.json'))
    if any(bindings.get(k)!=v for k,v in wanted.items()) or bindings.get('execution')!='standalone':
        raise ValueError('Scoring inputs differ from standalone inference bindings')
    expected_code = {'scripts/run_holdout_dev_retrieval.py', 'scripts/eval_insuranceqa_scale.py',
                     'src/insurerag_vlm/holdout_corpus.py', 'src/insurerag_vlm/holdout_dev_runner.py',
                     'src/insurerag_vlm/holdout_retrieval.py', 'src/insurerag_vlm/query_adaptation.py',
                     'src/insurerag_vlm/reranker.py', 'src/insurerag_vlm/domain_reranker.py',
                     'src/insurerag_vlm/holdout_standalone.py', 'src/insurerag_vlm/holdout_dev_scoring.py',
                     'src/insurerag_vlm/holdout_experiment.py', 'scripts/score_holdout_dev.py'}
    if set(bindings.get('code', {})) != expected_code:
        raise ValueError('Complete registered code bindings required')
    if not bindings.get('models') or any(not files.get('model.safetensors') or not files.get('tokenizer_config.json')
                                       for files in bindings['models'].values()):
        raise ValueError('Retrieval model and tokenizer bindings required')
    for name,digest in bindings['code'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest:
            raise ValueError('Inference/scoring code changed after DEV run')
    chunks = [json.loads(line) for line in (args.run/'corpus_chunks.jsonl').read_text(encoding='utf8').splitlines()]
    if fingerprint(chunks)!=bindings['chunks_sha256']:
        raise ValueError('Run corpus chunk binding mismatch')
    ids = [c['id'] for c in chunks]
    run, records = read_records(requests,ids,args.run/'standalone',bindings)
    mapping = read(args.support_mapping)
    report = score_run(requests,ids,mapping,run,records)
    report['input_files_sha256'] = wanted | dict(support_mapping_sha256=sha(args.support_mapping),
                                               retained_pool_sha256=clearance['retained_pool_sha256'])
    report['selection'] = select_arm(report)
    frozen = None
    if args.lock:
        if args.freeze_artifacts is None:
            parser.error('--lock requires reviewed --freeze-artifacts')
        specification = read(args.freeze_artifacts)
        if (specification.get('status')!='REVIEWED_FREEZE_ARTIFACTS'
                or specification.get('model_and_tokenizer_hashes_verified') is not True):
            raise ValueError('Explicit reviewed model/tokenizer freeze receipt required')
        names = {'prompt','packing','rubric','generation_model_receipt'}
        if set(specification.get('files',{}))!=names:
            raise ValueError('Prompt, packing, rubric and generation-model receipt required')
        files = {}
        for name, entry in specification['files'].items():
            path = (args.freeze_artifacts.parent/entry['path']).resolve()
            if sha(path)!=entry['sha256']:
                raise ValueError('Freeze artifact file fingerprint mismatch')
            files[name] = path
        generation = read(files['generation_model_receipt'])
        protocol = read(ROOT/'reports/holdout1000_v1/evaluation_protocol_v1.json')
        if (generation.get('status')!='VERIFIED_LOCAL_RUNTIME'
                or generation.get('model_digest')!=protocol['generation']['digest']
                or generation.get('tokenizer_bound_to_model_digest') is not True):
            raise ValueError('Registered generation runtime/tokenizer not verified')
        frozen = dict(dev_data=fingerprint(report['input_files_sha256'] | dict(requests_sha256=fingerprint(requests))),
                      corpus=bindings['chunks_sha256'],
                      models=fingerprint(dict(retrieval=bindings['models'],generation=sha(files['generation_model_receipt']))),
                      tokenizers=fingerprint(dict(retrieval={k:{n:h for n,h in v.items() if not n.endswith('.safetensors')}
                                                             for k,v in bindings['models'].items()},
                                                  generation=sha(files['generation_model_receipt']))),
                      code=fingerprint(bindings['code']), protocol=bindings['protocol_sha256'],
                      **{n:sha(files[n]) for n in ['prompt','packing','rubric']})
        if args.lock.exists():
            raise FileExistsError('Preserve the existing immutable DEV selection lock')
    with args.output.open('x',encoding='utf8') as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
        handle.write('\n')
    if frozen is not None:
        write_selection_lock(args.lock,report,frozen)
    print(json.dumps(dict(scored_requests=len(requests), selected_arm=report['selection']['arm'],
                          selection_locked=frozen is not None,test_inference_authorized=False)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
