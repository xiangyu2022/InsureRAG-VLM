"""No-key CLI smoke using existing public PDF snapshots; not a quality study."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,write,sha

def main():
    out=LOCAL/'offline_smoke';out.mkdir(exist_ok=False)
    docs=out/'documents';docs.mkdir();index=out/'index';index.mkdir()
    files=[]
    for name in ['wi_renters_pi017.pdf','wi_auto_pi057.pdf']:
        source=ROOT/'data/benchmarks/research_v2/sources'/name
        shutil.copyfile(source,docs/name)
        files.append({'repository_path':str(source.relative_to(ROOT)).replace('\\','/'),'sha256':sha(source)})
    env=dict(os.environ,INSURERAG_USE_OLLAMA='0',INSURERAG_VLM_MODEL='local-extractive',
             INSURERAG_RETRIEVAL_MODEL='local-hashing',INSURERAG_CORPUS_SOURCE='documents',PYTHONIOENCODING='utf-8')
    for key in ['OPENAI_API_KEY','HF_API_TOKEN','ANTHROPIC_API_KEY']:env.pop(key,None)
    relative=lambda p:str(p.relative_to(ROOT)).replace('\\','/')
    commands=[['build-index',relative(docs),'--index-dir',relative(index),'--corpus-source','documents'],
              ['generate-qa',relative(docs),'--output-dir',relative(index),'--target-count','50'],
              ['retrieval-metrics',relative(docs),relative(index/'qa_pairs.jsonl'),'--index-dir',relative(index),'--top-k','10','--corpus-source','documents'],
              ['query',relative(docs),'What coverage limits or deductibles are described in the documents?',
               '--index-dir',relative(index),'--corpus-source','documents','--json']]
    stages=[]
    for i,args in enumerate(commands):
        start=time.perf_counter();result=subprocess.run([sys.executable,'main.py',*args],cwd=ROOT,env=env,capture_output=True,encoding='utf8')
        (out/(str(i)+'.stdout.txt')).write_text(result.stdout,encoding='utf8')
        (out/(str(i)+'.stderr.txt')).write_text(result.stderr,encoding='utf8')
        stages.append({'command':['python','main.py',*args],'exit_code':result.returncode,'seconds':time.perf_counter()-start})
        if result.returncode:raise RuntimeError('Offline smoke stage failed; inspect local logs: '+str(i))
    artifacts={name:{'rows':sum(bool(l.strip()) for l in (index/name).read_text(encoding='utf8').splitlines()),'sha256':sha(index/name)}
               for name in ['qa_pairs.jsonl','hard_negatives.jsonl']}
    if any(v['rows']==0 for v in artifacts.values()):raise ValueError('Smoke generated empty artifacts')
    report={'status':'passed','stages':stages,'input_existing_public_snapshots':files,'artifacts':artifacts,
            'backends':{'retrieval':'local-hashing','generation':'local-extractive'},'external_api_keys_removed':True,
            'downloads':False,'training':False,'meaning':'CLI execution and nonempty artifact check only. Generated QA smoke labels do not establish retrieval or answer quality.'}
    write(ROOT/'reports/source_holdout_v1/offline_smoke.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
