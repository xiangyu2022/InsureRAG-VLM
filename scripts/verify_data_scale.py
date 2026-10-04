"""Fail on altered source bytes, labels, corpus records, or printed-page maps."""
from pathlib import Path
import json,sys
import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha,read_jsonl,verify_fixture
from scripts.build_policy_wordings import printed_footer


def verify():
    counts={}
    for name in ['insuranceqa_v2','opm_reference_navigation_v1']:
        folder=ROOT/'data/benchmarks'/name
        counts[name]=verify_fixture(folder).get('counts',{})
    for name in ['policy_wordings_v1','scaled_public_v2']:
        folder=ROOT/'data/research_corpus'/name
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        for path,digest in manifest['files'].items():
            if sha(folder/path)!=digest:raise ValueError(f'Hash mismatch: {name}/{path}')
        counts[name]=manifest['counts']
    source=ROOT/'data/research_corpus/policy_wordings_v1'
    pages={(p['doc_id'],p['page']):p for p in read_jsonl(source/'rag_pages.jsonl')}
    maps={(p['doc_id'],p['physical_page']):p for p in json.loads((source/'printed_page_maps.json').read_text())}
    checked=0
    for doc in json.loads((source/'acquisition.json').read_text()):
        with fitz.open(source/doc['file']) as pdf:
            for i,page in enumerate(pdf,1):
                row=pages[(doc['doc_id'],i)]
                if ' '.join(page.get_text('text',sort=True).split())!=row['text']:raise ValueError('Text extraction mismatch')
                label,_=printed_footer(page)
                if label!=maps[(doc['doc_id'],i)]['printed_page_label']:raise ValueError('Footer mapping mismatch')
                checked+=1
    print(json.dumps({'status':'verified','source_pages_reextracted':checked,'counts':counts},indent=2))


if __name__=='__main__':verify()
