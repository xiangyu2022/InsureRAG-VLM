"""Verify source spans, source isolation, training negatives and frozen inputs."""
from collections import Counter
import json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_multidomain_data import canonical_url,clean
from scripts.prepare_reranker_training import normalize


def run():
    fixture=ROOT/'data/benchmarks/multidomain_v1';training=ROOT/'data/training/retention_v2'
    for folder in [fixture,training]:
        manifest=json.loads((folder/'manifest.lock.json').read_text(encoding='utf8'))
        for name,expected in manifest['files'].items():assert sha(folder/name)==expected,(folder,name)
    cases={s:read_jsonl(fixture/f'{s}.jsonl') for s in ['train','valid','test']}
    answers={a['id']:a for a in read_jsonl(training/'answers.jsonl')}
    records={r['id']:r for r in read_jsonl(ROOT/'data/research_corpus/hicric_public_v1/records.jsonl')}
    old=read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/questions.jsonl')
    oldurls={canonical_url(c['source_url']) for c in old}
    source_sets={s:{c['source_group'] for c in cs} for s,cs in cases.items()}
    for a,b in [('train','valid'),('train','test'),('valid','test')]:assert not source_sets[a]&source_sets[b]
    originals={}
    for split in ['train','dev']:
        payload=json.loads((ROOT/f'../squad_download/squad_{split}_v1.1.json').read_text(encoding='utf8'))
        for article in payload['data']:
            for paragraph in article['paragraphs']:
                for q in paragraph['qas']:originals['squad_q_'+q['id']]=(paragraph['context'],q)
    government=0;squad=0;sample={};canonical_offset_variants=[]
    for split,cs in cases.items():
        for c in cs:
            answer=answers[c['gold_answer_ids'][0]]['text']
            if c['domain']=='government':
                rec=records[c['source_record_id']];text=rec['text']
                assert clean(text[slice(*c['question_span'])])==c['question']
                assert clean(text[slice(*c['answer_span'])])==answer
                assert not {canonical_url(p['source_url']) for p in rec['provenance']}&oldurls
                government+=1;sample.setdefault(c['source_group'],{'split':split,'question':c['question'],'answer':answer,'source_url':c['source_url'],'id':c['id']})
            else:
                original,q=originals[c['id']]
                assert q['question']==c['question'] and q['answers']==c['original_answer_spans']
                assert normalize(original)==normalize(answer)
                assert all(original[a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in c['original_answer_spans'])
                if not all(answer[a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in c['original_answer_spans']):
                    canonical_offset_variants.append({'id':c['id'],'split':split,'reason':'Paragraph ID deduplicates normalized text; original offsets refer to original source paragraph, not the canonical punctuation variant.'})
                squad+=1
    groups=read_jsonl(training/'train_groups.jsonl');counts=Counter(normalize(g['question']) for g in groups)
    assert all(n==1 for n in counts.values())
    hold=cases['valid']+cases['test']+old
    legacy=ROOT/'data/benchmarks/insuranceqa_v2'
    hold+=read_jsonl(legacy/'valid.jsonl')+read_jsonl(legacy/'test.jsonl')
    oldanswers={a['id']:a for a in read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/answers.jsonl')}
    allanswers={**answers,**oldanswers}
    heldq={normalize(c['question']) for c in hold};helda={normalize(allanswers[a]['text']) for c in hold for a in c['gold_answer_ids']}
    new_held_sources=source_sets['valid']|source_sets['test']
    forbidden_texts={normalize(a['text']) for a in answers.values() if a.get('source_group') in new_held_sources}
    for g in groups:
        assert normalize(g['question']) not in heldq
        pos=set(g['positive_ids']);neg=set(g['negative_ids'])
        assert not pos&neg
        positive_text={normalize(answers[a]['text']) for a in pos}
        assert not positive_text&{normalize(answers[a]['text']) for a in neg}
        assert not positive_text&helda
        assert not {normalize(answers[a]['text']) for a in pos|neg}&forbidden_texts
        assert pos|neg<=g['teacher_scores'].keys()
        assert np.isfinite(list(g['teacher_scores'].values())).all()
        assert Counter(g['negative_sources'].values())=={'dense_hard':4,'lexical_hard':4,'random':1}
    result={'status':'passed','unique_training_questions':len(groups),'government_source_spans_verified':government,
        'squad_original_spans_verified':squad,'government_sources':len(sample),'source_group_overlap':0,
        'old_government_test_source_overlap':0,'exact_duplicate_training_questions':0,
        'training_positive_vs_heldout_positive_text_overlap':0,'positive_negative_text_overlap':0,
        'teacher_scores_finite_and_complete':True,'new_heldout_source_training_pair_overlap':0,
        'canonical_paragraph_offset_variants':canonical_offset_variants,'code_sha256':sha(Path(__file__)),
        'training_manifest_sha256':sha(training/'manifest.lock.json'),'fixture_manifest_sha256':sha(fixture/'manifest.lock.json'),
        'caveat':'Mechanical source alignment; not expert annotation or verification of present-day policy correctness.'}
    write_json(result,ROOT/'reports/retention_v2/data_audit.json')
    write_json(list(sample.values()),ROOT/'reports/retention_v2/source_review_sample.json');print(json.dumps(result,indent=2))

if __name__=='__main__':run()
