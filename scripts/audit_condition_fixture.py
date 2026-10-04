"""Audit new labels against their original source before any new test scoring."""
from collections import Counter
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_reranker_training import normalize

def run():
    fixture=ROOT/'data/benchmarks/condition_v1'
    lock=json.loads((fixture/'manifest.lock.json').read_text(encoding='utf8'))
    for name,h in lock['files'].items():assert sha(fixture/name)==h
    train=read_jsonl(fixture/'train_additions.jsonl');test=read_jsonl(fixture/'test.jsonl')
    answers={a['id']:a['text'] for a in read_jsonl(fixture/'answers.jsonl')}
    records={r['id']:r for path in ['hicric_public_v1','dol_additional_v1'] for r in read_jsonl(ROOT/f'data/research_corpus/{path}/records.jsonl')}
    old=read_jsonl(ROOT/'data/training/retention_v2/train_groups.jsonl')
    oldpositive={a for g in old for a in g['positive_ids']}
    originals={q['id']:(p['context'],q) for article in json.loads((ROOT/'../squad_download/squad_train_v1.1.json').read_text(encoding='utf8'))['data'] for p in article['paragraphs'] for q in p['qas']}
    counts=Counter();mismatches=[]
    for c in train+test:
        if c['domain']=='government':
            text=records[c['source_record_id']]['text'];qs,qe=c['question_span'];s,e=c['answer_span']
            if normalize(text[s:e])!=normalize(answers[c['gold_answer_ids'][0]]):mismatches.append([c['id'],'answer'])
            # Publisher line wraps are collapsed by extraction; preserve every non-whitespace character.
            if ' '.join(c['question'].split()) not in ' '.join(text[qs:qe].split()):mismatches.append([c['id'],'question'])
            counts['government_exact_source_spans_checked']+=1
        else:
            context,q=originals[c['id'].removeprefix('squad_q_')]
            assert c['question']==q['question'] and normalize(context)==normalize(answers[c['gold_answer_ids'][0]])
            assert all(context[a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in c['original_answer_spans'])
            counts['original_crowdworker_questions_checked']+=1
    general=[c for c in train if c['domain']=='general'];positive=[c['gold_answer_ids'][0] for c in general]
    assert len(positive)==len(set(positive)), 'Duplicate new paragraph positives'
    assert not set(positive)&oldpositive
    assert len({normalize(c['question']) for c in train})==len(train)
    assert not {c['source_group'] for c in train}&{c['source_group'] for c in test}
    oldpairtexts={normalize(answers[a]) for g in old for a in g['positive_ids']+g['negative_ids']}
    assert not {normalize(answers[a]) for c in test for a in c['gold_answer_ids']}&oldpairtexts
    result={'checks':dict(counts),'mismatches':mismatches,'new_general_distinct_paragraphs':len(set(positive)),
            'fresh_test_source_groups':len({c['source_group'] for c in test}),'fresh_test_text_in_previous_training_pairs':0,
            'fixture_sha256':sha(fixture/'manifest.lock.json'),'code_sha256':sha(Path(__file__)),'expert_adjudication':False}
    write_json(result,ROOT/'reports/condition_v1/fixture_verification.json')
    print(json.dumps(result,indent=2));assert not mismatches

if __name__=='__main__':run()
