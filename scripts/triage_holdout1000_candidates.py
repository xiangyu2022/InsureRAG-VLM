"""Create a deterministic conservative review queue, never accepted records."""
import argparse,hashlib,json,re,sys,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.insurerag_vlm.holdout_quality import normalize

class Components:
    def __init__(self,keys):self.parent={key:key for key in keys}
    def find(self,key):
        while self.parent[key]!=key:
            self.parent[key]=self.parent[self.parent[key]];key=self.parent[key]
        return key
    def merge(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--semantic',type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh triage output directory')
    rows=[json.loads(l) for l in (a.snapshot/'candidates.jsonl').read_text(encoding='utf8').splitlines()]
    docs={r['id']:r for r in map(json.loads,(a.snapshot/'documents.jsonl').read_text(encoding='utf8').splitlines())}
    exact=json.loads((a.snapshot/'history_exact.json').read_text());near=json.loads((a.snapshot/'history_near.json').read_text())
    candidates_hash=hashlib.sha256((a.snapshot/'candidates.jsonl').read_bytes()).hexdigest()
    assert exact['candidate_file_sha256']==near['input_sha256']['candidates']==candidates_hash
    assignment=json.loads((ROOT/'reports/holdout1000_v1/split_assignment.json').read_text())
    splits={pub:split for split in ['test','dev'] for pub in assignment[split+'_publishers']}
    history={r['id'] for r in exact['rows'] if r['excluded_from_independent_holdout']}|{r['id'] for r in near['rows'] if r['historical_flag']}
    semantic=json.loads(a.semantic.read_text()) if a.semantic else None
    if semantic:
        assert candidates_hash in semantic['candidate_file_sha256'].values(),'Semantic audit uses different candidate snapshot'
        history.update(r['id'] for r in semantic['rows'] if r['flag_for_review'] or r['candidate_truncated'])
    families=Components(docs);by_text=defaultdict(list)
    for d in docs.values():by_text[d['normalized_text_sha256']].append(d['id'])
    for group in by_text.values():
        for key in group[1:]:families.merge(group[0],key)
    historical_families={families.find(r['document_group']) for r in exact['rows'] if r['historical_document_url_origin']}
    family_splits=defaultdict(set)
    for d in docs.values():family_splits[families.find(d['id'])].add(splits.get(d['publisher']))
    components=Components(r['id'] for r in rows)
    for pair in near['candidate_pair_flags']:components.merge(pair['first'],pair['second'])
    if semantic:
        for pair in semantic['pairs']:
            if pair['first'] in components.parent and pair['second'] in components.parent:components.merge(pair['first'],pair['second'])
    for field in ['question','answer']:
        groups=defaultdict(list)
        for r in rows:groups[normalize(r[field])].append(r['id'])
        for group in groups.values():
            for key in group[1:]:components.merge(group[0],key)
    document_rows=defaultdict(list)
    for r in rows:document_rows[families.find(r['document_group'])].append(r)
    overlap_edges=0
    for group in document_rows.values():
        for i,row in enumerate(group):
            span=row['evidence'][0]
            if span['start']<0:continue
            for other in group[:i]:
                second=other['evidence'][0]
                if second['start']<0 or span['document_id']!=second['document_id']:continue
                overlap=max(0,min(span['end'],second['end'])-max(span['start'],second['start']))
                if overlap/min(span['end']-span['start'],second['end']-second['start'])>=.85:
                    components.merge(row['id'],other['id']);overlap_edges+=1
    groups=defaultdict(list)
    for r in rows:groups[components.find(r['id'])].append(r)
    disposition=[];queue=[];nonrepresentatives=0
    for component,members in groups.items():
        contaminated=any(r['id'] in history for r in members)
        cross_split=len({splits.get(r['publisher']) for r in members})>1
        ranked=sorted(members,key=lambda r:(len(r['automatic_flags']),len(r['answer'].split())>600,r['id']))
        representative=ranked[0]['id']
        for r in members:
            reasons=[];text=docs[r['document_group']]['text'];title=r['source_title'];url=r['source_url']
            if contaminated:reasons.append('history_flag_in_near_duplicate_component')
            if cross_split:reasons.append('duplicate_component_crosses_publisher_split')
            if families.find(r['document_group']) in historical_families:reasons.append('historical_source_url_in_document_family')
            if len(family_splits[families.find(r['document_group'])])>1:reasons.append('identical_document_family_crosses_split')
            if r['id']!=representative:reasons.append('duplicate_component_nonrepresentative');nonrepresentatives+=1
            if {'evidence_alignment_failed','encoding_replacement_character','answer_boundary_or_excess_length'} & set(r['automatic_flags']):reasons.append('unresolved_extraction_defect')
            if re.search(r'credit repair|freeze your credit|mortgage (?:loan|broker)|student loan|foreclosure|banking|money transmitter|bail (?:agent|bond)',title+' '+url,re.I):reasons.append('out_of_consumer_insurance_scope')
            if re.search(r'(?:copyright|reprinted|reproduced).{0,120}(?:NAIC|National Association of Insurance Commissioners)|(?:NAIC|National Association of Insurance Commissioners).{0,80}copyright',text,re.I):reasons.append('third_party_permission_unresolved')
            if re.search(r'\b(?:NAIC|National Association of Insurance Commissioners)\b',title):reasons.append('third_party_publication_unresolved')
            if r['publisher']=='northcarolina_doi' and re.search('medicare',title+' '+url,re.I):reasons.append('critical_sample_error_requires_full_currency_review')
            if r['publisher']=='northcarolina_doi' and 'Health Insurance Continuation Rights' in title:reasons.append('obsolete_hipaa_fallback_requires_full_page_review')
            if r['publisher']=='utah_uid' and 'HIPAA-Health Insurance Portability' in title:reasons.append('superseded_hipaa_certificate_rule_requires_full_page_review')
            if r['publisher']=='california_cdi' and title=='Long Term Care Insurance':reasons.append('medicare_home_health_daily_care_error_requires_full_page_review')
            if r['publisher']=='us_opm' and title in {'Health Savings Accounts','Temporary Continuation of Coverage'}:reasons.append('critical_opm_currency_error_requires_full_page_review')
            if r['publisher']=='us_opm' and title=='Federal Employees Health Benefits (FEHB) Program Carriers':reasons.append('carrier_procurement_scope_and_merged_faq_review_required')
            if r['publisher']=='us_va' and title=='Totally disabled or terminally ill policyholders':reasons.append('contradictory_accelerated_benefit_proxy_rules_require_full_page_review')
            if r['publisher']=='us_va' and re.search(r'(?:call(?:ing)?(?: us)? at|fax the form to)\s*\.',r['answer'],re.I):reasons.append('custom_component_contact_value_missing')
            if r['publisher']=='texas_tdi' and title=='Do I need to buy insurance when I rent a car?':reasons.append('credit_card_secondary_coverage_overgeneralization_requires_full_page_review')
            if re.search(r'Want to share more feedback\?|Complete our 3-question survey',r['answer'],re.I):reasons.append('website_feedback_leaks_into_answer')
            if r['publisher']=='wisconsin_oci' and title=='Fact Sheet on Continuation Rights in Health Insurance Policies':reasons.append('cobra_initial_payment_deadline_conflict_requires_full_page_review')
            if r['publisher']=='us_tricare' and title=='Pharmacy Costs':reasons.append('pharmacy_population_and_table_boundary_review_required')
            if any(unicodedata.category(c)=='Cf' for c in r['question']+r['answer']):reasons.append('invisible_format_chars_require_context_repair_and_dual_audit')
            if '\u00bf' in r['question'] or re.search(r'gu[i\u00ed]a del seguro|seguros de|seguro de auto',title,re.I):reasons.append('non_english_requires_validated_multilingual_contamination_audit')
            record={'id':r['id'],'publisher':r['publisher'],'split':splits.get(r['publisher']),
              'document_family':families.find(r['document_group']),'duplicate_component':component,'component_size':len(members),
              'representative':representative,'status':'hold_or_exclude' if reasons else 'queued_not_accepted',
              'reasons':reasons,'automatic_flags':r['automatic_flags']}
            disposition.append(record)
            if not reasons:queue.append({**r,'split':record['split'],'document_family':record['document_family'],'duplicate_component':component})
    a.output.mkdir(parents=True)
    for name,data in [('dispositions',disposition),('review_queue',queue)]:
        with (a.output/(name+'.jsonl')).open('x',encoding='utf8') as handle:
            for row in data:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    result={'candidate_count':len(rows),'document_families':len({families.find(k) for k in docs}),
      'duplicate_components':len(groups),'evidence_overlap_edges':overlap_edges,'nonrepresentatives':nonrepresentatives,
      'queue_by_split':dict(Counter(r['split'] for r in queue)),'queue_by_publisher':dict(Counter(r['publisher'] for r in queue)),
      'hold_reason_counts':dict(Counter(reason for r in disposition for reason in r['reasons'])),
      'candidate_file_sha256':candidates_hash,'semantic_audit_sha256':hashlib.sha256(a.semantic.read_bytes()).hexdigest() if a.semantic else None,'accepted_test_items':0,
      'limitations':['Queue membership is not source-currency, permission, semantic novelty or content clearance.','Threshold/component exclusions are intentionally conservative and may drop substantively different questions.','No document cap, publisher fraction or task quota is fulfilled merely by this queue.']}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
