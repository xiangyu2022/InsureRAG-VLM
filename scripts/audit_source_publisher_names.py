"""Supplement URL screening with explicit publisher-name and broader-domain hits.

Names are screening probes, not automatic evidence of organizational identity.
Only origin aliases, matched probe names and hashes are exported.
"""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.retrieve_source_holdout import LOCAL,read,write,sha

PROBES={
 'ccpc_full':'competition and consumer protection commission',
 'ccpc_acronym':'ccpc',
 'hia_full':'health insurance authority',
 'fsra_full':'financial services regulatory authority',
 'fsra_acronym':'fsra',
 'fsco_full':'financial services commission of ontario',
 'australia_private_health_ombudsman':'private health insurance ombudsman',
 'australia_commonwealth_ombudsman':'commonwealth ombudsman',
 'oregon_dfr':'oregon division of financial regulation',
 'oregon_dcbs':'oregon department of consumer and business services',
}

def main():
    inventory=read(LOCAL/'history_v2/historical_inventory.json')
    regex={name:re.compile(r'(?<!\w)'+re.escape(value)+r'(?!\w)') for name,value in PROBES.items()}
    found={name:[] for name in PROBES}
    path=LOCAL/'history_v2/historical_texts.jsonl'
    with path.open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            for name,pattern in regex.items():
                if pattern.search(row['text']):found[name].append({'origin':row['origin'],'normalized_string_sha256':hashlib.sha256(row['text'].encode()).hexdigest()})
    domains=['oregon.gov','ombudsman.gov.au','fsco.gov.on.ca','fsrao.ca','ccpc.ie','consumerhelp.ie','hia.ie','privatehealth.gov.au']
    report={'normalized_strings_screened':inventory['text_count'],'literal_name_probes':PROBES,
            'name_hit_counts':{k:len(v) for k,v in found.items()},'hits':found,
            'broader_domain_hits':{d:{h:n for h,n in inventory['hosts'].items() if h==d or h.endswith('.'+d)} for d in domains},
            'history_sha256':sha(path),'timing':'Supplemental post-lock audit; no reselection or question removal.',
            'limitation':'Absence of these names/domains cannot establish unseen model pretraining or exclude unnamed/inaccessible publisher exposure.'}
    write(ROOT/'reports/source_holdout_v1/publisher_name_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='hits'},indent=2))
if __name__=='__main__':main()
