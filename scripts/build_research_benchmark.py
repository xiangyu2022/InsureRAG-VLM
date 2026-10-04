#!/usr/bin/env python3
"""Materialize agent-inspected annotations once, before any model evaluation.

This file records benchmark authorship; it is not a synthetic-data generator.
Do not edit an already evaluated benchmark version to improve reported scores.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/benchmarks/research_v1'
AUTHOR = 'AI-assisted/manual-agent-verified; no human-expert adjudication'
DOCS = {
    'de_auto_insurance_guide': ('dev', 'Delaware Auto Insurance Guide', 'https://insurance.delaware.gov/divisions/consumerhp/auto/'),
    'de_homeowners_guide': ('dev', 'Delaware Homeowners Insurance Guide', 'https://insurance.delaware.gov/wp-content/uploads/sites/15/2022/09/Homeowners-Guide.pdf'),
    'md_auto_insurance_guide': ('test', 'Maryland Consumer Guide to Auto Insurance', 'https://insurance.maryland.gov/Consumer/Documents/publications/autoinsuranceguide.pdf'),
    'md_homeowners_insurance_guide': ('test', 'Maryland Consumer Guide to Homeowners Insurance', 'https://insurance.maryland.gov/Consumer/Documents/publications/homeownersinsguide.pdf'),
    'nc_disability_insurance_guide': ('test', 'North Carolina Consumers Guide to Disability Insurance', 'https://www.ncdoi.gov/consumers-guide-disability-insurance/open'),
    'nc_travel_insurance_guide': ('test', 'North Carolina Consumers Guide to Travel Insurance', 'https://www.ncdoi.gov/consumers-guide-travel-insurance/open'),
}
# Tuple: scope, physical PDF page, question body, reference answer, exact evidence span,
# and required answer-key groups (each group contains interchangeable surface forms).
SUPPORTED = [
 ('de_auto_insurance_guide',4,'what is the minimum fine stated for a first conviction for driving without proper insurance?', '$1,500', 'not less than $1500 for the first offense', [['1500']]),
 ('de_auto_insurance_guide',5,'what maximum funeral expense benefit is included in the PIP coverage described?', '$5,000', 'Also included in PIP coverage is up to $5,000 for funeral expenses.', [['5000']]),
 ('de_auto_insurance_guide',6,'what automatic property-damage deductible accompanies the uninsured motorist coverage described?', '$250', 'The coverage comes at an automatic $250 deductible for property damage', [['250']]),
 ('de_auto_insurance_guide',9,'what percentage discount follows an initial approved defensive driving course, and how long does it last?', '10 percent for three years', 'you can receive 10 percent off a portion of your auto insurance for three years.', [['10 percent','10%'], ['three years','3 years']]),
 ('de_homeowners_guide',6,'which HO form number is designed for renters?', 'HO-4', 'policies designed for renters (HO-4, see page 13)', [['HO-4']]),
 ('de_homeowners_guide',8,'what typical category limit is listed for computer equipment?', '$5,000', '$5,000 for computer equipment', [['5000']]),
 ('de_homeowners_guide',9,'in the ten-year-old television example, how much would a replacement-cost policy pay?', '$500', 'A replacement cost policy would give you $500.', [['500']]),
 ('de_homeowners_guide',10,'what percentage of property value is described as a typical hurricane deductible?', '2 percent', "deductible (typically 2% of the property's value)", [['2%','2 percent']]),
 ('md_auto_insurance_guide',10,'what maximum percentage surcharge or discount based on credit history does the guide state?', '40 percent', 'apply a surcharge or discount of more than 40% based on credit history', [['40%','40 percent']]),
 ('md_auto_insurance_guide',13,'what minimum property-damage liability limit is listed?', '$15,000', '$15,000 property damage', [['15000']]),
 ('md_auto_insurance_guide',16,'what minimum Personal Injury Protection amount must insurers offer, according to this archived guide?', '$2,500', 'at least $2,500 in Personal Injury Protection (PIP) coverage.', [['2500']]),
 ('md_auto_insurance_guide',17,'in the example with $1,200 of collision damage and a $500 deductible, how much does the insurer pay?', '$700', 'then the insurer will pay $700', [['700']]),
 ('md_homeowners_insurance_guide',7,'what peril coverage does the Special Form HO-3 provide for the building versus the contents?', 'Open-peril coverage on the building; named-peril coverage for contents.', 'It provides named-peril coverage for the contents of your home.', [['open peril','open perils'], ['named peril','named perils']]),
 ('md_homeowners_insurance_guide',10,'which two coverages are exempt from the illustrated property-damage deductible?', 'Liability and medical payments', 'This deductible does not apply to claims under the liability or medical payments coverages.', [['liability'], ['medical payments']]),
 ('md_homeowners_insurance_guide',11,'in the $3,000 hurricane-damage example with a $100,000 dwelling limit and a 2% hurricane deductible, how much would the insurer pay?', '$1,000', 'the insurer would pay $1,000 towards the damage.', [['1000']]),
 ('md_homeowners_insurance_guide',12,'are contents automatically covered by the flood policy described, or must that coverage be purchased separately?', 'Contents coverage must be purchased separately.', 'You need to purchase this coverage separately and in addition to the coverage for your home.', [['separately','separate coverage']]),
 ('nc_disability_insurance_guide',2,'what is the name of the period after disability begins during which benefits are not payable?', 'Elimination period', 'The elimination period is a specified period of time, stated in the policy contract, following the beginning of disability during which benefits are not payable.', [['elimination period']]),
 ('nc_disability_insurance_guide',3,'what percentage range of pre-disability wages can the income benefit formula use?', '50 to 75 percent', 'The percentage varies from policy to policy and may range from 50 to 75 percent.', [['50'], ['75'], ['percent','%']]),
 ('nc_disability_insurance_guide',3,'what benefit-duration range is generally stated for short-term disability policies?', 'Six months to two years', 'Short term policies generally provide benefits from six months to two years', [['six months','6 months'], ['two years','2 years']]),
 ('nc_disability_insurance_guide',4,'which rider provides a benefit when someone returns to work full-time but income is not fully restored?', 'Residual disability rider', 'RESIDUAL DISABILITY RIDER This rider provides a benefit if the insured can return to work full-time but income is not fully restored.', [['residual disability']]),
 ('nc_travel_insurance_guide',2,'from whom must a claimant first seek reimbursement before the trip-cancellation insurer considers the claim?', 'The trip provider', 'You must first seek reimbursement from the trip provider before the insurance company will consider the claim.', [['trip provider']]),
 ('nc_travel_insurance_guide',2,'which named coverage protects belongings that are lost, stolen, or damaged during a trip?', 'Baggage insurance', 'Baggage Insurance Provides coverage if your belongings (luggage and personal possessions) are lost, stolen or damaged during a trip.', [['baggage insurance','baggage coverage']]),
 ('nc_travel_insurance_guide',3,'are cancellation waivers offered by cruise or tour operators described as insurance?', 'No, cancellation waivers are not insurance.', 'cancellation waivers, which are not insurance', [['not insurance','not considered insurance','not an insurance']]),
 ('nc_travel_insurance_guide',3,'what orders are the payment of weather-related claims described as contingent upon?', 'Mandatory evacuation orders', 'Claims paid due to weather conditions are contingent upon mandatory evacuation orders.', [['mandatory evacuation']]),
]
MISSING = {
    'de_auto_insurance_guide': ('what exact annual premium did I pay for my own automobile policy?', 'what is my homeowners policy number?'),
    'de_homeowners_guide': ('what exact dollar amount is the deductible on my own homeowners policy?', 'what is my personal automobile vehicle identification number?'),
    'md_auto_insurance_guide': ('what exact dollar amount is the collision deductible on my own issued auto policy, rather than an example deductible?', 'what is the account number of my personal bank account?'),
    'md_homeowners_insurance_guide': ('what exact Coverage A dwelling limit appears on my own issued declarations page?', 'what is the password for my insurance company account?'),
    'nc_disability_insurance_guide': ('what exact monthly dollar benefit is payable under my own disability policy?', 'what is the dollar deductible of my own automobile policy?'),
    'nc_travel_insurance_guide': ('how many days is the free-look period for my own travel policy?', 'what is the mortgage account number for my home?'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = DEST / 'manifest.lock.json'
    if lock.exists():
        raise RuntimeError('Benchmark is frozen. Create a new version instead of replacing this lock.')
    pages_path = ROOT / 'data/04_curated/rag_pages.jsonl'
    snippets_path = ROOT / 'data/04_curated/rag_snippets.jsonl'
    pages = [json.loads(line) for line in pages_path.read_text(encoding='utf-8').splitlines()]
    lookup = {(row['doc_id'], row['page']): row for row in pages}
    rows = []
    documents = []
    for doc_id, (split, title, origin) in DOCS.items():
        available = [row for row in pages if row['doc_id'] == doc_id]
        assert len(available) >= 3, doc_id
        documents.append({'doc_id': doc_id, 'split': split, 'title': title, 'public_origin_reference': origin,
            'available_physical_pages': sorted(row['page'] for row in available),
            'scope_note': 'All available curated pages in this named document; no gold-page filtering.'})
        selected = [item for item in SUPPORTED if item[0] == doc_id]
        for index, (_, page, body, answer, span, keys) in enumerate(selected, 1):
            record = lookup[(doc_id,page)]
            assert span in record['text'], (doc_id,page,span)
            rows.append({'id': f'{split}_{doc_id}_{index:02d}', 'split': split,
                'document_scope': [doc_id], 'document_title': title,
                'question': f'Using only the archived {title}, {body}', 'answerable': True,
                'reference_answer': answer, 'answer_keys': keys,
                'gold': [{'source': record['citation'], 'page': page, 'evidence_span': span}],
                'annotation_author': AUTHOR})
        for offset, body in enumerate(MISSING[doc_id], 5):
            rows.append({'id': f'{split}_{doc_id}_{offset:02d}', 'split': split,
                'document_scope': [doc_id], 'document_title': title,
                'question': f'Using only the archived {title}, {body}', 'answerable': False,
                'reference_answer': '', 'answer_keys': [], 'gold': [],
                'unsupported_type': 'missing_in_scope_specific_value' if offset == 5 else 'unrelated_private_fact',
                'unsupported_reason': 'The public consumer guide is not this person\'s issued policy or private record; generic examples do not establish this fact.',
                'annotation_author': AUTHOR})
    assert len(rows) == 36 and sum(row['split'] == 'dev' for row in rows) == 12
    for split in ['dev', 'test']:
        path = DEST / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows if row['split'] == split), encoding='utf-8')
    (DEST / 'documents.json').write_text(json.dumps(documents, indent=2), encoding='utf-8')
    payload = {'benchmark_version': 'research_v1', 'frozen_before_model_inference': True,
        'annotation_author': AUTHOR, 'counts': {'dev': 12, 'test': 24, 'answerable': 24, 'unsupported': 12},
        'question_split_policy': 'Document-disjoint dev/test questions across six named archived public guides.',
        'training_exposure_note': 'These public documents exist in the legacy SFT corpus; this is NOT a document holdout from previous fine-tuning. Compare unadapted Ollama base instruct models only and report possible public pretraining exposure.',
        'files': {name: digest(DEST/name) for name in ['dev.jsonl','test.jsonl','documents.json']},
        'corpus_files': {'data/04_curated/rag_pages.jsonl':digest(pages_path), 'data/04_curated/rag_snippets.jsonl':digest(snippets_path)}}
    lock.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__ == '__main__':
    main()
