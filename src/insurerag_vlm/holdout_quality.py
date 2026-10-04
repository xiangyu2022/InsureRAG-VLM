"""Fail-closed quality gates for a new, independently frozen insurance holdout.

Passing structural checks is not semantic or human review. Both audit and
content-review dispositions must be provided before any record is accepted.
"""
from collections import Counter,defaultdict
from decimal import Decimal,DecimalException,localcontext
import ast,hashlib,re,unicodedata

TASKS={'ordinary_qa','numerical_calculation','multi_evidence','insufficient_evidence'}
def normalize(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))
def digest(text):return hashlib.sha256(text.encode('utf8')).hexdigest()
def _text(value):return isinstance(value,str) and bool(value.strip())

def evaluate_calculation(formula, operands):
    """Evaluate a small arithmetic language; never execute supplied Python."""
    if len(formula)>1000:raise ValueError('Formula too long')
    values={}
    for operand in operands:
        name=operand['name']
        if not isinstance(name,str) or not name.isidentifier() or name in values:raise ValueError('Invalid or duplicate operand')
        value=Decimal(str(operand['value']))
        if not value.is_finite():raise ValueError('Non-finite operand')
        values[name]=value
    tree=ast.parse(formula,mode='eval')
    if sum(1 for _ in ast.walk(tree))>100:raise ValueError('Formula too complex')
    used={node.id for node in ast.walk(tree) if isinstance(node,ast.Name) and node.id not in {'min','max'}}
    if used!=set(values):raise ValueError('Formula must use exactly the supplied operands')
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            return Decimal(ast.get_source_segment(formula,node))
        if isinstance(node,ast.Name) and node.id in values:return values[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            value=visit(node.operand);return -value if isinstance(node.op,ast.USub) else value
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            left,right=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):return left+right
            if isinstance(node.op,ast.Sub):return left-right
            if isinstance(node.op,ast.Mult):return left*right
            return left/right
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('min','max') and node.args and not node.keywords:
            return (min if node.func.id=='min' else max)(visit(arg) for arg in node.args)
        raise ValueError('Unsupported arithmetic expression')
    with localcontext() as context:
        context.prec=40
        result=visit(tree.body)
    if not result.is_finite():raise ValueError('Non-finite result')
    return result

def verify_item(item,documents,source_approvals):
    errors=[]
    for key in ['id','publisher','jurisdiction','insurance_type','question','answer','document_group','origin']:
        if not _text(item.get(key)):errors.append('missing_'+key)
    if item.get('task') not in TASKS:errors.append('invalid_task')
    if item.get('publisher') not in source_approvals:errors.append('unapproved_source')
    evidence=item.get('evidence',[])
    if not isinstance(evidence,list) or not evidence:errors.append('missing_evidence');evidence=[]
    texts=[];signatures=[]
    for span in evidence:
        if not isinstance(span,dict):errors.append('invalid_evidence');continue
        document=documents.get(span.get('document_id'))
        if document is None:errors.append('unknown_evidence_document');continue
        if not _text(document.get('source_title')) or not _text(document.get('acquired_utc')) or not str(document.get('source_url','')).startswith('https://'):
            errors.append('incomplete_document_provenance')
        if not re.fullmatch('[0-9a-f]{64}',str(document.get('source_sha256',''))):errors.append('missing_source_snapshot_hash')
        if document.get('publisher')!=item.get('publisher'):errors.append('cross_publisher_evidence_requires_separate_protocol')
        if document.get('jurisdiction')!=item.get('jurisdiction'):errors.append('jurisdiction_mismatch')
        if digest(document['text'])!=document.get('normalized_text_sha256'):errors.append('document_text_hash_mismatch')
        start,end=span.get('start'),span.get('end')
        if type(start) is not int or type(end) is not int or not 0<=start<end<=len(document['text']):errors.append('invalid_evidence_offsets');continue
        text=document['text'][start:end];texts.append(text);signatures.append((span['document_id'],start,end))
        if not text.strip() or digest(text)!=span.get('sha256'):errors.append('evidence_hash_mismatch')
    if len(signatures)!=len(set(signatures)):errors.append('duplicate_evidence')
    for i,(doc,s,e) in enumerate(signatures):
        if any(doc==d and max(s,a)<min(e,b) for d,a,b in signatures[:i]):errors.append('overlapping_evidence')
    if any('\ufffd' in item.get(k,'') for k in ['question','answer']):errors.append('text_encoding_error')
    if item.get('automatic_flags'):errors.append('unresolved_automatic_flags')
    audit=item.get('contamination_audit',{})
    if audit.get('status')!='clear_in_accessible_scope' or not _text(audit.get('report_sha256')):errors.append('missing_contamination_clearance')
    if item.get('duplicate_audit',{}).get('status')!='unique_information_need':errors.append('missing_duplicate_clearance')
    review=item.get('content_review') or {}
    if review.get('decision') in {'reject','hold'} or review.get('unresolved_errors'):errors.append('content_review_rejection_or_hold')
    if item.get('task')!='ordinary_qa':
        if review.get('decision')!='pass' or review.get('reviewer_type')!='codex_agent_content_review' or not _text(review.get('rationale')):
            errors.append('complex_item_requires_content_review')
    if item.get('task')=='multi_evidence':
        roles=item.get('evidence_roles',[])
        if len(signatures)<2 or len(roles)!=len(signatures) or any(not _text(r) for r in roles):errors.append('missing_necessary_evidence_roles')
        if review.get('each_evidence_necessary') is not True:errors.append('multi_evidence_necessity_unreviewed')
    if item.get('task')=='numerical_calculation':
        calculation=item.get('calculation',{})
        for key in ['formula','unit','rounding','independent_verification']:
            if not _text(calculation.get(key)):errors.append('calculation_missing_'+key)
        operands=calculation.get('operands',[])
        if not isinstance(operands,list) or not operands:errors.append('calculation_missing_operands')
        else:
            for operand in operands:
                if not isinstance(operand,dict) or not _text(operand.get('name')) or not _text(operand.get('provenance')):errors.append('calculation_operand_provenance_missing')
        try:
            value=Decimal(str(calculation['result']));check=Decimal(str(calculation['verified_result']))
            if not value.is_finite() or value!=check:errors.append('calculation_result_mismatch')
            computed=evaluate_calculation(calculation['formula'],operands)
            quantum=calculation.get('rounding_quantum')
            if quantum is not None:
                quantum=Decimal(str(quantum))
                if not quantum.is_finite() or quantum<=0 or quantum.normalize().as_tuple().digits!=(1,):raise ValueError('Rounding quantum must be a positive power of ten')
                if calculation.get('rounding_mode') not in {'ROUND_HALF_UP','ROUND_HALF_EVEN','ROUND_DOWN','ROUND_UP'}:raise ValueError('Unspecified rounding mode')
                computed=computed.quantize(quantum,rounding=calculation['rounding_mode'])
            if value!=computed:errors.append('calculation_formula_mismatch')
        except (KeyError,DecimalException,ValueError,SyntaxError,TypeError):errors.append('invalid_calculation_result')
    if item.get('task')=='insufficient_evidence':
        missing=item.get('information_gap',{})
        if not missing.get('missing_facts') or not _text(missing.get('why_required')) or not _text(missing.get('available_evidence_not_sufficient')):
            errors.append('unsubstantiated_information_gap')
        if review.get('realistic_information_gap') is not True or review.get('not_answerable_from_corpus') is not True:
            errors.append('refusal_gap_unreviewed')
        if missing.get('empty_context_only') is not False:errors.append('empty_context_shortcut')
    return sorted(set(errors))

def verify_dataset(test,dev,documents,source_approvals,protocol):
    errors=[];details={};combined=[('test',r) for r in test]+[('dev',r) for r in dev]
    ids=[r.get('id') for _,r in combined]
    if len(set(ids))!=len(ids):errors.append('duplicate_or_cross_split_item_ids')
    for split,rows in [('test',test),('dev',dev)]:
        for row in rows:
            issues=verify_item(row,documents,source_approvals)
            if issues:details[split+':'+str(row.get('id'))]=issues
    if details:errors.append('item_quality_gate_failed')
    for key in ['publisher','document_group']:
        if {r.get(key) for r in test}&{r.get(key) for r in dev}:errors.append('cross_split_'+key)
    split_evidence=[]
    for rows in [test,dev]:
        split_evidence.append({span.get('document_id') for row in rows for span in row.get('evidence',[])})
    if split_evidence[0]&split_evidence[1]:errors.append('cross_split_evidence_document')
    if {normalize(r['question']) for r in test}&{normalize(r['question']) for r in dev}:errors.append('cross_split_question_duplicate')
    if len(test)<protocol['target_test_accepted']:errors.append('test_count_below_target')
    if len(dev)<protocol['target_dev_accepted']:errors.append('dev_count_below_target')
    publishers=Counter(r.get('publisher') for r in test);groups=Counter(r.get('document_group') for r in test)
    targets=protocol['source_targets']
    if len(publishers)<targets['minimum_independent_test_publishers']:errors.append('too_few_test_publishers')
    if publishers and max(publishers.values())/len(test)>targets['maximum_publisher_fraction']:errors.append('publisher_overrepresentation')
    if len(groups)<targets['minimum_test_document_groups']:errors.append('too_few_document_groups')
    if groups and max(groups.values())>targets['maximum_questions_per_document_group']:errors.append('document_overrepresentation')
    tasks=Counter(r.get('task') for r in test)
    for task,target in protocol['test_task_targets'].items():
        if tasks[task]<target:errors.append('task_count_below_target:'+task)
    ordinary=defaultdict(list)
    for split,row in combined:
        if row.get('task')=='ordinary_qa':
            ordinary[(split,row.get('publisher'),row.get('insurance_type'))].append(row)
            ordinary[(split,'document_group',row.get('document_group'))].append(row)
    for stratum,rows in ordinary.items():
        reviewed=sum((r.get('content_review') or {}).get('decision')=='pass' and (r.get('content_review') or {}).get('reviewer_type')=='codex_agent_content_review' for r in rows)
        if reviewed/len(rows)<.25:errors.append('insufficient_ordinary_review:'+str(stratum))
    return {'passed':not errors,'errors':errors,'item_errors':details,'test_n':len(test),'dev_n':len(dev),
            'publishers':dict(publishers),'tasks':dict(tasks),'document_groups':len(groups),
            'accepted_test_items':len(test) if not errors else 0,
            'review_label':'Automated gates plus recorded agent content review; not human/expert adjudication.'}
