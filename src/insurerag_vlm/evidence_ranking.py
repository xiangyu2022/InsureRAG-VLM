"""Head-sensitive rank loss with frozen-teacher retention and original labels."""
import numpy as np


def head_losses(logits, teacher, confidence, pair_weight=.5):
    import torch
    if logits.ndim!=2 or logits.shape[1]<2 or logits.shape!=teacher.shape or confidence.shape!=(logits.shape[0],logits.shape[1]-1):
        raise ValueError('Rank loss shape mismatch')
    if not all(torch.isfinite(x).all() for x in [logits,teacher,confidence]) or not ((confidence>0)&(confidence<=1)).all():
        raise ValueError('Invalid rank logits or confidence')
    if not 0<=pair_weight<=1:raise ValueError('Invalid pairwise loss weight')
    adjusted=torch.cat([logits[:,:1],logits[:,1:]+confidence.log()],dim=1)
    ce=torch.nn.functional.cross_entropy(adjusted,torch.zeros(len(logits),device=logits.device,dtype=torch.long),label_smoothing=.02,reduction='none')
    # Rank weights are stop-gradient; no labels beyond the sampled author-positive.
    ranks=torch.argsort(torch.argsort(-logits.detach(),dim=1,stable=True),dim=1,stable=True).float()+1
    delta=(1/ranks[:,:1]-1/ranks[:,1:]).abs().clamp_min(.05)
    weights=delta*confidence
    pair=(torch.nn.functional.softplus(1+logits[:,1:]-logits[:,:1])*weights).sum(dim=1)/weights.sum(dim=1)
    kl=torch.nn.functional.kl_div(torch.log_softmax(logits/2,dim=1),torch.softmax(teacher.detach()/2,dim=1),reduction='none').sum(dim=1)*4
    return ce+pair_weight*pair,kl


def sample_training_group(group,rng,presentation):
    positive=group['positive_ids'][presentation%len(group['positive_ids'])]
    negatives=[]
    for kind,count in [('current_hard',2),('lexical_hard',2),('random',1)]:
        pool=[a for a in group['negative_ids'] if group['negative_sources'][a]==kind and a not in negatives]
        negatives.extend(rng.sample(pool,min(count,len(pool))))
    remaining=[a for a in group['negative_ids'] if a not in negatives]
    negatives.extend(rng.sample(remaining,5-len(negatives)))
    if len(set(negatives))!=5 or set(negatives)&set(group['positive_ids']):raise ValueError('Invalid negative sample')
    return [positive]+negatives


def evidence_metrics(order,gold,candidates):
    from scripts.eval_query_adaptation import measure
    gold=set(gold)
    return {**measure(order,gold,candidates),
            'all_evidence_at_5':float(gold<=set(order[:5])),
            'all_evidence_at_10':float(gold<=set(order[:10])),
            'evidence_recall_at_5':len(gold&set(order[:5]))/len(gold),
            'candidate_all_evidence':float(gold<=set(candidates))}


def summarize_evidence(rows):
    from scripts.eval_query_adaptation import summarize
    return {**summarize(rows),**{k:float(np.mean([r[k] for r in rows])) for k in
                               ['all_evidence_at_5','all_evidence_at_10','evidence_recall_at_5','candidate_all_evidence']}}
