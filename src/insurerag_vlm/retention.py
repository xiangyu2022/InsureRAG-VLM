"""Label-free public/domain reranker interpolation for explicit research profiles."""
from .reranker import minmax_scores


def anchored_cross_scores(domain_scores,public_scores,domain_weight=1.):
    if not 0<=domain_weight<=1:raise ValueError('Domain weight must be in [0,1]')
    domain=minmax_scores(domain_scores);public=minmax_scores(public_scores)
    if domain.shape!=public.shape:raise ValueError('Reranker candidate sets must match')
    return domain_weight*domain+(1-domain_weight)*public
