"""The six registered retrieval arms; score fusion never accepts gold labels."""
import numpy as np
from .reranker import blend_scores

ARMS = ('bm25', 'dense_original', 'hybrid_original', 'hybrid_adapted',
        'rerank_original', 'rerank_adapted')


def rank_six_arms(chunk_ids, dense_original, dense_adapted, sparse, cross_score):
    """Rank global scores, with shared cross logits for identical query/chunk pairs.

    cross_score receives only the union of chunk IDs. Its caller maps them to
    source text and scores against the question; gold data is not an argument.
    Each dense+positive-BM25 candidate union is capped at 200. The two unions may
    differ; shared reranker work is reused without changing either arm's pool.
    """
    ids = list(chunk_ids)
    if (not ids or len(set(ids)) != len(ids)
            or any(not isinstance(x, str) or not x or x != x.strip() for x in ids)):
        raise ValueError('Unique nonempty corpus chunk IDs required')
    def vector(values, length):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError('Every corpus chunk requires a finite score')
        return values
    d, a, b = [vector(v, len(ids)) for v in (dense_original, dense_adapted, sparse)]
    if np.any(b < 0):
        raise ValueError('BM25 scores cannot be negative')
    def ranked(values, indices, limit):
        return sorted(indices, key=lambda i: (-float(values[i]), ids[i]))[:limit]
    whole = range(len(ids))
    lexical = ranked(b, [i for i in whole if b[i] > 0], 100)
    original = ranked(d, whole, 100)
    adapted = ranked(a, whole, 100)
    pools = {name: sorted(set(order) | set(lexical), key=lambda i: ids[i])
             for name, order in [('original', original), ('adapted', adapted)]}
    combined = sorted(set(pools['original']) | set(pools['adapted']), key=lambda i: ids[i])
    cross = vector(cross_score([ids[i] for i in combined]), len(combined))
    cross_by_index = dict(zip(combined, cross))
    orders = {'bm25': [ids[i] for i in lexical[:10]],
              'dense_original': [ids[i] for i in original[:10]]}
    traces = {}
    for name, values in [('original', d), ('adapted', a)]:
        pool = pools[name]
        channels = (values[pool], b[pool], [cross_by_index[i] for i in pool])
        traces[name] = dict(candidate_ids=[ids[i] for i in pool],
                            dense=channels[0].tolist(), sparse=channels[1].tolist(),
                            cross=[float(v) for v in channels[2]])
        for prefix, weight in [('hybrid', 0.0), ('rerank', 0.5)]:
            scores = blend_scores(*channels, lexical_weight=0.2, cross_weight=weight)
            order = sorted(range(len(pool)), key=lambda j: (-float(scores[j]), ids[pool[j]]))[:10]
            orders[prefix+'_'+name] = [ids[pool[j]] for j in order]
    return dict(orders=orders, candidate_scores=traces,
                unique_cross_pairs=len(combined), cross_calls=1,
                candidate_budget_per_arm=200, depth=10,
                lexical_weight=0.2, cross_weight=0.5)
