# Explicit insurance reference-path diagnostic

120 synthetic questions across 30 invented packets and 300 text records: 24 development questions and 96 test questions. Each packet has ten records, one two-hop clause-reference chain, similar but unrelated amounts, and four question categories (direct, one-hop, multi-hop, unsupported). These are text records, not PDF pages or real policies.

Six authored scenario templates are shared across the two splits. Identifiers and values differ, but this is a controlled regression and mechanism experiment, **not an unseen-template generalization benchmark**. The graph is constructed from literal source text, without answer keys or gold-source IDs. The source text and labels were frozen before retrieval experiments; code selection used development cases only.

`scripts/graph_retrieval_study.py` compares BM25, dense, hybrid, hybrid with explicit references, and hybrid with candidate relations enabled. All arms use the same embedding checkpoint, top-three context budget, and ranking machinery outside the named retrieval/graph switches. Accordingly, the BM25/dense arms include the shared insurance heuristic reranker; they are not pure raw-score search baselines.

The primary endpoint is complete annotated evidence in the actual packed context. Page recall alone can conceal a missing condition or limit. Unsupported cases are retrieved but do not receive an abstention score: no model generation runs in this experiment. The companion 240-question public-guide experiment evaluates actual Qwen generation.

Exact fixture hashes, graph paths, full contexts, per-category results, and a paired packet bootstrap are saved. Bootstrap intervals describe variation within the six constructed templates; they cannot establish real-policy reliability. Read `docs/GRAPH_RESEARCH_PROTOTYPE.md` for the interpretation and commands.
