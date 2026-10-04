# Method notes: noisy negatives and specialist retention

This prototype combines established training ideas and a small local heuristic.
It does not claim a new state-of-the-art retrieval algorithm or a reproduction
of any cited system. The BGE encoder is fixed; only the MiniLM cross-encoder
receives gradient updates.

## Why a mined negative is not necessarily wrong

Incomplete relevance labels can turn a top-retrieved but unannotated passage
into a false negative. RocketQA explicitly studies this issue and denoised hard
negatives for dense retrieval. That motivates auditing the negatives here; its
dataset-specific findings are not evidence that the same false-negative rate
holds for insurance. [Qu et al., NAACL 2021](https://aclanthology.org/2021.naacl-main.466/)

Our public-teacher confidence rule is an uncalibrated relative-score heuristic:
for a sampled author-positive passage with public logit `u+`, a negative receives
weight 0.25 if `u- >= u+ - 1`, otherwise 1. No passage is relabeled positive by
this rule. The blinded training review queue remains unannotated until a reviewer
actually supplies and adjudicates judgments.

## The two recorded objectives

Let `s0` be the author-positive student score and `s1..s5` the five sampled
negative scores. The first recipe minimizes the average of
`ci * softplus(0.5 - s0 + si)`, plus public-teacher KL. Averaging across five
competitors and lowering ambiguous-negative weights reduces the supervision's
scale relative to that KL term. Both seeds failed the registered validation
promotion rule; the artifacts remain in `reports/condition_v1/`.

The follow-up uses listwise probabilities:

\[
p_i = \frac{\exp(s_i + \log c_i)}{\sum_{j=0}^{5}\exp(s_j + \log c_j)},
\qquad c_0=1.
\]

Cross entropy uses label smoothing 0.02. The confidence-adjusted denominator
retains competition among candidates while reducing the repulsion of a suspected
false negative. The additional retention loss is

\[
L = L_{CE} + \lambda T^2\,\mathrm{KL}
\left(\mathrm{softmax}(t/T)\;\Vert\;\mathrm{softmax}(s/T)\right),
\quad T=2.
\]

Here `t` is the **previous insurance-specialist** score vector; KL uses raw logits,
without confidence adjustment. Lambda is 0.1 for FAQ and 0.2 for government/general.
The public model is still used for the negative-confidence rule, but not as this
follow-up's retention target. The old specialist is fixed and supplies no new
human labels.

Listwise supervision and relevance-distribution distillation are established
retrieval techniques; RocketQAv2 combines related ingredients while jointly
training retrieval and reranking. This project trains only the reranker and
uses a frozen teacher, so it is a different experiment.
[Ren et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.224/)

Preserving previous capabilities through distillation also has precedent in
Learning without Forgetting. That work studies a different task/data setting;
our experiment additionally rehearses old labeled examples and cannot be
described as its direct implementation.
[Li and Hoiem, Learning without Forgetting](https://arxiv.org/abs/1606.09282)

## What the experiment can establish

The follow-up keeps questions, positive/negative IDs and starting weights fixed
relative to the rejected first recipe. It changes the supervised objective,
teacher, and regularization strength together. Therefore any improvement is
evidence for the combined recipe, not a clean causal estimate for one ingredient.
The validation set is reused across recipes. Fresh government testing is delayed
until the final selection, and previous-model controls use the same ranking
configuration to separate training effects from fusion/candidate-pool effects.

A stronger attribution study would require independent ablations of the loss,
teacher and confidence rule, more held-out insurance document families, and
expert adjudication of uncertain labels. Those experiments have not been claimed
as completed here. The executed training and measured results are recorded in
`CONDITION_UPDATE_ZH.md`; retrieval metrics do not validate generated advice.
