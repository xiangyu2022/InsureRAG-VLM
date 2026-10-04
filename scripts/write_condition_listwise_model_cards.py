"""Attach actual training/selection provenance to both locally produced checkpoints."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run():
    reports=ROOT/'reports/condition_listwise_v1';lock=json.loads((reports/'selection.lock.json').read_text(encoding='utf8'))
    for seed in [42,123]:
        parent=ROOT/f'../models/insurerag-condition-listwise-v1-seed-{seed}';model=parent/'epoch-1'
        done=json.loads((reports/f'train_seed_{seed}/completion.json').read_text(encoding='utf8'));provenance=json.loads((model/'training_provenance.json').read_text(encoding='utf8'))
        chosen=lock['selected_weights_sha256']==provenance['weights_sha256']
        text=f'''# InsureRAG condition-oriented MiniLM reranker — seed {seed}

Research-only local fine-tuning of `cross-encoder/ms-marco-MiniLM-L6-v2`, through
the preceding InsureRAG domain and retention training checkpoints. This is a
22,713,601-parameter passage reranker, not a Qwen generator or an embedding model.

- Actual CUDA training: one epoch, seed {seed}, {done['optimizer_steps']:,} optimizer updates,
  {done['pair_exposures']:,} question/passage pair presentations.
- Training questions: 25,987 unique (11,729 InsuranceQA, 259 government, 13,999 general).
- Inputs: query and passage only; one raw relevance logit, not a probability.
- Sequence maximum: 512 tokens, longest-first truncation.
- Objective: confidence-adjusted listwise cross entropy plus previous-specialist KL.
- Initial weights SHA-256: `{provenance['initial_weights_sha256']}`.
- Produced weights SHA-256: `{provenance['weights_sha256']}`.
- Training manifest SHA-256: `{provenance['training_manifest_sha256']}`.
- Selected for the frozen test evaluation: **{chosen}**.

The two seeds and six candidate/fusion configurations are selected using
validation only. Historical tests have previously been inspected. The fresh
government test has 157 mechanically extracted publisher Q/A pairs across
28 URLs. Unlabeled plausible answers, source clustering, dated government
guidance and unknown public-pretraining exposure limit interpretation.
No generated-answer accuracy, GraphRAG reasoning gain, production readiness,
or medical/legal correctness is established by retrieval metrics.

See `InsureRAG-VLM/docs/CONDITION_UPDATE_ZH.md`, `CONDITION_REPRODUCTION.md`,
and `reports/condition_listwise_v1/` for every control, interval, failure case and
training/model/code/data hash. Do not substitute this checkpoint for the
selected model unless its weight hash matches the frozen selection lock.

The public base model's license/card are preserved alongside this card. Dataset
terms remain separate: InsuranceQA research restrictions and HICRIC/SQuAD
attribution/share-alike terms must be reviewed before redistribution or commercial
use. Fine-tuning does not remove those conditions.
'''
        (parent/'MODEL_CARD.md').write_text(text,encoding='utf8')
        for name in ['BASE_LICENSE','BASE_MODEL_CARD.md']:
            original=ROOT/'../models/insurerag-retention-v2'/name
            if original.exists():shutil.copyfile(original,parent/name)
if __name__=='__main__':run()
