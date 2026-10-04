# Local Qwen adaptation experiment (2026-10-04)

This isolated experiment starts from PR #4 commit
`4779c1a44985334f9e1f58aa752564b9611aece3`. It does not alter historical scores.
Results are written under `reports/local_20261004/`; adapters under
`models/local_20261004_lora/`. No cloud inference or paid compute is used.

## Evaluation fixes

PDF preprocessing now emits the same corpus-relative `file#page=N` source as the
text loader. Subdirectories are retained; basenames alone are not identifiers.
Visual page IDs also include a relative-path hash so different packets with the
same PDF filename/content cannot share an ID. Existing indexes must be rebuilt
from their original corpus root: do not reinterpret an arbitrary legacy basename
as a unique document. Original paths remain in `source_path` for provenance.

Visual local reranking uses a fixed candidate budget of 40. `top_k` only controls
output depth; requesting more than the candidate budget raises an error. Hybrid
retrieval likewise uses configured page/snippet/merge budgets independently of
the output depth. Recall@5, Hit@5 and the @10 diagnostics still share the PR4
multi-gold scorer. Output five pages and output ten pages now have the same
five-page prefix under the same corpus/configuration. Answer packing still uses
at most three pages, so retrieval availability is not answer quality.

The real regression tests build PDFs, render them, build actual text and visual
indexes and invoke the QA, text-benchmark and visual evaluators. Rankings are not
mocked. Duplicate basenames in nested packets and output-depth consistency are
tested separately. The original 42 tests plus these two tests passed before
starting model experiments.

The final suite additionally covers a cosine-rank-21 rerank rescue (which fails
under the former 20-versus-40 candidate budgets), numeric hallucination detection,
abstention confusion counts and prevention of gold-source leakage in retrieved
context prompts. Text-path PDF image filenames also include a relative-path hash
to prevent same-named packet PDFs from overwriting each other's rendered images.

## Model and environment

Official model: https://huggingface.co/Qwen/Qwen3.5-2B (Apache-2.0).
Pinned revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`.
The official Qwen organization lists newer much larger Qwen3.8 models; they are
not suitable for this 8GB GPU. This is a resource-fit upgrade, not a claim that
Qwen3.5 is the newest family regardless of hardware.

Hardware: RTX 4070 Laptop 8GB; 32GB system RAM. Exact observations are saved in
`hardware.json`. Isolated Python 3.12.10 environment uses torch 2.11.0+cu128,
transformers 5.18.0 and peft 0.21.2. Full versions are in `requirements-lock.txt`.

The Unsloth Qwen3.5 guide warns against 4-bit QLoRA for this family:
https://unsloth.ai/docs/models/qwen3.5/fine-tune . Therefore the experiment uses
BF16 LoRA, with the vision encoder frozen and text-only inputs. Native Transformers
reference PyTorch DeltaNet kernels are used on Windows; no Triton/Unsloth install
is required. No claim is made about visual fine-tuning.

The longest selected training sample (443 tokens) passed a forward/backward and
optimizer step at 4572.5 MiB peak allocated CUDA memory. Preflight updates are
discarded; baseline and training each reload the unchanged official model.

## Data and frozen protocol

`prepare_local_experiment.py` reads the committed 3,850-row SFT file. It groups
original documents sharing exact or >=0.85 Jaccard 5-word-shingle evidence before
splitting. It removes duplicate question/evidence pairs and selected obvious
topic/evidence mismatches. Generated answer boilerplate is removed, and answerable
references must be extractive spans of the supplied evidence. Documents and exact
evidence hashes cannot overlap between splits. Semantic duplicates below the
lexical threshold remain possible.

Seed: 20261004. Caps chosen before model evaluation: train 600, dev 48, test 96.
The selected train/dev/test sets contain 28/9/9 documents and 480/25/62 answerable
examples respectively. Document groups are disjoint. Hashes and group membership
are in `data_audit.json`; exact records and provenance are in the split JSONLs.

Evidence originates in real public regulator documents; questions, reference
answers and unsupported labels are rule-generated. This is NOT a human-reviewed
or unrestricted insurance benchmark. The primary adaptation experiment supplies
evidence directly. It isolates answer adaptation rather than end-to-end retrieval.

Fixed training plan: rank 8, alpha 16, dropout 0; text attention and MLP projections;
batch size 1, accumulation 4; two passes over 600 examples, at most 300 optimizer
steps; AdamW, initial LR 1e-4, 15-step warmup and linear decay; gradient clipping 1.
Training has a 90-minute limit. Checkpoint selection uses dev token NLL at steps
150 and 300, including the unchanged base as a candidate. Test metrics are never
used for hyperparameter or checkpoint selection. Greedy decoding with thinking
disabled and at most 128 generated tokens is identical before and after training.

Answerable content F1 excludes citation suffixes and generated template headers.
Citation precision checks exact source strings; a separate lexical-support proxy
checks evidence token coverage and numbers. Neither proves semantic entailment.
Abstention precision and recall use synthetic labels. Latency is local batch-1
wall time including prefill/generation, with CUDA synchronization.

The supplementary retrieval baseline uses local hashing and fixed hybrid budgets
on document-scoped synthetic queries formed only from the held-out splits. It
consolidates judged positive pages. It is NOT a rerun of trained-BGE and must not
be compared with historical first-hit scores.

## Reproduction (PowerShell, repository root)

Use the isolated environment `..\.venv-insurerag\Scripts\python.exe`.
Reproduce in a fresh checkout: these commands overwrite this experiment's output
directory. Preserve the delivered evidence directory first. On Windows/Python
3.12, activate an isolated venv and install the captured dependency versions:

```powershell
python -m venv ..\.venv-insurerag
..\.venv-insurerag\Scripts\Activate.ps1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r reports/local_20261004/requirements-lock.txt
```

```powershell
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN='1'
$env:HF_HOME=(Join-Path (Get-Location) '.hf-cache')
python scripts/local_model_setup.py --revision 15852e8c16360a2fea060d615a32b45270f8a8fc
python scripts/prepare_local_experiment.py
$env:HF_HUB_OFFLINE='1'
$env:INSURERAG_USE_OLLAMA='0'
python scripts/run_local_lora_experiment.py --stage preflight
python scripts/local_retrieval_baseline.py
python scripts/run_local_lora_experiment.py --stage baseline
python scripts/run_local_lora_experiment.py --stage train
python scripts/run_local_lora_experiment.py --stage eval
python scripts/audit_local_splits.py
python scripts/audit_local_predictions.py
python scripts/summarize_local_experiment.py
python -m unittest discover tests -v
```

Reproduction requires the pinned model revision and dependency lock; scripts load
model weights offline after download. Do not regenerate splits during a run.
Training scripts write incremental logs and predictions so interrupted runs remain
auditable. Results and completion status must be read from produced artifacts,
not inferred from the presence of training scripts.

## Optional retrieved-context diagnostic and interactive local query

`local_retrieval_baseline.py` also writes a frozen retrieved-context diagnostic
for the 62 answerable test rows. It uses five retrieved pages and the existing
three-page/context-budget packer. Gold source IDs are withheld from the prompt;
the source identifiers in the retrieved evidence are the only citation options.
Page-one URL aliases are mapped to the retriever's displayed source for exact
citation scoring. Unsupported rows are excluded: those labels concern the original
supplied evidence and cannot establish that the whole corpus lacks an answer.

Run both the unchanged base and selected adapter against the identical file:

```powershell
python scripts/run_local_lora_experiment.py --stage rag-eval
python scripts/run_local_lora_experiment.py --stage rag-eval --adapter models/local_20261004_lora/step-300
```

Use the actual `selected_adapter` in `training_summary.json`, which may differ
from the example path. This supplemental diagnostic was added after the primary
protocol was frozen. It is not used for training or model selection and is not a
second independent benchmark.

For a local interactive check (one query per invocation, no network server):

```powershell
python scripts/query_local_adapter.py "What does the evidence say about collision deductibles?"
```

The published CLI defaults to the official base. An adapter requires explicit
`--adapter models/local_20261004_lora/step-300`; it is not promoted automatically.
It uses local-hashing hybrid retrieval and prints the answer, source list, packed
context and timings. The original application defaults are preserved. The saved
first-round CLI result used the then-selected adapter, as recorded in its JSON.

## Completed run and interpretation

The fixed step-300 adapter completed all 300 planned steps, with 1,200 sample
presentations (600 unique training rows), in 588.2 seconds. The independent
96-example test has 62 answerable examples; content F1 on those 62 only changed
from 0.2962 to 0.5153. Synthetic abstention precision/recall changed from
0.8824/0.8824 to 1.0/1.0 (base TP/FP/FN=30/4/4; adapter=34/0/0). See
`../reports/local_20261004/RESULTS.md` for complete metrics and supplemental results.

This supports using the adapter as a local research candidate for concise,
source-formatted extractive answers. It does not support deploying it for insurance
decisions or claiming general answer correctness: 23 of 62 answerable cases have
lower F1 than base, and exact citations/lexical support can be correct while the
answer selects an irrelevant sentence. For example, the actual-cash-value question
`a0c5012fc2fc17b97854` receives a deductible definition after adaptation. Test
results were not used for checkpoint selection or further training.

Independent split checks revalidated original-input evidence across assigned
splits, not just sampled rows: no >=0.85 5-shingle Jaccard cross-split pair, no
>=0.8 shorter-span containment pair, and no canonical URL/path alias crossing.
All declared document groups and exact evidence hashes are disjoint. Train/test
still share 33 question templates, and semantic duplicate publication versions
or model-pretraining overlap are not ruled out. See `independent_split_audit.json`.

The final pure scoring/prompt module was extracted from the GPU runner during
primary inference to keep CPU-only CI independent of torch/transformers. Saved
predictions were re-scored with that final implementation and matched every saved
metric. No primary scoring/protocol change or test-driven tuning occurred.
Invocation/hash logging was added after the baseline/training process started;
therefore those initial invocation files are absent. Their frozen plan, complete
predictions, training log and summary remain available. Do not infer an earlier
source snapshot hash from the final delivered source manifest.

The complete 49-test suite passed both in the isolated GPU environment and the
existing CPU-only environment. A real public-PDF smoke processed 20 PDFs/168 pages
and verified identity alignment and fixed-depth ranking prefixes. Historical
trained-BGE and Qwen2.5 QLoRA/pilot figures were not rerun by this experiment.

### Supplemental prompt defect and paired rerun

The first retrieved-context diagnostic reused a single-source prompt but placed
an instruction string in its `Source:` value. That contradicted the prompt's
instruction to copy the supplied source exactly. The adapter often cited the
instruction itself. Those complete outputs, metrics, invocation records and CLI
smoke are preserved under `prompt_v1_*` in the report directory. V1 content F1
was base 0.1373 / adapter 0.0856; exact-source citation precision was 0.3103 / 0.
These are diagnostic implementation-failure results, not valid deployment scores.

The corrected multi-source prompt requests one SOURCE identifier from the packed
evidence and does not append a misleading single-source field. Both base and the
unchanged step-300 adapter are rerun on identical frozen contexts, same decoding
parameters and scoring. No primary given-evidence prompt or primary result was
changed; no training or checkpoint selection was repeated. A targeted prompt
regression test guards against the defect. The final supplemental results use
this paired rerun, with the first version retained rather than hidden.

The corrected diagnostic completed: content F1 is 0.1329 (base) versus 0.0830
(adapter), and exact-source citation precision is 0.2308 versus 0.2703. Only
12/62 gold sources survive context packing despite 34/62 appearing in retrieval
top-5. Do not promote the adapter into the end-to-end answer path. No defaults
were switched and no further training or test-based tuning was performed.
