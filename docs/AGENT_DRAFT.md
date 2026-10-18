# Consumer insurance QA Agent draft

This runnable scaffold adds a bounded controller loop around the existing hybrid
retrieval and insurance reranker. It addresses consumer rules and information,
not personal claim adjudication. It is independent of the holdout experiment.
No held-out questions, answers, gold spans or task labels enter the controller.
The current demonstration uses entirely fictional Cedar documents written for
these tests. They are not insurance guidance or part of the benchmark.

## Run the offline demonstration

From the repository root, with the project's existing Python dependencies:

```bash
python -m src.insurerag_vlm.agent_demo --scenario numeric --trace numeric-trace.json
python -m src.insurerag_vlm.agent_demo --scenario all --trace all-traces.json
python -m pytest -q tests/test_agent_draft.py
```

Trace files are created exclusively; choose a new filename for another run.
The default `scripted` controller is a synthetic fixture driver, not an LLM or
evidence of model reasoning quality. The retrieval is real project code:
`retrieve_text_candidates` combines local hashing dense and sparse channels;
`rerank_multimodal_candidates` supplies the existing insurance rule reranker.
Temporary synthetic documents and indices are removed after the run. No model
download, paid API, service startup, new framework or MCP server is required.

Expected outcomes: `ordinary`, `numeric` and `multi` answer with citations;
`refusal` abstains on a personal claim; `injection` rejects an attempted SHELL
action. The numeric example reads explicit 120 USD and 10 percent inputs and
calculates a 12.0 USD illustrative discount. The multi-evidence example reads
both policy-copy and complaint-escalation rules.

## Controller contract and limits

Initial retrieval always precedes controller actions. The only allowed actions
are SEARCH, READ, CALCULATE, ANSWER and ABSTAIN. Each controller response must be
one JSON object with exactly `action`, `parameters`, and `gaps`. The system prompt
in `agent_draft.py` specifies the complete schema. Unknown fields/actions,
duplicate JSON keys, malformed parameters and invalid citations fail closed.

SEARCH requires a concrete gap and permits at most three supplemental searches
by default; `--max-searches 0..3` can reduce that cap. Three is an initial cost
limit, not an optimized setting. Normalized repeated queries and repeated
actions stop the run. READ can access at most three previously retrieved IDs,
never paths or URLs. Evidence is clipped to 1,800 characters and marked if
truncated. No controller can write files, use a shell, browse or change limits.

CALCULATE accepts two bounded decimal literals with an exact quote and source
(`question` or an already read ID). Units must be explicit in the quote. Its
only operations are addition, subtraction, multiplication and division. Addition
and subtraction require identical units; multiplication/division require a
scalar, count or percent on the right. No arbitrary code, implicit exchange
rate, date arithmetic, unit conversion or automatic monetary rounding exists.
The trace preserves operands, units, sources and the decimal result.

ANSWER requires no declared gaps, bounded claims and exact quotes from read
evidence. ABSTAIN requires a concrete gap and reason. State records candidates,
read evidence, gaps, calculation outputs, called parameters, remaining budgets,
reported model usage (when available), errors and stop reason.

Defaults are eight controller actions, 512 output tokens per model call,
24,000 cumulative charged tokens, 8,192 context tokens, a 60-second workflow
deadline and a 15-second per-call deadline. Charging reserves the complete output
allowance and counts UTF-8 prompt bytes plus a 256-token overhead allowance;
this is deliberately conservative accounting, **not measured token usage or a
tokenizer-specific proof**. Actual Ollama counts are recorded separately and a
reported overrun stops the workflow. Output/context caps also go to the model
adapter. No result arriving after the deadline is accepted.

Synchronous local retrieval uses cooperative deadline checks before/after calls;
a blocked Python/native dependency cannot be forcibly interrupted in this draft.
Ollama inventory/generation requests have bounded HTTP timeouts; network read
timeouts are not a hard process wall-clock kill. Production hard cancellation
would need an isolated worker. The draft makes no such guarantee.

## Connect an existing local model or reranker

On October 18, 2026, the existing Agent contract tests passed **48/48** again.
Implementation and synthetic-trace hashes match the original delivery. The
[current verification report](../reports/agent_draft/status_20261018.json)
records the test, runtime probe and installed-weight inventory.

The read-only probe of `http://127.0.0.1:11434/api/tags` still refused connection.
No Ollama process, executable on PATH, standard per-user model store or configured
alternate endpoint was found in the checked locations. The project has BGE and
three BERT retrieval adaptation/reranker checkpoints; these are not generative
controller models. No installation, model download or service startup was attempted.
**No real-model controller loop was run.** Mock adapter tests verify only the call
contract and usage extraction. The artifact remains a runnable synthetic scaffold.

When an already installed local Ollama model is available, run just one
synthetic case with its verified digest:

```bash
python -m src.insurerag_vlm.agent_demo --controller ollama --scenario ordinary --model ollama:qwen3.5:4b --expected-digest YOUR_VERIFIED_DIGEST --trace local-model-trace.json
```

The adapter reuses `VLMClient`, requires an explicit model/digest and a localhost
HTTP endpoint, requests JSON with thinking disabled, and records available
`prompt_eval_count`/`eval_count`. It never silently substitutes a backend.
`--scenario all` is prohibited with the model controller to keep first runs
small. The historical intended Qwen digest was not reverified in this task;
availability and identity must be established before a future real-model run.

An existing local cross encoder can be connected with
`--reranker-path PATH_TO_EXISTING_CHECKPOINT`. This uses `LocalCrossEncoder` with
local-only loading and no remote code. Its integration is tested with a stub;
no cross-encoder weight inference is claimed for the default demo. Application
code may inject an already configured `DocumentRetrievalPipeline` into
`HybridRetrieverAdapter` to reuse its cached index and local encoder. Keep the
application corpus separate from all TEST material.

## Scope of evidence and remaining work

Synthetic tests cover ordinary, numeric, multi-evidence and refusal paths,
successful supplemental retrieval, unknown/invalid tools, repeat queries and
reads, action/search/token/context/time budgets, provenance, citations and
document prompt injection. The injection test deliberately has a controller
attempt an unauthorized tool after reading the attack; the dispatcher rejects
it. This proves the capability boundary in that test, not semantic immunity.

Exact quotes do not prove that a generated claim follows from them. The draft
does not independently verify entailment, temporal applicability, jurisdiction,
all arithmetic input interpretations, or a controller's decision to clear gaps.
Personal-claim exclusion and conflict handling also require real-model quality
validation. Until that exists, use synthetic examples only. Do not use model
results as a personal coverage or claim decision.

The next connection point is the existing local-model adapter: verify the
runtime, run a few synthetic full loops, inspect actions/citations and actual
usage, then evaluate on separately designated development material. Never tune
this controller on true TEST outputs. Current holdout progress is recorded in the
[aggregate checkpoint](../reports/holdout1000_v1/stage451_progress.json): 1,000
provisional TEST and 57 DEV, accepted=0 and frozen=false. Source and global
acceptance work continues independently of the unavailable generation runtime.
The six-baseline protocol and freeze-before-final-evaluation rule remain unchanged.
