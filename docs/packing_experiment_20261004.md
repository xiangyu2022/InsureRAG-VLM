# Packing repair and exploratory regression

The original experiment and model checkpoint are preserved. This phase performs
no training or checkpoint selection. Its test rows were
already observed: all test-side measurements are exploratory regression results.
Publication is limited to an independent draft PR stacked on PR #4; no merge is
performed. The local CLI defaults to the official base; the adapter requires an
explicit `--adapter` argument and is not promoted into application defaults.

## Change

`pack_long_context_legacy` retains the exact old role-first/global-trim behavior.
The default packer now preserves incoming retrieval rank and takes up to three
distinct nonempty sources. It selects sentence fragments using question terms
(excluding the document-scoping URL), suppresses navigation noise and duplicates,
and grows separate source budgets in round-robin increments. Existing limits of
900 evidence characters per source and 3,200 context characters are unchanged.
No gold source, answer, page or span enters the packer.

`pack_context_with_audit` returns the context, complete source IDs, individual
fragments with source associations, truncation flags, and dropped-page reasons.
The local Qwen path supplies a callback counting the **complete chat prompt with
the actual local tokenizer**, including instructions/question. The cap is 2,048
input tokens; inference verifies that count again and reserves 128 output tokens
within the model context limit. There is no tokenizer input truncation. Generic
provider-agnostic callers retain the character fallback and must supply their
own provider-specific counter for the same guarantee.

## Development and freeze

Development uses the existing 25 positive dev rows plus generated-PDF and Unicode
capacity fixtures. The initial rank/budget fix raised dev gold-source presence
from 5 to 11, while raw snippets still contained navigation and scope-URL matches.
Query-only sentence selection was then added on dev, before any new test run.
The initial dev packing summary is preserved. Source retention is not answer
coverage: exact reference-span retention remained poor, and no claim of semantic
correctness follows from a source header being present.

The full 53-test suite passes, including actual PDF retrieval and long-first-page
budget regressions. A byte-heavy multilingual fixture uses the actual Qwen
tokenizer to retain three intact source identifiers within 400 complete-prompt
tokens. Each data-preparation run verifies source/fragment associations.

`frozen_protocol.json` records dev outcomes and hashes of the selected packer,
evaluation code, original exposed-test input, dev contexts and unchanged step-300
checkpoint before test contexts are prepared. Test execution checks these hashes.
All 62 original positive test rows are retained. Original negative labels concern
their original evidence only and are not repurposed as retrieved-context labels.
Both base and step-300 receive the same old/new contexts and decoding settings.

Queries retain the first experiment's explicit document scoping. This is not an
unrestricted open-domain query benchmark, and no gold page is inserted. Some
rule-generated questions are under-specified (e.g. summarize a glossary while
the reference names one particular definition). Content F1 cannot resolve that
label ambiguity. No low-scoring rows are removed.

## New holdout feasibility

After excluding all prior SFT sources and the travel document inspected in the
CLI smoke, eight source identifiers remain. Five have high shared-clause or
duplicate-version risk under conservative lexical checks. Three remain lexically
unflagged, but have no independent answerability/reference labels or verified
semantic version independence. No new confirmatory benchmark is claimed or
manufactured. See `holdout_feasibility.json` for all candidates and scores.

## Reproduce in a fresh checkout

Use the first experiment's pinned model, isolated environment and dependency lock.
Published reports contain lightweight summaries, not raw split/prediction files,
local indexes, model weights or optimizer states. First reproduce the initial
experiment to obtain its local splits/index and selected adapter. Archive the
published packing report directory under a new name in your fresh checkout, then
create an empty `reports/packing_20261004` directory; the freeze command deliberately
refuses to overwrite an existing protocol. Do not overwrite the delivered evidence
directory. From the repository root:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN='1'
$env:INSURERAG_USE_OLLAMA='0'
python -m unittest discover -s tests -v *> reports/packing_20261004/unit_tests.log
python scripts/prepare_packing_experiment.py --split dev
python scripts/audit_packing_holdout.py
python scripts/run_packing_evaluation.py --split dev --model-kind base
python scripts/run_packing_evaluation.py --split dev --model-kind adapter
python scripts/freeze_packing_protocol.py
python scripts/prepare_packing_experiment.py --split test
python scripts/run_packing_evaluation.py --split test --model-kind base
python scripts/run_packing_evaluation.py --split test --model-kind adapter
python scripts/query_local_adapter.py "What should I check before buying travel insurance?"
python scripts/summarize_packing_experiment.py
```

Only test source-availability measurements use the old gold IDs, after packing.
No context-level refusal precision/recall is reported without new labels. Gold
reference absence is not automatically an unsupported label. Citation resolution
means a generated identifier points to a packed source; it does not prove that
source entails the answer. Review the complete result, proxies, failure examples,
per-row predictions and capacity audits under `reports/packing_20261004/`.
