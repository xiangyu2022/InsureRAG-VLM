# No-image control: independent review

September 20, 2026 UTC. AI-assisted/manual-agent review of four saved responses; not human expert certification. Raw predictions, fixtures, and automated metrics are unchanged. No model was called during this review.

**Result: three semantic refusals and one unsupported guess.** Only one refusal satisfies the stricter requirement that both answer and evidence strings be empty. Thus the recorded **1/4 strict-empty-refusal** result is correct, but it does not mean three answers invented policy facts.

| Question | Raw answer / abstain | Manual interpretation | Strict result |
|---|---|---|---|
| Dwelling limit | `Not visible` / true; empty evidence | Refuses to supply a value; nonempty answer violates the required empty-string convention | Fail |
| Wind/hail deductible | `Not shown` / true; empty evidence | Refuses to supply a value; same convention failure | Fail |
| Explicit exclusion | `War` / false; evidence `War is excluded` | Unsupported policy-specific guess and invented evidence; no document was supplied | Fail |
| Annual premium | Empty answer and evidence / true | Correct empty refusal | Pass |

All four responses are valid schema-shaped JSON, complete with `done_reason=stop`, and untruncated. The first two failures concern the abstention content convention, not malformed JSON or an invented dollar amount. The exclusion response is a substantive grounding failure even though its JSON is valid.

## Control identity

Compared with `qwen35_v1`, the saved system prompt, JSON schema, four question strings, fixture-lock hash, generation options, resolved model, model digest, Ollama version, endpoint, and `thinking=false` setting match exactly. The baseline contains those same four question forms repeated across three images; this control executes each unique text form once. The baseline prediction hash also matches the hash recorded by the control runner.

Both use Qwen3.5 `qwen3.5:4b`, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`, temperature 0, seed 42, 8,192 context tokens, 192 output tokens, top-k 40, top-p 1, repeat penalty 1, and presence penalty 0.

Code inspection confirms that the control calls `generate_chat(SYSTEM, question, response_format=SCHEMA)`. The shared `VLMClient` sends only fresh system/user messages and attaches an `images` field only when image bytes are provided. The control supplies none. All four saved records report `input_modality=text`, `image_count=0`, and `images=[]`; the 12 baseline records report one image. Neither path adds OCR text, expected answers, or fixture transcriptions to the user message. The shared runtime source hash is identical across runs. This is code-and-metadata verification; the report does not claim a separately recorded network packet capture.

## Interpretation and limits

With images, the baseline produced all nine changed target answers correctly across three clean synthetic declarations pages, plus three empty refusals for the absent premium. With images removed, it supplied no dwelling or deductible value, but guessed an exclusion. Together, these observations support sensitivity to the supplied image content in this narrow diagnostic and expose a missing-evidence failure. They do not prove a particular internal mechanism or safe behavior on arbitrary insurance documents.

Do not describe this as accuracy falling from 100% to 25%: the baseline asks for readable image facts, whereas every no-image request lacks necessary evidence and should be refused. Four unique question forms, one seed, and three synthetic layouts do not support a population hallucination rate. No confidence interval or general visual-accuracy claim is warranted from this control alone.

Integrity hashes:

- Baseline `qwen35_v1/predictions.jsonl`: `a4c40f9e701cd0f9791cca5da1487d20698113dc62a148dca2bf4e2bbc3f34d7`.
- Control `no_image_v1/predictions.jsonl`: `2344fc03c690637df326538adfb21ab4cd31bf6b533d84f59f2761ded83bb6ea`.
- Shared runtime `src/insurerag_vlm/vlm.py`: `109f2555cc767d4ef8119296d3edb54433abe1306157974c740a5b6637cada75`.
