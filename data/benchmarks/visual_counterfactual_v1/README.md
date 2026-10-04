# Image-only counterfactual diagnostic v1

Three AI-authored declarations pages use the same layout and question wording, while three target facts change: dwelling limit, wind/hail deductible, and explicit excluded cause. Personal-property/liability limits remain as distractors. Annual premium is absent on all pages. All names/documents are fictional; there is no customer information or legal force.

There are twelve cases: the same four questions on each of three images, comprising nine supported facts and three missing-fact abstentions. Seed 314159 fixes the invented variants. PDF and PNG assets, readable design specification, and exact expected keys were frozen using `manifest.lock.json` before inference. All three 1224 × 1584 PNGs were visually inspected for legibility and clipping. PyMuPDF renders at 144 DPI.

Frozen fixture verification always checks exact original file hashes; altered bytes fail even if visible content is unchanged. Regeneration tests separately require two builds in the same environment to produce identical PDF/PNG bytes, and compare the rebuilt PDF text, page/image dimensions, target facts, distractors, and absent premium with the frozen originals. They also check that the rebuilt PNG pixels match a fresh render of its PDF. Rebuilt compression or font bytes may differ across operating systems; regenerated assets never replace the frozen inference inputs. `requirements-dev.txt` pins PyMuPDF 1.28.2 and Pillow 12.3.0; both declare Python >=3.10 and support the configured Python 3.11/3.12 test matrix. These checks do not imply that remote CI jobs have executed.

## What is actually tested

The evaluator calls `VLMClient.generate_with_images` using the exact question as user text and one attached PNG. The system message only defines the answer/evidence/abstain JSON contract and the instruction to use the image. **No OCR, PDF extraction, page transcription, reference answer, design values, or expected label is appended to the request.** The model never receives the PDF file itself. Request text, frozen PNG hash, and runtime-reported transmitted image hash are saved for inspection. A mocked-transport test inspects all twelve actual API payloads to verify this data flow without calling a model.

The requests are separate stateless chats. For each requested field, question text is identical across A/B/C; the changed image is the only task-content input that varies. Getting all three differing target values correct provides a small check that the image path is active. It does not establish general visual accuracy or prove performance on scans, dense tables, rotated pages, handwriting, or real policy disputes.

## Run

```powershell
python scripts/eval_visual_counterfactual.py --verify-only --output reports/visual_counterfactual_v1/validation
python scripts/eval_visual_counterfactual.py --endpoint http://localhost:11435 --model qwen3.5:4b --num-predict 192 --timeout 600 --output reports/visual_counterfactual_v1/my_new_run
```

The model must already exist locally and advertise vision capability. No runtime weights are stored in this repository, no alternate model is selected, and no paid/cloud API is used. Keep existing result directories immutable. The builder refuses to overwrite an existing frozen manifest; later changes need a new fixture version.

## Outputs and limitations

- Boundary-aware answer-key match reports whether the expected value/cause occurs in the answer. It is not semantic correctness.
- The separate evidence-key check requires the field label and expected value in the model's transcription. It is not exact OCR or a verified quotation metric; valid abbreviations can fail it.
- Strict unsupported abstention requires `abstain=true` with empty answer/evidence.
- Only exact three-key JSON and completed, non-truncated generation can pass.
- Per-field counterfactual results show all three predicted values and whether every changed target was answered correctly.
- Raw response, generation timings/tokens, model digest, options, code/fixture/image hashes, exact request text, and image-transport metadata remain recorded.
- Counts and Wilson intervals are descriptive. Nine facts and three absent-fact cases share three clean authored pages; intervals ignore that clustering and cannot establish reliability on a document population.

This is an image-input regression diagnostic, separate from the public-guide text benchmark and the application PDF/repair stress suite. It does not exercise retrieval, the application answer-repair layer, or adapter training.
