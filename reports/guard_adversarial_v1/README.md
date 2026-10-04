# Adversarial guard diagnostic v1

**Synthetic, deliberately false candidates. No model inference or retrieval run.**

The citation helper accepted 6 of these 10 false candidates; the served-answer path accepted 6. These hand-selected cases do not estimate a real-world error rate.

The one-page ranking was supplied at a fixed heuristic score of 0.95 to isolate postprocessing. The initial run used explicit declarations/general metadata. A secondary check applied the current metadata inference to each of the same evidence strings; all 10 accept/reject outcomes were unchanged, including the six false acceptances. See inferred_metadata_check.json. All evidence, candidate answers, falsity rationales, and initial results are retained in cases.jsonl/results.jsonl. No runtime source or frozen fixture was changed.

| Case | Deliberately false claim | Helper | Served path | Reason |
| --- | --- | --- | --- | --- |
| G01 | same page deductible swap | FALSE ACCEPT | FALSE ACCEPT | supported |
| G02 | amount belongs to another scheduled item | FALSE ACCEPT | FALSE ACCEPT | supported |
| G03 | per day vs per claim unit | FALSE ACCEPT | FALSE ACCEPT | supported |
| G04 | percent vs currency unit | reject | reject | answer_amount_not_in_citation |
| G05 | minimum maximum value swap | FALSE ACCEPT | FALSE ACCEPT | supported |
| G06 | minimum only does not establish maximum | reject | reject | specific_question_terms_not_in_citation |
| G07 | coverage negation reversal | FALSE ACCEPT | FALSE ACCEPT | supported |
| G08 | effective expiration date swap | FALSE ACCEPT | FALSE ACCEPT | supported |
| G09 | wrong policy identifier | reject | reject | identifier_not_in_citation |
| G10 | repair estimate as unapproved payment | reject | reject | approved_payment_not_established |

The accepted cases show that matching terms, dates, and amounts is insufficient to establish field/value associations, units, bounds, or negation. High heuristic evidence scores can accompany false statements. These scores are not calibrated correctness probabilities.

The rejected controls show narrower checks for absent identifiers, unmatched currency/percentage forms, missing maximum evidence, and expressly unapproved payments. They do not establish general semantic reasoning.

See [known limitations](../../docs/KNOWN_LIMITATIONS.md) and [runtime follow-up](../packet_stress_v1/runtime_guard_followup.md). Reproduce with `python reports/guard_adversarial_v1/run.py --output <new-directory>`; existing reports are never overwritten.
