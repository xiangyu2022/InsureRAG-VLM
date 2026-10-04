# Excluded before validation

This attempt is retained for audit only. On 2026-10-01, before any validation or new test inference, a source-isolation audit found 3,916 negative pairs from new held-out source groups in 2,522 training groups. This is not a held-out question-label leak, but violates the intended strict source separation. No checkpoint from this attempt participates in selection.

`scripts/isolate_retention_sources.py` removes all held-out source texts from positive and negative training pairs. Retention v2 restarts from the original selected InsuranceQA v2 checkpoint, not this attempt. Its selection criteria are unchanged; a new protocol records the corrected training manifest. No model performance influenced this correction.
