This run aborted during index construction before any model inference. The diagnostic runner used the old internal `corpus_source='local'` spelling after runtime validation had standardized the explicit mode to `documents`. No prediction or accuracy result was produced.

The runner was corrected to request `documents`; the subsequent complete run uses a new output directory. This aborted run and metadata are retained rather than presented as successful evaluation.
