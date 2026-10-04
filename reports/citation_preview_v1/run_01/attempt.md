# Initial diagnostic attempt

The first diagnostic tried the normal local-extractive answer path on a synthetic
declarations page. Upload succeeded, but the application abstained with
`insufficient_retrieved_evidence` (heuristic score 0.19), so it produced no served
citation and could not exercise preview binding. Its raw extraction included the
correct $735 amount. This attempt is not a model-generation or preview success.

The follow-up isolates the intended preview contract with a controlled answer
fixture. Upload, indexing, HTTP dispatch, preview registration, PDF-byte checking,
and rendering still execute normally. No runtime answer guards were changed to
make this diagnostic pass.
