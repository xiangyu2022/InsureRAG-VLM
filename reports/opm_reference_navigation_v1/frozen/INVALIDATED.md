This development run is invalidated. The direct-lookup arm used the gold target
as an oracle rather than parsing the supplied anchor. Its 100% result is not a
measured lookup baseline. Raw outputs are retained for traceability only.

The corrected run, `../corrected_frozen`, parses literal references from the
input anchor and resolves printed page labels without reading target labels.
The graph and BM25 configurations are unchanged. This correction occurred after
test execution, so the corrected result is an exploratory integration diagnostic.
