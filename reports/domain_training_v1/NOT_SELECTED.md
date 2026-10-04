This development training run is retained for provenance and is not the final model.

After validation selection, an additional data-quality audit found 10 retained training
questions whose positive answer texts exactly duplicate held-out positive texts under
different author IDs. The existing exclusion rule covered answer IDs but not these texts.
The test scoring process was interrupted after partial scoring; no test metric was computed
or used to choose this correction. Its incomplete raw scores are retained.

The corrected `insuranceqa_hardneg_v2` fixture excludes those 10 whole training questions.
`domain_training_v2` repeats the same three-epoch training and validation protocol from
the original public base model. The government transfer benchmark was not scored during v1.
