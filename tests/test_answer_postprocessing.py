from pathlib import Path
from unittest.mock import patch

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline


def page(source, text, role="declarations"):
    return {"source": source, "text_snippet": text, "score": 0.95,
            "document_type": role, "primary_clause_type": "general", "table_fields": []}


def serve(question, raw, pages, context=None):
    pipeline = DocumentRetrievalPipeline(ModelConfig(vlm_model="local-extractive", retrieval_model="local-hashing"))
    result = {"answer": raw, "source_ranking": pages, "generation_used": True,
              "answer_backend": "test-model", "backend_metadata": {}}
    if context is not None:
        result["retrieval_context"] = context
    with patch.object(pipeline, "query_with_ranking", return_value=result):
        return pipeline.query_structured(question, Path("unused"))


def supported(question, answer, evidence):
    return DocumentRetrievalPipeline._citation_support_details(
        question, answer, [{"evidence_text": evidence, "document_type": "declarations"}],
    )


def test_markdown_source_resolves_exact_page_instead_of_a_higher_ranked_distractor():
    result = serve("What comprehensive deductible is actually declared for policy ZX-14?",
                   "The comprehensive deductible for policy ZX-14 is $750.\n\n* **Source:** declarations.pdf#page=2 (issued policy)",
                   [page("guide.pdf#page=1", "A comprehensive deductible example is $500."),
                    page("declarations.pdf#page=2", "Policy ZX-14. Comprehensive deductible: $750.")])
    assert not result["abstain"]
    assert result["citations"][0]["source"] == "declarations.pdf#page=2"
    assert result["citation_origin"] == "model_source"
    assert result["citations"][0]["origin"] == "model_source"
    assert "Source:" not in result["answer"]


def test_explicit_unknown_citation_cannot_be_replaced_even_if_known_source_is_mentioned():
    result = serve("What is the collision deductible?",
                   "The collision deductible is $750 according to policy.pdf#page=1.\nSOURCE: invented.pdf#page=1",
                   [page("policy.pdf#page=1", "Collision deductible: $750.")])
    assert result["abstain"]
    assert result["citation_support_reason"] == "missing_citation"
    assert result["citation_origin"] is None


def test_automatic_citation_requires_support_before_retrieval_rank():
    pages = [page("guide.pdf#page=1", "Educational example only. Collision deductible: $500.", "consumer_guide"),
             page("declarations.pdf#page=1", "Policy ZX-14. Comprehensive deductible: $750.")]
    pages[0]["score"] = 1.0
    pages[1]["score"] = 0.7
    result = serve("What comprehensive deductible is declared for policy ZX-14?",
                   "The comprehensive deductible for policy ZX-14 is $750.\n\nThe $500 in guide.pdf is an educational example.", pages)
    assert not result["abstain"]
    assert result["citation_origin"] == "evidence_selection"
    assert result["citations"][0]["origin"] == "evidence_selection"
    assert result["citations"][0]["source"] == "declarations.pdf#page=1"
    assert result["answer_repaired"]
    assert "$500" not in result["answer"]


def test_automatic_citation_returns_none_when_no_candidate_supports_answer():
    result = serve("What is the comprehensive deductible?", "The comprehensive deductible is $9,999.",
                   [page("policy.pdf#page=1", "Comprehensive deductible: $750.")])
    assert result["abstain"] and result["citation_origin"] is None
    assert not result["citations"] and not result["answer_repaired"]


def test_automatic_citation_cannot_use_a_fact_outside_packed_context():
    result = serve("What is the comprehensive deductible?", "The comprehensive deductible is $750.",
                   [page("policy.pdf#page=1", "Comprehensive deductible: $750.")],
                   context="SOURCE: policy.pdf#page=1\nComprehensive coverage details.")
    assert result["abstain"] and result["citation_origin"] is None


def test_scoped_document_lookup_normalizes_plural_and_maximum_without_losing_document_identity():
    evidence_page = page("arizona-auto-guide.pdf#page=4", "PIP coverage includes up to $4,500 for funeral expenses.", "consumer_guide")
    evidence_page["snippet_support"] = ["Arizona Auto Guide. PIP coverage includes up to $4,500 for funeral expenses."]
    question = "Using the uploaded Arizona auto guide, what maximum funeral expense benefit is included in PIP coverage?"
    result = serve(question, "The maximum funeral expense benefit included in PIP coverage is $4,500.\n**Source:** arizona-auto-guide.pdf#page=4", [evidence_page])
    assert not result["abstain"] and result["citation_origin"] == "model_source"
    wrong = serve(question.replace("Arizona", "Ohio"), result["raw_answer"], [evidence_page])
    assert wrong["abstain"] and wrong["citation_support_reason"] == "document_scope_not_in_citation"


def test_plural_normalization_does_not_confuse_minimum_with_maximum():
    assert supported("What is the maximum funeral expense benefit?", "Funeral expenses are covered up to $4,500.",
                     "Funeral expense coverage is up to $4,500.")[0]
    assert not supported("What is the minimum funeral expense benefit?", "The minimum funeral expense benefit is $4,500.",
                         "Funeral expense coverage is up to $4,500.")[0]


def test_page_one_is_not_a_substring_citation_match_for_page_ten():
    pages = [page("policy.pdf#page=1", "Collision deductible $500."),
             page("policy.pdf#page=10", "Collision deductible $750.")]
    assert DocumentRetrievalPipeline._extract_answer_source("**Source:** policy.pdf#page=10", pages) == "policy.pdf#page=10"
    assert DocumentRetrievalPipeline._extract_answer_source("SOURCE: policy.pdf#page=100", pages) == "policy.pdf#page=100"


def test_amount_delimiter_is_not_part_of_value_but_distinct_amounts_remain_distinct():
    question = "What is the collision deductible?"
    assert supported(question, "The collision deductible is $750.", "The collision deductible is $750, per claim.")[0]
    for amount in ("$7,500", "$750.50", "$75"):
        assert not supported(question, f"The collision deductible is {amount}.", "Collision deductible: $750.")[0]


def test_lookup_wording_does_not_replace_exact_policy_identity_constraint():
    assert supported("What dwelling limit is declared for policy ZX-14?", "The dwelling limit for ZX-14 is $280,000.",
                     "Policy ZX-14. Dwelling limit: $280,000.")[0]
    assert not supported("What dwelling limit is declared for policy ZX-15?", "The dwelling limit is $280,000.",
                         "Policy ZX-14. Dwelling limit: $280,000.")[0]


def test_printed_identifier_lookup_does_not_require_literal_printed_word():
    assert supported("What policy identifier is printed on claim report CL-Z?", "The policy identifier on claim CL-Z is ZX-14.",
                     "Claim report CL-Z. Policy identifier: ZX-14.")[0]
    assert not supported("What policy identifier is printed on claim report CL-Y?", "The policy identifier is ZX-14.",
                         "Claim report CL-Z. Policy identifier: ZX-14.")[0]
    assert not supported("What policy identifier is printed on claim report cl-y?", "The policy identifier is ZX-14.",
                         "Claim report cl-z. Policy identifier: ZX-14.")[0]


def test_date_lookup_preserves_both_exact_date_and_effective_vs_expiration_field():
    evidence = "Policy ZX-14. Endorsement RX-03 effective June 2, 2027."
    question = "What effective date is stated on endorsement RX-03 for policy ZX-14?"
    assert supported(question, "The effective date is June 2, 2027.", evidence)[0]
    assert not supported(question, "The effective date is June 3, 2027.", evidence)[0]
    assert not supported("What expiration date is stated for policy ZX-14?", "The expiration date is June 2, 2027.", evidence)[0]


def test_coverage_and_scheduled_item_discriminators_remain_required():
    assert not supported("What is the comprehensive deductible?", "The deductible is $750.", "Collision deductible: $750.")[0]
    assert not supported("What is the laptop coverage limit?", "The limit is $4,000.", "Camera coverage limit: $4,000.")[0]


def test_uncited_comparison_is_removed_only_when_complete_primary_paragraph_is_supported():
    raw = "The water backup limit in endorsement RX-03 is $3,700.\n\nThe historical base limit was $5,700.\n\n**Source:** endorsement.pdf#page=1"
    result = serve("What water backup limit is stated in endorsement RX-03?", raw,
                   [page("endorsement.pdf#page=1", "Endorsement RX-03. Water backup limit is $3,700.", "endorsement")])
    assert not result["abstain"]
    assert result["answer_repaired"]
    assert result["raw_answer"] == raw
    assert result["answer"] == "The water backup limit in endorsement RX-03 is $3,700."
    assert result["answer_backend"] == "deterministic-evidence-repair"


def test_mixed_supported_and_unsupported_amounts_in_same_claim_are_not_accepted():
    result = serve("What water backup limit is stated in endorsement RX-03?",
                   "The water backup limit in endorsement RX-03 is $3,700 or $5,700.\n\nSOURCE: endorsement.pdf#page=1",
                   [page("endorsement.pdf#page=1", "Endorsement RX-03. Water backup limit is $3,700.", "endorsement")])
    assert result["abstain"]
    assert not result["answer_repaired"]


def test_primary_paragraph_repair_cannot_drop_second_requested_deductible():
    result = serve("What is the collision deductible and the comprehensive deductible?",
                   "The collision deductible is $750.\n\nThe comprehensive deductible is $9,999.\n\nSOURCE: policy.pdf#page=1",
                   [page("policy.pdf#page=1", "Collision deductible: $750. Comprehensive deductible: $1,500.")])
    assert result["abstain"]
    assert not result["answer_repaired"]
    assert result["answer"] == ""


def test_retrieved_but_unpacked_fact_cannot_validate_answer():
    result = serve("What is the collision deductible?", "The collision deductible is $750.\nSOURCE: policy.pdf#page=1",
                   [page("policy.pdf#page=1", "Collision deductible: $750.")],
                   context="SOURCE: policy.pdf#page=1\nROLE: declarations / general\nCollision coverage details.")
    assert result["abstain"]
    assert result["citation_support_reason"] == "answer_amount_not_in_citation"


def test_unsequenced_version_cannot_be_declared_controlling_but_scoped_version_fact_can():
    evidence = "Policy ZX-14. Version label A. Equipment coverage limit: $3,700. Which version controls is not established."
    assert not supported("Which equipment limit controls policy ZX-14?", "The controlling equipment limit is $3,700.", evidence)[0]
    assert supported("Without deciding which version controls, what equipment coverage limit is printed in version A of ZX-14?",
                     "The equipment coverage limit printed in version A of ZX-14 is $3,700.", evidence)[0]
    assert not supported("What equipment coverage limit is printed in version B of ZX-14?", "The limit is $3,700.", evidence)[0]
    assert not supported("Without deciding which version controls, what equipment coverage limit is printed in version B of ZX-14?",
                         "The equipment limit is $3,700.", evidence)[0]


def test_repair_estimate_cannot_be_substituted_for_unapproved_insurer_payment():
    evidence = "Claim report CL-Z. Policy ZX-14. Repair estimate: $9,400. No coverage acceptance or approved insurer payment is recorded."
    assert not supported("What approved insurer payment is established for claim CL-Z under policy ZX-14?",
                         "The approved insurer payment is $9,400.", evidence)[0]


def test_excluding_an_example_does_not_allow_example_to_establish_policy_value():
    question = "What collision deductible is actually declared for policy ZX-14, rather than the guide example?"
    assert supported(question, "The collision deductible for ZX-14 is $750.", "Policy ZX-14. Collision deductible: $750.")[0]
    value = DocumentRetrievalPipeline._citation_support_details(question, "The collision deductible for ZX-14 is $500.",
             [{"evidence_text": "Educational example only for ZX-14: collision deductible $500.", "document_type": "consumer_guide"}])
    assert not value[0]
    direct = DocumentRetrievalPipeline._citation_support_details(
        "What collision deductible is declared for policy ZX-14?", "The collision deductible for ZX-14 is $500.",
        [{"evidence_text": "Educational example only for ZX-14: collision deductible $500.", "document_type": "consumer_guide"}])
    assert not direct[0]
