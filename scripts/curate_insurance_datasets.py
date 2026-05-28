import argparse
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.insurerag_vlm.pdf import extract_text_by_page


CORE_SFT_DOCS = {
    "md_homeowners_declarations_page.pdf",
    "md_homeowners_insurance_guide.pdf",
    "md_auto_insurance_guide.pdf",
    "nc_homeowners_guide_archive.pdf",
    "md_homeowners_disclosure_notice.pdf",
    "nc_disability_insurance_guide.pdf",
}

EXTERNAL_WEB_SOURCES = [
    {
        "source_id": "naic_consumer_glossary",
        "name": "NAIC Glossary of Insurance Terms",
        "url": "https://content.naic.org/consumer_glossary",
        "source_file": "naic_consumer_glossary.html",
        "authority": "NAIC",
        "insurance_domain": ["insurance"],
        "content_type": "glossary",
        "sft_eligible": True,
    },
    {
        "source_id": "ca_doi_auto_guide",
        "name": "California DOI Automobile Insurance Guide",
        "url": "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/01-auto/auto101.cfm",
        "source_file": "ca_doi_auto_guide.html",
        "authority": "California Department of Insurance",
        "insurance_domain": ["auto"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "ca_doi_auto_terms",
        "name": "California DOI Automobile Insurance Terms",
        "url": "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/01-auto/autoterms.cfm",
        "source_file": "ca_doi_auto_terms.html",
        "authority": "California Department of Insurance",
        "insurance_domain": ["auto"],
        "content_type": "glossary",
        "sft_eligible": True,
    },
    {
        "source_id": "ca_doi_glossary",
        "name": "California DOI Glossary of Insurance Terms",
        "url": "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/20-Glossary/",
        "source_file": "ca_doi_glossary.html",
        "authority": "California Department of Insurance",
        "insurance_domain": ["insurance"],
        "content_type": "glossary",
        "sft_eligible": True,
    },
    {
        "source_id": "ca_doi_residential_guide",
        "name": "California DOI Residential Insurance Guide",
        "url": "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/03-res/Residential_Insurance.cfm",
        "source_file": "ca_doi_residential_guide.html",
        "authority": "California Department of Insurance",
        "insurance_domain": ["homeowners", "renters"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "tx_doi_auto_guide",
        "name": "Texas DOI Auto Insurance Guide",
        "url": "https://www.tdi.texas.gov/pubs/consumer/cb020.html",
        "source_file": "tx_doi_auto_guide.html",
        "authority": "Texas Department of Insurance",
        "insurance_domain": ["auto"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "tx_doi_home_guide",
        "name": "Texas DOI Home Insurance Guide",
        "url": "https://www.tdi.texas.gov/pubs/consumer/cb025.html",
        "source_file": "tx_doi_home_guide.html",
        "authority": "Texas Department of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "tx_doi_deductibles",
        "name": "Texas DOI Deductibles Guide",
        "url": "https://www.tdi.texas.gov/tips/deductibles.html",
        "source_file": "tx_doi_deductibles.html",
        "authority": "Texas Department of Insurance",
        "insurance_domain": ["auto", "homeowners"],
        "content_type": "deductible_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "tx_doi_replacement_cost_acv",
        "name": "Texas DOI Home Policies Replacement Cost or Actual Cash Value",
        "url": "https://www.tdi.texas.gov/tips/home-insurance-policies-replacement-cost-or-actual-cash-value.html",
        "source_file": "tx_doi_replacement_cost_acv.html",
        "authority": "Texas Department of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "settlement_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "ny_dfs_home_basic_coverage",
        "name": "New York DFS Basic Homeowners Coverage",
        "url": "https://www.dfs.ny.gov/consumers/help_for_homeowners/insurance/basic_coverage",
        "source_file": "ny_dfs_home_basic_coverage.html",
        "authority": "New York Department of Financial Services",
        "insurance_domain": ["homeowners"],
        "content_type": "coverage_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "ny_dfs_choosing_policy",
        "name": "New York DFS Choosing a Homeowners Policy",
        "url": "https://www.dfs.ny.gov/consumers/help_for_homeowners/insurance/choosing_a_policy",
        "source_file": "ny_dfs_choosing_policy.html",
        "authority": "New York Department of Financial Services",
        "insurance_domain": ["homeowners"],
        "content_type": "policy_form_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "ny_dfs_how_much_insurance",
        "name": "New York DFS Determining How Much Homeowners Insurance You Need",
        "url": "https://www.dfs.ny.gov/consumers/help_for_homeowners/insurance/determining_how_much_insurance_you_need",
        "source_file": "ny_dfs_how_much_insurance.html",
        "authority": "New York Department of Financial Services",
        "insurance_domain": ["homeowners"],
        "content_type": "settlement_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "ny_dfs_problems_obtaining",
        "name": "New York DFS Homeowners Problems Obtaining Insurance",
        "url": "https://www.dfs.ny.gov/consumers/help_for_homeowners/insurance/problems_obtaining_insurance",
        "source_file": "ny_dfs_problems_obtaining.html",
        "authority": "New York Department of Financial Services",
        "insurance_domain": ["homeowners"],
        "content_type": "coverage_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "wa_oic_home_claims",
        "name": "Washington OIC Filing a Homeowner Insurance Claim",
        "url": "https://www.insurance.wa.gov/insurance-resources/home-insurance/home-insurance-claims/filing-homeowner-insurance-claim",
        "source_file": "wa_oic_home_claims.html",
        "authority": "Washington Office of the Insurance Commissioner",
        "insurance_domain": ["homeowners"],
        "content_type": "claims_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "wa_oic_consumer_glossary",
        "name": "Washington OIC Consumer Insurance Glossary",
        "url": "https://www.insurance.wa.gov/consumers-insurance-glossary",
        "source_file": "wa_oic_consumer_glossary.html",
        "authority": "Washington Office of the Insurance Commissioner",
        "insurance_domain": ["insurance"],
        "content_type": "glossary",
        "sft_eligible": True,
    },
    {
        "source_id": "wa_oic_homeowner_guide_pdf",
        "name": "Washington OIC Homeowner Insurance Guide PDF",
        "url": "https://www.insurance.wa.gov/sites/default/files/documents/homeowner-insurance-guide_1.pdf",
        "source_file": "wa_oic_homeowner_guide_pdf.html",
        "authority": "Washington Office of the Insurance Commissioner",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "wa_oic_water_damage_mold",
        "name": "Washington OIC Leaks, Water Damage and Mold",
        "url": "https://www.insurance.wa.gov/insurance-resources/home-insurance/how-home-insurance-works/leaks-water-damage-and-mold",
        "source_file": "wa_oic_water_damage_mold.html",
        "authority": "Washington Office of the Insurance Commissioner",
        "insurance_domain": ["homeowners"],
        "content_type": "coverage_limitation_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "wa_oic_home_insurance_works",
        "name": "Washington OIC How Home Insurance Works",
        "url": "https://www.insurance.wa.gov/insurance-resources/home-insurance/how-home-insurance-works",
        "source_file": "wa_oic_home_insurance_works.html",
        "authority": "Washington Office of the Insurance Commissioner",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "wa_oic_homeowner_insurance",
        "name": "Washington OIC Learn How Homeowner Insurance Works",
        "url": "https://www.insurance.wa.gov/what-homeowner-insurance",
        "source_file": "wa_oic_homeowner_insurance.html",
        "authority": "Washington Office of the Insurance Commissioner",
        "insurance_domain": ["homeowners"],
        "content_type": "settlement_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "wa_oic_renter_insurance",
        "name": "Washington OIC How Renter Insurance Works",
        "url": "https://www.insurance.wa.gov/insurance-resources/home-insurance/how-home-insurance-works/how-renter-insurance-works",
        "source_file": "wa_oic_renter_insurance.html",
        "authority": "Washington Office of the Insurance Commissioner",
        "insurance_domain": ["renters", "homeowners"],
        "content_type": "coverage_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "wa_oic_lender_placed",
        "name": "Washington OIC Lender-Placed Insurance",
        "url": "https://www.insurance.wa.gov/insurance-resources/home-insurance/how-home-insurance-works/lender-placed-insurance",
        "source_file": "wa_oic_lender_placed.html",
        "authority": "Washington Office of the Insurance Commissioner",
        "insurance_domain": ["homeowners"],
        "content_type": "declarations_page_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "wa_oic_totaled_car",
        "name": "Washington OIC What Happens After Your Car Gets Totaled",
        "url": "https://www.insurance.wa.gov/insurance-resources/auto-insurance/auto-insurance-claims/what-happens-after-your-car-gets-totaled",
        "source_file": "wa_oic_totaled_car.html",
        "authority": "Washington Office of the Insurance Commissioner",
        "insurance_domain": ["auto"],
        "content_type": "settlement_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "ma_doi_auto_101",
        "name": "Massachusetts DOI Auto Insurance 101 and 102",
        "url": "https://www.mass.gov/info-details/auto-insurance-101-and-102-a-crash-course-in-coverage-and-claims",
        "source_file": "ma_doi_auto_101.html",
        "authority": "Massachusetts Division of Insurance",
        "insurance_domain": ["auto"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "ma_doi_auto_claims_faq",
        "name": "Massachusetts DOI Auto Insurance Claims FAQ",
        "url": "https://www.mass.gov/info-details/frequently-asked-questions-about-auto-insurance-claims",
        "source_file": "ma_doi_auto_claims_faq.html",
        "authority": "Massachusetts Division of Insurance",
        "insurance_domain": ["auto"],
        "content_type": "claims_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "ma_doi_home_insurance",
        "name": "Massachusetts DOI Understanding Home Insurance",
        "url": "https://www.mass.gov/info-details/understanding-home-insurance",
        "source_file": "ma_doi_home_insurance.html",
        "authority": "Massachusetts Division of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "ma_doi_homeowners_faq",
        "name": "Massachusetts DOI Homeowners Insurance FAQ",
        "url": "https://www.mass.gov/info-details/frequently-asked-questions-about-homeowners-insurance",
        "source_file": "ma_doi_homeowners_faq.html",
        "authority": "Massachusetts Division of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "ma_doi_flood_damage",
        "name": "Massachusetts DOI Is My Flood Damage Covered",
        "url": "https://www.mass.gov/info-details/is-my-flood-damage-covered",
        "source_file": "ma_doi_flood_damage.html",
        "authority": "Massachusetts Division of Insurance",
        "insurance_domain": ["homeowners", "flood"],
        "content_type": "coverage_limitation_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "mn_doi_auto_basics",
        "name": "Minnesota Commerce Auto Insurance Basics",
        "url": "https://mn.gov/commerce/insurance/auto/basics/index.jsp",
        "source_file": "mn_doi_auto_basics.html",
        "authority": "Minnesota Department of Commerce",
        "insurance_domain": ["auto"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "mn_doi_auto_shopping",
        "name": "Minnesota Commerce Shopping for Auto Insurance",
        "url": "https://mn.gov/commerce/insurance/auto/shopping/index.jsp",
        "source_file": "mn_doi_auto_shopping.html",
        "authority": "Minnesota Department of Commerce",
        "insurance_domain": ["auto"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "mn_doi_homeowner_guide",
        "name": "Minnesota Commerce Homeowner Insurance",
        "url": "https://mn.gov/commerce/insurance/home/index.jsp",
        "source_file": "mn_doi_homeowner_guide.html",
        "authority": "Minnesota Department of Commerce",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "mn_doi_property_coverage",
        "name": "Minnesota Commerce Property Coverage",
        "url": "https://mn.gov/commerce/insurance/home/old/insurance-basics/property-coverage.jsp",
        "source_file": "mn_doi_property_coverage.html",
        "authority": "Minnesota Department of Commerce",
        "insurance_domain": ["homeowners"],
        "content_type": "coverage_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "mn_doi_home_claims",
        "name": "Minnesota Commerce Home Insurance Claims Process",
        "url": "https://mn.gov/commerce/insurance/home/old/insurance-basics/claims-process.jsp",
        "source_file": "mn_doi_home_claims.html",
        "authority": "Minnesota Department of Commerce",
        "insurance_domain": ["homeowners"],
        "content_type": "claims_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "sc_doi_policy_basics",
        "name": "South Carolina DOI Understanding Your Insurance Policy",
        "url": "https://doi.sc.gov/957/Understanding-Your-Insurance-Policy",
        "source_file": "sc_doi_policy_basics.html",
        "authority": "South Carolina Department of Insurance",
        "insurance_domain": ["insurance"],
        "content_type": "policy_form_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "md_doi_home_declarations_page",
        "name": "Maryland DOI Understanding Homeowners Insurance Declarations Page",
        "url": "https://insurance.maryland.gov/Consumer/Pages/Home-Understanding-Declarations.aspx",
        "source_file": "md_doi_home_declarations_page.html",
        "authority": "Maryland Insurance Administration",
        "insurance_domain": ["homeowners"],
        "content_type": "declarations_page_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "md_doi_auto_declarations_page",
        "name": "Maryland DOI Understanding Auto Insurance Declarations Page",
        "url": "https://insurance.maryland.gov/Consumer/Pages/Auto-Understanding-Declarations.aspx",
        "source_file": "md_doi_auto_declarations_page.html",
        "authority": "Maryland Insurance Administration",
        "insurance_domain": ["auto"],
        "content_type": "declarations_page_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "sc_doi_home_coverage_types",
        "name": "South Carolina DOI Types of Homeowners Coverage",
        "url": "https://doi.sc.gov/615/Types-of-Coverage-in-Homeowners-Insuranc",
        "source_file": "sc_doi_home_coverage_types.html",
        "authority": "South Carolina Department of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "coverage_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "sc_doi_home_purchase",
        "name": "South Carolina DOI Purchasing Home Insurance Knowledge",
        "url": "https://doi.sc.gov/618/Purchasing-Home-Insurance-Knowledge",
        "source_file": "sc_doi_home_purchase.html",
        "authority": "South Carolina Department of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "sc_doi_auto_basics",
        "name": "South Carolina DOI Automobile Insurance",
        "url": "https://doi.sc.gov/auto",
        "source_file": "sc_doi_auto_basics.html",
        "authority": "South Carolina Department of Insurance",
        "insurance_domain": ["auto"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "sc_doi_deductible",
        "name": "South Carolina DOI Understanding Your Deductible",
        "url": "https://www.doi.sc.gov/1019/Understanding-Your-Deductible",
        "source_file": "sc_doi_deductible.html",
        "authority": "South Carolina Department of Insurance",
        "insurance_domain": ["insurance"],
        "content_type": "deductible_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "de_doi_homeowners_guide_pdf",
        "name": "Delaware DOI Homeowners Guide PDF",
        "url": "https://insurance.delaware.gov/wp-content/uploads/sites/15/2022/09/Homeowners-Guide.pdf",
        "source_file": "de_doi_homeowners_guide_pdf.html",
        "authority": "Delaware Department of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "de_doi_flood",
        "name": "Delaware DOI Flood Insurance",
        "url": "https://insurance.delaware.gov/divisions/consumerhp/flood/",
        "source_file": "de_doi_flood.html",
        "authority": "Delaware Department of Insurance",
        "insurance_domain": ["homeowners", "flood"],
        "content_type": "coverage_limitation_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "fl_doi_homeowners_overview",
        "name": "Florida CFO Homeowners Insurance Overview",
        "url": "https://www.myfloridacfo.com/division/consumers/understanding-insurance/homeownersinsuranceoverview",
        "source_file": "fl_doi_homeowners_overview.html",
        "authority": "Florida Department of Financial Services",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "fl_doi_home_endorsements",
        "name": "Florida CFO Homeowners Policy Endorsements",
        "url": "https://www.myfloridacfo.com/division/ica/fullcoverage/homeowners/policyendorsements",
        "source_file": "fl_doi_home_endorsements.html",
        "authority": "Florida Department of Financial Services",
        "insurance_domain": ["homeowners"],
        "content_type": "policy_form_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "fl_doi_auto_overview",
        "name": "Florida CFO Automobile Insurance",
        "url": "https://myfloridacfo.com/division/ica/fullcoverage/auto",
        "source_file": "fl_doi_auto_overview.html",
        "authority": "Florida Department of Financial Services",
        "insurance_domain": ["auto"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "ne_doi_home_policy_forms",
        "name": "Nebraska DOI Policy Forms: What You Need to Know",
        "url": "https://doi.nebraska.gov/policy-forms-what-you-need-know",
        "source_file": "ne_doi_home_policy_forms.html",
        "authority": "Nebraska Department of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "policy_form_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "ne_doi_home_coverages_limits",
        "name": "Nebraska DOI Types of Coverages and Their Limits",
        "url": "https://doi.nebraska.gov/types-coverages-and-their-limits",
        "source_file": "ne_doi_home_coverages_limits.html",
        "authority": "Nebraska Department of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "coverage_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "ne_doi_home_terms",
        "name": "Nebraska DOI Common Homeowners Insurance Terms",
        "url": "https://doi.nebraska.gov/common-homeowners-insurance-terms",
        "source_file": "ne_doi_home_terms.html",
        "authority": "Nebraska Department of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "glossary",
        "sft_eligible": True,
    },
    {
        "source_id": "ne_doi_auto_coverages",
        "name": "Nebraska DOI Importance of Auto Insurance and Types of Coverage",
        "url": "https://doi.nebraska.gov/important-auto-insurance-and-types-coverage",
        "source_file": "ne_doi_auto_coverages.html",
        "authority": "Nebraska Department of Insurance",
        "insurance_domain": ["auto"],
        "content_type": "coverage_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "ne_doi_auto_shopping",
        "name": "Nebraska DOI Shopping for Auto Insurance",
        "url": "https://doi.nebraska.gov/shopping-auto-insurance-where-go",
        "source_file": "ne_doi_auto_shopping.html",
        "authority": "Nebraska Department of Insurance",
        "insurance_domain": ["auto"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "wi_oci_homeowners_guide",
        "name": "Wisconsin OCI Consumer Guide to Homeowners Insurance",
        "url": "https://oci.wi.gov/Pages/Consumers/PI-015.aspx",
        "source_file": "wi_oci_homeowners_guide.html",
        "authority": "Wisconsin Office of the Commissioner of Insurance",
        "insurance_domain": ["homeowners", "renters"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
    {
        "source_id": "wi_oci_homeowners_faq",
        "name": "Wisconsin OCI Homeowners Insurance FAQ",
        "url": "https://oci.wi.gov/Pages/Consumers/PI-232.aspx",
        "source_file": "wi_oci_homeowners_faq.html",
        "authority": "Wisconsin Office of the Commissioner of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "coverage_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "wi_oci_home_savings",
        "name": "Wisconsin OCI Tips for Saving on Homeowners Insurance",
        "url": "https://oci.wi.gov/Pages/Consumers/PI-219.aspx",
        "source_file": "wi_oci_home_savings.html",
        "authority": "Wisconsin Office of the Commissioner of Insurance",
        "insurance_domain": ["homeowners"],
        "content_type": "settlement_explainer",
        "sft_eligible": True,
    },
    {
        "source_id": "pa_doi_homeowners_guide_pdf",
        "name": "Pennsylvania DOI Homeowners Insurance Guide PDF",
        "url": "https://www.insurance.pa.gov/Coverage/homeowners/Documents/Homeowners%20Insurance%20Guide.pdf",
        "source_file": "pa_doi_homeowners_guide_pdf.html",
        "authority": "Pennsylvania Insurance Department",
        "insurance_domain": ["homeowners"],
        "content_type": "consumer_guide",
        "sft_eligible": True,
    },
]

EXCLUDED_EXTERNAL_SOURCES = [
    {
        "source_id": "iii_ho3_sample_policy",
        "name": "Insurance Information Institute HO-3 Sample Policy PDF",
        "url": "https://www.iii.org/sites/default/files/docs/pdf/HO3_sample.pdf",
        "reason": "Potential ISO/copyright restrictions; register as a local-only candidate, do not redistribute in generated datasets.",
    }
]

FETCH_FAILURES: List[Dict[str, str]] = []

DOMAIN_BY_FILE = {
    "auto": ["auto", "motorist", "driver", "vehicle"],
    "homeowners": ["homeowner", "home", "dwelling", "declarations"],
    "disability": ["disability"],
    "travel": ["travel"],
    "health": ["health"],
    "flood": ["flood"],
    "business": ["business", "commercial"],
    "life": ["life"],
    "renters": ["renters"],
}

CONTENT_TYPES = [
    ("declarations_page", ["declarations page"]),
    ("disclosure_notice", ["disclosure notice", "required by maryland law"]),
    ("consumer_guide", ["consumer guide"]),
    ("claims_guide", ["claim", "claims"]),
    ("policy_form_explainer", ["coverage a", "coverage b", "ho-3", "ho 00"]),
    ("coverage_explainer", ["coverage", "covered"]),
]

INSURANCE_TERMS = [
    "policy",
    "coverage",
    "insured",
    "insurer",
    "premium",
    "deductible",
    "limit",
    "liability",
    "claim",
    "loss",
    "exclusion",
    "endorsement",
    "declarations",
    "benefit",
    "peril",
    "replacement cost",
    "actual cash value",
    "medical payments",
    "uninsured motorist",
    "underinsured motorist",
    "disability",
]

SFT_TOPICS = {
    "declarations page": "What is a declarations page and why is it important?",
    "deductible": "What does this page say about the deductible?",
    "coverage": "What coverage information is explained on this page?",
    "liability": "What does this page explain about liability coverage?",
    "claim": "What does this page say about insurance claims?",
    "premium": "What does this page explain about premiums?",
    "exclusion": "What does this page say about exclusions?",
    "endorsement": "What does this page say about endorsements?",
    "actual cash value": "What does actual cash value mean in this insurance context?",
    "replacement cost": "What does replacement cost mean in this insurance context?",
    "medical payments": "What does this page say about Medical Payments coverage?",
    "uninsured motorist": "What does this page explain about uninsured motorist coverage?",
    "underinsured motorist": "What does this page explain about underinsured motorist coverage?",
    "disability": "What does this page explain about disability income insurance?",
    "benefits": "What does this page say about insurance benefits?",
}

TOPIC_MARKERS = {
    "declarations page": [
        "declarations page",
        "declaration page",
        "declarations",
        "declaration",
        "policy declarations",
        "coverage summary",
    ],
    "deductible": ["deductible", "deductibles"],
    "coverage": ["coverage", "covered", "coverages", "insured against", "policy provides"],
    "liability": ["liability", "bodily injury", "property damage", "personal liability"],
    "claim": ["claim", "claims", "proof of loss", "settlement", "adjuster"],
    "premium": ["premium", "premiums", "rate", "rates"],
    "exclusion": ["exclusion", "exclusions", "not covered", "does not cover", "excluded", "limitations"],
    "endorsement": ["endorsement", "endorsements", "rider", "amendment", "add-on", "floater"],
    "actual cash value": ["actual cash value", "acv", "depreciation", "depreciated value"],
    "replacement cost": ["replacement cost", "replacement value", "replace", "rebuild"],
    "medical payments": ["medical payments", "med pay", "medical payment"],
    "uninsured motorist": ["uninsured motorist", "uninsured motorists"],
    "underinsured motorist": ["underinsured motorist", "underinsured motorists"],
    "disability": ["disability", "disabled", "disability income"],
    "benefits": ["benefit", "benefits"],
}

TARGET_TOPIC_MIN_COUNTS = {
    "exclusion": 120,
    "endorsement": 120,
    "actual cash value": 120,
    "replacement cost": 120,
    "declarations page": 120,
    "liability": 120,
}

QUESTION_VARIANTS = {
    "declarations page": [
        "What is a declarations page and why is it important?",
        "How should an employee explain a declarations page?",
        "What policy information is found on the declarations page?",
        "Why should the policyholder review the declarations page?",
        "What should the reader verify on the declarations page?",
        "How does the declarations page help summarize the policy?",
    ],
    "deductible": [
        "What does this evidence say about the deductible?",
        "How should the deductible be explained from this page?",
        "What happens when the insured chooses this deductible option?",
        "What deductible-related point should be noted from the evidence?",
    ],
    "coverage": [
        "What coverage information is explained by the evidence?",
        "How would you summarize the coverage point on this page?",
        "What does the evidence say is covered or addressed?",
        "What should an insurance employee note about coverage here?",
    ],
    "liability": [
        "What does the evidence explain about liability coverage?",
        "How should liability coverage be summarized from this page?",
        "What liability-related point is supported by the evidence?",
        "What does this page say about liability protection?",
    ],
    "claim": [
        "What does the evidence say about insurance claims?",
        "How should the claims guidance on this page be explained?",
        "What claim-related action or rule is supported here?",
        "What should a policyholder know about claims from this evidence?",
    ],
    "premium": [
        "What does the evidence explain about premiums?",
        "How should the premium-related point be summarized?",
        "What affects the premium according to this evidence?",
        "What should an employee explain about premium from this page?",
    ],
    "exclusion": [
        "What does the evidence say about exclusions?",
        "How should this exclusion-related point be explained?",
        "What limitation or exclusion is supported by this evidence?",
        "What should the reader understand about exclusions here?",
    ],
    "endorsement": [
        "What does the evidence say about endorsements?",
        "How should an endorsement be explained from this page?",
        "What policy change can an endorsement make?",
        "What endorsement-related point is supported by the evidence?",
    ],
    "actual cash value": [
        "What does actual cash value mean in this insurance context?",
        "How should actual cash value be explained using this evidence?",
        "What does the page say about actual cash value settlement?",
        "What should a policyholder understand about actual cash value?",
    ],
    "replacement cost": [
        "What does replacement cost mean in this insurance context?",
        "How should replacement cost coverage be explained from this evidence?",
        "What replacement-cost point is supported by this page?",
        "What should a policyholder understand about replacement cost?",
    ],
    "medical payments": [
        "What does the evidence say about Medical Payments coverage?",
        "How should Medical Payments coverage be explained?",
        "What Medical Payments coverage point is supported by this page?",
        "What should the reader know about Medical Payments coverage?",
    ],
    "uninsured motorist": [
        "What does the evidence explain about uninsured motorist coverage?",
        "How should uninsured motorist coverage be summarized?",
        "What uninsured-motorist point is supported by this evidence?",
        "What should the reader understand about uninsured motorist claims?",
    ],
    "underinsured motorist": [
        "What does the evidence explain about underinsured motorist coverage?",
        "How should underinsured motorist coverage be summarized?",
        "What underinsured-motorist point is supported by this evidence?",
        "What should the reader understand about underinsured motorist claims?",
    ],
    "disability": [
        "What does the evidence explain about disability income insurance?",
        "How should disability income insurance be summarized from this page?",
        "What disability benefit point is supported by this evidence?",
        "What should the reader understand about disability coverage?",
    ],
    "benefits": [
        "What does the evidence say about insurance benefits?",
        "How should the benefits described here be explained?",
        "What benefit-related point is supported by this evidence?",
        "What should the reader understand about these benefits?",
    ],
}

GENERIC_QUESTION_VARIANTS = [
    "Summarize the insurance-specific point supported by this evidence.",
    "What should an insurance employee explain from this evidence?",
    "What is the key policy-review takeaway from this evidence?",
    "Explain this evidence in plain insurance terms.",
]

UNSUPPORTED_QUESTIONS = [
    "Does this policy include cyber liability coverage?",
    "What is the earthquake deductible?",
    "Does this policy include scheduled jewelry coverage?",
    "What is the cyber liability sublimit?",
    "Does this policy include aviation liability coverage?",
    "What is the flood deductible listed in this homeowners policy?",
    "Does this page confirm umbrella liability coverage?",
    "What is the separate windstorm deductible?",
    "Does this page provide a terrorism coverage sublimit?",
    "Does this evidence confirm employment practices liability coverage?",
    "What is the pet injury coverage limit?",
    "Does the policy include identity theft expense reimbursement?",
    "What is the ordinance or law coverage percentage?",
    "Does this evidence confirm equipment breakdown coverage?",
    "What is the mold remediation sublimit?",
    "Does this page confirm personal cyber fraud coverage?",
    "What is the service line coverage deductible?",
    "Does this policy include inland marine coverage?",
    "What is the data breach response limit?",
]


def normalize_text(text: str) -> str:
    replacements = {
        "\x03": " ",
        "\x07": " ",
        "\x0e": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\u2026": "...",
        "\u221a": "-",
        "\ufffd": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "section"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            data = data.strip()
            if data:
                self.parts.append(data)

    def text(self) -> str:
        return normalize_text(" ".join(self.parts))


def html_to_text(html: str) -> str:
    parser = TextHTMLParser()
    parser.feed(html)
    return parser.text()


def pdf_bytes_to_text(content: bytes) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is required to parse PDF web sources.") from exc
    document = fitz.open(stream=content, filetype="pdf")
    return normalize_text(" ".join(page.get_text("text") for page in document))


def fetch_external_sources(cache_dir: Path, refresh: bool = False) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in EXTERNAL_WEB_SOURCES:
        cache_path = cache_dir / f"{source['source_id']}.txt"
        try:
            if cache_path.exists() and not refresh:
                text = cache_path.read_text(encoding="utf-8", errors="ignore")
            else:
                response = requests.get(
                    source["url"],
                    timeout=30,
                    headers={"User-Agent": "InsureRAG-VLM data curation/0.1"},
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" in content_type or source["url"].lower().endswith(".pdf"):
                    text = pdf_bytes_to_text(response.content)
                else:
                    text = html_to_text(response.text)
                cache_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            FETCH_FAILURES.append(
                {
                    "source_id": source["source_id"],
                    "url": source["url"],
                    "error": str(exc),
                }
            )
            continue
        text = normalize_text(text)
        if word_count(text) < 120 or term_count(text) < 2:
            FETCH_FAILURES.append(
                {
                    "source_id": source["source_id"],
                    "url": source["url"],
                    "error": "Skipped after parsing because it did not meet minimum word/insurance-term thresholds.",
                }
            )
            continue
        item = dict(source)
        item["text"] = text
        item["cache_path"] = str(cache_path)
        records.append(item)
    return records


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z\-']+|\$\d[\d,]*(?:\.\d+)?%?", text))


def sentence_split(text: str) -> List[str]:
    text = normalize_text(text)
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if 45 <= len(piece.strip()) <= 450]


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_snippets(text: str, max_words: int = 120) -> List[Tuple[int, str]]:
    """Build paragraph-like evidence snippets while keeping sentence boundaries."""
    sentences = sentence_split(text)
    snippets: List[Tuple[int, str]] = []
    current: List[str] = []
    chunk_id = 0
    for sentence in sentences:
        candidate = compact_whitespace(" ".join(current + [sentence]))
        if current and word_count(candidate) > max_words:
            snippet = compact_whitespace(" ".join(current))
            if word_count(snippet) >= 35:
                chunk_id += 1
                snippets.append((chunk_id, snippet))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        snippet = compact_whitespace(" ".join(current))
        if word_count(snippet) >= 35:
            chunk_id += 1
            snippets.append((chunk_id, snippet))
    return snippets


def detect_domains(file_name: str, text: str) -> List[str]:
    haystack = file_name.lower()
    domains = [domain for domain, markers in DOMAIN_BY_FILE.items() if any(marker in haystack for marker in markers)]
    return sorted(set(domains)) or ["insurance"]


def detect_content_type(text: str) -> str:
    lowered = text.lower()
    for content_type, markers in CONTENT_TYPES:
        if any(marker in lowered for marker in markers):
            return content_type
    return "insurance_reference"


def term_count(text: str) -> int:
    lowered = text.lower()
    return sum(lowered.count(term) for term in INSURANCE_TERMS)


def matched_topics(text: str) -> List[str]:
    lowered = text.lower()
    topics = []
    for topic in SFT_TOPICS:
        markers = TOPIC_MARKERS.get(topic, [topic])
        if any(marker in lowered for marker in markers):
            topics.append(topic)
    return topics


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def build_rag_pages(data_dir: Path, external_sources: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        pages = extract_text_by_page(pdf_path)
        for page_number, page_text in enumerate(pages, start=1):
            text = normalize_text(page_text)
            words = word_count(text)
            terms = term_count(text)
            if words < 120 or terms < 2:
                continue
            records.append(
                {
                    "record_id": f"rag_page::{pdf_path.stem}::p{page_number:04d}",
                    "record_type": "page",
                    "doc_id": pdf_path.stem,
                    "source_file": pdf_path.name,
                    "page": page_number,
                    "content_type": detect_content_type(text),
                    "insurance_domain": detect_domains(pdf_path.name, text),
                    "text": text,
                    "citation": f"{pdf_path.name}#page={page_number}",
                    "quality": {
                        "word_count": words,
                        "insurance_term_hits": terms,
                        "usable_for": ["retrieval", "grounded_answering"],
                    },
                }
            )
    for source in external_sources or []:
        text = normalize_text(source["text"])
        words = word_count(text)
        terms = term_count(text)
        if words < 120 or terms < 2:
            continue
        records.append(
            {
                "record_id": f"rag_page::{source['source_id']}::p0001",
                "record_type": "page",
                "doc_id": source["source_id"],
                "source_file": source["source_file"],
                "source_url": source["url"],
                "authority": source["authority"],
                "page": 1,
                "content_type": source["content_type"],
                "insurance_domain": source["insurance_domain"],
                "text": text,
                "citation": source["url"],
                "quality": {
                    "word_count": words,
                    "insurance_term_hits": terms,
                    "usable_for": ["retrieval", "grounded_answering"],
                    "source_kind": "official_web",
                },
            }
        )
    return records


def build_rag_snippets(page_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for page in page_records:
        for chunk_id, snippet in split_into_snippets(page["text"]):
            words = word_count(snippet)
            terms = term_count(snippet)
            if words < 35 or terms < 2:
                continue
            records.append(
                {
                    "record_id": f"rag_snippet::{page['doc_id']}::p{page['page']:04d}::c{chunk_id:03d}",
                    "record_type": "snippet",
                    "parent_page_id": page["record_id"],
                    "doc_id": page["doc_id"],
                    "source_file": page["source_file"],
                    **({"source_url": page["source_url"]} if page.get("source_url") else {}),
                    **({"authority": page["authority"]} if page.get("authority") else {}),
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "content_type": page["content_type"],
                    "insurance_domain": page["insurance_domain"],
                    "text": snippet,
                    "citation": page["citation"],
                    "quality": {
                        "word_count": words,
                        "insurance_term_hits": terms,
                        "usable_for": ["retrieval", "snippet_selection", "grounded_answering"],
                    },
                }
            )
    return records


def build_rag_corpus(
    data_dir: Path,
    external_sources: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pages = build_rag_pages(data_dir, external_sources=external_sources)
    snippets = build_rag_snippets(pages)
    return pages, snippets, pages + snippets


def best_evidence_for_topic(sentences: List[str], topic: str) -> Optional[str]:
    lowered_topic = topic.lower()
    candidates = [sentence for sentence in sentences if lowered_topic in sentence.lower()]
    if not candidates and topic == "benefits":
        candidates = [sentence for sentence in sentences if "benefit" in sentence.lower()]
    if not candidates:
        return None
    bad_markers = ["www.", "800-", "contact us", "phone", "suite", "copyright"]
    filtered = [s for s in candidates if not any(marker in s.lower() for marker in bad_markers)]
    candidates = filtered or candidates
    return max(candidates, key=lambda value: (term_count(value), len(value)))


def concise_support(evidence: str, max_words: int = 42) -> str:
    sentences = sentence_split(evidence)
    if not sentences:
        words = evidence.split()
        return " ".join(words[:max_words]).rstrip(" ,;:") + ("..." if len(words) > max_words else "")
    ranked = sorted(sentences, key=lambda sentence: (term_count(sentence), word_count(sentence)), reverse=True)
    support = ranked[0]
    words = support.split()
    if len(words) > max_words:
        support = " ".join(words[:max_words]).rstrip(" ,;:") + "..."
    return support


def answer_for_topic(topic: str, evidence: str, source_file: str, page: int) -> str:
    term = topic.title() if topic not in {"actual cash value", "replacement cost"} else topic
    support = concise_support(evidence)
    if topic in {"actual cash value", "replacement cost"}:
        body = (
            f"**{term}** is a policy settlement concept that should be interpreted from the cited evidence, "
            f"not assumed from general insurance knowledge. The supported point is: {support}"
        )
    else:
        body = (
            f"The cited page supports a **{term}** explanation. In practical terms, the reader should rely on "
            f"the stated policy language: {support}"
        )
    return f"{body} Source: {source_file}, Page {page}"


def answer_for_generic(question: str, evidence: str, source_file: str, page: int) -> str:
    support = concise_support(evidence)
    if "plain insurance terms" in question.lower():
        body = f"In plain insurance terms, the evidence means the policy point should be explained as: {support}"
    elif "employee" in question.lower():
        body = (
            "An insurance employee should explain only the supported policy point and avoid adding coverage assumptions. "
            f"The supported point is: {support}"
        )
    elif "takeaway" in question.lower():
        body = f"The key policy-review takeaway is limited to what the evidence states: {support}"
    else:
        body = f"The insurance-specific point supported by the evidence is: {support}"
    return f"{body} Source: {source_file}, Page {page}"


def make_sft_record(
    record_id: str,
    question: str,
    evidence: str,
    answer: str,
    source_file: str,
    page: int,
    domains: List[str],
    task_type: str,
    topic: str,
    parent_record_id: str,
    generation_method: str,
    answerable: bool = True,
) -> Dict[str, Any]:
    return {
        "record_id": record_id,
        "instruction": (
            "Answer the insurance question using only the provided evidence. "
            "Explain insurance-specific terminology when useful and cite the source page. "
            "If the evidence is insufficient, say so."
        ),
        "question": question,
        "evidence": evidence,
        "answer": answer,
        "source": f"{source_file}#page={page}",
        "answerable": answerable,
        "task_type": task_type,
        "topic": topic,
        "insurance_domain": domains,
        "quality": {
            "source_set": "core_sft_docs",
            "generation_method": generation_method,
            "parent_record_id": parent_record_id,
            "evidence_word_count": word_count(evidence),
            "insurance_term_hits": term_count(evidence),
        },
    }


def unsupported_forbidden_terms(question: str) -> List[str]:
    lowered = question.lower()
    explicit = {
        "cyber liability": ["cyber"],
        "earthquake deductible": ["earthquake"],
        "scheduled jewelry": ["scheduled jewelry", "jewelry"],
        "cyber liability sublimit": ["cyber", "sublimit"],
        "aviation liability": ["aviation"],
        "flood deductible": ["flood"],
        "umbrella liability": ["umbrella"],
        "windstorm deductible": ["windstorm"],
        "terrorism coverage": ["terrorism"],
        "employment practices liability": ["employment practices"],
        "pet injury": ["pet injury", "pet"],
        "identity theft": ["identity theft"],
        "ordinance or law": ["ordinance"],
        "equipment breakdown": ["equipment breakdown"],
        "mold remediation": ["mold"],
        "personal cyber fraud": ["cyber", "fraud"],
        "service line": ["service line"],
        "inland marine": ["inland marine"],
        "data breach": ["data breach"],
    }
    for marker, terms in explicit.items():
        if marker in lowered:
            return terms
    return [token for token in re.findall(r"[a-z][a-z\-]+", lowered) if len(token) >= 7]


def evidence_supports_unsupported_question(question: str, evidence: str) -> bool:
    lowered_evidence = evidence.lower()
    return any(term in lowered_evidence for term in unsupported_forbidden_terms(question))


def build_sft_dataset(
    data_dir: Path,
    external_sources: Optional[List[Dict[str, Any]]] = None,
    target_answerable: int = 1000,
    unsupported_count: int = 120,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    used_questions = Counter()

    all_pages = build_rag_pages(data_dir, external_sources=external_sources)
    core_pages = [
        page for page in all_pages
        if page["source_file"] in CORE_SFT_DOCS and page["quality"]["insurance_term_hits"] >= 3
    ]
    web_sft_pages = [
        page for page in all_pages
        if page.get("source_url")
        and any(source["source_file"] == page["source_file"] and source.get("sft_eligible") for source in external_sources or [])
        and page["quality"]["insurance_term_hits"] >= 3
    ]
    sft_pages = core_pages + web_sft_pages
    core_snippets = [
        snippet for snippet in build_rag_snippets(sft_pages)
        if 35 <= snippet["quality"]["word_count"] <= 140 and snippet["quality"]["insurance_term_hits"] >= 2
    ]
    core_snippets.sort(
        key=lambda item: (
            -item["quality"]["insurance_term_hits"],
            -item["quality"]["word_count"],
            item["source_file"],
            item["page"],
            item["chunk_id"],
        )
    )

    deferred_variants: List[Tuple[Dict[str, Any], str, str, str]] = []
    topic_counts: Counter[str] = Counter()

    def add_answerable_record(snippet: Dict[str, Any], topic: str, question: str, task_type: str) -> bool:
        if len(records) >= target_answerable:
            return False
        evidence = snippet["text"]
        key = (snippet["record_id"], question)
        if used_questions[key]:
            return False
        used_questions[key] += 1
        citation_source = snippet.get("source_url") or snippet["source_file"]
        answer = (
            answer_for_generic(question, evidence, citation_source, snippet["page"])
            if topic == "general_policy_review"
            else answer_for_topic(topic, evidence, citation_source, snippet["page"])
        )
        records.append(
            make_sft_record(
                record_id=f"sft::{snippet['doc_id']}::p{snippet['page']:04d}::c{snippet['chunk_id']:03d}::{len(records):04d}",
                question=question,
                evidence=evidence,
                answer=answer,
                source_file=citation_source,
                page=snippet["page"],
                domains=snippet["insurance_domain"],
                task_type=task_type,
                topic=topic,
                parent_record_id=snippet["record_id"],
                generation_method="rule_based_high_precision_snippet_paraphrase",
            )
        )
        topic_counts[topic] += 1
        return True

    for target_topic, min_count in TARGET_TOPIC_MIN_COUNTS.items():
        target_snippets = [snippet for snippet in core_snippets if target_topic in matched_topics(snippet["text"])]
        for snippet in target_snippets:
            if topic_counts[target_topic] >= min_count or len(records) >= target_answerable:
                break
            for question in QUESTION_VARIANTS.get(target_topic, [SFT_TOPICS[target_topic]]):
                if topic_counts[target_topic] >= min_count or len(records) >= target_answerable:
                    break
                add_answerable_record(snippet, target_topic, question, "targeted_topic_grounded_qa")

    for snippet in core_snippets:
        evidence = snippet["text"]
        topics = matched_topics(evidence)
        if not topics:
            topics = ["coverage"]
        topic_variants: List[Tuple[str, str, str]] = []
        for topic in topics[:3]:
            for question in QUESTION_VARIANTS.get(topic, [SFT_TOPICS[topic]]):
                topic_variants.append((topic, question, "topic_grounded_qa"))
        generic_variants = [
            ("general_policy_review", question, "policy_review_explanation")
            for question in GENERIC_QUESTION_VARIANTS
        ]
        immediate_variants = topic_variants[:2] + [generic_variants[0]]
        remaining_variants = topic_variants[2:] + generic_variants[1:]

        for topic, question, task_type in immediate_variants:
            add_answerable_record(snippet, topic, question, task_type)
        for topic, question, task_type in remaining_variants:
            deferred_variants.append((snippet, topic, question, task_type))
        if len(records) >= target_answerable:
            break

    if len(records) < target_answerable:
        for snippet, topic, question, task_type in deferred_variants:
            if len(records) >= target_answerable:
                break
            evidence = snippet["text"]
            add_answerable_record(snippet, topic, question, task_type)

    unsupported_records: List[Dict[str, Any]] = []
    evidence_pool = [record for record in records if record["answerable"]]
    for idx in range(min(unsupported_count, len(evidence_pool))):
        question = UNSUPPORTED_QUESTIONS[idx % len(UNSUPPORTED_QUESTIONS)]
        if idx >= len(evidence_pool):
            break
        evidence_record = None
        for offset in range(len(evidence_pool)):
            candidate = evidence_pool[-(((idx + offset) % len(evidence_pool)) + 1)]
            if not evidence_supports_unsupported_question(question, candidate["evidence"]):
                evidence_record = candidate
                break
        if evidence_record is None:
            continue
        source_file, page_part = evidence_record["source"].split("#page=", 1)
        page = int(page_part)
        unsupported_records.append(
            make_sft_record(
                record_id=f"sft::unsupported::{idx:04d}",
                question=question,
                evidence=evidence_record["evidence"],
                answer=(
                    "The provided evidence does not support that answer. "
                    "I cannot confirm this coverage, deductible, or sublimit from the supplied page. "
                    "Source: insufficient evidence"
                ),
                source_file=source_file,
                page=page,
                domains=evidence_record["insurance_domain"],
                task_type="unsupported_question_abstention",
                topic="unsupported",
                parent_record_id=evidence_record["quality"]["parent_record_id"],
                generation_method="rule_based_unsupported_counterexample",
                answerable=False,
            )
        )
        unsupported_records[-1]["quality"]["label"] = "unsupported_by_evidence"

    return records + unsupported_records


def summarize(
    rag_pages: List[Dict[str, Any]],
    rag_snippets: List[Dict[str, Any]],
    rag_records: List[Dict[str, Any]],
    sft_records: List[Dict[str, Any]],
    output_dir: Path,
    external_sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rag_docs = sorted({record["source_file"] for record in rag_records})
    sft_docs = sorted({record["source"].split("#page=", 1)[0] for record in sft_records if record["answerable"]})
    return {
        "rag_pages": {
            "path": str(output_dir / "rag_pages.jsonl"),
            "records": len(rag_pages),
            "source_documents": len(sorted({record["source_file"] for record in rag_pages})),
        },
        "rag_snippets": {
            "path": str(output_dir / "rag_snippets.jsonl"),
            "records": len(rag_snippets),
            "source_documents": len(sorted({record["source_file"] for record in rag_snippets})),
        },
        "rag_corpus": {
            "path": str(output_dir / "rag_corpus.jsonl"),
            "records": len(rag_records),
            "page_records": len(rag_pages),
            "snippet_records": len(rag_snippets),
            "source_documents": len(rag_docs),
            "documents": rag_docs,
        },
        "sft_dataset": {
            "path": str(output_dir / "sft_dataset.jsonl"),
            "records": len(sft_records),
            "answerable_records": sum(1 for record in sft_records if record["answerable"]),
            "unsupported_records": sum(1 for record in sft_records if not record["answerable"]),
            "source_documents": len(sft_docs),
            "documents": sft_docs,
        },
        "external_sources": {
            "included": len(external_sources or []),
            "included_sources": [
                {
                    "source_id": source["source_id"],
                    "name": source["name"],
                    "url": source["url"],
                    "sft_eligible": source.get("sft_eligible", False),
                }
                for source in external_sources or []
            ],
            "excluded_sources": EXCLUDED_EXTERNAL_SOURCES,
            "fetch_failures": FETCH_FAILURES,
        },
        "quality_policy": {
            "rag": "Hybrid RAG includes page-level parent records plus sentence-boundary snippet records for second-stage evidence selection.",
            "sft": "Uses only core high-signal insurance PDFs, natural questions, explicit snippet evidence, page citations, and abstention counterexamples.",
            "excluded": "Pages with empty text extraction, weak insurance signal, or contact-only/low-information content are excluded.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate RAG and SFT JSONL datasets for InsureRAG-VLM.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/00_raw/external/public_docs"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/04_curated"))
    parser.add_argument("--sft-target-answerable", type=int, default=1000)
    parser.add_argument("--sft-unsupported-count", type=int, default=120)
    parser.add_argument("--web-cache-dir", type=Path, default=Path("data/04_curated/source_cache"))
    parser.add_argument("--skip-web-sources", action="store_true")
    parser.add_argument("--refresh-web-cache", action="store_true")
    args = parser.parse_args()

    external_sources = [] if args.skip_web_sources else fetch_external_sources(args.web_cache_dir, args.refresh_web_cache)
    rag_pages, rag_snippets, rag_records = build_rag_corpus(args.data_dir, external_sources=external_sources)
    sft_records = build_sft_dataset(
        args.data_dir,
        external_sources=external_sources,
        target_answerable=args.sft_target_answerable,
        unsupported_count=args.sft_unsupported_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rag_pages, args.output_dir / "rag_pages.jsonl")
    write_jsonl(rag_snippets, args.output_dir / "rag_snippets.jsonl")
    write_jsonl(rag_records, args.output_dir / "rag_corpus.jsonl")
    write_jsonl(sft_records, args.output_dir / "sft_dataset.jsonl")
    summary = summarize(rag_pages, rag_snippets, rag_records, sft_records, args.output_dir, external_sources)
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
