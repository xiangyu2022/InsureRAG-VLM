"""Insurance industry knowledge base for internal employee Q&A.

Covers common acronyms, terms, concepts, and role definitions used across
the insurance industry so new employees can get instant, clear explanations
without needing to search external resources.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class KnowledgeEntry:
    term: str
    short: str          # one-line definition shown in badges
    detail: str         # full explanation paragraph
    category: str = "concept"   # "acronym" | "concept" | "role" | "metric"
    see_also: list[str] = field(default_factory=list)


# ── Acronyms ─────────────────────────────────────────────────────────────────

ACRONYMS: dict[str, KnowledgeEntry] = {
    "PI": KnowledgeEntry(
        "PI", "Personal Insurance / Personal Lines",
        (
            "PI stands for Personal Insurance, also called Personal Lines. "
            "It covers individuals and families rather than businesses. "
            "Common PI products include: personal auto, homeowners (HO), renters, "
            "personal umbrella liability, boat, and personal jewelry/art floaters. "
            "PI is the counterpart to Commercial Lines (CI), which covers business entities. "
            "In a typical insurer, the PI and CI divisions operate with separate underwriting "
            "teams, pricing models, and distribution channels."
        ),
        category="acronym",
        see_also=["CI", "LOB", "HO", "P&C"],
    ),
    "CI": KnowledgeEntry(
        "CI", "Commercial Insurance / Commercial Lines",
        (
            "CI stands for Commercial Insurance or Commercial Lines. "
            "It covers businesses, organisations, and commercial entities. "
            "Key CI products include General Liability (GL), Commercial Property, "
            "Business Owners Policy (BOP), Workers' Compensation (WC), "
            "Professional Liability (E&O, D&O), and Cyber Liability. "
            "Commercial accounts are typically more complex and higher-premium than personal lines."
        ),
        category="acronym",
        see_also=["PI", "GL", "BOP", "WC", "E&O", "D&O"],
    ),
    "GL": KnowledgeEntry(
        "GL", "General Liability",
        (
            "GL stands for General Liability insurance. "
            "It protects businesses against claims of bodily injury (BI), property damage (PD), "
            "and personal/advertising injury arising from their operations, products, or premises. "
            "Also called CGL — Commercial General Liability. "
            "GL policies have both per-occurrence and aggregate limits. "
            "Almost every business carries GL as a baseline coverage."
        ),
        category="acronym",
        see_also=["CGL", "BI", "PD", "CI"],
    ),
    "CGL": KnowledgeEntry(
        "CGL", "Commercial General Liability",
        (
            "CGL stands for Commercial General Liability — the standard liability policy for most businesses. "
            "It covers bodily injury, property damage, personal injury, and advertising injury claims "
            "arising from the insured's operations, products, or completed work. "
            "CGL policies use ISO standard forms (e.g., CG 00 01) and typically include "
            "per-occurrence and annual aggregate limits."
        ),
        category="acronym",
        see_also=["GL", "ISO"],
    ),
    "E&O": KnowledgeEntry(
        "E&O", "Errors & Omissions (Professional Liability)",
        (
            "E&O stands for Errors & Omissions insurance, also known as Professional Liability. "
            "It protects professionals — consultants, insurance agents, IT firms, lawyers, accountants, etc. "
            "— against claims of negligence, errors, or failure to perform professional duties "
            "that result in financial loss to a client. "
            "E&O policies are typically written on a claims-made basis, "
            "meaning the claim must be filed during the active policy period."
        ),
        category="acronym",
        see_also=["D&O", "EPL", "claims-made"],
    ),
    "D&O": KnowledgeEntry(
        "D&O", "Directors & Officers Liability",
        (
            "D&O stands for Directors & Officers Liability insurance. "
            "It protects corporate directors and officers from personal financial losses "
            "if they are sued for alleged wrongful acts in managing a company — "
            "such as breach of fiduciary duty, misrepresentation, or mismanagement. "
            "D&O covers legal defence costs and settlements, but not intentional fraud or criminal acts. "
            "Publicly traded and non-profit boards routinely require D&O coverage."
        ),
        category="acronym",
        see_also=["E&O", "EPL"],
    ),
    "EPL": KnowledgeEntry(
        "EPL", "Employment Practices Liability",
        (
            "EPL stands for Employment Practices Liability insurance. "
            "It protects employers against claims of wrongful termination, discrimination "
            "(race, gender, age, disability), sexual harassment, failure to promote, "
            "retaliation, and other employment-related allegations. "
            "EPL is particularly important for companies with 15+ employees "
            "and is often bundled with D&O in a Management Liability package."
        ),
        category="acronym",
        see_also=["D&O"],
    ),
    "WC": KnowledgeEntry(
        "WC", "Workers' Compensation",
        (
            "WC stands for Workers' Compensation insurance. "
            "It provides wage replacement and medical benefits to employees injured on the job, "
            "in exchange for the employee giving up the right to sue the employer for negligence. "
            "WC is mandatory in almost every US state and most other jurisdictions worldwide. "
            "Premiums are based on payroll size and job classification (hazard level). "
            "Key benefits: medical treatment, temporary/permanent disability payments, and death benefits."
        ),
        category="acronym",
        see_also=["CI", "GL"],
    ),
    "BI": KnowledgeEntry(
        "BI", "Bodily Injury  —OR—  Business Interruption",
        (
            "BI is a dual-use acronym in insurance — context determines the meaning:\n\n"
            "1. Bodily Injury: Physical harm caused to a person. "
            "A standard coverage component in liability policies, written as 'BI/PD' limits "
            "(e.g., $100,000 per person / $300,000 per accident).\n\n"
            "2. Business Interruption: A property coverage that replaces lost income "
            "when a business cannot operate due to a covered physical loss (e.g., fire, storm). "
            "Also called 'Business Income' coverage. BI coverage typically has a waiting period "
            "and an indemnity period (maximum recovery period)."
        ),
        category="acronym",
        see_also=["PD", "GL", "CI"],
    ),
    "PD": KnowledgeEntry(
        "PD", "Property Damage",
        (
            "PD stands for Property Damage — physical damage to tangible property belonging to a third party. "
            "In liability policies, PD coverage pays when the insured accidentally damages someone else's property. "
            "Always paired with Bodily Injury in standard auto and GL liability limits, "
            "written as 'BI/PD limits' (e.g., $100K/$300K BI, $100K PD)."
        ),
        category="acronym",
        see_also=["BI", "GL"],
    ),
    "PIP": KnowledgeEntry(
        "PIP", "Personal Injury Protection",
        (
            "PIP stands for Personal Injury Protection, also called no-fault coverage. "
            "It pays medical expenses (and sometimes lost wages and funeral costs) for "
            "the policyholder and passengers after an auto accident, regardless of fault. "
            "Required in no-fault states such as Florida, Michigan, New York, and New Jersey. "
            "PIP eliminates the need to determine fault for basic medical claims."
        ),
        category="acronym",
        see_also=["UM", "UIM", "PI"],
    ),
    "UM": KnowledgeEntry(
        "UM", "Uninsured Motorist",
        (
            "UM stands for Uninsured Motorist coverage. "
            "It protects you if you are hit by a driver who carries no auto insurance. "
            "Covers medical bills, lost wages, and in some states property damage. "
            "Required or offered as mandatory in most US states. "
            "Works alongside UIM (Underinsured Motorist) coverage."
        ),
        category="acronym",
        see_also=["UIM", "PIP"],
    ),
    "UIM": KnowledgeEntry(
        "UIM", "Underinsured Motorist",
        (
            "UIM stands for Underinsured Motorist coverage. "
            "It pays the gap between your damages and an at-fault driver's insufficient liability limits. "
            "Example: your damages are $100K but the at-fault driver only has $25K in liability — "
            "UIM covers the remaining $75K (up to your UIM limit). "
            "Usually purchased alongside Uninsured Motorist (UM) coverage."
        ),
        category="acronym",
        see_also=["UM", "PIP"],
    ),
    "HO": KnowledgeEntry(
        "HO", "Homeowners Insurance",
        (
            "HO refers to Homeowners Insurance and its standard policy forms. "
            "The most common forms are:\n"
            "• HO-3 (Special Form): covers the dwelling on an open-perils basis, contents on named-perils basis — most common.\n"
            "• HO-5 (Comprehensive): open-perils for both dwelling and contents — broadest coverage.\n"
            "• HO-4: renters insurance (no dwelling coverage).\n"
            "• HO-6: condominium unit owners.\n"
            "HO policies cover the structure, personal property, liability, and additional living expenses (ALE)."
        ),
        category="acronym",
        see_also=["PI", "LOB"],
    ),
    "BOP": KnowledgeEntry(
        "BOP", "Business Owners Policy",
        (
            "BOP stands for Business Owners Policy. "
            "It bundles Commercial Property and General Liability (GL) into one cost-effective package "
            "designed for small-to-medium-sized businesses. "
            "BOPs are typically cheaper than purchasing coverages separately and "
            "may include business interruption (BI) coverage automatically. "
            "Not all businesses qualify — generally must be low-hazard and below revenue thresholds."
        ),
        category="acronym",
        see_also=["CI", "GL", "CPP"],
    ),
    "CPP": KnowledgeEntry(
        "CPP", "Commercial Package Policy",
        (
            "CPP stands for Commercial Package Policy. "
            "More flexible than a BOP, it allows combining multiple commercial coverage parts "
            "(commercial property, general liability, inland marine, crime, equipment breakdown, etc.) "
            "into one policy for mid-to-large businesses. "
            "Each coverage part can be tailored with different limits and deductibles."
        ),
        category="acronym",
        see_also=["BOP", "CI"],
    ),
    "UW": KnowledgeEntry(
        "UW", "Underwriting",
        (
            "UW stands for Underwriting — the core process by which an insurer evaluates risk, "
            "decides whether to accept it, and determines the appropriate premium, terms, and conditions. "
            "Underwriters analyse applications, loss history, inspection reports, financial statements, "
            "credit scores, and industry data. "
            "A 'UW' as a job title means Underwriter — the person making these decisions. "
            "'UW appetite' refers to the types of risk an insurer is willing to write."
        ),
        category="acronym",
        see_also=["LOB", "LR", "actuarial"],
    ),
    "LOB": KnowledgeEntry(
        "LOB", "Line of Business",
        (
            "LOB stands for Line of Business — a category of insurance product. "
            "Common LOBs include: Personal Auto, Homeowners, Commercial Property, "
            "General Liability, Workers' Compensation, Professional Liability, Cyber, Marine, etc. "
            "Each LOB has its own pricing actuaries, underwriters, claims teams, "
            "regulatory requirements, and loss development patterns. "
            "When someone asks 'which LOB does this fall under?' they're asking which product category applies."
        ),
        category="acronym",
        see_also=["PI", "CI", "UW"],
    ),
    "LR": KnowledgeEntry(
        "LR", "Loss Ratio",
        (
            "LR stands for Loss Ratio — a key underwriting profitability metric. "
            "Formula: (Incurred Losses + LAE) ÷ Earned Premium × 100%. "
            "A ratio below ~60-65% is generally considered profitable underwriting. "
            "Combined with the Expense Ratio (operating costs ÷ premium) to give the Combined Ratio (CR). "
            "Example: LR 62% + Expense Ratio 30% = CR 92% → profitable underwriting."
        ),
        category="metric",
        see_also=["CR", "LAE", "GWP"],
    ),
    "CR": KnowledgeEntry(
        "CR", "Combined Ratio",
        (
            "CR stands for Combined Ratio = Loss Ratio + Expense Ratio. "
            "It measures overall underwriting profitability:\n"
            "• CR < 100%: underwriting profit.\n"
            "• CR > 100%: underwriting loss (offset by investment income).\n"
            "• CR = 100%: breakeven on underwriting.\n"
            "The industry benchmark varies by line, but most insurers target a CR of 90-98%. "
            "Investment income allows some insurers to sustain CRs slightly above 100%."
        ),
        category="metric",
        see_also=["LR"],
    ),
    "GWP": KnowledgeEntry(
        "GWP", "Gross Written Premium",
        (
            "GWP stands for Gross Written Premium — the total premium an insurer contractually "
            "commits to collect from policyholders during a period, before ceding any to reinsurers. "
            "GWP is the top-line revenue figure for insurance companies and is often used as "
            "a measure of market size or company growth. "
            "GWP minus reinsurance ceded = Net Written Premium (NWP)."
        ),
        category="metric",
        see_also=["NWP", "NEP", "reinsurance"],
    ),
    "NWP": KnowledgeEntry(
        "NWP", "Net Written Premium",
        (
            "NWP stands for Net Written Premium = Gross Written Premium (GWP) minus ceded reinsurance premium. "
            "It represents the premium an insurer retains after transferring risk to reinsurers. "
            "NWP is a better measure of the insurer's actual retained risk than GWP."
        ),
        category="metric",
        see_also=["GWP", "NEP"],
    ),
    "NEP": KnowledgeEntry(
        "NEP", "Net Earned Premium",
        (
            "NEP stands for Net Earned Premium — the portion of NWP that has been 'earned' "
            "as the policy coverage period elapses. "
            "Example: a $1,200 annual premium earns at $100/month, so after 3 months NEP = $300. "
            "NEP is the denominator in the Loss Ratio calculation."
        ),
        category="metric",
        see_also=["NWP", "LR"],
    ),
    "LAE": KnowledgeEntry(
        "LAE", "Loss Adjustment Expenses",
        (
            "LAE stands for Loss Adjustment Expenses — costs incurred to investigate and settle claims. "
            "Two components:\n"
            "• ALAE (Allocated LAE): costs tied to a specific claim — defence attorneys, independent adjusters, expert witnesses.\n"
            "• ULAE (Unallocated LAE): general claims department overhead — salaries, rent, IT systems.\n"
            "LAE is added to losses in the Loss Ratio calculation."
        ),
        category="metric",
        see_also=["LR", "IBNR"],
    ),
    "IBNR": KnowledgeEntry(
        "IBNR", "Incurred But Not Reported",
        (
            "IBNR stands for Incurred But Not Reported — a reserve for claims that have occurred "
            "but have not yet been filed with the insurer. "
            "Actuaries estimate IBNR using historical loss development triangles and tail factors. "
            "IBNR is a critical liability on an insurer's balance sheet, especially in "
            "long-tail lines like WC, GL, and professional liability where claims can emerge years later. "
            "Getting IBNR wrong can lead to under-reserving and financial instability."
        ),
        category="metric",
        see_also=["LAE", "actuarial"],
    ),
    "PML": KnowledgeEntry(
        "PML", "Probable Maximum Loss",
        (
            "PML stands for Probable Maximum Loss — the estimated worst-case loss from a single event "
            "under foreseeable (non-catastrophic) circumstances. "
            "Used in property underwriting and reinsurance to set line sizes and attachment points. "
            "Example: PML on a factory considers that fire suppression systems and fire walls "
            "will limit the spread, even in a worst-case fire. "
            "PML is distinct from EML (Estimated Maximum Loss), though the terms are sometimes used interchangeably."
        ),
        category="acronym",
        see_also=["EML", "CAT", "TIV"],
    ),
    "EML": KnowledgeEntry(
        "EML", "Estimated Maximum Loss",
        (
            "EML stands for Estimated Maximum Loss — the maximum plausible single-event loss "
            "assuming a worst-case realistic scenario (but not an impossible catastrophe). "
            "Lloyd's of London uses EML as a key metric for line-setting and capacity decisions. "
            "EML tends to be more conservative than PML in some market conventions."
        ),
        category="acronym",
        see_also=["PML", "TIV"],
    ),
    "CAT": KnowledgeEntry(
        "CAT", "Catastrophe",
        (
            "CAT stands for Catastrophe in the insurance context — a large-scale event "
            "(hurricane, earthquake, flood, wildfire, tornado outbreak, etc.) that triggers "
            "widespread losses across many policyholders simultaneously. "
            "Industry bodies (ISO PCS in the US) declare CAT events when industry-wide losses "
            "exceed a threshold (currently $25M in the US). "
            "CAT modelling firms (RMS, AIR, Verisk) estimate probable losses by simulating "
            "thousands of event scenarios — critical for pricing CAT reinsurance and managing accumulation."
        ),
        category="acronym",
        see_also=["PML", "AAL", "EQ"],
    ),
    "EQ": KnowledgeEntry(
        "EQ", "Earthquake",
        (
            "EQ stands for Earthquake in insurance underwriting and catastrophe modelling. "
            "Earthquake coverage is almost always excluded from standard property policies "
            "and must be purchased as a separate policy or endorsement. "
            "Key EQ underwriting factors: seismic zone, soil type, building construction, age. "
            "EQ deductibles are typically percentage-based (e.g., 10-15% of insured value) "
            "rather than a flat dollar amount."
        ),
        category="acronym",
        see_also=["CAT", "PML"],
    ),
    "TIV": KnowledgeEntry(
        "TIV", "Total Insured Value",
        (
            "TIV stands for Total Insured Value — the sum of all insured asset values in a policy "
            "or across a portfolio. "
            "For a commercial property policy covering 10 buildings worth $5M each, TIV = $50M. "
            "TIV is used to calculate premium, set limits, and measure catastrophe accumulation. "
            "Insurers track TIV by geography to avoid concentrating too much exposure in one area."
        ),
        category="acronym",
        see_also=["PML", "SOV"],
    ),
    "SOV": KnowledgeEntry(
        "SOV", "Schedule of Values",
        (
            "SOV stands for Schedule of Values — a detailed list submitted by the insured "
            "showing all locations and assets to be insured, with their individual replacement values. "
            "A SOV for a national retail chain might list 200 store locations with building and "
            "contents values for each. Underwriters use SOVs to price the risk and identify "
            "accumulation hotspots (e.g., many locations in a Florida CAT zone)."
        ),
        category="acronym",
        see_also=["TIV"],
    ),
    "MGA": KnowledgeEntry(
        "MGA", "Managing General Agent",
        (
            "MGA stands for Managing General Agent — a specialised insurance intermediary "
            "granted binding authority by an insurer (the 'capacity provider') to underwrite, "
            "price, and bind coverage on their behalf. "
            "MGAs often focus on niche markets (e.g., cyber, cannabis, marine, specialty construction) "
            "where they have deeper expertise than the capacity provider. "
            "MGAs are increasingly important in the specialty and E&S (excess & surplus lines) market."
        ),
        category="acronym",
        see_also=["TPA", "UW"],
    ),
    "TPA": KnowledgeEntry(
        "TPA", "Third Party Administrator",
        (
            "TPA stands for Third Party Administrator — a company that handles insurance claims "
            "and administrative tasks on behalf of insurers or self-insured employers. "
            "TPAs are used to: reduce costs vs. in-house claims staff, access specialist expertise "
            "(e.g., complex WC or medical claims), and provide scalable capacity. "
            "Self-insured companies often use TPAs to manage their claims while retaining the risk."
        ),
        category="acronym",
        see_also=["MGA"],
    ),
    "NCD": KnowledgeEntry(
        "NCD", "No Claims Discount",
        (
            "NCD (also called NCB — No Claims Bonus) is a discount applied to renewal premiums "
            "for policyholders who have not made any claims during preceding policy years. "
            "Common in personal auto and home insurance. "
            "Each claim-free year earns a higher discount level (e.g., 0%, 20%, 30%, 40%, 50%). "
            "Policyholders often purchase 'NCD Protection' riders to preserve their discount "
            "even after one claim."
        ),
        category="acronym",
        see_also=["NCB", "PI"],
    ),
    "NCB": KnowledgeEntry(
        "NCB", "No Claims Bonus",
        (
            "NCB is synonymous with NCD (No Claims Discount) — a premium discount rewarding "
            "claim-free driving or homeownership history. "
            "The terminology differs by market: 'NCB' is common in the UK and Asia, "
            "'NCD' is used in the UK, Australia, and some Asian markets."
        ),
        category="acronym",
        see_also=["NCD"],
    ),
    "P&C": KnowledgeEntry(
        "P&C", "Property and Casualty Insurance",
        (
            "P&C stands for Property and Casualty — the broad insurance segment covering:\n"
            "• Property: buildings, contents, vehicles, marine cargo (first-party losses).\n"
            "• Casualty/Liability: legal liability to third parties for bodily injury or property damage.\n"
            "P&C is distinct from Life & Health (L&H) insurance. "
            "Outside North America, P&C is often called 'General Insurance' (GI). "
            "Major P&C insurers include Berkshire Hathaway, Travelers, Chubb, AIG, and Zurich."
        ),
        category="acronym",
        see_also=["CI", "PI", "L&H"],
    ),
    "L&H": KnowledgeEntry(
        "L&H", "Life and Health Insurance",
        (
            "L&H stands for Life and Health insurance — the segment covering:\n"
            "• Life insurance: pays a death benefit to beneficiaries.\n"
            "• Health/Medical insurance: pays medical expenses.\n"
            "• Disability Income: replaces income if the insured cannot work.\n"
            "• Annuities: provide income streams, often for retirement.\n"
            "L&H is actuarially and regulatorily distinct from P&C insurance."
        ),
        category="acronym",
        see_also=["P&C"],
    ),
    "ISO": KnowledgeEntry(
        "ISO", "Insurance Services Office (now Verisk)",
        (
            "ISO stands for Insurance Services Office (now part of Verisk Analytics). "
            "ISO develops standardised policy forms, rating manuals, and statistical reporting "
            "programmes used across the US P&C insurance industry. "
            "When underwriters refer to 'ISO forms' they mean standard policy language "
            "(e.g., CG 00 01 for CGL, CP 00 10 for Commercial Property). "
            "ISO also collects industry loss statistics used in actuarial pricing."
        ),
        category="acronym",
        see_also=["ACORD"],
    ),
    "ACORD": KnowledgeEntry(
        "ACORD", "Association for Cooperative Operations Research and Development",
        (
            "ACORD is the global insurance standards body. It develops:\n"
            "• Standardised data exchange forms used by agents, brokers, and insurers "
            "(e.g., ACORD 125 Commercial Insurance Application, ACORD 35 for WC).\n"
            "• XML and API data standards for electronic policy transactions.\n"
            "When someone says 'fill out the ACORD form' they mean the standard application form."
        ),
        category="acronym",
        see_also=["ISO"],
    ),
    "AAL": KnowledgeEntry(
        "AAL", "Average Annual Loss",
        (
            "AAL stands for Average Annual Loss — the long-run expected average annual loss "
            "from catastrophe events, as calculated by a CAT model. "
            "It averages losses across all simulated event scenarios weighted by probability. "
            "AAL is the actuarially expected annual cost of catastrophe exposure "
            "and is the primary metric used in pricing CAT reinsurance and setting CAT loadings in premiums."
        ),
        category="metric",
        see_also=["CAT", "PML"],
    ),
    "E&S": KnowledgeEntry(
        "E&S", "Excess and Surplus Lines",
        (
            "E&S stands for Excess and Surplus Lines — the market for insuring risks that "
            "admitted (standard) insurers are unwilling or unable to cover. "
            "E&S carriers (e.g., Lloyd's syndicates, non-admitted US carriers) have more "
            "pricing and form flexibility. They are not bound by state-filed rates and forms. "
            "Examples of E&S risks: cannabis businesses, new tech companies, habitational properties "
            "with poor loss history, and large unique risks. "
            "Risks must typically be 'diligently sought' in the admitted market before going E&S."
        ),
        category="acronym",
        see_also=["MGA", "CI"],
    ),
    "ROE": KnowledgeEntry(
        "ROE", "Return on Equity",
        (
            "ROE = Net Income ÷ Shareholders' Equity × 100%. "
            "A standard financial performance metric. "
            "For insurance companies, ROE combines underwriting profit (measured by CR) "
            "and investment income. "
            "Industry target ROE varies but is typically 10-15% for P&C insurers."
        ),
        category="metric",
        see_also=["CR", "LR"],
    ),
}

# ── General Concepts ─────────────────────────────────────────────────────────

TERMS: dict[str, KnowledgeEntry] = {
    "deductible": KnowledgeEntry(
        "Deductible", "The amount the insured pays before insurance pays",
        (
            "A deductible is the portion of a covered loss the policyholder must pay "
            "before the insurance company contributes. "
            "Example: with a $500 deductible on a $2,000 claim, the insured pays $500 "
            "and the insurer pays $1,500. "
            "Higher deductibles = lower premiums (you retain more risk). "
            "Types: flat-dollar deductibles (most common), percentage deductibles "
            "(common in earthquake and hurricane coverage — e.g., 5% of TIV)."
        ),
        category="concept",
        see_also=["premium", "coverage limit"],
    ),
    "premium": KnowledgeEntry(
        "Premium", "The price paid for insurance coverage",
        (
            "The premium is the amount paid by the policyholder — monthly, quarterly, "
            "or annually — in exchange for the insurer's promise to pay covered losses. "
            "Premiums are calculated based on risk factors: "
            "age, location, claims history, coverage amount, deductible, credit score, "
            "building construction, occupancy type, etc. "
            "Gross Written Premium (GWP) is the industry aggregate measure of premiums."
        ),
        category="concept",
        see_also=["deductible", "GWP", "UW"],
    ),
    "endorsement": KnowledgeEntry(
        "Endorsement", "A written amendment that modifies an insurance policy",
        (
            "An endorsement (also called a rider) is a document attached to a policy "
            "that adds, removes, or changes coverage terms. "
            "Common examples:\n"
            "• Adding earthquake coverage to a property policy.\n"
            "• Removing an excluded driver from an auto policy.\n"
            "• Adding a certificate holder or additional insured.\n"
            "• Raising a coverage limit mid-term.\n"
            "Endorsements are legally part of the policy and take precedence over standard form language."
        ),
        category="concept",
        see_also=["exclusion", "rider"],
    ),
    "rider": KnowledgeEntry(
        "Rider", "Policy modification — synonymous with endorsement",
        (
            "A rider is another term for an endorsement — a modification to an insurance policy. "
            "'Rider' is more commonly used in life and health insurance; "
            "'endorsement' is more common in P&C. "
            "Examples: waiver of premium rider (waives premiums if the insured becomes disabled), "
            "accidental death benefit rider."
        ),
        category="concept",
        see_also=["endorsement"],
    ),
    "exclusion": KnowledgeEntry(
        "Exclusion", "A policy provision that eliminates coverage for specified situations",
        (
            "An exclusion is a provision in an insurance policy that removes coverage "
            "for specific perils, persons, property types, or circumstances. "
            "Common exclusions:\n"
            "• Intentional acts.\n"
            "• War and nuclear hazard.\n"
            "• Flood (from standard property policies — requires separate NFIP or private flood policy).\n"
            "• Business use on personal auto policies.\n"
            "• Pre-existing conditions (some health policies).\n"
            "Exclusions define the boundaries of coverage and are always listed clearly in the policy."
        ),
        category="concept",
        see_also=["endorsement", "coverage limit"],
    ),
    "indemnity": KnowledgeEntry(
        "Indemnity", "The principle of restoring the insured to their pre-loss financial position",
        (
            "The principle of indemnity is a foundational insurance concept: "
            "insurance should restore the insured to the same financial position they were in "
            "before a loss — no better, no worse. "
            "This prevents 'profiting' from an insurance claim. "
            "Exception: Replacement Cost Value (RCV) policies pay to replace without depreciation, "
            "which can result in a slight improvement in some cases."
        ),
        category="concept",
        see_also=["subrogation"],
    ),
    "subrogation": KnowledgeEntry(
        "Subrogation", "The insurer's right to recover from the party responsible for a loss",
        (
            "Subrogation gives an insurer the legal right to pursue a third party "
            "that caused an insured loss — after the insurer has already paid the claim. "
            "Example: your insurer pays your $15,000 auto claim after a negligent driver hits you. "
            "Your insurer then sues the at-fault driver to recover the $15,000. "
            "The insured typically must cooperate and cannot waive subrogation rights without consent."
        ),
        category="concept",
        see_also=["indemnity"],
    ),
    "coverage limit": KnowledgeEntry(
        "Coverage Limit", "The maximum amount an insurer will pay for covered losses",
        (
            "The coverage limit (or policy limit) is the maximum dollar amount an insurer "
            "will pay for a covered claim or in a policy period. "
            "Key limit types:\n"
            "• Per-occurrence limit: max per individual event.\n"
            "• Per-person limit: max paid to any one person (common in auto BI).\n"
            "• Aggregate limit: total max paid in the policy period across all claims.\n"
            "Once the aggregate limit is exhausted, no further claims are paid until renewal."
        ),
        category="concept",
        see_also=["deductible", "aggregate limit"],
    ),
    "aggregate limit": KnowledgeEntry(
        "Aggregate Limit", "The maximum total payout across all claims in a policy period",
        (
            "The aggregate limit caps an insurer's total payout across all claims "
            "during a policy period (usually 12 months). "
            "Once exhausted, the policy provides no further coverage until it renews. "
            "Example: a GL policy with $1M per-occurrence and $2M aggregate means "
            "the insurer will pay at most $1M per claim, but no more than $2M total "
            "across all claims in the year."
        ),
        category="concept",
        see_also=["coverage limit"],
    ),
    "underwriting": KnowledgeEntry(
        "Underwriting", "The process of evaluating, pricing, and accepting insurance risk",
        (
            "Underwriting is the core process by which an insurer:\n"
            "1. Evaluates a risk (reviews the application, loss history, inspections).\n"
            "2. Decides whether to offer coverage (accept, decline, or offer with restrictions).\n"
            "3. Determines the premium, deductible, and any coverage exclusions.\n"
            "Underwriters are the professionals who make these decisions. "
            "'UW appetite' refers to the types and sizes of risk an insurer is willing to take."
        ),
        category="concept",
        see_also=["UW", "actuarial"],
    ),
    "actuarial": KnowledgeEntry(
        "Actuarial Science", "Mathematical discipline for measuring and pricing insurance risk",
        (
            "Actuarial science applies probability, statistics, and financial mathematics "
            "to quantify and manage insurance risk. Actuaries:\n"
            "• Set premium rates (pricing).\n"
            "• Establish claim reserves (including IBNR).\n"
            "• Model catastrophe losses (CAT modelling).\n"
            "• Ensure solvency (capital modelling).\n"
            "Designations: FCAS (Fellow of the Casualty Actuarial Society) for P&C; "
            "FSA (Fellow of the Society of Actuaries) for L&H."
        ),
        category="concept",
        see_also=["IBNR", "LR", "underwriting"],
    ),
    "reinsurance": KnowledgeEntry(
        "Reinsurance", "Insurance purchased by insurers to transfer risk to other insurers",
        (
            "Reinsurance is 'insurance for insurance companies.' "
            "An insurer (the 'cedant') transfers part of its risk portfolio to a reinsurer "
            "in exchange for a share of the premium. "
            "Key types:\n"
            "• Quota Share (QS): reinsurer takes a fixed % of every policy written.\n"
            "• Excess of Loss (XL): reinsurer pays losses above the cedant's retention.\n"
            "• Catastrophe XL: covers aggregate CAT losses above a threshold.\n"
            "Major reinsurers: Munich Re, Swiss Re, Hannover Re, Berkshire Hathaway Re, Lloyd's."
        ),
        category="concept",
        see_also=["GWP", "NWP", "CAT"],
    ),
    "claims adjuster": KnowledgeEntry(
        "Claims Adjuster", "Professional who investigates and settles insurance claims",
        (
            "A claims adjuster (also: claims handler, loss adjuster) investigates insurance claims, "
            "determines coverage, assesses damages, and negotiates settlements. "
            "Three types:\n"
            "• Staff adjuster: employee of the insurer.\n"
            "• Independent adjuster (IA): third-party contractor hired by insurers.\n"
            "• Public adjuster (PA): hired by the policyholder to advocate for the insured.\n"
            "Adjusters may specialise by line: auto, property, liability, CAT, or complex commercial."
        ),
        category="concept",
        see_also=["TPA", "LAE"],
    ),
    "first party": KnowledgeEntry(
        "First-Party Coverage", "Coverage protecting the policyholder's own losses",
        (
            "First-party coverage pays the policyholder (the 'first party') directly for their own losses. "
            "Examples: collision/comprehensive coverage on your own car, property coverage for your building, "
            "PIP medical coverage after an accident. "
            "Contrast with third-party (liability) coverage, which pays claims made by others against you."
        ),
        category="concept",
        see_also=["third party"],
    ),
    "third party": KnowledgeEntry(
        "Third-Party Coverage", "Liability coverage for claims made by others against the policyholder",
        (
            "Third-party coverage (liability insurance) pays when a third party "
            "— someone other than the insurer or insured — makes a claim against the policyholder "
            "for bodily injury or property damage. "
            "Examples: auto liability, General Liability (GL), professional liability (E&O, D&O). "
            "The 'three parties' are: 1) the insurer, 2) the insured, 3) the claimant."
        ),
        category="concept",
        see_also=["first party", "GL"],
    ),
    "occurrence policy": KnowledgeEntry(
        "Occurrence-Based Policy", "Covers events that happen during the policy period, whenever claimed",
        (
            "An occurrence policy covers any claim arising from an incident that occurred "
            "during the active policy period — even if the claim is filed years later after the policy expires. "
            "Example: a GL occurrence policy in 2024 will still respond to a 2024 incident "
            "even if the lawsuit isn't filed until 2027. "
            "Common in: General Liability, Personal Auto, Homeowners."
        ),
        category="concept",
        see_also=["claims-made policy"],
    ),
    "claims-made policy": KnowledgeEntry(
        "Claims-Made Policy", "Coverage only for claims reported during the active policy period",
        (
            "A claims-made policy covers only claims that are both made AND reported "
            "to the insurer during the active policy period. "
            "If the policy expires before a claim is filed, there is no coverage "
            "unless a 'tail' (Extended Reporting Period) is purchased. "
            "Common in: Professional Liability (E&O), D&O, Cyber, Medical Malpractice. "
            "Important for new employees: always check if a professional liability policy is "
            "claims-made and whether tail coverage is maintained."
        ),
        category="concept",
        see_also=["occurrence policy", "E&O"],
    ),
    "declarations page": KnowledgeEntry(
        "Declarations Page", "The policy 'summary page' showing all key coverage details",
        (
            "The declarations page (dec page) is the first section of an insurance policy. "
            "It summarises all key information at a glance:\n"
            "• Named insured and address.\n"
            "• Policy period (effective and expiration dates).\n"
            "• Coverage types, limits, and deductibles.\n"
            "• Premium amount.\n"
            "• Property or vehicle described.\n"
            "The dec page is what agents first look at when reviewing a policy "
            "and what lenders or landlords request as proof of coverage."
        ),
        category="concept",
        see_also=["named insured", "coverage limit"],
    ),
    "named insured": KnowledgeEntry(
        "Named Insured", "The person or entity listed on the policy as the primary policyholder",
        (
            "The named insured is the policyholder — specifically named on the declarations page. "
            "They have the broadest rights under the policy: to make changes, cancel, "
            "and file claims. "
            "Additional insureds can be added by endorsement (e.g., a landlord added to a tenant's GL policy) "
            "but have narrower rights — typically only coverage for liability arising from the named insured's operations."
        ),
        category="concept",
        see_also=["declarations page", "endorsement"],
    ),
    "coinsurance": KnowledgeEntry(
        "Coinsurance", "Requirement to insure property to a minimum percentage of value",
        (
            "In property insurance, coinsurance requires the insured to carry coverage equal to at least "
            "a specified percentage (typically 80-90%) of the property's full replacement value. "
            "If under-insured, a coinsurance penalty applies: "
            "the insurer pays only the proportion of the loss equal to "
            "(amount of insurance carried ÷ required amount) × loss amount. "
            "Example: 80% coinsurance clause, building worth $1M, insured for only $600K. "
            "Required: $800K. On a $100K loss: insurer pays ($600K/$800K) × $100K = $75K."
        ),
        category="concept",
        see_also=["coverage limit", "deductible"],
    ),
    "binder": KnowledgeEntry(
        "Binder", "Temporary evidence of coverage while the formal policy is being issued",
        (
            "A binder is a temporary document confirming that insurance is in force "
            "while the full policy is being underwritten, processed, and issued. "
            "Typically valid for 30-90 days. "
            "Used in real estate closings (lenders require proof of property insurance at closing), "
            "commercial transactions, and new business submissions. "
            "The binder is replaced by the formal policy once issued."
        ),
        category="concept",
        see_also=["declarations page"],
    ),
}


# ── Search ────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return text.lower().strip()


def _is_definition_question(question: str) -> bool:
    """Return True if the question is asking for a definition or explanation."""
    q = _normalise(question)
    triggers = [
        "what is", "what does", "what are", "what do", "what's",
        "stand for", "mean", "means", "meaning", "define", "definition",
        "explain", "tell me about", "describe", "difference between",
        "what kind", "how does", "how do", "why is", "who is",
    ]
    return any(t in q for t in triggers)


def search_knowledge(question: str, max_results: int = 3) -> list[KnowledgeEntry]:
    """Search the knowledge base and return matching entries, best-first."""
    results: list[tuple[int, KnowledgeEntry]] = []
    q_raw = question.strip()
    q_lower = q_raw.lower()

    # 1. Find uppercase acronyms in the question
    acronyms_in_q = re.findall(r"\b([A-Z][A-Z&]+)\b", q_raw)
    for acr in acronyms_in_q:
        if acr in ACRONYMS:
            results.append((100, ACRONYMS[acr]))

    # 2. Check if the question contains a known term literally
    for key, entry in ACRONYMS.items():
        if key.lower() in q_lower and entry not in [r[1] for r in results]:
            results.append((80, entry))

    # 3. Check TERMS dict
    for key, entry in TERMS.items():
        if key.lower() in q_lower:
            results.append((70, entry))
        elif entry.term.lower() in q_lower:
            results.append((65, entry))

    # 4. Keyword scan of short/detail text for definition questions
    if _is_definition_question(question):
        q_words = set(re.findall(r"[a-z]{3,}", q_lower))
        for key, entry in {**ACRONYMS, **TERMS}.items():
            if entry in [r[1] for r in results]:
                continue
            entry_words = set(re.findall(r"[a-z]{3,}", (key + " " + entry.short).lower()))
            overlap = len(q_words & entry_words)
            if overlap >= 2:
                results.append((40 + overlap, entry))

    # De-duplicate, sort by score descending, return top N
    seen: set[str] = set()
    unique: list[KnowledgeEntry] = []
    for _, entry in sorted(results, key=lambda x: -x[0]):
        if entry.term not in seen:
            seen.add(entry.term)
            unique.append(entry)
        if len(unique) >= max_results:
            break
    return unique


def format_knowledge_answer(entries: list[KnowledgeEntry]) -> str:
    """Format knowledge entries into a readable answer string."""
    if not entries:
        return ""
    parts = []
    for entry in entries:
        header = f"**{entry.term}** — {entry.short}"
        body = entry.detail
        see = ""
        if entry.see_also:
            see = "See also: " + ", ".join(entry.see_also)
        parts.append("\n".join(filter(None, [header, body, see])))
    return "\n\n---\n\n".join(parts)
