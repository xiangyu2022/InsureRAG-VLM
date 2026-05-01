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


# ── Extended Insurance Glossary ───────────────────────────────────────────────
#
# Keep the core dictionaries above readable, then add broader industry coverage
# here. These entries are intentionally concise: the knowledge base is used as a
# deterministic first stop before the RAG/LLM path, so definitions should be
# quick to retrieve and easy to cite in the demo UI.

def _entry(
    term: str,
    short: str,
    detail: str,
    category: str = "concept",
    see_also: list[str] | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(term, short, detail, category=category, see_also=see_also or [])


ACRONYMS.update(
    {
        "ACV": _entry(
            "ACV",
            "Actual Cash Value",
            "ACV is a valuation method equal to replacement cost minus depreciation. If a 7-year-old roof is damaged, ACV pays for the roof's depreciated value rather than the cost of a brand-new roof.",
            "acronym",
            ["RCV", "depreciation"],
        ),
        "RCV": _entry(
            "RCV",
            "Replacement Cost Value",
            "RCV pays the cost to repair or replace damaged property with similar new property, without deducting depreciation, subject to policy terms and limits.",
            "acronym",
            ["ACV", "recoverable depreciation"],
        ),
        "ALE": _entry(
            "ALE",
            "Additional Living Expense",
            "ALE pays extra costs to live elsewhere when a covered homeowners loss makes the residence uninhabitable, such as hotel, meals above normal cost, and temporary housing.",
            "acronym",
            ["loss of use", "HO"],
        ),
        "AOP": _entry(
            "AOP",
            "All Other Perils",
            "AOP refers to covered causes of loss other than specially separated perils such as wind, hail, hurricane, flood, or earthquake. Deductibles are often shown as AOP vs. named CAT deductibles.",
            "acronym",
            ["deductible", "CAT"],
        ),
        "PAP": _entry(
            "PAP",
            "Personal Auto Policy",
            "PAP is the standard policy form for personal auto insurance, covering liability, medical/PIP, uninsured motorist, collision, comprehensive, and optional endorsements.",
            "acronym",
            ["PI", "collision coverage", "comprehensive coverage"],
        ),
        "BAP": _entry(
            "BAP",
            "Business Auto Policy",
            "BAP is the commercial auto policy used for vehicles owned, hired, or used by a business. It can include liability, physical damage, hired auto, and non-owned auto coverage.",
            "acronym",
            ["HNOA", "CI"],
        ),
        "HNOA": _entry(
            "HNOA",
            "Hired and Non-Owned Auto",
            "HNOA covers business auto liability arising from rented vehicles or employees' personal vehicles used for business, when the company does not own the auto.",
            "acronym",
            ["BAP", "third party"],
        ),
        "CSL": _entry(
            "CSL",
            "Combined Single Limit",
            "CSL is one liability limit that applies to both bodily injury and property damage from an auto accident, instead of split limits by person, accident, and property damage.",
            "acronym",
            ["BI", "PD"],
        ),
        "UMPD": _entry(
            "UMPD",
            "Uninsured Motorist Property Damage",
            "UMPD pays for damage to the insured vehicle or property caused by an uninsured driver, when available under state law and selected on the policy.",
            "acronym",
            ["UM", "UIM", "PD"],
        ),
        "MEDPAY": _entry(
            "MedPay",
            "Medical Payments Coverage",
            "MedPay pays reasonable medical expenses for the insured and passengers after an auto accident, usually regardless of fault, up to a small selected limit.",
            "acronym",
            ["PIP", "PAP"],
        ),
        "MVR": _entry(
            "MVR",
            "Motor Vehicle Record",
            "MVR is a driver's record of accidents, violations, license status, and sometimes suspensions. Auto underwriters use MVRs to price and accept personal or commercial auto risks.",
            "acronym",
            ["PAP", "BAP", "underwriting"],
        ),
        "VIN": _entry(
            "VIN",
            "Vehicle Identification Number",
            "VIN is the unique vehicle identifier used to determine vehicle year, make, model, trim, safety features, theft exposure, and insurance rating characteristics.",
            "acronym",
            ["PAP", "BAP"],
        ),
        "CLUE": _entry(
            "CLUE",
            "Comprehensive Loss Underwriting Exchange",
            "CLUE is a claims history database used by many insurers to review prior auto and property losses associated with a person or property.",
            "acronym",
            ["loss history", "underwriting"],
        ),
        "FNOL": _entry(
            "FNOL",
            "First Notice of Loss",
            "FNOL is the first report of a claim to an insurer. It starts claim intake and captures facts such as date, location, parties involved, cause of loss, and initial damages.",
            "acronym",
            ["notice of loss", "claims adjuster"],
        ),
        "SIU": _entry(
            "SIU",
            "Special Investigation Unit",
            "SIU is the insurer team that investigates suspected fraud, staged accidents, inflated damages, arson, false statements, or organized claim schemes.",
            "acronym",
            ["fraud", "claims adjuster"],
        ),
        "COI": _entry(
            "COI",
            "Certificate of Insurance",
            "COI is a summary document proving that insurance exists. It lists the named insured, insurer, policy dates, coverages, limits, and certificate holder, but it does not amend coverage.",
            "acronym",
            ["certificate of insurance", "additional insured"],
        ),
        "AI": _entry(
            "AI",
            "Additional Insured",
            "In insurance documents, AI often means Additional Insured: a person or organization added to another party's policy for limited liability protection.",
            "acronym",
            ["additional insured", "endorsement"],
        ),
        "NFIP": _entry(
            "NFIP",
            "National Flood Insurance Program",
            "NFIP is the US federal flood insurance program administered through FEMA. Standard homeowners policies usually exclude flood, so flood coverage is bought through NFIP or private flood insurers.",
            "acronym",
            ["FEMA", "flood insurance"],
        ),
        "FEMA": _entry(
            "FEMA",
            "Federal Emergency Management Agency",
            "FEMA administers disaster response programs and the National Flood Insurance Program in the US. FEMA flood maps are important for flood underwriting and lender requirements.",
            "acronym",
            ["NFIP", "flood insurance"],
        ),
        "FAIR": _entry(
            "FAIR Plan",
            "Fair Access to Insurance Requirements Plan",
            "A FAIR Plan is a state-backed residual market mechanism for property owners who cannot obtain coverage in the standard market, often because of wildfire, coastal, or urban exposure.",
            "acronym",
            ["residual market", "admitted carrier"],
        ),
        "DIC": _entry(
            "DIC",
            "Difference in Conditions",
            "DIC coverage fills gaps in a standard property policy, often for excluded perils such as flood or earthquake. It is common for large commercial property accounts.",
            "acronym",
            ["flood insurance", "earthquake endorsement"],
        ),
        "DIL": _entry(
            "DIL",
            "Difference in Limits",
            "DIL coverage provides excess limits above an underlying policy or program. It is often discussed together with DIC in layered property placements.",
            "acronym",
            ["DIC", "excess policy"],
        ),
        "IM": _entry(
            "IM",
            "Inland Marine",
            "Inland Marine covers movable property, property in transit, equipment, tools, fine arts, installation exposures, and other property that does not fit neatly into standard property forms.",
            "acronym",
            ["scheduled personal property", "bailee coverage"],
        ),
        "BPP": _entry(
            "BPP",
            "Business Personal Property",
            "BPP is business-owned contents, furniture, inventory, machinery, and equipment at an insured location under a commercial property policy.",
            "acronym",
            ["commercial property", "TIV"],
        ),
        "EE": _entry(
            "EE",
            "Extra Expense",
            "Extra Expense coverage pays necessary additional costs incurred to continue operations after a covered property loss, such as temporary space, expedited shipping, or emergency equipment rental.",
            "acronym",
            ["business income", "BI"],
        ),
        "CBI": _entry(
            "CBI",
            "Contingent Business Interruption",
            "CBI covers income loss caused by covered damage to a dependent property, supplier, customer, or key business partner rather than to the insured's own premises.",
            "acronym",
            ["business income", "supply chain risk"],
        ),
        "EB": _entry(
            "EB",
            "Equipment Breakdown",
            "EB covers sudden mechanical, electrical, or pressure-system breakdown of covered equipment, such as boilers, HVAC, electrical panels, or production machinery.",
            "acronym",
            ["commercial property", "BPP"],
        ),
        "EPLI": _entry(
            "EPLI",
            "Employment Practices Liability Insurance",
            "EPLI is another common acronym for Employment Practices Liability. It covers employment-related claims such as discrimination, harassment, retaliation, and wrongful termination.",
            "acronym",
            ["EPL", "D&O"],
        ),
        "RBC": _entry(
            "RBC",
            "Risk-Based Capital",
            "RBC is a regulatory capital framework that compares an insurer's available capital to the riskiness of its assets, underwriting, credit, and operational exposures.",
            "metric",
            ["NAIC", "solvency"],
        ),
        "NAIC": _entry(
            "NAIC",
            "National Association of Insurance Commissioners",
            "NAIC is the US organization of state insurance regulators. It develops model laws, solvency tools, data reporting frameworks, and regulatory guidance.",
            "acronym",
            ["DOI", "RBC", "SERFF"],
        ),
        "DOI": _entry(
            "DOI",
            "Department of Insurance",
            "DOI usually means a state Department of Insurance. It regulates insurers, agents, rates, policy forms, solvency, consumer complaints, and market conduct.",
            "acronym",
            ["NAIC", "rate filing"],
        ),
        "SERFF": _entry(
            "SERFF",
            "System for Electronic Rates and Forms Filing",
            "SERFF is the NAIC electronic platform many insurers use to submit rate, rule, and policy-form filings to state insurance regulators.",
            "acronym",
            ["NAIC", "rate filing", "form filing"],
        ),
        "AMB": _entry(
            "AM Best",
            "Insurance financial strength rating agency",
            "AM Best is a rating agency focused on insurance companies. Its financial strength ratings are used by brokers, policyholders, reinsurers, and regulators to assess insurer credit quality.",
            "acronym",
            ["solvency", "policyholder surplus"],
        ),
        "SIR": _entry(
            "SIR",
            "Self-Insured Retention",
            "SIR is the amount the insured must pay and often handle before the insurer's obligation begins. Unlike a deductible, the insured may manage claims within the SIR layer.",
            "acronym",
            ["self-insured retention", "deductible"],
        ),
        "XOL": _entry(
            "XOL",
            "Excess of Loss Reinsurance",
            "XOL reinsurance responds when losses exceed the cedant's retention or attachment point, up to the reinsurance limit.",
            "acronym",
            ["excess of loss", "attachment point"],
        ),
        "QS": _entry(
            "QS",
            "Quota Share Reinsurance",
            "QS is proportional reinsurance where the reinsurer takes a fixed percentage of premiums and losses on every covered policy.",
            "acronym",
            ["quota share", "reinsurance"],
        ),
        "AAD": _entry(
            "AAD",
            "Annual Aggregate Deductible",
            "AAD is an aggregate deductible that must be exhausted across losses during a treaty year before reinsurance responds.",
            "acronym",
            ["XOL", "reinsurance"],
        ),
        "OEP": _entry(
            "OEP",
            "Occurrence Exceedance Probability",
            "OEP is a catastrophe-model metric showing the probability that the largest single event loss in a year exceeds a given amount.",
            "metric",
            ["AEP", "CAT", "PML"],
        ),
        "AEP": _entry(
            "AEP",
            "Aggregate Exceedance Probability",
            "AEP is a catastrophe-model metric showing the probability that total annual losses from all events exceed a given amount.",
            "metric",
            ["OEP", "CAT", "AAL"],
        ),
        "TVaR": _entry(
            "TVaR",
            "Tail Value at Risk",
            "TVaR is the average loss severity beyond a selected percentile, such as the average of losses worse than the 99th percentile. It is used for capital and catastrophe risk.",
            "metric",
            ["CAT", "RBC"],
        ),
        "AD&D": _entry(
            "AD&D",
            "Accidental Death and Dismemberment",
            "AD&D pays benefits for death or certain severe injuries caused by an accident. It is narrower than life insurance because non-accidental death is not covered.",
            "acronym",
            ["L&H", "beneficiary"],
        ),
        "LTC": _entry(
            "LTC",
            "Long-Term Care",
            "LTC insurance helps pay for extended care services such as nursing homes, assisted living, or in-home care when the insured cannot perform activities of daily living.",
            "acronym",
            ["elimination period", "L&H"],
        ),
        "DI": _entry(
            "DI",
            "Disability Income",
            "DI insurance replaces part of income when the insured cannot work because of illness or injury, subject to an elimination period, benefit period, and disability definition.",
            "acronym",
            ["elimination period", "L&H"],
        ),
        "HMO": _entry(
            "HMO",
            "Health Maintenance Organization",
            "An HMO is a managed care health plan that typically requires members to use in-network providers and obtain referrals for specialists.",
            "acronym",
            ["PPO", "EPO"],
        ),
        "PPO": _entry(
            "PPO",
            "Preferred Provider Organization",
            "A PPO health plan gives more flexibility to use out-of-network providers than an HMO, usually with higher member cost-sharing.",
            "acronym",
            ["HMO", "EPO"],
        ),
        "EPO": _entry(
            "EPO",
            "Exclusive Provider Organization",
            "An EPO health plan generally covers only in-network care except emergencies, but may not require primary-care referrals like an HMO.",
            "acronym",
            ["HMO", "PPO"],
        ),
        "HDHP": _entry(
            "HDHP",
            "High-Deductible Health Plan",
            "An HDHP is a health plan with a high deductible that may qualify the member to contribute to a Health Savings Account if other requirements are met.",
            "acronym",
            ["HSA", "deductible"],
        ),
        "HSA": _entry(
            "HSA",
            "Health Savings Account",
            "An HSA is a tax-advantaged account used to pay qualified medical expenses, generally paired with a qualifying high-deductible health plan.",
            "acronym",
            ["HDHP", "L&H"],
        ),
    }
)


TERMS.update(
    {
        "actual cash value": ACRONYMS["ACV"],
        "replacement cost value": ACRONYMS["RCV"],
        "additional living expense": ACRONYMS["ALE"],
        "loss of use": _entry(
            "Loss of Use",
            "Coverage for extra costs when property cannot be used after a covered loss",
            "Loss of Use pays additional expenses or lost rental value when a covered loss prevents normal use of the insured property. In homeowners policies, ALE is the common loss-of-use benefit.",
            "concept",
            ["ALE"],
        ),
        "dwelling coverage": _entry(
            "Dwelling Coverage",
            "Homeowners Coverage A for the house structure",
            "Dwelling coverage protects the main residence structure, including attached structures, against covered causes of loss. It is usually the anchor limit for other homeowners coverages.",
            "concept",
            ["HO", "other structures"],
        ),
        "other structures": _entry(
            "Other Structures",
            "Coverage for detached structures on the residence premises",
            "Other Structures coverage protects detached garages, fences, sheds, and similar structures. It is often a percentage of the dwelling limit.",
            "concept",
            ["dwelling coverage", "HO"],
        ),
        "personal property": _entry(
            "Personal Property",
            "Coverage for contents owned or used by the insured",
            "Personal Property coverage protects contents such as furniture, clothing, electronics, and household goods, subject to limits, special limits, and exclusions.",
            "concept",
            ["scheduled personal property", "special limit"],
        ),
        "scheduled personal property": _entry(
            "Scheduled Personal Property",
            "Itemized coverage for valuable property",
            "Scheduled personal property lists valuable items such as jewelry, fine art, cameras, or musical instruments with specific values, often providing broader coverage than unscheduled contents.",
            "concept",
            ["personal articles floater", "special limit"],
        ),
        "personal articles floater": _entry(
            "Personal Articles Floater",
            "Standalone or endorsed coverage for high-value personal items",
            "A personal articles floater insures valuable personal property, often worldwide and on broader terms than a standard homeowners contents limit.",
            "concept",
            ["scheduled personal property", "IM"],
        ),
        "special limit": _entry(
            "Special Limit of Liability",
            "A lower sublimit for specific categories of property",
            "A special limit caps coverage for certain property categories such as jewelry, cash, firearms, silverware, or business property, even when the overall personal property limit is higher.",
            "concept",
            ["sublimit", "personal property"],
        ),
        "sublimit": _entry(
            "Sublimit",
            "A smaller limit inside a broader policy limit",
            "A sublimit restricts how much the policy will pay for a specific type of loss, property, or expense within the larger coverage limit.",
            "concept",
            ["coverage limit", "special limit"],
        ),
        "blanket limit": _entry(
            "Blanket Limit",
            "One shared limit across multiple items or locations",
            "A blanket limit applies across multiple insured properties, locations, or categories instead of assigning a separate scheduled limit to each item.",
            "concept",
            ["scheduled personal property", "TIV"],
        ),
        "ordinance or law": _entry(
            "Ordinance or Law",
            "Coverage for increased costs from building code enforcement",
            "Ordinance or Law coverage pays for demolition, increased construction cost, or undamaged portions of a building when current building codes apply after a covered loss.",
            "concept",
            ["replacement cost value", "commercial property"],
        ),
        "water backup": _entry(
            "Water Backup",
            "Coverage for sewer or drain backup losses",
            "Water backup coverage pays for damage from water backing up through sewers, drains, or sump systems. It is usually excluded or limited unless endorsed.",
            "concept",
            ["endorsement", "exclusion"],
        ),
        "flood insurance": _entry(
            "Flood Insurance",
            "Separate coverage for surface-water flooding",
            "Flood insurance covers direct physical loss from flood, which standard homeowners and many commercial property policies exclude. It may be purchased through NFIP or private flood markets.",
            "concept",
            ["NFIP", "FEMA", "exclusion"],
        ),
        "earthquake endorsement": _entry(
            "Earthquake Endorsement",
            "Added coverage for earthquake loss",
            "An earthquake endorsement adds earthquake coverage to a property policy, often with a percentage deductible and separate limits or exclusions.",
            "concept",
            ["EQ", "endorsement"],
        ),
        "wind hail deductible": _entry(
            "Wind/Hail Deductible",
            "Separate deductible for windstorm or hail losses",
            "A wind/hail deductible applies specifically to windstorm or hail claims and may be a flat amount or a percentage of insured value, especially in coastal or severe-convective-storm areas.",
            "concept",
            ["AOP", "hurricane deductible"],
        ),
        "hurricane deductible": _entry(
            "Hurricane Deductible",
            "Separate deductible triggered by hurricane loss",
            "A hurricane deductible applies when a loss is caused by a hurricane under the policy's trigger language. It is often a percentage of dwelling or property value.",
            "concept",
            ["wind hail deductible", "CAT"],
        ),
        "named peril": _entry(
            "Named Peril",
            "Coverage only for causes of loss listed in the policy",
            "A named-peril form covers only the specific perils named in the policy, such as fire, lightning, windstorm, theft, or vandalism.",
            "concept",
            ["open peril", "basic form"],
        ),
        "open peril": _entry(
            "Open Peril",
            "Coverage for all causes of loss except excluded ones",
            "An open-peril form covers direct physical loss unless the policy excludes the cause. It is broader than named-peril coverage.",
            "concept",
            ["named peril", "special form"],
        ),
        "all risk": _entry(
            "All Risk",
            "Older term for open-peril coverage",
            "All Risk generally means open-peril coverage: losses are covered unless excluded. The term can be misleading because no policy covers every possible loss.",
            "concept",
            ["open peril", "exclusion"],
        ),
        "basic form": _entry(
            "Basic Form",
            "Narrow named-peril property coverage",
            "Basic Form property coverage usually includes a short list of perils such as fire, lightning, explosion, windstorm, smoke, aircraft, vehicles, riot, vandalism, sprinkler leakage, sinkhole, and volcanic action.",
            "concept",
            ["named peril", "broad form"],
        ),
        "broad form": _entry(
            "Broad Form",
            "Named-peril property coverage broader than Basic Form",
            "Broad Form adds perils such as falling objects, weight of ice/snow/sleet, water damage, and collapse to Basic Form coverage.",
            "concept",
            ["basic form", "special form"],
        ),
        "special form": _entry(
            "Special Form",
            "Open-peril property coverage",
            "Special Form generally covers direct physical loss unless excluded, making it broader than Basic or Broad Form named-peril coverage.",
            "concept",
            ["open peril", "HO"],
        ),
        "vacancy clause": _entry(
            "Vacancy Clause",
            "Policy condition restricting coverage when property is vacant",
            "A vacancy clause reduces or eliminates coverage for certain losses when a building has been vacant beyond the allowed period, often 30 or 60 days.",
            "concept",
            ["occupancy", "exclusion"],
        ),
        "occupancy": _entry(
            "Occupancy",
            "How property is used or lived in",
            "Occupancy describes whether a property is owner-occupied, tenant-occupied, vacant, seasonal, commercial, or mixed-use. It materially affects underwriting and coverage.",
            "concept",
            ["vacancy clause", "underwriting"],
        ),
        "mortgagee clause": _entry(
            "Mortgagee Clause",
            "Policy language protecting the lender's interest",
            "A mortgagee clause gives a lender rights under a property policy, often allowing claim payment to the lender even if the insured violates certain policy conditions.",
            "concept",
            ["loss payee", "declarations page"],
        ),
        "loss payee": _entry(
            "Loss Payee",
            "Party entitled to claim payments for covered property",
            "A loss payee is a lender, lessor, or other party named to receive claim payments for property in which it has a financial interest.",
            "concept",
            ["mortgagee clause", "lender loss payable"],
        ),
        "lender loss payable": _entry(
            "Lender Loss Payable",
            "Enhanced loss-payee protection for lenders",
            "A lender loss payable clause gives the lender stronger rights than a simple loss payee, commonly used for financed equipment or commercial property.",
            "concept",
            ["loss payee", "mortgagee clause"],
        ),
        "additional insured": ACRONYMS["AI"],
        "certificate of insurance": ACRONYMS["COI"],
        "waiver of subrogation": _entry(
            "Waiver of Subrogation",
            "Agreement that the insurer will not pursue recovery from a specified party",
            "A waiver of subrogation prevents the insurer from recovering from a party that might otherwise be legally responsible. It is common in construction, leases, and vendor contracts.",
            "concept",
            ["subrogation", "endorsement"],
        ),
        "primary and noncontributory": _entry(
            "Primary and Noncontributory",
            "Coverage responds first and does not seek contribution from another policy",
            "Primary and noncontributory language means the insured's policy pays before another party's insurance and will not ask that other insurance to share the loss.",
            "concept",
            ["additional insured", "COI"],
        ),
        "contractual liability": _entry(
            "Contractual Liability",
            "Liability assumed under a contract",
            "Contractual liability is liability one party assumes in a contract, such as an indemnity agreement. CGL policies cover some insured contracts but exclude many broad obligations.",
            "concept",
            ["hold harmless agreement", "GL"],
        ),
        "hold harmless agreement": _entry(
            "Hold Harmless Agreement",
            "Contract provision shifting liability from one party to another",
            "A hold harmless agreement requires one party to protect another from certain losses or claims. Underwriters review these provisions because they can expand the insured's assumed liability.",
            "concept",
            ["contractual liability", "indemnity"],
        ),
        "duty to defend": _entry(
            "Duty to Defend",
            "Insurer's obligation to provide legal defense for covered allegations",
            "Duty to defend means the insurer must defend the insured against suits that potentially seek covered damages, often broader than the duty to indemnify.",
            "concept",
            ["duty to indemnify", "reservation of rights"],
        ),
        "duty to indemnify": _entry(
            "Duty to Indemnify",
            "Insurer's obligation to pay covered judgments or settlements",
            "Duty to indemnify is the obligation to pay covered damages for which the insured is legally liable, subject to policy terms and limits.",
            "concept",
            ["duty to defend", "coverage limit"],
        ),
        "reservation of rights": _entry(
            "Reservation of Rights",
            "Notice that the insurer may defend while preserving coverage defenses",
            "A reservation of rights letter tells the insured the insurer will investigate or defend while reserving the right to deny coverage later based on policy terms.",
            "concept",
            ["coverage investigation", "denial letter"],
        ),
        "coverage investigation": _entry(
            "Coverage Investigation",
            "Claim review focused on whether policy coverage applies",
            "A coverage investigation examines facts, policy language, exclusions, conditions, and endorsements to decide whether a claim is covered.",
            "concept",
            ["reservation of rights", "claims adjuster"],
        ),
        "denial letter": _entry(
            "Denial Letter",
            "Written explanation that a claim is not covered",
            "A denial letter explains why an insurer is denying all or part of a claim, usually citing relevant policy provisions, facts, and applicable exclusions or conditions.",
            "concept",
            ["coverage investigation", "exclusion"],
        ),
        "notice of loss": _entry(
            "Notice of Loss",
            "The insured's report of a loss to the insurer",
            "Notice of loss is the initial claim notice required by policy conditions. Late notice can create coverage issues if it prejudices the insurer.",
            "concept",
            ["FNOL", "proof of loss"],
        ),
        "proof of loss": _entry(
            "Proof of Loss",
            "Formal sworn statement of claimed damages",
            "Proof of loss is a signed statement documenting the amount claimed, cause of loss, and supporting information. Some policies require it within a stated deadline.",
            "concept",
            ["notice of loss", "claims adjuster"],
        ),
        "appraisal clause": _entry(
            "Appraisal Clause",
            "Policy process for resolving amount-of-loss disputes",
            "An appraisal clause lets the insured and insurer each choose an appraiser, with an umpire resolving differences, when they disagree about the amount of loss rather than coverage.",
            "concept",
            ["proof of loss", "claim settlement"],
        ),
        "claim settlement": _entry(
            "Claim Settlement",
            "Resolution and payment of a covered claim",
            "Claim settlement is the process of agreeing on covered damages and issuing payment, repair authorization, replacement, or denial according to policy terms.",
            "concept",
            ["claims adjuster", "coverage investigation"],
        ),
        "total loss": _entry(
            "Total Loss",
            "Loss where repair cost or damage severity exceeds the total-loss threshold",
            "A total loss occurs when property is destroyed or when repair cost exceeds an economic or statutory threshold. Auto claims often compare repair cost to actual cash value.",
            "concept",
            ["ACV", "salvage"],
        ),
        "salvage": _entry(
            "Salvage",
            "Damaged property retained or sold after a claim payment",
            "Salvage is damaged property that still has residual value after the insurer pays a loss. The insurer may sell salvage to offset claim cost.",
            "concept",
            ["total loss", "subrogation"],
        ),
        "depreciation": _entry(
            "Depreciation",
            "Reduction in value due to age, wear, or obsolescence",
            "Depreciation is deducted under ACV valuation. In some RCV claims, depreciation is withheld first and then paid as recoverable depreciation after repairs are completed.",
            "concept",
            ["ACV", "recoverable depreciation"],
        ),
        "recoverable depreciation": _entry(
            "Recoverable Depreciation",
            "Depreciation paid back after completed repair or replacement",
            "Recoverable depreciation is the withheld depreciation that can be paid once the insured completes eligible repair or replacement under an RCV policy.",
            "concept",
            ["RCV", "depreciation"],
        ),
        "betterment": _entry(
            "Betterment",
            "Improvement beyond pre-loss condition",
            "Betterment is an improvement that leaves the insured better off than before the loss. Insurers may subtract betterment to preserve the principle of indemnity.",
            "concept",
            ["indemnity", "depreciation"],
        ),
        "collision coverage": _entry(
            "Collision Coverage",
            "Auto physical damage coverage for collision or upset",
            "Collision coverage pays for damage to the insured vehicle from collision with another object or vehicle, or vehicle upset, subject to deductible.",
            "concept",
            ["comprehensive coverage", "PAP"],
        ),
        "comprehensive coverage": _entry(
            "Comprehensive Coverage",
            "Auto physical damage coverage for non-collision losses",
            "Comprehensive coverage pays for auto losses such as theft, vandalism, fire, falling objects, flood, hail, or animal impact, subject to deductible.",
            "concept",
            ["collision coverage", "PAP"],
        ),
        "rental reimbursement": _entry(
            "Rental Reimbursement",
            "Optional auto coverage for temporary transportation costs",
            "Rental reimbursement pays for a rental car or transportation expense while the insured vehicle is being repaired after a covered physical damage claim.",
            "concept",
            ["endorsement", "PAP"],
        ),
        "towing and labor": _entry(
            "Towing and Labor",
            "Optional auto roadside assistance coverage",
            "Towing and labor coverage reimburses certain roadside expenses such as towing, jump starts, lockout service, or tire changes, subject to policy limits.",
            "concept",
            ["PAP", "endorsement"],
        ),
        "gap coverage": _entry(
            "GAP Coverage",
            "Pays loan or lease gap after a total loss",
            "GAP coverage helps pay the difference between a vehicle's actual cash value and the outstanding loan or lease balance after a covered total loss.",
            "concept",
            ["total loss", "ACV"],
        ),
        "permissive use": _entry(
            "Permissive Use",
            "Coverage for someone driving with the insured's permission",
            "Permissive use refers to coverage for a driver who has permission to use the insured vehicle, subject to policy definitions, exclusions, and state rules.",
            "concept",
            ["excluded driver", "PAP"],
        ),
        "excluded driver": _entry(
            "Excluded Driver",
            "Driver specifically removed from coverage",
            "An excluded driver endorsement removes or restricts coverage when a named person operates the vehicle. It is used when a household driver has unacceptable risk.",
            "concept",
            ["endorsement", "MVR"],
        ),
        "garaging address": _entry(
            "Garaging Address",
            "Location where a vehicle is primarily kept",
            "Garaging address affects auto rates because theft, weather, traffic density, and claim frequency vary by territory.",
            "concept",
            ["territory", "PAP"],
        ),
        "telematics": _entry(
            "Telematics",
            "Use of driving data for insurance pricing or risk selection",
            "Telematics programs collect driving behavior such as mileage, speed, braking, cornering, time of day, or phone use to support usage-based insurance pricing.",
            "concept",
            ["rating factor", "underwriting"],
        ),
        "loss history": _entry(
            "Loss History",
            "Record of prior claims or losses",
            "Loss history shows the frequency, severity, and type of prior claims. It is a major underwriting and pricing input across personal and commercial lines.",
            "concept",
            ["CLUE", "underwriting"],
        ),
        "experience rating": _entry(
            "Experience Rating",
            "Pricing based on the insured's own historical losses",
            "Experience rating adjusts premium using the insured's historical loss experience, common in workers' compensation, commercial auto, and large commercial accounts.",
            "concept",
            ["loss history", "credibility"],
        ),
        "schedule rating": _entry(
            "Schedule Rating",
            "Underwriter judgment adjustment to manual premium",
            "Schedule rating applies debit or credit factors for risk characteristics not fully captured by standard rating variables, such as management quality or safety controls.",
            "concept",
            ["underwriting", "rating factor"],
        ),
        "class code": _entry(
            "Class Code",
            "Code grouping similar risks for rating",
            "A class code groups risks with similar exposure characteristics, such as workers' compensation job classifications or GL business classes.",
            "concept",
            ["rate", "exposure"],
        ),
        "territory": _entry(
            "Territory",
            "Geographic rating area",
            "Territory is a geographic rating variable reflecting local loss cost differences such as weather, crime, traffic, litigation, repair cost, or catastrophe exposure.",
            "concept",
            ["garaging address", "rating factor"],
        ),
        "exposure": _entry(
            "Exposure",
            "Unit measuring how much risk is insured",
            "Exposure is the measurable basis for premium, such as payroll, sales, vehicle count, building value, insured value, car-years, or policy count.",
            "concept",
            ["exposure base", "rate"],
        ),
        "exposure base": _entry(
            "Exposure Base",
            "The denominator used to apply an insurance rate",
            "Exposure base is the unit to which a rate is applied, such as payroll per $100, sales per $1,000, vehicle-years, or insured value per $100.",
            "concept",
            ["exposure", "rate"],
        ),
        "rate": _entry(
            "Rate",
            "Price per unit of exposure",
            "A rate is the price charged per exposure unit. Premium is commonly calculated as rate times exposure, adjusted for modifiers, deductibles, limits, and fees.",
            "concept",
            ["exposure", "rating factor"],
        ),
        "rating factor": _entry(
            "Rating Factor",
            "Variable used to adjust insurance price",
            "Rating factors are variables such as age, location, vehicle type, construction, credit, class code, limits, deductible, and loss history used to estimate expected loss and premium.",
            "concept",
            ["rate", "underwriting"],
        ),
        "loss cost": _entry(
            "Loss Cost",
            "Expected loss component before expense and profit load",
            "Loss cost is the expected claim cost per exposure unit. Insurers add expenses, profit, contingencies, and other loads to convert loss cost into a final rate.",
            "metric",
            ["pure premium", "rate"],
        ),
        "pure premium": _entry(
            "Pure Premium",
            "Expected loss per exposure unit",
            "Pure premium equals expected losses divided by exposure. It is the actuarial foundation of rate indications before expense and profit loads.",
            "metric",
            ["loss cost", "frequency", "severity"],
        ),
        "credibility": _entry(
            "Credibility",
            "Weight assigned to observed experience",
            "Credibility measures how much trust to place in a dataset or insured's experience. Larger, more stable experience receives more credibility in actuarial pricing.",
            "metric",
            ["experience rating", "actuarial"],
        ),
        "adverse selection": _entry(
            "Adverse Selection",
            "When higher-risk buyers are more likely to seek coverage",
            "Adverse selection occurs when applicants know more about their risk than the insurer, causing high-risk insureds to buy more coverage or lower-risk insureds to leave.",
            "concept",
            ["underwriting", "moral hazard"],
        ),
        "moral hazard": _entry(
            "Moral Hazard",
            "Behavior change because insurance exists",
            "Moral hazard occurs when an insured behaves less carefully because insurance reduces the financial consequences of loss.",
            "concept",
            ["morale hazard", "adverse selection"],
        ),
        "morale hazard": _entry(
            "Morale Hazard",
            "Carelessness due to indifference rather than intent",
            "Morale hazard is increased loss risk from indifference or carelessness, such as neglecting maintenance because the property is insured.",
            "concept",
            ["moral hazard", "underwriting"],
        ),
        "law of large numbers": _entry(
            "Law of Large Numbers",
            "Statistical principle that larger risk pools produce more stable averages",
            "The law of large numbers explains why insurers pool many independent risks: aggregate actual losses become more predictable as exposure count increases.",
            "concept",
            ["actuarial", "frequency"],
        ),
        "frequency": _entry(
            "Frequency",
            "How often claims occur",
            "Frequency measures claim count relative to exposure, such as claims per 100 vehicles or claims per 1,000 policies.",
            "metric",
            ["severity", "pure premium"],
        ),
        "severity": _entry(
            "Severity",
            "How large claims are when they occur",
            "Severity measures average or distributional claim size. Pure premium is often decomposed into frequency times severity.",
            "metric",
            ["frequency", "pure premium"],
        ),
        "trend": _entry(
            "Trend",
            "Expected change in losses, expenses, or premiums over time",
            "Trend adjusts historical insurance data to a future cost level, reflecting inflation, repair costs, medical costs, litigation, social inflation, or exposure changes.",
            "metric",
            ["loss development", "actuarial"],
        ),
        "loss development": _entry(
            "Loss Development",
            "Change in reported losses as claims mature",
            "Loss development is the pattern by which claim amounts grow or shrink over time as claims are reported, adjusted, settled, reopened, or closed.",
            "metric",
            ["loss triangle", "IBNR"],
        ),
        "loss triangle": _entry(
            "Loss Triangle",
            "Actuarial table showing losses by accident period and development age",
            "A loss triangle organizes claims by origin period and maturity. Actuaries use it to estimate ultimate losses, IBNR, and reserve adequacy.",
            "metric",
            ["loss development", "IBNR"],
        ),
        "reserve": _entry(
            "Reserve",
            "Liability estimate for unpaid claims or obligations",
            "A reserve is money set aside for future claim payments or policy obligations. Claim reserves include case reserves and IBNR.",
            "metric",
            ["case reserve", "IBNR"],
        ),
        "case reserve": _entry(
            "Case Reserve",
            "Claim-specific estimate of future payment",
            "A case reserve is the adjuster's estimate of remaining unpaid cost for a specific reported claim.",
            "metric",
            ["reserve", "claims adjuster"],
        ),
        "earned premium": _entry(
            "Earned Premium",
            "Premium recognized as coverage time passes",
            "Earned premium is the portion of written premium corresponding to expired coverage. A one-year policy earns roughly one-twelfth of annual premium each month.",
            "metric",
            ["unearned premium", "NEP"],
        ),
        "unearned premium": _entry(
            "Unearned Premium",
            "Premium for the remaining unexpired policy period",
            "Unearned premium is the portion of written premium not yet earned because future coverage remains. It is a liability until coverage is provided.",
            "metric",
            ["earned premium", "GWP"],
        ),
        "cancellation": _entry(
            "Cancellation",
            "Ending a policy before expiration",
            "Cancellation terminates a policy before its scheduled expiration date. It can be initiated by the insured or insurer subject to notice and regulatory rules.",
            "concept",
            ["nonrenewal", "lapse"],
        ),
        "nonrenewal": _entry(
            "Nonrenewal",
            "Decision not to continue coverage at the next term",
            "Nonrenewal means the insurer or insured does not continue the policy after expiration. It differs from cancellation because it occurs at renewal.",
            "concept",
            ["cancellation", "policy period"],
        ),
        "lapse": _entry(
            "Lapse",
            "Loss of coverage from nonpayment or failure to renew",
            "A lapse occurs when coverage ends, often because premium was not paid by the due date or grace period.",
            "concept",
            ["grace period", "cancellation"],
        ),
        "grace period": _entry(
            "Grace Period",
            "Extra time after premium due date before lapse",
            "A grace period gives the policyholder a limited time to pay overdue premium before coverage lapses, subject to policy and regulatory rules.",
            "concept",
            ["lapse", "premium"],
        ),
        "policy period": _entry(
            "Policy Period",
            "Effective dates when coverage is in force",
            "The policy period is the time between the effective date and expiration date shown on the declarations page.",
            "concept",
            ["declarations page", "occurrence policy"],
        ),
        "retroactive date": _entry(
            "Retroactive Date",
            "Earliest date from which claims-made incidents may be covered",
            "In claims-made policies, the retroactive date bars coverage for incidents that occurred before that date, even if the claim is made during the policy period.",
            "concept",
            ["claims-made policy", "prior acts coverage"],
        ),
        "extended reporting period": _entry(
            "Extended Reporting Period",
            "Tail coverage for reporting claims after policy expiration",
            "An extended reporting period lets the insured report claims after a claims-made policy expires for incidents that occurred after the retroactive date and before expiration.",
            "concept",
            ["claims-made policy", "tail coverage"],
        ),
        "tail coverage": _entry(
            "Tail Coverage",
            "Common name for extended reporting period coverage",
            "Tail coverage is an extended reporting period for claims-made policies. It does not cover new acts after expiration; it only extends the reporting window.",
            "concept",
            ["extended reporting period", "claims-made policy"],
        ),
        "prior acts coverage": _entry(
            "Prior Acts Coverage",
            "Claims-made coverage for acts before the current policy period",
            "Prior acts coverage protects against claims arising from professional services before the current policy period, as long as they occurred after the retroactive date.",
            "concept",
            ["retroactive date", "claims-made policy"],
        ),
        "hammer clause": _entry(
            "Hammer Clause",
            "Settlement-consent clause that can limit insurer payment",
            "A hammer clause applies when the insured refuses a settlement recommended by the insurer. The insurer may limit payment to the amount for which the claim could have settled plus defense costs to that point.",
            "concept",
            ["E&O", "D&O"],
        ),
        "self-insured retention": ACRONYMS["SIR"],
        "retention": _entry(
            "Retention",
            "Risk amount retained before insurance or reinsurance responds",
            "Retention is the amount of loss retained by the insured, insurer, or cedant before excess insurance or reinsurance pays.",
            "concept",
            ["SIR", "attachment point"],
        ),
        "umbrella policy": _entry(
            "Umbrella Policy",
            "Excess liability policy that may broaden coverage",
            "An umbrella policy sits above underlying liability policies and may provide additional limits and sometimes broader coverage than the underlying forms.",
            "concept",
            ["excess policy", "underlying insurance"],
        ),
        "excess policy": _entry(
            "Excess Policy",
            "Policy that pays after underlying limits are exhausted",
            "An excess policy provides additional limits above scheduled underlying insurance and usually follows the underlying policy terms more closely than an umbrella.",
            "concept",
            ["umbrella policy", "attachment point"],
        ),
        "underlying insurance": _entry(
            "Underlying Insurance",
            "Primary policy beneath umbrella or excess coverage",
            "Underlying insurance is the policy or layer that must respond before an umbrella, excess, or reinsurance layer pays.",
            "concept",
            ["umbrella policy", "excess policy"],
        ),
        "attachment point": _entry(
            "Attachment Point",
            "Loss level where an excess layer begins paying",
            "The attachment point is the threshold above which an excess insurance or reinsurance layer attaches and begins to pay losses.",
            "concept",
            ["retention", "XOL"],
        ),
        "treaty reinsurance": _entry(
            "Treaty Reinsurance",
            "Reinsurance covering a portfolio of policies",
            "Treaty reinsurance automatically covers a defined book or class of business under agreed terms, rather than reinsuring one risk at a time.",
            "concept",
            ["facultative reinsurance", "reinsurance"],
        ),
        "facultative reinsurance": _entry(
            "Facultative Reinsurance",
            "Reinsurance negotiated for one specific risk",
            "Facultative reinsurance is purchased risk-by-risk, often for large, unusual, or high-limit accounts that exceed treaty capacity.",
            "concept",
            ["treaty reinsurance", "reinsurance"],
        ),
        "quota share": ACRONYMS["QS"],
        "surplus share": _entry(
            "Surplus Share",
            "Proportional reinsurance based on amount above retained line",
            "Surplus share reinsurance lets the cedant retain a fixed line and cede amounts above that line proportionally, often used for property capacity management.",
            "concept",
            ["quota share", "reinsurance"],
        ),
        "excess of loss": ACRONYMS["XOL"],
        "ceding commission": _entry(
            "Ceding Commission",
            "Commission paid by reinsurer to cedant",
            "Ceding commission compensates the cedant for acquisition and underwriting expenses on business ceded to a reinsurer.",
            "metric",
            ["quota share", "reinsurance"],
        ),
        "bordereau": _entry(
            "Bordereau",
            "Detailed schedule of premiums, policies, claims, or exposures",
            "A bordereau is a periodic report exchanged between cedants, MGAs, brokers, and reinsurers listing policies, premiums, claims, or exposures.",
            "concept",
            ["MGA", "reinsurance"],
        ),
        "retrocession": _entry(
            "Retrocession",
            "Reinsurance purchased by a reinsurer",
            "Retrocession is reinsurance for reinsurers. It lets a reinsurer transfer part of its assumed risk to another reinsurer or capital provider.",
            "concept",
            ["reinsurance", "CAT"],
        ),
        "admitted carrier": _entry(
            "Admitted Carrier",
            "Insurer licensed and regulated in the state where policy is written",
            "An admitted carrier has state approval to write coverage and is subject to state rate/form rules and guaranty fund protection.",
            "concept",
            ["non-admitted carrier", "DOI"],
        ),
        "non-admitted carrier": _entry(
            "Non-Admitted Carrier",
            "Insurer not licensed as admitted in the risk's state",
            "A non-admitted carrier can write eligible surplus lines risks with more pricing and form flexibility, but policies usually lack guaranty fund protection.",
            "concept",
            ["E&S", "surplus lines"],
        ),
        "surplus lines": _entry(
            "Surplus Lines",
            "Coverage placed outside the admitted market",
            "Surplus lines coverage is used for unusual, distressed, high-hazard, or hard-to-place risks that standard admitted carriers will not write.",
            "concept",
            ["E&S", "non-admitted carrier"],
        ),
        "residual market": _entry(
            "Residual Market",
            "Insurance market of last resort",
            "Residual markets provide access to coverage when applicants cannot obtain insurance in the voluntary market, such as assigned risk auto, FAIR Plans, or workers' compensation pools.",
            "concept",
            ["FAIR", "assigned risk plan"],
        ),
        "assigned risk plan": _entry(
            "Assigned Risk Plan",
            "Mechanism assigning hard-to-place risks to insurers",
            "An assigned risk plan distributes applicants who cannot obtain coverage voluntarily among insurers required to participate, commonly in auto and workers' compensation.",
            "concept",
            ["residual market", "admitted carrier"],
        ),
        "guaranty fund": _entry(
            "Guaranty Fund",
            "State safety net for certain unpaid claims after insurer insolvency",
            "A guaranty fund pays eligible claims up to statutory limits if an admitted insurer becomes insolvent. It generally does not apply to surplus lines policies.",
            "concept",
            ["admitted carrier", "solvency"],
        ),
        "rate filing": _entry(
            "Rate Filing",
            "Regulatory filing of insurance rates or rating rules",
            "A rate filing submits proposed rates, rating factors, manuals, or supporting actuarial indications to a Department of Insurance for review or approval.",
            "concept",
            ["SERFF", "DOI"],
        ),
        "form filing": _entry(
            "Form Filing",
            "Regulatory filing of policy forms and endorsements",
            "A form filing submits policy language, endorsements, applications, or notices to regulators for approval or informational review depending on state law.",
            "concept",
            ["SERFF", "ISO"],
        ),
        "underwriting appetite": _entry(
            "Underwriting Appetite",
            "Types of risks an insurer wants to write",
            "Underwriting appetite defines preferred classes, geographies, limits, industries, risk quality, and hazards the insurer is willing to accept.",
            "concept",
            ["UW", "capacity"],
        ),
        "binding authority": _entry(
            "Binding Authority",
            "Permission to commit an insurer to coverage",
            "Binding authority lets an underwriter, agent, MGA, or delegated partner bind coverage on behalf of the insurer within specified guidelines.",
            "concept",
            ["MGA", "delegated authority"],
        ),
        "delegated authority": _entry(
            "Delegated Authority",
            "Underwriting or claims authority granted to another party",
            "Delegated authority allows an MGA, coverholder, TPA, or partner to underwrite, bind, price, or handle claims under contractually defined limits.",
            "concept",
            ["MGA", "TPA"],
        ),
        "broker of record": _entry(
            "Broker of Record",
            "Broker authorized to represent the insured for a policy or account",
            "A broker of record letter designates which broker may negotiate with insurers and receive commissions for a specific account.",
            "role",
            ["broker", "producer"],
        ),
        "producer": _entry(
            "Producer",
            "Licensed person who sells or solicits insurance",
            "Producer is a regulatory term for licensed agents or brokers who sell, solicit, or negotiate insurance.",
            "role",
            ["agent", "broker"],
        ),
        "agent": _entry(
            "Agent",
            "Insurance intermediary who usually represents an insurer",
            "An agent sells insurance and may have authority to quote or bind coverage for one or more insurers, depending on appointment and authority.",
            "role",
            ["broker", "producer"],
        ),
        "broker": _entry(
            "Broker",
            "Insurance intermediary who usually represents the buyer",
            "A broker helps the insured shop coverage across insurers and negotiate terms, especially in commercial and specialty markets.",
            "role",
            ["agent", "wholesaler"],
        ),
        "wholesaler": _entry(
            "Wholesaler",
            "Intermediary connecting retail agents with specialty markets",
            "A wholesale broker accesses E&S, specialty, or hard-to-place markets on behalf of retail agents and brokers.",
            "role",
            ["E&S", "broker"],
        ),
        "capacity": _entry(
            "Capacity",
            "Amount of insurance limit an insurer or market can provide",
            "Capacity is the maximum amount of risk or limit an insurer, reinsurer, or market is willing to deploy on a policy, layer, class, or portfolio.",
            "concept",
            ["underwriting appetite", "reinsurance"],
        ),
        "policyholder surplus": _entry(
            "Policyholder Surplus",
            "Insurer capital cushion after liabilities",
            "Policyholder surplus is admitted assets minus liabilities under statutory accounting. It supports underwriting capacity and solvency.",
            "metric",
            ["RBC", "solvency"],
        ),
        "statutory accounting": _entry(
            "Statutory Accounting",
            "Insurance regulatory accounting basis",
            "Statutory accounting is the conservative accounting framework used for insurer solvency reporting to regulators.",
            "concept",
            ["NAIC", "policyholder surplus"],
        ),
        "solvency": _entry(
            "Solvency",
            "Ability of insurer to meet policy obligations",
            "Solvency means an insurer has enough assets and capital to pay claims and obligations as they come due.",
            "concept",
            ["RBC", "policyholder surplus"],
        ),
        "float": _entry(
            "Float",
            "Funds held between premium receipt and claim payment",
            "Insurance float is money collected as premiums and held before claims are paid. Insurers invest float, making underwriting discipline and investment income both important.",
            "metric",
            ["GWP", "CR"],
        ),
        "underwriting profit": _entry(
            "Underwriting Profit",
            "Profit from insurance operations before investment income",
            "Underwriting profit occurs when earned premiums exceed losses, loss adjustment expenses, and underwriting expenses. A combined ratio below 100% indicates underwriting profit.",
            "metric",
            ["CR", "LR"],
        ),
        "commercial property": _entry(
            "Commercial Property",
            "Coverage for business buildings, contents, and income exposures",
            "Commercial property insurance covers business-owned buildings, business personal property, and often business income or extra expense from covered property losses.",
            "concept",
            ["BPP", "business income"],
        ),
        "business income": _entry(
            "Business Income",
            "Coverage for lost income from suspended operations after covered property damage",
            "Business income coverage replaces lost net income and continuing expenses when operations are suspended because of covered physical damage.",
            "concept",
            ["BI", "EE", "CBI"],
        ),
        "civil authority": _entry(
            "Civil Authority",
            "Business income coverage triggered by government access restrictions",
            "Civil authority coverage may apply when a government order prohibits access to premises because of covered damage nearby, subject to waiting periods and time limits.",
            "concept",
            ["business income", "commercial property"],
        ),
        "bailee coverage": _entry(
            "Bailee Coverage",
            "Coverage for customers' property in the insured's care",
            "Bailee coverage protects property of others while in the insured's custody, such as dry cleaners, repair shops, warehouses, or art handlers.",
            "concept",
            ["IM", "commercial property"],
        ),
        "builders risk": _entry(
            "Builders Risk",
            "Property coverage for buildings under construction",
            "Builders risk covers property during construction, renovation, or installation, including materials, temporary structures, and sometimes soft costs or delay in completion.",
            "concept",
            ["commercial property", "IM"],
        ),
        "ocean marine": _entry(
            "Ocean Marine",
            "Coverage for vessels, cargo, freight, and marine liabilities",
            "Ocean marine insurance covers seaborne property and liability exposures such as hull, cargo, protection and indemnity, and freight.",
            "concept",
            ["IM", "cargo insurance"],
        ),
        "cargo insurance": _entry(
            "Cargo Insurance",
            "Coverage for goods in transit",
            "Cargo insurance covers loss or damage to goods while being transported by truck, ship, rail, air, or multimodal routes.",
            "concept",
            ["ocean marine", "IM"],
        ),
        "crime insurance": _entry(
            "Crime Insurance",
            "Coverage for theft, fraud, and dishonesty losses",
            "Crime insurance covers losses such as employee theft, forgery, computer fraud, funds transfer fraud, and theft of money or securities.",
            "concept",
            ["employee dishonesty", "cyber liability"],
        ),
        "employee dishonesty": _entry(
            "Employee Dishonesty",
            "Crime coverage for theft by employees",
            "Employee dishonesty coverage pays for theft of money, securities, or property by employees, subject to policy terms and discovery/reporting conditions.",
            "concept",
            ["crime insurance"],
        ),
        "cyber liability": _entry(
            "Cyber Liability",
            "Coverage for data breach, network security, privacy, and cyber extortion losses",
            "Cyber liability policies can cover breach response, ransomware, business interruption, privacy liability, regulatory defense, PCI fines, and network security claims.",
            "concept",
            ["tech E&O", "claims-made policy"],
        ),
        "tech e&o": _entry(
            "Technology E&O",
            "Professional liability for technology products or services",
            "Technology E&O covers financial loss arising from technology services, software, platforms, or IT consulting failures, often paired with cyber liability.",
            "concept",
            ["E&O", "cyber liability"],
        ),
        "fiduciary liability": _entry(
            "Fiduciary Liability",
            "Coverage for benefit-plan fiduciary claims",
            "Fiduciary liability covers claims alleging mismanagement of employee benefit plans, such as ERISA fiduciary breaches.",
            "concept",
            ["D&O", "EPL"],
        ),
        "liquor liability": _entry(
            "Liquor Liability",
            "Coverage for alcohol-related injury or damage claims",
            "Liquor liability covers businesses that manufacture, sell, serve, or furnish alcohol against claims arising from intoxicated patrons.",
            "concept",
            ["GL", "exclusion"],
        ),
        "products completed operations": _entry(
            "Products-Completed Operations",
            "CGL coverage for products or completed work after possession or completion",
            "Products-completed operations coverage applies to bodily injury or property damage arising from the insured's products or completed work away from premises.",
            "concept",
            ["CGL", "aggregate limit"],
        ),
        "premises operations": _entry(
            "Premises Operations",
            "Liability arising from ongoing business operations or premises",
            "Premises operations coverage applies to injury or damage arising from ongoing operations or conditions at the insured's premises.",
            "concept",
            ["CGL", "third party"],
        ),
        "garagekeepers": _entry(
            "Garagekeepers",
            "Coverage for customers' autos in a garage business's care",
            "Garagekeepers coverage protects customers' vehicles while in the care, custody, or control of an auto repair shop, dealership, valet, or parking operation.",
            "concept",
            ["bailee coverage", "BAP"],
        ),
        "professional liability": _entry(
            "Professional Liability",
            "Liability for professional errors or failure to perform services",
            "Professional liability covers alleged negligence, errors, omissions, or failure to deliver professional services that cause financial harm.",
            "concept",
            ["E&O", "claims-made policy"],
        ),
        "malpractice": _entry(
            "Malpractice",
            "Professional liability for healthcare or legal professionals",
            "Malpractice insurance is professional liability coverage for doctors, lawyers, and other professionals whose errors can cause financial or bodily harm.",
            "concept",
            ["professional liability", "claims-made policy"],
        ),
        "beneficiary": _entry(
            "Beneficiary",
            "Person or entity designated to receive policy benefits",
            "A beneficiary receives benefits from a life insurance policy, annuity, retirement account, or similar contract when conditions are met.",
            "concept",
            ["L&H", "term life"],
        ),
        "term life": _entry(
            "Term Life",
            "Life insurance for a specified period",
            "Term life insurance pays a death benefit if the insured dies during the policy term. It has no cash value and is usually cheaper than permanent life insurance.",
            "concept",
            ["whole life", "beneficiary"],
        ),
        "whole life": _entry(
            "Whole Life",
            "Permanent life insurance with guaranteed cash value features",
            "Whole life provides lifetime coverage if premiums are paid and builds cash value under guaranteed assumptions.",
            "concept",
            ["term life", "cash value"],
        ),
        "universal life": _entry(
            "Universal Life",
            "Flexible permanent life insurance",
            "Universal life combines life insurance protection with flexible premiums and cash value credited according to policy terms.",
            "concept",
            ["whole life", "cash value"],
        ),
        "cash value": _entry(
            "Cash Value",
            "Savings-like value inside permanent life insurance",
            "Cash value is the accumulated policy value that may be borrowed against, withdrawn, or surrendered, subject to policy terms and charges.",
            "concept",
            ["whole life", "surrender charge"],
        ),
        "surrender charge": _entry(
            "Surrender Charge",
            "Fee for canceling certain life or annuity contracts early",
            "A surrender charge is deducted if the policyholder withdraws or cancels cash-value life insurance or annuities during the surrender period.",
            "concept",
            ["cash value", "annuity"],
        ),
        "annuity": _entry(
            "Annuity",
            "Insurance contract designed to provide income or accumulated value",
            "An annuity is a contract that can accumulate funds tax-deferred and later provide periodic payments, often used for retirement income planning.",
            "concept",
            ["L&H", "surrender charge"],
        ),
        "elimination period": _entry(
            "Elimination Period",
            "Waiting period before disability or long-term-care benefits begin",
            "An elimination period is the time the insured must satisfy after a covered disability or care need before benefits are payable.",
            "concept",
            ["DI", "LTC"],
        ),
    }
)


ALIASES: dict[str, str] = {
    "dec page": "declarations page",
    "declaration page": "declarations page",
    "policy limit": "coverage limit",
    "limits of insurance": "coverage limit",
    "liability limit": "coverage limit",
    "deductibles": "deductible",
    "premiums": "premium",
    "riders": "rider",
    "endorsements": "endorsement",
    "exclusions": "exclusion",
    "claim adjuster": "claims adjuster",
    "claim handler": "claims adjuster",
    "loss adjuster": "claims adjuster",
    "first notice": "notice of loss",
    "first notice of loss": "FNOL",
    "med pay": "MEDPAY",
    "medical payments": "MEDPAY",
    "uninsured motorist property damage": "UMPD",
    "personal auto": "PAP",
    "business auto": "BAP",
    "commercial auto": "BAP",
    "hired non owned auto": "HNOA",
    "hired and non owned auto": "HNOA",
    "actual cash value coverage": "actual cash value",
    "replacement cost": "replacement cost value",
    "replacement cost coverage": "replacement cost value",
    "additional living expenses": "additional living expense",
    "sewer backup": "water backup",
    "wind hail": "wind hail deductible",
    "wind/hail": "wind hail deductible",
    "quake": "EQ",
    "earthquake": "EQ",
    "flood": "flood insurance",
    "tail": "tail coverage",
    "tail policy": "tail coverage",
    "erp": "extended reporting period",
    "prior acts": "prior acts coverage",
    "sir": "SIR",
    "self insured retention": "self-insured retention",
    "epli": "EPLI",
    "employment practices liability insurance": "EPLI",
    "professional liability insurance": "professional liability",
    "errors and omissions": "E&O",
    "errors & omissions": "E&O",
    "directors and officers": "D&O",
    "directors & officers": "D&O",
    "business owners policy": "BOP",
    "businessowners policy": "BOP",
    "commercial package": "CPP",
    "commercial package policy": "CPP",
    "business personal property": "BPP",
    "extra expense": "EE",
    "equipment breakdown": "EB",
    "contingent business interruption": "CBI",
    "certificate holder": "certificate of insurance",
    "additional insured endorsement": "additional insured",
    "waiver of subro": "waiver of subrogation",
    "primary noncontributory": "primary and noncontributory",
    "pnc": "primary and noncontributory",
    "proof of claim": "proof of loss",
    "ror": "reservation of rights",
    "loss run": "loss history",
    "loss runs": "loss history",
    "no claims bonus": "NCB",
    "no claims discount": "NCD",
    "assigned risk": "assigned risk plan",
    "fair plan": "FAIR",
    "surplus line": "surplus lines",
    "non admitted": "non-admitted carrier",
    "nonadmitted": "non-admitted carrier",
    "admitted insurer": "admitted carrier",
    "non admitted insurer": "non-admitted carrier",
    "department of insurance": "DOI",
    "insurance department": "DOI",
    "a.m. best": "AMB",
    "am best": "AMB",
    "quota share reinsurance": "quota share",
    "excess of loss reinsurance": "excess of loss",
    "fac": "facultative reinsurance",
    "facultative": "facultative reinsurance",
    "treaty": "treaty reinsurance",
    "risk based capital": "RBC",
    "health savings account": "HSA",
    "high deductible health plan": "HDHP",
    "long term care": "LTC",
    "disability income": "DI",
    "accidental death": "AD&D",
}


def knowledge_base_size() -> int:
    """Return the number of distinct display terms in the deterministic glossary."""
    return len({entry.term for entry in [*ACRONYMS.values(), *TERMS.values()]})


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


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match glossary keys as standalone phrases, not inside unrelated words."""
    pattern = rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _contains_acronym_token(text: str, acronym: str) -> bool:
    """Match acronyms as standalone tokens, not inside words like liability/tail."""
    return _contains_phrase(text, acronym)


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

    # 2. Resolve common aliases and alternate spellings
    for alias, target in ALIASES.items():
        if _contains_phrase(q_lower, alias):
            entry = ACRONYMS.get(target) or TERMS.get(target.lower()) or TERMS.get(target)
            if entry and entry not in [r[1] for r in results]:
                results.append((90, entry))

    # 3. Check if the question contains a known term literally
    for key, entry in ACRONYMS.items():
        if _contains_acronym_token(q_lower, key) and entry not in [r[1] for r in results]:
            results.append((80, entry))

    # 4. Check TERMS dict
    for key, entry in TERMS.items():
        if _contains_phrase(q_lower, key):
            results.append((70, entry))
        elif _contains_phrase(q_lower, entry.term):
            results.append((65, entry))

    # 5. Keyword scan of short/detail text for definition questions
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
