# SFT Adapter Spot Check

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Adapter dir: `models/qwen7b-insurerag-lora-retrieval`
- Samples: `8`

## Summary

| variant | overall_f1 | answerable_f1 | unsupported_abstain_rate |
| --- | ---: | ---: | ---: |
| base | 0.2168 | 0.3035 | 0.7500 |
| adapter | 0.7443 | 0.4886 | 1.0000 |

## Samples

### Sample 1: sft::md_homeowners_insurance_guide::p0007::c002::0000

- Answerable: `True`
- Source: `md_homeowners_insurance_guide.pdf#page=7`
- Question: What does the evidence say about exclusions?

**Reference**

The cited page supports a **Exclusion** explanation. In practical terms, the reader should rely on the stated policy language: An open perils policy, or all-risk policy, provides coverage for all causes of loss unless the specific cause of loss is excluded from coverage under the policy. Source: md_homeowners_insurance_guide.pdf, Page 7

**Base**

The evidence does not explicitly mention exclusions in the context of named peril policies or open perils (all-risk) policies. However, it implies that named peril policies cover most, but not all, of the common causes of damage, suggesting there are some exclusions. In contrast, open perils policies provide coverage for all causes of loss unless the specific cause of loss is excluded from coverage under the policy.

This means that while named peril policies have a list of specific perils they cover, other causes of loss are not covered unless they are included. On the other hand, open perils policies cover almost everything but have a shorter list of exclusions.

For a more detailed understanding of exclusions, one would need to refer to the specific policy documents or the insurer’s guidelines, which are not

**Adapter**

The cited page supports a **Exclusion** explanation. In practical terms, the reader should rely on the stated policy language: 5 Maryland Insurance Administration - 800-492-6116 - www.insurance.maryland.gov A C O N S U M E R G U I D E TO H O M E O W N E R S I N S U R A N C E... Source: md_homeowners_insurance_guide.pdf, Page 7

### Sample 2: sft::md_auto_insurance_guide::p0014::c001::0004

- Answerable: `True`
- Source: `md_auto_insurance_guide.pdf#page=14`
- Question: What does the evidence say about exclusions?

**Reference**

The cited page supports a **Exclusion** explanation. In practical terms, the reader should rely on the stated policy language: In other words, if the claim for bodily injury is covered by your policy, the insurer will pay the claim up to the dollar limits of the coverage you purchased; you would only need to pay if the claim exceeds the policy... Source: md_auto_insurance_guide.pdf, Page 14

**Base**

The provided evidence does not specifically mention exclusions. It discusses how claims for bodily injury and property damage are handled within the policy limits but does not address what is excluded from coverage. To understand exclusions in an insurance policy, one would typically need to refer to the specific terms and conditions outlined in the policy document, which often include a section detailing what is not covered. For more detailed information on exclusions, you would need to consult the relevant part of the insurance policy or the document provided (md_auto_insurance_guide.pdf). Source: md_auto_insurance_guide.pdf#page=14

**Adapter**

The cited page supports a **Exclusion** explanation. In practical terms, the reader should rely on the stated policy language: 10 Maryland Insurance Administration - 800-492-6116 - www.insurance.maryland.gov A C O N S U M E R G U I D E TO A U TO I N S U R A N C E - Property Damage Liability Coverage (Mandatory): Property... Source: md_auto_insurance_guide.pdf, Page 14

### Sample 3: sft::ca_doi_auto_terms::p0001::c015::0008

- Answerable: `True`
- Source: `https://www.insurance.ca.gov/01-consumers/105-type/95-guides/01-auto/autoterms.cfm#page=1`
- Question: What does the evidence say about exclusions?

**Reference**

The cited page supports a **Exclusion** explanation. In practical terms, the reader should rely on the stated policy language: Exclusion - A contractual provision in an insurance policy that denies or restricts coverage for certain perils, persons, property, or locations. Source: https://www.insurance.ca.gov/01-consumers/105-type/95-guides/01-auto/autoterms.cfm, Page 1

**Base**

According to the evidence provided, exclusions are "contractual provisions in an insurance policy that deny or restrict coverage for certain perils, persons, property, or locations." This means that exclusions specify what the insurance policy will not cover, effectively limiting the scope of the insurer's obligations under the policy.

Source: https://www.insurance.ca.gov/01-consumers/105-type/95-guides/01-auto/autoterms.cfm#page=1

**Adapter**

The cited page supports a **Exclusion** explanation. In practical terms, the reader should rely on the stated policy language: Uninsured/Underinsured Motorist (UIM): This coverage will protect you if someone driving without insurance or with insufficient insurance causes damage to your property or injures you or your passenger(s). Source: https://www.insurance.ca.gov/01-consumers/105-type/95-guides/01-auto/autoterms.cfm, Page 1

### Sample 4: sft::nc_homeowners_guide_archive::p0015::c001::0012

- Answerable: `True`
- Source: `nc_homeowners_guide_archive.pdf#page=15`
- Question: What does the evidence say about exclusions?

**Reference**

The cited page supports a **Exclusion** explanation. In practical terms, the reader should rely on the stated policy language: If your policy contains this endorsement and the Coverage A limit stated in your policy is not enough to cover the loss to your dwelling, the insurance company will pay the difference, subject to a specified maximum. Source: nc_homeowners_guide_archive.pdf, Page 15

**Base**

The evidence provided does not explicitly mention exclusions related to Guaranteed Replacement Cost Coverage. However, it does indicate that there might be exclusions or conditions associated with this type of coverage. The relevant part states:

"Check with your insurance agent to determine if an additional premium is required and what exclusions or conditions may apply."

This suggests that while Guaranteed Replacement Cost Coverage provides comprehensive protection, there could be specific exclusions or conditions that limit its application. These would need to be discussed with your insurance agent to understand fully. (Source: nc_homeowners_guide_archive.pdf#page=15)

**Adapter**

The cited page supports a **Exclusion** explanation. In practical terms, the reader should rely on the stated policy language: Cancellation Termination of an insurance policy by the insurance company or policyholder before a policy expires. Source: nc_homeowners_guide_archive.pdf, Page 15

### Sample 5: sft::unsupported::0000

- Answerable: `False`
- Source: `https://content.naic.org/consumer_glossary#page=1`
- Question: Does this policy include cyber liability coverage?

**Reference**

The provided evidence does not support that answer. I cannot confirm this coverage, deductible, or sublimit from the supplied page. Source: insufficient evidence

**Base**

The provided evidence does not specify whether the Credit Placed Insurance includes cyber liability coverage. The definition given pertains to the type of insurance purchased by creditors to protect their interests in collateral, such as homes, mobile homes, and vehicles. It does not mention cyber liability or any digital or technology-related risks.

To determine if a policy includes cyber liability coverage, you would need to review the specific terms and conditions of the policy or consult with the insurer. Cyber liability coverage typically protects against losses related to data breaches, cyber attacks, and other digital risks.

For more detailed information on cyber liability coverage, you may refer to the NAIC (National Association of Insurance Commissioners) consumer glossary or contact your insurance provider directly.

**Adapter**

The provided evidence does not support that answer. I cannot confirm this coverage, deductible, or sublimit from the supplied page. Source: insufficient evidence

### Sample 6: sft::unsupported::0003

- Answerable: `False`
- Source: `https://www.tdi.texas.gov/pubs/consumer/cb020.html#page=1`
- Question: What is the cyber liability sublimit?

**Reference**

The provided evidence does not support that answer. I cannot confirm this coverage, deductible, or sublimit from the supplied page. Source: insufficient evidence

**Base**

The term "cyber liability sublimit" is not mentioned in the provided evidence. The given text focuses on what to do after a car accident, including reporting to the police, exchanging information, and dealing with insurance claims. There is no information related to cyber liability or sublimits in this context.

If you need information about cyber liability sublimits, it would typically be found in a commercial insurance policy or a business owner's policy (BOP) that includes cyber liability coverage. Cyber liability sublimits refer to the specific amount of coverage within a policy dedicated to addressing cyber-related risks such as data breaches, cyber extortion, and network security failures. However, this detail is beyond the scope of the provided evidence.

**Adapter**

The provided evidence does not support that answer. I cannot confirm this coverage, deductible, or sublimit from the supplied page. Source: insufficient evidence

### Sample 7: sft::unsupported::0006

- Answerable: `False`
- Source: `md_auto_insurance_guide.pdf#page=10`
- Question: Does this page confirm umbrella liability coverage?

**Reference**

The provided evidence does not support that answer. I cannot confirm this coverage, deductible, or sublimit from the supplied page. Source: insufficient evidence

**Base**

The provided evidence does not mention umbrella liability coverage. The text discusses the right to request a recheck of your credit history by your insurer and the ability to review and correct errors in your credit report. It also mentions the availability of free credit reports from the Federal Trade Commission. Therefore, based on the given evidence, there is no confirmation regarding umbrella liability coverage.

For clarification, umbrella liability coverage is an additional layer of liability insurance that provides protection beyond the limits of your primary auto, home, or rental insurance policies. It can help cover damages and legal costs if you are found liable in a lawsuit where your primary insurance limits are exceeded.

**Adapter**

The provided evidence does not support that answer. I cannot confirm this coverage, deductible, or sublimit from the supplied page. Source: insufficient evidence

### Sample 8: sft::unsupported::0009

- Answerable: `False`
- Source: `https://www.insurance.wa.gov/consumers-insurance-glossary#page=1`
- Question: Does this evidence confirm employment practices liability coverage?

**Reference**

The provided evidence does not support that answer. I cannot confirm this coverage, deductible, or sublimit from the supplied page. Source: insufficient evidence

**Base**

The provided evidence does not confirm anything about employment practices liability coverage. The information given pertains to long-term care insurance benefits, eligibility criteria, and definitions such as "activities of daily living" and "actual cash value," which are related to long-term care insurance and property insurance respectively. Employment practices liability coverage is not mentioned in the given evidence. Therefore, based on the provided information, it cannot be confirmed whether employment practices liability coverage is applicable or relevant here.

**Adapter**

The provided evidence does not support that answer. I cannot confirm this coverage, deductible, or sublimit from the supplied page. Source: insufficient evidence
