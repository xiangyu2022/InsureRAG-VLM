---
language:
- en
license: cc-by-sa-4.0
configs:
- config_name: case-description
  data_files:
  - split: train
    path: case-description/train-*
- config_name: clinical-guidelines
  data_files:
  - split: train
    path: clinical-guidelines/train-*
- config_name: contract-coverage-rule-medical-policy
  data_files:
  - split: train
    path: contract-coverage-rule-medical-policy/train-*
- config_name: legal
  data_files:
  - split: train
    path: legal/train-*
- config_name: opinion-policy-summary
  data_files:
  - split: train
    path: opinion-policy-summary/train-*
- config_name: regulatory-guidance
  data_files:
  - split: train
    path: regulatory-guidance/train-*
dataset_info:
- config_name: case-description
  features:
  - name: text
    dtype: string
  - name: tags
    sequence: string
  - name: date_accessed
    dtype: string
  - name: source_url
    dtype: string
  - name: source_md5
    dtype: string
  - name: relative_path
    dtype: string
  - name: decision
    dtype: string
  - name: appeal_type
    dtype: string
  splits:
  - name: train
    num_bytes: 1694360988
    num_examples: 990065
  download_size: 402963667
  dataset_size: 1694360988
- config_name: clinical-guidelines
  features:
  - name: text
    dtype: string
  - name: tags
    sequence: string
  - name: date_accessed
    dtype: string
  - name: source_url
    dtype: string
  - name: source_md5
    dtype: string
  - name: relative_path
    dtype: string
  splits:
  - name: train
    num_bytes: 562276777
    num_examples: 40110
  download_size: 266629038
  dataset_size: 562276777
- config_name: contract-coverage-rule-medical-policy
  features:
  - name: text
    dtype: string
  - name: tags
    sequence: string
  - name: date_accessed
    dtype: string
  - name: source_url
    dtype: string
  - name: source_md5
    dtype: string
  - name: relative_path
    dtype: string
  splits:
  - name: train
    num_bytes: 30225716
    num_examples: 3661
  download_size: 12426008
  dataset_size: 30225716
- config_name: legal
  features:
  - name: text
    dtype: string
  - name: tags
    sequence: string
  - name: date_accessed
    dtype: string
  - name: source_url
    dtype: string
  - name: source_md5
    dtype: string
  - name: relative_path
    dtype: string
  splits:
  - name: train
    num_bytes: 596329842
    num_examples: 1348
  download_size: 238155541
  dataset_size: 596329842
- config_name: opinion-policy-summary
  features:
  - name: text
    dtype: string
  - name: tags
    sequence: string
  - name: date_accessed
    dtype: string
  - name: source_url
    dtype: string
  - name: source_md5
    dtype: string
  - name: relative_path
    dtype: string
  splits:
  - name: train
    num_bytes: 131693040
    num_examples: 2081
  download_size: 62275853
  dataset_size: 131693040
- config_name: regulatory-guidance
  features:
  - name: text
    dtype: string
  - name: tags
    sequence: string
  - name: date_accessed
    dtype: string
  - name: source_url
    dtype: string
  - name: source_md5
    dtype: string
  - name: relative_path
    dtype: string
  splits:
  - name: train
    num_bytes: 38848315
    num_examples: 1110
  download_size: 17339277
  dataset_size: 38848315
---
## HICRIC: A Dataset of Law, Policy, and Regulatory Guidance for Health Insurance Coverage Understanding

**H**ealth **I**nsurance **C**overage **R**ules **I**nterpretation **C**orpus is a collection of unannotated text curated to support
applications that require understanding of U.S. health insurance coverage rules.

It consists of a small, unlabeled corpus of authoritative and non-authoritative text related to law, insurance contracts, and medicine. The corpus is intended for use in pretraining language models and, independently, for use as a standalone knowledge base in retrieval pipelines.

This data repository houses the processed, unlabeled data referenced in a more general work documented here: [https://github.com/TPAFS/hicric](https://github.com/TPAFS/hicric).

### Redistribution

Please consult the licenses for all source data for yourself if you plan to redistribute any of it. To the best of our knowledge, our redistributions abide by all such licenses.

### Risks

We believe there are numerous risks associated with our released data, which we've done our best to mitigate. Our main concerns involve:

- Potential for Propagation of Bias
- Potential for Misuse

Please see our forthcoming paper for a thorough discussion of these perceived risks.

### Limitations

There are many limitations associated with our released data, and our advice is to consider and weigh these limitations carefully to
inform resonsible and effective use. The main categories of limitation are:

- Task Shortcomings
- Simplicity of the Benchmark
- Corpus Deficiencies

Please see our forthcoming paper for a thorough discussion of these perceived risks.

## Dataset Breakdown

Each document in our dataset comes equipped with a set of plain-text _tags_. In constructing the
data we formulated a particular privileged set of partitioning tags: these are a set of tags with
the property that each document in the dataset is associated with exactly one tag in the set,
and none of the tags are unused.

The tags are the following:


- **legal**

- **regulatory-guidance**

- **contract-coverage-rule-medical-policy**

-  **opinion-policy-summary**

-  **case-description**

-  **clinical-guidelines**

In addition to this set of partitioning tags, we intoduce another privileged tag:

- **kb** This tag indicates that a document is suitable for use in a knowledge base.

    This is a subjective determination, but the intent is to label text that comes from a reputable,
    _definitive_ source. For example, a summary of Medicaid rules as stated by an employee of HHS
    during congressional testimony would not be labeled with the `kb` tag, because such testimony is
    not the definitive source for the ground truth of such rules. On the other hand, federal
    law describing those same rules would be labeled with the `kb` tag.


A high level characterization of the distribution of text in our corpus in terms of these 
privileged tags is shown in the table below.

| Category | Num Documents | Words | Chars | Size (GB) |                                                                                                                                     
| -------- | ------------- | ----- | ----- | --------- |                                                                                                                                     
| All Partition Parts | 8,310 | 417,617,646 | 2,699,256,987 | 2.81 |                                                                                                                                     
| kb | 1,434 | 170,717,368 | 1,120,961,295 | 1.13 |                                                                                                                                          
| legal | 335 | 92,357,802 | 596,044,008 | 0.60 |                                                                                                                                            
| regulatory-guidance | 1,110 | 5,536,585 | 38,607,587 | 0.04 |                                                                                                                              
| contract-coverage-rule-medical-policy | 7 | 196,156,813 | 1,228,184,524 | 1.31 |                                                                                                           
| opinion-policy-summary | 2,094 | 19,462,399 | 133,049,956 | 0.14 |                                                                                                                         
| case-description | 2,629 | 214,267,074 | 1,351,074,791 | 1.45 |
| clinical-guidelines | 2,150 | 81,955,020 | 553,041,990 | 0.56 |



## Contact
For questions or comments, please reach out to `info@persius.org`.