import ast
import csv
import hashlib
import html
import json
import math
import random
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from .data import PageDocument, load_documents
from .retriever import load_index


PUBLIC_DATA_SOURCES = {
    "cuad": {
        "name": "CUAD",
        "license": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/master_clauses.csv",
        "local_path": "data/00_raw/external/cuad/master_clauses.csv",
        "description": "Clause extraction and contract QA supervision from The Atticus Project.",
    },
    "acord": {
        "name": "ACORD",
        "license": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/theatticusproject/acord/resolve/main/ACORD%20Dataset%20%26%20ReadMe.zip",
        "local_path": "data/00_raw/external/acord/acord.zip",
        "description": "BEIR-style clause retrieval benchmark from The Atticus Project.",
    },
    "insuranceqa": {
        "name": "InsuranceQA",
        "license": "research purpose only",
        "url": "https://github.com/shuzi/insuranceQA",
        "local_path": "data/00_raw/external/insuranceqa",
        "description": "Insurance question language style data. Manifest-only by default due license restrictions.",
    },
    "public_docs": {
        "name": "Public Insurance PDF Samples",
        "license": "public web documents; verify source terms before redistribution",
        "url": "multiple",
        "local_path": "data/00_raw/external/public_docs",
        "description": "Real public insurance PDFs from state departments for page-image preprocessing and eval.",
        "documents": [
            {
                "name": "Maryland Homeowners Insurance Declarations Page Example",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/Consumer/Documents/publications/understandinghodeclarationspage.pdf",
                "local_name": "md_homeowners_declarations_page.pdf",
                "description": "Consumer-facing explanation of homeowners declarations page fields.",
            },
            {
                "name": "Maryland Consumer Guide to Auto Insurance",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/Consumer/Documents/publications/autoinsuranceguide.pdf",
                "local_name": "md_auto_insurance_guide.pdf",
                "description": "Public consumer guide with auto coverage, premium, and claims concepts.",
            },
            {
                "name": "NC Consumer Guide to Auto Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/media/4197/download?language=en",
                "local_name": "nc_auto_guide.pdf",
                "description": "Scanned consumer guide — demonstrates visual-only retrieval use case.",
            },
            {
                "name": "NC Consumer Guide to Home Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/media/4199/download?language=en",
                "local_name": "nc_home_guide.pdf",
                "description": "Scanned homeowners insurance guide.",
            },
            {
                "name": "NC Consumer Guide to Life Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/media/4203/download?language=en",
                "local_name": "nc_life_guide.pdf",
                "description": "Scanned life insurance consumer guide.",
            },
            {
                "name": "NC Consumer Guide to Renters Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/media/4201/download?language=en",
                "local_name": "nc_renters_guide.pdf",
                "description": "Scanned renters insurance consumer guide.",
            },
            {
                "name": "NC Consumer Guide to Homeowners Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/media/4200/download?language=en",
                "local_name": "nc_homeowners_guide.pdf",
                "description": "Scanned homeowners insurance consumer guide.",
            },
            {
                "name": "NC Consumer Guide to Flood Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/media/4205/download?language=en",
                "local_name": "nc_flood_guide.pdf",
                "description": "Scanned flood insurance consumer guide.",
            },
            {
                "name": "NC Consumer Guide to Health Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/media/4202/download?language=en",
                "local_name": "nc_health_guide.pdf",
                "description": "Scanned health insurance consumer guide.",
            },
            {
                "name": "NC Consumer Guide to Business Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/media/4198/download?language=en",
                "local_name": "nc_business_guide.pdf",
                "description": "Scanned business/commercial insurance consumer guide.",
            },
            {
                "name": "NC Consumers Guide to Disability Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/consumers-guide-disability-insurance/open",
                "local_name": "nc_disability_insurance_guide.pdf",
                "description": "Consumer guide for disability income insurance.",
            },
            {
                "name": "NC Consumers Guide to Travel Insurance",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/consumers-guide-travel-insurance/open",
                "local_name": "nc_travel_insurance_guide.pdf",
                "description": "Consumer guide for travel insurance.",
            },
            {
                "name": "Maryland Consumer Guide to Homeowners Insurance",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/Consumer/Documents/publications/homeownersinsguide.pdf",
                "local_name": "md_homeowners_insurance_guide.pdf",
                "description": "Consumer guide to homeowners insurance coverage, claims, and rates.",
            },
            {
                "name": "Maryland Homeowners Disclosure Notice",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/Consumer/Documents/HomeownersDisclosureNotice.pdf",
                "local_name": "md_homeowners_disclosure_notice.pdf",
                "description": "Required homeowners insurance disclosure notice.",
            },
            {
                "name": "Maryland Uninsured Motorist Claims Advisory",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/Consumer/Documents/publications/ConsumerAdvisory-What-You-Need-To-Know-About-Uninsured-Motorist-Claims.pdf",
                "local_name": "md_uninsured_motorist_claims_advisory.pdf",
                "description": "Consumer advisory on uninsured motorist claims.",
            },
            {
                "name": "Maryland College-Bound Child Insurance Advisory",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/Consumer/Documents/publications/Consumer-Advisory-Hitting-the-Books-on-Insurance-for-Your-College-Bound-Child.pdf",
                "local_name": "md_college_bound_child_insurance_advisory.pdf",
                "description": "Consumer advisory about property, auto, and health insurance for college-bound students.",
            },
            {
                "name": "Maryland Licensed Drivers in Household Advisory",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/Consumer/Documents/publicnew/ConsumerAdvisory-Licensed-Drivers-Household.pdf",
                "local_name": "md_licensed_drivers_household_advisory.pdf",
                "description": "Consumer advisory on undisclosed licensed drivers and auto coverage.",
            },
            {
                "name": "Maryland Insurance Coverage Review Tips Advisory",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/Consumer/Documents/publicnew/ConsumerAdvisory-Insurance-Coverage-Tips.pdf",
                "local_name": "md_insurance_coverage_tips_advisory.pdf",
                "description": "Consumer advisory with tips for reviewing insurance coverage.",
            },
            {
                "name": "NC Consumers Guide to Homeowners Insurance Archive",
                "source": "North Carolina Department of Insurance",
                "url": "https://www.ncdoi.gov/consumers-guide-homeowners-insurance/open",
                "local_name": "nc_homeowners_guide_archive.pdf",
                "description": "Archived consumers guide to homeowners insurance.",
            },
            {
                "name": "Maryland Homeowners Insurance Still Important Advisory",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/Consumer/Documents/publicnew/ConsumerAlert-Homeowners-3152022.pdf",
                "local_name": "md_homeowners_still_important_advisory.pdf",
                "description": "Consumer advisory about maintaining homeowners insurance after paying off a mortgage.",
            },
        ],
    },
    "real_domain_mix": {
        "name": "Real Insurance Domain Mix",
        "license": "mixed public web documents; verify source terms before redistribution",
        "url": "multiple",
        "local_path": "data/00_raw/external/real_domain_mix",
        "description": (
            "Real insurance-domain text sources aligned to domain datasets, web pages, "
            "news, reports/white papers, social/forum pointers, and conversation/FAQ data."
        ),
        "documents": [
            {
                "category": "domain_specific_text_dataset",
                "name": "CUAD master clauses",
                "source": "The Atticus Project / Hugging Face",
                "url": "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/master_clauses.csv",
                "local_name": "domain_text/cuad_master_clauses.csv",
                "description": "Contract clause extraction and QA supervision dataset.",
            },
            {
                "category": "domain_specific_text_dataset",
                "name": "ACORD dataset and readme",
                "source": "The Atticus Project / Hugging Face",
                "url": "https://huggingface.co/datasets/theatticusproject/acord/resolve/main/ACORD%20Dataset%20%26%20ReadMe.zip",
                "local_name": "domain_text/acord.zip",
                "description": "Clause retrieval benchmark for insurance/reinsurance contracts.",
            },
            {
                "category": "domain_web_content",
                "name": "Maryland insurance consumer publications",
                "source": "Maryland Insurance Administration",
                "url": "https://insurance.maryland.gov/consumer/pages/consumerpublications.aspx",
                "local_name": "web/md_consumer_publications.txt",
                "description": "Consumer insurance publication index with auto, homeowners, health, life, travel, and fraud topics.",
            },
            {
                "category": "domain_web_content",
                "name": "NAIC transparency and readability topic",
                "source": "National Association of Insurance Commissioners",
                "url": "https://content.naic.org/insurance-topics/transparency-and-readability-of-consumer-information",
                "local_name": "web/naic_transparency_readability.txt",
                "description": "Regulatory topic page about consumer access to readable personal-lines policy information.",
            },
            {
                "category": "domain_news_article",
                "name": "Triple-I severe convective storms insured losses news",
                "source": "Insurance Information Institute",
                "url": "https://www.iii.org/press-release/triple-i-severe-convective-storms-generate-more-than-50b-in-insured-losses-for-third-consecutive-year-041326",
                "local_name": "news/iii_severe_convective_storms_2026.txt",
                "description": "News release on 2025 severe convective storm insured losses.",
            },
            {
                "category": "domain_news_article",
                "name": "Triple-I flood insurance state of the risk news",
                "source": "Insurance Information Institute",
                "url": "https://www.iii.org/press-release/record-2025-us-flooding-highlights-urgent-need-for-flood-insurance-and-resilience-measures-triple-is-new-issues-brief-explains-021626",
                "local_name": "news/iii_flood_insurance_state_of_risk_2026.txt",
                "description": "News release summarizing recent flooding trends and flood insurance need.",
            },
            {
                "category": "industry_report_whitepaper",
                "name": "NAIC Artificial Intelligence and Insurance Regulation",
                "source": "NAIC Journal of Insurance Regulation",
                "url": "https://content.naic.org/research/jir/artificial-intelligence-and-insurance-regulation",
                "local_name": "reports/naic_ai_insurance_regulation.txt",
                "description": "Research article page on AI use, model bulletins, and insurance regulation.",
            },
            {
                "category": "industry_report_whitepaper",
                "name": "NAIC Trial by Fire: Reimagining Wildfire Insurance in California",
                "source": "NAIC Journal of Insurance Regulation",
                "url": "https://content.naic.org/research/jir/trial-fire-reimagining-wildfire-insurance-california",
                "local_name": "reports/naic_wildfire_insurance_california.txt",
                "description": "Research article page on a public-private wildfire insurance/reinsurance mechanism.",
            },
            {
                "category": "conversation_qa_data",
                "name": "NAIC health reform frequently asked questions",
                "source": "National Association of Insurance Commissioners",
                "url": "https://content.naic.org/index.php/index_health_reform_faq.htm",
                "local_name": "conversations/naic_health_reform_faq.txt",
                "description": "Consumer and employer FAQ page about health reform and state insurance regulation.",
            },
            {
                "category": "conversation_qa_data",
                "name": "InsuranceQA repository pointer",
                "source": "GitHub / shuzi",
                "url": "https://github.com/shuzi/insuranceQA",
                "local_name": "conversations/insuranceqa_source_manifest.txt",
                "description": "Manifest pointer to InsuranceQA question-answer language data; local cloning is left explicit because terms are research-use oriented.",
                "manifest_only": True,
            },
            {
                "category": "social_media_data",
                "name": "Reddit insurance discussions source pointer",
                "source": "Reddit",
                "url": "https://www.reddit.com/r/Insurance/",
                "local_name": "social/reddit_insurance_source_manifest.txt",
                "description": "Manifest pointer for opt-in/API-based collection of insurance discussion posts; importer does not scrape social platforms by default.",
                "manifest_only": True,
            },
        ],
    },
}


@dataclass
class QAResult:
    qa_path: Path
    hard_negatives_path: Path
    qa_count: int
    hard_negative_count: int
    splits_path: Optional[Path] = None


@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    evaluated_count: int


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class _HTMLTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
        "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main", "nav",
        "ol", "p", "section", "table", "td", "th", "tr", "ul",
    }
    _SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = html.unescape(data).strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        text = " ".join(part.strip() for part in self._parts)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def _html_to_text(content: bytes) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(content.decode("utf-8", errors="ignore"))
    return parser.text()


def _write_source_pointer(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"name: {doc.get('name', '')}",
        f"category: {doc.get('category', '')}",
        f"source: {doc.get('source', '')}",
        f"url: {doc.get('url', '')}",
        f"description: {doc.get('description', '')}",
        "",
        "This source is recorded as manifest-only. Collect it through the platform's",
        "official export/API or after reviewing its license and redistribution terms.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _download_document(doc: Dict[str, Any], doc_path: Path) -> Dict[str, Any]:
    doc_record = {
        "name": doc["name"],
        "category": doc.get("category", ""),
        "source": doc.get("source", ""),
        "url": doc["url"],
        "description": doc.get("description", ""),
        "local_path": str(doc_path),
        "status": "pending",
    }

    if doc.get("manifest_only"):
        _write_source_pointer(doc_path, doc)
        doc_record["status"] = "manifest_only"
        return doc_record

    response = requests.get(
        doc["url"],
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.content
    content_type = response.headers.get("content-type", "").lower()
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    if content.lstrip().startswith(b"%PDF"):
        if doc_path.suffix.lower() != ".pdf":
            doc_path = doc_path.with_suffix(".pdf")
            doc_record["local_path"] = str(doc_path)
        doc_path.write_bytes(content)
    elif doc_path.suffix.lower() in {".zip", ".csv", ".json", ".jsonl"}:
        doc_path.write_bytes(content)
    elif "html" in content_type or doc_path.suffix.lower() == ".txt":
        text = _html_to_text(content)
        if not text:
            text = content.decode("utf-8", errors="ignore")
        header = [
            f"Title: {doc.get('name', '')}",
            f"Category: {doc.get('category', '')}",
            f"Source: {doc.get('source', '')}",
            f"URL: {doc.get('url', '')}",
            "",
        ]
        doc_path.write_text("\n".join(header) + text, encoding="utf-8")
    else:
        doc_path.write_bytes(content)

    doc_record["status"] = "downloaded"
    doc_record["bytes"] = doc_path.stat().st_size
    doc_record["sha256"] = hashlib.sha256(doc_path.read_bytes()).hexdigest()
    return doc_record


def _download_document_collection(source: Dict[str, Any], local_path: Path) -> Dict[str, Any]:
    local_path.mkdir(parents=True, exist_ok=True)
    documents = []
    downloaded_count = 0
    manifest_count = 0
    for doc in source.get("documents", []):
        doc_path = local_path / doc["local_name"]
        try:
            doc_record = _download_document(doc, doc_path)
            actual_path = Path(doc_record.get("local_path", doc_path))
            if doc_record["status"] == "downloaded" and actual_path.suffix.lower() == ".zip":
                extract_dir = actual_path.parent / actual_path.stem
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(actual_path) as archive:
                    archive.extractall(extract_dir)
                doc_record["extract_dir"] = str(extract_dir)
            if doc_record["status"] == "downloaded":
                downloaded_count += 1
            elif doc_record["status"] == "manifest_only":
                manifest_count += 1
        except Exception as exc:
            doc_record = {
                "name": doc["name"],
                "category": doc.get("category", ""),
                "source": doc.get("source", ""),
                "url": doc["url"],
                "description": doc.get("description", ""),
                "local_path": str(doc_path),
                "status": "error",
                "error": str(exc),
            }
        documents.append(doc_record)
    metadata_path = local_path / "source_metadata.jsonl"
    _write_jsonl(documents, metadata_path)
    status = "downloaded" if downloaded_count else "manifest_only" if manifest_count else "error"
    return {
        "status": status,
        "documents": documents,
        "downloaded_count": downloaded_count,
        "manifest_only_count": manifest_count,
        "metadata_path": str(metadata_path),
    }


def download_public_datasets(
    output_root: Path = Path("data"),
    datasets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    selected = datasets or ["cuad", "acord", "insuranceqa"]
    manifest: List[Dict[str, Any]] = []
    output_root = Path(output_root)

    for dataset_name in selected:
        if dataset_name not in PUBLIC_DATA_SOURCES:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        source = dict(PUBLIC_DATA_SOURCES[dataset_name])
        local_path = output_root / Path(source["local_path"]).relative_to("data")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "dataset": dataset_name,
            "name": source["name"],
            "license": source["license"],
            "url": source["url"],
            "local_path": str(local_path),
            "description": source["description"],
            "status": "manifest_only",
        }

        if dataset_name in {"public_docs", "real_domain_mix"}:
            collection_result = _download_document_collection(source, local_path)
            record.update(collection_result)
            manifest.append(record)
            continue

        if dataset_name in {"cuad", "acord"}:
            response = requests.get(source["url"], timeout=120)
            response.raise_for_status()
            local_path.write_bytes(response.content)
            record["status"] = "downloaded"
            record["bytes"] = len(response.content)
            if dataset_name == "acord":
                extract_dir = local_path.parent / "extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(local_path) as archive:
                    archive.extractall(extract_dir)
                record["extract_dir"] = str(extract_dir)

        manifest.append(record)

    manifest_path = output_root / "manifests" / "external_data_manifest.jsonl"
    _write_jsonl(manifest, manifest_path)
    return {"manifest_path": str(manifest_path), "records": manifest}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "[]", "nan", "NaN"}:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return " ".join(str(item).strip() for item in parsed if str(item).strip())
        except Exception:
            pass
    return re.sub(r"\s+", " ", text)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9$%]+", text.lower()) if len(token) > 2}


def _content_tokens(text: str) -> List[str]:
    stopwords = {
        "about", "after", "also", "and", "are", "before", "between", "from", "have",
        "into", "may", "more", "must", "not", "that", "the", "their", "there", "this",
        "when", "where", "which", "will", "with", "would", "your", "you", "they",
        "them", "than", "then", "what", "does", "policy", "insurance", "coverage",
        "company", "companies", "consumer", "guide", "page",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if token not in stopwords and len(token) > 2
    ]


TOPIC_KEYWORDS = {
    "deductible": ["deductible", "deductibles"],
    "liability coverage": ["liability", "bodily injury", "property damage"],
    "premium": ["premium", "premiums", "rate", "rates"],
    "discounts": ["discount", "discounts"],
    "declarations page": ["declaration", "declarations"],
    "claims": ["claim", "claims", "loss"],
    "underwriting": ["underwriting", "risk characteristics", "risk"],
    "credit history": ["credit", "bankruptcy", "late payments"],
    "vehicle information": ["vehicle", "vehicles", "make", "model"],
    "homeowners coverage": ["homeowners", "dwelling", "personal property"],
    "limits": ["limit", "limits"],
    "exclusions": ["exclude", "excludes", "exclusion", "exclusions"],
    "uninsured motorists": ["uninsured", "underinsured"],
    "personal injury protection": ["personal injury protection", "pip"],
    "collision coverage": ["collision"],
    "comprehensive coverage": ["comprehensive"],
}


def _source_to_page_id(source: str) -> str:
    return source.replace("/", "_").replace("#page=", "_p")


def _make_qa(
    qa_id: str,
    question: str,
    answer: str,
    evidence_source: str,
    evidence_text: str,
    qa_source: str,
    answerable: bool = True,
    page_number: Optional[int] = None,
    doc_id: Optional[str] = None,
    policy_family_id: Optional[str] = None,
    version_id: Optional[str] = None,
) -> Dict[str, Any]:
    page_id = _source_to_page_id(evidence_source) if evidence_source else None
    return {
        "qa_id": qa_id,
        "question": question,
        "answer": answer,
        "ground_truth": answer,
        "evidence_page_ids": [page_id] if page_id else [],
        "evidence_sources": [evidence_source] if evidence_source else [],
        "citations": [evidence_source] if evidence_source else [],
        "evidence_text": evidence_text,
        "doc_id": doc_id,
        "policy_family_id": policy_family_id,
        "version_id": version_id,
        "page_number": page_number,
        "qa_source": qa_source,
        "answerable": answerable,
    }


def _policy_qa_from_page(doc: PageDocument, qa_index: int) -> List[Dict[str, Any]]:
    text = doc.text or ""
    lines = [line.strip() for line in re.split(r"\n+", text) if line.strip()]
    source = doc.metadata.get("source", doc.doc_id)
    doc_id = doc.metadata.get("path") or doc.doc_id
    page_number = doc.page_number
    policy_family_id = doc.metadata.get("policy_family_id")
    version_id = doc.metadata.get("version_id")
    records: List[Dict[str, Any]] = []

    def add(question: str, answer: str, evidence: str, label: str) -> None:
        records.append(
            _make_qa(
                qa_id=f"policy_{qa_index}_{len(records)}_{label}",
                question=question,
                answer=answer,
                evidence_source=source,
                evidence_text=evidence,
                qa_source="policy_rules",
                page_number=page_number,
                doc_id=doc_id,
                policy_family_id=policy_family_id,
                version_id=version_id,
            )
        )

    for line in lines:
        lowered = line.lower()
        if "deductible" in lowered and re.search(r"\$[\d,]+", line):
            if "comprehensive" in lowered:
                add("What is the comprehensive deductible?", line, line, "comprehensive_deductible")
            elif "collision" in lowered:
                add("What is the collision deductible?", line, line, "collision_deductible")
            else:
                add("What deductible is listed in the policy?", line, line, "deductible")
        if "liability" in lowered and "limit" in lowered:
            add("What is the liability coverage limit?", line, line, "liability_limit")
        if "endorsement" in lowered and ("provides" in lowered or "covers" in lowered):
            add("What does the endorsement provide?", line, line, "endorsement")
        if "excludes coverage" in lowered or lowered.startswith("this policy excludes"):
            add("What exclusions are listed in the policy?", line, line, "exclusions")
        if "insured must" in lowered or "must promptly notify" in lowered:
            add("What duties does the insured have after a loss?", line, line, "duties")
        if "medical payments" in lowered:
            add("What does Medical Payments Coverage apply to?", line, line, "medical_payments")

    return records


def _split_evidence_chunks(text: str, max_chars: int = 420) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|(?<=:)\s+", cleaned)
        if sentence.strip()
    ]
    chunks = []
    current = ""
    for sentence in sentences:
        if len(sentence) < 35:
            continue
        if len(sentence) > max_chars:
            sentence = sentence[:max_chars].rsplit(" ", 1)[0].strip()
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current:
        chunks.append(current)
    return chunks


def _infer_topic(evidence: str) -> str:
    lowered = evidence.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    tokens = _content_tokens(evidence)
    if not tokens:
        return "the insurance document"
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0].replace("-", " ")


def _distinctive_phrase(evidence: str, max_terms: int = 4) -> str:
    tokens = _content_tokens(evidence)
    if not tokens:
        return ""
    counts: Dict[str, int] = {}
    ordered_tokens = []
    for token in tokens:
        if token not in counts:
            ordered_tokens.append(token)
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(ordered_tokens, key=lambda token: (-counts[token], ordered_tokens.index(token)))
    return " ".join(ranked[:max_terms]).replace("-", " ")


def _expanded_qa_from_page(doc: PageDocument, qa_index: int, max_per_page: int = 8) -> List[Dict[str, Any]]:
    source = doc.metadata.get("source", doc.doc_id)
    doc_id = doc.metadata.get("path") or doc.doc_id
    page_number = doc.page_number
    chunks = _split_evidence_chunks(doc.text or "")
    records: List[Dict[str, Any]] = []
    seen_questions = set()

    for chunk_idx, chunk in enumerate(chunks):
        topic = _infer_topic(chunk)
        phrase = _distinctive_phrase(chunk)
        anchor = f" mentioning {phrase}" if phrase else ""
        templates = [
            f"What does the page{anchor} say about {topic}?",
            f"Which evidence{anchor} explains {topic}?",
            f"Summarize the policy guidance{anchor} on {topic}.",
            f"Find the page evidence{anchor} related to {topic}.",
            f"How is {topic} described in the evidence{anchor}?",
            f"What should a consumer know about {topic} from the passage{anchor}?",
        ]
        if "$" in chunk or re.search(r"\b\d+%|\b\d+/\d+/\d+\b", chunk):
            templates.append(f"What amount, limit, or numeric detail{anchor} is stated about {topic}?")
        for template_idx, question in enumerate(templates):
            key = question.lower()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            records.append(
                _make_qa(
                    qa_id=f"realpdf_{qa_index}_{chunk_idx}_{template_idx}",
                    question=question,
                    answer=chunk,
                    evidence_source=source,
                    evidence_text=chunk,
                    qa_source="real_pdf_evidence_rules",
                    page_number=page_number,
                    doc_id=doc_id,
                )
            )
            if len(records) >= max_per_page:
                return records
    return records


def _augment_multi_positive_sources(
    qa_pairs: List[Dict[str, Any]],
    documents: List[PageDocument],
    max_sources: int = 5,
) -> None:
    docs_by_name: Dict[str, List[PageDocument]] = {}
    page_token_cache: Dict[str, set[str]] = {}
    for doc in documents:
        source = doc.metadata.get("source", doc.doc_id)
        doc_name = source.split("#page=", 1)[0]
        docs_by_name.setdefault(doc_name, []).append(doc)
        page_token_cache[source] = set(_content_tokens(doc.text or ""))

    for qa in qa_pairs:
        if not qa.get("answerable", True) or qa.get("qa_source") != "real_pdf_evidence_rules":
            continue
        evidence_sources = list(qa.get("evidence_sources", []))
        if not evidence_sources:
            continue
        doc_name = evidence_sources[0].split("#page=", 1)[0]
        query_terms = set(_content_tokens(f"{qa.get('question', '')} {qa.get('evidence_text', '')}"))
        if not query_terms:
            continue
        scored = []
        for doc in docs_by_name.get(doc_name, []):
            source = doc.metadata.get("source", doc.doc_id)
            if source in evidence_sources:
                continue
            page_terms = page_token_cache.get(source, set())
            overlap = len(query_terms & page_terms)
            if overlap >= 3:
                scored.append((overlap, source))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for _, source in scored[: max(0, max_sources - len(evidence_sources))]:
            evidence_sources.append(source)

        qa["evidence_sources"] = evidence_sources
        qa["citations"] = evidence_sources
        qa["evidence_page_ids"] = [_source_to_page_id(source) for source in evidence_sources]


def _build_document_split_records(
    documents: List[PageDocument],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    doc_ids = sorted({doc.metadata.get("path") or doc.doc_id.split("#page=", 1)[0] for doc in documents})
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    total = len(doc_ids)
    if total == 0:
        return []

    if total == 1:
        split_names = ["train"]
    else:
        train_n = max(1, int(total * train_ratio))
        valid_n = max(1, int(total * valid_ratio)) if total >= 3 else 0
        if train_n + valid_n >= total:
            train_n = max(1, total - 1)
            valid_n = 0 if total == 2 else 1
        test_n = max(0, total - train_n - valid_n)
        if total >= 3 and test_n == 0:
            train_n = max(1, train_n - 1)
            test_n = 1
        split_names = ["train"] * train_n + ["valid"] * valid_n + ["test"] * test_n

    return [
        {"doc_id": doc_id, "split": split_name}
        for doc_id, split_name in zip(doc_ids, split_names)
    ]


def _assign_qa_document_splits(
    qa_pairs: List[Dict[str, Any]],
    documents: List[PageDocument],
    output_dir: Path,
) -> Path:
    split_records = _build_document_split_records(documents)
    split_by_doc = {record["doc_id"]: record["split"] for record in split_records}
    for qa in qa_pairs:
        doc_id = qa.get("doc_id")
        if doc_id and doc_id in split_by_doc:
            qa["split"] = split_by_doc[doc_id]
            qa["split_doc_id"] = doc_id
        elif qa.get("answerable", True):
            qa["split"] = "train"
            qa["split_doc_id"] = doc_id
        else:
            qa["split"] = "test"
            qa["split_doc_id"] = None

    splits_path = Path(output_dir) / "qa_splits.jsonl"
    _write_jsonl(split_records, splits_path)
    return splits_path


def generate_policy_qa_pairs(
    data_folder: Path,
    output_dir: Path = Path("data/02_processed"),
    include_unsupported: bool = True,
    target_count: Optional[int] = None,
) -> QAResult:
    documents = load_documents(Path(data_folder), render_pdf_pages=False)
    qa_pairs: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        qa_pairs.extend(_policy_qa_from_page(doc, idx))

    if target_count and len([item for item in qa_pairs if item.get("answerable", True)]) < target_count:
        per_page = max(8, min(24, math.ceil(target_count / max(1, len(documents))) + 6))
        seen = {
            (item.get("question"), item.get("answer"), tuple(item.get("evidence_sources", [])))
            for item in qa_pairs
        }
        expanded_by_doc: List[List[Dict[str, Any]]] = []
        for idx, doc in enumerate(documents):
            doc_records: List[Dict[str, Any]] = []
            for item in _expanded_qa_from_page(doc, idx, max_per_page=per_page):
                key = (item.get("question"), item.get("answer"), tuple(item.get("evidence_sources", [])))
                if key in seen:
                    continue
                seen.add(key)
                doc_records.append(item)
            if doc_records:
                expanded_by_doc.append(doc_records)

        expanded: List[Dict[str, Any]] = []
        offsets = [0 for _ in expanded_by_doc]
        while len(qa_pairs) + len(expanded) < target_count:
            added_this_round = False
            for bucket_idx, bucket in enumerate(expanded_by_doc):
                offset = offsets[bucket_idx]
                if offset >= len(bucket):
                    continue
                expanded.append(bucket[offset])
                offsets[bucket_idx] += 1
                added_this_round = True
                if len(qa_pairs) + len(expanded) >= target_count:
                    break
            if not added_this_round:
                break
        qa_pairs.extend(expanded)

    _augment_multi_positive_sources(qa_pairs, documents)

    if include_unsupported:
        unsupported_questions = [
            "What is the earthquake deductible?",
            "Does this policy include jewelry scheduled property coverage?",
            "What is the cyber liability sublimit?",
        ]
        for idx, question in enumerate(unsupported_questions):
            qa_pairs.append(
                _make_qa(
                    qa_id=f"unsupported_{idx}",
                    question=question,
                    answer="",
                    evidence_source="",
                    evidence_text="",
                    qa_source="unsupported_rules",
                    answerable=False,
                )
            )

    splits_path = _assign_qa_document_splits(qa_pairs, documents, Path(output_dir))
    hard_negatives = build_hard_negatives(qa_pairs, documents)
    qa_path = Path(output_dir) / "qa_pairs.jsonl"
    hard_path = Path(output_dir) / "hard_negatives.jsonl"
    _write_jsonl(qa_pairs, qa_path)
    _write_jsonl(hard_negatives, hard_path)
    return QAResult(qa_path, hard_path, len(qa_pairs), len(hard_negatives), splits_path)


def build_hard_negatives(
    qa_pairs: List[Dict[str, Any]],
    documents: List[PageDocument],
    negatives_per_question: int = 3,
) -> List[Dict[str, Any]]:
    candidates = []
    for doc in documents:
        source = doc.metadata.get("source", doc.doc_id)
        candidates.append(
            {
                "page_id": _source_to_page_id(source),
                "source": source,
                "text": doc.text or "",
                "tokens": _tokenize(doc.text or ""),
            }
        )

    negatives: List[Dict[str, Any]] = []
    for qa in qa_pairs:
        if not qa.get("answerable", True):
            continue
        positive_sources = set(qa.get("evidence_sources", []))
        q_terms = _tokenize(qa.get("question", ""))
        scored = []
        for candidate in candidates:
            if candidate["source"] in positive_sources:
                continue
            score = len(q_terms & candidate["tokens"])
            scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        for rank, (_, candidate) in enumerate(scored[:negatives_per_question]):
            negatives.append(
                {
                    "qa_id": qa["qa_id"],
                    "negative_page_id": candidate["page_id"],
                    "negative_source": candidate["source"],
                    "rank": rank + 1,
                    "negative_type": "lexical_hard_negative",
                }
            )
    return negatives


def import_cuad_qa(
    master_clauses_path: Path,
    output_path: Path,
    max_examples: int = 2000,
) -> int:
    records: List[Dict[str, Any]] = []
    with Path(master_clauses_path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        context_columns = [
            name
            for name in fieldnames
            if name != "Filename" and not name.lower().replace(" ", "").endswith("-answer")
        ]
        for row_idx, row in enumerate(reader):
            filename = row.get("Filename", f"cuad_doc_{row_idx}")
            for column in context_columns:
                context = _clean_text(row.get(column))
                if not context:
                    continue
                answer = _clean_text(row.get(f"{column}-Answer")) or _clean_text(row.get(f"{column}- Answer"))
                if not answer or answer.lower() == "no":
                    continue
                question = f"What is the {column} provision or answer?"
                records.append(
                    _make_qa(
                        qa_id=f"cuad_{len(records)}",
                        question=question,
                        answer=answer,
                        evidence_source=filename,
                        evidence_text=context[:2000],
                        qa_source="cuad_master_clauses",
                        doc_id=filename,
                    )
                )
                if len(records) >= max_examples:
                    _write_jsonl(records, output_path)
                    return len(records)
    _write_jsonl(records, output_path)
    return len(records)


def import_acord_qa(
    acord_root: Path,
    output_path: Path,
    hard_negatives_path: Path,
    min_score: int = 2,
    max_examples: int = 3000,
) -> QAResult:
    root = _find_acord_root(Path(acord_root))
    queries = {item["_id"]: item for item in _read_jsonl(root / "queries.jsonl")}
    corpus = {item["_id"]: item for item in _read_jsonl(root / "corpus.jsonl")}
    qrel_paths = [root / "qrels" / name for name in ["train.tsv", "valid.tsv", "test.tsv"]]

    qa_pairs: List[Dict[str, Any]] = []
    hard_negatives: List[Dict[str, Any]] = []
    for qrel_path in qrel_paths:
        if not qrel_path.exists():
            continue
        split = qrel_path.stem
        with qrel_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                query_id = row["query-id"]
                corpus_id = row["corpus-id"]
                score = int(row["score"])
                if query_id not in queries or corpus_id not in corpus:
                    continue
                if score >= min_score and len(qa_pairs) < max_examples:
                    query = queries[query_id]
                    clause = corpus[corpus_id]
                    qa_pairs.append(
                        _make_qa(
                            qa_id=f"acord_{len(qa_pairs)}",
                            question=query["text"],
                            answer=clause["text"],
                            evidence_source=corpus_id,
                            evidence_text=clause["text"],
                            qa_source=f"acord_{split}",
                            doc_id=corpus_id,
                        )
                    )
                elif score < min_score:
                    hard_negatives.append(
                        {
                            "qa_id": query_id,
                            "negative_page_id": corpus_id,
                            "negative_source": corpus_id,
                            "score": score,
                            "negative_type": "acord_low_relevance",
                        }
                    )
    _write_jsonl(qa_pairs, output_path)
    _write_jsonl(hard_negatives, hard_negatives_path)
    return QAResult(output_path, hard_negatives_path, len(qa_pairs), len(hard_negatives))


def _find_acord_root(path: Path) -> Path:
    if (path / "queries.jsonl").exists():
        return path
    for candidate in path.rglob("queries.jsonl"):
        return candidate.parent
    raise FileNotFoundError(f"Could not find ACORD queries.jsonl under {path}")


def merge_qa_files(paths: List[Path], output_path: Path) -> int:
    records: List[Dict[str, Any]] = []
    seen = set()
    for path in paths:
        for item in _read_jsonl(Path(path)):
            key = (item.get("question"), item.get("answer"), tuple(item.get("evidence_sources", [])))
            if key in seen:
                continue
            seen.add(key)
            item["qa_id"] = f"merged_{len(records)}"
            records.append(item)
    _write_jsonl(records, output_path)
    return len(records)


def compute_retrieval_metrics(
    pipeline: Any,
    data_folder: Path,
    qa_path: Path,
    top_k: int = 10,
) -> RetrievalMetrics:
    examples = [item for item in _read_jsonl(qa_path) if item.get("answerable", True)]
    if not examples:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0)

    recall_1 = 0
    recall_5 = 0
    mrr_10 = 0.0
    ndcg_10 = 0.0

    for item in examples:
        gold_sources = set(item.get("evidence_sources") or item.get("citations") or [])
        ranked = pipeline.rank_pages(item["question"], Path(data_folder), top_k=top_k)
        ranked_sources = [candidate["source"] for candidate in ranked]
        hit_positions = [
            idx + 1
            for idx, source in enumerate(ranked_sources[:10])
            if source in gold_sources
        ]
        if ranked_sources[:1] and ranked_sources[0] in gold_sources:
            recall_1 += 1
        if any(source in gold_sources for source in ranked_sources[:5]):
            recall_5 += 1
        if hit_positions:
            first_hit = hit_positions[0]
            mrr_10 += 1.0 / first_hit
            ndcg_10 += 1.0 / math.log2(first_hit + 1)

    count = len(examples)
    return RetrievalMetrics(
        recall_at_1=recall_1 / count,
        recall_at_5=recall_5 / count,
        mrr_at_10=mrr_10 / count,
        ndcg_at_10=ndcg_10 / count,
        evaluated_count=count,
    )
