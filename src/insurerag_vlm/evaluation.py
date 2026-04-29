import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ANSWER_CITATION_PATTERN = re.compile(r"(?:SOURCE:\s*|source:\s*)([^\n]+)", re.IGNORECASE)


@dataclass
class EvaluationExample:
    question: str
    ground_truth: str
    citations: List[str]


def load_evaluation_examples(path: Path) -> List[EvaluationExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    examples: List[EvaluationExample] = []
    for item in raw:
        examples.append(
            EvaluationExample(
                question=item["question"],
                ground_truth=item.get("answer", item.get("ground_truth", "")).strip(),
                citations=item.get("citations", []),
            )
        )
    return examples


def _normalize_citations(value):
    if isinstance(value, str):
        return [x.strip() for x in value.split("|") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def generate_evaluation_examples(
    input_path: Path,
    output_path: Path,
    input_format: str = "auto",
    question_key: str = "question",
    answer_key: str = "answer",
    citations_key: str = "citations",
) -> None:
    if input_format == "auto":
        if input_path.suffix.lower() == ".csv":
            input_format = "csv"
        elif input_path.suffix.lower() == ".jsonl":
            input_format = "jsonl"
        elif input_path.suffix.lower() == ".json":
            input_format = "json"
        else:
            raise ValueError("Unsupported eval input format. Use csv, jsonl, or json.")

    examples: List[Dict[str, object]] = []
    if input_format == "csv":
        import csv

        with input_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                examples.append(
                    {
                        "question": row[question_key].strip(),
                        "answer": row[answer_key].strip(),
                        "citations": _normalize_citations(row.get(citations_key, "")),
                    }
                )
    elif input_format == "jsonl":
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                examples.append(
                    {
                        "question": str(item[question_key]).strip(),
                        "answer": str(item[answer_key]).strip(),
                        "citations": _normalize_citations(item.get(citations_key, [])),
                    }
                )
    elif input_format == "json":
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw:
            examples.append(
                {
                    "question": str(item[question_key]).strip(),
                    "answer": str(item[answer_key]).strip(),
                    "citations": _normalize_citations(item.get(citations_key, [])),
                }
            )
    else:
        raise ValueError("Unsupported eval input format. Use csv, jsonl, or json.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def exact_match(prediction: str, reference: str) -> int:
    return int(normalize_text(prediction) == normalize_text(reference))


def f1_score(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_citations(answer_text: str) -> List[str]:
    citations = ANSWER_CITATION_PATTERN.findall(answer_text)
    return [citation.strip() for citation in citations if citation.strip()]


def compute_citation_precision(predicted: List[str], gold: List[str]) -> float:
    if not predicted:
        return 0.0
    matched = sum(1 for item in predicted if any(normalize_text(item) in normalize_text(g) or normalize_text(g) in normalize_text(item) for g in gold))
    return matched / len(predicted)


def evaluate_predictions(
    predictions: Iterable[Tuple[str, str]],
    examples: Iterable[EvaluationExample],
) -> Dict[str, float]:
    predictions = list(predictions)
    examples = list(examples)
    if len(predictions) != len(examples):
        raise ValueError("Number of predictions and examples must match.")

    total_em = 0
    total_f1 = 0.0
    total_citation_precision = 0.0

    for (question, prediction), example in zip(predictions, examples):
        total_em += exact_match(prediction, example.ground_truth)
        total_f1 += f1_score(prediction, example.ground_truth)
        citations = extract_citations(prediction)
        total_citation_precision += compute_citation_precision(citations, example.citations)

    count = len(predictions)
    return {
        "em": total_em / count if count else 0.0,
        "f1": total_f1 / count if count else 0.0,
        "citation_precision": total_citation_precision / count if count else 0.0,
    }
