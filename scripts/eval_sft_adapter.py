#!/usr/bin/env python3
import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

from src.insurerag_vlm.evaluation import exact_match, f1_score
from src.insurerag_vlm.sft import DEFAULT_QWEN_7B_MODEL, format_sft_messages, read_sft_records


def _is_unsupported(record: Dict[str, Any]) -> bool:
    return not bool(record.get("answerable", True))


def _abstains(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "insufficient",
        "cannot support",
        "can't support",
        "not enough information",
        "do not have enough information",
        "don't have enough information",
        "provided evidence does not",
        "not stated in the provided evidence",
        "cannot determine from the provided evidence",
    ]
    return any(marker in lowered for marker in markers)


def _pick_records(records: List[Dict[str, Any]], answerable_count: int, unsupported_count: int) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    seen_answerable_sources = set()
    seen_unsupported_sources = set()

    for record in records:
        source = str(record.get("source") or "")
        if not _is_unsupported(record):
            if source in seen_answerable_sources:
                continue
            chosen.append(record)
            seen_answerable_sources.add(source)
            if len(seen_answerable_sources) >= answerable_count:
                break

    for record in records:
        source = str(record.get("source") or "")
        if _is_unsupported(record):
            if source in seen_unsupported_sources:
                continue
            chosen.append(record)
            seen_unsupported_sources.add(source)
            if len(seen_unsupported_sources) >= unsupported_count:
                break

    return chosen


def _load_generation_stack(model_name: str, adapter_dir: Path | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        dtype=compute_dtype,
        quantization_config=quantization_config,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return tokenizer, model


def _generate_for_records(
    model_name: str,
    records: List[Dict[str, Any]],
    adapter_dir: Path | None,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    import torch

    tokenizer, model = _load_generation_stack(model_name, adapter_dir)
    rows: List[Dict[str, Any]] = []
    try:
        for record in records:
            messages = format_sft_messages(record)
            prompt = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            batch = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(output[0][batch["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            reference = str(record.get("answer") or "").strip()
            unsupported = _is_unsupported(record)
            rows.append(
                {
                    "record_id": record.get("record_id"),
                    "question": record.get("question"),
                    "source": record.get("source"),
                    "answerable": not unsupported,
                    "reference": reference,
                    "prediction": generated,
                    "exact_match": exact_match(generated, reference),
                    "f1": f1_score(generated, reference),
                    "reference_abstains": _abstains(reference),
                    "prediction_abstains": _abstains(generated),
                }
            )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    unsupported = [row for row in rows if not row["answerable"]]

    def avg(key: str, subset: List[Dict[str, Any]]) -> float:
        if not subset:
            return 0.0
        return sum(float(item[key]) for item in subset) / len(subset)

    def rate(key: str, subset: List[Dict[str, Any]]) -> float:
        if not subset:
            return 0.0
        return sum(1 for item in subset if item[key]) / len(subset)

    return {
        "count": len(rows),
        "overall_em": avg("exact_match", rows),
        "overall_f1": avg("f1", rows),
        "answerable_count": len(answerable),
        "answerable_em": avg("exact_match", answerable),
        "answerable_f1": avg("f1", answerable),
        "unsupported_count": len(unsupported),
        "unsupported_reference_abstain_rate": rate("reference_abstains", unsupported),
        "unsupported_prediction_abstain_rate": rate("prediction_abstains", unsupported),
    }


def _write_markdown(output_md: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# SFT Adapter Spot Check",
        "",
        f"- Base model: `{payload['model_name']}`",
        f"- Adapter dir: `{payload['adapter_dir']}`",
        f"- Samples: `{payload['sample_count']}`",
        "",
        "## Summary",
        "",
        "| variant | overall_f1 | answerable_f1 | unsupported_abstain_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for variant in ["base", "adapter"]:
        summary = payload["summary"][variant]
        lines.append(
            f"| {variant} | {summary['overall_f1']:.4f} | {summary['answerable_f1']:.4f} | {summary['unsupported_prediction_abstain_rate']:.4f} |"
        )

    lines.extend(["", "## Samples", ""])
    for idx, record in enumerate(payload["records"], start=1):
        lines.extend(
            [
                f"### Sample {idx}: {record['record_id']}",
                "",
                f"- Answerable: `{record['answerable']}`",
                f"- Source: `{record['source']}`",
                f"- Question: {record['question']}",
                "",
                f"**Reference**\n\n{record['reference']}",
                "",
                f"**Base**\n\n{record['base_prediction']}",
                "",
                f"**Adapter**\n\n{record['adapter_prediction']}",
                "",
            ]
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Spot-check a trained LoRA adapter against the base model.")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/04_curated/sft_dataset.jsonl"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/qwen7b-insurerag-lora"))
    parser.add_argument("--model-name", type=str, default=DEFAULT_QWEN_7B_MODEL)
    parser.add_argument("--answerable-count", type=int, default=4)
    parser.add_argument("--unsupported-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-json", type=Path, default=Path("reports/sft_eval/adapter_spot_check.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/sft_eval/adapter_spot_check.md"))
    args = parser.parse_args()

    records = read_sft_records(args.dataset_path)
    chosen = _pick_records(records, args.answerable_count, args.unsupported_count)
    if len(chosen) < args.answerable_count + args.unsupported_count:
        raise ValueError("Not enough records to build the requested spot-check set.")

    base_rows = _generate_for_records(args.model_name, chosen, adapter_dir=None, max_new_tokens=args.max_new_tokens)
    adapter_rows = _generate_for_records(args.model_name, chosen, adapter_dir=args.adapter_dir, max_new_tokens=args.max_new_tokens)

    merged_records: List[Dict[str, Any]] = []
    for base_row, adapter_row in zip(base_rows, adapter_rows):
        merged_records.append(
            {
                "record_id": base_row["record_id"],
                "question": base_row["question"],
                "source": base_row["source"],
                "answerable": base_row["answerable"],
                "reference": base_row["reference"],
                "base_prediction": base_row["prediction"],
                "base_exact_match": base_row["exact_match"],
                "base_f1": base_row["f1"],
                "base_prediction_abstains": base_row["prediction_abstains"],
                "adapter_prediction": adapter_row["prediction"],
                "adapter_exact_match": adapter_row["exact_match"],
                "adapter_f1": adapter_row["f1"],
                "adapter_prediction_abstains": adapter_row["prediction_abstains"],
            }
        )

    payload = {
        "model_name": args.model_name,
        "adapter_dir": str(args.adapter_dir),
        "dataset_path": str(args.dataset_path),
        "sample_count": len(chosen),
        "summary": {
            "base": _summarize(base_rows),
            "adapter": _summarize(adapter_rows),
        },
        "records": merged_records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(args.output_md, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
