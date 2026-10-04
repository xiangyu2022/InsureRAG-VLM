"""Offline paired baseline/LoRA experiment with frozen document splits.

Only public, rule-labeled evidence is used. Answer content scoring excludes citation
boilerplate. The visual encoder is frozen and this experiment trains TEXT ONLY.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import random
import re
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModelForImageTextToText

ROOT = Path("reports/local_20261004")
SEED = 20261004
from src.insurerag_vlm.local_answer_metrics import SYSTEM, prompt_ids, score, aggregate


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False


def encode(tokenizer, row):
    prompt = prompt_ids(tokenizer, row)
    answer = tokenizer(row["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return prompt, answer


def load_model(model_path, adapter=None):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model


def token_loss(model, prompt, answer):
    ids = torch.tensor([prompt + answer], device="cuda")
    # Compute logits only where an assistant target is supervised. This avoids
    # materializing prompt-length x 248k vocabulary logits, without changing loss.
    positions = torch.arange(len(prompt) - 1, ids.shape[1] - 1, device="cuda")
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits
    targets = torch.tensor(answer, device="cuda")
    return torch.nn.functional.cross_entropy(logits[0].float(), targets)


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def content_f1(prediction, reference):
    a, b = Counter(words(prediction)), Counter(words(reference))
    overlap = sum((a & b).values())
    return 2 * overlap / (sum(a.values()) + sum(b.values())) if a or b else 1.0


def score(row, prediction):
    content = re.split(r"\bSource\s*:", prediction, flags=re.I)[0].strip()
    abstain = bool(re.search(r"INSUFFICIENT_EVIDENCE|insufficient evidence|cannot (?:confirm|determine)|does not (?:provide|support|mention)|not (?:specified|stated|supported)", prediction, re.I))
    citations = re.findall(r"\bSource\s*:\s*(.+)", prediction, re.I)
    correct_citation = bool(citations) and citations[-1].strip().rstrip(".") == row["source"].rstrip(".")
    tokens = words(content)
    evidence_tokens = set(words(row["evidence"]))
    coverage = sum(t in evidence_tokens for t in tokens) / max(1, len(tokens))
    numbers = set(re.findall(r"\d[\d,.]*(?:%|\b)", content))
    evidence_numbers = set(re.findall(r"\d[\d,.]*(?:%|\b)", row["evidence"]))
    return {"content_f1": content_f1(content, row["reference_content"]) if row["answerable"] else None,
            "abstains": abstain, "has_citation": bool(citations), "citation_correct": correct_citation,
            "evidence_token_coverage": coverage if not abstain else None,
            "numbers_supported": numbers <= evidence_numbers if not abstain else None,
            "lexical_support_proxy": bool(correct_citation and coverage >= 0.8 and numbers <= evidence_numbers and not abstain)}


def aggregate(rows):
    answerable = [x for x in rows if x["answerable"]]
    tp = sum(x["abstains"] and not x["answerable"] for x in rows)
    fp = sum(x["abstains"] and x["answerable"] for x in rows)
    fn = sum(not x["abstains"] and not x["answerable"] for x in rows)
    cited = sum(x["has_citation"] for x in rows)
    return {"n": len(rows), "n_answerable": len(answerable), "n_unsupported": len(rows) - len(answerable),
            "answerable_content_f1": sum(x["content_f1"] for x in answerable) / max(1, len(answerable)),
            "citation_precision_exact_source": sum(x["citation_correct"] for x in rows) / max(1, cited),
            "answerable_citation_rate": sum(x["citation_correct"] for x in answerable) / max(1, len(answerable)),
            "answerable_lexical_support_proxy": sum(x["lexical_support_proxy"] for x in answerable) / max(1, len(answerable)),
            "abstention_precision": tp / max(1, tp + fp), "abstention_recall": tp / max(1, tp + fn),
            "abstention_counts": {"tp": tp, "fp": fp, "fn": fn},
            "latency_p50_seconds": float(np.percentile([x["seconds"] for x in rows], 50)),
            "latency_p95_seconds": float(np.percentile([x["seconds"] for x in rows], 95)),
            "generated_tokens": sum(x["generated_tokens"] for x in rows)}


def evaluate(model, tokenizer, rows, name, max_new_tokens):
    model.eval()
    out = ROOT / f"{name}_predictions.jsonl"
    results = []
    with out.open("w", encoding="utf-8") as stream:
        for i, row in enumerate(rows):
            ids = torch.tensor([prompt_ids(tokenizer, row)], device="cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            generated = output[0, ids.shape[1]:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            result = {"id": row["id"], "doc_id": row["doc_id"], "answerable": row["answerable"],
                      "prediction": prediction, "seconds": seconds, "generated_tokens": len(generated),
                      "prompt_tokens": ids.shape[1], **score(row, prediction)}
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if (i + 1) % 8 == 0:
                print(f"{name}: {i+1}/{len(rows)}", flush=True)
    metrics = aggregate(results)
    write_json(ROOT / f"{name}_metrics.json", metrics)
    print(json.dumps({name: metrics}), flush=True)
    return metrics


def dev_loss(model, encoded):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for prompt, answer in encoded:
            loss = token_loss(model, prompt, answer)
            total += loss.item() * len(answer)
            count += len(answer)
    return total / count


def train(model, tokenizer, train_rows, dev_rows, args):
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules="model.language_model.layers.*.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    train_encoded = [encode(tokenizer, r) for r in train_rows]
    dev_encoded = [encode(tokenizer, r) for r in dev_rows]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    artifact_root = Path("models/local_20261004_lora")
    artifact_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    initial_dev = dev_loss(model, dev_encoded)
    best_dev = initial_dev
    best_path = None
    losses = []
    logs = ROOT / "training_log.jsonl"
    step = 0
    actual_samples = 0
    with logs.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"step": 0, "dev_nll": initial_dev}) + "\n")
        for epoch in range(args.epochs):
            order = list(range(len(train_encoded)))
            random.Random(SEED + epoch).shuffle(order)
            for offset in range(0, len(order), args.accumulation):
                if step >= args.max_steps or time.time() - start > args.max_train_seconds:
                    break
                model.train()
                indices = order[offset:offset + args.accumulation]
                loss_value = 0.0
                for index in indices:
                    prompt, answer = train_encoded[index]
                    loss = token_loss(model, prompt, answer)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss; stopping without claiming completion")
                    (loss / len(indices)).backward()
                    loss_value += loss.item() / len(indices)
                    actual_samples += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                warmup = min(1.0, (step + 1) / 15)
                decay = max(0.1, 1 - step / max(1, args.max_steps))
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * warmup * decay
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                losses.append(loss_value)
                item = {"step": step, "epoch": epoch + 1, "samples_seen": actual_samples,
                        "loss": loss_value, "grad_norm": float(grad_norm), "seconds": time.time() - start,
                        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
                if step % args.checkpoint_steps == 0 or step == args.max_steps:
                    validation = dev_loss(model, dev_encoded)
                    checkpoint = artifact_root / f"step-{step}"
                    model.save_pretrained(checkpoint)
                    tokenizer.save_pretrained(checkpoint)
                    torch.save({"optimizer": optimizer.state_dict(), "step": step, "seed": SEED}, checkpoint / "optimizer_state.pt")
                    item["dev_nll"] = validation
                    if validation < best_dev:
                        best_dev, best_path = validation, str(checkpoint)
                log.write(json.dumps(item) + "\n")
                log.flush()
                if step % 10 == 0:
                    print(json.dumps(item), flush=True)
            if step >= args.max_steps or time.time() - start > args.max_train_seconds:
                break
    final_dir = artifact_root / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    summary = {"steps": step, "samples_seen": actual_samples, "seconds": time.time() - start,
               "mean_train_loss": float(np.mean(losses)), "initial_dev_nll": initial_dev,
               "best_dev_nll": best_dev, "selected_adapter": best_path,
               "selection_rule": "minimum dev token NLL, including unmodified base as candidate",
               "final_adapter": str(final_dir), "trainable_parameters": sum(p.numel() for p in params),
               "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
               "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
               "training_completed_planned_steps": step == args.max_steps}
    write_json(ROOT / "training_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "baseline", "train", "eval", "rag-eval"], required=True)
    parser.add_argument("--model", default="models/Qwen3.5-2B")
    parser.add_argument("--adapter")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-steps", type=int, default=150)
    parser.add_argument("--max-train-seconds", type=int, default=5400)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    seed_all()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    splits = {s: read(ROOT / f"{s}.jsonl") for s in ("train", "dev", "test")}
    audit = json.loads((ROOT / "data_audit.json").read_text())
    for split in splits:
        assert hashlib.sha256((ROOT / f"{split}.jsonl").read_bytes()).hexdigest() == audit["splits"][split]["sha256"]
    if args.stage != "preflight":
        plan = json.loads((ROOT / "frozen_experiment_plan.json").read_text(encoding="utf-8"))
        for key in ("model", "max_steps", "epochs", "accumulation", "learning_rate",
                    "checkpoint_steps", "max_train_seconds", "max_new_tokens"):
            if getattr(args, key) != plan["config"][key]:
                raise ValueError(f"Frozen experiment setting changed: {key}. Use a new experiment directory.")
    write_json(ROOT / f"{args.stage.replace('-', '_')}_{'adapter' if args.adapter else 'default'}_invocation.json",
               {"arguments": vars(args), "utc": datetime.now(timezone.utc).isoformat(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "data_hashes": {s: audit["splits"][s]["sha256"] for s in splits}})
    if args.stage == "rag-eval":
        rows = read(ROOT / "retrieved_context_test.jsonl")
        model = load_model(args.model, args.adapter)
        name = "adapter_retrieved_context_test" if args.adapter else "base_retrieved_context_test"
        evaluate(model, tokenizer, rows, name, args.max_new_tokens)
    elif args.stage == "preflight":
        lengths = {s: [sum(map(len, encode(tokenizer, r))) for r in rows] for s, rows in splits.items()}
        plan = {"model": args.model, "seed": SEED, "dtype": "bfloat16", "quantization": None,
                "training_modality": "text_only_vision_encoder_frozen", "config": vars(args),
                "selection_rule": "minimum dev token NLL, including base; no test-based tuning",
                "prompt": SYSTEM, "decode": "greedy, thinking disabled, max_new_tokens=128",
                "data_hashes": {s: audit["splits"][s]["sha256"] for s in splits},
                "max_total_tokens": {s: max(v) for s, v in lengths.items()},
                "scope": "Given-evidence answer adaptation, not end-to-end RAG or production quality",
                "metrics": "answerable content F1; exact source citation; lexical support proxy; abstention precision/recall; latency",
                "created_utc": datetime.now(timezone.utc).isoformat()}
        write_json(ROOT / "frozen_experiment_plan.json", plan)
        worst = max(splits["train"], key=lambda r: sum(map(len, encode(tokenizer, r))))
        model = load_model(args.model)
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules="model.language_model.layers.*.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", task_type="CAUSAL_LM"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        started = time.time()
        loss = token_loss(model, *encode(tokenizer, worst))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        result = {"status": "passed", "tokens": max(lengths["train"]), "loss": loss.item(),
                  "seconds": time.time() - started, "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
        write_json(ROOT / "real_length_preflight.json", result)
        print(json.dumps(result), flush=True)
    elif args.stage == "baseline":
        model = load_model(args.model)
        evaluate(model, tokenizer, splits["dev"], "base_dev", args.max_new_tokens)
        evaluate(model, tokenizer, splits["test"], "base_test", args.max_new_tokens)
    elif args.stage == "train":
        model = load_model(args.model)
        train(model, tokenizer, splits["train"], splits["dev"], args)
    else:
        summary = json.loads((ROOT / "training_summary.json").read_text())
        adapter = args.adapter or summary["selected_adapter"] or summary["final_adapter"]
        model = load_model(args.model, adapter)
        write_json(ROOT / "evaluated_adapter.json", {"adapter": adapter, "selected_over_base": summary["selected_adapter"] is not None})
        evaluate(model, tokenizer, splits["dev"], "adapter_dev", args.max_new_tokens)
        evaluate(model, tokenizer, splits["test"], "adapter_test", args.max_new_tokens)


if __name__ == "__main__":
    main()
