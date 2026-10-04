"""Finite, offline single-step BF16 LoRA feasibility check on a public model."""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForImageTextToText


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3.5-2B")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules="model.language_model.layers.*.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    messages = [{"role": "system", "content": "Answer only from the evidence."},
                {"role": "user", "content": "Evidence: The collision deductible is $500. What is the deductible?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer("The collision deductible is $500." + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prompt_ids + answer_ids], device="cuda")
    labels = ids.clone()
    labels[:, :len(prompt_ids)] = -100
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    step_start = time.time()
    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    output = {"status": "passed", "model": args.model, "loss": loss.item(), "tokens": ids.shape[1],
              "step_seconds": time.time() - step_start, "total_seconds": time.time() - started,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20}
    path = Path("reports/local_20261004") / (Path(args.model).name + "_smoke.json")
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
