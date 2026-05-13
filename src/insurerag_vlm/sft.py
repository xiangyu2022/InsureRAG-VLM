import json
import os
import signal
from hashlib import sha256
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_QWEN_7B_MODEL = os.environ.get("INSURERAG_QWEN_7B_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class QwenLoraSFTConfig:
    dataset_path: Path = Path("data/04_curated/sft_dataset.jsonl")
    output_dir: Path = Path("models/qwen7b-insurerag-lora")
    model_name: str = DEFAULT_QWEN_7B_MODEL
    max_samples: Optional[int] = None
    max_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    load_in_4bit: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True
    resume_from_checkpoint: Optional[Path] = None
    auto_resume: bool = False
    lora_target_modules: List[str] = None

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)
        if self.lora_target_modules is None:
            self.lora_target_modules = list(DEFAULT_LORA_TARGET_MODULES)


@dataclass
class SFTSmokeResult:
    dataset_path: Path
    model_name: str
    sample_count: int
    first_record_id: str
    first_prompt_chars: int
    first_answer_chars: int
    first_token_count: int
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: Optional[str]


def read_sft_records(path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if max_samples is not None and len(records) >= max_samples:
                    break
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list or JSONL records at {path}")
        records = payload[:max_samples] if max_samples is not None else payload

    cleaned = []
    for record in records:
        if not record.get("question") or not record.get("answer"):
            continue
        cleaned.append(record)
    if not cleaned:
        raise ValueError(f"No usable SFT records found in {path}")
    return cleaned


def format_sft_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    instruction = str(record.get("instruction") or "").strip()
    evidence = str(record.get("evidence") or "").strip()
    question = str(record.get("question") or "").strip()
    source = str(record.get("source") or "").strip()
    answer = str(record.get("answer") or "").strip()

    user_parts = []
    if instruction:
        user_parts.append(f"Instruction:\n{instruction}")
    if evidence:
        user_parts.append(f"Evidence:\n{evidence}")
    if source:
        user_parts.append(f"Source:\n{source}")
    user_parts.append(f"Question:\n{question}")

    return [
        {
            "role": "system",
            "content": (
                "You are InsureRAG, an insurance-domain assistant. Answer only from the "
                "provided evidence, explain insurance terms when useful, and cite the source page."
            ),
        },
        {"role": "user", "content": "\n\n".join(user_parts)},
        {"role": "assistant", "content": answer},
    ]


def _render_prompt_and_full_text(tokenizer: Any, messages: List[Dict[str, str]]) -> tuple[str, str]:
    prompt_messages = messages[:-1]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        prompt_text = (
            f"System: {prompt_messages[0]['content']}\n\n"
            f"User: {prompt_messages[1]['content']}\n\nAssistant:"
        )
        full_text = f"{prompt_text} {messages[-1]['content']}"
    if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
        full_text += tokenizer.eos_token
    return prompt_text, full_text


def tokenize_sft_record(tokenizer: Any, record: Dict[str, Any], max_length: int) -> Dict[str, List[int]]:
    messages = format_sft_messages(record)
    prompt_text, full_text = _render_prompt_and_full_text(tokenizer, messages)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = full["input_ids"]
    labels = list(input_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(label == -100 for label in labels):
        labels[-1] = input_ids[-1]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class SFTDataset:
    def __init__(self, records: List[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples = [tokenize_sft_record(tokenizer, record, max_length) for record in records]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


class CausalLMDataCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class _FormatOnlyTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0
    chat_template = None

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        del add_special_tokens
        tokens = text.split()
        ids = list(range(1, len(tokens) + 1))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids or [1]}


class _TrainerCallbackCompat:
    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            return lambda args, state, control, **kwargs: control
        raise AttributeError(name)


def _require_cuda() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for Qwen LoRA SFT. Install requirements-gpu.txt first.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen 7B LoRA SFT, but torch.cuda.is_available() is false.")
    return torch


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for Qwen LoRA SFT.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _iter_trainable_parameters(model: Any) -> Iterable[tuple[str, Any]]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def _find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = []
    for candidate in Path(output_dir).glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        try:
            step = int(candidate.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, candidate))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


class _SignalSaveCallback(_TrainerCallbackCompat):
    def __init__(self) -> None:
        self._signal_name: Optional[str] = None
        self._previous_handlers: Dict[int, Any] = {}

    def register(self) -> None:
        for signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if signum is None:
                continue
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def restore(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self._signal_name = signal.Signals(signum).name
        print(f"Received {self._signal_name}; will save a checkpoint and stop training.", flush=True)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, state, kwargs
        if self._signal_name:
            control.should_save = True
            control.should_training_stop = True
        return control


class _SFTProgressCallback(_TrainerCallbackCompat):
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.progress_path = self.output_dir / "sft_progress.json"

    def _write_progress(self, payload: Dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, control, kwargs
        self._write_progress(
            {
                "status": "running",
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": float(state.epoch or 0.0),
                "last_checkpoint": None,
            }
        )

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        del args, control, kwargs
        payload = {
            "status": "running",
            "global_step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "epoch": float(state.epoch or 0.0),
            "logs": logs or {},
        }
        if self.progress_path.exists():
            existing = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if existing.get("last_checkpoint"):
                payload["last_checkpoint"] = existing["last_checkpoint"]
        self._write_progress(payload)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, control, kwargs
        checkpoint_dir = self.output_dir / f"checkpoint-{int(state.global_step)}"
        payload = {
            "status": "running",
            "global_step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "epoch": float(state.epoch or 0.0),
            "last_checkpoint": str(checkpoint_dir),
        }
        self._write_progress(payload)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, control, kwargs
        payload = {
            "status": "completed",
            "global_step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "epoch": float(state.epoch or 0.0),
        }
        latest = _find_latest_checkpoint(self.output_dir)
        if latest is not None:
            payload["last_checkpoint"] = str(latest)
        self._write_progress(payload)


def run_lora_sft(config: QwenLoraSFTConfig) -> Dict[str, Any]:
    torch = _require_cuda()
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("Install GPU SFT dependencies: pip install -r requirements-gpu.txt") from exc
    if config.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The current CUDA device does not support bf16. Re-run with --fp16.")

    tokenizer = _load_tokenizer(config.model_name)
    records = read_sft_records(config.dataset_path, max_samples=config.max_samples)
    train_dataset = SFTDataset(records, tokenizer, config.max_length)

    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=not config.bf16,
        optim="paged_adamw_8bit" if config.load_in_4bit else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        seed=config.seed,
    )
    signal_callback = _SignalSaveCallback()
    progress_callback = _SFTProgressCallback(config.output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalLMDataCollator(tokenizer),
        callbacks=[signal_callback, progress_callback],
    )
    resume_checkpoint = None
    if config.resume_from_checkpoint is not None:
        resume_checkpoint = str(config.resume_from_checkpoint)
    elif config.auto_resume:
        latest_checkpoint = _find_latest_checkpoint(config.output_dir)
        if latest_checkpoint is not None:
            resume_checkpoint = str(latest_checkpoint)
            print(f"Auto-resuming from checkpoint: {resume_checkpoint}", flush=True)

    signal_callback.register()
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    finally:
        signal_callback.restore()
    peak_memory_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    trainable_params = sum(parameter.numel() for _, parameter in _iter_trainable_parameters(model))
    total_params = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "train_samples": len(records),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_loss": getattr(train_result, "training_loss", None),
        "dataset_sha256": _file_sha256(config.dataset_path),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "peak_cuda_memory_mb": peak_memory_mb,
        "resume_from_checkpoint": resume_checkpoint,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
    }
    (config.output_dir / "sft_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_lora_smoke_test(
    dataset_path: Path,
    model_name: str = DEFAULT_QWEN_7B_MODEL,
    max_length: int = 1024,
    skip_cuda_check: bool = False,
    format_only: bool = False,
) -> SFTSmokeResult:
    torch = None
    if not format_only:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for the LoRA smoke test. Install requirements-gpu.txt first.") from exc
        if not skip_cuda_check and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for the smoke test. Use --skip-cuda-check only for CPU syntax checks."
            )

    tokenizer = _FormatOnlyTokenizer() if format_only else _load_tokenizer(model_name)
    records = read_sft_records(dataset_path, max_samples=1)
    tokenized = tokenize_sft_record(tokenizer, records[0], max_length=max_length)
    messages = format_sft_messages(records[0])
    prompt_text, _ = _render_prompt_and_full_text(tokenizer, messages)
    cuda_available = bool(torch and torch.cuda.is_available())
    return SFTSmokeResult(
        dataset_path=Path(dataset_path),
        model_name=model_name,
        sample_count=1,
        first_record_id=str(records[0].get("record_id") or ""),
        first_prompt_chars=len(prompt_text),
        first_answer_chars=len(str(records[0].get("answer") or "")),
        first_token_count=len(tokenized["input_ids"]),
        cuda_available=cuda_available,
        cuda_device_count=torch.cuda.device_count() if torch else 0,
        cuda_device_name=torch.cuda.get_device_name(0) if cuda_available else None,
    )
