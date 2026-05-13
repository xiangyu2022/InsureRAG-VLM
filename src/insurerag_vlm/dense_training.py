import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DENSE_BASE_MODEL = os.environ.get("INSURERAG_DENSE_BASE_MODEL", "BAAI/bge-base-en-v1.5")


@dataclass
class DenseRetrieverTrainConfig:
    dataset_path: Path
    output_dir: Path = Path("models/retrieval/bge-base-insurerag")
    model_name: str = DEFAULT_DENSE_BASE_MODEL
    max_samples: Optional[int] = None
    max_length: int = 384
    learning_rate: float = 2e-5
    num_train_epochs: float = 2.0
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    seed: int = 42
    bf16: bool = True
    margin: float = 0.2
    resume_from_checkpoint: Optional[Path] = None
    auto_resume: bool = False

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)


def _read_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("positive_page_texts") and row.get("hard_negative_texts"):
                rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No usable dense-retriever training rows found in {path}")
    return rows


class DenseTripletDataset:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.rows[idx]
        return {
            "query": str(row.get("question") or ""),
            "positive": str((row.get("positive_page_texts") or [""])[0]),
            "negative": str((row.get("hard_negative_texts") or [""])[0]),
        }


class DenseTripletCollator:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, Any]:
        query = self.tokenizer(
            [feature["query"] for feature in features],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        positive = self.tokenizer(
            [feature["positive"] for feature in features],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        negative = self.tokenizer(
            [feature["negative"] for feature in features],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {}
        for prefix, values in [("query", query), ("positive", positive), ("negative", negative)]:
            for key, tensor in values.items():
                batch[f"{prefix}_{key}"] = tensor
        return batch


def _auto_resume_checkpoint(output_dir: Path) -> Optional[Path]:
    candidates = sorted(
        [
            path for path in Path(output_dir).glob("checkpoint-*")
            if path.is_dir() and path.name.split("-")[-1].isdigit()
        ],
        key=lambda path: int(path.name.split("-")[-1]),
    )
    return candidates[-1] if candidates else None


def run_dense_retriever_training(config: DenseRetrieverTrainConfig) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

    rows = _read_jsonl(config.dataset_path, max_samples=config.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    class DenseTripletModel(torch.nn.Module):
        def __init__(self, model_name: str, margin: float):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.margin = margin

        @staticmethod
        def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            masked = last_hidden_state * attention_mask.unsqueeze(-1)
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return summed / counts

        def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embedding = self._mean_pool(outputs.last_hidden_state, attention_mask)
            return F.normalize(embedding, p=2, dim=-1)

        def forward(self, **batch: Any) -> Dict[str, torch.Tensor]:
            query = self._encode(batch["query_input_ids"], batch["query_attention_mask"])
            positive = self._encode(batch["positive_input_ids"], batch["positive_attention_mask"])
            negative = self._encode(batch["negative_input_ids"], batch["negative_attention_mask"])
            pos_scores = (query * positive).sum(dim=-1)
            neg_scores = (query * negative).sum(dim=-1)
            loss = F.relu(self.margin - pos_scores + neg_scores).mean()
            return {"loss": loss, "pos_scores": pos_scores.detach(), "neg_scores": neg_scores.detach()}

    train_dataset = DenseTripletDataset(rows)
    data_collator = DenseTripletCollator(tokenizer, max_length=config.max_length)
    model = DenseTripletModel(config.model_name, margin=config.margin)

    resume_from_checkpoint = config.resume_from_checkpoint
    if config.auto_resume and resume_from_checkpoint is None:
        resume_from_checkpoint = _auto_resume_checkpoint(config.output_dir)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        remove_unused_columns=False,
        bf16=config.bf16 and torch.cuda.is_available(),
        fp16=(not config.bf16) and torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "train_samples": len(train_dataset),
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    metadata_path = config.output_dir / "dense_retriever_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_dir": str(config.output_dir),
        "metadata_path": str(metadata_path),
        "train_samples": len(train_dataset),
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
    }
