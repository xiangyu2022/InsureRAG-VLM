"""Optional local cross-encoder. Query and passage text are its only inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np


def minmax_scores(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Scores must be a finite one-dimensional vector')
    if not len(values):
        return values
    span = float(np.max(values) - np.min(values))
    return (values - np.min(values)) / span if span > 0 else np.zeros_like(values)


def blend_scores(dense_scores, lexical_scores, cross_scores=None,
                 lexical_weight=0.2, cross_weight=0.0):
    """Normalize each channel over the same candidate set; no labels accepted."""
    if not 0 <= lexical_weight <= 1 or not 0 <= cross_weight <= 1:
        raise ValueError('Fusion weights must lie in [0, 1]')
    dense, lexical = minmax_scores(dense_scores), minmax_scores(lexical_scores)
    if dense.shape != lexical.shape:
        raise ValueError('Dense and lexical scores must describe the same candidates')
    first_stage = (1-lexical_weight)*dense + lexical_weight*lexical
    if cross_weight == 0:
        return first_stage
    if cross_scores is None:
        raise ValueError('A positive cross weight requires cross-encoder scores')
    cross = minmax_scores(cross_scores)
    if cross.shape != dense.shape:
        raise ValueError('Cross scores must describe the same candidates')
    return (1-cross_weight)*first_stage + cross_weight*cross


class LocalCrossEncoder:
    def __init__(self, model_path, device='cpu', batch_size=64, max_length=512):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.path = Path(model_path).resolve()
        if not self.path.is_dir():
            raise ValueError('Reranker requires an existing local checkpoint directory')
        if device not in {'cpu', 'cuda'} or (device == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Requested reranker device is unavailable')
        if batch_size < 1 or max_length < 1:
            raise ValueError('Batch size and maximum length must be positive')
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
            dtype=torch.float16 if device == 'cuda' else torch.float32)
        if self.model.config.num_labels != 1:
            raise ValueError('Reranker must emit exactly one relevance logit per pair')
        if max_length > self.model.config.max_position_embeddings:
            raise ValueError('Requested sequence length exceeds checkpoint capacity')
        self.model.to(device).eval()
        self.pairs_scored = 0

    def fingerprint(self):
        relevant = [p for p in sorted(self.path.iterdir()) if p.is_file()
                    and (p.suffix in {'.json', '.safetensors', '.txt'})]
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in relevant}
        return {'model_path': str(self.path), 'files_sha256': files,
                'checkpoint_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                'device': self.device, 'dtype': str(next(self.model.parameters()).dtype),
                'max_length': self.max_length, 'batch_size': self.batch_size,
                'truncation': 'longest_first', 'score': 'raw single relevance logit',
                'remote_code': False, 'fine_tuned_in_this_project': False}

    def score_pairs(self, pairs):
        import torch
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        # Length bucketing reduces padding without changing the caller's order.
        order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0])+len(pairs[i][1]), i))
        result = np.empty(len(pairs), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start+self.batch_size]
                features = self.tokenizer([pairs[i][0] for i in batch], [pairs[i][1] for i in batch],
                    padding=True, truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)
                logits = self.model(**features).logits.reshape(-1).float().cpu().numpy()
                if len(logits) != len(batch) or not np.isfinite(logits).all():
                    raise ValueError('Reranker emitted invalid relevance scores')
                result[batch] = logits
        self.pairs_scored += len(pairs)
        return result

    def score(self, question, passages):
        return self.score_pairs([(question, passage) for passage in passages])
