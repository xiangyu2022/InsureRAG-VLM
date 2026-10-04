"""Asymmetric query adaptation against an unchanged BGE document encoder."""
from pathlib import Path
import numpy as np

QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '


def blend_queries(original, learned, alpha):
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(learned, dtype=np.float32)
    if a.ndim != 2 or a.shape != b.shape or not 0 <= alpha <= 1:
        raise ValueError('Query blend shape/weight mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite query embeddings')
    mixed = (1-alpha)*a + alpha*b
    norm = np.linalg.norm(mixed, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError('Zero query embedding')
    return mixed / norm


def multi_positive_losses(logits, positives, teacher_logits):
    """All author positives/text aliases share probability mass, not negatives."""
    import torch
    if logits.ndim != 2 or logits.shape != positives.shape or logits.shape != teacher_logits.shape:
        raise ValueError('Loss shape mismatch')
    if positives.dtype != torch.bool or not positives.any(dim=1).all():
        raise ValueError('Each query needs a positive mask')
    if not torch.isfinite(logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError('Nonfinite retrieval logits')
    supervised = torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    retention = torch.nn.functional.kl_div(torch.log_softmax(logits, dim=1),
                                         torch.softmax(teacher_logits.detach(), dim=1),
                                         reduction='none').sum(dim=1)
    return supervised, retention


class QueryEncoder:
    """Encode queries only; callers must keep the pinned original document index."""
    def __init__(self, model_path, device='cpu'):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.path = Path(model_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True, trust_remote_code=False,
                                              dtype=torch.float32).to(device).eval()

    def encode(self, questions, batch_size=16):
        import torch
        output = []
        with torch.inference_mode():
            for start in range(0, len(questions), batch_size):
                features = self.tokenizer([QUERY_INSTRUCTION + q for q in questions[start:start+batch_size]],
                                          padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
                hidden = self.model(**features).last_hidden_state[:, 0].float()
                output.append(torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy())
        return np.vstack(output) if output else np.empty((0, 384), dtype=np.float32)
