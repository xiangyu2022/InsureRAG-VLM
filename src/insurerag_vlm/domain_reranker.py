"""Training-aware provenance adapter; frozen baseline scorer remains unchanged."""
import json
from .reranker import LocalCrossEncoder


class DomainCrossEncoder(LocalCrossEncoder):
    def fingerprint(self):
        payload = super().fingerprint()
        metadata = self.path/'training_provenance.json'
        payload['fine_tuned_in_this_project'] = metadata.exists()
        if metadata.exists():
            payload['training_provenance'] = json.loads(metadata.read_text(encoding='utf8'))
        return payload
