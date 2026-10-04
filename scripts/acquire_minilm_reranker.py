"""Download an explicitly pinned public reranker; never download executable code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L6-v2'


def acquire(output: Path, revision: str | None = None):
    info = HfApi().model_info(MODEL_ID, revision=revision)
    pinned = info.sha
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, revision=pinned, local_dir=output,
                      allow_patterns=['config.json', 'tokenizer_config.json', 'tokenizer.json',
                                      'special_tokens_map.json', 'vocab.txt', 'model.safetensors',
                                      'README.md', 'LICENSE*'])
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.iterdir()) if p.is_file() and p.name != 'download_provenance.json'}
    manifest = {'model_id': MODEL_ID, 'revision': pinned,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'source': f'https://huggingface.co/{MODEL_ID}/tree/{pinned}',
                'files_sha256': files}
    (output/'download_provenance.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision')
    args = parser.parse_args()
    acquire(args.output, args.revision)
