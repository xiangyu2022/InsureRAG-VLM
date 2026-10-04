"""Wait for both completed training seeds, then perform registered validation."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    deadline = time.monotonic()+3600
    paths = [ROOT / f'reports/query_adaptation_v1/train_seed_{seed}/completion.json' for seed in [42, 123]]
    while not all(p.exists() for p in paths):
        if time.monotonic() > deadline: raise TimeoutError('Training did not complete')
        time.sleep(10)
    assert all(json.loads(p.read_text(encoding='utf8'))['status'] == 'completed' for p in paths)
    subprocess.run([sys.executable, 'scripts/eval_query_adaptation.py', 'valid'], cwd=ROOT, check=True)
