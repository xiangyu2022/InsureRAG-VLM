"""Run both registered seeds sequentially without sharing optimizer state."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    for seed in [42, 123]:
        subprocess.run([sys.executable, 'scripts/train_query_adaptation.py', '--seed', str(seed)], cwd=ROOT, check=True)
