"""Verify completed controls, then run the once-selected test evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    run = ROOT / 'reports/query_adaptation_v1'
    selection = run / 'selection.lock.json'
    expected = hashlib.sha256(selection.read_bytes()).hexdigest()
    complete = run / 'data_ablation/valid/summary.json'
    deadline = time.monotonic()+3600
    while not complete.exists():
        if time.monotonic() > deadline: raise TimeoutError('Ablation validation did not complete')
        time.sleep(10)
    assert json.loads(complete.read_text(encoding='utf8'))['status'] == 'completed'
    assert not (run / 'data_ablation/selection.lock.json').exists()
    assert hashlib.sha256(selection.read_bytes()).hexdigest() == expected
    subprocess.run([sys.executable, 'scripts/verify_query_adaptation.py', '--require-ablation'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'scripts/eval_query_adaptation.py', 'test'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'scripts/verify_query_adaptation.py', '--require-ablation', '--require-test'], cwd=ROOT, check=True)
