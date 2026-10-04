"""Wait for the final validation lock, then execute its one registered test run."""
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def run():
    lock=ROOT/'reports/condition_listwise_v1/selection.lock.json';start=time.monotonic()
    while not lock.exists():
        if time.monotonic()-start>3600:raise TimeoutError('Final selection did not complete')
        time.sleep(5)
    for script in ['run_condition_listwise_tests.py','verify_condition_listwise.py','analyze_condition_listwise_results.py','write_condition_listwise_model_cards.py']:
        print('Running '+script,flush=True);subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,check=True)
if __name__=='__main__':run()
