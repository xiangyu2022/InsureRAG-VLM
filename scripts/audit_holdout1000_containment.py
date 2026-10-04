"""Preserve a fresh evidence-containment audit; never auto-accept candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.insurerag_vlm.holdout_containment import evidence_windows, scan_history, verify_evidence_projection


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--field', choices=['question', 'answer'], default='answer')
    parser.add_argument('--documents', type=Path, help='Source JSONL required for verified evidence mode')
    parser.add_argument('--text-mode', action='store_true', help='Generic text screening only; not verified source-evidence coverage')
    parser.add_argument('--window-words', type=int, default=16)
    parser.add_argument('--minimum-words', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous audit; use a fresh output path')
    started = time.perf_counter()
    with args.candidates.open(encoding='utf8') as stream:
        candidates = [json.loads(line) for line in stream if line.strip()]
    if not args.text_mode:
        if args.documents is None or args.field!='answer':
            parser.error('Evidence mode requires --documents and --field answer; use --text-mode for generic text')
        with args.documents.open(encoding='utf8') as stream:
            documents={row['id']:row for row in (json.loads(line) for line in stream if line.strip())}
        verify_evidence_projection(candidates,documents)
    index = evidence_windows(candidates, args.field, args.window_words, args.minimum_words)
    def progress(count, matches):
        print(json.dumps({'historical_processed': count, 'matched_windows': matches}), flush=True)
    with args.history.open(encoding='utf8') as stream:
        result = scan_history(index, (json.loads(line) for line in stream if line.strip()), progress)
    report = {
        'status': 'REVIEW_FLAGS_NOT_INDEPENDENCE_CLEARANCE',
        'method': 'Every contiguous normalized '+('text' if args.text_mode else 'verified evidence')+' window; original and Cf-stripped candidate variants; histories normalized record by record without cross-record joins.',
        'candidate_count': len(candidates), 'field': args.field,
        'evidence_projection_verified':not args.text_mode,
        'window_words': args.window_words, 'minimum_words': args.minimum_words,
        'input_sha256': {'candidates': digest(args.candidates), 'history': digest(args.history)},
        **result, 'seconds': time.perf_counter() - started,
        'accepted_items': 0,
        'limitations': [
            'Common wording can match without duplicate information needs; every flag requires disposition.',
            'No-match results can miss paraphrases or overlaps shorter than the configured window.',
            'Coverage combines matches across historical records; it does not imply a single historical record contains the entire evidence.',
            'Previously normalized history cannot reconstruct characters or word joins lost by its exporter.',
            'Denied paths, unextracted historical PDFs and foundation-model pretraining remain outside this audit.',
        ],
    }
    if not args.text_mode:report['input_sha256']['documents']=digest(args.documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'matches', 'coverage'}}, indent=2))


if __name__ == '__main__':
    main()
