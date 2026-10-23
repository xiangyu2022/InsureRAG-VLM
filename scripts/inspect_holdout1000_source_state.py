#!/usr/bin/env python3
"""Inspect existing local source decisions before authoring another holdout batch."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.insurerag_vlm.holdout_source_state import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Holdout report directory containing local/")
    parser.add_argument("--pool-number", type=int, required=True, help="Authoritative provisional pool version")
    parser.add_argument("--document", action="append", required=True, help="Document ID; repeat for a batch")
    args = parser.parse_args()
    if args.pool_number < 1:
        parser.error("--pool-number must be positive")
    print(json.dumps(source_state(args.root, args.document, args.pool_number), indent=2))


if __name__ == "__main__":
    main()
