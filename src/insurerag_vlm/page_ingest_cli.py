"""Strict page-ingest -> isolated schema-v2 Chroma collection."""
import argparse
import json
from pathlib import Path

from .chroma_knowledge import ClauseStore
from .page_ingest import ingest_pages, LocalVisionRecognizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--doc-id', required=True, help='Stable document identity, independent of source revision')
    parser.add_argument('--work-dir', type=Path, required=True, help='New private directory; never overwrite prior artifacts')
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--collection', default='page_clauses_v2')
    parser.add_argument('--vision-model')
    parser.add_argument('--vision-digest')
    parser.add_argument('--max-chars', type=int, default=1600, help='Soft budget at semantic boundaries; never raw truncation')
    parser.add_argument('--max-edge', type=int, default=2000)
    parser.add_argument('--jpeg-quality', type=int, default=85)
    args = parser.parse_args(argv)
    if bool(args.vision_model) != bool(args.vision_digest):
        parser.error('Explicit existing local vision model and digest must be supplied together')
    recognizer = LocalVisionRecognizer(args.vision_model, args.vision_digest) if args.vision_model else None
    records, report = ingest_pages(args.source, args.work_dir, doc_id=args.doc_id, recognizer=recognizer,
                                  maximum=args.max_chars, max_edge=args.max_edge, quality=args.jpeg_quality)
    if report['status'] != 'complete_unreviewed':
        print(json.dumps({**report, 'indexed': False}, ensure_ascii=True))
        return 2
    with ClauseStore(args.store, collection=args.collection, writable=True, record_schema=2) as store:
        result = store.upsert(records)
    print(json.dumps({**report, 'indexed': True, 'index': result, 'collection': args.collection}, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
