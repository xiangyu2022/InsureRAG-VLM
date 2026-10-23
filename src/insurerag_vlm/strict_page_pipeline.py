"""Bridge the new page policy into existing hybrid/UI code without replacing it."""
from hashlib import sha256
import json
from pathlib import Path

from .data import _load_packet_manifest
from .page_ingest import ingest_pages, LocalVisionRecognizer


def strict_document_corpus(pipeline, folder):
    from .hybrid_pipeline import DocumentTextUnavailableError
    folder = Path(folder)
    originals = [p for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix.lower() in
                 {'.pdf', '.txt', '.md', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}]
    defaults = _load_packet_manifest(folder)
    records, statuses = [], []
    code_identity = sha256(Path(__file__).with_name('page_ingest.py').read_bytes() +
                           Path(__file__).with_name('page_schema.py').read_bytes()).hexdigest()
    for source in originals:
        relative = source.relative_to(folder).as_posix()
        identity = sha256((relative + sha256(source.read_bytes()).hexdigest() + code_identity +
                           str(pipeline.config.page_vision_model) + str(pipeline.config.page_vision_digest)).encode()).hexdigest()
        destination = Path(pipeline.config.index_dir) / 'strict_pages_v2' / identity
        if (destination / 'report.json').exists():
            report = json.loads((destination / 'report.json').read_text(encoding='utf-8'))
            chunks = [json.loads(line) for line in (destination / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        else:
            recognizer = None
            if pipeline.config.page_vision_model:
                recognizer = LocalVisionRecognizer(pipeline.config.page_vision_model, pipeline.config.page_vision_digest)
            chunks, report = ingest_pages(source, destination, doc_id=relative, recognizer=recognizer)
        for page in report['pages']:
            statuses.append({'source': f'{relative}#page={page["page_number"]}',
                             'status': page['status'], 'route': page['route'],
                             'text_characters': sum(len(r['text']) for r in chunks if r['page_number'] == page['page_number']),
                             'warning': page.get('reason')})
        records += chunks
    pipeline.ingestion_report = {'policy': 'page_v2', 'answer_input': 'page-routed', 'total_pages': len(statuses),
        'readable_pages': sum(s['status'] == 'extracted_unreviewed' for s in statuses), 'blank_pages': 0,
        'unreadable_pages': sum(s['status'] != 'extracted_unreviewed' for s in statuses), 'pages': statuses}
    if not statuses or pipeline.ingestion_report['unreadable_pages']:
        raise DocumentTextUnavailableError('Strict page ingestion pending: compressed multimodal recognition is required '
            'for visual/mixed pages; no OCR or text-layer fallback. Use ingest-pages with an available approved local vision model. '
            'This upload was not activated.')
    pages, snippets, seen = [], [], set()
    for record in records:
        relative = record['doc_id']
        extra = defaults.get(relative, {})
        if any(key in record and record[key] != value for key, value in extra.items()):
            raise ValueError('Packet metadata conflicts with mandatory page provenance')
        page_key = record['page_id']
        metadata = {**extra, **record, 'record_type': 'snippet', 'page_key': page_key,
                    'parent_page_id': page_key, 'source': f'{relative}#page={record["page_number"]}',
                    'image_path': None}
        snippets.append(pipeline._augment_record_metadata(metadata))
        if page_key not in seen:
            seen.add(page_key)
            page = {**metadata, 'record_id': page_key, 'record_type': 'page',
                    'text': '\n\n'.join(b['text'] for b in record['page_metadata']['blocks'])}
            pages.append(pipeline._augment_record_metadata(page))
    return {'pages': pages, 'snippets': snippets}
