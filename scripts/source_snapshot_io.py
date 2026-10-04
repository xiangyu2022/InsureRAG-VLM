"""Verify local source snapshots before trusting cached provenance."""
import hashlib,json,re
from pathlib import Path

def verified_metadata(path,expected_group=None,expected_url=None):
    path=Path(path);meta=json.loads(path.read_text(encoding='utf8'))
    if expected_group is not None and meta.get('group')!=expected_group:
        raise ValueError('Cached URL has a different publisher assignment')
    if expected_url is not None and meta.get('url')!=expected_url:
        raise ValueError('Cached metadata URL mismatch')
    artifact=meta.get('artifact')
    if artifact is None:
        if meta.get('error'):return meta
        raise ValueError('Successful snapshot metadata is missing its artifact')
    if not isinstance(artifact,str) or Path(artifact).name!=artifact or '/' in artifact or '\\' in artifact or artifact in {'.','..'}:
        raise ValueError('Snapshot artifact must be a local basename')
    digest=meta.get('sha256','')
    if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('Snapshot SHA256 missing or malformed')
    target=(path.parent/artifact).resolve()
    if target.parent!=path.parent.resolve():raise ValueError('Snapshot artifact resolves outside its cache directory')
    raw=target.read_bytes()
    if type(meta.get('bytes')) is not int or len(raw)!=meta['bytes'] or hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('Cached snapshot bytes do not match recorded provenance')
    return meta
