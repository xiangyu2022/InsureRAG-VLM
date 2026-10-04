import hashlib

import pytest

from scripts.audit_qwen35_adapter import REVISION,authenticate_files,git_blob_sha1


def test_git_blob_authentication_uses_git_object_hash_instead_of_plain_file_sha1(tmp_path):
    empty=tmp_path/'empty.txt'
    empty.write_bytes(b'')
    assert git_blob_sha1(empty)=='e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
    assert git_blob_sha1(empty)!=hashlib.sha1(b'').hexdigest()


def test_hub_audit_supports_lfs_sha256_and_small_file_git_objects(tmp_path):
    (tmp_path/'config.json').write_bytes(b'{}')
    (tmp_path/'weights.safetensors').write_bytes(b'synthetic fixture')
    local={name:{'sha256':hashlib.sha256((tmp_path/name).read_bytes()).hexdigest(),
                 'bytes':(tmp_path/name).stat().st_size} for name in ['config.json','weights.safetensors']}
    info={'sha':REVISION,'siblings':[
        {'rfilename':'config.json','size':2,'blobId':git_blob_sha1(tmp_path/'config.json')},
        {'rfilename':'weights.safetensors','size':17,'lfs':{'sha256':local['weights.safetensors']['sha256']}},
    ]}
    assert all(row['matched'] for row in authenticate_files(tmp_path,local,info).values())
    info['siblings'][1]['lfs']['sha256']='wrong'
    assert not authenticate_files(tmp_path,local,info)['weights.safetensors']['matched']


def test_official_metadata_revision_must_match_the_requested_commit(tmp_path):
    with pytest.raises(ValueError,match='different revision'):
        authenticate_files(tmp_path,{}, {'sha':'untrusted','siblings':[]})
