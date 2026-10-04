"""Restore checksum-verified release artifacts without replacing Git-tracked files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import urllib.request
import zipfile


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def destination(name: str, repo: Path) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    if not relative.parts:
        raise ValueError("Empty archive member")
    if relative.parts[0] == "InsureRAG-VLM":
        target = repo.joinpath(*relative.parts[1:])
    else:
        target = repo.parent.joinpath(*relative.parts)
    resolved = target.resolve()
    if not resolved.is_relative_to(repo.parent.resolve()):
        raise ValueError(f"Archive member escapes the workspace: {name}")
    if any(p in {".git", ".venv", ".ssh"} for p in relative.parts):
        raise ValueError(f"Protected archive member: {name}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    top = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=repo, text=True).strip()).resolve()
    if top != repo.resolve():
        raise RuntimeError("Run this script from its Git clone, not a source ZIP")
    tracked = {
        (repo / name.decode("utf8")).resolve()
        for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).split(b"\0") if name
    }
    manifest = json.loads((repo / "docs/releases/2026-10-03/research-assets.json").read_text(encoding="utf8"))
    archive_dir = (args.archives_dir or repo.parent / "research-release-downloads").resolve()
    for asset in manifest["archives"]:
        archive = archive_dir / asset["filename"]
        valid = archive.is_file() and archive.stat().st_size == asset["bytes"] and sha256(archive) == asset["sha256"]
        if not valid:
            if args.offline or args.dry_run:
                raise RuntimeError(f"Missing or checksum-mismatched archive: {archive}")
            archive_dir.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_suffix(archive.suffix + ".part")
            print(f"Downloading {asset['filename']} ({asset['bytes']:,} bytes)", flush=True)
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "InsureRAG-research-restore"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            if temporary.stat().st_size != asset["bytes"] or sha256(temporary) != asset["sha256"]:
                raise RuntimeError(f"Downloaded checksum mismatch: {temporary}")
            temporary.replace(archive)
        restored = preserved = 0
        with zipfile.ZipFile(archive) as contents:
            for entry in contents.infolist():
                if entry.is_dir():
                    continue
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f"Symlink member is not allowed: {entry.filename}")
                target = destination(entry.filename, repo)
                if target in tracked:
                    preserved += 1
                    continue
                if not args.dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with contents.open(entry) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                restored += 1
        print(json.dumps({"archive": asset["filename"], "sha256_verified": True,
                          "preserved_git_files": preserved, "restored_files": restored,
                          "dry_run": args.dry_run}), flush=True)


if __name__ == "__main__":
    main()
