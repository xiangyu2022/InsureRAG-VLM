"""Hash final source/evidence/checkpoint files without changing experiment data."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("reports/local_20261004")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    files = set(ROOT.glob("*.*"))
    files.discard(ROOT / "deliverable_hashes.json")
    for folder in ("src", "scripts", "tests"):
        files.update(Path(folder).rglob("*.py"))
    files.add(Path("docs/local_experiment_20261004.md"))
    files.update(Path("models/local_20261004_lora/step-300").glob("*"))
    result = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "scope": "Final delivered files; not a retroactive claim about source bytes at initial training launch.",
              "files": {p.as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                        for p in sorted(files) if p.is_file()}}
    (ROOT / "deliverable_hashes.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"files": len(result["files"]),
                      "selected_adapter": result["files"]["models/local_20261004_lora/step-300/adapter_model.safetensors"]}, indent=2))


if __name__ == "__main__":
    main()
