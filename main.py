import sys

from src.insurerag_vlm.cli import main

if __name__ == "__main__":
    # JSON and public PDF excerpts contain Unicode even on legacy Windows
    # consoles. Keep redirected output decodable without locale-specific loss.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
