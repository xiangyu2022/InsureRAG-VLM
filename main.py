import sys

from src.insurerag_vlm.cli import main

if __name__ == "__main__":
    # JSON and public PDF excerpts contain Unicode even on legacy Windows
    # consoles. Keep redirected output decodable without locale-specific loss.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == 'ingest-pages':
        from src.insurerag_vlm.page_ingest_cli import main as ingest_pages_main
        raise SystemExit(ingest_pages_main(sys.argv[2:]))
    main()
