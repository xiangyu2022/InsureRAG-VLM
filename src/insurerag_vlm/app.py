import json
import re
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import ModelConfig
from .diff import summarize_policy_diff
from .knowledge import format_knowledge_answer, knowledge_base_size, search_knowledge
from .pdf import extract_text_by_page
from .pipeline import DocumentRetrievalPipeline
from .vlm import _ANTHROPIC_SYSTEM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / "data" / "00_raw" / "external" / "public_docs"
INDEX_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / "reports" / "demo_uploads" / "current"
UPLOAD_INDEX_DIR = PROJECT_ROOT / "reports" / "demo_uploads" / "index"
PAGE_CACHE_DIR = PROJECT_ROOT / "reports" / "demo_uploads" / "page_cache"

_DIFF_TRIGGERS = re.compile(
    r"\b(compar|diff|version|drift|v1|v2|chang|updat)\w*\b", re.I
)


def _is_diff_query(question: str) -> bool:
    hits = _DIFF_TRIGGERS.findall(question)
    return len(hits) >= 2


def _is_document_first_query(question: str) -> bool:
    return bool(re.search(r"\b(document|policy|guide|pdf|uploaded|file)\b", question, re.I))


def _backend_label(pipeline: DocumentRetrievalPipeline) -> str:
    return pipeline.vlm_client.backend_label()


def _first_two_pdfs(folder: Path) -> tuple[Path, Path] | None:
    pdfs = sorted(Path(folder).rglob("*.pdf")) if Path(folder).exists() else []
    if len(pdfs) < 2:
        return None
    return pdfs[0], pdfs[1]


def _retrieval_trace(rag_result: dict, limit: int = 3) -> list[dict]:
    trace = []
    for rank, page in enumerate(rag_result.get("source_ranking", [])[:limit], start=1):
        trace.append(
            {
                "rank": rank,
                "source": page.get("source"),
                "score": round(float(page.get("score", 0.0)), 4),
                "page_number": page.get("page_number"),
                "snippet": page.get("text_snippet", ""),
            }
        )
    return trace


def build_chat_response(question: str, pipeline: DocumentRetrievalPipeline, data_folder: Path) -> dict:
    """Route question through knowledge base + RAG and return a unified response."""
    if _is_diff_query(question):
        return {"source": "diff"}

    vlm = pipeline.vlm_client
    kb_entries = search_knowledge(question)
    document_first = _is_document_first_query(question)

    # ── Path A: Ollama available ─────────────────────────────────────────────
    # Small local LLMs are best used as a last-mile explainer. For exact
    # glossary matches and cited document snippets, deterministic extraction is
    # faster and more faithful than asking a reasoning model to regenerate facts.
    if vlm.is_ollama():
        rag = pipeline.query_structured(question, data_folder, top_k=3, force_extractive=True)
        has_kb = len(kb_entries) > 0
        has_doc = not rag.get("abstain") and bool(rag.get("answer"))

        if has_doc and document_first:
            return {
                "source": "document",
                "answer": rag["answer"],
                "knowledge_terms": [],
                "citations": rag.get("citations", []),
                "confidence": rag.get("confidence", 0.0),
                "abstain": False,
                "abstain_reason": None,
                "backend": "deterministic evidence extraction",
                "retrieval_trace": _retrieval_trace(rag),
            }
        if has_kb and has_doc:
            return {
                "source": "combined",
                "answer": f"{format_knowledge_answer(kb_entries)}\n\n**From your policy documents:**\n{rag['answer']}",
                "knowledge_terms": [entry.term for entry in kb_entries],
                "citations": rag.get("citations", []),
                "confidence": rag.get("confidence", 0.0),
                "abstain": False,
                "abstain_reason": None,
                "backend": f"{_backend_label(pipeline)} + deterministic evidence extraction",
                "retrieval_trace": _retrieval_trace(rag),
            }
        if has_kb:
            return {
                "source": "knowledge",
                "answer": format_knowledge_answer(kb_entries),
                "knowledge_terms": [entry.term for entry in kb_entries],
                "citations": [],
                "confidence": 1.0,
                "abstain": False,
                "abstain_reason": None,
                "backend": "knowledge-base deterministic answer",
            }
        if has_doc:
            return {
                "source": "document",
                "answer": rag["answer"],
                "knowledge_terms": [],
                "citations": rag.get("citations", []),
                "confidence": rag.get("confidence", 0.0),
                "abstain": False,
                "abstain_reason": None,
                "backend": "deterministic evidence extraction",
                "retrieval_trace": _retrieval_trace(rag),
            }

        answer = vlm.generate_chat(
            _ANTHROPIC_SYSTEM,
            f"Answer briefly. If unsure, say you are not sure.\n\nQuestion: {question}",
        )
        return {
            "source": "llm",
            "answer": answer,
            "knowledge_terms": [],
            "citations": [],
            "confidence": 0.5,
            "abstain": False,
            "abstain_reason": None,
            "backend": _backend_label(pipeline),
            "retrieval_trace": [],
        }

    # ── Path B: real hosted LLM available (Claude / OpenAI / HF) ──────────────
    if vlm.is_real_llm():
        # Retrieve document context
        rag = pipeline.query_structured(question, data_folder, top_k=3)
        ranked = rag.get("source_ranking", [])

        # Build knowledge context
        kb_section = ""
        if kb_entries:
            kb_section = (
                "## Insurance Knowledge Base\n"
                + format_knowledge_answer(kb_entries)
                + "\n\n"
            )

        # Build document context
        doc_section = ""
        if ranked:
            doc_lines = []
            for p in ranked[:3]:
                doc_lines.append(f"[{p['source']}]\n{p['text_snippet']}")
            doc_section = "## Policy Document Excerpts\n" + "\n\n".join(doc_lines) + "\n\n"

        context = (kb_section + doc_section).strip()
        user_prompt = (
            f"{context}\n\n## Question\n{question}"
            if context
            else question
        )

        answer = vlm.generate_chat(_ANTHROPIC_SYSTEM, user_prompt)

        # Determine source label
        if ranked and not rag.get("abstain") and document_first:
            source = "document"
        elif kb_entries and ranked and not rag.get("abstain"):
            source = "combined"
        elif kb_entries:
            source = "knowledge"
        elif ranked and not rag.get("abstain"):
            source = "document"
        else:
            source = "knowledge"  # Claude answered from training knowledge

        return {
            "source": source,
            "answer": answer,
            "knowledge_terms": [] if document_first and ranked and not rag.get("abstain") else [e.term for e in kb_entries],
            "citations": rag.get("citations", []) if not rag.get("abstain") else [],
            "confidence": 1.0,
            "abstain": False,
            "abstain_reason": None,
            "backend": _backend_label(pipeline),
            "retrieval_trace": _retrieval_trace(rag),
        }

    # ── Path C: local-extractive fallback ─────────────────────────────────────
    rag = pipeline.query_structured(question, data_folder, top_k=3)

    has_kb = len(kb_entries) > 0
    has_doc = not rag.get("abstain") and bool(rag.get("answer"))

    if has_doc and document_first:
        answer = rag["answer"]
        source = "document"
    elif has_kb and has_doc:
        kb_text = format_knowledge_answer(kb_entries)
        answer = f"{kb_text}\n\n**From your policy documents:**\n{rag['answer']}"
        source = "combined"
    elif has_kb:
        answer = format_knowledge_answer(kb_entries)
        source = "knowledge"
    elif has_doc:
        answer = rag["answer"]
        source = "document"
    elif rag.get("abstain"):
        return {
            "source": "abstain",
            "answer": "",
            "abstain_reason": (
                "No matching evidence found. Try asking with 'What is…', 'What does…', "
                "or 'Explain…' for insurance terms, or set ANTHROPIC_API_KEY for open-ended Q&A."
            ),
            "citations": [],
            "confidence": 0.0,
            "backend": _backend_label(pipeline),
            "retrieval_trace": _retrieval_trace(rag),
        }
    else:
        answer = (
            "I couldn't find a specific answer. "
            "For open-ended questions, set ANTHROPIC_API_KEY to enable Claude as the LLM backend. "
            "For insurance terms, try: 'What is business insurance?' or 'Explain CI'."
        )
        source = "none"

    return {
        "source": source,
        "answer": answer,
        "knowledge_terms": [] if source == "document" and document_first else [e.term for e in kb_entries],
        "citations": rag.get("citations", []),
        "confidence": rag.get("confidence", 1.0 if has_kb else 0.0),
        "abstain": False,
        "abstain_reason": None,
        "backend": _backend_label(pipeline),
        "retrieval_trace": _retrieval_trace(rag),
    }


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>InsureRAG-VLM — Policy Assistant</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --sidebar:   #171717;
      --sb-border: rgba(255,255,255,.09);
      --sb-hover:  rgba(255,255,255,.07);
      --accent:    #19c37d;
      --accent-dk: #0fa060;
      --accent-lt: #d1fae5;
      --blue:      #60a5fa;
      --blue-lt:   #dbeafe;
      --purple:    #a78bfa;
      --purple-lt: #ede9fe;
      --warn:      #f87171;
      --warn-lt:   #fee2e2;
      --bg:        #212121;
      --card:      #2a2a2a;
      --border:    #383838;
      --txt:       #ececec;
      --muted:     #8e8ea0;
      --white:     #ffffff;
      --radius:    12px;
    }
    html, body { height: 100%; font-family: "Söhne", ui-sans-serif, system-ui, -apple-system, sans-serif; font-size: 15px; color: var(--txt); background: var(--bg); overflow: hidden; }

    /* ─── LAYOUT ─── */
    .app { display: flex; height: 100vh; }

    /* ─── SIDEBAR ─── */
    .sidebar {
      width: 260px; flex-shrink: 0;
      background: var(--sidebar);
      display: flex; flex-direction: column;
      border-right: 1px solid var(--sb-border);
      overflow: hidden;
    }
    .sb-top { padding: 12px 8px; }
    .new-chat-btn {
      display: flex; align-items: center; gap: 8px;
      width: 100%; padding: 10px 12px;
      border-radius: 8px; border: 1px solid var(--sb-border);
      background: transparent; color: var(--txt);
      font: inherit; font-size: 13px; font-weight: 500;
      cursor: pointer; transition: background .15s;
    }
    .new-chat-btn:hover { background: var(--sb-hover); }
    .new-chat-btn .ico { font-size: 15px; }

    .sb-divider { height: 1px; background: var(--sb-border); margin: 6px 8px; }

    .sb-section { padding: 8px; }
    .sb-label {
      font-size: 11px; font-weight: 600; letter-spacing: .08em;
      text-transform: uppercase; color: var(--muted);
      padding: 4px 4px 8px;
    }

    /* Upload */
    .upload-zone {
      border: 1.5px dashed rgba(255,255,255,.15);
      border-radius: 8px; padding: 14px 12px;
      text-align: center; cursor: pointer;
      transition: border-color .2s, background .2s;
      background: rgba(255,255,255,.02);
      position: relative;
    }
    .upload-zone:hover { border-color: var(--accent); background: rgba(25,195,125,.08); }
    .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
    .uz-icon { font-size: 20px; margin-bottom: 5px; }
    .uz-text { font-size: 12px; font-weight: 600; color: #b4b4b4; }
    .uz-hint { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .file-status { font-size: 11px; color: var(--muted); padding: 6px 2px; line-height: 1.4; }
    .file-status.ok { color: var(--accent); }
    .file-status.err { color: var(--warn); }

    /* Presets */
    .presets { display: flex; flex-direction: column; gap: 3px; }
    .preset-btn {
      width: 100%; text-align: left;
      padding: 8px 10px; border-radius: 7px;
      border: none; background: transparent;
      color: #b4b4b4; font: inherit; font-size: 12px;
      cursor: pointer; transition: background .15s;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .preset-btn:hover { background: var(--sb-hover); color: var(--txt); }

    .sb-footer {
      margin-top: auto; padding: 12px;
      font-size: 11px; color: var(--muted);
      border-top: 1px solid var(--sb-border);
      line-height: 1.5;
    }
    .sb-footer strong { color: #b4b4b4; }

    /* ─── CHAT AREA ─── */
    .chat-wrapper {
      flex: 1; display: flex; flex-direction: column; overflow: hidden;
      background: var(--bg);
    }

    /* Top bar */
    .chat-topbar {
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; gap: 10px;
      background: var(--bg); flex-shrink: 0;
    }
    .topbar-logo { font-size: 20px; }
    .topbar-title { font-size: 16px; font-weight: 700; }
    .topbar-sub { font-size: 12px; color: var(--muted); }
    .topbar-badge {
      margin-left: auto;
      padding: 3px 10px; border-radius: 999px;
      font-size: 11px; font-weight: 700;
      background: rgba(25,195,125,.15); color: var(--accent);
      border: 1px solid rgba(25,195,125,.3);
    }

    /* Messages */
    .messages {
      flex: 1; overflow-y: auto;
      padding: 24px 0 12px;
      display: flex; flex-direction: column; gap: 0;
    }
    .messages::-webkit-scrollbar { width: 5px; }
    .messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    .msg-row {
      padding: 10px 0;
      display: flex; justify-content: center;
    }
    .msg-row.user-row { background: transparent; }
    .msg-row.asst-row { background: rgba(255,255,255,.025); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }

    .msg-inner { width: 100%; max-width: 720px; padding: 0 24px; display: flex; gap: 14px; }

    /* Avatar */
    .avatar {
      width: 32px; height: 32px; border-radius: 6px;
      display: flex; align-items: center; justify-content: center;
      font-size: 15px; font-weight: 700; flex-shrink: 0; margin-top: 2px;
    }
    .avatar.user-av { background: #19c37d; color: #fff; font-size: 12px; }
    .avatar.asst-av { background: #ffffff; font-size: 16px; }

    /* Bubble content */
    .bubble { flex: 1; min-width: 0; }
    .bubble-name { font-size: 12px; font-weight: 700; color: var(--muted); margin-bottom: 6px; }

    .bubble-text {
      font-size: 15px; line-height: 1.65; color: var(--txt);
      white-space: pre-wrap; word-break: break-word;
    }
    .bubble-text strong { color: var(--white); }
    .bubble-text .kb-term { color: var(--accent); font-weight: 700; }
    .bubble-text .section-sep { display: block; height: 1px; background: var(--border); margin: 12px 0; }
    .bubble-text .policy-label {
      display: inline-block; padding: 2px 8px; border-radius: 4px;
      background: rgba(96,165,250,.15); color: var(--blue);
      font-size: 11px; font-weight: 700; margin-bottom: 6px;
    }

    /* Source badges */
    .bubble-meta { margin-top: 10px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .source-badge {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 3px 10px; border-radius: 999px;
      font-size: 11px; font-weight: 700; border: 1px solid;
    }
    .source-badge.kb { background: rgba(167,139,250,.12); color: var(--purple); border-color: rgba(167,139,250,.3); }
    .source-badge.doc { background: rgba(96,165,250,.12); color: var(--blue); border-color: rgba(96,165,250,.3); }
    .source-badge.combo { background: rgba(25,195,125,.1); color: var(--accent); border-color: rgba(25,195,125,.3); }
    .source-badge.warn { background: rgba(248,113,113,.1); color: var(--warn); border-color: rgba(248,113,113,.3); }
    .source-badge.diff-b { background: rgba(251,191,36,.1); color: #fbbf24; border-color: rgba(251,191,36,.3); }

    /* Citations */
    .citations-wrap { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
    details.citation {
      border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    }
    details.citation summary {
      padding: 8px 12px;
      background: rgba(255,255,255,.03);
      font-size: 12px; font-weight: 600; color: var(--blue);
      cursor: pointer; list-style: none; display: flex; align-items: center; gap: 6px;
    }
    details.citation summary::-webkit-details-marker { display: none; }
    details.citation summary::before { content: "▶"; font-size: 9px; transition: transform .2s; }
    details.citation[open] summary::before { transform: rotate(90deg); }
    .citation-body {
      padding: 10px 14px;
      border-top: 1px solid var(--border);
      font-size: 12.5px; color: #c9c9c9; line-height: 1.55;
      border-left: 3px solid var(--blue);
      background: rgba(96,165,250,.04);
    }
    .citation-preview {
      display: grid;
      grid-template-columns: 116px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
    }
    .citation-thumb {
      width: 116px;
      max-height: 154px;
      object-fit: contain;
      background: #111;
      border: 1px solid var(--border);
      border-radius: 6px;
    }
    .citation-evidence {
      min-width: 0;
      border-left: 3px solid var(--accent);
      padding: 8px 10px;
      border-radius: 6px;
      background: rgba(25,195,125,.055);
      color: #d9d9d9;
      overflow-wrap: anywhere;
    }
    .citation-evidence mark {
      color: inherit;
      background: rgba(251,191,36,.22);
      border-bottom: 1px solid rgba(251,191,36,.42);
      border-radius: 3px;
      padding: 0 2px;
    }

    /* Retrieval trace */
    details.trace {
      margin-top: 10px;
      border: 1px solid rgba(25,195,125,.24);
      border-radius: 8px;
      overflow: hidden;
      background: rgba(25,195,125,.035);
    }
    details.trace summary {
      padding: 8px 12px;
      cursor: pointer;
      list-style: none;
      font-size: 12px;
      font-weight: 700;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    details.trace summary::-webkit-details-marker { display: none; }
    details.trace summary::before { content: "▶"; font-size: 9px; transition: transform .2s; }
    details.trace[open] summary::before { transform: rotate(90deg); }
    .trace-body { border-top: 1px solid rgba(25,195,125,.18); padding: 8px 10px; display: flex; flex-direction: column; gap: 7px; }
    .trace-item {
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 8px 10px;
      background: rgba(255,255,255,.025);
    }
    .trace-head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 11px; color: var(--muted); margin-bottom: 5px; }
    .trace-rank { color: var(--accent); font-weight: 800; }
    .trace-score { color: var(--blue); font-weight: 700; }
    .trace-source { color: #c9c9c9; overflow-wrap: anywhere; }
    .trace-snippet { font-size: 12px; line-height: 1.45; color: #bdbdbd; }

    /* Diff blocks */
    .diff-wrap { margin-top: 8px; display: flex; flex-direction: column; gap: 5px; }
    .diff-row { display: flex; gap: 8px; padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 13px; line-height: 1.45; }
    .diff-row:last-child { border-bottom: none; }
    .diff-badge { flex-shrink: 0; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-weight: 800; height: fit-content; margin-top: 2px; }
    .diff-badge.add { background: #14532d; color: #4ade80; }
    .diff-badge.rem { background: #450a0a; color: #f87171; }
    .diff-txt { color: #c9c9c9; }

    /* Confidence bar */
    .conf-wrap { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
    .conf-label { font-size: 11px; color: var(--muted); }
    .conf-bar { flex: 1; max-width: 120px; height: 4px; background: var(--border); border-radius: 999px; overflow: hidden; }
    .conf-fill { height: 100%; border-radius: 999px; background: var(--accent); transition: width .6s ease; }
    .conf-pct { font-size: 11px; color: var(--accent); font-weight: 700; }

    /* Typing indicator */
    .typing { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
    .typing span {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--muted); animation: blink 1.2s ease infinite;
    }
    .typing span:nth-child(2) { animation-delay: .2s; }
    .typing span:nth-child(3) { animation-delay: .4s; }
    @keyframes blink { 0%,80%,100% { opacity:.25; } 40% { opacity:1; } }

    /* Welcome screen */
    .welcome {
      flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
      padding: 32px 24px; text-align: center; gap: 24px;
    }
    .welcome-logo { font-size: 48px; }
    .welcome-title { font-size: 24px; font-weight: 700; color: var(--txt); }
    .welcome-sub { font-size: 15px; color: var(--muted); max-width: 480px; line-height: 1.6; }
    .suggest-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; max-width: 600px; width: 100%; }
    .suggest-card {
      padding: 14px 16px; border-radius: 10px;
      border: 1px solid var(--border); background: var(--card);
      text-align: left; cursor: pointer;
      font: inherit; font-size: 13px; color: #c9c9c9;
      transition: border-color .15s, background .15s;
    }
    .suggest-card:hover { border-color: var(--accent); background: rgba(25,195,125,.08); color: var(--txt); }
    .suggest-card .sc-label { font-size: 11px; font-weight: 700; color: var(--muted); margin-bottom: 5px; }

    /* Input area */
    .input-area {
      padding: 14px 20px 20px;
      display: flex; justify-content: center;
      background: var(--bg);
      border-top: 1px solid var(--border);
      flex-shrink: 0;
    }
    .input-box-wrap {
      width: 100%; max-width: 720px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--card);
      display: flex; align-items: flex-end; gap: 0;
      transition: border-color .2s;
      overflow: hidden;
    }
    .input-box-wrap:focus-within { border-color: rgba(25,195,125,.5); }
    #inputBox {
      flex: 1;
      background: transparent;
      border: none; outline: none;
      color: var(--txt); font: inherit; font-size: 15px;
      padding: 14px 16px;
      resize: none;
      max-height: 200px;
      overflow-y: auto;
      line-height: 1.5;
    }
    #inputBox::placeholder { color: var(--muted); }
    #inputBox::-webkit-scrollbar { width: 3px; }
    #inputBox::-webkit-scrollbar-thumb { background: var(--border); }
    .send-btn {
      flex-shrink: 0;
      margin: 8px;
      width: 36px; height: 36px;
      border-radius: 8px;
      border: none;
      background: var(--accent);
      color: var(--sidebar);
      font-size: 16px;
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: background .15s, opacity .15s;
    }
    .send-btn:disabled { background: var(--border); color: var(--muted); cursor: default; }
    .send-btn:not(:disabled):hover { background: var(--accent-dk); }
    .input-hint {
      text-align: center; margin-top: 8px;
      font-size: 11px; color: var(--muted);
    }

    @media (max-width: 780px) {
      .sidebar { display: none; }
      .suggest-grid { grid-template-columns: 1fr; }
      .citation-preview { grid-template-columns: 1fr; }
      .citation-thumb { width: 100%; max-height: 220px; }
    }
  </style>
</head>
<body>
<div class="app">

  <!-- ═════ SIDEBAR ═════ -->
  <aside class="sidebar">
    <div class="sb-top">
      <button class="new-chat-btn" id="newChatBtn">
        <span class="ico">&#x270F;</span> New chat
      </button>
    </div>

    <div class="sb-divider"></div>

    <div class="sb-section">
      <div class="sb-label">Document</div>
      <div class="upload-zone">
        <div class="uz-icon">&#x1F4C4;</div>
        <div class="uz-text">Upload policy PDF</div>
        <div class="uz-hint">Click or drag &amp; drop</div>
        <input id="fileInput" type="file" accept=".pdf" />
      </div>
      <div id="fileStatus" class="file-status">No file uploaded</div>
    </div>

    <div class="sb-divider"></div>

    <div class="sb-section">
      <div class="sb-label">Try asking</div>
      <div class="presets">
        <button class="preset-btn" data-q="What is PI in insurance?">What is PI in insurance?</button>
        <button class="preset-btn" data-q="What does E&O stand for?">What does E&amp;O stand for?</button>
        <button class="preset-btn" data-q="What is the difference between ACV and RCV?">ACV vs RCV?</button>
        <button class="preset-btn" data-q="What is waiver of subrogation?">Waiver of subrogation?</button>
        <button class="preset-btn" data-q="Explain the difference between occurrence and claims-made policies.">Occurrence vs claims-made?</button>
        <button class="preset-btn" data-q="What coverage limits are described in the document?">Coverage limits?</button>
        <button class="preset-btn" data-q="What exclusions are described in the document?">Exclusions?</button>
        <button class="preset-btn" data-q="Compare the first two uploaded or indexed policy documents and what changed.">Compare documents</button>
      </div>
    </div>

    <div class="sb-footer">
      <strong>InsureRAG-VLM</strong><br>
      LLM: <strong id="backendLabel">detecting…</strong><br>
      Knowledge base: __KNOWLEDGE_BASE_SIZE__ terms
    </div>
  </aside>

  <!-- ═════ CHAT ═════ -->
  <div class="chat-wrapper">
    <div class="chat-topbar">
      <span class="topbar-logo">&#x1F6E1;</span>
      <div>
        <div class="topbar-title">InsureRAG-VLM Policy Assistant</div>
        <div class="topbar-sub">Insurance knowledge + document Q&amp;A</div>
      </div>
      <span class="topbar-badge" id="topBadge">Ready</span>
    </div>

    <!-- Welcome screen (shown until first message) -->
    <div class="welcome" id="welcomeScreen">
      <div class="welcome-logo">&#x1F6E1;</div>
      <div class="welcome-title">InsureRAG-VLM Policy Assistant</div>
      <div class="welcome-sub">
        Ask about insurance industry terms and acronyms, or query your uploaded policy documents — all in one place.
      </div>
      <div class="suggest-grid">
        <button class="suggest-card" data-q="What does PI stand for in insurance?">
          <div class="sc-label">&#x1F4DA; Industry term</div>
          What does PI stand for in insurance?
        </button>
        <button class="suggest-card" data-q="What does E&O stand for and when do I need it?">
          <div class="sc-label">&#x1F4DA; Acronym</div>
          What does E&amp;O stand for and when do I need it?
        </button>
        <button class="suggest-card" data-q="What is the comprehensive deductible in my policy?">
          <div class="sc-label">&#x1F4C4; Policy document</div>
          What is the comprehensive deductible?
        </button>
        <button class="suggest-card" data-q="Compare the two policy versions and explain what changed.">
          <div class="sc-label">&#x1F504; Version diff</div>
          Compare the two policy versions
        </button>
      </div>
    </div>

    <!-- Message list (hidden until first message) -->
    <div class="messages" id="messages" style="display:none"></div>

    <div class="input-area">
      <div style="width:100%;max-width:720px">
        <div class="input-box-wrap">
          <textarea id="inputBox" rows="1" placeholder="Ask about insurance terms or your policy..."></textarea>
          <button class="send-btn" id="sendBtn" title="Send (Enter)">&#x27A4;</button>
        </div>
        <div class="input-hint">Enter to send &nbsp;&bull;&nbsp; Shift+Enter for new line</div>
      </div>
    </div>
  </div>

</div>
<script>
/* ── detect backend ── */
(async () => {
  try {
    const r = await fetch('/api/backend');
    const d = await r.json();
    const el = document.getElementById('backendLabel');
    if (el) el.textContent = d.backend;
  } catch {}
})();

/* ── refs ── */
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesEl    = document.getElementById('messages');
const inputBox      = document.getElementById('inputBox');
const sendBtn       = document.getElementById('sendBtn');
const fileInput     = document.getElementById('fileInput');
const fileStatus    = document.getElementById('fileStatus');
const newChatBtn    = document.getElementById('newChatBtn');
const topBadge      = document.getElementById('topBadge');

/* ── state ── */
let busy = false;

/* ── auto-resize textarea ── */
inputBox.addEventListener('input', () => {
  inputBox.style.height = 'auto';
  inputBox.style.height = Math.min(inputBox.scrollHeight, 200) + 'px';
});

/* ── Enter to send / Shift+Enter for newline ── */
inputBox.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!busy) send();
  }
});
sendBtn.addEventListener('click', () => { if (!busy) send(); });

/* ── preset / suggestion clicks ── */
document.querySelectorAll('[data-q]').forEach(btn => {
  btn.addEventListener('click', () => {
    inputBox.value = btn.dataset.q;
    inputBox.style.height = 'auto';
    inputBox.style.height = Math.min(inputBox.scrollHeight, 200) + 'px';
    if (!busy) send();
  });
});

/* ── new chat ── */
newChatBtn.addEventListener('click', () => {
  messagesEl.innerHTML = '';
  messagesEl.style.display = 'none';
  welcomeScreen.style.display = 'flex';
  inputBox.value = '';
  inputBox.style.height = 'auto';
  topBadge.textContent = 'Ready';
  topBadge.style.cssText = '';
});

/* ── file upload ── */
fileInput.addEventListener('change', async () => {
  const f = fileInput.files[0];
  if (!f) return;
  fileStatus.textContent = 'Indexing ' + f.name + '…';
  fileStatus.className = 'file-status';
  const body = new FormData();
  body.append('file', f);
  try {
    const res  = await fetch('/api/upload', { method: 'POST', body });
    const data = await res.json();
    if (data.ok) {
      fileStatus.textContent = '● ' + data.filename + ' ready for Q&A';
      fileStatus.className = 'file-status ok';
    } else {
      fileStatus.textContent = data.error || 'Upload failed.';
      fileStatus.className = 'file-status err';
    }
  } catch { fileStatus.textContent = 'Upload error.'; fileStatus.className = 'file-status err'; }
});

/* ────────────────────────────────────────────────────────────
   RENDER helpers
──────────────────────────────────────────────────────────── */
function escHtml(s) {
  s = String(s ?? '');
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function mdToHtml(text) {
  // bold **...**
  let s = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // code `...`
  s = s.replace(/`(.+?)`/g, '<code>$1</code>');
  // diff separator ---
  s = s.replace(/^---$/gm, '<span class="section-sep"></span>');
  // policy label marker  **From your policy documents:**
  s = s.replace(/<strong>(From your policy documents:)<\/strong>/g,
    '<span class="policy-label">&#x1F4C4; From your policy documents</span>');
  // line breaks
  s = s.replace(/\n/g, '<br>');
  return s;
}

function appendRow(role, html, meta) {
  if (welcomeScreen.style.display !== 'none') {
    welcomeScreen.style.display = 'none';
    messagesEl.style.display = 'flex';
  }

  const row  = document.createElement('div');
  row.className = 'msg-row ' + (role === 'user' ? 'user-row' : 'asst-row');

  const inner = document.createElement('div');
  inner.className = 'msg-inner';

  const av = document.createElement('div');
  av.className = 'avatar ' + (role === 'user' ? 'user-av' : 'asst-av');
  av.textContent = role === 'user' ? 'You' : '🛡';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  const name = document.createElement('div');
  name.className = 'bubble-name';
  name.textContent = role === 'user' ? 'You' : 'InsureRAG Assistant';

  const txt = document.createElement('div');
  txt.className = 'bubble-text';
  txt.innerHTML = html;

  bubble.appendChild(name);
  bubble.appendChild(txt);

  if (meta) {
    /* source badge */
    if (meta.source && meta.source !== 'none' && meta.source !== 'diff') {
      const metaRow = document.createElement('div');
      metaRow.className = 'bubble-meta';
      const badge = document.createElement('span');
      const configs = {
        knowledge: ['kb',    '&#x1F4DA; Insurance Knowledge Base'],
        document:  ['doc',   '&#x1F4C4; Policy Document'],
        combined:  ['combo', '&#x1F4DA; Knowledge + &#x1F4C4; Policy'],
        abstain:   ['warn',  '&#x26A0; Insufficient Evidence'],
        diff:      ['diff-b','&#x1F504; Version Diff'],
      };
      const [cls, label] = configs[meta.source] || ['doc', meta.source];
      badge.className = 'source-badge ' + cls;
      badge.innerHTML = label;
      metaRow.appendChild(badge);

      /* confidence bar */
      if (meta.confidence != null && meta.confidence > 0 && meta.source !== 'knowledge') {
        const cw = document.createElement('div');
        cw.className = 'conf-wrap';
        const pct = Math.round(meta.confidence * 100);
        cw.innerHTML = `<span class="conf-label">Confidence</span>
          <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
          <span class="conf-pct">${pct}%</span>`;
        metaRow.appendChild(cw);
      }
      bubble.appendChild(metaRow);
    }

    /* citations */
    if (meta.citations && meta.citations.length > 0) {
      const cw = document.createElement('div');
      cw.className = 'citations-wrap';
      meta.citations.forEach(c => {
        const det = document.createElement('details');
        det.className = 'citation';
        const sum = document.createElement('summary');
        sum.innerHTML = '&#x1F4CE; ' + escHtml(c.source || 'cited page');
        const body = document.createElement('div');
        body.className = 'citation-body';
        const preview = document.createElement('div');
        preview.className = 'citation-preview';
        const thumb = document.createElement('img');
        thumb.className = 'citation-thumb';
        thumb.loading = 'lazy';
        thumb.alt = 'Cited PDF page preview';
        thumb.src = '/api/page-image?source=' + encodeURIComponent(c.source || '');
        thumb.onerror = () => { thumb.remove(); preview.style.gridTemplateColumns = '1fr'; };
        const evidence = document.createElement('div');
        evidence.className = 'citation-evidence';
        const snippet = escHtml(c.evidence_text || 'Retrieved cited page.');
        evidence.innerHTML = '<mark>' + snippet + '</mark>';
        preview.appendChild(thumb);
        preview.appendChild(evidence);
        body.appendChild(preview);
        det.appendChild(sum); det.appendChild(body);
        cw.appendChild(det);
      });
      bubble.appendChild(cw);
    }

    /* retrieval trace */
    if (meta.retrieval_trace && meta.retrieval_trace.length > 0) {
      const det = document.createElement('details');
      det.className = 'trace';
      const sum = document.createElement('summary');
      sum.innerHTML = '&#x1F50E; Retrieval trace';
      const body = document.createElement('div');
      body.className = 'trace-body';
      meta.retrieval_trace.forEach(item => {
        const row = document.createElement('div');
        row.className = 'trace-item';
        const score = item.score == null ? '' : Number(item.score).toFixed(3);
        const page = item.page_number ? 'page ' + item.page_number : 'page ?';
        row.innerHTML = `<div class="trace-head">
          <span class="trace-rank">#${item.rank}</span>
          <span class="trace-score">score ${score}</span>
          <span>${escHtml(page)}</span>
          <span class="trace-source">${escHtml(item.source || '')}</span>
        </div>
        <div class="trace-snippet">${escHtml((item.snippet || '').slice(0, 260))}</div>`;
        body.appendChild(row);
      });
      det.appendChild(sum);
      det.appendChild(body);
      bubble.appendChild(det);
    }

    /* diff items */
    if (meta.diff_items && meta.diff_items.length > 0) {
      const dw = document.createElement('div');
      dw.className = 'diff-wrap';
      meta.diff_items.forEach(item => {
        const row = document.createElement('div');
        row.className = 'diff-row';
        const badge = document.createElement('span');
        badge.className = 'diff-badge ' + (item.type === 'added' ? 'add' : 'rem');
        badge.textContent = item.type === 'added' ? '+ NEW' : '− OLD';
        const dtxt = document.createElement('span');
        dtxt.className = 'diff-txt';
        dtxt.textContent = item.text;
        row.appendChild(badge); row.appendChild(dtxt);
        dw.appendChild(row);
      });
      bubble.appendChild(dw);
    }
  }

  inner.appendChild(av);
  inner.appendChild(bubble);
  row.appendChild(inner);
  messagesEl.appendChild(row);
  row.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return { row, txt };
}

function appendTyping() {
  const { row, txt } = appendRow('assistant', '', null);
  txt.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  return row;
}

/* ────────────────────────────────────────────────────────────
   SEND
──────────────────────────────────────────────────────────── */
async function send() {
  const q = inputBox.value.trim();
  if (!q) return;
  busy = true;
  sendBtn.disabled = true;
  inputBox.value = '';
  inputBox.style.height = 'auto';
  topBadge.textContent = 'Thinking…';

  appendRow('user', escHtml(q), null);
  const typingRow = appendTyping();

  try {
    const isDiff = /\b(compar|diff|version|drift|v1|v2|chang)\w*/i.test(q) && q.split(/\s+/).length > 1;

    if (isDiff) {
      const res  = await fetch('/api/diff');
      const data = await res.json();
      typingRow.remove();

      let diffItems = [];
      const sections = [
        ['Deductible', data.deductible_changes],
        ['Coverage',   data.coverage_changes],
        ['Endorsement',data.endorsement_drift],
        ['Exclusion',  data.exclusion_changes],
        ['Duties',     data.duties_after_loss_changes],
      ];
      sections.forEach(([label, items]) => {
        if (!items) return;
        items.slice(0, 4).forEach(item => {
          diffItems.push({ type: item.change_type, text: '[' + label + '] ' + item.text.slice(0, 110) });
        });
      });

      appendRow('assistant',
        '<strong>Policy Version Diff: v1 → v2</strong><br>' + escHtml(data.summary || ''),
        { source: 'diff', diff_items: diffItems }
      );
    } else {
      const res  = await fetch('/api/chat?q=' + encodeURIComponent(q));
      const data = await res.json();
      typingRow.remove();

      if (data.abstain) {
        appendRow('assistant',
          escHtml(data.abstain_reason || 'Insufficient evidence in the retrieved documents.'),
          { source: 'abstain', retrieval_trace: data.retrieval_trace }
        );
      } else {
        appendRow('assistant', mdToHtml(data.answer), {
          source:     data.source,
          citations:  data.citations,
          confidence: data.confidence,
          retrieval_trace: data.retrieval_trace,
        });
      }
    }
    topBadge.textContent = 'Ready';
  } catch (err) {
    typingRow.remove();
    appendRow('assistant', 'Sorry, an error occurred: ' + escHtml(String(err)), null);
    topBadge.textContent = 'Error';
  }

  busy = false;
  sendBtn.disabled = false;
  inputBox.focus();
}
</script>
</body>
</html>
"""

HTML = HTML.replace("__KNOWLEDGE_BASE_SIZE__", str(knowledge_base_size()))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class DemoHandler(BaseHTTPRequestHandler):
    _pipeline: DocumentRetrievalPipeline | None = None
    _data_folder: Path = DATA_FOLDER
    _index_dir: Path = INDEX_DIR

    @classmethod
    def pipeline(cls) -> DocumentRetrievalPipeline:
        if cls._pipeline is None:
            config = ModelConfig(index_dir=cls._index_dir)
            pipeline = DocumentRetrievalPipeline(config)
            if not config.index_path.exists() or not config.metadata_path.exists():
                pipeline.build_index(cls._data_folder)
            cls._pipeline = pipeline
        return cls._pipeline

    @classmethod
    def use_uploaded_folder(cls, data_folder: Path, index_dir: Path) -> None:
        config = ModelConfig(index_dir=index_dir)
        pipeline = DocumentRetrievalPipeline(config)
        pipeline.build_index(data_folder)
        cls._data_folder = data_folder
        cls._index_dir = index_dir
        cls._pipeline = pipeline

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/backend":
            label = _backend_label(self.pipeline())
            self._send(200, json.dumps({"backend": label}).encode(), "application/json; charset=utf-8")
            return

        if parsed.path == "/api/chat":
            question = parse_qs(parsed.query).get("q", [""])[0].strip()
            if not question:
                self._send_json({"error": "Missing question parameter q"}, status=400)
                return
            result = build_chat_response(question, self.pipeline(), self._data_folder)
            if result.get("source") == "diff":
                pair = _first_two_pdfs(self._data_folder)
                if pair is None:
                    self._send_json(
                        {
                            "source": "diff",
                            "summary": "Upload or index at least two real PDF documents before running policy diff.",
                        },
                        status=400,
                    )
                    return
                old_text = "\n\n".join(extract_text_by_page(pair[0]))
                new_text = "\n\n".join(extract_text_by_page(pair[1]))
                diff = summarize_policy_diff(old_text, new_text)
                self._send(200, json.dumps(diff, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            else:
                self._send(200, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            return

        if parsed.path == "/api/page-image":
            source = parse_qs(parsed.query).get("source", [""])[0].strip()
            page_image = self._render_source_page_image(source)
            if page_image and page_image.exists():
                image_bytes = page_image.read_bytes()
                self._send(200, image_bytes, "image/png")
            else:
                self._send(404, b"Page image not available", "text/plain")
            return

        if parsed.path == "/api/diff":
            pair = _first_two_pdfs(self._data_folder)
            if pair is None:
                self._send_json(
                    {
                        "source": "diff",
                        "summary": "Upload or index at least two real PDF documents before running policy diff.",
                    },
                    status=400,
                )
                return
            old_text = "\n\n".join(extract_text_by_page(pair[0]))
            new_text = "\n\n".join(extract_text_by_page(pair[1]))
            result = summarize_policy_diff(old_text, new_text)
            self._send(200, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            return

        self._send(404, b"Not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/upload":
            self._send(404, b"Not found", "text/plain")
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type or "boundary=" not in content_type:
            self._send_json({"ok": False, "error": "Expected multipart PDF upload."}, status=400)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            filename, content = self._extract_upload(content_type, body)
            if not filename or not content:
                self._send_json({"ok": False, "error": "No file found in upload."}, status=400)
                return
            safe_name = self._safe_filename(filename)
            if Path(safe_name).suffix.lower() != ".pdf":
                self._send_json({"ok": False, "error": "Please upload a PDF file."}, status=400)
                return
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            upload_path = UPLOAD_DIR / safe_name
            upload_path.write_bytes(content)
            self.use_uploaded_folder(UPLOAD_DIR, UPLOAD_INDEX_DIR)
            self._send_json({"ok": True, "filename": safe_name})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    @classmethod
    def _render_source_page_image(cls, source: str) -> Path | None:
        pdf_path, page_number = cls._resolve_source_pdf(source)
        if pdf_path is None:
            return None
        PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha1(
            f"{pdf_path.resolve()}:{pdf_path.stat().st_mtime_ns}:{page_number}".encode("utf-8")
        ).hexdigest()[:16]
        output_path = PAGE_CACHE_DIR / f"{cache_key}_p{page_number:04d}.png"
        if output_path.exists():
            return output_path
        try:
            import fitz

            with fitz.open(pdf_path) as document:
                if page_number < 1 or page_number > len(document):
                    return None
                pix = document[page_number - 1].get_pixmap(dpi=92, annots=False)
                pix.save(str(output_path))
            return output_path
        except Exception:
            return None

    @classmethod
    def _resolve_source_pdf(cls, source: str) -> tuple[Path | None, int]:
        if not source:
            return None, 1
        doc_ref, page_number = source, 1
        if "#page=" in source:
            doc_ref, page_blob = source.split("#page=", 1)
            match = re.search(r"\d+", page_blob)
            page_number = int(match.group(0)) if match else 1

        candidate_ref = Path(doc_ref)
        candidates: list[Path] = []
        if candidate_ref.is_absolute():
            candidates.append(candidate_ref)
        else:
            candidates.extend(
                [
                    cls._data_folder / candidate_ref,
                    UPLOAD_DIR / candidate_ref.name,
                    DATA_FOLDER / candidate_ref,
                    PROJECT_ROOT / candidate_ref,
                ]
            )
            for root in [cls._data_folder, UPLOAD_DIR, DATA_FOLDER]:
                if root.exists():
                    candidates.extend(root.rglob(candidate_ref.name))

        safe_roots = [PROJECT_ROOT / "data", PROJECT_ROOT / "reports" / "demo_uploads"]
        for candidate in candidates:
            if candidate.suffix.lower() != ".pdf" or not candidate.exists():
                continue
            resolved = candidate.resolve()
            if any(_is_relative_to(resolved, root.resolve()) for root in safe_roots if root.exists()):
                return resolved, page_number
        return None, page_number

    @staticmethod
    def _safe_filename(filename: str) -> str:
        cleaned = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in filename)
        return cleaned or "uploaded_policy.pdf"

    @staticmethod
    def _extract_upload(content_type: str, body: bytes) -> tuple[str, bytes]:
        boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode()
        delimiter = b"--" + boundary
        for part in body.split(delimiter):
            if b"Content-Disposition" not in part or b"filename=" not in part:
                continue
            header_blob, _, content = part.partition(b"\r\n\r\n")
            if not content:
                continue
            disposition = header_blob.decode("utf-8", errors="ignore")
            match = re.search(r'filename="([^"]+)"', disposition)
            filename = match.group(1) if match else ""
            content = content.rsplit(b"\r\n", 1)[0]
            return filename, content
        return "", b""

    def log_message(self, format: str, *args) -> None:
        return


def run_demo_server(host: str = "127.0.0.1", port: int = 7860) -> None:
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"InsureRAG-VLM demo running at http://{host}:{port}")
    server.serve_forever()
