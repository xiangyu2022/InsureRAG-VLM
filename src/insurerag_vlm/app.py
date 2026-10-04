import json
import re
import secrets
import hashlib
import logging
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import ModelConfig
from .diff import summarize_policy_diff
from .knowledge import format_knowledge_answer, knowledge_base_size, search_knowledge
from .pdf import extract_text_by_page
from .pipeline import DocumentRetrievalPipeline
from .hybrid_pipeline import DocumentTextUnavailableError
from .retriever import EmbeddingBackendError
from .vlm import BackendConfigurationError, BackendUnavailableError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / "data" / "00_raw" / "external" / "public_docs"
INDEX_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / "reports" / "demo_uploads" / "current"
UPLOAD_INDEX_DIR = PROJECT_ROOT / "reports" / "demo_uploads" / "index"
PAGE_CACHE_DIR = PROJECT_ROOT / "reports" / "demo_uploads" / "page_cache"
LOGGER = logging.getLogger(__name__)

_DIFF_TRIGGERS = re.compile(
    r"\b(compar|diff|version|drift|v1|v2|chang|updat)\w*\b", re.I
)


def _is_diff_query(question: str) -> bool:
    hits = _DIFF_TRIGGERS.findall(question)
    return len(hits) >= 2


_POLICY_CONCEPT = r"(?:polic(?:y|ies)|insurance|coverage|deductibles?|limits?|premiums?|endorsements?|exclusions?|benefits?|claims?|copay(?:ment)?s?|coinsurance|plan)"


def _is_personal_policy_query(question: str) -> bool:
    # Permit coverage names and ordinary modifiers between the possessive and
    # policy concept: "my collision deductible", "our actual annual premium".
    # A bounded phrase avoids interpreting "my question is: what is ..." as a
    # request for a private policy fact.
    modifiers = r"(?:(?!(?:question|example|definition|understanding)\b)[\w'-]+[\s-]+){0,6}"
    if re.search(r"\b(?:my|our)\s+" + modifiers + _POLICY_CONCEPT + r"\b", question, re.I):
        return True
    if re.search(r"\b(?:am\s+i|are\s+we)\s+covered\b", question, re.I):
        return True
    return bool(re.search(r"\b" + _POLICY_CONCEPT + r"\b", question, re.I)) and bool(
        re.search(r"\b(?:do\s+(?:i|we)\s+have|(?:i|we)\s+(?:pay|owe)|appl(?:y|ies)\s+to\s+(?:me|us))\b", question, re.I)
    )


def _is_document_first_query(question: str) -> bool:
    return _is_personal_policy_query(question) or bool(
        re.search(r"\b(documents?|polic(?:y|ies)|guides?|pdf|uploaded|files?)\b", question, re.I)
    ) or bool(re.search(r"\bthis\s+(?:[\w-]+\s+){0,4}" + _POLICY_CONCEPT + r"\b", question, re.I))


def _backend_label(pipeline: DocumentRetrievalPipeline) -> str:
    return pipeline.vlm_client.backend_label()


def _abstention_message(reason: str | None) -> str:
    messages = {
        "personal_policy_not_available": "The active documents are public reference guides, which cannot establish your own policy's coverage or amounts. Upload your policy or declarations pages to answer this question.",
        "generation_truncated": "The model's response ended before it was complete. Try a shorter or more specific question.",
        "missing_policy_packet_counterevidence": "The available pages leave a policy exception or conflicting clause unresolved. Add the related policy pages to establish the answer.",
    }
    return messages.get(reason, "The available document evidence does not establish this answer. Upload the relevant policy pages or ask a more specific question.")


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
    """Return the pipeline's validated answer without a second generation step."""
    if Path(data_folder).resolve() == DATA_FOLDER.resolve() and _is_personal_policy_query(question):
        trace = pipeline.vlm_client.answer_trace(invoked=False)
        return {
            "source": "abstain", "answer": "", "knowledge_terms": [], "citations": [],
            "citation_origin": None,
            "confidence": None, "abstain": True,
            "abstain_reason": _abstention_message("personal_policy_not_available"),
            "abstain_reason_code": "personal_policy_not_available",
            "backend": "corpus-abstention", "configured_backend": _backend_label(pipeline),
            "generation_used": False, "answer_repaired": False,
            "backend_metadata": trace["backend_metadata"], "retrieval_trace": [],
        }
    if _is_diff_query(question):
        return {"source": "diff"}

    document_first = _is_document_first_query(question)
    kb_entries = search_knowledge(question)
    # Glossary answers are a separately labeled deterministic feature. A
    # policy-specific request must always go through document evidence checks.
    if kb_entries and not document_first:
        return {
            "source": "knowledge",
            "answer": format_knowledge_answer(kb_entries),
            "knowledge_terms": [entry.term for entry in kb_entries],
            "citations": [],
            "citation_origin": None,
            "confidence": None,
            "abstain": False,
            "abstain_reason": None,
            "abstain_reason_code": None,
            "backend": "knowledge-base deterministic answer",
            "configured_backend": _backend_label(pipeline),
            "generation_used": False,
            "answer_repaired": False,
            "retrieval_trace": [],
        }

    # Honor the selected answer model. Forcing extraction here would make an
    # installed Qwen model appear active while never generating document answers.
    rag = pipeline.query_structured(question, data_folder, top_k=3)
    abstain = bool(rag.get("abstain"))
    answer = str(rag.get("answer") or "")
    if not answer and not abstain:
        abstain = True
    return {
        "source": "abstain" if abstain else "document",
        "answer": "" if abstain else answer,
        "knowledge_terms": [],
        "citations": [] if abstain else rag.get("citations", []),
        "citation_origin": None if abstain else rag.get("citation_origin"),
        "confidence": rag.get("confidence", 0.0),
        "abstain": abstain,
        "abstain_reason": _abstention_message(rag.get("abstain_reason")) if abstain else None,
        "abstain_reason_code": rag.get("abstain_reason") if abstain else None,
        "backend": rag.get("answer_backend") or _backend_label(pipeline),
        "configured_backend": _backend_label(pipeline),
        "generation_used": bool(rag.get("generation_used", False)),
        "answer_repaired": bool(rag.get("answer_repaired", False)),
        "raw_answer": str(rag.get("raw_answer") or ""),
        "backend_metadata": rag.get("backend_metadata", {}),
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
      <div id="fileStatus" class="file-status">Checking active documents…</div>
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
      Document answer mode: <strong id="backendLabel">checking…</strong><br>
      Knowledge base: __KNOWLEDGE_BASE_SIZE__ terms
    </div>
  </aside>

  <!-- ═════ CHAT ═════ -->
  <div class="chat-wrapper">
    <div class="chat-topbar">
      <span class="topbar-logo">&#x1F6E1;</span>
      <div>
        <div class="topbar-title">InsureRAG-VLM Policy Assistant</div>
        <div class="topbar-sub">Research prototype · glossary + cited document Q&amp;A</div>
      </div>
      <span class="topbar-badge" id="topBadge">Ready</span>
    </div>

    <!-- Welcome screen (shown until first message) -->
    <div class="welcome" id="welcomeScreen">
      <div class="welcome-logo">&#x1F6E1;</div>
      <div class="welcome-title">InsureRAG-VLM Policy Assistant</div>
      <div class="welcome-sub">
        Explore insurance terms or ask about policy documents. Check the cited evidence; this research prototype can make mistakes.
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
    if (el) el.textContent = r.ok ? d.backend : (d.error || 'unavailable');
    if (d.corpus) showCorpusStatus(d.corpus);
  } catch {
    const el = document.getElementById('backendLabel');
    if (el) el.textContent = 'status unavailable';
  }
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

function showCorpusStatus(corpus) {
  const el = document.getElementById('fileStatus');
  if (!el || !corpus) return;
  el.textContent = 'Documents: ' + corpus.label;
  const ingestion = corpus.ingestion;
  if (ingestion && ingestion.total_pages) {
    el.textContent += ' · text pages ' + ingestion.readable_pages + '/' + ingestion.total_pages;
    if (ingestion.unreadable_pages) el.textContent += ' · ' + ingestion.unreadable_pages + ' unreadable page(s) excluded; OCR required';
    else if (ingestion.blank_pages) el.textContent += ' · blank pages skipped';
  }
  el.className = 'file-status' + (corpus.mode === 'uploaded' ? ' ok' : '');
}

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
      showCorpusStatus(data.corpus);
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
  let s = escHtml(String(text ?? '')).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
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

      /* This retrieval score is a heuristic, not calibrated correctness. */
      if (meta.confidence != null && meta.confidence > 0 && meta.source !== 'knowledge') {
        const cw = document.createElement('div');
        cw.className = 'conf-wrap';
        cw.textContent = 'Evidence score: ' + Number(meta.confidence).toFixed(2) + ' (heuristic)';
        metaRow.appendChild(cw);
      }
      bubble.appendChild(metaRow);
    }

    if (meta.backend) {
      const backend = document.createElement('div');
      backend.className = 'conf-label';
      backend.textContent = 'Answer source: ' + meta.backend;
      if (meta.answer_repaired) backend.textContent += ' · answer shortened or replaced after evidence checks';
      if (meta.citation_origin === 'evidence_selection') backend.textContent += ' · citation selected from retrieved evidence';
      if (meta.citation_origin === 'model_source') backend.textContent += ' · citation supplied by model';
      bubble.appendChild(backend);
    }

    if (meta.generation_used && meta.raw_answer) {
      const det = document.createElement('details');
      det.className = 'trace';
      const sum = document.createElement('summary');
      sum.textContent = 'Original model output (unvalidated)';
      const body = document.createElement('div');
      body.className = 'trace-body';
      body.style.whiteSpace = 'pre-wrap';
      body.textContent = meta.raw_answer;
      det.appendChild(sum); det.appendChild(body);
      bubble.appendChild(det);
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
        if (c.preview_url) thumb.src = c.preview_url;
        thumb.onerror = () => { thumb.remove(); preview.style.gridTemplateColumns = '1fr'; };
        const evidence = document.createElement('div');
        evidence.className = 'citation-evidence';
        const snippet = escHtml(c.evidence_text || 'Retrieved cited page.');
        evidence.innerHTML = '<mark>' + snippet + '</mark>';
        if (c.preview_url) preview.appendChild(thumb);
        else preview.style.gridTemplateColumns = '1fr';
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
      if (!res.ok) throw new Error(data.error || data.summary || 'Document comparison failed.');
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
      if (!res.ok) throw new Error(data.error || 'The selected answer backend failed.');
      typingRow.remove();

      if (data.abstain) {
        appendRow('assistant',
          escHtml(data.abstain_reason || 'Insufficient evidence in the retrieved documents.'),
          { source: 'abstain', backend: data.backend, retrieval_trace: data.retrieval_trace,
            generation_used: data.generation_used, raw_answer: data.raw_answer }
        );
      } else {
        appendRow('assistant', mdToHtml(data.answer), {
          source:     data.source,
          citations:  data.citations,
          confidence: data.confidence,
          backend: data.backend,
          answer_repaired: data.answer_repaired,
          citation_origin: data.citation_origin,
          generation_used: data.generation_used,
          raw_answer: data.raw_answer,
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
    _state_lock = threading.RLock()
    _preview_bindings: dict[str, dict] = {}

    @classmethod
    def corpus_status(cls) -> dict:
        folder = cls._data_folder.resolve()
        if _is_relative_to(folder, UPLOAD_DIR.parent.resolve()):
            filenames = sorted(path.name for path in folder.rglob("*.pdf")) if folder.exists() else []
            return {"mode": "uploaded", "label": ", ".join(filenames) or "Uploaded documents", "filenames": filenames,
                    "ingestion": getattr(cls._pipeline, "ingestion_report", {})}
        if folder == DATA_FOLDER.resolve():
            return {"mode": "public_reference", "label": "Public reference guides", "filenames": []}
        filenames = sorted(path.name for path in folder.rglob("*.pdf")) if folder.exists() else []
        return {"mode": "indexed", "label": ", ".join(filenames) or "Indexed reference documents", "filenames": filenames}

    @classmethod
    def pipeline(cls) -> DocumentRetrievalPipeline:
        if cls._pipeline is None:
            # The demo's public references are an explicit bundled snapshot
            # when the optional original PDFs are not present. Other folders
            # retain auto's strict "use this folder" behavior.
            public_snapshot = cls._data_folder.resolve() == DATA_FOLDER.resolve() and not cls._data_folder.is_dir()
            config = ModelConfig(
                index_dir=cls._index_dir,
                corpus_source="curated" if public_snapshot else "auto",
                curated_dataset_dir=PROJECT_ROOT / "data" / "04_curated",
            )
            pipeline = DocumentRetrievalPipeline(config)
            if not config.index_path.exists() or not config.metadata_path.exists():
                pipeline.build_index(cls._data_folder)
            cls._pipeline = pipeline
        return cls._pipeline

    @classmethod
    def use_uploaded_folder(cls, data_folder: Path, index_dir: Path, required_document: str | None = None) -> None:
        config = ModelConfig(index_dir=index_dir, corpus_source="documents", pdf_render_dir=index_dir / "rendered_pages")
        pipeline = DocumentRetrievalPipeline(config)
        pipeline.build_index(data_folder)
        if required_document:
            uploaded_pages = [row for row in pipeline.ingestion_report.get("pages", [])
                              if row["source"] == required_document or row["source"].startswith(required_document + "#page=")]
            if not any(row["text_characters"] for row in uploaded_pages):
                details = next((row["warning"] for row in uploaded_pages if row.get("warning")), "Upload a searchable PDF.")
                raise DocumentTextUnavailableError(f"No readable text was found in {required_document}. {details} This upload was not activated.")
        with cls._state_lock:
            cls._data_folder = data_folder
            cls._index_dir = index_dir
            cls._pipeline = pipeline

    @classmethod
    def activate_upload(cls, filename: str, content: bytes) -> None:
        """Build an unpublished candidate; activate it only after success.

        Successful directories keep their stable paths so index provenance and
        rendered-page references remain valid. A failed candidate is discarded.
        """
        sessions_root = (UPLOAD_DIR.parent / "sessions").resolve()
        sessions_root.mkdir(parents=True, exist_ok=True)
        candidate_root = Path(tempfile.mkdtemp(prefix="candidate-", dir=sessions_root)).resolve()
        candidate_docs = candidate_root / "documents"
        try:
            with cls._state_lock:
                active_folder = cls._data_folder.resolve()
                if active_folder.is_dir() and _is_relative_to(active_folder, UPLOAD_DIR.parent.resolve()):
                    shutil.copytree(active_folder, candidate_docs)
                else:
                    candidate_docs.mkdir()
                (candidate_docs / filename).write_bytes(content)
                cls.use_uploaded_folder(candidate_docs, candidate_root / "index", required_document=filename)
        except Exception:
            # Both paths are created locally above; never delete an active or
            # user-selected folder while rolling back a failed upload.
            if candidate_root.parent == sessions_root and candidate_root.name.startswith("candidate-"):
                shutil.rmtree(candidate_root)
            raise

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        try:
            self._handle_get()
        except (BackendConfigurationError, BackendUnavailableError, EmbeddingBackendError) as exc:
            self._send_json({"source": "error", "error": str(exc), "fallback_used": False}, status=503)
        except Exception:
            LOGGER.exception("Demo request failed")
            self._send_json({"source": "error", "error": "The request failed. Check the server logs."}, status=500)

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/backend":
            pipeline = self.pipeline()
            self._send_json({
                "backend": _backend_label(pipeline),
                "status": "configured",
                "answer_model": pipeline.vlm_client.backend_metadata(),
                "embedding_model": pipeline.retriever.backend_metadata(),
                "corpus": self.corpus_status(),
            })
            return

        if parsed.path == "/api/chat":
            question = parse_qs(parsed.query).get("q", [""])[0].strip()
            if not question:
                self._send_json({"error": "Missing question parameter q"}, status=400)
                return
            with self._state_lock:
                pipeline, data_folder = self.pipeline(), self._data_folder
            result = build_chat_response(question, pipeline, data_folder)
            self._bind_citation_previews(result, data_folder)
            if result.get("source") == "diff":
                pair = _first_two_pdfs(data_folder)
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
            token = parse_qs(parsed.query).get("token", [""])[0].strip()
            if not token:
                self._send(400, b"A citation-bound preview token is required", "text/plain")
                return
            page_image = self._render_bound_page_image(token)
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
            self.activate_upload(safe_name, content)
            self._send_json({"ok": True, "filename": safe_name, "corpus": self.corpus_status()})
        except DocumentTextUnavailableError as exc:
            self._send_json({"ok": False, "error": str(exc), "active_documents_unchanged": True}, status=422)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    @classmethod
    def _preview_safe_roots(cls) -> list[Path]:
        return [(PROJECT_ROOT / "data").resolve(), UPLOAD_DIR.parent.resolve()]

    @classmethod
    def _bind_citation_previews(cls, result: dict, data_folder: Path) -> None:
        """Bind each citation to the answer's snapshot, never the next upload.

        Tokens live only for this local server session. Retained upload folders
        supply old snapshots; missing or modified files fail closed at rendering.
        """
        for citation in result.get("citations", []):
            citation["preview_url"] = None
            citation["preview_sha256"] = None
            pdf_path, page_number = cls._resolve_source_pdf(str(citation.get("source") or ""), data_folder)
            if pdf_path is None:
                continue
            try:
                digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            except OSError:
                continue
            token = secrets.token_hex(24)
            with cls._state_lock:
                # Subclass handlers used by isolated local servers do not share
                # their preview registry with another server's handler class.
                if "_preview_bindings" not in cls.__dict__:
                    cls._preview_bindings = {}
                cls._preview_bindings[token] = {
                    "path": str(pdf_path), "root": str(Path(data_folder).resolve()),
                    "sha256": digest, "page_number": page_number,
                }
            citation["preview_url"] = f"/api/page-image?token={token}"
            citation["preview_sha256"] = digest

    @classmethod
    def _render_bound_page_image(cls, token: str) -> Path | None:
        if not re.fullmatch(r"[0-9a-f]{48}", token):
            return None
        with cls._state_lock:
            binding = cls.__dict__.get("_preview_bindings", {}).get(token)
            binding = dict(binding) if binding else None
        if not binding:
            return None
        try:
            import fitz
            root = Path(binding["root"]).resolve()
            pdf_path = Path(binding["path"]).resolve()
            if (pdf_path.suffix.lower() != ".pdf"
                    or not _is_relative_to(pdf_path, root)
                    or not any(_is_relative_to(root, safe_root) for safe_root in cls._preview_safe_roots())):
                return None
            # Render the same bytes that were checked, avoiding a file reread
            # between validation and rendering. Validate even on cache hits.
            pdf_bytes = pdf_path.read_bytes()
            digest = hashlib.sha256(pdf_bytes).hexdigest()
            if digest != binding["sha256"]:
                return None
            page_number = int(binding["page_number"])
            PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            output_path = PAGE_CACHE_DIR / f"{digest}_p{page_number:04d}_v1.png"
            if output_path.exists():
                return output_path
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                if page_number < 1 or page_number > len(document):
                    return None
                pix = document[page_number - 1].get_pixmap(dpi=92, annots=False)
                pix.save(str(output_path))
            return output_path
        except Exception:
            return None

    @classmethod
    def _resolve_source_pdf(cls, source: str, data_folder: Path | None = None) -> tuple[Path | None, int]:
        if not source:
            return None, 1
        doc_ref, page_number = source, 1
        if "#page=" in source:
            doc_ref, page_blob = source.split("#page=", 1)
            if not re.fullmatch(r"[1-9]\d*", page_blob):
                return None, 1
            page_number = int(page_blob)

        root = Path(data_folder if data_folder is not None else cls._data_folder).resolve()
        if not any(_is_relative_to(root, safe_root) for safe_root in cls._preview_safe_roots()):
            return None, page_number
        candidate_ref = Path(doc_ref)
        candidates: list[Path] = []
        if candidate_ref.is_absolute():
            candidates.append(candidate_ref)
        else:
            candidates.extend(
                [
                    root / candidate_ref,
                    PROJECT_ROOT / candidate_ref,
                ]
            )
            if root.exists() and str(candidate_ref) == candidate_ref.name:
                candidates.extend(root.rglob(candidate_ref.name))

        matches: set[Path] = set()
        for candidate in candidates:
            if candidate.suffix.lower() != ".pdf" or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if _is_relative_to(resolved, root):
                matches.add(resolved)
        if len(matches) == 1:
            return matches.pop(), page_number
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
