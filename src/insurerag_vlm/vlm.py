import json
import os
import re
from typing import Optional

import requests


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

_SYSTEM_PROMPT = """You are InsureRAG, an expert insurance industry assistant serving internal company employees.

Your role:
- Explain insurance terminology, acronyms, and concepts clearly for employees at any level.
- Answer questions about uploaded policy documents with precise citations.
- When answering about a specific policy document, quote the relevant text and cite the source.
- If you are not confident, say so rather than guessing.
- Be concise but complete. Use plain language; avoid unnecessary jargon unless explaining it.
- When relevant, mention related terms the employee might want to know about.

Format rules:
- Use **bold** for key terms, amounts, and important phrases.
- Use bullet points for lists.
- Keep answers focused — 2-5 sentences for simple questions, more only when needed.
- For policy-document answers, end with: Source: [document name], Page [N]
- Do NOT include <think>...</think> reasoning blocks in your final answer."""

_OLLAMA_SYSTEM_PROMPT = """You are InsureRAG. Answer insurance questions concisely using only the supplied evidence when evidence is present. Cite sources exactly as given. If evidence is insufficient, say so."""

# Public alias used by app.py
_ANTHROPIC_SYSTEM = _SYSTEM_PROMPT


def _detect_ollama() -> Optional[str]:
    """Return the first available Ollama model name, or None if Ollama is not running."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if not models:
                return None
            names = [m["name"] for m in models]
            requested = os.environ.get("OLLAMA_MODEL")
            if requested:
                for name in names:
                    if name == requested or name.startswith(f"{requested}:"):
                        return name
            preferred = [
                "qwen2.5:3b",
                "llama3.2:3b",
                "llama3.2",
                "gemma3:4b",
                "phi4-mini",
                "mistral",
            ]
            for target in preferred:
                for name in names:
                    if name == target or name.startswith(f"{target}:"):
                        return name
            non_reasoning = [name for name in names if "r1" not in name.lower() and "reason" not in name.lower()]
            if non_reasoning:
                return non_reasoning[0]
            return names[0]
    except Exception:
        pass
    return None


class VLMClient:
    def __init__(
        self,
        model_name: str,
        hf_api_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        use_hf_api: bool = True,
    ):
        self.model_name = model_name
        self.hf_api_token = hf_api_token or os.environ.get("HF_API_TOKEN")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.use_hf_api = use_hf_api
        use_ollama = os.environ.get("INSURERAG_USE_OLLAMA", "1").lower() not in {"0", "false", "no"}
        # Detect Ollama once at init.
        self._ollama_model: Optional[str] = _detect_ollama() if use_ollama else None

    def is_real_llm(self) -> bool:
        if self._ollama_model:
            return True
        if self.anthropic_api_key:
            return True
        if self.openai_api_key:
            return True
        if self.hf_api_token and not self.model_name.startswith("local-"):
            return True
        return False

    def backend_label(self) -> str:
        if self._ollama_model:
            return f"Ollama · {self._ollama_model}"
        if self.anthropic_api_key:
            return f"Claude · {self.model_name}"
        if self.openai_api_key:
            return f"OpenAI · {self.model_name}"
        if self.hf_api_token:
            return f"HuggingFace · {self.model_name}"
        return "local-extractive (no LLM)"

    def is_ollama(self) -> bool:
        return self._ollama_model is not None

    def generate(self, prompt: str) -> str:
        if self._ollama_model:
            return self._call_ollama_chat(_OLLAMA_SYSTEM_PROMPT, prompt)
        if self.anthropic_api_key:
            return self._call_anthropic_chat(_SYSTEM_PROMPT, prompt)
        if self.openai_api_key:
            return self._call_openai_chat(_SYSTEM_PROMPT, prompt)
        if self.hf_api_token and not self.model_name.startswith("local-"):
            return self._call_huggingface(prompt)
        return self._local_extractive_answer(prompt)

    def generate_chat(self, system: str, user: str) -> str:
        if self._ollama_model:
            return self._call_ollama_chat(system, user)
        if self.anthropic_api_key:
            return self._call_anthropic_chat(system, user)
        if self.openai_api_key:
            return self._call_openai_chat(system, user)
        combined = f"{system}\n\nUser: {user}\nAssistant:"
        return self.generate(combined)

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _call_ollama_chat(self, system: str, user: str) -> str:
        model = self._ollama_model or OLLAMA_DEFAULT_MODEL
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.0")),
                "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "384")),
                "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
            },
        }
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        # Strip <think>...</think> blocks that DeepSeek-R1 emits
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw

    def generate_extractive(self, prompt: str) -> str:
        return self._local_extractive_answer(prompt)

    # ── Anthropic / Claude ────────────────────────────────────────────────────

    def _call_anthropic_chat(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        model = self.model_name if not self.model_name.startswith("local-") else "claude-haiku-4-5"
        client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _call_openai_chat(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        model = self.model_name if not self.model_name.startswith("local-") else "gpt-4o-mini"
        client = OpenAI(api_key=self.openai_api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()

    # ── Hugging Face ──────────────────────────────────────────────────────────

    def _call_huggingface(self, prompt: str) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {"Authorization": f"Bearer {self.hf_api_token}", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"inputs": prompt, "options": {"wait_for_model": True}}, timeout=120)
        resp.raise_for_status()
        out = resp.json()
        if isinstance(out, list) and out:
            return out[0].get("generated_text", "").strip()
        return json.dumps(out)

    # ── Local extractive fallback ─────────────────────────────────────────────

    def _local_extractive_answer(self, prompt: str) -> str:
        question_match = re.search(r"Question:\s*(.*?)\n\nAnswer:", prompt, flags=re.DOTALL)
        context_match = re.search(r"Context:\s*(.*?)\n\nQuestion:", prompt, flags=re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""
        context = context_match.group(1).strip() if context_match else prompt

        question_terms = {t for t in re.findall(r"[a-zA-Z0-9$%]+", question.lower()) if len(t) > 2}
        generic = {
            "what", "which", "does", "the", "this", "that", "policy", "coverage",
            "deductible", "provide", "provides", "listed", "limit", "limits", "after",
            "loss", "insured", "have", "apply", "applies",
        }
        key_terms = question_terms - generic

        sources = []
        for block in context.split("\n---\n"):
            sm = re.search(r"SOURCE:\s*(.*)", block)
            source = sm.group(1).strip() if sm else "unknown"
            text = re.sub(r"^SOURCE:.*\n?", "", block).strip()
            if text:
                sources.append((source, text))

        best_source, best_sentence, best_score = "unknown", "", -1.0
        for source, text in sources:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                s = sentence.strip()
                if not s:
                    continue
                sterms = set(re.findall(r"[a-zA-Z0-9$%]+", s.lower()))
                score = float(len(question_terms & sterms))
                if "deductible" in question_terms and "$" in s and "deductible" in sterms:
                    score += 2.0
                if "limit" in question_terms and "$" in s and ("limit" in sterms or "limits" in sterms):
                    score += 2.0
                if key_terms and not (key_terms & sterms):
                    score -= 1.5
                if score > best_score:
                    best_score, best_source, best_sentence = score, source, s

        bterms = set(re.findall(r"[a-zA-Z0-9$%]+", best_sentence.lower()))
        if not best_sentence or best_score <= 0 or (key_terms and not (key_terms & bterms)):
            return "I cannot support an answer from the retrieved evidence. SOURCE: insufficient_evidence"
        return f"{best_sentence}\n\nSOURCE: {best_source}"


def format_prompt(context: str, question: str, template: str) -> str:
    return template.format(context=context.strip(), question=question.strip())
