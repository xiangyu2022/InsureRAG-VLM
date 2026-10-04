import json
import os
import re
import time
import base64
import hashlib
from io import BytesIO
from copy import deepcopy
from pathlib import Path
from threading import local
from typing import Any, Optional
from urllib.parse import urlparse

import requests


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = "qwen3.5:4b"

_SYSTEM_PROMPT = """You are InsureRAG, an expert insurance industry assistant serving internal company employees.

Your role:
- Explain insurance terminology, acronyms, and concepts clearly for employees at any level.
- Answer questions about uploaded policy documents with precise citations.
- When answering about a specific policy document, quote the relevant text and cite the source.
- A person's own policy amount requires evidence from the relevant policy, declarations, or endorsement. Public guides and illustrative examples do not establish that person's coverage, even if the guide was uploaded.
- If you are not confident, say so rather than guessing.
- Be concise but complete. Use plain language; avoid unnecessary jargon unless explaining it.
- When relevant, mention related terms the employee might want to know about.

Format rules:
- Use **bold** for key terms, amounts, and important phrases.
- Use bullet points for lists.
- Keep answers focused — 2-5 sentences for simple questions, more only when needed.
- For policy-document answers, end with: Source: [document name], Page [N]
- Do NOT include <think>...</think> reasoning blocks in your final answer."""

_OLLAMA_SYSTEM_PROMPT = """You are InsureRAG. Answer insurance questions concisely using only the supplied evidence when evidence is present. Cite sources exactly as given. A person's own policy amount requires evidence from the relevant policy, declarations, or endorsement. Public guides and illustrative examples do not establish that person's coverage, even if the guide was uploaded. If evidence is insufficient, say so."""

# Public alias used by app.py
_ANTHROPIC_SYSTEM = _SYSTEM_PROMPT


class BackendConfigurationError(ValueError):
    """The requested provider/model cannot be selected unambiguously."""


class BackendUnavailableError(RuntimeError):
    """The explicitly requested backend is unavailable; no substitute was used."""


def _resolve_provider(model_name: str, provider: Optional[str] = None) -> tuple[str, str]:
    requested = model_name.strip()
    if requested == "local-extractive":
        if provider not in (None, "local", "local-extractive"):
            raise BackendConfigurationError("local-extractive conflicts with the requested provider")
        return "local-extractive", requested
    aliases = {"ollama": "ollama", "openai": "openai", "anthropic": "anthropic", "hf": "huggingface", "huggingface": "huggingface"}
    prefix, separator, remainder = requested.partition(":")
    if separator and prefix in aliases:
        selected = aliases[prefix]
        if provider and aliases.get(provider, provider) != selected:
            raise BackendConfigurationError("Model prefix conflicts with the requested provider")
        provider, requested = selected, remainder
    elif provider:
        provider = aliases.get(provider, provider)
    elif requested.startswith("claude-"):
        provider = "anthropic"
    elif requested.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        provider = "openai"
    elif ":" in requested:
        # Bare Ollama name:tag remains supported by the existing CLI.
        provider = "ollama"
    else:
        raise BackendConfigurationError(
            "Choose local-extractive or an explicit provider:model, for example "
            "ollama:qwen3.5:4b, openai:gpt-4o-mini, or hf:organization/model."
        )
    if provider not in set(aliases.values()) or not requested or requested.startswith("local-"):
        raise BackendConfigurationError(f"Invalid provider/model selection: {provider!r} / {requested!r}")
    if provider == "ollama" and ":" not in requested.rsplit("/", 1)[-1]:
        requested += ":latest"
    return provider, requested


class VLMClient:
    def __init__(
        self,
        model_name: str,
        hf_api_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        use_hf_api: bool = True,
        *,
        provider: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        generation_options: Optional[dict[str, Any]] = None,
        thinking: bool = False,
        request_timeout: float = 300,
        expected_model_digest: Optional[str] = None,
    ):
        self.requested_model = model_name
        self.provider, self.model_name = _resolve_provider(model_name, provider)
        self.hf_api_token = hf_api_token or os.environ.get("HF_API_TOKEN")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.use_hf_api = use_hf_api
        self.ollama_base_url = (ollama_base_url or os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)).rstrip("/")
        parsed_url = urlparse(self.ollama_base_url)
        if self.provider == "ollama" and (parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname or parsed_url.username or parsed_url.password):
            raise BackendConfigurationError("Ollama base URL must be an HTTP(S) URL without embedded credentials")
        self.thinking = thinking
        if not isinstance(thinking, bool):
            raise BackendConfigurationError("thinking must be a boolean")
        self.request_timeout = request_timeout
        self.generation_options = {
            "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.0")),
            "seed": int(os.environ.get("OLLAMA_SEED", "42")),
            "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "384")),
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
            "top_k": 40,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
        } if self.provider == "ollama" else {}
        self.generation_options.update(generation_options or {})
        if self.request_timeout <= 0 or (self.is_ollama() and (self.generation_options["num_predict"] <= 0 or self.generation_options["num_ctx"] <= 0)):
            raise BackendConfigurationError("Timeout, num_predict, and num_ctx must be positive")
        self._ollama_model: Optional[str] = None
        self._model_info: dict[str, Any] = {}
        self._ollama_version: Optional[str] = None
        self._generation_state = local()
        self.last_generation_metadata: dict[str, Any] = {}
        if self.provider == "ollama":
            if os.environ.get("INSURERAG_USE_OLLAMA", "1").lower() in {"0", "false", "no"}:
                raise BackendConfigurationError("Ollama was explicitly selected but INSURERAG_USE_OLLAMA disables it")
            self._model_info = self._lookup_ollama_model()
            self._ollama_model = self.model_name
            if expected_model_digest and self._model_info.get("digest") != expected_model_digest:
                raise BackendConfigurationError("Requested Ollama model digest does not match the installed model")
            try:
                response = requests.get(f"{self.ollama_base_url}/api/version", timeout=5)
                response.raise_for_status()
                self._ollama_version = response.json().get("version")
            except (requests.RequestException, ValueError):
                # Version metadata is optional; tag + digest are required.
                self._ollama_version = None
        else:
            required = {"openai": self.openai_api_key, "anthropic": self.anthropic_api_key, "huggingface": self.hf_api_token}
            if self.provider in required and not required[self.provider]:
                raise BackendConfigurationError(f"An API credential is required for explicitly selected {self.provider}")
            if self.provider == "huggingface" and not self.use_hf_api:
                raise BackendConfigurationError("Hugging Face was selected but use_hf_api is false")

    @property
    def last_generation_metadata(self) -> dict[str, Any]:
        return getattr(self._generation_state, "metadata", {})

    @last_generation_metadata.setter
    def last_generation_metadata(self, value: dict[str, Any]) -> None:
        self._generation_state.metadata = value

    def answer_trace(self, *, invoked: bool, force_extractive: bool = False) -> dict[str, Any]:
        used = invoked and not force_extractive and self.is_real_llm()
        metadata = self.backend_metadata()
        if not used:
            metadata["last_generation"] = {}
        return {
            "generation_used": used,
            "answer_backend": self.backend_label() if used else "local-extractive" if invoked else "retrieval-abstention",
            "backend_metadata": metadata,
        }

    def _lookup_ollama_model(self) -> dict[str, Any]:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise BackendUnavailableError(f"Cannot reach requested Ollama backend at {self.ollama_base_url}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise BackendUnavailableError("Ollama returned an invalid model inventory")
        models = payload["models"]
        if any(not isinstance(entry, dict) for entry in models):
            raise BackendUnavailableError("Ollama returned an invalid model inventory")
        match = next((entry for entry in models if entry.get("name") == self.model_name or entry.get("model") == self.model_name), None)
        if match is None:
            raise BackendUnavailableError(
                f"Requested Ollama model {self.model_name!r} is not installed. "
                f"Install that exact tag with `ollama pull {self.model_name}`. No fallback was used."
            )
        if not match.get("digest"):
            raise BackendUnavailableError("Ollama did not return a model digest; reproducible model identity is unavailable")
        return dict(match)

    def is_real_llm(self) -> bool:
        return self.provider != "local-extractive"

    def backend_label(self) -> str:
        labels = {"ollama": "Ollama", "anthropic": "Claude", "openai": "OpenAI", "huggingface": "HuggingFace"}
        if self.provider == "local-extractive":
            return "local-extractive (no LLM)"
        return f"{labels[self.provider]} · {self.model_name}"

    def backend_metadata(self) -> dict[str, Any]:
        return deepcopy({
            "provider": self.provider,
            "requested_model": self.requested_model,
            "resolved_model": self.model_name,
            "model_digest": self._model_info.get("digest"),
            "model_details": self._model_info.get("details", {}),
            "ollama_version": self._ollama_version,
            "base_url": self.ollama_base_url if self.is_ollama() else None,
            "generation_options": self.generation_options if self.is_ollama() else None,
            "thinking": self.thinking if self.is_ollama() else None,
            "last_generation": self.last_generation_metadata,
        })

    def is_ollama(self) -> bool:
        return self.provider == "ollama"

    def generate(self, prompt: str) -> str:
        if self.is_ollama():
            return self._call_ollama_chat(_OLLAMA_SYSTEM_PROMPT, prompt)
        if self.provider == "anthropic":
            return self._call_anthropic_chat(_SYSTEM_PROMPT, prompt)
        if self.provider == "openai":
            return self._call_openai_chat(_SYSTEM_PROMPT, prompt)
        if self.provider == "huggingface":
            return self._call_huggingface(prompt)
        return self._local_extractive_answer(prompt)

    @staticmethod
    def _validate_response_format(response_format: Optional[str | dict[str, Any]]) -> None:
        if response_format is None:
            return
        if response_format != "json" and not isinstance(response_format, dict):
            raise BackendConfigurationError("response_format must be 'json' or a JSON schema object")
        try:
            json.dumps(response_format, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise BackendConfigurationError("response_format must contain valid JSON values") from exc

    def generate_chat(self, system: str, user: str, *, response_format: Optional[str | dict[str, Any]] = None) -> str:
        self._validate_response_format(response_format)
        if self.is_ollama():
            return self._call_ollama_chat(system, user, response_format=response_format)
        if response_format is not None:
            raise BackendConfigurationError("response_format is currently supported only for explicitly selected Ollama models")
        if self.provider == "anthropic":
            return self._call_anthropic_chat(system, user)
        if self.provider == "openai":
            return self._call_openai_chat(system, user)
        combined = f"{system}\n\nUser: {user}\nAssistant:"
        return self.generate(combined)

    def generate_with_images(self, prompt: str, image_paths: list[Path], *, system: str = _OLLAMA_SYSTEM_PROMPT,
                             response_format: Optional[str | dict[str, Any]] = None) -> str:
        """Explicit image-QA path; normal document RAG remains text-only.

        Images are loaded only from caller-supplied local paths. No remote media
        are fetched, and an unsupported backend never discards the images.
        """
        if not self.is_ollama():
            raise BackendConfigurationError("Image QA currently requires an explicitly selected Ollama vision model")
        self._validate_response_format(response_format)
        if not 1 <= len(image_paths) <= 4:
            raise BackendConfigurationError("Image QA requires between one and four local images")
        capabilities = self._model_info.get("capabilities")
        if capabilities is None:
            try:
                response = requests.post(f"{self.ollama_base_url}/api/show", json={"model": self.model_name}, timeout=5)
                response.raise_for_status()
                capabilities = response.json().get("capabilities", [])
            except (requests.RequestException, ValueError, AttributeError) as exc:
                raise BackendUnavailableError("Cannot verify the selected model's vision capability") from exc
        if "vision" not in capabilities:
            raise BackendConfigurationError(f"Selected model {self.model_name!r} does not advertise vision support")
        from PIL import Image

        encoded, provenance = [], []
        for value in image_paths:
            path = Path(value)
            try:
                if path.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError("image exceeds 20 MiB")
                content = path.read_bytes()
                with Image.open(BytesIO(content)) as image:
                    width, height = image.size
                    image_format = image.format
                    if image_format not in {"PNG", "JPEG", "WEBP"}:
                        raise ValueError("image must be PNG, JPEG, or WEBP")
                    if width * height > 20_000_000:
                        raise ValueError("image exceeds 20 million pixels")
                    image.verify()
            except (OSError, ValueError) as exc:
                raise BackendConfigurationError(f"Cannot use image {path.name!r}: {exc}") from exc
            encoded.append(base64.b64encode(content).decode("ascii"))
            provenance.append({"name": path.name, "sha256": hashlib.sha256(content).hexdigest(),
                               "width": width, "height": height, "format": image_format})
        return self._call_ollama_chat(system, prompt, images=encoded, image_provenance=provenance,
                                      response_format=response_format)

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _call_ollama_chat(self, system: str, user: str, *, images: Optional[list[str]] = None,
                          image_provenance: Optional[list[dict[str, Any]]] = None,
                          response_format: Optional[str | dict[str, Any]] = None) -> str:
        self.last_generation_metadata = {}
        current_model = self._lookup_ollama_model()
        if current_model["digest"] != self._model_info["digest"]:
            raise BackendUnavailableError("Ollama model digest changed during this run; initialize a new client for the new model")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Qwen3.5 does not support the old /nothink prompt switch.
            # Ollama's API uses this top-level flag, not an options field.
            "think": self.thinking,
            "options": dict(self.generation_options),
        }
        if images:
            payload["messages"][1]["images"] = images
        if response_format is not None:
            payload["format"] = deepcopy(response_format)
        started = time.perf_counter()
        try:
            resp = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            result = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise BackendUnavailableError(f"Ollama generation failed for {self.model_name!r}; no fallback was used") from exc
        if not isinstance(result, dict) or result.get("done") is not True:
            raise BackendUnavailableError("Ollama returned an invalid or incomplete non-streaming response")
        if result.get("model") != self.model_name:
            raise BackendUnavailableError(f"Ollama answered using an unexpected model: {result.get('model')!r}")
        message = result.get("message") or {}
        if not isinstance(message, dict):
            raise BackendUnavailableError("Ollama returned an invalid assistant message")
        raw = message.get("content")
        if not isinstance(raw, str):
            raise BackendUnavailableError("Ollama returned no assistant text")
        self.last_generation_metadata = {
            "wall_seconds": time.perf_counter() - started,
            "done_reason": result.get("done_reason"),
            "truncated": result.get("done_reason") == "length",
            "thinking_returned": bool(message.get("thinking")),
            "input_modality": "text+image" if images else "text",
            "image_count": len(images or []),
            "images": image_provenance or [],
            "response_format": deepcopy(response_format),
            **{key: result.get(key) for key in (
                "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"
            )},
        }
        # Handle older servers that embed reasoning tags; do not return hidden reasoning.
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if "<think>" in raw or not raw:
            raise BackendUnavailableError("Ollama returned no usable final answer (possibly exhausted the generation budget)")
        return raw

    def generate_extractive(self, prompt: str) -> str:
        return self._local_extractive_answer(prompt)

    # ── Anthropic / Claude ────────────────────────────────────────────────────

    def _call_anthropic_chat(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        msg = client.messages.create(
            model=self.model_name,
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

        client = OpenAI(api_key=self.openai_api_key)
        resp = client.chat.completions.create(
            model=self.model_name,
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
