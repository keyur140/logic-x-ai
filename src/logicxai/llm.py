# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""One client for every model provider.

Groq, OpenRouter, Hugging Face and Ollama all speak the same
"OpenAI-compatible" chat API, so one small client covers them all.

Every call is:
  * deterministic where the provider allows it (temperature 0, fixed seed),
  * cached on disk by the SHA-256 of the exact request (so a crashed run
    resumes for free and re-runs give identical answers),
  * logged to a JSONL file with request hash, response hash, timing and
    the provider's reported model version.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .provenance import canonical_json, sha256_json, sha256_text, utc_now

PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "huggingface": {"base_url": "https://router.huggingface.co/v1", "key_env": "HF_TOKEN"},
    # OLLAMA_BASE_URL lets the same code use Ollama on another machine (e.g. a rented GPU)
    "ollama": {"base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1",
               "key_env": ""},
    "mock": {"base_url": "", "key_env": ""},  # offline test provider (see mock.py)
}


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so parallel threads never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    """One model on one provider. `alias` is the stable name used in the data."""
    alias: str
    provider: str
    model_id: str
    fallback_provider: str | None = None
    fallback_model_id: str | None = None
    # Extra request fields for one model (e.g. reasoning_effort), as sorted (key, value) pairs.
    extra: tuple = ()
    # Extra output tokens for models whose hidden reasoning counts against max_tokens.
    token_allowance: int = 0


class LLMClient:
    def __init__(self, cache_dir: str | Path, log_path: str | Path,
                 seed: int = 20261026, max_retries: int | None = None, timeout: int = 120):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        # Free tiers can be slow: retries and the longest wait are settable (pilot 2026-11-07).
        self.max_retries = max_retries or int(os.environ.get("LOGICXAI_MAX_RETRIES", "6"))
        self.max_sleep = float(os.environ.get("LOGICXAI_MAX_SLEEP", "120"))
        self.timeout = timeout
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.mock_answer_key: dict = {}

    # ---------------------------------------------------------------- utils
    def _cache_file(self, req_hash: str) -> Path:
        return self.cache_dir / req_hash[:2] / f"{req_hash}.json"

    def _log(self, record: dict[str, Any]) -> None:
        with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(canonical_json(record) + "\n")

    @staticmethod
    def available(provider: str) -> bool:
        env = PROVIDERS[provider]["key_env"]
        return provider in ("ollama", "mock") or bool(env and os.environ.get(env))

    # ----------------------------------------------------------------- main
    def chat(self, spec: ModelSpec, messages: list[dict[str, str]],
             max_tokens: int = 1024, purpose: str = "") -> dict[str, Any]:
        """Return {"text", "model_reported", "request_sha256", "response_sha256", "cached"}."""
        routes = [(spec.provider, spec.model_id)]
        if spec.fallback_provider and spec.fallback_model_id:
            routes.append((spec.fallback_provider, spec.fallback_model_id))
        last_err: Exception | None = None
        for provider, model_id in routes:
            if not self.available(provider):
                last_err = LLMError(f"no key for {provider}")
                continue
            try:
                return self._chat_one(spec.alias, provider, model_id, messages,
                                      max_tokens + spec.token_allowance, purpose, dict(spec.extra))
            except LLMError as e:
                last_err = e
        raise LLMError(f"{spec.alias}: all routes failed: {last_err}")

    def _chat_one(self, alias: str, provider: str, model_id: str,
                  messages: list[dict[str, str]], max_tokens: int, purpose: str,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {
            "model": model_id,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "seed": self.seed,
            "max_tokens": max_tokens,
        }
        if extra:
            body.update(extra)
        # The cache key ignores the provider so a fallback route with the
        # same weights is not re-queried; the provider is still logged.
        req_hash = sha256_json({"alias": alias, "body": body})
        cf = self._cache_file(req_hash)
        # Synthetic replies are never cached: they depend on the simulated world
        # (amplify / null), which is not part of the request.
        if provider != "mock" and cf.exists():
            data = json.loads(cf.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return {**data, "cached": True}

        if provider == "mock":
            from . import mock
            text = mock.respond(alias, messages, self.mock_answer_key)
            result = {"text": text, "alias": alias, "provider": "mock", "model_requested": model_id,
                      "model_reported": model_id, "system_fingerprint": None, "usage": {},
                      "request_sha256": req_hash, "response_sha256": sha256_text(text),
                      "timestamp_utc": utc_now()}
            self.calls += 1
            self._log({**{k: v for k, v in result.items() if k != "text"}, "purpose": purpose,
                       "attempt": 1, "latency_s": 0.0, "status": "ok"})
            return {**result, "cached": False}

        cfg = PROVIDERS[provider]
        headers = {"Content-Type": "application/json"}
        if cfg["key_env"]:
            headers["Authorization"] = f"Bearer {os.environ[cfg['key_env']]}"
        url = cfg["base_url"] + "/chat/completions"

        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                err = f"network: {e}"
            else:
                if r.status_code == 200:
                    payload = r.json()
                    try:
                        text = payload["choices"][0]["message"]["content"] or ""
                    except (KeyError, IndexError, TypeError):
                        err = f"bad payload: {str(payload)[:200]}"
                    else:
                        result = {
                            "text": text,
                            "alias": alias,
                            "provider": provider,
                            "model_requested": model_id,
                            "model_reported": payload.get("model", model_id),
                            "system_fingerprint": payload.get("system_fingerprint"),
                            "usage": payload.get("usage", {}),
                            "request_sha256": req_hash,
                            "response_sha256": sha256_text(text),
                            "timestamp_utc": utc_now(),
                        }
                        _atomic_write(cf, canonical_json(result))
                        self.calls += 1
                        self._log({**{k: v for k, v in result.items() if k != "text"},
                                   "purpose": purpose, "attempt": attempt,
                                   "latency_s": round(time.time() - t0, 3), "status": "ok"})
                        return {**result, "cached": False}
                retryable = r.status_code in (408, 409, 425, 429, 500, 502, 503, 504)
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if not retryable:
                    self._log({"alias": alias, "provider": provider, "request_sha256": req_hash,
                               "purpose": purpose, "status": "error", "error": err,
                               "timestamp_utc": utc_now()})
                    raise LLMError(err)
                ra = r.headers.get("retry-after")
                if ra:
                    try:
                        delay = max(delay, float(ra))
                    except ValueError:
                        pass
            self._log({"alias": alias, "provider": provider, "request_sha256": req_hash,
                       "purpose": purpose, "status": "retry", "attempt": attempt,
                       "error": err, "timestamp_utc": utc_now()})
            time.sleep(min(delay, self.max_sleep))
            delay *= 2
        raise LLMError(f"{provider}/{model_id}: gave up after {self.max_retries} tries")
