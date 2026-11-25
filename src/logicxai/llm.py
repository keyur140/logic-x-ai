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
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .provenance import canonical_json, sha256_json, sha256_text, utc_now

# key_envs: every environment variable that may hold a key for this provider, tried in
# order (v0.2.1, decision 18: free tiers cap each key per day, so several keys rotate).
PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "key_envs": ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4"]},
    # Google AI Studio (free tier; Gemma open-weights models) through its OpenAI-compatible endpoint
    "google": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_envs": ["GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY_2"]},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "key_envs": ["MISTRAL_API_KEY", "MISTRAL_API_KEY_2"]},
    "cerebras": {"base_url": "https://api.cerebras.ai/v1", "key_envs": ["CEREBRAS_API_KEY", "CEREBRAS_API_KEY_2"]},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "key_envs": ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"]},
    "huggingface": {"base_url": "https://router.huggingface.co/v1", "key_envs": ["HF_TOKEN", "HF_TOKEN_2"]},
    # OLLAMA_BASE_URL lets the same code use Ollama on another machine
    "ollama": {"base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1",
               "key_envs": []},
    "mock": {"base_url": "", "key_envs": []},  # offline test provider (see mock.py)
}

# Providers that do not accept every OpenAI field (the request still records what was sent).
NO_SEED = {"google", "mistral"}

USER_AGENT = "logicxai/0.2.1 (research study; python-requests)"

# A 429 whose text matches this is a DAILY limit, not a per-minute one.
DAILY_RE = re.compile(r"per day|\bTPD\b|\bRPD\b|daily|PerDay|quota exceeded|free-models-per-day", re.I)
WAIT_RE = re.compile(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", re.I)


def key_names(provider: str) -> list[str]:
    """Names (never values) of the key variables that are set for a provider."""
    return [k for k in PROVIDERS[provider]["key_envs"] if os.environ.get(k)]


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so parallel threads never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class LLMError(RuntimeError):
    pass


class DailyCapReached(LLMError):
    """Every key for this model has used its free daily allowance. The run pauses
    (saved answers are kept in the cache) and continues when it is started again."""


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
        # Daily-cap state: "provider|KEY_NAME|model" -> unix time when it may be tried again.
        # Kept next to the cache so a paused run knows tomorrow which keys were spent.
        self.budget_path = self.cache_dir / "_budget_state.json"
        try:
            self._spent: dict[str, float] = json.loads(self.budget_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._spent = {}
        self.max_cap_wait = float(os.environ.get("LOGICXAI_MAX_CAP_WAIT", "900"))
        self.keys_used: dict[str, int] = {}

    # ---------------------------------------------------------------- utils
    def _cache_file(self, req_hash: str) -> Path:
        return self.cache_dir / req_hash[:2] / f"{req_hash}.json"

    def _log(self, record: dict[str, Any]) -> None:
        with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(canonical_json(record) + "\n")

    @staticmethod
    def available(provider: str) -> bool:
        return provider in ("ollama", "mock") or bool(key_names(provider))

    def _mark_spent(self, slot: str, until: float) -> None:
        with self._lock:
            self._spent[slot] = until
            _atomic_write(self.budget_path, json.dumps(self._spent, indent=1, sort_keys=True))

    def _key_slots(self, provider: str, model_id: str) -> list[tuple[str, str]]:
        """(KEY_NAME, slot) pairs for keys that are not spent right now."""
        now = time.time()
        if not PROVIDERS[provider]["key_envs"]:
            return [("", f"{provider}||{model_id}")]
        out = []
        for k in key_names(provider):
            slot = f"{provider}|{k}|{model_id}"
            if self._spent.get(slot, 0) <= now:
                out.append((k, slot))
        return out

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
            except DailyCapReached as e:
                # Default: wait for tomorrow rather than switch to another copy of the weights
                # mid-run (e.g. a quantised local copy). LOGICXAI_CAP_FALLBACK=1 allows it.
                last_err = e
                if os.environ.get("LOGICXAI_CAP_FALLBACK") != "1":
                    raise
            except LLMError as e:
                last_err = e
        if isinstance(last_err, DailyCapReached):
            raise last_err
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
        url = cfg["base_url"] + "/chat/completions"
        send = {k: v for k, v in body.items() if not (k == "seed" and provider in NO_SEED)}

        slots = self._key_slots(provider, model_id)
        if len(slots) > 1:  # spread calls over the keys (per-minute limits are per key too)
            with self._lock:
                self._rr = getattr(self, "_rr", 0) + 1
                k = self._rr % len(slots)
            slots = slots[k:] + slots[:k]
        if not slots:
            raise DailyCapReached(f"{provider}/{model_id}: every key has used its free daily limit")
        delay = 2.0
        attempt = 0
        while attempt < self.max_retries:
            if not slots:
                raise DailyCapReached(f"{provider}/{model_id}: every key has used its free daily limit")
            key_name, slot = slots[0]
            headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
            if key_name:
                headers["Authorization"] = f"Bearer {os.environ[key_name]}"
            attempt += 1
            t0 = time.time()
            try:
                r = requests.post(url, headers=headers, json=send, timeout=self.timeout)
            except requests.RequestException as e:
                err = f"network: {e}"
            else:
                if r.status_code == 200:
                    try:
                        payload = r.json()
                        raw = payload["choices"][0]["message"]["content"] or ""
                    except (KeyError, IndexError, TypeError, ValueError):
                        err = f"bad payload: {r.text[:200]}"
                    else:
                        # v0.2.1: always-thinking models (e.g. Gemma 4 on Google) put their
                        # reasoning in <thought>...</thought> inside the reply. It is kept in the
                        # cache and log but never scored and never shown to a solver (an
                        # assistant's thoughts could reveal a planted error).
                        text, thought = strip_thoughts(raw)
                        result = {
                            "text": text,
                            "thought_text": thought,
                            "raw_sha256": sha256_text(raw),
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
                        with self._lock:
                            self.calls += 1
                            self.keys_used[key_name or provider] = self.keys_used.get(key_name or provider, 0) + 1
                        self._log({**{k: v for k, v in result.items() if k not in ("text", "thought_text")},
                                   "purpose": purpose, "attempt": attempt, "key_name": key_name,
                                   "latency_s": round(time.time() - t0, 3), "status": "ok"})
                        return {**result, "cached": False}
                else:
                    err = f"HTTP {r.status_code}: {r.text[:300]}"
                    if r.status_code == 429 and DAILY_RE.search(r.text or ""):
                        wait = _parse_wait(r.text)
                        if wait is not None and wait <= self.max_cap_wait:
                            # rolling daily window that frees up soon: wait, same key
                            self._log({"alias": alias, "provider": provider, "request_sha256": req_hash,
                                       "purpose": purpose, "status": "cap_wait", "key_name": key_name,
                                       "wait_s": wait, "error": err[:200], "timestamp_utc": utc_now()})
                            time.sleep(wait + 1)
                            attempt -= 1  # a short wait is not a failed try
                            continue
                        until = time.time() + (wait if wait is not None else 3600)
                        self._mark_spent(slot, until)
                        self._log({"alias": alias, "provider": provider, "request_sha256": req_hash,
                                   "purpose": purpose, "status": "daily_cap", "key_name": key_name,
                                   "retry_after_utc": utc_now(until), "error": err[:200],
                                   "timestamp_utc": utc_now()})
                        slots = slots[1:]  # rotate to the next key
                        attempt -= 1
                        continue
                    if r.status_code in (401, 403) and len(slots) > 1:
                        # this key is invalid or expired: try the next one
                        self._log({"alias": alias, "provider": provider, "request_sha256": req_hash,
                                   "purpose": purpose, "status": "key_rejected", "key_name": key_name,
                                   "error": err[:200], "timestamp_utc": utc_now()})
                        self._mark_spent(slot, time.time() + 6 * 3600)
                        slots = slots[1:]
                        attempt -= 1
                        continue
                    retryable = r.status_code in (408, 409, 425, 429, 500, 502, 503, 504)
                    if not retryable:
                        self._log({"alias": alias, "provider": provider, "request_sha256": req_hash,
                                   "purpose": purpose, "status": "error", "key_name": key_name,
                                   "error": err, "timestamp_utc": utc_now()})
                        raise LLMError(err)
                    ra = r.headers.get("retry-after")
                    if ra:
                        try:
                            delay = max(delay, float(ra))
                        except ValueError:
                            pass
            self._log({"alias": alias, "provider": provider, "request_sha256": req_hash,
                       "purpose": purpose, "status": "retry", "attempt": attempt, "key_name": key_name,
                       "error": err, "timestamp_utc": utc_now()})
            if attempt < self.max_retries:
                time.sleep(min(delay, self.max_sleep))
                delay *= 2
        raise LLMError(f"{provider}/{model_id}: gave up after {self.max_retries} tries")


THOUGHT_RE = re.compile(r"<(thought|think|thinking)>(.*?)(?:</\1>|$)", re.S | re.I)


def strip_thoughts(raw: str) -> tuple[str, str | None]:
    """Split a reply into (answer text, hidden reasoning). An unclosed tag (reply cut off
    while thinking) counts as all reasoning, so a draft answer inside it is never read."""
    thoughts = [m.group(2) for m in THOUGHT_RE.finditer(raw or "")]
    if not thoughts:
        return raw, None
    return THOUGHT_RE.sub("", raw).strip(), "\n".join(thoughts)


def _parse_wait(text: str) -> float | None:
    """'Please try again in 7m26.4s' -> 446.4 seconds (None if no time is given)."""
    m = WAIT_RE.search(text or "")
    if not m or not any(m.groups()):
        return None
    h, mi, sec = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + sec
