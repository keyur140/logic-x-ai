# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Produce the assistant's replies that solvers will see.

  easy_math       natural assistant reply (tests H2: normal, easy-to-check help)
  hard_deduction  scripted reply; for a random half of items the assistant writes a
                  confident solution with ONE planted slip that reaches a wrong answer
                  (tests H1 and H4). The other half get a matched correct solution.
  logic_creative  natural suggestions (H3, exploratory)

Every scripted reply is checked: its FINAL answer must match the target and it
must not leak words like "mistake". Failures are retried, then logged and the
item is excluded for that assistant (a preregistered rule).
"""
from __future__ import annotations

import random
from typing import Any

from . import prompts as P
from .llm import DailyCapReached, LLMClient, LLMError, ModelSpec
from .provenance import sha256_text
from .scoring import parse_choice, score_item

MAX_SCRIPT_TRIES = 4


def planted_set(item_ids: list[str], seed: int, rate: float = 0.5) -> set[str]:
    """Exactly round(rate * n) items get a planted error, chosen at random.
    Blocked by answer type (TRUE/FALSE/UNKNOWN are balanced in the item set), the
    choice depends only on item ids and seed, so it is identical for both assistants
    and anyone can re-derive it."""
    ranked = sorted(item_ids, key=lambda i: sha256_text(f"plant|{seed}|{i}"))
    return set(ranked[: round(rate * len(ranked))])


def wrong_target(item: dict[str, Any], seed: int) -> str:
    rng = random.Random(f"wrong|{seed}|{item['item_id']}")
    return rng.choice([c for c in item["choices"] if c != item["answer"]])


def _leaks(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in P.LEAK_WORDS)


def assistant_reply(client: LLMClient, spec: ModelSpec, item: dict[str, Any], seed: int,
                    planted: bool = False) -> dict[str, Any]:
    fam = item["family"]
    msgs = lambda content: [{"role": "system", "content": P.ASSISTANT_SYSTEM},
                            {"role": "user", "content": content}]
    base = {"assistant": spec.alias, "item_id": item["item_id"], "family": fam}

    if fam == "easy_math":
        r = client.chat(spec, msgs(P.ASSISTANT_NATURAL.format(prompt=item["prompt"])),
                        purpose=f"assistant:{fam}")
        sc = score_item(item, r["text"])
        return {**base, "text": r["text"], "condition": "natural", "planted": False,
                "assistant_pred": sc["pred"], "assistant_correct": sc["correct"],
                "error_present": int(not sc["correct"]), "ok": True,
                "response_sha256": r["response_sha256"], "model_reported": r["model_reported"]}

    if fam == "logic_creative":
        r = client.chat(spec, msgs(P.ASSISTANT_CREATIVE.format(prompt=item["prompt"])),
                        purpose=f"assistant:{fam}")
        return {**base, "text": r["text"], "condition": "natural", "planted": False, "ok": True,
                "response_sha256": r["response_sha256"], "model_reported": r["model_reported"]}

    if fam not in ("hard_deduction", "catch"):
        raise ValueError(fam)

    target = wrong_target(item, seed) if planted else item["answer"]
    failures: list[str] = []
    for attempt in range(1, MAX_SCRIPT_TRIES + 1):
        retry = "" if attempt == 1 else f"\n(Version {attempt}.)"
        if planted:
            content = P.ASSISTANT_SCRIPTED_ERROR.format(prompt=item["prompt"], wrong=target, retry=retry)
        else:
            content = P.ASSISTANT_SCRIPTED_CLEAN.format(prompt=item["prompt"], key=target) + retry
        try:
            r = client.chat(spec, msgs(content), purpose=f"assistant:{fam}:{'error' if planted else 'clean'}")
        except DailyCapReached:
            raise  # pause the run; never count a daily limit as a failed reply
        except LLMError as e:
            failures.append(f"api: {e}")
            continue
        pred = parse_choice(r["text"], item["choices"])
        if pred != target:
            failures.append(f"final={pred} target={target}")
            continue
        if _leaks(r["text"]):
            failures.append("leak")
            continue
        return {**base, "text": r["text"], "condition": "error" if planted else "clean",
                "planted": planted, "target": target, "assistant_pred": pred,
                "assistant_correct": int(pred == item["answer"]),
                "error_present": int(pred != item["answer"]), "ok": True, "attempts": attempt,
                "response_sha256": r["response_sha256"], "model_reported": r["model_reported"]}
    return {**base, "text": None, "condition": "error" if planted else "clean", "planted": planted,
            "target": target, "ok": False, "failures": failures}
