# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Synthetic ("mock") models for offline runs. NOT study data.

Used for three things:
1. Running the whole pipeline with no internet and no keys.
2. Planted-effect recovery: in the "amplify" world, logic raises the chance of
   catching a planted AI error. The analysis should find H1 and H4.
3. Null-effect check: in the "null" world, catching errors does NOT depend on logic
   (only on solo ability). The analysis should NOT find H1 or H4.

Each synthetic model has hidden traits drawn from a fixed seed:
  theta  = logic ability, kappa = knowledge (correlated with theta), size.
Reasoning mode shifts theta (direct -0.7, brief 0, careful +0.7).
Two synthetic models answer at random (they show the chance floor) and one
always gives the same answer (it must be removed by the collapse gate G2).
"""
from __future__ import annotations

import hashlib
import math
import random
import re

from .scoring import parse_choice

WORLD = "amplify"           # set by the runner: "amplify" or "null"
AMPLIFY_GAMMA = 1.6          # logit slope of error-catching on logic (amplify world)
NULL_CATCH = 0.35            # constant catch probability (null world)
MODE_SHIFT = {"direct": -0.7, "brief": 0.0, "careful": 0.7}
N_MODELS = 28
RANDOM_MODELS = {25, 26}
COLLAPSED_MODELS = {27}


def _u(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:13], 16) / 16**13


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def traits(k: int) -> dict[str, float]:
    rng = random.Random(f"mock-traits-{k}")
    theta = rng.gauss(0, 1)
    kappa = 0.5 * theta + 0.87 * rng.gauss(0, 1)
    log10_params = 0.75 + 0.45 * kappa + 0.25 * rng.gauss(0, 1)
    return {"theta": theta, "kappa": kappa, "params_b": round(10 ** log10_params, 2)}


def roster() -> tuple[list[dict], list[dict]]:
    solvers = []
    for k in range(N_MODELS):
        kind = "random" if k in RANDOM_MODELS else "collapsed" if k in COLLAPSED_MODELS else "regular"
        solvers.append({"alias": f"mock-solver-{k:02d}", "provider": "mock", "model_id": f"mock-{k:02d}",
                        "params_b": traits(k)["params_b"], "family": f"mockfam{k % 7}",
                        "synthetic_kind": kind})
    assistants = [{"alias": "mock-assistant-a", "provider": "mock", "model_id": "mock-a", "params_b": 70},
                  {"alias": "mock-assistant-b", "provider": "mock", "model_id": "mock-b", "params_b": 72}]
    return solvers, assistants


def lookup_key(user: str, answer_key: dict) -> object:
    return lookup(user, answer_key)[0]


def lookup(user: str, answer_key: dict) -> tuple[object, str | None]:
    """(answer, item family) for the item whose prompt starts this message."""
    for prompt, ans, fam in answer_key.get(user[:160], []):
        if user.startswith(prompt):
            return ans, fam
    return None, None


def build_answer_key(items: list[dict]) -> dict[str, list]:
    idx: dict[str, list] = {}
    for it in items:
        idx.setdefault(it["prompt"][:160], []).append((it["prompt"], it["answer"], it.get("family")))
    return idx


def _mode(text: str) -> str:
    if "immediately" in text:
        return "direct"
    if "briefly" in text:
        return "brief"
    return "careful"


def _choices(user: str) -> list[str]:
    if "Answer VALID or INVALID" in user:
        return ["VALID", "INVALID"]
    if "Answer TRUE if" in user:
        return ["TRUE", "FALSE", "UNKNOWN"]
    if "Answer with the letter" in user:
        return list("ABCD")
    return []


def _fmt(ans: object, choices: list[str]) -> str:
    if choices and len(choices[0]) == 1:
        return f"Answer: {ans}\nFINAL: {ans}"
    return f"FINAL: {ans}"


# ---------------------------------------------------------------- assistant
def _assistant(alias: str, user: str, key: object) -> str:
    if "Give 5 conclusions" in user:
        return "CONCLUSIONS:\nX0 -> X1\n~X1\nX1 | X2\n~X0\nX2"
    tgt = re.search(r"FINAL: (\w+)(?:\s*\(Version \d+\.\))?\s*$", user.strip())
    if tgt:
        # Same template and length distribution for clean and planted replies, so
        # no surface cue gives the answer away (checked by gate G5).
        # v0.2.1: the fixed format the real assistants must follow (6 numbered steps,
        # 450-800 characters), with a random length inside the band.
        pad = int(_u(alias, user, "steps") * 4)
        steps = "\n".join(f"{i + 1}. Apply statement {1 + int(_u(alias, user, str(i)) * 8)} "
                          f"to what we have so far, which gives the next fact"
                          + ", and nothing contradicts it" * (1 if i < pad else 0) + "."
                          for i in range(6))
        return f"{steps}\nFINAL: {tgt.group(1)}"
    if isinstance(key, int):
        ans = key if _u(alias, user, "nat") < 0.9 else key + 7
        return f"Working it out step by step.\nFINAL: {ans}"
    return "FINAL: 0"


# ------------------------------------------------------------------- solver
def respond(alias: str, messages: list[dict[str, str]], answer_key: dict) -> str:
    user = messages[-1]["content"]
    key, family = lookup(user, answer_key)
    if alias.startswith("mock-assistant"):
        return _assistant(alias, user, key)
    k = int(re.search(r"(\d+)$", alias).group(1))
    choices = _choices(user)
    assisted = "----- ASSISTANT REPLY -----" in user
    creative = "DIFFERENT conclusions" in user
    r = lambda *t: _u(alias, user, *t)  # noqa: E731

    if k in COLLAPSED_MODELS:
        if creative:
            return "CONCLUSIONS:\nX0"
        a = choices[0] if choices else 0
        return ("ASSISTANT_ERROR: NO\n" if assisted else "") + _fmt(a, choices)
    if k in RANDOM_MODELS:
        if creative:
            return "CONCLUSIONS:\n" + "\n".join(["~X1", "X0 & X4", "X2"][: int(r("n") * 4)])
        a = choices[int(r("c") * len(choices))] if choices else int(r("i") * 1000)
        flag = "YES" if r("f") < 0.5 else "NO"
        return (f"ASSISTANT_ERROR: {flag}\n" if assisted else "") + _fmt(a, choices)

    t = traits(k)
    theta = t["theta"] + MODE_SHIFT[_mode(user)]
    kappa = t["kappa"]
    fam_knowledge = "MMLU-STYLE" in user
    fam_logic_bench = "FOLIO-STYLE" in user or "LOGIQA-STYLE" in user

    if creative:
        n = max(0, int(round(2 + 1.4 * theta + r("cr") + (1.5 if "ASSISTANT SUGGESTIONS" in user else 0))))
        pool = ["~X1", "~X0", "X2", "~X3", "X4", "X2 | X3", "(~X4) -> X1", "X1 -> X3", "X0 -> X1"]
        return "CONCLUSIONS:\n" + "\n".join(pool[:n])

    # solo success probability, with a chance floor for choice items
    c = 1 / len(choices) if choices else 0.0
    if fam_knowledge:
        p_solo = c + (1 - c) * _sig(1.3 * kappa - 0.2)
    elif choices:
        p_solo = c + (1 - c) * _sig(1.2 * theta - 0.3 + (0.3 if fam_logic_bench else 0))
    else:  # integer answers (CRT, arithmetic, GSM8K-style)
        p_solo = _sig(0.9 + 0.9 * theta + 0.4 * kappa)

    def solo_answer(tag: str, exclude: object = None) -> object:
        if choices:
            right = key if isinstance(key, str) else choices[0]
            if r("solo", tag) < p_solo:
                return right
            wrong = [x for x in choices if x != right and x != exclude] or [x for x in choices if x != right]
            return wrong[int(r("w", tag) * len(wrong))]
        if isinstance(key, int):
            return key if r("solo", tag) < p_solo else key + 3
        return 0

    if not assisted:
        return _fmt(solo_answer("alone"), choices)

    block = user.split("----- ASSISTANT REPLY -----")[1].split("----- END")[0]
    if choices:
        a_ans = parse_choice(block, choices)
        right = key if isinstance(key, str) else choices[0]
    else:
        m = re.findall(r"FINAL:\s*(-?\d+)", block)
        a_ans = int(m[-1]) if m else None
        right = key
    if a_ans == right:
        if r("fa") < 0.04 + 0.08 * _sig(-theta):
            return f"ASSISTANT_ERROR: YES\n{_fmt(solo_answer('fa', exclude=a_ans), choices)}"
        return f"ASSISTANT_ERROR: NO\n{_fmt(a_ans, choices)}"
    # the assistant is wrong
    is_catch = family == "catch"
    if not choices:                         # easy arithmetic: easy to check for everyone
        p_catch = 0.85
    elif is_catch:                           # blatant contradiction of a stated fact
        p_catch = 0.9
    elif WORLD == "amplify":
        p_catch = _sig(-0.8 + AMPLIFY_GAMMA * theta)
    else:
        p_catch = NULL_CATCH
    if r("catch") < p_catch:
        return f"ASSISTANT_ERROR: YES\n{_fmt(solo_answer('fix', exclude=a_ans), choices)}"
    return f"ASSISTANT_ERROR: NO\n{_fmt(a_ans, choices)}"
