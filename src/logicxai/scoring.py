# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Turn model replies into scores. Rule-based and exact: no AI judge needed."""
from __future__ import annotations

import ast
import re
from typing import Any

from . import logic_engine as L

FINAL_RE = re.compile(r"FINAL\s*[:=]\s*\**\s*([A-Za-z]+|-?[\d,]+(?:\.\d+)?)", re.I)
FLAG_RE = re.compile(r"ASSISTANT[_ ]ERROR\s*[:=]\s*\**\s*(YES|NO)", re.I)


def parse_choice(text: str, choices: list[str]) -> str | None:
    """Prefer an explicit 'FINAL: X'; otherwise the last choice word mentioned."""
    m = FINAL_RE.findall(text or "")
    for cand in reversed(m):
        c = cand.upper()
        if c in choices:
            return c
    if all(len(c) == 1 for c in choices):
        # Single letters: never scan free text (the word "a" would match "A").
        pats = [r"(?i:answer|option)\s*(?:is)?\s*[:\-]?\s*\(?([A-E])\)?\b", r"^\s*\(?([A-E])[).]?\s*$"]
        hits = []
        for pat in pats:
            for mm in re.finditer(pat, text or "", re.M):
                if mm.group(1) in choices:
                    hits.append((mm.start(), mm.group(1)))
        return max(hits)[1] if hits else None
    up = (text or "").upper()
    best, pos = None, -1
    for c in choices:
        for mm in re.finditer(rf"\b{re.escape(c)}\b", up):
            # 'INVALID' contains 'VALID' only as a substring; \b handles it.
            if mm.start() > pos:
                best, pos = c, mm.start()
    return best


def parse_int(text: str) -> int | None:
    m = FINAL_RE.findall(text or "")
    cands = m or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    for c in reversed(cands):
        c = c.replace(",", "")
        try:
            v = float(c)
        except ValueError:
            continue
        if v.is_integer():
            return int(v)
    return None


def parse_flag(text: str) -> bool | None:
    m = FLAG_RE.findall(text or "")
    if not m:
        return None
    return m[-1].upper() == "YES"


def score_item(item: dict[str, Any], text: str) -> dict[str, Any]:
    t = item["answer_type"]
    if t == "choice":
        pred = parse_choice(text, item["choices"])
        return {"pred": pred, "correct": int(pred == item["answer"]), "parsed": pred is not None}
    if t == "int":
        pred = parse_int(text)
        return {"pred": pred, "correct": int(pred == item["answer"]), "parsed": pred is not None}
    raise ValueError(t)


# ---------------------------------------------------- logical creativity
_TOKEN = re.compile(r"\s*(->|[()~&|]|X\d+)", re.I)


def parse_formula(s: str) -> L.Formula:
    """Parse '~', '&', '|', '->' with usual precedence (~ > & > | > ->, -> right-assoc)."""
    s = s.replace("¬", "~").replace("∧", "&").replace("∨", "|").replace("→", "->").replace("=>", "->")
    tokens: list[str] = []
    pos = 0
    s = s.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ValueError(f"bad token at {pos}: {s[pos:pos+8]!r}")
        tokens.append(m.group(1).upper())
        pos = m.end()
        while pos < len(s) and s[pos].isspace():
            pos += 1
    i = 0

    def peek():
        return tokens[i] if i < len(tokens) else None

    def eat(tok=None):
        nonlocal i
        t = peek()
        if t is None or (tok and t != tok):
            raise ValueError(f"expected {tok}, got {t}")
        i += 1
        return t

    def imp():
        left = disj()
        if peek() == "->":
            eat("->")
            return L.Imp(left, imp())
        return left

    def disj():
        left = conj()
        while peek() == "|":
            eat("|")
            left = L.Or(left, conj())
        return left

    def conj():
        left = unary()
        while peek() == "&":
            eat("&")
            left = L.And(left, unary())
        return left

    def unary():
        t = peek()
        if t == "~":
            eat("~")
            return L.Not(unary())
        if t == "(":
            eat("(")
            f = imp()
            eat(")")
            return f
        if t and t.startswith("X"):
            eat()
            return L.V("x" + t[1:])
        raise ValueError(f"unexpected {t}")

    f = imp()
    if i != len(tokens):
        raise ValueError("trailing tokens")
    return f


def truth_table_key(f: L.Formula, atoms: list[str]) -> str:
    """Canonical meaning of a formula: its truth table over the item's atoms.
    Two conclusions that say the same thing get the same key."""
    import itertools
    bits = []
    for vals in itertools.product((False, True), repeat=len(atoms)):
        bits.append("1" if L.evaluate(f, dict(zip(atoms, vals))) else "0")
    return "".join(bits)


def score_creative(item: dict[str, Any], text: str) -> dict[str, Any]:
    """Count distinct, valid, non-trivial conclusions (by meaning, not wording)."""
    premises = [ast.literal_eval(p) for p in item["meta"]["premises"]]
    atoms = item["meta"]["atoms"]
    premise_keys = {truth_table_key(p, atoms) for p in premises}
    lines = [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", ln).strip().strip("`")
             for ln in (text or "").splitlines()]
    valid_keys: list[str] = []
    n_lines = n_parsed = n_invalid = n_trivial = 0
    for ln in lines:
        if not ln or not re.search(r"X\d", ln, re.I):
            continue
        n_lines += 1
        try:
            f = parse_formula(ln)
        except ValueError:
            continue
        if not L.variables(f) <= set(atoms):
            continue
        n_parsed += 1
        key = truth_table_key(f, atoms)
        if not L.entails(premises, f):
            n_invalid += 1
            continue
        if L.tautology(f) or key in premise_keys:
            n_trivial += 1
            continue
        if key not in valid_keys:
            valid_keys.append(key)
    return {"n_lines": n_lines, "n_parsed": n_parsed, "n_valid_distinct": len(valid_keys),
            "n_invalid": n_invalid, "n_trivial": n_trivial, "valid_keys": valid_keys,
            "validity_rate": (len(valid_keys) / n_parsed) if n_parsed else 0.0}
