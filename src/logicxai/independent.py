"""Independent re-implementations, used only by the self-checks.

Every answer key in the study comes from logic_engine.py (brute-force truth
tables and Venn-region models). To catch a bug there, the self-checks
recompute the keys with methods that share no code with it:

  * propositional entailment: a small DPLL SAT solver on clause form
  * syllogism validity: the traditional rules of the syllogism
    (distribution, quality, existential import), not model checking
"""
from __future__ import annotations

from typing import Iterable

Clause = frozenset  # of (name, polarity)


# ------------------------------------------------------------- CNF + DPLL
def _nnf(f: tuple, neg: bool = False) -> tuple:
    op = f[0]
    if op == "var":
        return ("lit", f[1], not neg)
    if op == "not":
        return _nnf(f[1], not neg)
    if op == "imp":  # a -> b  ==  ~a | b
        return _nnf(("or", ("not", f[1]), f[2]), neg)
    if op in ("and", "or"):
        real = op if not neg else ("or" if op == "and" else "and")
        return (real, _nnf(f[1], neg), _nnf(f[2], neg))
    raise ValueError(op)


def _cnf(f: tuple) -> list[set]:
    op = f[0]
    if op == "lit":
        return [{(f[1], f[2])}]
    if op == "and":
        return _cnf(f[1]) + _cnf(f[2])
    if op == "or":
        left, right = _cnf(f[1]), _cnf(f[2])
        return [a | b for a in left for b in right]
    raise ValueError(op)


def clauses(formulas: Iterable[tuple]) -> list[Clause]:
    out = []
    for f in formulas:
        for c in _cnf(_nnf(f)):
            if not any((n, not p) in c for n, p in c):  # drop tautological clauses
                out.append(frozenset(c))
    return out


def dpll(cls: list[Clause], assign: dict[str, bool] | None = None) -> bool:
    """True if the clause set is satisfiable."""
    assign = dict(assign or {})
    cls = list(cls)
    while True:
        new = []
        unit = None
        for c in cls:
            if any(assign.get(n) == p for n, p in c):
                continue
            rest = frozenset((n, p) for n, p in c if n not in assign)
            if not rest:
                return False
            if len(rest) == 1 and unit is None:
                unit = next(iter(rest))
            new.append(rest)
        cls = new
        if not cls:
            return True
        if unit is None:
            break
        assign[unit[0]] = unit[1]
    name = next(iter(next(iter(cls))))[0]
    return dpll(cls, {**assign, name: True}) or dpll(cls, {**assign, name: False})


def status_dpll(premises: list[tuple], query: tuple) -> str:
    base = clauses(premises)
    can_true = dpll(base + clauses([query]))
    can_false = dpll(base + clauses([("not", query)]))
    if can_true and can_false:
        return "unknown"
    if can_true:
        return "true"
    if can_false:
        return "false"
    return "inconsistent"


# ------------------------------------------------- rules of the syllogism
# Distribution: A distributes subject; E both; I neither; O predicate.
_DIST = {"A": (True, False), "E": (True, True), "I": (False, False), "O": (False, True)}
_FIG = {1: (("M", "P"), ("S", "M")), 2: (("P", "M"), ("S", "M")),
        3: (("M", "P"), ("M", "S")), 4: (("P", "M"), ("M", "S"))}


def _distributed(mood: str, subj: str, pred: str) -> set[str]:
    ds, dp = _DIST[mood]
    return ({subj} if ds else set()) | ({pred} if dp else set())


def syllogism_valid_by_rules(major: str, minor: str, concl: str, figure: int) -> bool:
    """Traditional rules (with existential import, i.e. all terms non-empty):
    1. the middle term is distributed at least once;
    2. a term distributed in the conclusion is distributed in its premise;
    3. not both premises negative;
    4. a negative premise iff a negative conclusion;
    5. two particular premises give nothing."""
    (a1, a2), (b1, b2) = _FIG[figure]
    d_major = _distributed(major, a1, a2)
    d_minor = _distributed(minor, b1, b2)
    d_concl = _distributed(concl, "S", "P")
    neg = {"E", "O"}
    if "M" not in d_major | d_minor:
        return False
    if "P" in d_concl and "P" not in d_major:
        return False
    if "S" in d_concl and "S" not in d_minor:
        return False
    if major in neg and minor in neg:
        return False
    if (major in neg or minor in neg) != (concl in neg):
        return False
    if major in "IO" and minor in "IO":
        return False
    return True
