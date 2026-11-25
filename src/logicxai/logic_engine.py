# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Small exact logic engine used to create and score items.

Nothing here uses an AI: every correct answer in the study is computed by
brute-force model checking, so the answer key is provably right.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable

# --------------------------------------------------------------- syllogisms
# A categorical statement relates two terms. Regions of the Venn diagram are
# subsets of {S, M, P}; a "model" says which regions are non-empty.
TERMS = ("S", "M", "P")
REGIONS = [frozenset(c) for r in range(1, 4) for c in itertools.combinations(TERMS, r)]
REGIONS.append(frozenset())  # outside every circle


def _holds(mood: str, x: str, y: str, model: frozenset) -> bool:
    xs = [r for r in model if x in r]
    if mood == "A":  # All x are y
        return all(y in r for r in xs)
    if mood == "E":  # No x are y
        return all(y not in r for r in xs)
    if mood == "I":  # Some x are y
        return any(y in r for r in xs)
    if mood == "O":  # Some x are not y
        return any(y not in r for r in xs)
    raise ValueError(mood)


def _models(existential_import: bool) -> Iterable[frozenset]:
    for bits in itertools.product((0, 1), repeat=len(REGIONS)):
        model = frozenset(r for r, b in zip(REGIONS, bits) if b)
        if existential_import and not all(any(t in r for r in model) for t in TERMS):
            continue
        yield model


FIGURES = {  # (major premise terms, minor premise terms)
    1: (("M", "P"), ("S", "M")),
    2: (("P", "M"), ("S", "M")),
    3: (("M", "P"), ("M", "S")),
    4: (("P", "M"), ("M", "S")),
}


def syllogism_valid(major: str, minor: str, concl: str, figure: int,
                    existential_import: bool = True) -> bool:
    """True if conclusion `concl` (about S then P) follows in every model."""
    (a1, a2), (b1, b2) = FIGURES[figure]
    for m in _models(existential_import):
        if _holds(major, a1, a2, m) and _holds(minor, b1, b2, m) and not _holds(concl, "S", "P", m):
            return False
    return True


def all_syllogism_forms(existential_import: bool = True):
    """Yield (major, minor, conclusion, figure, valid) for all 256 forms."""
    for fig in FIGURES:
        for a, b, c in itertools.product("AEIO", repeat=3):
            yield a, b, c, fig, syllogism_valid(a, b, c, fig, existential_import)


# ------------------------------------------------------ propositional logic
# Formulas are nested tuples: ("var","p"), ("not",f), ("and",f,g),
# ("or",f,g), ("imp",f,g). Small and easy to check.
Formula = tuple


def V(name: str) -> Formula: return ("var", name)
def Not(f: Formula) -> Formula: return ("not", f)
def neg(f: Formula) -> Formula:
    """Negate, collapsing double negation so sentences stay readable."""
    return f[1] if f[0] == "not" else ("not", f)


def And(f: Formula, g: Formula) -> Formula: return ("and", f, g)
def Or(f: Formula, g: Formula) -> Formula: return ("or", f, g)
def Imp(f: Formula, g: Formula) -> Formula: return ("imp", f, g)


def evaluate(f: Formula, env: dict[str, bool]) -> bool:
    op = f[0]
    if op == "var":
        return env[f[1]]
    if op == "not":
        return not evaluate(f[1], env)
    if op == "and":
        return evaluate(f[1], env) and evaluate(f[2], env)
    if op == "or":
        return evaluate(f[1], env) or evaluate(f[2], env)
    if op == "imp":
        return (not evaluate(f[1], env)) or evaluate(f[2], env)
    raise ValueError(op)


def variables(f: Formula, acc: set[str] | None = None) -> set[str]:
    acc = set() if acc is None else acc
    if f[0] == "var":
        acc.add(f[1])
    else:
        for sub in f[1:]:
            variables(sub, acc)
    return acc


def entailment_status(premises: list[Formula], query: Formula) -> str:
    """'true' if premises entail query, 'false' if they entail its negation,
    'unknown' otherwise, 'inconsistent' if premises have no model."""
    names = sorted(set().union(*(variables(p) for p in premises), variables(query)))
    can_true = can_false = False
    for vals in itertools.product((False, True), repeat=len(names)):
        env = dict(zip(names, vals))
        if all(evaluate(p, env) for p in premises):
            if evaluate(query, env):
                can_true = True
            else:
                can_false = True
            if can_true and can_false:
                return "unknown"
    if can_true:
        return "true"
    if can_false:
        return "false"
    return "inconsistent"


def entails(premises: list[Formula], query: Formula) -> bool:
    return entailment_status(premises, query) == "true"


@dataclass(frozen=True)
class Proof:
    """A forward-chaining derivation: list of (derived literal, rule index)."""
    steps: tuple[tuple[str, bool, int], ...]


def forward_chain(facts: dict[str, bool], rules: list[tuple[list[tuple[str, bool]], tuple[str, bool]]]
                  ) -> tuple[dict[str, bool], Proof]:
    """Horn-style chaining over literals. Rule = (body literals, head literal).
    Used only to write the step-by-step solutions the assistant may show."""
    known = dict(facts)
    steps: list[tuple[str, bool, int]] = []
    changed = True
    while changed:
        changed = False
        for i, (body, (hv, hval)) in enumerate(rules):
            if hv in known:
                continue
            if all(known.get(v) == val for v, val in body):
                known[hv] = hval
                steps.append((hv, hval, i))
                changed = True
    return known, Proof(tuple(steps))


def tautology(f: Formula) -> bool:
    names = sorted(variables(f))
    return all(evaluate(f, dict(zip(names, vals)))
               for vals in itertools.product((False, True), repeat=len(names)))


def to_text(f: Formula, name: Callable[[str], str] = lambda s: s) -> str:
    op = f[0]
    if op == "var":
        return name(f[1])
    if op == "not":
        return f"it is not the case that {to_text(f[1], name)}"
    if op == "and":
        return f"({to_text(f[1], name)} and {to_text(f[2], name)})"
    if op == "or":
        return f"(either {to_text(f[1], name)} or {to_text(f[2], name)}, or both)"
    if op == "imp":
        return f"(if {to_text(f[1], name)}, then {to_text(f[2], name)})"
    raise ValueError(op)
