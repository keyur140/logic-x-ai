# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Seeded item generators. Same seed -> byte-identical items -> same SHA-256.

Families
  Logic battery (measures each solver's logic, always solved alone):
    syllogism      belief-bias syllogisms (valid/invalid x believable/unbelievable/neutral)
    conditional    MP / MT / AC / DA inferences with real and nonsense content
    crt            cognitive-reflection style items with an intuitive wrong answer
  Tasks (solved alone AND with the assistant):
    easy_math      multi-step word problems, one integer answer (easy to verify)
    hard_deduction rule-chain puzzles: true / false / cannot be determined
                   (long chains, hard to verify; used for planted errors)
    logic_creative list as many distinct valid conclusions as you can (H3, exploratory)
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from . import logic_engine as L
from .provenance import sha256_json


@dataclass
class Item:
    family: str
    item_id: str
    prompt: str
    answer: Any
    answer_type: str            # "choice" | "int" | "set"
    choices: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["item_sha256"] = sha256_json({k: v for k, v in d.items()})
        return d


# ------------------------------------------------------------- word lists
NONSENSE = ["wugs", "blickets", "daxes", "feps", "toves", "zorbs", "glims", "pluks",
            "vards", "snerks", "quals", "mibs", "trons", "yeeps", "kelbs", "frids",
            "norps", "jasks", "plimbs", "wexes"]

# (subset, superset) pairs that are true in the real world, plus disjoint pairs.
SUBSET = [("robins", "birds"), ("salmon", "fish"), ("oaks", "trees"), ("roses", "flowers"),
          ("poodles", "dogs"), ("dogs", "mammals"), ("sparrows", "birds"), ("cats", "mammals"),
          ("apples", "fruits"), ("carrots", "vegetables"), ("whales", "mammals"),
          ("trout", "fish"), ("tulips", "flowers"), ("pines", "trees"), ("eagles", "birds"),
          ("violins", "instruments"), ("hammers", "tools"), ("rubies", "gems"),
          ("cars", "vehicles"), ("lions", "mammals")]
DISJOINT = [("fish", "mammals"), ("birds", "fish"), ("trees", "animals"), ("tools", "fruits"),
            ("gems", "vegetables"), ("vehicles", "flowers"), ("instruments", "mammals"),
            ("rocks", "birds")]

MOOD_TEXT = {"A": "All {x} are {y}.", "E": "No {x} are {y}.",
             "I": "Some {x} are {y}.", "O": "Some {x} are not {y}."}


def _real_truth(mood: str, s: str, p: str) -> bool | None:
    """Real-world truth of '<mood> s are p' if we know it, else None."""
    if (s, p) in SUBSET:
        return {"A": True, "E": False, "I": True, "O": False}[mood]
    if (p, s) in SUBSET:
        return {"A": False, "E": False, "I": True, "O": True}[mood]
    if (s, p) in DISJOINT or (p, s) in DISJOINT:
        return {"A": False, "E": True, "I": False, "O": True}[mood]
    return None


def _id(family: str, i: int, seed: int) -> str:
    return f"{family}-{seed}-{i:05d}"


# --------------------------------------------------------------- syllogism
def gen_syllogisms(n: int, seed: int) -> list[Item]:
    """Balanced over validity x content (believable / unbelievable / neutral)."""
    rng = random.Random(f"syllogism-{seed}")
    forms = list(L.all_syllogism_forms(existential_import=True))
    valid = [f for f in forms if f[4]]
    invalid = [f for f in forms if not f[4]]
    cells = [(v, c) for v in (True, False) for c in ("believable", "unbelievable", "neutral")]
    items: list[Item] = []
    i = 0
    while len(items) < n:
        want_valid, content = cells[i % len(cells)]
        i += 1
        major, minor, concl, fig, ok = rng.choice(valid if want_valid else invalid)
        m = rng.choice(NONSENSE)
        if content == "neutral":
            s, p = rng.sample([w for w in NONSENSE if w != m], 2)
        else:
            want_true = content == "believable"
            pool = [(a, b) for a, b in SUBSET + DISJOINT] + [(b, a) for a, b in SUBSET + DISJOINT]
            rng.shuffle(pool)
            pick = next(((a, b) for a, b in pool if _real_truth(concl, a, b) is want_true), None)
            if pick is None:
                continue
            s, p = pick
        names = {"S": s, "M": m, "P": p}
        (a1, a2), (b1, b2) = L.FIGURES[fig]
        prem1 = MOOD_TEXT[major].format(x=names[a1], y=names[a2])
        prem2 = MOOD_TEXT[minor].format(x=names[b1], y=names[b2])
        conc = MOOD_TEXT[concl].format(x=s, y=p)
        prompt = (
            "Assume the two premises are true, even if they sound odd. Decide only whether the "
            "conclusion follows necessarily from them. Ignore what you know about the world.\n\n"
            f"Premise 1: {prem1}\nPremise 2: {prem2}\nConclusion: {conc}\n\n"
            "Does the conclusion follow necessarily? Answer VALID or INVALID."
        )
        conflict = content != "neutral" and ((content == "believable") != want_valid)
        items.append(Item("syllogism", _id("syllogism", len(items), seed), prompt,
                          "VALID" if ok else "INVALID", "choice", ["VALID", "INVALID"],
                          {"form": f"{major}{minor}{concl}-{fig}", "valid": ok,
                           "content": content, "belief_conflict": conflict}))
    return items


# ------------------------------------------------------------- conditional
COND_CONTENT = [
    ("it rains", "the street gets wet"), ("the alarm rings", "the guards wake up"),
    ("a number is divisible by 4", "it is even"), ("the plant gets no water", "it wilts"),
    ("Ana studies", "Ana passes"), ("the switch is on", "the lamp glows"),
]


def gen_conditionals(n: int, seed: int) -> list[Item]:
    rng = random.Random(f"conditional-{seed}")
    kinds = ["MP", "MT", "AC", "DA"]
    items = []
    for i in range(n):
        kind = kinds[i % 4]
        if rng.random() < 0.5:
            p, q = rng.choice(COND_CONTENT)
            content = "real"
        else:
            a, b = rng.sample(NONSENSE, 2)
            p, q = f"there are {a} in the box", f"there are {b} in the jar"
            content = "nonsense"
        rule = f"If {p}, then {q}."
        fact, concl = {
            "MP": (f"{p[0].upper()+p[1:]}.", q),
            "MT": (f"It is not the case that {q}.", f"it is not the case that {p}"),
            "AC": (f"{q[0].upper()+q[1:]}.", p),
            "DA": (f"It is not the case that {p}.", f"it is not the case that {q}"),
        }[kind]
        valid = kind in ("MP", "MT")
        prompt = ("Assume the statements are true. Decide whether the conclusion follows necessarily.\n\n"
                  f"Rule: {rule}\nFact: {fact}\nConclusion: Therefore, {concl}.\n\n"
                  "Answer VALID or INVALID.")
        items.append(Item("conditional", _id("conditional", i, seed), prompt,
                          "VALID" if valid else "INVALID", "choice", ["VALID", "INVALID"],
                          {"inference": kind, "valid": valid, "content": content}))
    return items


# --------------------------------------------------------------------- CRT
def gen_crt(n: int, seed: int) -> list[Item]:
    """Fresh numbers in three classic structures so answers can't be memorised."""
    rng = random.Random(f"crt-{seed}")
    items = []
    for i in range(n):
        kind = i % 3
        if kind == 0:  # bat and ball
            diff = rng.choice([100, 150, 200, 250, 300])
            ball = rng.choice([5, 10, 15, 20, 25, 30, 40, 45])
            total = diff + 2 * ball
            prompt = (f"A pen and a notebook cost {total} pence in total. The notebook costs "
                      f"{diff} pence more than the pen. How many pence does the pen cost?")
            ans, lure = ball, total - diff
        elif kind == 1:  # widgets
            k = rng.choice([3, 4, 5, 6, 7, 8])
            m = rng.choice([50, 80, 100, 120])
            prompt = (f"If it takes {k} printers {k} minutes to print {k} posters, how many "
                      f"minutes would it take {m} printers to print {m} posters?")
            ans, lure = k, m
        else:  # lily pads
            d = rng.choice([24, 30, 36, 40, 48, 60])
            prompt = (f"A patch of algae doubles in size every day. It takes {d} days to cover "
                      f"the whole pond. How many days does it take to cover half of the pond?")
            ans, lure = d - 1, d // 2
        prompt += "\n\nGive only the final number."
        items.append(Item("crt", _id("crt", i, seed), prompt, ans, "int", [],
                          {"structure": ["bat_ball", "widgets", "lily"][kind], "lure": lure}))
    return items


# --------------------------------------------------------------- easy math
NAMES = ["Asha", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goran", "Hana", "Ivo", "Jun"]
GOODS = [("apples", 3), ("notebooks", 4), ("stickers", 2), ("candles", 5), ("pens", 2)]


def gen_easy_math(n: int, seed: int) -> list[Item]:
    """3-5 step problems with one integer answer; easy to check by recomputing."""
    rng = random.Random(f"easy_math-{seed}")
    items = []
    for i in range(n):
        name = rng.choice(NAMES)
        good, price = rng.choice(GOODS)
        price = price + rng.randint(0, 6)
        boxes = rng.randint(2, 9)
        per_box = rng.randint(3, 12)
        gave = rng.randint(1, boxes * per_box // 3)
        bought_more = rng.randint(1, 10)
        steps = rng.randint(3, 5)
        total = boxes * per_box
        text = f"{name} buys {boxes} boxes of {good} with {per_box} {good} in each box."
        total -= gave
        text += f" {name} gives {gave} {good} to a friend."
        if steps >= 4:
            total += bought_more
            text += f" Later {name} buys {bought_more} more {good}."
        if steps >= 5:
            total *= 2
            text += f" Then a shop doubles everything {name} has as a reward."
        q = f" Each {good[:-1] if good.endswith('s') else good} is worth {price} pounds. " \
            f"What is the total value, in pounds, of the {good} {name} has now?"
        ans = total * price
        prompt = text + q + "\n\nGive only the final number."
        items.append(Item("easy_math", _id("easy_math", i, seed), prompt, ans, "int", [],
                          {"steps": steps}))
    return items


# ---------------------------------------------------------- hard deduction
ATOM_TEXT = ["the red door is open", "the bell is ringing", "the lamp is lit",
             "the gate is locked", "the dog is barking", "the window is shut",
             "the clock is stopped", "the tap is running", "the fan is on",
             "the box is sealed", "the flag is raised", "the drum is beating"]


def gen_hard_deduction(n: int, seed: int, n_atoms: int = 8, chain_len: int = 5) -> list[Item]:
    """Facts + rules (with negations, disjunctions and distractors); the question is
    whether one statement is TRUE, FALSE, or UNKNOWN. Answer by exact model checking.
    Balanced across the three answers."""
    rng = random.Random(f"hard_deduction-{seed}")
    targets = ["true", "false", "unknown"]
    items: list[Item] = []
    attempts = 0
    while len(items) < n:
        attempts += 1
        want = targets[len(items) % 3]
        atoms = [f"x{k}" for k in range(n_atoms)]
        text = dict(zip(atoms, rng.sample(ATOM_TEXT, n_atoms)))
        order = atoms[:]
        rng.shuffle(order)
        premises: list[L.Formula] = []
        start = order[0]
        premises.append(L.V(start) if rng.random() < 0.5 else L.Not(L.V(start)))
        cur = premises[0]
        # main chain
        for k in range(1, chain_len):
            nxt_atom = order[k]
            nxt = L.V(nxt_atom) if rng.random() < 0.6 else L.Not(L.V(nxt_atom))
            form = rng.random()
            if form < 0.6:
                premises.append(L.Imp(cur, nxt))
            elif form < 0.85:  # contrapositive form: if not nxt then not cur
                premises.append(L.Imp(L.neg(nxt), L.neg(cur)))
            else:  # disjunction form: not cur or nxt
                premises.append(L.Or(L.neg(cur), nxt))
            cur = nxt
        # distractors on remaining atoms (and some affirming-the-consequent bait)
        rest = order[chain_len:]
        for a in rest:
            b = rng.choice(order[:chain_len])
            premises.append(L.Imp(L.V(a), L.V(b)) if rng.random() < 0.5 else L.Imp(L.V(b), L.V(a)))
        query_atom = order[chain_len - 1] if want != "unknown" else rng.choice(rest or order)
        query = L.V(query_atom)
        status = L.entailment_status(premises, query)
        if status != want:  # also rejects inconsistent premise sets
            if attempts > 200000:
                raise RuntimeError("hard_deduction generator stuck; check parameters")
            continue
        rng.shuffle(premises)
        name = lambda v: text[v]
        lines = [f"{j+1}. " + _sentence(L.to_text(p, name)) for j, p in enumerate(premises)]
        prompt = ("Read the statements. Assume all of them are true. Use strict logic only.\n\n"
                  + "\n".join(lines) +
                  f"\n\nQuestion: Is it true that {text[query_atom]}?\n"
                  "Answer TRUE if it must be true, FALSE if it must be false, or UNKNOWN if the "
                  "statements do not settle it.")
        items.append(Item("hard_deduction", _id("hard_deduction", len(items), seed), prompt,
                          want.upper(), "choice", ["TRUE", "FALSE", "UNKNOWN"],
                          {"n_premises": len(premises), "chain_len": chain_len,
                           "premises": [repr(p) for p in premises], "query": query_atom,
                           "atom_text": text}))
    return items


def _sentence(s: str) -> str:
    s = s[1:-1] if s.startswith("(") and s.endswith(")") else s
    return s[0].upper() + s[1:] + "."


# -------------------------------------------------------- logical creative
def gen_logic_creative(n: int, seed: int) -> list[Item]:
    """Give 4-5 premises; ask for up to 10 distinct conclusions that follow.
    Each conclusion the solver writes is checked exactly (validity), and
    originality is scored later by how rare the conclusion is across solvers."""
    rng = random.Random(f"logic_creative-{seed}")
    items = []
    for i in range(n):
        atoms = [f"x{k}" for k in range(5)]
        text = dict(zip(atoms, rng.sample(ATOM_TEXT, 5)))
        a, b, c, d, e = atoms
        premises = [L.Imp(L.V(a), L.V(b)), L.Or(L.V(b), L.V(c)),
                    L.Imp(L.V(c), L.Not(L.V(d))), L.Imp(L.Not(L.V(e)), L.V(a))]
        if rng.random() < 0.5:
            premises.append(L.Not(L.V(b)))
        rng.shuffle(premises)
        name = lambda v: text[v]
        lines = [f"{j+1}. " + _sentence(L.to_text(p, name)) for j, p in enumerate(premises)]
        legend = "\n".join(f"  {k.upper()} = {text[k]}" for k in atoms)
        prompt = ("Assume these statements are true:\n\n" + "\n".join(lines) +
                  "\n\nWrite up to 10 DIFFERENT conclusions that follow necessarily from them. "
                  "Be as original as you can, but every conclusion must be logically valid and must "
                  "not just repeat a statement. Write each conclusion as a formula on its own line "
                  "using these letters:\n" + legend +
                  "\nand the symbols ~ (not), & (and), | (or), -> (if...then), with brackets. "
                  "Example line: (X0 & ~X3) -> X4")
        items.append(Item("logic_creative", _id("logic_creative", i, seed), prompt,
                          None, "set", [],
                          {"premises": [repr(p) for p in premises], "atoms": atoms,
                           "atom_text": text}))
    return items


# ------------------------------------------------------------ catch items
def gen_catch(n: int, seed: int) -> list[Item]:
    """Known-answer check for the error flag. Each item states the answer outright
    ("1. The lamp is lit."), plus distractor rules. The assistant is always scripted
    to contradict that stated fact. Any solver that reads the statements can see the
    error, so the flag rate on these items must be clearly above the false-alarm rate
    on clean items, or the error-flag measure is treated as uninformative (gate G6).
    Not used in the hypothesis tests."""
    rng = random.Random(f"catch-{seed}")
    items = []
    for i in range(n):
        atoms = [f"x{k}" for k in range(4)]
        text = dict(zip(atoms, rng.sample(ATOM_TEXT, 4)))
        q = atoms[0]
        fact = L.V(q) if i % 2 == 0 else L.Not(L.V(q))
        premises = [fact, L.Imp(L.V(atoms[1]), L.V(atoms[2])), L.Or(L.V(atoms[2]), L.V(atoms[3]))]
        rng.shuffle(premises)
        status = L.entailment_status(premises, L.V(q)).upper()
        name = lambda v: text[v]
        lines = [f"{j+1}. " + _sentence(L.to_text(p, name)) for j, p in enumerate(premises)]
        prompt = ("Read the statements. Assume all of them are true. Use strict logic only.\n\n"
                  + "\n".join(lines) +
                  f"\n\nQuestion: Is it true that {text[q]}?\n"
                  "Answer TRUE if it must be true, FALSE if it must be false, or UNKNOWN if the "
                  "statements do not settle it.")
        items.append(Item("catch", _id("catch", i, seed), prompt, status, "choice",
                          ["TRUE", "FALSE", "UNKNOWN"],
                          {"premises": [repr(p) for p in premises], "query": q, "atom_text": text}))
    return items


# ------------------------------------------- synthetic stand-ins (offline only)
def gen_synthetic_benchmark(name: str, n: int, seed: int) -> list[Item]:
    """Placeholder items with the same answer format as FOLIO / LogiQA / GSM8K / MMLU,
    used ONLY by the offline synthetic (mock) profiles, which cannot download data.
    They are labelled synthetic in every table and never used with real models."""
    rng = random.Random(f"synthetic-{name}-{seed}")
    items = []
    for i in range(n):
        if name in ("logiqa", "mmlu"):
            ans = rng.choice("ABCD")
            prompt = (f"[SYNTHETIC {name.upper()}-STYLE ITEM {i}] Which option is correct?\n\n"
                      "A. option one\nB. option two\nC. option three\nD. option four\n\n"
                      "Answer with the letter of the best option.")
            items.append(Item(name, f"{name}-syn-{seed}-{i:05d}", prompt, ans, "choice",
                              list("ABCD"), {"synthetic": True}))
        elif name == "folio":
            ans = rng.choice(["TRUE", "FALSE", "UNKNOWN"])
            prompt = (f"[SYNTHETIC FOLIO-STYLE ITEM {i}] Read the statements. Assume all of them are "
                      "true. Use strict logic only.\n\nQuestion: Is it true that the claim holds?\n"
                      "Answer TRUE if it must be true, FALSE if it must be false, or UNKNOWN if the "
                      "statements do not settle it.")
            items.append(Item(name, f"{name}-syn-{seed}-{i:05d}", prompt, ans, "choice",
                              ["TRUE", "FALSE", "UNKNOWN"], {"synthetic": True}))
        else:  # gsm8k
            ans = rng.randint(10, 999)
            prompt = f"[SYNTHETIC GSM8K-STYLE ITEM {i}] Solve the word problem.\n\nGive only the final number."
            items.append(Item(name, f"{name}-syn-{seed}-{i:05d}", prompt, ans, "int", [], {"synthetic": True}))
    return items


GENERATORS = {
    "syllogism": gen_syllogisms, "conditional": gen_conditionals, "crt": gen_crt,
    "easy_math": gen_easy_math, "hard_deduction": gen_hard_deduction,
    "logic_creative": gen_logic_creative, "catch": gen_catch,
}
