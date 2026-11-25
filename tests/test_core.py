# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Core checks: the answer keys are right, items are reproducible, scoring is exact."""
import itertools

import pytest

from logicxai import logic_engine as L
from logicxai.items import GENERATORS
from logicxai.provenance import sha256_json
from logicxai.scoring import parse_choice, parse_flag, parse_formula, parse_int, score_creative


def test_classic_syllogism_counts():
    # Textbook facts: 24 valid forms with existential import, 15 without.
    assert sum(f[4] for f in L.all_syllogism_forms(True)) == 24
    assert sum(f[4] for f in L.all_syllogism_forms(False)) == 15
    assert L.syllogism_valid("A", "A", "A", 1)       # Barbara
    assert not L.syllogism_valid("A", "A", "A", 2)   # undistributed middle


def test_conditional_inferences():
    p, q = L.V("p"), L.V("q")
    assert L.entails([L.Imp(p, q), p], q)                       # modus ponens
    assert L.entails([L.Imp(p, q), L.Not(q)], L.Not(p))         # modus tollens
    assert not L.entails([L.Imp(p, q), q], p)                   # affirming the consequent
    assert not L.entails([L.Imp(p, q), L.Not(p)], L.Not(q))     # denying the antecedent


@pytest.mark.parametrize("fam", list(GENERATORS))
def test_items_reproducible(fam):
    a = [i.to_dict()["item_sha256"] for i in GENERATORS[fam](30, 11)]
    b = [i.to_dict()["item_sha256"] for i in GENERATORS[fam](30, 11)]
    c = [i.to_dict()["item_sha256"] for i in GENERATORS[fam](30, 12)]
    assert a == b and a != c and len(set(a)) > 1


def test_hard_deduction_keys_are_correct_and_balanced():
    items = GENERATORS["hard_deduction"](60, 3)
    assert {i.answer for i in items} == {"TRUE", "FALSE", "UNKNOWN"}
    for it in items:
        import ast
        prem = [ast.literal_eval(p) for p in it.meta["premises"]]
        assert L.entailment_status(prem, L.V(it.meta["query"])).upper() == it.answer


def test_syllogism_keys_match_engine():
    for it in GENERATORS["syllogism"](60, 5):
        form = it.meta["form"]
        ok = L.syllogism_valid(form[0], form[1], form[2], int(form.split("-")[1]))
        assert (it.answer == "VALID") == ok


def test_parsers():
    assert parse_choice("blah VALID ... FINAL: INVALID", ["VALID", "INVALID"]) == "INVALID"
    assert parse_choice("so it is invalid", ["VALID", "INVALID"]) == "INVALID"
    assert parse_choice("This is a test", ["A", "B", "C", "D"]) is None
    assert parse_choice("Answer: C", ["A", "B", "C", "D"]) == "C"
    assert parse_int("total 1,332 pounds") == 1332
    assert parse_flag("ASSISTANT_ERROR: no\nFINAL: TRUE") is False


def test_formula_parser_precedence():
    f = parse_formula("~X0 & X1 -> X2 | X3")
    env = {"x0": False, "x1": True, "x2": False, "x3": False}
    assert L.evaluate(f, env) is False


def test_creative_scoring_counts_meaning_not_wording():
    it = GENERATORS["logic_creative"](1, 3)[0].to_dict()
    txt = "~X0\n~~~X0\nX0 -> X1\nX4 & X0"
    sc = score_creative(it, txt)
    # "~X0" and "~~~X0" mean the same thing -> one distinct conclusion at most
    assert sc["n_valid_distinct"] <= 1 + 0 and sc["n_parsed"] == 4


def test_canonical_hash_stable():
    assert sha256_json({"b": 1, "a": [1, 2]}) == sha256_json({"a": [1, 2], "b": 1})
