# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)


"""v0.2.0 safeguards: independent key checks, freeze/lock, gates, validation, mock worlds."""
import ast
import itertools
from pathlib import Path

import numpy as np

from logicxai import logic_engine as L
from logicxai import mock
from logicxai.checks import gate_parser
from logicxai.freeze import freeze_status, study_files, write_freeze
from logicxai.independent import status_dpll, syllogism_valid_by_rules
from logicxai.items import GENERATORS


def test_dpll_agrees_with_engine_on_hard_items():
    for it in GENERATORS["hard_deduction"](80, 21):
        prem = [ast.literal_eval(p) for p in it.meta["premises"]]
        assert status_dpll(prem, L.V(it.meta["query"])).upper() == it.answer


def test_syllogism_rules_agree_with_engine_on_all_256_forms():
    for a, b, c in itertools.product("AEIO", repeat=3):
        for fig in (1, 2, 3, 4):
            assert syllogism_valid_by_rules(a, b, c, fig) == L.syllogism_valid(a, b, c, fig)


def test_catch_items_state_the_answer():
    for it in GENERATORS["catch"](12, 4):
        assert it.answer in ("TRUE", "FALSE")


def test_parser_gate_passes():
    g = gate_parser()
    assert g["pass"], g


def test_freeze_detects_any_change(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "study.yaml").write_text("x: 1\n")
    (tmp_path / "src").mkdir()
    f = tmp_path / "src" / "a.py"
    f.write_text("print(1)\n")
    assert freeze_status(tmp_path)["status"] == "NOT_FROZEN"
    cfg = {"study": {"title": "t", "seed": 1, "dev_seed": 2},
           "profiles": {"full": {"items": {"syllogism": 4}}}}
    write_freeze(cfg, tmp_path)
    assert freeze_status(tmp_path)["status"] == "MATCHES_FREEZE"
    f.write_text("print(2)\n")
    st = freeze_status(tmp_path)
    assert st["status"] == "CHANGED_SINCE_FREEZE" and st["changed"] == ["src/a.py"]
    assert all("freeze" not in p.parts for p in study_files(tmp_path))


def test_mock_worlds_differ_and_are_not_cached_together():
    item = GENERATORS["hard_deduction"](1, 9)[0].to_dict()
    key = mock.build_answer_key([item])
    wrong = next(x for x in ("TRUE", "FALSE", "UNKNOWN") if x != item["answer"])
    user = (item["prompt"] + "\n\n----- ASSISTANT REPLY -----\nStep 1: x\nFINAL: " + wrong +
            "\n----- END OF ASSISTANT REPLY -----\nThink carefully.")
    out = {}
    for world in ("amplify", "null"):
        mock.WORLD = world
        out[world] = sum("ASSISTANT_ERROR: YES" in mock.respond(f"mock-solver-{k:02d}",
                                                                [{"role": "user", "content": user}], key)
                         for k in range(25))
    mock.WORLD = "amplify"
    assert out["amplify"] != out["null"]


def test_validation_gain_score_holds_alpha_and_ancova_does_not():
    from logicxai.validate import _one_h1
    rng = np.random.default_rng(5)
    items = {"syllogism": 120, "conditional": 80, "crt": 30, "mmlu": 400, "hard_error": 150, "easy_math": 200}
    sims = [_one_h1(rng, "ignore", 25, items, 0.05) for _ in range(300)]
    assert np.mean([s["gain_reject"] for s in sims]) < 0.10
    assert np.mean([s["ancova_reject"] for s in sims]) > 0.5
