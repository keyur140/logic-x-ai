# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""v0.2.1: difficulty ladder, lenient number reading, leak-scan fix, mediation guard,
free-tier key rotation and daily-limit pause, thinking-model pooling."""
import ast
import collections
import json

import numpy as np
import pandas as pd
import pytest

from logicxai import llm as LLM
from logicxai import logic_engine as L
from logicxai.analysis import mediation
from logicxai.independent import status_dpll
from logicxai.items import HARD_LEVELS, gen_hard_deduction
from logicxai.provenance import sha256_json
from logicxai.runner import LEAK_RE, _leak_scan_text
from logicxai.scoring import parse_int

# SHA-256 of the first 75 registered-seed hard items, computed with the v0.2.0 generator.
V020_LEVEL1_SHA = "2ab96094e608f524a2e89a1630d935e91efecf84c63dc921c5427731486bb59f"


def test_level1_is_byte_identical_to_v020():
    items = gen_hard_deduction(300, 20261026)
    assert sha256_json([i.to_dict() for i in items][:75]) == V020_LEVEL1_SHA


def test_ladder_is_balanced_and_keys_are_right():
    items = gen_hard_deduction(96, 7110526)
    cnt = collections.Counter((i.meta.get("level", 1), i.answer) for i in items)
    assert set(cnt.values()) == {8} and len(cnt) == 12
    for it in items:  # independent SAT solver agrees with the truth-table key
        prem = [ast.literal_eval(p) for p in it.meta["premises"]]
        assert status_dpll(prem, L.V(it.meta["query"])).upper() == it.answer


def test_harder_levels_are_longer_and_use_either_or():
    items = gen_hard_deduction(120, 5)
    by = collections.defaultdict(list)
    for i in items:
        by[i.meta.get("level", 1)].append(i)
    sizes = [np.mean([i.meta["n_premises"] for i in by[lv]]) for lv in (1, 2, 3, 4)]
    assert sizes == sorted(sizes) and sizes[-1] >= 2 * sizes[0]
    for lv in (3, 4):
        assert all("either" in i.prompt.lower() for i in by[lv])
        assert all(i.meta["case_splits"] == lv - 2 for i in by[lv])
    assert [lv["level"] for lv in HARD_LEVELS] == [1, 2, 3, 4]


def test_fast_and_slow_entailment_agree():
    for it in gen_hard_deduction(30, 11)[:10]:  # level-1 items are small enough for the slow check
        prem = [ast.literal_eval(p) for p in it.meta["premises"]]
        q = L.V(it.meta["query"])
        assert L.entailment_status(prem, q) == L.entailment_status_fast(prem, q)


@pytest.mark.parametrize("text,want", [
    ("FINAL: Tanya earned $365", 365), ("FINAL: Tanya earned $365.", 365),
    ("FINAL: **$1,250**", 1250), ("Work: 3 x 12\nFINAL: 36 pens", 36),
    ("FINAL: 42", 42), ("The pen costs 5 pence.", 5), ("FINAL: -7", -7), ("No number", None),
])
def test_lenient_number_reading(text, want):
    assert parse_int(text) == want


def test_leak_scan_ignores_the_item_itself():
    item = "All judges deliberate before they rule. Is it true that Sam rules?"
    content = item + "\n\nAn AI assistant was given the same question."
    assert LEAK_RE.search(content)                       # v0.2.0 false alarm
    assert not LEAK_RE.search(_leak_scan_text(content, item))
    assert LEAK_RE.search(_leak_scan_text(content + " It made a deliberate slip.", item))


def test_mediation_is_not_testable_with_one_model():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=3), "m": rng.normal(size=3), "y": rng.normal(size=3),
                       "c": rng.normal(size=3), "solver": ["a", "a", "a"]})
    out = mediation(df, "x", "m", "y", ["c"], "solver", reps=99, seed=1)
    assert "error" in out


class _Resp:
    def __init__(self, code, text="", payload=None):
        self.status_code, self.text, self._p, self.headers = code, text, payload, {}

    def json(self):
        return self._p


def _ok(model):
    return _Resp(200, "ok", {"model": model, "choices": [{"message": {"content": "FINAL: 4"}}]})


def test_daily_limit_rotates_keys_then_pauses(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_4", raising=False)
    used = []

    def post(url, headers, json, timeout):  # noqa: A002
        used.append(headers["Authorization"])
        if headers["Authorization"] == "Bearer k1":
            return _Resp(429, "Rate limit reached on tokens per day (TPD): Limit 200000. "
                              "Please try again in 2h10m5s.")
        return _ok(json["model"])

    monkeypatch.setattr(LLM.requests, "post", post)
    c = LLM.LLMClient(tmp_path / "cache", tmp_path / "log.jsonl", max_retries=3)
    spec = LLM.ModelSpec("m", "groq", "model-x")
    c._rr = -1  # round robin starts at the first key for this test
    r = c.chat(spec, [{"role": "user", "content": "2+2"}])
    assert r["text"] == "FINAL: 4" and used == ["Bearer k1", "Bearer k2"]
    state = json.loads((tmp_path / "cache" / "_budget_state.json").read_text())
    assert "groq|GROQ_API_KEY|model-x" in state
    log = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(x["status"] == "daily_cap" for x in log) and all("k1" not in json.dumps(x) for x in log)

    # second key spent too -> the run must pause, not count a failure
    def post2(url, headers, json, timeout):  # noqa: A002
        return _Resp(429, "Rate limit reached on requests per day (RPD). Please try again in 5h.")
    monkeypatch.setattr(LLM.requests, "post", post2)
    with pytest.raises(LLM.DailyCapReached):
        c.chat(spec, [{"role": "user", "content": "3+3"}])
    # a fresh client (tomorrow's run) remembers which keys are spent without calling
    c2 = LLM.LLMClient(tmp_path / "cache", tmp_path / "log2.jsonl")
    with pytest.raises(LLM.DailyCapReached):
        c2.chat(spec, [{"role": "user", "content": "5+5"}])


def test_parse_wait():
    assert LLM._parse_wait("Please try again in 7m26.4s.") == pytest.approx(446.4)
    assert LLM._parse_wait("try again in 1h") == 3600
    assert LLM._parse_wait("no time here") is None


def test_thinking_models_are_pooled(tmp_path, monkeypatch):
    from logicxai import mock
    from logicxai.runner import run_study
    orig = mock.roster

    def roster():
        s, a = orig()
        s[3] = {**s[3], "thinking": True}
        return s, a
    monkeypatch.setattr(mock, "roster", roster)
    run = run_study("configs/study.yaml", "mock_quick", tmp_path / "runs", tmp_path / "cache")
    sl = pd.read_csv(run / "scored" / "solver_level.csv")
    t = sl[sl["solver"] == "mock-solver-03"]
    assert list(t["mode"]) == ["pooled"] and int(t["thinking"].iloc[0]) == 1
    assert (sl[sl["solver"] != "mock-solver-03"]["mode"] != "pooled").all()


def test_scripted_reply_format_band():
    from logicxai.assistant import _format_problem
    good = "\n".join(f"{i}. " + "Statement %d gives the next fact in the chain of rules." % i
                     for i in range(1, 7)) + "\n" + "x" * 120 + "\nFINAL: TRUE"
    assert _format_problem(good) is None
    assert _format_problem("1. a\n2. b\nFINAL: TRUE").startswith("format")
    assert _format_problem(good.replace("6. ", "six ")).startswith("format: 5 steps")


def test_hidden_reasoning_is_stripped():
    text, thought = LLM.strip_thoughts("<thought>FINAL: FALSE draft</thought>\nFINAL: TRUE")
    assert text == "FINAL: TRUE" and "draft" in thought
    text, thought = LLM.strip_thoughts("<thought>cut off while thinking FINAL: 3")
    assert text == "" and parse_int(text) is None
