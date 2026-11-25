# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Public benchmark items (medium-sized, seeded samples).

  folio   (logic, convergent validity)     TRUE / FALSE / UNKNOWN
  logiqa  (logic, convergent validity)     A-D
  gsm8k   (math, checks easy_math)         integer
  mmlu    (general-knowledge CONTROL)      A-D, logic and maths subjects removed so
                                           the control is knowledge, not reasoning

Each dataset is loaded at a pinned Hugging Face revision (commit hash). If the
config leaves the revision empty, the current one is resolved and recorded in
the run manifest so the exact data can be re-fetched later.
"""
from __future__ import annotations

import random
import re
from typing import Any

from .items import Item

SOURCES = {
    "folio": {"repo": "tasksource/folio", "config": None, "split": "validation"},
    "logiqa": {"repo": "lucasmccabe/logiqa", "config": None, "split": "test"},
    "gsm8k": {"repo": "openai/gsm8k", "config": "main", "split": "test"},
    "mmlu": {"repo": "cais/mmlu", "config": "all", "split": "test"},
}

MMLU_EXCLUDE = {"formal_logic", "logical_fallacies", "abstract_algebra", "college_mathematics",
                "elementary_mathematics", "high_school_mathematics", "high_school_statistics",
                "college_physics", "high_school_physics", "college_computer_science",
                "machine_learning", "econometrics", "college_chemistry", "high_school_chemistry"}
LETTERS = ["A", "B", "C", "D", "E"]


def resolve_revision(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def _load(name: str, revision: str | None):
    from datasets import load_dataset
    src = SOURCES[name]
    kw = {"split": src["split"]}
    if revision:
        kw["revision"] = revision
    if src["config"]:
        return load_dataset(src["repo"], src["config"], **kw)
    return load_dataset(src["repo"], **kw)


def _mc_prompt(context: str, question: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    ctx = f"{context.strip()}\n\n" if context else ""
    return f"{ctx}Question: {question.strip()}\n\n{opts}\n\nAnswer with the letter of the best option."


def load_benchmark(name: str, n: int, seed: int, revision: str | None = None) -> list[Item]:
    ds = _load(name, revision)
    rows = list(ds)
    rng = random.Random(f"bench-{name}-{seed}")
    items: list[Item] = []
    if name == "mmlu":
        rows = [r for r in rows if r["subject"] not in MMLU_EXCLUDE]
        by_subj: dict[str, list] = {}
        for r in rows:
            by_subj.setdefault(r["subject"], []).append(r)
        subjects = sorted(by_subj)
        for s in subjects:
            rng.shuffle(by_subj[s])
        picked, k = [], 0
        while len(picked) < min(n, len(rows)):  # round-robin = stratified by subject
            s = subjects[k % len(subjects)]
            if by_subj[s]:
                picked.append(by_subj[s].pop())
            k += 1
        for i, r in enumerate(picked):
            opts = list(r["choices"])
            items.append(Item("mmlu", f"mmlu-{seed}-{i:05d}", _mc_prompt("", r["question"], opts),
                              LETTERS[int(r["answer"])], "choice", LETTERS[:len(opts)],
                              {"subject": r["subject"]}))
        return items
    rng.shuffle(rows)
    rows = rows[:n]
    for i, r in enumerate(rows):
        if name == "folio":
            lab = str(r["label"]).strip().lower()
            ans = {"true": "TRUE", "false": "FALSE", "uncertain": "UNKNOWN", "unknown": "UNKNOWN"}[lab]
            prem = r["premises"] if isinstance(r["premises"], str) else "\n".join(r["premises"])
            prompt = ("Read the statements. Assume all of them are true. Use strict logic only.\n\n"
                      f"{prem}\n\nQuestion: Is it true that {r['conclusion'].strip().rstrip('.')}?\n"
                      "Answer TRUE if it must be true, FALSE if it must be false, or UNKNOWN if the "
                      "statements do not settle it.")
            items.append(Item("folio", f"folio-{seed}-{i:05d}", prompt, ans, "choice",
                              ["TRUE", "FALSE", "UNKNOWN"], {}))
        elif name == "logiqa":
            opts = list(r["options"])
            items.append(Item("logiqa", f"logiqa-{seed}-{i:05d}",
                              _mc_prompt(r["context"], r["query"], opts),
                              LETTERS[int(r["correct_option"])], "choice", LETTERS[:len(opts)], {}))
        elif name == "gsm8k":
            m = re.search(r"####\s*(-?[\d,]+)", r["answer"])
            if not m:
                continue
            ans = int(m.group(1).replace(",", ""))
            items.append(Item("gsm8k", f"gsm8k-{seed}-{i:05d}",
                              r["question"].strip() + "\n\nGive only the final number.", ans, "int", [], {}))
    return items
