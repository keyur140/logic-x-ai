# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Run the whole study: items -> assistant replies -> solver trials -> scores ->
analysis -> reports, with a manifest and SHA-256 checksums for everything."""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from . import AUTHOR, __version__
from . import prompts as P
from .assistant import assistant_reply, planted_set
from .items import GENERATORS
from .llm import LLMClient, LLMError, ModelSpec
from .provenance import (canonical_json, environment_snapshot, new_run_id, sha256_file,
                         sha256_json, utc_now, write_sha256sums)
from .scoring import parse_flag, score_creative, score_item

TASK_FAMILIES = ("easy_math", "hard_deduction", "logic_creative")
BATTERY_FAMILIES = ("syllogism", "conditional", "crt")
LOGIC_BENCH = ("folio", "logiqa")
log = logging.getLogger("logicxai")


# ------------------------------------------------------------------ helpers
def _spec(d: dict[str, Any], route: str = "api") -> ModelSpec:
    """Resolve a model entry to a route. 'local' sends everything to Ollama."""
    if route == "local" and d["provider"] != "mock":
        tag = d.get("ollama_tag") or (d["model_id"] if d["provider"] == "ollama" else None) \
            or (d.get("fallback_model_id") if d.get("fallback_provider") == "ollama" else None)
        if not tag:
            raise ValueError(f"{d['alias']}: no Ollama tag for the local route")
        return ModelSpec(d["alias"], "ollama", tag)
    return ModelSpec(d["alias"], d["provider"], d["model_id"],
                     d.get("fallback_provider"), d.get("fallback_model_id"),
                     tuple(sorted((d.get("extra") or {}).items())), int(d.get("token_allowance") or 0))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json(r) + "\n")


def _setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def _library_deviations() -> list[dict[str, str]]:
    """Installed versions that differ from the pinned requirements (reported, not fatal)."""
    from importlib import metadata
    from .freeze import project_root
    req = project_root() / "requirements.txt"
    out = []
    if not req.exists():
        return out
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, want = line.split("==", 1)
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "not installed"
        if have != want:
            out.append({"package": name, "pinned": want, "installed": have})
    return out


def _ollama_digests(tags: list[str]) -> dict[str, str]:
    """Exact model fingerprints from the Ollama server (empty if it is not running)."""
    import requests
    from .llm import PROVIDERS
    base = PROVIDERS["ollama"]["base_url"].rsplit("/v1", 1)[0]
    try:
        r = requests.get(base + "/api/tags", timeout=10)
        models = {m["name"]: m.get("digest", "") for m in r.json().get("models", [])}
    except Exception:
        return {}
    return {t: models.get(t) or models.get(t + ":latest") or "not pulled" for t in tags}


# --------------------------------------------------------------------- main
def run_study(config_path: str | Path, profile: str, out_root: str | Path = "runs",
              cache_dir: str | Path = ".cache/llm", run_id: str | None = None,
              skip_analysis: bool = False, route: str | None = None) -> Path:
    from .checks import gate_assistant, gate_parser, gate_solvers, self_checks
    from .freeze import freeze_status, registered_item_hashes
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prof = cfg["profiles"][profile]
    st = cfg["study"]
    route = route or cfg.get("route", "api")
    is_mock = prof["solvers"] == "mock"
    item_set = prof.get("item_set", "dev")
    seed = int(st["seed"]) if item_set == "registered" else int(st["dev_seed"])

    # ---- lock: the registered items may only go to models after the freeze
    fz = freeze_status()
    if item_set == "registered" and fz["status"] != "MATCHES_FREEZE":
        raise SystemExit(
            f"LOCKED: the registered study needs a matching freeze (status: {fz['status']}). "
            "Run `python -m logicxai freeze`, register on OSF, and do not change any study file after.")

    run_id = run_id or f"{new_run_id()}-{profile}"
    run_dir = Path(out_root) / run_id
    for sub in ("config", "items", "assistant", "raw", "scored", "analysis", "reports", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir / "logs" / "run.log")
    t_start = time.time()
    log.info("run %s profile=%s item_set=%s seed=%s route=%s", run_id, profile, item_set, seed, route)

    shutil.copy(config_path, run_dir / "config" / "study.yaml")
    prompts_src = Path(P.__file__)
    shutil.copy(prompts_src, run_dir / "config" / "prompts.py")

    manifest: dict[str, Any] = {
        "run_id": run_id, "profile": profile, "synthetic": is_mock,
        "world": prof.get("world") if is_mock else None, "package_version": __version__,
        "author": AUTHOR, "started_utc": utc_now(), "route": route if not is_mock else "mock",
        "config_sha256": sha256_file(config_path), "prompts_sha256": sha256_file(prompts_src),
        "seed": seed, "item_set": item_set, "freeze": fz, "environment": environment_snapshot(),
        "library_deviations": _library_deviations(), "stages": {}, "gates": {},
    }

    # ---- G1: the answer parser must read every known reply correctly
    g1 = gate_parser()
    manifest["gates"]["G1"] = g1
    if not g1["pass"]:
        raise SystemExit(f"Gate G1 failed: the answer parser misread {g1['failures']}")

    if is_mock:
        from . import mock
        mock.WORLD = prof.get("world", "amplify")
        solvers, assistants = mock.roster()
    else:
        solvers, assistants = cfg["solvers"], cfg["assistants"]
        if prof.get("solver_limit"):
            solvers = solvers[: prof["solver_limit"]]
    manifest["solvers"] = solvers
    manifest["assistants"] = assistants
    if not is_mock:
        tags = [_spec(d, "local").model_id for d in solvers + assistants
                if route == "local" or d["provider"] == "ollama" or d.get("fallback_provider") == "ollama"]
        manifest["ollama_digests"] = _ollama_digests(sorted(set(tags)))
    modes = st["modes"]
    client = LLMClient(cache_dir, run_dir / "logs" / "llm_calls.jsonl", seed=seed)

    # ----------------------------------------------------------- 1. items
    items: dict[str, list[dict]] = {}
    for fam, n in prof["items"].items():
        items[fam] = [it.to_dict() for it in GENERATORS[fam](n, seed)]
    bench_info: dict[str, Any] = {}
    for name, n in (prof.get("benchmarks") or {}).items():
        if is_mock:
            from .items import gen_synthetic_benchmark
            items[name] = [it.to_dict() for it in gen_synthetic_benchmark(name, n, seed)]
            bench_info[name] = {"repo": "SYNTHETIC stand-in (offline run)", "n": n, "status": "synthetic"}
            continue
        from .benchmarks import SOURCES, load_benchmark, resolve_revision
        rev = (cfg.get("benchmark_revisions") or {}).get(name) or resolve_revision(SOURCES[name]["repo"])
        try:
            items[name] = [it.to_dict() for it in load_benchmark(name, n, seed, rev)]
            bench_info[name] = {"repo": SOURCES[name]["repo"], "revision": rev, "n": len(items[name]),
                                "status": "ok"}
        except Exception as e:  # benchmark trouble must not kill the main study
            log.warning("benchmark %s failed: %s", name, e)
            bench_info[name] = {"repo": SOURCES[name]["repo"], "revision": rev, "status": f"failed: {e}"}
    for fam, rows in items.items():
        _write_jsonl(run_dir / "items" / f"{fam}.jsonl", rows)
    manifest["stages"]["items"] = {
        "counts": {f: len(r) for f, r in items.items()},
        "sha256": {f: sha256_file(run_dir / "items" / f"{f}.jsonl") for f in items},
        "benchmarks": bench_info,
    }
    # lock check: generated items of this run vs the registered set
    reg = registered_item_hashes(cfg)
    gen_fams = [f for f in prof["items"] if f in reg]
    mine = {f: sha256_file(run_dir / "items" / f"{f}.jsonl") for f in gen_fams}
    reg_ids = set()
    if item_set == "dev":
        for fam in gen_fams:
            reg_ids |= {it.to_dict()["item_sha256"] for it in GENERATORS[fam](cfg["profiles"]["full"]["items"][fam],
                                                                                 int(st["seed"]))}
        mine_ids = {it["item_sha256"] for f in gen_fams for it in items[f]}
        overlap = len(mine_ids & reg_ids)
        lock_ok = overlap == 0
    else:
        overlap = 0
        lock_ok = all(mine[f] == reg[f] for f in gen_fams if prof["items"][f] == cfg["profiles"]["full"]["items"][f])
    manifest["lock"] = {"item_set": item_set, "overlap": overlap, "ok": lock_ok}
    if not lock_ok:
        raise SystemExit("Lock check failed: development and registered items overlap, or registered items changed.")

    all_items = [it for rows in items.values() for it in rows]
    if is_mock:
        from .mock import build_answer_key
        client.mock_answer_key = build_answer_key(all_items)
    log.info("items ready: %s", manifest["stages"]["items"]["counts"])

    # ------------------------------------------------------ 2. assistants
    task_items = [it for f in TASK_FAMILIES + ("catch",) for it in items.get(f, [])]
    hard = items.get("hard_deduction", [])
    plant: set[str] = set()
    for ans in ("TRUE", "FALSE", "UNKNOWN"):  # exact half within each answer type
        plant |= planted_set([it["item_id"] for it in hard if it["answer"] == ans], seed, st["error_rate"])
    manifest["planted_items"] = sorted(plant)
    assist: dict[str, dict[str, dict]] = defaultdict(dict)
    all_assist_rows: list[dict] = []
    for a in assistants:
        spec = _spec(a, route)
        rows = []
        with ThreadPoolExecutor(max_workers=cfg["concurrency"].get(spec.provider, 2)) as ex:
            futs = {ex.submit(_safe_assistant, client, spec, it, seed,
                              it["family"] == "catch" or it["item_id"] in plant): it for it in task_items}
            for fu in as_completed(futs):
                r = fu.result()
                rows.append(r)
                assist[a["alias"]][r["item_id"]] = r
        rows.sort(key=lambda r: r["item_id"])
        all_assist_rows += rows
        _write_jsonl(run_dir / "assistant" / f"{a['alias']}.jsonl", rows)
        hd = [r for r in rows if r["family"] == "hard_deduction"]
        g45 = gate_assistant(rows)
        log.info("assistant %s: %d replies, %d failed manipulation checks; G4 %s, G5 worst AUC %.2f",
                 a["alias"], len(rows), sum(not r["ok"] for r in rows), g45["G4_manipulation_pass"],
                 g45["G5_worst_auc"])
        manifest["gates"].setdefault("G4_G5", {})[a["alias"]] = g45
        manifest["stages"].setdefault("assistants", {})[a["alias"]] = {
            "n": len(rows), "failed": sum(not r["ok"] for r in rows),
            "hard_planted": sum(r["planted"] for r in hd), "hard_clean": sum(not r["planted"] for r in hd),
            "models_reported": sorted({r.get("model_reported") or "" for r in rows} - {""}),
            "sha256": sha256_file(run_dir / "assistant" / f"{a['alias']}.jsonl"),
        }

    # ------------------------------------------------------- 3. solver trials
    jobs: list[dict[str, Any]] = []
    for s in solvers:
        for mode in modes:
            for fam in BATTERY_FAMILIES + LOGIC_BENCH + ("gsm8k",):
                for it in items.get(fam, []):
                    jobs.append({"solver": s, "mode": mode, "item": it, "assistant": None})
            for fam in TASK_FAMILIES + ("catch",):
                for it in items.get(fam, []):
                    if fam != "catch":
                        jobs.append({"solver": s, "mode": mode, "item": it, "assistant": None})
                    for a in assistants:
                        ar = assist[a["alias"]].get(it["item_id"])
                        if ar and ar["ok"]:
                            jobs.append({"solver": s, "mode": mode, "item": it, "assistant": ar})
        for it in items.get("mmlu", []):  # knowledge control: once per model
            jobs.append({"solver": s, "mode": "direct", "item": it, "assistant": None})
    log.info("solver trials to run: %d", len(jobs))

    # One worker pool per model: free API tiers limit each model separately,
    # so models run side by side (scheduling only; answers do not change).
    by_model: dict[tuple, list] = defaultdict(list)
    for j in jobs:
        sp = _spec(j["solver"], route)
        by_model[(sp.provider, sp.model_id)].append(j)
    trials: list[dict[str, Any]] = []
    done = 0
    pools = [(ThreadPoolExecutor(max_workers=cfg["concurrency"].get(prov, 1)), pj)
             for (prov, _m), pj in by_model.items()]
    futs = [ex.submit(_run_trial, client, j, st, run_id, route) for ex, pj in pools for j in pj]
    for fu in as_completed(futs):
        trials.append(fu.result())
        done += 1
        if done % 500 == 0:
            log.info("trials %d / %d", done, len(jobs))
    for ex, _ in pools:
        ex.shutdown()
    trials.sort(key=lambda r: (r["solver"], r["mode"], r["family"], r["item_id"], r["condition"],
                               r["assistant"] or ""))
    _write_jsonl(run_dir / "raw" / "trials.jsonl", trials)
    manifest["stages"]["trials"] = {
        "n": len(trials), "api_errors": sum(r["status"] != "ok" for r in trials),
        "unparsed": sum(r["status"] == "ok" and not r["parsed"] for r in trials),
        "llm_calls_new": client.calls, "llm_cache_hits": client.cache_hits,
        "models_reported": sorted({f"{r['solver']}={r.get('model_reported')}" for r in trials
                                   if r.get("model_reported")}),
        "sha256": sha256_file(run_dir / "raw" / "trials.jsonl"),
    }

    # ---------------------------------------- 4. exclusion gates + scored tables
    import pandas as pd
    tdf = pd.DataFrame([{k: v for k, v in t.items() if k not in ("text", "creative")} for t in trials])
    excl = gate_solvers(tdf)
    excl.to_csv(run_dir / "scored" / "exclusions.csv", index=False)
    manifest["gates"]["G2_G3"] = {"n_units": int(len(excl)), "n_excluded": int(excl["excluded"].sum()),
                                  "excluded": excl.loc[excl["excluded"], ["solver_unit", "reason"]].to_dict("records")}
    log.info("gates G2/G3: %d of %d solver units excluded", int(excl["excluded"].sum()), len(excl))
    from .tables import build_tables
    tables = build_tables(trials, solvers, run_dir / "scored",
                          excluded=set(excl.loc[excl["excluded"], "solver_unit"]))
    manifest["stages"]["scored"] = {k: sha256_file(v) for k, v in tables.items()}

    # ------------------------------------------------- 5. analysis + reports
    if not skip_analysis:
        from .analysis import run_analysis
        from .report import write_reports
        results = run_analysis(run_dir / "scored", run_dir / "analysis", st, seed, manifest)
        manifest["self_checks"] = self_checks(run_dir, items, all_assist_rows, trials,
                                              pd.read_csv(run_dir / "scored" / "solver_level.csv"),
                                              results, manifest, plant, seed, st["error_rate"])
        n_fail = sum(c["result"] == "FAIL" for c in manifest["self_checks"])
        log.info("self-checks: %d passed, %d failed", sum(c["result"] == "pass" for c in manifest["self_checks"]), n_fail)
        manifest["stages"]["analysis"] = {"results_sha256": sha256_file(run_dir / "analysis" / "results.json")}
        manifest["finished_utc"] = utc_now()
        manifest["duration_s"] = round(time.time() - t_start, 1)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
        write_reports(run_dir, manifest, results)

    manifest["finished_utc"] = utc_now()
    manifest["duration_s"] = round(time.time() - t_start, 1)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    log.info("run finished in %.1fs -> %s", manifest["duration_s"], run_dir)
    for h in list(log.handlers):  # close the log before hashing so it never changes after
        h.flush()
        h.close()
        log.removeHandler(h)
    write_sha256sums(run_dir)
    print(f"SHA256SUMS.txt written for {run_dir}")
    return run_dir


def _safe_assistant(client, spec, item, seed, planted):
    try:
        return assistant_reply(client, spec, item, seed, planted)
    except LLMError as e:
        return {"assistant": spec.alias, "item_id": item["item_id"], "family": item["family"],
                "ok": False, "planted": False, "text": None, "failures": [str(e)]}


LEAK_RE = re.compile(r"correct answer is|planted|subtle slip|deliberate|for a study", re.I)


def _run_trial(client: LLMClient, job: dict[str, Any], st: dict[str, Any], run_id: str,
               route: str = "api") -> dict[str, Any]:
    s, mode, it, ar = job["solver"], job["mode"], job["item"], job["assistant"]
    fam = it["family"]
    mode_text = P.MODES[mode]
    if fam == "logic_creative":
        content = it["prompt"]
        if ar:
            content += P.CREATIVE_ASSISTED_BLOCK.format(assistant=ar["text"])
        content += P.CREATIVE_SUFFIX.format(mode=mode_text)
        max_tok = max(st["max_tokens"][mode], 384)
    elif ar:
        content = it["prompt"] + P.ASSISTED_BLOCK.format(assistant=ar["text"]) + \
            P.ASSISTED_SUFFIX.format(mode=mode_text)
        max_tok = st["max_tokens"][mode] + 32
    else:
        content = it["prompt"] + P.ALONE_SUFFIX.format(mode=mode_text)
        max_tok = st["max_tokens"][mode]
    msgs = [{"role": "system", "content": P.SOLVER_SYSTEM}, {"role": "user", "content": content}]
    row: dict[str, Any] = {
        "run_id": run_id, "solver": s["alias"], "solver_provider": s["provider"],
        "params_b": s.get("params_b"), "model_family": s.get("family"), "mode": mode,
        "family": fam, "item_id": it["item_id"], "item_sha256": it["item_sha256"],
        "condition": "assisted" if ar else "alone", "assistant": ar["assistant"] if ar else None,
        "planted": bool(ar and ar.get("planted")), "assistant_error_present":
            (ar.get("error_present") if ar else None),
        "prompt_sha256": sha256_json(msgs), "prompt_leak": bool(LEAK_RE.search(content)),
        "content": it["meta"].get("content"), "belief_conflict": it["meta"].get("belief_conflict"),
    }
    try:
        r = client.chat(_spec(s, route), msgs, max_tokens=max_tok, purpose=f"solver:{fam}:{row['condition']}")
    except LLMError as e:
        return {**row, "status": "error", "error": str(e), "parsed": False, "correct": None,
                "pred": None, "flag": None, "text": None}
    row.update({"status": "ok", "text": r["text"], "response_sha256": r["response_sha256"],
                "model_reported": r["model_reported"], "route_provider": r["provider"]})
    if fam == "logic_creative":
        sc = score_creative(it, r["text"])
        row.update({"parsed": sc["n_parsed"] > 0, "pred": None, "correct": None, "flag": None,
                    "creative": sc})
    else:
        sc = score_item(it, r["text"])
        row.update({"parsed": sc["parsed"], "pred": sc["pred"], "correct": sc["correct"],
                    "flag": parse_flag(r["text"]) if ar else None})
    return row
