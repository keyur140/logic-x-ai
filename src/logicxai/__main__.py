# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Command line: python -m logicxai <command>

  run       --profile mock|mock_null|mock_quick|pilot|repilot|full [--route api|local]
            [--run-id ID] [--until-done]   (--until-done waits out free daily limits)
  validate  pipeline validation: planted effect, no-effect world, false-positive rate
  verify    RUN_DIR                 check every file of a run against SHA256SUMS.txt
  check                             ping every model once (fast, parallel); record Ollama digests
  plan      [--profile full]        estimate calls, tokens and days under the free limits
  items     --profile P --out DIR   only generate items (inspection / OSF upload)
  power                             power simulation used in the preregistration
  freeze                            hash every study file and the registered items (OSF)
  status                            NOT_FROZEN | MATCHES_FREEZE | CHANGED_SINCE_FREEZE
  package   [--out DIR]             build the release zip and print its SHA-256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROFILES = ["mock", "mock_null", "mock_quick", "pilot", "repilot", "repilot_gemma", "repilot_assist", "full"]


def _load_env(path: str = ".env") -> None:
    import os
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and k not in os.environ:
            os.environ[k] = v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="logicxai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", default="pilot", choices=PROFILES)
    r.add_argument("--config", default="configs/study.yaml")
    r.add_argument("--out", default="runs")
    r.add_argument("--cache", default=".cache/llm")
    r.add_argument("--route", default=None, choices=["api", "local"])
    r.add_argument("--run-id", default=None, help="reuse a run id to resume into the same folder")
    r.add_argument("--until-done", action="store_true",
                   help="when the free daily limits pause the run, wait and continue by itself")
    r.add_argument("--wait-minutes", type=float, default=60.0,
                   help="with --until-done: minutes to wait after a pause before trying again")
    va = sub.add_parser("validate")
    va.add_argument("--config", default="configs/study.yaml")
    va.add_argument("--out", default="runs")
    va.add_argument("--sims", type=int, default=2000)
    v = sub.add_parser("verify")
    v.add_argument("run_dir")
    c = sub.add_parser("check")
    c.add_argument("--config", default="configs/study.yaml")
    c.add_argument("--route", default=None, choices=["api", "local"])
    pl = sub.add_parser("plan")
    pl.add_argument("--profile", default="full", choices=PROFILES)
    pl.add_argument("--config", default="configs/study.yaml")
    i = sub.add_parser("items")
    i.add_argument("--profile", default="pilot", choices=PROFILES)
    i.add_argument("--config", default="configs/study.yaml")
    i.add_argument("--out", default="items_preview")
    sub.add_parser("power")
    fz = sub.add_parser("freeze")
    fz.add_argument("--config", default="configs/study.yaml")
    sub.add_parser("status")
    pk = sub.add_parser("package")
    pk.add_argument("--out", default="dist")
    a = ap.parse_args(argv)
    _load_env()

    if a.cmd == "run":
        from .runner import run_study
        if not a.until_done:
            out = run_study(a.config, a.profile, a.out, a.cache, a.run_id, route=a.route)
            print(f"\nDone. Reports: {out / 'reports'}")
            return 0
        # v0.2.1 (decision 18): free tiers cap each key per day. Keep the same run id and
        # continue from the saved answers after each pause until everything is answered.
        import time
        from .provenance import new_run_id
        run_id = a.run_id or f"{new_run_id()}-{a.profile}"
        attempt = 0
        while True:
            attempt += 1
            try:
                out = run_study(a.config, a.profile, a.out, a.cache, run_id, route=a.route)
                print(f"\nDone. Reports: {out / 'reports'}")
                return 0
            except SystemExit as e:
                if e.code != 3:
                    raise
                print(f"Paused (free daily limits), pass {attempt}. Waiting {a.wait_minutes:.0f} min, "
                      f"then continuing run {run_id}. Safe to stop with Ctrl+C and re-run later with "
                      f"--run-id {run_id}.", flush=True)
                time.sleep(a.wait_minutes * 60)
    if a.cmd == "validate":
        from .validate import run_validation
        out = run_validation(a.config, a.out, a.sims)
        print(f"\nPipeline validation written to {out}")
        return 0
    if a.cmd == "verify":
        from .provenance import verify_sha256sums
        probs = verify_sha256sums(a.run_dir)
        print("All files match SHA256SUMS.txt" if not probs else "\n".join(probs))
        return 0 if not probs else 1
    if a.cmd == "check":
        return _check(a.config, a.route)
    if a.cmd == "plan":
        from .plan import print_plan
        return print_plan(a.config, a.profile)
    if a.cmd == "items":
        import yaml
        from .items import GENERATORS
        from .provenance import canonical_json, write_sha256sums
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        prof = cfg["profiles"][a.profile]
        if prof.get("item_set") == "registered":
            from .freeze import freeze_status
            if freeze_status()["status"] == "NOT_FROZEN":
                print("Note: these are the registered items. Inspecting them is fine; sending them "
                      "to models before the freeze is not.")
        seed = int(cfg["study"]["seed"] if prof.get("item_set") == "registered" else cfg["study"]["dev_seed"])
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for fam, n in prof["items"].items():
            with open(out / f"{fam}.jsonl", "w", encoding="utf-8") as f:
                for it in GENERATORS[fam](n, seed):
                    f.write(canonical_json(it.to_dict()) + "\n")
        write_sha256sums(out)
        print(f"Items written to {out} with SHA256SUMS.txt (seed {seed})")
        return 0
    if a.cmd == "power":
        from .power import main as pmain
        pmain()
        return 0
    if a.cmd == "freeze":
        import yaml
        from .freeze import freeze_status, write_freeze
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        out = write_freeze(cfg)
        st = freeze_status()
        print(f"Freeze written: {out}\nFreeze SHA-256: {st['freeze_sha256']}\n"
              f"Code SHA-256:   {st['code_sha256_frozen']}\nStatus: {st['status']}\n"
              "Next: python -m logicxai package, then register both SHA-256 values on OSF.")
        return 0
    if a.cmd == "status":
        from .freeze import freeze_status
        print(json.dumps(freeze_status(), indent=2))
        return 0
    if a.cmd == "package":
        from .freeze import build_package, freeze_status
        z, digest = build_package(Path(a.out))
        print(f"Package: {z}\nSHA-256: {digest}\nFreeze status: {freeze_status()['status']}")
        return 0
    return 1


def _check(config: str, route: str | None) -> int:
    """v0.2.1 fix 10: never hang. Each provider is probed once (5 s); models on a provider
    that cannot be reached are reported at once; every model gets one short try (no long
    retry waits), all in parallel."""
    import os
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    import requests
    import yaml
    from .llm import PROVIDERS, LLMClient, LLMError, key_names
    from .runner import _ollama_digests, _spec
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    route = route or cfg.get("route", "api")
    tmp = Path(tempfile.mkdtemp())
    os.environ.setdefault("LOGICXAI_MAX_SLEEP", "3")
    client = LLMClient(tmp / "cache", tmp / "log.jsonl", max_retries=2, timeout=40)
    rows = [(role[:-1], d, _spec(d, route)) for role in ("assistants", "solvers") for d in cfg[role]]

    def reachable(prov: str) -> str:
        if prov == "mock":
            return ""
        base = PROVIDERS[prov]["base_url"]
        try:
            requests.get(base.rsplit("/v1", 1)[0] if prov == "ollama" else base, timeout=5)
            return ""
        except requests.RequestException as e:
            return f"cannot reach {base} ({type(e).__name__})"

    provs = sorted({sp.provider for _, _, sp in rows} | {d.get("fallback_provider") for _, d, _ in rows} - {None})
    with ThreadPoolExecutor(max_workers=8) as ex:
        reach = dict(zip(provs, ex.map(reachable, provs)))
    print("Providers:")
    for p in provs:
        keys = "" if p in ("ollama", "mock") else f"  keys set: {', '.join(key_names(p)) or 'NONE'}"
        print(f"  {p:12s} {'reachable' if not reach[p] else reach[p]}{keys}")

    def one(row):
        role, d, spec = row
        if reach.get(spec.provider) and not (d.get("fallback_provider") and not reach.get(d["fallback_provider"])):
            return f"FAIL  {role:9s} {d['alias']:22s} {reach[spec.provider]}", False
        try:
            res = client.chat(spec, [{"role": "user", "content": "Reply with the word OK."}], max_tokens=16)
            return f"OK    {role:9s} {d['alias']:22s} via {res['provider']} ({res['model_reported']})", True
        except LLMError as e:
            return f"FAIL  {role:9s} {d['alias']:22s} {str(e)[:140]}", False

    print("\nModels:")
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, rows))
    for line, _ in results:
        print(line)
    tags = sorted({sp.model_id for _, _, sp in rows if sp.provider == "ollama"}
                  | {d["fallback_model_id"] for _, d, _ in rows if d.get("fallback_provider") == "ollama"})
    if tags and not reach.get("ollama"):
        print("\nOllama digests (exact model fingerprints):")
        for t, dg in _ollama_digests(tags).items():
            print(f"  {t:28s} {dg}")
    n_fail = sum(not ok for _, ok in results)
    print(f"\n{len(results) - n_fail} of {len(results)} models answered.")
    return 0 if not n_fail else 1


if __name__ == "__main__":
    sys.exit(main())
