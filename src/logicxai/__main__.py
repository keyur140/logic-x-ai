# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Command line: python -m logicxai <command>

  run       --profile mock|mock_null|mock_quick|pilot|full [--route api|local]
  validate  pipeline validation: planted effect, no-effect world, false-positive rate
  verify    RUN_DIR                 check every file of a run against SHA256SUMS.txt
  check                             ping every model once; record Ollama digests
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

PROFILES = ["mock", "mock_null", "mock_quick", "pilot", "full"]


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
    va = sub.add_parser("validate")
    va.add_argument("--config", default="configs/study.yaml")
    va.add_argument("--out", default="runs")
    va.add_argument("--sims", type=int, default=2000)
    v = sub.add_parser("verify")
    v.add_argument("run_dir")
    c = sub.add_parser("check")
    c.add_argument("--config", default="configs/study.yaml")
    c.add_argument("--route", default=None, choices=["api", "local"])
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
        out = run_study(a.config, a.profile, a.out, a.cache, a.run_id, route=a.route)
        print(f"\nDone. Reports: {out / 'reports'}")
        return 0
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
    import tempfile
    import yaml
    from .llm import LLMClient, LLMError
    from .runner import _ollama_digests, _spec
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    route = route or cfg.get("route", "api")
    tmp = Path(tempfile.mkdtemp())
    client = LLMClient(tmp / "cache", tmp / "log.jsonl")
    ok_all = True
    tags = []
    for role in ("assistants", "solvers"):
        for d in cfg[role]:
            spec = _spec(d, route)
            if spec.provider == "ollama":
                tags.append(spec.model_id)
            try:
                res = client.chat(spec, [{"role": "user", "content": "Reply with the word OK."}], max_tokens=5)
                print(f"OK    {role[:-1]:9s} {d['alias']:22s} via {res['provider']} ({res['model_reported']})")
            except LLMError as e:
                ok_all = False
                print(f"FAIL  {role[:-1]:9s} {d['alias']:22s} {str(e)[:120]}")
    if tags:
        print("\nOllama digests (exact model fingerprints):")
        for t, dg in _ollama_digests(sorted(set(tags))).items():
            print(f"  {t:28s} {dg}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
