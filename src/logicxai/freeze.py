"""Freeze, lock and package.

  freeze   writes freeze/FREEZE.json: SHA-256 of every study file (code, config,
           prompts, tests, requirements) and of every registered item file.
           Its own hash and the package zip hash go into the OSF registration.
  status   compares the files now on disk with FREEZE.json:
             NOT_FROZEN | MATCHES_FREEZE | CHANGED_SINCE_FREEZE (+ the changed files)
  lock     the registered items (master seed) may only be sent to models when the
           status is MATCHES_FREEZE. Pilot and exploratory runs use the dev seed.
  package  builds the release zip from the frozen files and records its SHA-256.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .provenance import canonical_json, sha256_file, sha256_text, utc_now

ROOT_FILES = ("README.md", "LICENSE", "CITATION.cff", "requirements.txt", "pyproject.toml",
              "run.sh", "run.bat", ".env.example", ".gitignore", "CHANGELOG.md", "ISSUES.md",
              "DEVIATIONS.md")
DIRS = ("src", "configs", "tests", "scripts", "preregistration", "docs", "exploratory_synthetic")
SKIP_PARTS = {"__pycache__", ".pytest_cache", "freeze"}
SKIP_SUFFIX = (".pyc",)


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "configs" / "study.yaml").exists():
            return p
    return Path.cwd()


def study_files(root: Path | None = None) -> list[Path]:
    root = root or project_root()
    files = [root / f for f in ROOT_FILES if (root / f).exists()]
    for d in DIRS:
        for p in sorted((root / d).rglob("*")):
            if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_PARTS) \
                    and not p.name.endswith(SKIP_SUFFIX) and ".egg-info" not in str(p):
                files.append(p)
    return sorted(set(files))


def registered_item_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    from .items import GENERATORS
    seed = int(cfg["study"]["seed"])
    out = {}
    for fam, n in cfg["profiles"]["full"]["items"].items():
        text = "".join(canonical_json(it.to_dict()) + "\n" for it in GENERATORS[fam](n, seed))
        out[fam] = sha256_text(text)
    return out


def write_freeze(cfg: dict[str, Any], root: Path | None = None) -> Path:
    from . import AUTHOR, __version__
    root = root or project_root()
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in study_files(root)}
    body = {
        "study": cfg["study"]["title"], "version": __version__, "author": AUTHOR,
        "frozen_utc": utc_now(), "registered_seed": cfg["study"]["seed"],
        "dev_seed": cfg["study"]["dev_seed"],
        "files": files, "registered_items": registered_item_hashes(cfg),
    }
    body["code_sha256"] = sha256_text(canonical_json(files))
    out = root / "freeze" / "FREEZE.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def freeze_status(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    f = root / "freeze" / "FREEZE.json"
    current = {p.relative_to(root).as_posix(): sha256_file(p) for p in study_files(root)}
    code_now = sha256_text(canonical_json(current))
    if not f.exists():
        return {"status": "NOT_FROZEN", "code_sha256_now": code_now, "changed": []}
    frozen = json.loads(f.read_text(encoding="utf-8"))
    changed = sorted(k for k in set(current) | set(frozen["files"])
                     if current.get(k) != frozen["files"].get(k))
    return {"status": "MATCHES_FREEZE" if not changed else "CHANGED_SINCE_FREEZE",
            "frozen_utc": frozen["frozen_utc"], "freeze_sha256": sha256_file(f),
            "code_sha256_frozen": frozen["code_sha256"], "code_sha256_now": code_now,
            "changed": changed}


def build_package(out_dir: Path, root: Path | None = None) -> tuple[Path, str]:
    """Zip every study file plus the freeze file. Fixed timestamps make the zip
    byte-identical for identical files, so its SHA-256 is reproducible."""
    from . import __version__
    root = root or project_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    z = out_dir / f"logic-x-ai-study-v{__version__}.zip"
    files = study_files(root)
    if (root / "freeze" / "FREEZE.json").exists():
        files.append(root / "freeze" / "FREEZE.json")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files):
            info = zipfile.ZipInfo(f"logic-x-ai/{p.relative_to(root).as_posix()}",
                                   date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if p.suffix in (".sh",) else 0o644) << 16
            zf.writestr(info, p.read_bytes())
    digest = sha256_file(z)
    (out_dir / (z.name + ".sha256")).write_text(f"{digest}  {z.name}\n", encoding="utf-8")
    return z, digest
