# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Provenance: SHA-256 hashing, run IDs and run manifests.

Every file the study writes is hashed so anyone can check that the
data and reports they have are exactly the ones produced by the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON (sorted keys, no extra spaces) so equal content hashes equally."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return sorted(line for line in out.stdout.splitlines() if line.strip())
    except Exception:  # pragma: no cover - environment dependent
        return []


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _pip_freeze(),
        "env_flags": {k: os.environ[k] for k in ("PYTHONHASHSEED",) if k in os.environ},
    }


def hash_tree(root: str | Path, exclude: tuple[str, ...] = ("SHA256SUMS.txt",)) -> dict[str, str]:
    """Hash every file under root. Keys are POSIX paths relative to root."""
    root = Path(root)
    result: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in exclude:
            result[p.relative_to(root).as_posix()] = sha256_file(p)
    return result


def write_sha256sums(root: str | Path) -> Path:
    """Write a SHA256SUMS.txt compatible with `sha256sum -c`."""
    root = Path(root)
    hashes = hash_tree(root)
    out = root / "SHA256SUMS.txt"
    out.write_text("".join(f"{h}  {p}\n" for p, h in hashes.items()), encoding="utf-8")
    return out


def verify_sha256sums(root: str | Path) -> list[str]:
    """Return a list of problems (empty list means everything matches)."""
    root = Path(root)
    problems: list[str] = []
    sums = root / "SHA256SUMS.txt"
    if not sums.exists():
        return ["SHA256SUMS.txt missing"]
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        p = root / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(p) != digest:
            problems.append(f"changed: {rel}")
    return problems
