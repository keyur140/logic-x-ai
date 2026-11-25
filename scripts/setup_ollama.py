# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Download (pull) every Ollama model named in configs/study.yaml.
Needs Ollama installed and running: https://ollama.com/download
Usage: python scripts/setup_ollama.py [--only-small]"""
import subprocess
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path("configs/study.yaml").read_text(encoding="utf-8"))
only_small = "--only-small" in sys.argv
tags = []
for d in cfg["solvers"] + cfg["assistants"]:
    for prov, mid in ((d["provider"], d["model_id"]), (d.get("fallback_provider"), d.get("fallback_model_id"))):
        if prov == "ollama" and mid and mid not in tags:
            if only_small and d.get("params_b", 0) > 10:
                continue
            tags.append(mid)
print(f"Pulling {len(tags)} models...")
failed = []
for t in tags:
    print(f"\n=== ollama pull {t}")
    if subprocess.run(["ollama", "pull", t]).returncode != 0:
        failed.append(t)
print("\nDone." if not failed else f"\nFailed: {failed}")
