# Logic x AI study (logicxai) - Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X)
# Pilot fill driver with several Groq keys (one per Groq account).
# Runs the pilot with one key; when that account hits its daily token cap
# ("tokens per day"), the run is stopped and the next key is used.
# A spent key is tried again after 30 minutes (Groq's daily window rolls).
# Key values are never printed; only env var names are logged.
import json, os, subprocess, sys, time

RUN_ID = "pilot-groq-20261107"
CACHE = "/mnt/project-files/logic-x-ai/pilot/_cache"
KEYS = [k for k in ("GROQ_API_KEY_3", "GROQ_API_KEY_2", "GROQ_API_KEY") if os.environ.get(k)]
LOG = f"runs/{RUN_ID}/logs/llm_calls.jsonl"
spent = {}  # key name -> time it hit the daily cap

def say(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)

def tpd_hit(offset):
    try:
        with open(LOG, "rb") as f:
            f.seek(offset)
            return b"tokens per day" in f.read()
    except FileNotFoundError:
        return False

for pass_no in range(1, 200):
    now = time.time()
    ready = [k for k in KEYS if now - spent.get(k, 0) > 1800]
    if not ready:
        say("all keys spent, waiting 10 min")
        time.sleep(600)
        continue
    key = ready[0]
    say(f"pass {pass_no} with {key}")
    offset = os.path.getsize(LOG) if os.path.exists(LOG) else 0
    env = dict(os.environ, GROQ_API_KEY=os.environ[key],
               LOGICXAI_MAX_RETRIES="80", LOGICXAI_MAX_SLEEP="90")
    p = subprocess.Popen([sys.executable, "-m", "logicxai", "run", "--config", "configs/pilot_groq.yaml",
                          "--profile", "pilot", "--cache", CACHE, "--run-id", RUN_ID, "--out", "runs"], env=env)
    capped = False
    while p.poll() is None:
        time.sleep(20)
        if tpd_hit(offset):
            capped = True
            say(f"{key}: daily token cap hit, stopping this pass")
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
    if capped:
        spent[key] = time.time()
        continue
    try:
        errs = json.load(open(f"runs/{RUN_ID}/manifest.json"))["stages"]["trials"]["api_errors"]
    except Exception as e:
        errs = f"unknown ({e})"
    say(f"pass {pass_no} done, api_errors={errs}")
    if errs == 0:
        say("ALL DONE")
        break
    time.sleep(120)
