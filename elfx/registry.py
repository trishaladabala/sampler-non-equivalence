"""Run directories, atomic JSON, and the single-source-of-truth registry.csv."""
import csv
import hashlib
import json
import os
import subprocess
import time

from . import CODE, ROOT, RUNS

REGISTRY_FILE = os.path.join(ROOT, "registry.csv")
REGISTRY_COLUMNS = [
    "run_id", "model", "method", "steps", "gamma", "sc_cfg", "hook", "seed", "n",
    "gen_ppl", "ppl_ci_lo", "ppl_ci_hi", "entropy", "mauve", "rep4", "distinct2", "degen",
    "nfe_elf", "nfe_enc", "wall_s", "rouge1", "rouge2", "rougeL", "bleu",
    "git_hash_or_zip_md5", "timestamp",
]


def run_dir(run_id):
    d = os.path.join(RUNS, str(run_id))
    os.makedirs(d, exist_ok=True)
    return d


def is_done(run_id):
    return os.path.exists(os.path.join(RUNS, str(run_id), "summary.json"))


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_jsonable)
    os.replace(tmp, path)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def log_jsonl(path, records, mode="w"):
    tmp = path + ".tmp" if mode == "w" else path
    with open(tmp, mode) as f:
        for r in records:
            f.write(json.dumps(r, default=_jsonable) + "\n")
    if mode == "w":
        os.replace(tmp, path)


_CODE_ID = None


def code_id():
    """md5 of CODE/elf_code.zip if present, else git HEAD of the ELF repo, else 'unknown'."""
    global _CODE_ID
    if _CODE_ID is None:
        z = os.path.join(CODE, "elf_code.zip")
        if os.path.exists(z):
            with open(z, "rb") as f:
                _CODE_ID = "zip:" + hashlib.md5(f.read()).hexdigest()[:12]
        else:
            try:
                from . import ELF_REPO
                _CODE_ID = "git:" + subprocess.check_output(
                    ["git", "-C", ELF_REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
            except Exception:
                _CODE_ID = "unknown"
    return _CODE_ID


def append_registry(record):
    """Append one row; rotates an old registry whose header differs."""
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE) as f:
            header = f.readline().strip().split(",")
        if header != REGISTRY_COLUMNS:
            os.replace(REGISTRY_FILE, REGISTRY_FILE.replace(".csv", f"_old_{int(time.time())}.csv"))
    new = not os.path.exists(REGISTRY_FILE)
    row = {k: record.get(k, "") for k in REGISTRY_COLUMNS}
    row["git_hash_or_zip_md5"] = row["git_hash_or_zip_md5"] or code_id()
    row["timestamp"] = row["timestamp"] or time.strftime("%Y-%m-%dT%H:%M:%S")
    for k, v in row.items():
        if isinstance(v, float):
            row[k] = f"{v:.6g}"
    with open(REGISTRY_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def read_registry():
    import pandas as pd
    if not os.path.exists(REGISTRY_FILE):
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    return pd.read_csv(REGISTRY_FILE)
