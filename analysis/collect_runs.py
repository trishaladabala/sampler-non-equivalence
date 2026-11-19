"""Collect every run summary into a single CSV (the single source of truth for tables).

Usage: python analysis/collect_runs.py
Reads:  content/drive/MyDrive/elf_research/runs/**/summary.json
Writes: analysis/results_db.csv
Only values present in summary.json files are emitted; nothing is imputed.
"""
import csv
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "content", "drive", "MyDrive", "elf_research", "runs")
OUT = os.path.join(ROOT, "analysis", "results_db.csv")

FIELDS = ["run_id", "source_path", "model", "method", "steps", "gamma", "sc_cfg", "seed", "n",
          "gen_ppl", "ppl_ci_lo", "ppl_ci_hi", "entropy", "mauve", "rep4", "distinct1",
          "distinct2", "distinct3", "degen", "mean_gpt2_len", "nfe_elf", "nfe_enc", "n_elf_fwd",
          "n_enc_fwd", "policy", "hook", "commit", "temperature", "top_p", "p10", "p50", "p90",
          "cont_ppl", "coherence", "bleu", "rouge1", "rouge2", "rougeL", "wall_s", "timestamp",
          "h_pool", "jsd_inter_sample", "self_bleu", "jcd_bits_per_byte",
          "within_distinct1", "within_distinct2", "within_distinct3", "within_distinct4"]


def clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return v


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS, "**", "summary.json"), recursive=True)):
        if "__MACOSX" in p or "p5_non_identifiability_artifacts" in p:
            continue
        try:
            d = json.load(open(p))
        except Exception as e:  # pragma: no cover
            print(f"[skip] {p}: {e}")
            continue
        ext_path = os.path.join(os.path.dirname(p), "extended_metrics.json")
        if os.path.isfile(ext_path):
            try:
                ext = json.load(open(ext_path))
                for k, v in ext.items():
                    if k not in d or d[k] in ("", None):
                        d[k] = v
            except Exception:
                pass
        r = {k: clean(d.get(k, "")) for k in FIELDS}
        r["source_path"] = os.path.relpath(p, ROOT)
        if not r["run_id"]:
            r["run_id"] = os.path.basename(os.path.dirname(p))
        rows.append(r)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
