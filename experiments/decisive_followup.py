#!/usr/bin/env python3
"""Decisive Follow-Up Suite for ACL 2027 Main-Track Submission.

When executed on a CUDA GPU (e.g., Google Colab A100), this script runs the
smallest high-leverage follow-up suite identified in the Statistical Audit:
  1. Complete the orthogonal 2x2 factorial sub-grid on ELF-M (64 steps) by
     adding the two off-diagonal cells:
       - SC-CFG = 1.0, gamma = 0.5 (seeds 0..4; 5 new runs)
       - SC-CFG = 4.0, gamma = 2.0 (seeds 0..4; 5 new runs)
  2. Extend the primary matched pairs from n=2 to n=5 independent seeds (seeds 0..4):
       - ELF-M Pair M1: (SC=1.0, gamma=2.0) vs (SC=4.0, gamma=0.5) [seeds 0,1 exist; seeds 2,3,4 pending = 6 new runs]
       - ELF-B Pair S1: (SC=1.0, gamma=2.0) vs (SC=4.0, gamma=1.5) [seeds 0,1 exist; seeds 2,3,4 pending = 6 new runs]
       - ELF-B Pair E3a: (SC=1.0, gamma=1.25) vs (SC=3.0, gamma=0.5) [seeds 0,1 exist; seeds 2,3,4 pending = 6 new runs]

Modes:
  - `--mode audit`   : Verifies existing vs. pending artifacts, uniqueness of run IDs,
                       checkpoint mapping, output locations, and estimated runtime.
  - `--mode dry-run` : Imports `run_single_config_and_ext_metrics` from
                       `experiments.non_identifiability_suite` and validates all 28
                       pending configurations and checkpoint specs without launching GPU runs.
  - `--mode gpu`     : Executes pending runs on a CUDA GPU without overwriting existing runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS_DIR = ROOT / "content" / "drive" / "MyDrive" / "elf_research" / "runs"

CHECKPOINT_MAP = {
    "ELF-B": {
        "hf_repo": "embedded-language-flows/ELF-B-owt-torch",
        "ckpt": "checkpoint_182985",
        "params_M": 105.0,
        "evaluator": "gpt2-large (774M)",
        "est_a100_min": 1.6,
    },
    "ELF-M": {
        "hf_repo": "embedded-language-flows/ELF-M-owt-torch",
        "ckpt": "checkpoint_76070",
        "params_M": 342.5,
        "evaluator": "gpt2-large (774M)",
        "est_a100_min": 7.3,
    },
}

FOLLOWUP_SPECS = [
    # 1. Complete 2x2 orthogonal sub-design on ELF-M (64 steps)
    {"model": "ELF-M", "steps": 64, "sc_cfg": 1.0, "gamma": 0.5, "seeds": [0, 1, 2, 3, 4], "role": "2x2_orthogonal_cell_low_sc_low_g"},
    {"model": "ELF-M", "steps": 64, "sc_cfg": 4.0, "gamma": 2.0, "seeds": [0, 1, 2, 3, 4], "role": "2x2_orthogonal_cell_high_sc_high_g"},
    # 2. 5-seed replication of ELF-M primary matched pair M1 (seeds 0,1 already on disk; 2,3,4 pending)
    {"model": "ELF-M", "steps": 64, "sc_cfg": 1.0, "gamma": 2.0, "seeds": [0, 1, 2, 3, 4], "role": "5seed_M1_low_sc"},
    {"model": "ELF-M", "steps": 64, "sc_cfg": 4.0, "gamma": 0.5, "seeds": [0, 1, 2, 3, 4], "role": "5seed_M1_high_sc"},
    # 3. 5-seed replication of ELF-B strict matched pair S1 and refined pair E3a (seeds 0,1 on disk; 2,3,4 pending)
    {"model": "ELF-B", "steps": 32, "sc_cfg": 1.0, "gamma": 2.0, "seeds": [0, 1, 2, 3, 4], "role": "5seed_S1_low_sc"},
    {"model": "ELF-B", "steps": 32, "sc_cfg": 4.0, "gamma": 1.5, "seeds": [0, 1, 2, 3, 4], "role": "5seed_S1_high_sc"},
    {"model": "ELF-B", "steps": 32, "sc_cfg": 1.0, "gamma": 1.25, "seeds": [0, 1, 2, 3, 4], "role": "5seed_E3a_low_sc"},
    {"model": "ELF-B", "steps": 32, "sc_cfg": 3.0, "gamma": 0.5, "seeds": [0, 1, 2, 3, 4], "role": "5seed_E3a_high_sc"},
]


def check_status():
    """Check which follow-up runs already exist on disk and which require GPU execution.
    Also enforces uniqueness of run_ids across the target matrix.
    """
    completed = []
    pending = []
    seen_run_ids = set()
    for spec in FOLLOWUP_SPECS:
        prefix = "p5_elfm_s64" if spec["model"] == "ELF-M" else f"p4_elfb_s{spec['steps']}"
        ckpt_info = CHECKPOINT_MAP[spec["model"]]
        for seed in spec["seeds"]:
            run_id = f"{prefix}_g{spec['gamma']}_sc{spec['sc_cfg']}_seed{seed}"
            if run_id in seen_run_ids:
                raise ValueError(f"Duplicate run_id in FOLLOWUP_SPECS: {run_id}")
            seen_run_ids.add(run_id)
            summary_path = RUNS_DIR / run_id / "summary.json"
            entry = {
                "run_id": run_id,
                "model": spec["model"],
                "steps": spec["steps"],
                "sc_cfg": spec["sc_cfg"],
                "gamma": spec["gamma"],
                "seed": seed,
                "n_samples": 1000,
                "role": spec["role"],
                "hf_repo": ckpt_info["hf_repo"],
                "ckpt": ckpt_info["ckpt"],
                "output_dir": str(RUNS_DIR / run_id),
                "est_a100_min": ckpt_info["est_a100_min"],
            }
            if summary_path.exists():
                completed.append(entry)
            else:
                pending.append(entry)
    return completed, pending


def main():
    parser = argparse.ArgumentParser(description="ACL 2027 Decisive Follow-Up Experiment Runner")
    parser.add_argument("--mode", choices=["audit", "dry-run", "gpu"], default="audit")
    args = parser.parse_args()

    # Always verify that the execution entrypoint imports cleanly
    from experiments.non_identifiability_suite import run_single_config_and_ext_metrics

    completed, pending = check_status()
    print("=== ACL 2027 Decisive Follow-Up Experiment Status ===")
    print(f"Target 5-seed matrix total runs: {len(completed) + len(pending)} (unique run IDs verified)")
    print(f"Existing completed runs on disk (preserved, skipped): {len(completed)}")
    print(f"Pending new GPU runs: {len(pending)}")
    for item in pending:
        print(
            f"  - {item['run_id']} ({item['model']} [{item['hf_repo']}::{item['ckpt']}], "
            f"steps={item['steps']}, gamma={item['gamma']}, sc={item['sc_cfg']}, "
            f"seed={item['seed']}, M={item['n_samples']}): ~{item['est_a100_min']:.1f} min on A100"
        )
    est_min = sum(x["est_a100_min"] for x in pending)
    print(f"Total estimated A100 time for {len(pending)} pending runs: {est_min:.1f} min ({est_min / 60.0:.2f} hours).")

    if args.mode == "audit":
        return

    if args.mode == "dry-run":
        validated = []
        for item in pending:
            res = run_single_config_and_ext_metrics(
                model_name=item["model"],
                steps=item["steps"],
                gamma=item["gamma"],
                sc_cfg=item["sc_cfg"],
                seed=item["seed"],
                n_samples=item["n_samples"],
                dry_run=True,
            )
            assert res["status"] == "dry_run_validated"
            assert not (Path(res["output_dir"]) / "summary.json").exists()
            validated.append(res["run_id"])
        print(f"[dry-run] Validated all {len(validated)} pending configurations and import paths with zero errors.")
        return

    # GPU execution path (requires torch + CUDA)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU mode requires a CUDA device (e.g., Google Colab A100). Run with --mode audit or --mode dry-run on CPU.")
    for item in pending:
        print(f"[GPU] Running {item['run_id']} ({item['role']})...")
        run_single_config_and_ext_metrics(
            model_name=item["model"],
            steps=item["steps"],
            gamma=item["gamma"],
            sc_cfg=item["sc_cfg"],
            seed=item["seed"],
            n_samples=item["n_samples"],
            dry_run=False,
        )


if __name__ == "__main__":
    main()
