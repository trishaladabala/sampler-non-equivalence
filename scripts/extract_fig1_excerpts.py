#!/usr/bin/env python3
"""
scripts/extract_fig1_excerpts.py
Extract deterministic, verbatim sample excerpts from Google Drive run directories
for Confirmatory Pair C3 (Seed 201) to populate Figure 1 Panel (c).
"""

import argparse
import hashlib
import json
from pathlib import Path


def extract_fig1_excerpts(runs_dir: Path, output_json: Path, fixed_sample_idx: int = 0) -> dict:
    # Confirmatory Pair C3: Seed 201
    # Arm A: SC = 1.0, gamma = 1.651
    # Arm B: SC = 4.0, gamma = 1.000
    dir_a_candidates = list(runs_dir.glob("p4_elfb_s32_*sc1.0*seed201*"))
    dir_b_candidates = list(runs_dir.glob("p4_elfb_s32_*sc4.0*seed201*"))

    if not dir_a_candidates or not dir_b_candidates:
        raise FileNotFoundError(
            f"Could not locate C3 Seed 201 run directories in {runs_dir}.\n"
            f"Found Arm A candidates: {dir_a_candidates}\n"
            f"Found Arm B candidates: {dir_b_candidates}"
        )

    run_dir_a = dir_a_candidates[0]
    run_dir_b = dir_b_candidates[0]

    samples_file_a = run_dir_a / "samples.jsonl"
    samples_file_b = run_dir_b / "samples.jsonl"

    if not samples_file_a.exists() or not samples_file_b.exists():
        raise FileNotFoundError(
            f"Missing samples.jsonl in run directories:\n"
            f"  Arm A: {samples_file_a} (exists={samples_file_a.exists()})\n"
            f"  Arm B: {samples_file_b} (exists={samples_file_b.exists()})"
        )

    def read_fixed_sample(jsonl_path: Path, idx: int) -> tuple[str, str]:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for cur_idx, line in enumerate(f):
                if cur_idx == idx:
                    record = json.loads(line)
                    text = record.get("text") or record.get("sample") or ""
                    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    return text.strip(), sha
        raise IndexError(f"File {jsonl_path} has fewer than {idx + 1} lines.")

    text_a, sha_a = read_fixed_sample(samples_file_a, fixed_sample_idx)
    text_b, sha_b = read_fixed_sample(samples_file_b, fixed_sample_idx)

    payload = {
        "pair": "C3",
        "seed": 201,
        "fixed_sample_index": fixed_sample_idx,
        "arm_a": {
            "config": "SC=1.0, gamma=1.651",
            "source_file": str(samples_file_a),
            "sha256": sha_a,
            "verbatim_text": text_a,
            "display_snippet": (text_a[:380] + "...") if len(text_a) > 380 else text_a,
        },
        "arm_b": {
            "config": "SC=4.0, gamma=1.000",
            "source_file": str(samples_file_b),
            "sha256": sha_b,
            "verbatim_text": text_b,
            "display_snippet": (text_b[:380] + "...") if len(text_b) > 380 else text_b,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Done] Deterministically extracted index {fixed_sample_idx} to {output_json}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=Path, default=Path("content/drive/MyDrive/elf_research/runs"))
    parser.add_argument("--output_json", type=Path, default=Path("paper_figures/fig1_excerpts.json"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    extract_fig1_excerpts(args.runs_dir, args.output_json, args.index)
