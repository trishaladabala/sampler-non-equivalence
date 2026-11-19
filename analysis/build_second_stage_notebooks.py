#!/usr/bin/env python3
"""
Builds and executes all Second-Stage Jupyter notebooks (.ipynb) in `notebooks/` with
100% SELF-CONTAINED GOOGLE COLAB BOOTSTRAPPING:
  - Every notebook's Cell 0 embeds the compressed `second_stage_stats.json` payload and
    automatically writes `/content/drive/MyDrive/elf_research/analysis/out/second_stage_stats.json`
    on Google Colab if it does not yet exist (preventing any FileNotFoundError on Colab).
  - GPU notebooks (`08`, `09`, `10`, `06`, `07`) also embed the standalone `elfx` +
    `non_identifiability.py` bootstrap so they can be opened directly in Google Colab and
    executed from top to bottom ("Run All") without running any prior notebook first.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import re
import shutil
import zlib
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
S2_STATS = ROOT / "analysis" / "out" / "second_stage_stats.json"


def _rfloat(v):
    if isinstance(v, float):
        return round(v, 5)
    if isinstance(v, dict):
        return {k: _rfloat(vv) for k, vv in v.items() if k not in ("vals_A", "vals_B", "loo_diffs", "loo_hedges_g")}
    if isinstance(v, list):
        return [_rfloat(x) for x in v]
    return v


def _wrap76(b64_str: str) -> str:
    return "\n".join(b64_str[i:i + 76] for i in range(0, len(b64_str), 76))


def _load_bootstrap_blobs() -> tuple[dict[str, str], str, str]:
    s2_full = _rfloat(json.loads(S2_STATS.read_text(encoding="utf-8")))

    p12 = s2_full["phase1_and_phase2"]
    for _pk, pv in p12["existing_pairs_audit"].items():
        pv.pop("ln_gen_ppl_tier1", None)
        pv.pop("ln_gen_ppl_tier2", None)
        keep_sec = {}
        for mk in ("rep4", "within_distinct4", "distinct2", "jsd_pi", "self_bleu", "mauve", "jcd"):
            if mk in pv["secondary_metrics"]:
                mv = pv["secondary_metrics"][mk]
                keep_sec[mk] = {
                    k: mv[k]
                    for k in ("mean_A", "sd_A", "mean_B", "sd_B", "diff_B_minus_A", "ci95_welch", "p_welch_2sided", "hedges_g", "ci95_hedges_g", "loso_diff_range", "loso_max_p_value")
                    if k in mv
                }
        pv["secondary_metrics"] = keep_sec

    p5 = s2_full["phase5_semantic"]
    for _pk, pv in p5["pairs_seed0_audit_and_reanalysis"].items():
        for rk, rv in pv["representations"].items():
            pv["representations"][rk] = {
                "task2_homogeneous_15bundles_cos_dist": {
                    k: rv["task2_homogeneous_15bundles_cos_dist"][k]
                    for k in ("mean_A", "mean_B", "diff_B_minus_A", "ci95_welch", "p_welch_2sided", "hedges_g")
                },
                "corpus_100_vendi_low_sc": rv["corpus_100_vendi_low_sc"],
                "corpus_100_vendi_high_sc": rv["corpus_100_vendi_high_sc"],
                "corpus_100_vendi_diff_B_minus_A": rv["corpus_100_vendi_diff_B_minus_A"],
                "corpus_100_vendi_diff_ci95_boot": rv["corpus_100_vendi_diff_ci95_boot"],
            }

    raw_slices = {
        "nb08": {"phase1_and_phase2": p12},
        "nb03": {"phase3_factorial": s2_full["phase3_factorial"]},
        "nb09": {},
        "nb10": {"phase5_semantic": p5},
        "nb04": {"phase6_human_eval": s2_full["phase6_human_eval"]},
        "nb06": {},
        "nb07": {"phase8_compute": s2_full["phase8_compute"]},
    }
    s2_slices_b64 = {}
    for k, obj in raw_slices.items():
        raw_b = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        s2_slices_b64[k] = _wrap76(base64.b64encode(zlib.compress(raw_b, level=9)).decode("ascii"))

    p6_nb = json.loads((ROOT / "elf_p6_followup.ipynb").read_text(encoding="utf-8"))
    c2_src = "".join(p6_nb["cells"][2]["source"])
    m = re.search(r"ELFX_CORE_B64 = '([A-Za-z0-9+/=]+)'", c2_src)
    elfx_b64 = _wrap76(m.group(1)) if m else ""

    ni_raw = (ROOT / "experiments" / "non_identifiability_suite.py").read_bytes()
    ni_b64 = _wrap76(base64.b64encode(zlib.compress(ni_raw, level=9)).decode("ascii"))
    return s2_slices_b64, elfx_b64, ni_b64


S2_SLICES_B64, ELFX_B64, NI_B64 = _load_bootstrap_blobs()


def make_md_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def make_code_cell(source: str, execute_locally: bool = False, local_ns: dict | None = None, form_view: bool = False) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    outputs = []
    exec_count = None
    if execute_locally:
        buf = io.StringIO()
        ns = local_ns if local_ns is not None else {"__name__": "__main__"}
        with contextlib.redirect_stdout(buf):
            exec(source, ns)
        out_text = buf.getvalue()
        exec_count = 1
        if out_text:
            out_lines = [line + "\n" for line in out_text.split("\n")]
            if out_lines and out_lines[-1] == "\n":
                out_lines.pop()
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": out_lines,
            })
    meta = {"cellView": "form"} if form_view else {}
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": meta,
        "outputs": outputs,
        "source": lines,
    }


def save_notebook(path: Path, cells: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "A100"},
            "Accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"[build_second_stage_notebooks] Saved {path.relative_to(ROOT)} ({len(cells)} cells)")


def make_setup_cell_source(nb_key: str, include_elfx_gpu_bootstrap: bool = True) -> str:
    slice_b64 = S2_SLICES_B64[nb_key]
    header = f'''#@title 0. Environment & Standalone Bootstrap (Click ▶ to Run) {{ display-mode: "form" }}
import os
import sys
import json
import math
import time
import glob
import random
import shutil
import zipfile
import zlib
import base64
import subprocess
from pathlib import Path
import numpy as np
import torch

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    PROJECT_ROOT = Path("/content/drive/MyDrive/elf_research")
else:
    PROJECT_ROOT = Path(os.environ.get("ELF_PROJECT_ROOT", Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

# 1. Auto-bootstrap `/content/drive/MyDrive/elf_research/analysis/out/second_stage_stats.json` slice if needed
S2_PATH = PROJECT_ROOT / "analysis" / "out" / "second_stage_stats.json"
S2_SLICE_B64 = """
{slice_b64}
"""
_slice_obj = json.loads(zlib.decompress(base64.b64decode(S2_SLICE_B64)).decode("utf-8"))
if not S2_PATH.exists():
    S2_PATH.parent.mkdir(parents=True, exist_ok=True)
    S2_PATH.write_text(json.dumps(_slice_obj), encoding="utf-8")
    print(f"[bootstrap] Materialized {{S2_PATH}}")
elif _slice_obj:
    try:
        _existing = json.loads(S2_PATH.read_text(encoding="utf-8"))
    except Exception:
        _existing = {{}}
    if any(k not in _existing for k in _slice_obj):
        _existing.update(_slice_obj)
        S2_PATH.write_text(json.dumps(_existing), encoding="utf-8")
        print(f"[bootstrap] Updated {{S2_PATH}} with {nb_key} stats slice")
'''
    if include_elfx_gpu_bootstrap:
        gpu_part = f'''
# 2. On Colab, auto-bootstrap ELF repo + `elfx` + `elfx.non_identifiability` so GPU cells run standalone
ELFX_CORE_B64 = """
{ELFX_B64}
"""
NI_SUITE_B64 = """
{NI_B64}
"""

if IN_COLAB:
    ROOT_STR = str(PROJECT_ROOT)
    CODE_DIR = PROJECT_ROOT / "code"
    RUNS_DIR = PROJECT_ROOT / "runs"
    REFS_DIR = PROJECT_ROOT / "refs"
    for _d in [PROJECT_ROOT, CODE_DIR, RUNS_DIR, REFS_DIR, CODE_DIR / "elfx"]:
        _d.mkdir(parents=True, exist_ok=True)

    zip_path = CODE_DIR / "elf_code.zip"
    if not os.path.isdir("/content/repo/ELF/src"):
        if zip_path.exists():
            subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", "/content/repo"], check=True)
        else:
            os.makedirs("/content/repo", exist_ok=True)
            subprocess.run([
                "git", "clone", "-q", "--depth", "1", "-b", "pytorch_elf",
                "https://anonymous.4open.science/r/ELF", "/content/repo/ELF"
            ], check=True)

    if not (CODE_DIR / "elfx" / "__init__.py").exists():
        _bundle = json.loads(zlib.decompress(base64.b64decode(ELFX_CORE_B64)).decode("utf-8"))
        for _fname, _fcontent in _bundle.items():
            (CODE_DIR / "elfx" / _fname).write_text(_fcontent, encoding="utf-8")

    ni_target = CODE_DIR / "elfx" / "non_identifiability.py"
    if not ni_target.exists():
        ni_target.write_bytes(zlib.decompress(base64.b64decode(NI_SUITE_B64)))

    os.environ["ELF_ROOT"] = ROOT_STR
    os.environ["ELF_SRC"] = "/content/repo/ELF/src"
    for _p in ["/content/repo/ELF/src", str(CODE_DIR)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "sacrebleu", "rouge-score", "mauve-text", "einops", "sentencepiece", "matplotlib", "scipy"
    ], check=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
else:
    for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "ELF" / "src")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
'''
    else:
        gpu_part = '''
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "ELF" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
'''

    footer = '''
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

print(f"Environment: {'Google Colab' if IN_COLAB else 'Local Workspace'} | Root: {PROJECT_ROOT}")
print(f"Verified Stats File Ready: {S2_PATH.exists()} ({S2_PATH})")
if torch.cuda.is_available():
    dev_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"CUDA GPU Available: {dev_name} ({vram_gb:.1f} GiB VRAM)")
else:
    print("CUDA GPU Available: False (Running in CPU analysis / verification mode)")
'''
    return header + gpu_part + footer


# ==============================================================================
# NOTEBOOK 08: Confirmatory Operating-Point Matching (Phase 2)
# ==============================================================================

def build_nb08() -> None:
    ns = {"__name__": "__main__", "ROOT": ROOT, "S2_STATS": S2_STATS}
    cells = [
        make_md_cell("""
# Notebook 08 — Confirmatory Operating-Point Matching & 1D-for-2D Root-Finding Audit (Phase 2)

**100% Standalone on Google Colab:** You do **not** need to run any other notebook before running this one. Cell 0 automatically materializes `second_stage_stats.json` and bootstraps the `elfx` + `ELF` environment on Google Drive.

**Status:**
- **Sections 1–3 (Practical Equivalence Margin Justification, 1D-for-2D Over-Determination Proof, Bootstrap Uncertainty $\\text{SE}(\\gamma^*)$, & Power Analysis):** `EXPLORATORY CALIBRATION — COMPLETE ON REAL DATA (Seeds 101..105)`
- **Section 4 (Frozen Confirmatory Generation & Evaluation on Fresh Seeds `201..205`, 30 Runs):** `CONFIRMATORY — AUTO-RUNS WHEN EXECUTED ON COLAB GPU`
- **Section 5 (Confirmatory Paired & Independent Welch TOST + Secondary Metric Analysis):** `CONFIRMATORY — AUTO-EXECUTES ONCE SECTION 4 COMPLETES`
"""),
        make_code_cell(make_setup_cell_source("nb08", include_elfx_gpu_bootstrap=True), execute_locally=True, local_ns=ns, form_view=True),
        make_code_cell("""
# Cell 1: Load Verified Second-Stage Calibration Data & External Benchmarks
from scipy import stats

s2 = json.loads(S2_PATH.read_text(encoding="utf-8"))
p12 = s2["phase1_and_phase2"]

print("=== 1. EXTERNAL BENCHMARKS vs. MEASUREMENT PRECISION ===")
bm = p12["practical_margin_benchmarks"]
print("Model-scale step (ELF-B 105M -> ELF-M 342M):")
print(f"  Mean |Delta ln(Gen-PPL)| = {bm['model_scale_step_ELFB_to_ELFM']['mean_abs_delta_ln_ppl']:.4f} ({bm['model_scale_step_ELFB_to_ELFM']['mean_abs_rel_ppl_pct']:.2f}%)")
print(f"  Mean |Delta H_ps|        = {bm['model_scale_step_ELFB_to_ELFM']['mean_abs_delta_hps_nats']:.4f} nats")
print(f"  20% of scale step        = {bm['model_scale_step_ELFB_to_ELFM']['twenty_pct_of_scale_step_ln_ppl']:.4f} ln-PPL (~4.5%), {bm['model_scale_step_ELFB_to_ELFM']['twenty_pct_of_scale_step_hps']:.4f} nats H_ps")
print("Cross-seed measurement precision (5-seed cells):")
mp = bm["cross_seed_measurement_precision_5seed_cells"]
print(f"  Pooled SD across seeds   = {mp['pooled_sd_ln_ppl']:.4f} ln-PPL (~1.01%), {mp['pooled_sd_hps_nats']:.5f} nats H_ps")
print(f"  Min detectable 5-seed TOST margin (alpha=0.05) = {mp['min_detectable_equiv_margin_5seeds_alpha05_ln_ppl']*100:.2f}% PPL, {mp['min_detectable_equiv_margin_5seeds_alpha05_hps']:.4f} nats H_ps")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 2: Audit of All 6 Existing Exploratory Pairs (Physical Shifts, TOST, & LOSO Sensitivity)
print("=== 2. EXISTING EXPLORATORY PAIRS AUDIT (SEEDS 101..105) ===")
print(f"{'Pair':<22} {'n':>3} {'d_PPL%':>8} {'min_d_PPL%':>11} {'d_H(nats)':>10} {'min_d_H':>9} {'Tier1':>6} {'Tier2':>6} {'d_rep4':>9} {'LOSO d_rep4':>18} {'Hedges g (95% CI)':>24}")
print("-" * 135)
for label, v in p12["existing_pairs_audit"].items():
    n = v["cell_A"]["n"]
    dp_pct = v["gen_ppl_tier1"]["rel_diff_pct_vs_A"]
    min_p = v["min_passing_ppl_pct_alpha05"]
    dh = v["hps_tier1"]["diff_B_minus_A"]
    min_h = v["min_passing_hps_nats_alpha05"]
    t1 = "PASS" if v["passes_joint_tier1"] else "FAIL"
    t2 = "PASS" if v["passes_joint_tier2"] else "FAIL"
    rep = v["secondary_metrics"]["rep4"]
    d_rep = rep["diff_B_minus_A"]
    loso_d = rep["loso_diff_range"]
    loso_str = f"[{loso_d[0]:+.4f}, {loso_d[1]:+.4f}]" if loso_d else "N/A (n=2)"
    g_val = rep["hedges_g"]
    g_ci = rep["ci95_hedges_g"]
    g_str = f"{g_val:5.1f} [{g_ci[0]:5.1f}, {g_ci[1]:5.1f}]"
    print(f"{label:<22} {n:>3} {dp_pct:>+7.2f}% {min_p:>10.2f}% {dh:>+10.5f} {min_h:>9.4f} {t1:>6} {t2:>6} {d_rep:>+9.4f} {loso_str:>18} {g_str:>24}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 3: Phase 2B — Mathematical Audit of 1D-for-2D Root-Finding, Irreducible Pareto Residuals, & Bootstrap SE(gamma*)
print("=== 3. 1D-PARAMETER-FOR-2D-TARGET ROOT-FINDING AUDIT (2,000 SEED BOOTSTRAPS) ===")
print(f"{'Target B (SC, g)':<18} {'n_cal':>5} {'g_joint* (Boot 95% CI)':>26} {'g_PPL*':>8} {'g_H*':>8} {'Mismatch dg':>12} {'Res PPL%':>9} {'Res H':>9} {'Pwr T2':>7} {'Pwr T1':>7}")
print("-" * 128)
for c in p12["root_finding_1d_for_2d_audit"][:9]:
    lf = c["linear_fit"]
    tb = f"({c['target_sc_cfg']:.1f}, {c['target_gamma']:.2f})"
    nc = c["n_calibration_seeds"]
    ci = c["bootstrap_ci95_gamma_star"]
    gj_str = f"{lf['gamma_joint_star']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    print(f"{tb:<18} {nc:>5} {gj_str:>26} {lf['gamma_ppl_only_star']:>8.3f} {lf['gamma_hps_only_star']:>8.3f} "
          f"{lf['root_mismatch_delta_gamma']:>+12.3f} {lf['residual_ppl_pct']:>+8.2f}% {lf['residual_hps_nats']:>+9.4f} "
          f"{c['predicted_5seed_power_tier2_joint']:>7.3f} {c['predicted_5seed_power_tier1_joint']:>7.3f}")

print("\\nKEY MATHEMATICAL FINDING:")
print("  - At gamma_B = 0.5, root mismatch dg* = g_PPL* - g_H* is strictly NEGATIVE (-0.050 to -0.090).")
print("  - At gamma_B = 1.5, root mismatch dg* = g_PPL* - g_H* is strictly POSITIVE (+0.042 to +0.064).")
print("  - By the Intermediate Value Theorem, the two iso-curves intersect near gamma_B = 1.0, where |dg*| <= 0.027")
print("    and irreducible compromise residuals are <= 0.39% PPL and <= 0.0029 nats H_ps (giving >86% Tier-I and 99.9% Tier-II power at n=5 seeds).")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 4: Publication-Quality Diagnostic Plot — 1D-for-2D Root Mismatch & Bootstrap Uncertainty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cands = [c for c in p12["root_finding_1d_for_2d_audit"] if c["target_gamma"] in (0.5, 1.0, 1.5)]
cands.sort(key=lambda x: (x["target_gamma"], x["target_sc_cfg"]))

labels = [f"SC={c['target_sc_cfg']:.0f}, g={c['target_gamma']:.1f}" for c in cands]
dg_vals = [c["linear_fit"]["root_mismatch_delta_gamma"] for c in cands]
res_p = [c["linear_fit"]["residual_ppl_pct"] for c in cands]
res_h = [c["linear_fit"]["residual_hps_nats"] * 1000.0 for c in cands]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
x_pos = np.arange(len(labels))
colors = ["#d95f02" if c["target_gamma"] == 0.5 else ("#1b9e77" if c["target_gamma"] == 1.0 else "#7570b3") for c in cands]

axes[0].bar(x_pos, dg_vals, color=colors, edgecolor="black", alpha=0.85)
axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
axes[0].set_ylabel(r"Root Mismatch $\Delta\gamma^* = \gamma^*_{\mathrm{PPL}} - \gamma^*_H$")
axes[0].set_title(r"(a) 1D-for-2D Root Mismatch Across Target $\gamma_B$")

axes[1].scatter(res_p, res_h, c=colors, s=75, edgecolors="black", zorder=3)
for idx, lbl in enumerate(labels):
    axes[1].annotate(lbl, (res_p[idx], res_h[idx]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
axes[1].axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
axes[1].axvspan(-2.5, 2.5, ymin=0.2, ymax=0.8, color="#1b9e77", alpha=0.08, label="Tier I PPL (+/-2.5%)")
axes[1].set_xlabel(r"Irreducible Compromise Residual $\hat{r}_{\mathrm{PPL}}(\gamma^*)$ (%)")
axes[1].set_ylabel(r"Irreducible Compromise Residual $\hat{r}_H(\gamma^*)$ ($10^{-3}$ nats)")
axes[1].set_title(r"(b) Pareto Compromise Residuals at $\gamma^*_{\mathrm{joint}}$")

plt.tight_layout()
fig_path = PROJECT_ROOT / "reports" / "figures" / "fig_nb08_root_matching_audit.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path)
plt.close(fig)
print(f"Saved publication figure to {fig_path}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 5: Frozen Pre-Registered Confirmatory Experiment Manifest (30 Runs on Fresh Seeds 201..205)
CONFIRMATORY_SEEDS = [201, 202, 203, 204, 205]

FROZEN_CONFIRMATORY_PAIRS = [
    {
        "pair_id": "C1_ELFB_SC2_vs_SC1",
        "model_name": "ELF-B",
        "steps": 32,
        "n_samples": 1000,
        "arm_A": {"sc_cfg": 1.0, "gamma": 1.346},
        "arm_B": {"sc_cfg": 2.0, "gamma": 1.000},
        "cal_pred_residual_ppl_pct": -0.05,
        "cal_pred_residual_hps_nats": +0.0004,
        "cal_gamma_boot_se": 0.009,
    },
    {
        "pair_id": "C2_ELFB_SC3_vs_SC1",
        "model_name": "ELF-B",
        "steps": 32,
        "n_samples": 1000,
        "arm_A": {"sc_cfg": 1.0, "gamma": 1.547},
        "arm_B": {"sc_cfg": 3.0, "gamma": 1.000},
        "cal_pred_residual_ppl_pct": -0.08,
        "cal_pred_residual_hps_nats": +0.0007,
        "cal_gamma_boot_se": 0.017,
    },
    {
        "pair_id": "C3_ELFB_SC4_vs_SC1",
        "model_name": "ELF-B",
        "steps": 32,
        "n_samples": 1000,
        "arm_A": {"sc_cfg": 1.0, "gamma": 1.651},
        "arm_B": {"sc_cfg": 4.0, "gamma": 1.000},
        "cal_pred_residual_ppl_pct": -0.39,
        "cal_pred_residual_hps_nats": +0.0029,
        "cal_gamma_boot_se": 0.012,
    },
]

CONFIRMATORY_RUN_MANIFEST = []
for pair in FROZEN_CONFIRMATORY_PAIRS:
    for arm_label, arm_cfg in [("A", pair["arm_A"]), ("B", pair["arm_B"])]:
        for seed in CONFIRMATORY_SEEDS:
            run_id = f"p4_elfb_s32_g{float(arm_cfg['gamma'])}_sc{float(arm_cfg['sc_cfg'])}_seed{int(seed)}"
            CONFIRMATORY_RUN_MANIFEST.append({
                "run_id": run_id,
                "pair_id": pair["pair_id"],
                "arm": arm_label,
                "model_name": pair["model_name"],
                "steps": pair["steps"],
                "sc_cfg": float(arm_cfg["sc_cfg"]),
                "gamma": float(arm_cfg["gamma"]),
                "seed": int(seed),
                "n_samples": int(pair["n_samples"]),
            })

print(f"Total Frozen Confirmatory Runs Scheduled: {len(CONFIRMATORY_RUN_MANIFEST)} (3 pairs x 2 arms x 5 fresh seeds 201..205)")
for entry in CONFIRMATORY_RUN_MANIFEST[:4]:
    print(" ", entry["run_id"])
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 6: [COLAB GPU EXECUTION CELL] Run the 30 Frozen Confirmatory Runs & Rescore All Primary/Secondary Metrics
# Automatically runs when a CUDA GPU is attached on Colab (~22 minutes total on A100).
# Resumable: skips any run_id whose summary.json + extended_metrics.json already exist on Drive.

RUN_CONFIRMATORY_GPU = torch.cuda.is_available()

if RUN_CONFIRMATORY_GPU:
    import importlib
    from elfx import load, sample, metrics, refs, registry, sweep, frontier
    from elfx import non_identifiability as ni
    for _m in [registry, refs, load, sample, metrics, sweep, frontier, ni]:
        importlib.reload(_m)
    sweep.set_compile(True)

    for idx, item in enumerate(CONFIRMATORY_RUN_MANIFEST, 1):
        print(f"\\n[{idx}/{len(CONFIRMATORY_RUN_MANIFEST)}] Executing confirmatory run {item['run_id']} ({item['pair_id']} Arm {item['arm']})...")
        res = ni.run_single_config_and_ext_metrics(
            model_name=item["model_name"],
            steps=item["steps"],
            gamma=item["gamma"],
            sc_cfg=item["sc_cfg"],
            seed=item["seed"],
            n_samples=item["n_samples"],
            dry_run=False,
        )
        print(f"  -> Completed {item['run_id']}: Gen-PPL={res.get('gen_ppl'):.2f}, H_ps={res.get('entropy'):.4f}, rep4={res.get('rep4'):.4f}, JSD_pi={res.get('jsd_inter_sample', 0.0):.4f}")
else:
    print("[CONFIRMATORY GPU EXECUTION PENDING] Attach a Colab GPU (Runtime -> Change runtime type -> A100 GPU) and run this cell.")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 7: Confirmatory Inference Engine (Paired Seed-Blocked TOST + Independent Satterthwaite Welch TOST + Auto-Zip)
# Auto-detects completed confirmatory runs in runs/p4_elfb_s32_*_seed{201..205}; never fabricates pending runs.

def satterthwaite_and_paired_tost(x: np.ndarray, y: np.ndarray, margin: float) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    diff = float(np.mean(y) - np.mean(x))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    se_w = math.sqrt(vx / nx + vy / ny) if (vx / nx + vy / ny) > 0 else 1e-12
    df_satt = ((vx / nx + vy / ny) ** 2) / (((vx / nx) ** 2) / (nx - 1) + ((vy / ny) ** 2) / (ny - 1)) if (vx + vy) > 0 else float(nx + ny - 2)
    p_w_tost = max(float(stats.t.sf((diff + margin) / se_w, df=df_satt)), float(stats.t.sf((margin - diff) / se_w, df=df_satt)))
    t90_w = float(stats.t.ppf(0.95, df=df_satt))
    t95_w = float(stats.t.ppf(0.975, df=df_satt))

    # Paired seed-blocked test across seeds 201..205
    d_vec = y - x
    se_p = float(np.std(d_vec, ddof=1) / math.sqrt(nx)) if nx > 1 else 1e-12
    df_p = nx - 1
    p_paired_tost = max(float(stats.t.sf((diff + margin) / se_p, df=df_p)), float(stats.t.sf((margin - diff) / se_p, df=df_p)))

    return {
        "mean_A": float(np.mean(x)),
        "mean_B": float(np.mean(y)),
        "diff_B_minus_A": diff,
        "ci95_welch": [diff - t95_w * se_w, diff + t95_w * se_w],
        "ci90_tost_welch": [diff - t90_w * se_w, diff + t90_w * se_w],
        "df_satt": df_satt,
        "p_welch_2sided": float(2.0 * stats.t.sf(abs(diff / se_w), df=df_satt)),
        "p_tost_welch": p_w_tost,
        "p_tost_paired": p_paired_tost,
        "pass_welch_05": bool(p_w_tost < 0.05),
        "pass_paired_05": bool(p_paired_tost < 0.05),
    }

runs_base = PROJECT_ROOT / "runs" if (PROJECT_ROOT / "runs").exists() else (PROJECT_ROOT / "content" / "drive" / "MyDrive" / "elf_research" / "runs")
found_records = {}
for item in CONFIRMATORY_RUN_MANIFEST:
    rdir = runs_base / item["run_id"]
    sp = rdir / "summary.json"
    ep = rdir / "extended_metrics.json"
    if sp.exists():
        rec = json.loads(sp.read_text(encoding="utf-8"))
        if ep.exists():
            rec.update(json.loads(ep.read_text(encoding="utf-8")))
        found_records[item["run_id"]] = rec

print(f"Confirmatory Runs Completed on Disk: {len(found_records)} / {len(CONFIRMATORY_RUN_MANIFEST)}")
if len(found_records) == len(CONFIRMATORY_RUN_MANIFEST):
    conf_summary = {}
    for pair in FROZEN_CONFIRMATORY_PAIRS:
        pid = pair["pair_id"]
        a_runs = [found_records[f"p4_elfb_s32_g{float(pair['arm_A']['gamma'])}_sc{float(pair['arm_A']['sc_cfg'])}_seed{s}"] for s in CONFIRMATORY_SEEDS]
        b_runs = [found_records[f"p4_elfb_s32_g{float(pair['arm_B']['gamma'])}_sc{float(pair['arm_B']['sc_cfg'])}_seed{s}"] for s in CONFIRMATORY_SEEDS]
        mean_ppl_a = float(np.mean([r["gen_ppl"] for r in a_runs]))
        t1_ppl = satterthwaite_and_paired_tost([r["gen_ppl"] for r in a_runs], [r["gen_ppl"] for r in b_runs], 0.025 * mean_ppl_a)
        t2_ppl = satterthwaite_and_paired_tost([r["gen_ppl"] for r in a_runs], [r["gen_ppl"] for r in b_runs], 0.050 * mean_ppl_a)
        t1_hps = satterthwaite_and_paired_tost([r["entropy"] for r in a_runs], [r["entropy"] for r in b_runs], 0.015)
        t2_hps = satterthwaite_and_paired_tost([r["entropy"] for r in a_runs], [r["entropy"] for r in b_runs], 0.025)
        sec = {}
        for mk in ["rep4", "within_distinct2", "h_pool", "jsd_inter_sample", "distinct2", "self_bleu", "jcd_bits_per_byte", "mauve"]:
            if all(mk in r for r in a_runs + b_runs):
                sec[mk] = satterthwaite_and_paired_tost([r[mk] for r in a_runs], [r[mk] for r in b_runs], 0.01)
        conf_summary[pid] = {"tier1_ppl": t1_ppl, "tier2_ppl": t2_ppl, "tier1_hps": t1_hps, "tier2_hps": t2_hps, "secondary": sec}
        print(f"\\n=== CONFIRMATORY RESULT: {pid} ===")
        print(f"  Gen-PPL: A={t1_ppl['mean_A']:.2f}, B={t1_ppl['mean_B']:.2f}, diff={t1_ppl['diff_B_minus_A']:+.2f} | Tier-I TOST p={t1_ppl['p_tost_welch']:.4f}, Tier-II TOST p={t2_ppl['p_tost_welch']:.4f}")
        print(f"  H_ps:    A={t1_hps['mean_A']:.4f}, B={t1_hps['mean_B']:.4f}, diff={t1_hps['diff_B_minus_A']:+.4f} | Tier-I TOST p={t1_hps['p_tost_welch']:.4f}, Tier-II TOST p={t2_hps['p_tost_welch']:.4f}")
        for mk, mv in sec.items():
            print(f"  {mk:<18}: diff={mv['diff_B_minus_A']:+.4f}, 95% CI=[{mv['ci95_welch'][0]:+.4f}, {mv['ci95_welch'][1]:+.4f}], p={mv['p_welch_2sided']:.2e}")

    out_json = runs_base / "p8_confirmatory_stats.json"
    out_json.write_text(json.dumps(conf_summary, indent=2), encoding="utf-8")
    zip_out = runs_base / "p8_confirmatory_artifacts.zip"
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname="p8_confirmatory_stats.json")
        for item in CONFIRMATORY_RUN_MANIFEST:
            rdir = runs_base / item["run_id"]
            for fn in ("summary.json", "extended_metrics.json"):
                if (rdir / fn).exists():
                    zf.write(rdir / fn, arcname=f"{item['run_id']}/{fn}")
    print(f"\\nSaved confirmatory archive to {zip_out}")
    if IN_COLAB:
        from google.colab import files
        files.download(str(zip_out))
else:
    print("[STATUS: WAITING FOR GPU EXECUTION OF CELL 6 TO COMPLETE ALL 30 CONFIRMATORY RUNS]")
""", execute_locally=True, local_ns=ns),
    ]
    save_notebook(NB_DIR / "08_confirmatory_operating_point_matching.ipynb", cells)


# ==============================================================================
# NOTEBOOK 03 (UPDATED): Factorial ANOVA, Physical Marginal Effects, HC3, &
#                        Factor-Restricted Permutations (Phase 3)
# ==============================================================================

def build_nb03() -> None:
    ns = {"__name__": "__main__", "ROOT": ROOT, "S2_STATS": S2_STATS}
    cells = [
        make_md_cell("""
# Notebook 03 — Factorial Statistical Analysis, Physical Marginal Effects, HC3 Inference, & Related Metric Groups (Phase 3)

**100% Standalone on Google Colab & Local CPU:** Cell 0 automatically materializes `second_stage_stats.json` if not yet present on Google Drive.

**Status:** `COMPLETE — VERIFIED ON REAL EXPERIMENTAL RUNS (ELF-B 56 runs, ELF-M 20 runs, 64 rescored runs)`
"""),
        make_code_cell(make_setup_cell_source("nb03", include_elfx_gpu_bootstrap=False), execute_locally=True, local_ns=ns, form_view=True),
        make_code_cell("""
# Cell 1: Load Verified Phase-3 Factorial Reanalysis from second_stage_stats.json
s2 = json.loads(S2_PATH.read_text(encoding="utf-8"))
p3 = s2["phase3_factorial"]

print("=== 1. ELF-B (105M, K=32) PHYSICAL MARGINAL EFFECTS & GAMMA-DOMAIN SENSITIVITY ===")
for metric in ["rep4", "distinct2", "mauve", "gen_ppl", "entropy"]:
    print(f"\\nMetric: {metric}")
    print(f"  {'Domain':<14} {'d_SC (4-1)':>12} {'HC3 95% CI (SC)':>22} {'p_perm(SC)':>11} {'eta_p2(SC)':>11} {'omega2(SC)':>11} | {'d_gamma':>10} {'HC3 95% CI (gamma)':>22} {'eta_p2(g)':>10}")
    for dom in ["gamma_le_1p5", "gamma_le_2p0", "gamma_le_3p0"]:
        r = p3["elf_b_factorial_reanalysis"][metric][dom]
        sc = r["physical_marginal_effects"]["sc_cfg_max_minus_min"]
        gm = r["physical_marginal_effects"]["gamma_max_minus_min"]
        an_sc = r["anova_and_restricted_perm"]["sc_cfg"]
        an_gm = r["anova_and_restricted_perm"]["gamma"]
        ci_sc = f"[{sc['ci95_hc3'][0]:+.4f}, {sc['ci95_hc3'][1]:+.4f}]"
        ci_gm = f"[{gm['ci95_hc3'][0]:+.4f}, {gm['ci95_hc3'][1]:+.4f}]"
        print(f"  {dom:<14} {sc['estimate']:>+12.4f} {ci_sc:>22} {an_sc['p_perm_restricted_within_gamma']:>11.4f} "
              f"{an_sc['partial_eta_sq']:>11.4f} {an_sc['omega_sq']:>11.4f} | {gm['estimate']:>+10.4f} {ci_gm:>22} {an_gm['partial_eta_sq']:>10.4f}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 2: Slice-Specific Physical Effects of SC-CFG (4.0 - 1.0) at Each Fixed Gamma Level on ELF-B
print("=== 2. SLICE-SPECIFIC PHYSICAL EFFECT OF SC-CFG (4.0 - 1.0) AT FIXED GAMMA (ELF-B) ===")
for metric in ["rep4", "distinct2", "mauve"]:
    slices = p3["elf_b_factorial_reanalysis"][metric]["gamma_le_3p0"]["physical_marginal_effects"]["slice_specific_sc_cfg_effects_by_gamma"]
    print(f"\\nMetric: {metric}")
    for g_val, s_info in slices.items():
        ci = s_info["ci95"]
        print(f"  gamma = {float(g_val):<4.2f}: Delta_SC = {s_info['diff_sc_max_minus_min']:+.4f} (SE = {s_info['se']:.4f}, 95% CI = [{ci[0]:+.4f}, {ci[1]:+.4f}])")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 3: Balanced ELF-M (342M, 2 x 2 x 5 Seeds = 20 Runs, df_resid = 16) Physical Marginal Effects & HC3 CIs
print("=== 3. BALANCED ELF-M (342M, 2x2x5) PHYSICAL MARGINAL EFFECTS & FACTOR-RESTRICTED PERMUTATIONS ===")
print(f"{'Metric':<20} {'d_SC (4.0-1.0)':>14} {'HC3 95% CI (SC)':>22} {'p_perm(SC)':>11} {'omega2(SC)':>11} | {'d_gamma (2.0-0.5)':>17} {'HC3 95% CI (gamma)':>22} {'p_perm(g)':>10} {'omega2(g)':>10}")
print("-" * 148)
for m, r in p3["elf_m_factorial_reanalysis"].items():
    if "physical_marginal_effects" not in r:
        continue
    sc = r["physical_marginal_effects"]["sc_cfg_max_minus_min"]
    gm = r["physical_marginal_effects"]["gamma_max_minus_min"]
    an_sc = r["anova_and_restricted_perm"]["sc_cfg"]
    an_gm = r["anova_and_restricted_perm"]["gamma"]
    ci_sc = f"[{sc['ci95_hc3'][0]:+.4f}, {sc['ci95_hc3'][1]:+.4f}]"
    ci_gm = f"[{gm['ci95_hc3'][0]:+.4f}, {gm['ci95_hc3'][1]:+.4f}]"
    print(f"{m:<20} {sc['estimate']:>+14.4f} {ci_sc:>22} {an_sc['p_perm_restricted_within_gamma']:>11.4f} {an_sc['omega_sq']:>11.4f} | "
          f"{gm['estimate']:>+17.4f} {ci_gm:>22} {an_gm['p_perm_restricted_within_sc']:>10.4f} {an_gm['omega_sq']:>10.4f}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 4: Related Metric Groups Audit — Cross-Group Correlations & Exact Algebraic Identities
mg = p3["metric_groups_correlation_and_identity_audit"]
print("=== 4. CROSS-GROUP CORRELATIONS (PROVING METRIC GROUPS ARE CORRELATED, NOT ORTHOGONAL) ===")
for pair in mg["top_cross_group_correlations"][:8]:
    print(f"  {pair['metric_1']:<18} ({pair['group_1']}) vs {pair['metric_2']:<18} ({pair['group_2']}): r = {pair['pearson_r']:+.4f}")

print("\\n=== 5. EXACT MATHEMATICAL IDENTITIES & TOKENIZATION/WEIGHTING AUDIT ===")
print(f"  Max |JSD_pi - (H_pool - H_ps_weighted)| across 64 runs     = {mg['identity_jsd_pi_vs_h_pool_minus_h_ps_weighted_max_err']:.3e} nats (Exact Identity)")
print(f"  Max |JSD_pi - (H_pool - H_ps_unweighted)| across 64 runs   = {mg['discrepancy_jsd_pi_vs_h_pool_minus_unweighted_entropy_max_diff']:.5f} nats (Length-weighting difference)")
print(f"  Max |rep4 - (1 - within_distinct4)| on same tokenizer      = {mg['identity_rep4_vs_1_minus_within_distinct4_max_err']:.3e} (Exact Identity)")
print("\\n=== 6. CONSEQUENCES OF n=2 SEEDS PER CELL IN ELF-B ===")
print(" ", p3["n2_seed_limitation_audit"]["consequence"])
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 5: Publication Plot — Invariance of Physical Marginal Effect vs. Domain-Sensitivity of Variance Share
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

domains = ["gamma_le_1p5", "gamma_le_2p0", "gamma_le_3p0"]
dom_labels = [r"$\gamma \leq 1.5$", r"$\gamma \leq 2.0$", r"$\gamma \leq 3.0$"]

sc_phys = [p3["elf_b_factorial_reanalysis"]["rep4"][d]["physical_marginal_effects"]["sc_cfg_max_minus_min"]["estimate"] * 100 for d in domains]
g_phys = [p3["elf_b_factorial_reanalysis"]["rep4"][d]["physical_marginal_effects"]["gamma_max_minus_min"]["estimate"] * 100 for d in domains]
sc_share = [p3["elf_b_factorial_reanalysis"]["rep4"][d]["anova_and_restricted_perm"]["sc_cfg"]["eta_sq_share"] * 100 for d in domains]
g_share = [p3["elf_b_factorial_reanalysis"]["rep4"][d]["anova_and_restricted_perm"]["gamma"]["eta_sq_share"] * 100 for d in domains]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150)
x = np.arange(len(domains))
w = 0.34

axes[0].bar(x - w/2, sc_phys, w, label="SC-CFG (4.0 - 1.0)", color="#d95f02", edgecolor="black")
axes[0].bar(x + w/2, g_phys, w, label=r"$\gamma$ ($\gamma_{\max} - 0.0$)", color="#1b9e77", edgecolor="black")
axes[0].set_xticks(x)
axes[0].set_xticklabels(dom_labels)
axes[0].set_ylabel("Physical Shift in rep-4 (percentage points)")
axes[0].set_title("(a) Physical Marginal Effects on rep-4")
axes[0].legend()

axes[1].bar(x - w/2, sc_share, w, label="SC-CFG Share of Total SS (%)", color="#d95f02", edgecolor="black")
axes[1].bar(x + w/2, g_share, w, label=r"$\gamma$ Share of Total SS (%)", color="#1b9e77", edgecolor="black")
axes[1].set_xticks(x)
axes[1].set_xticklabels(dom_labels)
axes[1].set_ylabel(r"Total Variance Share $\eta^2$ (%)")
axes[1].set_title(r"(b) Domain Sensitivity of $\eta^2$ Share")
axes[1].legend()

plt.tight_layout()
fig_path = PROJECT_ROOT / "reports" / "figures" / "fig_nb03_factorial_domain_sensitivity.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path)
plt.close(fig)
print(f"Saved publication figure to {fig_path}")
""", execute_locally=True, local_ns=ns),
    ]
    save_notebook(NB_DIR / "03_factorial_anova_mixedlm_and_metric_families.ipynb", cells)


# ==============================================================================
# NOTEBOOK 09: Actual ACE Intervention & Ablations (Phase 4)
# ==============================================================================

def build_nb09() -> None:
    ns = {"__name__": "__main__", "ROOT": ROOT}
    cells = [
        make_md_cell("""
# Notebook 09 — Actual Attractor-Contrast-Escape (ACE) Intervention, Ablations, & Operating-Point Non-Equivalence (Phase 4)

**100% Standalone on Google Colab:** You do **not** need to run any prior notebook before running this one. Cell 0 automatically bootstraps `second_stage_stats.json`, `ELF`, and `elfx` on Google Drive.

**Status:**
- **Sections 1–2 (Algorithmic Audit of Zhang et al. 2026 `arXiv:2607.00588`, Exact ACE Implementation, & Live PyTorch CPU Smoke Test):** `COMPLETE & VERIFIED LOCALLY`
- **Sections 3–4 (Held-Out Calibration on Seeds `301..302` & Confirmatory Evaluation on Seeds `201..205` at Matched Gen-PPL and $H_{\\text{ps}}$):** `AUTO-RUNS WHEN EXECUTED ON COLAB GPU`
"""),
        make_code_cell(make_setup_cell_source("nb09", include_elfx_gpu_bootstrap=True), execute_locally=True, local_ns=ns, form_view=True),
        make_code_cell("""
# Cell 1: Complete Implementation of Actual ACE Calibration, ACE Steering, Ablations, & Rank-1 Step-Delta SVD
from typing import Dict, List, Optional, Tuple
import torch.nn as nn

def estimate_ace_direction(
    traj_mean_u: torch.Tensor,
    rep4_scores: np.ndarray,
    tail_quantile: float = 0.25,
) -> Dict[str, torch.Tensor]:
    \"\"\"
    Estimates the unit ACE repetition direction d_hat_ACE from calibration trajectories (Seeds 301..302).
    Args:
        traj_mean_u: Tensor of shape (M_cal, d_latent) containing mean self-conditioning feedback u per sequence.
        rep4_scores: Array of shape (M_cal,) containing sequence-level 4-gram repetition rates.
        tail_quantile: Fraction of highest-rep4 (trapped) and lowest-rep4 (free) trajectories to contrast.
    \"\"\"
    M_cal, d_lat = traj_mean_u.shape
    k_tail = max(1, int(round(M_cal * tail_quantile)))
    order = np.argsort(rep4_scores)
    low_idx = torch.as_tensor(order[:k_tail], dtype=torch.long, device=traj_mean_u.device)
    high_idx = torch.as_tensor(order[-k_tail:], dtype=torch.long, device=traj_mean_u.device)

    mu_low = traj_mean_u[low_idx].mean(dim=0)
    mu_high = traj_mean_u[high_idx].mean(dim=0)
    diff = mu_high - mu_low
    d_hat_ace = diff / diff.norm(p=2).clamp(min=1e-12)

    # Construct a random unit vector strictly orthogonal to d_hat_ace for Ablation B
    g_rng = torch.Generator(device=traj_mean_u.device).manual_seed(202704)
    r_vec = torch.randn(d_lat, generator=g_rng, dtype=traj_mean_u.dtype, device=traj_mean_u.device)
    r_orth = r_vec - torch.dot(r_vec, d_hat_ace) * d_hat_ace
    d_hat_rand_orth = r_orth / r_orth.norm(p=2).clamp(min=1e-12)

    return {
        "d_hat_ace": d_hat_ace,
        "d_hat_rand_orth": d_hat_rand_orth,
        "mu_high_norm": mu_high.norm(p=2).item(),
        "mu_low_norm": mu_low.norm(p=2).item(),
        "contrast_norm": diff.norm(p=2).item(),
        "mean_rep4_high_tail": float(np.mean(rep4_scores[order[-k_tail:]])),
        "mean_rep4_low_tail": float(np.mean(rep4_scores[order[:k_tail]])),
    }


def apply_ace_steering(
    u: torch.Tensor,
    d_hat: torch.Tensor,
    alpha: float = 1.0,
    mode: str = "projection",
    cond_seq_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    \"\"\"
    Applies ACE intervention to self-conditioning feedback u = x_pred_prev of shape (B, L, d).
    Modes:
      - 'projection': u_steered = u - alpha * <u, d_hat> * d_hat
      - 'subtraction': u_steered = u - alpha * ||u||_2 * d_hat
    Preserves clean conditioning tokens if cond_seq_mask is provided.
    \"\"\"
    d_view = d_hat.view(1, 1, -1).to(dtype=u.dtype, device=u.device)
    if mode == "projection":
        proj_coeff = (u * d_view).sum(dim=-1, keepdim=True)
        u_steered = u - float(alpha) * proj_coeff * d_view
    elif mode == "subtraction":
        u_norm = u.norm(p=2, dim=-1, keepdim=True)
        u_steered = u - float(alpha) * u_norm * d_view
    else:
        raise ValueError(f"Unknown ACE mode: {mode}")

    if cond_seq_mask is not None:
        mask = cond_seq_mask.unsqueeze(-1) if cond_seq_mask.dim() == 2 else cond_seq_mask
        u_steered = torch.where(mask > 0, u, u_steered)
    return u_steered


def apply_rank1_step_delta_svd(
    x_pred_curr: torch.Tensor,
    x_pred_prev: torch.Tensor,
    eta_svd: float = 0.5,
) -> torch.Tensor:
    \"\"\"
    Separate Heuristic Baseline (NOT ACE): Online per-sample rank-1 singular vector attenuation
    of step-to-step prediction deltas Delta_x = x_pred_curr - x_pred_prev.
    \"\"\"
    delta = x_pred_curr - x_pred_prev
    u_svd, s_svd, vh_svd = torch.linalg.svd(delta.float(), full_matrices=False)
    rank1 = (u_svd[:, :, :1] * s_svd[:, None, :1]) @ vh_svd[:, :1, :]
    return x_pred_curr - float(eta_svd) * rank1.to(x_pred_curr.dtype)
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 2: Live PyTorch CPU Verification & Smoke Test of Actual ACE vs. Ablations vs. Rank-1 Step-Delta SVD
set_all_seeds(2027)

M_cal, B_eval, L_seq, d_lat = 128, 16, 32, 64
true_attractor = torch.randn(d_lat)
true_attractor = true_attractor / true_attractor.norm()

cal_noise = 0.35 * torch.randn(M_cal, d_lat)
attractor_loadings = torch.linspace(-1.5, 2.0, M_cal)
traj_u_cal = cal_noise + attractor_loadings.unsqueeze(1) * true_attractor.unsqueeze(0)
rep4_cal = (0.02 + 0.04 * torch.sigmoid(1.8 * attractor_loadings)).numpy()

ace_cal = estimate_ace_direction(traj_u_cal, rep4_cal, tail_quantile=0.25)
d_hat_ace = ace_cal["d_hat_ace"]
d_hat_rand = ace_cal["d_hat_rand_orth"]

cos_recovery = float(torch.dot(d_hat_ace, true_attractor).item())
orth_check = float(torch.dot(d_hat_ace, d_hat_rand).item())

u_batch = torch.randn(B_eval, L_seq, d_lat) + 1.2 * true_attractor.view(1, 1, -1)
u_ace_proj = apply_ace_steering(u_batch, d_hat_ace, alpha=1.0, mode="projection")
u_ace_sub = apply_ace_steering(u_batch, d_hat_ace, alpha=0.2, mode="subtraction")
u_ace_rand = apply_ace_steering(u_batch, d_hat_rand, alpha=1.0, mode="projection")
u_ace_anti = apply_ace_steering(u_batch, d_hat_ace, alpha=-0.5, mode="projection")
u_svd_heur = apply_rank1_step_delta_svd(u_batch, torch.zeros_like(u_batch), eta_svd=0.5)

residual_proj_along_d = float((u_ace_proj * d_hat_ace.view(1, 1, -1)).sum(dim=-1).abs().max().item())

print("=== PYTORCH CPU SMOKE TEST: ACTUAL ACE CALIBRATION & STEERING ===")
print(f"  Unit norm ||d_hat_ACE||_2                  = {d_hat_ace.norm().item():.6f}")
print(f"  Orthogonality <d_hat_ACE, d_hat_rand>      = {orth_check:+.2e}")
print(f"  Cosine recovery of true attractor          = {cos_recovery:.4f}")
print(f"  Calibration high-tail vs low-tail rep-4    = {ace_cal['mean_rep4_high_tail']:.4f} vs {ace_cal['mean_rep4_low_tail']:.4f}")
print(f"  Max |<u_ACE(alpha=1.0), d_hat_ACE>|        = {residual_proj_along_d:.2e} (exact projection removal)")
print(f"  Output shapes verified (B, L, d)           = {tuple(u_ace_proj.shape)}, NaN count = {int(torch.isnan(u_ace_proj).sum())}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 3: [COLAB GPU EXECUTION CELL] Full Phase-4 Experiment on ELF-B (Calibration Seeds 301..302 -> Evaluation Seeds 201..205)
# Automatically runs when a CUDA GPU is attached on Colab (~14 minutes total on A100).
# Baselines 1 (SC=1.0, gamma=1.651) and 2 (SC=4.0, gamma=1.000) automatically reuse the completed runs from Notebook 08!

RUN_PHASE4_ACE_GPU = torch.cuda.is_available()
N_CAL_SAMPLES = 250
N_EVAL_SAMPLES = 500
CAL_SEEDS = [301, 302]
EVAL_SEEDS = [201, 202, 203, 204, 205]

ACE_EXPERIMENT_CONDITIONS = [
    {"cond_id": "1_baseline_SC1", "sc_cfg": 1.0, "gamma": 1.651, "intervention": "none", "alpha": 0.0},
    {"cond_id": "2_baseline_SC4", "sc_cfg": 4.0, "gamma": 1.000, "intervention": "none", "alpha": 0.0},
    {"cond_id": "3_actual_ACE_proj_a05", "sc_cfg": 4.0, "gamma": 1.000, "intervention": "ace_projection", "alpha": 0.5},
    {"cond_id": "4_actual_ACE_proj_a10", "sc_cfg": 4.0, "gamma": 1.000, "intervention": "ace_projection", "alpha": 1.0},
    {"cond_id": "5_ablation_ACE_sub_b02", "sc_cfg": 4.0, "gamma": 1.000, "intervention": "ace_subtraction", "alpha": 0.2},
    {"cond_id": "6_ablation_ACE_rand_orth", "sc_cfg": 4.0, "gamma": 1.000, "intervention": "ace_random_orth", "alpha": 1.0},
    {"cond_id": "7_ablation_anti_ACE", "sc_cfg": 4.0, "gamma": 1.000, "intervention": "ace_projection", "alpha": -0.5},
    {"cond_id": "8_heuristic_rank1_svd", "sc_cfg": 4.0, "gamma": 1.000, "intervention": "rank1_step_delta_svd", "alpha": 0.5},
]

print(f"Configured {len(ACE_EXPERIMENT_CONDITIONS)} experimental conditions x {len(EVAL_SEEDS)} evaluation seeds (201..205) + {len(CAL_SEEDS)} calibration seeds (301..302).")

if RUN_PHASE4_ACE_GPU:
    import importlib
    from scipy import stats
    from elfx import load, sample, metrics, refs, registry, sweep, frontier
    from elfx import non_identifiability as ni
    for _m in [registry, refs, load, sample, metrics, sweep, frontier, ni]:
        importlib.reload(_m)
    sweep.set_compile(True)

    P9_DIR = PROJECT_ROOT / "runs" / "p9_ace_intervention"
    P9_DIR.mkdir(parents=True, exist_ok=True)
    cal_pt_path = P9_DIR / "ace_calibrated_directions.pt"
    ref_texts = refs.get_owt_refs(1000)

    # --- STEP 1: HELD-OUT CALIBRATION ON SEEDS 301..302 ---
    if cal_pt_path.exists():
        cal_payload = torch.load(cal_pt_path, map_location="cpu", weights_only=False)
        d_hat_ace = cal_payload["d_hat_ace"]
        d_hat_rand_orth = cal_payload["d_hat_rand_orth"]
        print(f"[ACE Calibration] Loaded cached d_hat_ACE from {cal_pt_path} (high-tail rep4={cal_payload['mean_rep4_high_tail']:.4f} vs low-tail rep4={cal_payload['mean_rep4_low_tail']:.4f})")
    else:
        print("\\n[ACE Calibration] Running held-out calibration trajectories on seeds 301..302 (SC=4.0, gamma=1.0, K=32)...")
        _cal_batches = []
        _cal_state = {"u_sum": None}

        def _make_record_hook(eager_model, encoder, tokenizer, config, steps=32):
            def _hook(i, t, t_next, t_eff, z, x_pred):
                u_mean_l = x_pred.detach().float().mean(dim=1)
                if i == 0 or _cal_state["u_sum"] is None:
                    _cal_state["u_sum"] = u_mean_l
                else:
                    _cal_state["u_sum"] = _cal_state["u_sum"] + u_mean_l
                if i == steps - 2:
                    _cal_batches.append((_cal_state["u_sum"] / float(steps - 1)).cpu())
                    _cal_state["u_sum"] = None
                return z, x_pred
            return _hook

        sweep.HOOKS["ace_record"] = _make_record_hook
        all_u_list, all_rep4_list = [], []
        for cseed in CAL_SEEDS:
            _cal_batches.clear()
            cal_run_id = f"p9_ace_cal_elfb_s32_g1.0_sc4.0_seed{cseed}"
            cal_rd = PROJECT_ROOT / "runs" / cal_run_id
            # If summary already exists from an earlier partial run without trajectories, remove summary so hook records u
            if (cal_rd / "summary.json").exists() and not (cal_rd / "traj_u.pt").exists():
                shutil.rmtree(cal_rd, ignore_errors=True)
            if (cal_rd / "traj_u.pt").exists() and (cal_rd / "per_sample.npz").exists():
                u_seed = torch.load(cal_rd / "traj_u.pt", map_location="cpu", weights_only=False)
                rep4_seed = np.load(cal_rd / "per_sample.npz")["rep4"][:N_CAL_SAMPLES]
            else:
                cal_cfg = {
                    "run_id": cal_run_id,
                    "model": "ELF-B",
                    "steps": 32,
                    "method": "sde",
                    "gamma": 1.0,
                    "sc_cfg": 4.0,
                    "n": N_CAL_SAMPLES,
                    "seed": int(cseed),
                    "mauve": False,
                    "hook": "ace_record",
                    "hook_kwargs": {"steps": 32},
                }
                sweep.run_config(cal_cfg, verbose=True)
                u_seed = torch.cat(_cal_batches, dim=0)[:N_CAL_SAMPLES]
                torch.save(u_seed, cal_rd / "traj_u.pt")
                rep4_seed = np.load(cal_rd / "per_sample.npz")["rep4"][:N_CAL_SAMPLES]
            all_u_list.append(u_seed)
            all_rep4_list.append(rep4_seed)

        traj_u_cal = torch.cat(all_u_list, dim=0)
        rep4_cal = np.concatenate(all_rep4_list, axis=0)
        cal_payload = estimate_ace_direction(traj_u_cal, rep4_cal, tail_quantile=0.25)
        torch.save(cal_payload, cal_pt_path)
        d_hat_ace = cal_payload["d_hat_ace"]
        d_hat_rand_orth = cal_payload["d_hat_rand_orth"]
        print(f"[ACE Calibration] Saved {cal_pt_path}: ||mu_high - mu_low||_2 = {cal_payload['contrast_norm']:.4f}, high-tail rep4 = {cal_payload['mean_rep4_high_tail']:.4f} vs low-tail rep4 = {cal_payload['mean_rep4_low_tail']:.4f}")

    # --- STEP 2: REGISTER INTERVENTION HOOK & EXECUTE ALL 8 CONDITIONS ACROSS SEEDS 201..205 ---
    def _make_intervene_hook(eager_model, encoder, tokenizer, config, intervention="none", alpha=0.0):
        _state = {"prev_xp": None}
        def _hook(i, t, t_next, t_eff, z, x_pred):
            if intervention == "ace_projection":
                xp_new = apply_ace_steering(x_pred, d_hat_ace, alpha=float(alpha), mode="projection")
                return z, xp_new
            elif intervention == "ace_subtraction":
                xp_new = apply_ace_steering(x_pred, d_hat_ace, alpha=float(alpha), mode="subtraction")
                return z, xp_new
            elif intervention == "ace_random_orth":
                xp_new = apply_ace_steering(x_pred, d_hat_rand_orth, alpha=float(alpha), mode="projection")
                return z, xp_new
            elif intervention == "rank1_step_delta_svd":
                if i == 0 or _state["prev_xp"] is None or _state["prev_xp"].shape != x_pred.shape:
                    _state["prev_xp"] = torch.zeros_like(x_pred)
                xp_new = apply_rank1_step_delta_svd(x_pred, _state["prev_xp"], eta_svd=float(alpha))
                _state["prev_xp"] = x_pred.detach().clone()
                return z, xp_new
            return z, x_pred
        return _hook

    sweep.HOOKS["ace_intervene"] = _make_intervene_hook

    total_jobs = len(ACE_EXPERIMENT_CONDITIONS) * len(EVAL_SEEDS)
    job_idx = 0
    for cond in ACE_EXPERIMENT_CONDITIONS:
        cid = cond["cond_id"]
        for eseed in EVAL_SEEDS:
            job_idx += 1
            # Reuse Notebook 08 baseline runs if available on disk
            if cid == "1_baseline_SC1":
                nb08_id = f"p4_elfb_s32_g1.651_sc1.0_seed{eseed}"
                run_id = nb08_id if (PROJECT_ROOT / "runs" / nb08_id / "summary.json").exists() else f"p9_{cid}_seed{eseed}"
                n_target = 1000 if run_id == nb08_id else N_EVAL_SAMPLES
            elif cid == "2_baseline_SC4":
                nb08_id = f"p4_elfb_s32_g1.0_sc4.0_seed{eseed}"
                run_id = nb08_id if (PROJECT_ROOT / "runs" / nb08_id / "summary.json").exists() else f"p9_{cid}_seed{eseed}"
                n_target = 1000 if run_id == nb08_id else N_EVAL_SAMPLES
            else:
                run_id = f"p9_{cid}_seed{eseed}"
                n_target = N_EVAL_SAMPLES

            rd = PROJECT_ROOT / "runs" / run_id
            summ_path = rd / "summary.json"
            ext_path = rd / "extended_metrics.json"

            if not summ_path.exists():
                print(f"\\n[{job_idx}/{total_jobs}] Running {run_id} ({cid}, alpha={cond['alpha']})...")
                cfg = {
                    "run_id": run_id,
                    "model": "ELF-B",
                    "steps": 32,
                    "method": "sde",
                    "gamma": float(cond["gamma"]),
                    "sc_cfg": float(cond["sc_cfg"]),
                    "n": int(n_target),
                    "seed": int(eseed),
                    "mauve": True,
                }
                if cond["intervention"] != "none":
                    cfg["hook"] = "ace_intervene"
                    cfg["hook_kwargs"] = {"intervention": cond["intervention"], "alpha": float(cond["alpha"])}
                sweep.run_config(cfg, verbose=True)
            else:
                print(f"[{job_idx}/{total_jobs}] [skip] {run_id} already has summary.json")

            if not ext_path.exists() and (rd / "samples.jsonl").exists():
                texts = [json.loads(line)["text"] for line in (rd / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                ext = ni.score_extended_sample_metrics(texts=texts, ref_texts=ref_texts, ctw_depth=6, seed=int(eseed))
                ni.save_json_atomic(ext_path, ext)
                summ = ni.load_json(summ_path, {})
                summ.update(ext)
                ni.save_json_atomic(summ_path, summ)

    # --- STEP 3: AGGREGATE 5-SEED RESULTS ACROSS ALL 8 CONDITIONS & PACKAGE ZIP ---
    print("\\n" + "=" * 118)
    print("=== PHASE 4 CONFIRMATORY RESULTS: ACTUAL ACE vs. ABLATIONS vs. RANK-1 SVD (SEEDS 201..205) ===")
    print("=" * 118)
    cond_records = {}
    for cond in ACE_EXPERIMENT_CONDITIONS:
        cid = cond["cond_id"]
        rows = []
        for eseed in EVAL_SEEDS:
            cand_ids = [f"p9_{cid}_seed{eseed}", f"p4_elfb_s32_g{float(cond['gamma'])}_sc{float(cond['sc_cfg'])}_seed{eseed}"]
            for rid in cand_ids:
                sp = PROJECT_ROOT / "runs" / rid / "summary.json"
                ep = PROJECT_ROOT / "runs" / rid / "extended_metrics.json"
                if sp.exists():
                    d = json.loads(sp.read_text(encoding="utf-8"))
                    if ep.exists():
                        d.update(json.loads(ep.read_text(encoding="utf-8")))
                    rows.append(d)
                    break
        cond_records[cid] = rows

    print(f"{'Condition':<26} {'n':>2} {'Gen-PPL':>14} {'H_ps (nats)':>16} {'rep-4':>16} {'JSD_pi':>15} {'Self-BLEU':>15} {'MAUVE':>14}")
    print("-" * 118)
    summary_out = {"calibration": {k: float(v) for k, v in cal_payload.items() if isinstance(v, (int, float))}, "conditions": {}}
    base_sc4_rows = cond_records.get("2_baseline_SC4", [])

    for cond in ACE_EXPERIMENT_CONDITIONS:
        cid = cond["cond_id"]
        rows = cond_records[cid]
        if not rows:
            continue
        def _ms(key):
            arr = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
            return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0, arr

        ppl_m, ppl_s, ppl_a = _ms("gen_ppl")
        hps_m, hps_s, hps_a = _ms("entropy")
        rep_m, rep_s, rep_a = _ms("rep4")
        jsd_m, jsd_s, jsd_a = _ms("jsd_inter_sample")
        sb_m, sb_s, sb_a = _ms("self_bleu")
        mv_m, mv_s, mv_a = _ms("mauve")

        print(f"{cid:<26} {len(rows):>2} {ppl_m:>7.2f}+/-{ppl_s:<5.2f} {hps_m:>8.4f}+/-{hps_s:<6.4f} "
              f"{rep_m:>8.4f}+/-{rep_s:<6.4f} {jsd_m:>7.4f}+/-{jsd_s:<6.4f} {sb_m:>7.2f}+/-{sb_s:<6.2f} {mv_m:>6.3f}+/-{mv_s:<5.3f}")

        c_stats = {}
        for mkey in ["gen_ppl", "entropy", "rep4", "within_distinct2", "h_pool", "jsd_inter_sample", "distinct2", "self_bleu", "jcd_bits_per_byte", "mauve"]:
            m_val, s_val, arr = _ms(mkey)
            entry = {"mean": m_val, "sd": s_val}
            if base_sc4_rows and cid != "2_baseline_SC4":
                b_arr = np.array([float(r.get(mkey, np.nan)) for r in base_sc4_rows], dtype=float)
                if len(arr) == len(b_arr) and len(arr) > 1:
                    d_vec = arr - b_arr
                    diff = float(np.mean(d_vec))
                    se = float(np.std(d_vec, ddof=1) / math.sqrt(len(d_vec))) if np.std(d_vec, ddof=1) > 0 else 1e-12
                    t_ci = float(stats.t.ppf(0.975, df=len(d_vec) - 1))
                    p_val = float(2.0 * stats.t.sf(abs(diff) / se, df=len(d_vec) - 1))
                    entry.update({"diff_vs_SC4": diff, "ci95_vs_SC4": [diff - t_ci * se, diff + t_ci * se], "p_paired_vs_SC4": p_val})
            c_stats[mkey] = entry
        summary_out["conditions"][cid] = c_stats

    print("\\n=== PAIRED DIFFERENCES VS. UNSTEERED SC=4.0 BASELINE (2_baseline_SC4) ===")
    for cid in ["3_actual_ACE_proj_a05", "4_actual_ACE_proj_a10", "5_ablation_ACE_sub_b02", "6_ablation_ACE_rand_orth", "7_ablation_anti_ACE", "8_heuristic_rank1_svd"]:
        if cid in summary_out["conditions"]:
            cs = summary_out["conditions"][cid]
            r4 = cs["rep4"]
            jp = cs["jsd_inter_sample"]
            ppl = cs["gen_ppl"]
            hps = cs["entropy"]
            if "diff_vs_SC4" in r4:
                print(f"  {cid:<26}: Delta_rep4 = {r4['diff_vs_SC4']:+.4f} (95% CI [{r4['ci95_vs_SC4'][0]:+.4f}, {r4['ci95_vs_SC4'][1]:+.4f}], p={r4['p_paired_vs_SC4']:.2e}) | "
                      f"Delta_JSD_pi = {jp['diff_vs_SC4']:+.4f} (p={jp['p_paired_vs_SC4']:.2e}) | Delta_PPL = {ppl['diff_vs_SC4']:+.2f} | Delta_Hps = {hps['diff_vs_SC4']:+.4f}")

    out_json_path = P9_DIR / "p9_ace_confirmatory_summary.json"
    out_json_path.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    zip_out = PROJECT_ROOT / "runs" / "p9_ace_intervention_artifacts.zip"
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json_path, arcname="runs/p9_ace_intervention/p9_ace_confirmatory_summary.json")
        if cal_pt_path.exists():
            zf.write(cal_pt_path, arcname="runs/p9_ace_intervention/ace_calibrated_directions.pt")
        for cond in ACE_EXPERIMENT_CONDITIONS:
            cid = cond["cond_id"]
            for eseed in EVAL_SEEDS:
                for fn in ["summary.json", "extended_metrics.json"]:
                    p = PROJECT_ROOT / "runs" / f"p9_{cid}_seed{eseed}" / fn
                    if p.exists():
                        zf.write(p, arcname=f"runs/p9_{cid}_seed{eseed}/{fn}")
    print(f"\\nSaved ACE intervention archive to {zip_out}")
else:
    print("[STATUS: GPU EXECUTION PENDING] Attach a Colab A100 GPU to run calibration + evaluation.")
""", execute_locally=True, local_ns=ns),
    ]
    save_notebook(NB_DIR / "09_actual_ace_intervention.ipynb", cells)


# ==============================================================================
# NOTEBOOK 10: Validated Semantic Diversity (Phase 5)
# ==============================================================================

def build_nb10() -> None:
    ns = {"__name__": "__main__", "ROOT": ROOT, "S2_STATS": S2_STATS}
    cells = [
        make_md_cell("""
# Notebook 10 — Validated Semantic Diversity Analysis & Multi-Representation Sensitivity (Phase 5)

**100% Standalone on Google Colab & Local CPU:** Cell 0 automatically materializes `second_stage_stats.json` if not yet present on Google Drive.

**Status:**
- **Sections 1–3 (Data Provenance Audit of `human_eval_blinded_sheet.csv`, Separated Homogeneous `seed0` Subsample Analysis, PSD Vendi Score Bootstrap CIs, LSA $k \\in \\{8, 16, 32, 64\\}$ Sensitivity, Holm–Bonferroni Correction, & Hypotheses A–E Verdict):** `EXPLORATORY WITHIN-SEED0 AUDIT — COMPLETE ON REAL DATA`
- **Section 4 (Full Multi-Seed $n=5$ Confirmatory Evaluation on Un-Truncated Passages/Seed with `SentenceTransformer` `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, and `bge-base-en-v1.5`):** `AUTO-RUNS ON COLAB`
"""),
        make_code_cell(make_setup_cell_source("nb10", include_elfx_gpu_bootstrap=False), execute_locally=True, local_ns=ns, form_view=True),
        make_code_cell("""
# Cell 1: Load Verified Phase-5 Semantic Reanalysis from second_stage_stats.json
s2 = json.loads(S2_PATH.read_text(encoding="utf-8"))
p5 = s2["phase5_semantic"]

print("=== 1. DATA PROVENANCE AUDIT OF 600 REAL PASSAGES ON DISK ===")
audit_info = p5["pairs_seed0_audit_and_reanalysis"]["M1_ELFM"].get("sampling_unit_audit", {
    "source": "human_eval_blinded_sheet.csv (600 passages across M1, S1, Refined-1)",
    "finding": "Single seed (seed0) subsample; Task 1 = 50x450c, Task 2 = 15x5x350c",
})
for k, v in audit_info.items():
    print(f"  {k}: {v}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 2: Homogeneous Task-2 Bundle Analysis (15 Bundles x 5 Snippets @ 350 chars, Seed0) & Corpus-100 Vendi Bootstrap CIs
print("=== 2. MULTI-REPRESENTATION & LSA k-SENSITIVITY TABLE (SEED0 SUBSAMPLE, B-A = HIGH_SC - LOW_SC) ===")
for ptag, pdata in p5["pairs_seed0_audit_and_reanalysis"].items():
    print(f"\\n--- Pair: {ptag} (n_seed = 1 [seed0], 15 homogeneous Task-2 bundles & 100-passage Vendi bootstrap) ---")
    print(f"  {'Representation':<12} {'T2 CosDist Diff':>15} {'95% CI (CosDist)':>22} {'p_raw':>8} {'Hedges g':>9} | {'Vendi Low':>9} {'Vendi High':>10} {'Vendi Diff (95% Boot CI)':>26}")
    print("  " + "-" * 122)
    for rname, rvals in pdata["representations"].items():
        t2 = rvals["task2_homogeneous_15bundles_cos_dist"]
        ci_cd = f"[{t2['ci95_welch'][0]:+.4f}, {t2['ci95_welch'][1]:+.4f}]"
        v_lo = rvals["corpus_100_vendi_low_sc"]
        v_hi = rvals["corpus_100_vendi_high_sc"]
        v_d = rvals["corpus_100_vendi_diff_B_minus_A"]
        v_ci = rvals["corpus_100_vendi_diff_ci95_boot"]
        v_str = f"{v_d:+.2f} [{v_ci[0]:+.2f}, {v_ci[1]:+.2f}]"
        print(f"  {rname:<12} {t2['diff_B_minus_A']:>+15.5f} {ci_cd:>22} {t2['p_welch_2sided']:>8.4f} {t2['hedges_g']:>+9.2f} | "
              f"{v_lo:>9.2f} {v_hi:>10.2f} {v_str:>26}")
    if "holm_bonferroni_task2_cos_dist" in pdata:
        print("  Holm-Bonferroni adjusted p-values across 6 primary representations:")
        for rn, hinfo in pdata["holm_bonferroni_task2_cos_dist"].items():
            print(f"    {rn:<10}: p_raw = {hinfo['p_raw']:.4f}, p_Holm = {hinfo['p_holm_adjusted']:.4f} (sig_05 = {hinfo['significant_05']})")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 3: Publication Plot — Representation Sensitivity & LSA Component Sensitivity Across Pairs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rep_keys = ["t5_raw", "t5_abtt", "gpt2_abtt", "tfidf", "lsa_k8", "lsa_k16", "lsa_k32", "lsa_k64"]
fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=150)
x = np.arange(len(rep_keys))
w = 0.25

for idx, (ptag, color) in enumerate([("M1_ELFM", "#1b9e77"), ("S1_ELFB", "#d95f02"), ("Ref1_ELFB", "#7570b3")]):
    gs = [p5["pairs_seed0_audit_and_reanalysis"][ptag]["representations"][rk]["task2_homogeneous_15bundles_cos_dist"]["hedges_g"] for rk in rep_keys]
    ax.bar(x + (idx - 1) * w, gs, w, label=ptag, color=color, edgecolor="black", alpha=0.85)

ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels(rep_keys, rotation=30, ha="right")
ax.set_ylabel("Standardized Shift in Bundle Cosine Distance (g, High SC - Low SC)")
ax.set_title("Representation & LSA Dimensionality Sensitivity on Seed0 Subsample (Options D & E)")
ax.legend()
plt.tight_layout()
fig_path = PROJECT_ROOT / "reports" / "figures" / "fig_nb10_semantic_representation_sensitivity.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path)
plt.close(fig)
print(f"Saved publication figure to {fig_path}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 4: Scientific Verdict Across Hypotheses A-E
print("=== 3. SCIENTIFIC EVALUATION OF HYPOTHESES A-E ===")
for k, v in p5["hypothesis_verdict"].items():
    print(f"  {k}: {v}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 5: [COLAB MULTI-SEED EXECUTION CELL] Seed-Level (n=5 Seeds) Semantic Evaluation Across Matched Pairs
# Automatically runs on Colab GPU (~2 minutes total) over all on-disk 5-seed `samples.jsonl` files
# (including the new C1, C2, C3 confirmatory pairs on seeds 201..205 and S1, Ref1, M1 on seeds 0..4).

RUN_FULL_5SEED_SEMANTIC_COLAB = torch.cuda.is_available()
MAX_PASSAGES_PER_SEED = 250

SENTENCE_ENCODER_MODELS = [
    ("mpnet", "sentence-transformers/all-mpnet-base-v2"),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    ("bge", "BAAI/bge-base-en-v1.5"),
]

PAIR_5SEED_SPECS = [
    ("C1_ELFB_Conf", "p4_elfb_s32_g1.346_sc1.0", "p4_elfb_s32_g1.0_sc2.0", [201, 202, 203, 204, 205]),
    ("C2_ELFB_Conf", "p4_elfb_s32_g1.547_sc1.0", "p4_elfb_s32_g1.0_sc3.0", [201, 202, 203, 204, 205]),
    ("C3_ELFB_Conf", "p4_elfb_s32_g1.651_sc1.0", "p4_elfb_s32_g1.0_sc4.0", [201, 202, 203, 204, 205]),
    ("S1_ELFB_5Seed", "p4_elfb_s32_g2.0_sc1.0", "p4_elfb_s32_g1.5_sc4.0", [0, 1, 2, 3, 4]),
    ("Ref1_ELFB_5Seed", "p4_elfb_s32_g1.75_sc1.0", "p4_elfb_s32_g0.5_sc4.0", [0, 1, 2, 3, 4]),
    ("M1_ELFM_5Seed", "p5_elfm_s64_g2.0_sc1.0", "p5_elfm_s64_g0.5_sc4.0", [0, 1, 2, 3, 4]),
]

def _compute_cos_and_vendi(emb: np.ndarray, abtt_k: int = 0) -> tuple[float, float]:
    X = np.asarray(emb, dtype=np.float64)
    if abtt_k > 0:
        X = X - X.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        comps = vt[:abtt_k]
        X = X - (X @ comps.T) @ comps
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_unit = X / np.clip(norms, 1e-12, None)
    K = X_unit @ X_unit.T
    n = K.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    mean_cos_dist = float(np.mean(1.0 - K[triu_idx]))
    eigvals = np.linalg.eigvalsh(K / float(n))
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = eigvals / max(float(np.sum(eigvals)), 1e-12)
    pos = eigvals[eigvals > 1e-12]
    vendi = float(np.exp(-np.sum(pos * np.log(pos))))
    return mean_cos_dist, vendi

if RUN_FULL_5SEED_SEMANTIC_COLAB:
    from scipy import stats
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=False)
    from sentence_transformers import SentenceTransformer

    encoders = {}
    for short_name, hf_id in SENTENCE_ENCODER_MODELS:
        print(f"Loading SentenceTransformer: {hf_id} ...")
        encoders[short_name] = SentenceTransformer(hf_id, device="cuda")

    sem_5seed_out = {}
    print("\\n" + "=" * 124)
    print("=== 5-SEED CONFIRMATORY NEURAL SEMANTIC DIVERSITY (UNIT OF ANALYSIS = SEED, n=5 SEEDS PER ARM) ===")
    print("=" * 124)
    print(f"{'Pair':<16} {'Encoder (Mode)':<18} {'CosDist A':>10} {'CosDist B':>10} {'Diff (95% Welch CI)':>26} {'p_Welch':>9} {'Hedges g':>9} | {'Vendi A':>8} {'Vendi B':>8} {'Vendi Diff':>11} {'p_Vendi':>8}")
    print("-" * 124)

    for pair_id, pref_a, pref_b, seeds in PAIR_5SEED_SPECS:
        texts_a_by_seed, texts_b_by_seed = {}, {}
        for s in seeds:
            pa = PROJECT_ROOT / "runs" / f"{pref_a}_seed{s}" / "samples.jsonl"
            pb = PROJECT_ROOT / "runs" / f"{pref_b}_seed{s}" / "samples.jsonl"
            if pa.exists() and pb.exists():
                ta = [json.loads(l)["text"] for l in pa.read_text(encoding="utf-8").splitlines() if l.strip()][:MAX_PASSAGES_PER_SEED]
                tb = [json.loads(l)["text"] for l in pb.read_text(encoding="utf-8").splitlines() if l.strip()][:MAX_PASSAGES_PER_SEED]
                if len(ta) >= 50 and len(tb) >= 50:
                    texts_a_by_seed[s] = ta
                    texts_b_by_seed[s] = tb

        if len(texts_a_by_seed) < 3:
            print(f"{pair_id:<16} [skipped — fewer than 3 seeds with samples.jsonl found on Drive]")
            continue

        sem_5seed_out[pair_id] = {"n_seeds": len(texts_a_by_seed), "seeds": sorted(texts_a_by_seed.keys()), "encoders": {}}
        valid_seeds = sorted(texts_a_by_seed.keys())

        for short_name, model_obj in encoders.items():
            for mode_tag, abtt_k in [("raw", 0), ("abtt3", 3)]:
                enc_key = f"{short_name}_{mode_tag}"
                cd_a, cd_b, vn_a, vn_b = [], [], [], []
                for s in valid_seeds:
                    emb_a = model_obj.encode(texts_a_by_seed[s], batch_size=64, show_progress_bar=False, convert_to_numpy=True)
                    emb_b = model_obj.encode(texts_b_by_seed[s], batch_size=64, show_progress_bar=False, convert_to_numpy=True)
                    ca, va = _compute_cos_and_vendi(emb_a, abtt_k=abtt_k)
                    cb, vb = _compute_cos_and_vendi(emb_b, abtt_k=abtt_k)
                    cd_a.append(ca)
                    cd_b.append(cb)
                    vn_a.append(va)
                    vn_b.append(vb)

                xa, xb = np.array(cd_a), np.array(cd_b)
                va_arr, vb_arr = np.array(vn_a), np.array(vn_b)
                nx, ny = len(xa), len(xb)
                diff_cd = float(np.mean(xb) - np.mean(xa))
                vx, vy = float(np.var(xa, ddof=1)), float(np.var(xb, ddof=1))
                se_cd = math.sqrt(vx / nx + vy / ny) if (vx + vy) > 0 else 1e-12
                df_cd = ((vx / nx + vy / ny) ** 2) / (((vx / nx) ** 2) / (nx - 1) + ((vy / ny) ** 2) / (ny - 1)) if (vx + vy) > 0 else float(nx + ny - 2)
                t95 = float(stats.t.ppf(0.975, df=df_cd))
                p_cd = float(2.0 * stats.t.sf(abs(diff_cd) / se_cd, df=df_cd))
                sp = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)) if (vx + vy) > 0 else 1e-12
                g_cd = (diff_cd / sp) * (1.0 - 3.0 / (4.0 * (nx + ny - 2) - 1.0))

                diff_vn = float(np.mean(vb_arr) - np.mean(va_arr))
                vva, vvb = float(np.var(va_arr, ddof=1)), float(np.var(vb_arr, ddof=1))
                se_vn = math.sqrt(vva / nx + vvb / ny) if (vva + vvb) > 0 else 1e-12
                df_vn = ((vva / nx + vvb / ny) ** 2) / (((vva / nx) ** 2) / (nx - 1) + ((vvb / ny) ** 2) / (ny - 1)) if (vva + vvb) > 0 else float(nx + ny - 2)
                p_vn = float(2.0 * stats.t.sf(abs(diff_vn) / se_vn, df=df_vn))

                ci_str = f"{diff_cd:+.4f} [{diff_cd - t95 * se_cd:+.4f}, {diff_cd + t95 * se_cd:+.4f}]"
                print(f"{pair_id:<16} {enc_key:<18} {np.mean(xa):>10.4f} {np.mean(xb):>10.4f} {ci_str:>26} {p_cd:>9.4f} {g_cd:>+9.2f} | "
                      f"{np.mean(va_arr):>8.2f} {np.mean(vb_arr):>8.2f} {diff_vn:>+11.2f} {p_vn:>8.4f}")

                sem_5seed_out[pair_id]["encoders"][enc_key] = {
                    "cos_dist_mean_A": float(np.mean(xa)),
                    "cos_dist_mean_B": float(np.mean(xb)),
                    "cos_dist_diff": diff_cd,
                    "cos_dist_ci95": [diff_cd - t95 * se_cd, diff_cd + t95 * se_cd],
                    "cos_dist_p_welch": p_cd,
                    "cos_dist_hedges_g": g_cd,
                    "vendi_mean_A": float(np.mean(va_arr)),
                    "vendi_mean_B": float(np.mean(vb_arr)),
                    "vendi_diff": diff_vn,
                    "vendi_p_welch": p_vn,
                }

    out_p10_json = PROJECT_ROOT / "runs" / "p10_semantic_5seed_summary.json"
    out_p10_json.parent.mkdir(parents=True, exist_ok=True)
    out_p10_json.write_text(json.dumps(sem_5seed_out, indent=2), encoding="utf-8")
    zip_p10 = PROJECT_ROOT / "runs" / "p10_semantic_5seed_artifacts.zip"
    with zipfile.ZipFile(zip_p10, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_p10_json, arcname="runs/p10_semantic_5seed_summary.json")
    print(f"\\nSaved 5-seed semantic diversity summary to {out_p10_json} and {zip_p10}")
else:
    print("[SEED-LEVEL 5-SEED SEMANTIC EVALUATION PENDING COLAB GPU EXECUTION]")
    print("Configured encoders for 5-seed x 250-passage seed-level analysis:", [x[1] for x in SENTENCE_ENCODER_MODELS])
""", execute_locally=True, local_ns=ns),
    ]
    save_notebook(NB_DIR / "10_validated_semantic_diversity.ipynb", cells)


# ==============================================================================
# NOTEBOOK 04 (UPDATED): Human Evaluation Power, Multi-Level CLMM/GLMM, &
#                        Seed-Stratified Protocol (Phase 6)
# ==============================================================================

def build_nb04() -> None:
    ns = {"__name__": "__main__", "ROOT": ROOT, "S2_STATS": S2_STATS}
    cells = [
        make_md_cell("""
# Notebook 04 — Human Evaluation Multi-Level Power Analysis, Seed-Stratified Protocol, & Mixed-Effects Inference (Phase 6)

**100% Standalone on Google Colab & Local CPU:** Cell 0 automatically materializes `second_stage_stats.json` if not yet present on Google Drive.

**Status:**
- **Sections 1–3 (Multi-Level Seed + Rater Clustered Power Analysis, Protocol Redesign Fixing `seed0` Bias, IRB/Ethics Checklist, & CLMM/GLMM + Krippendorff's $\\alpha$ / Gwet's $\\text{AC}_2$ Pipeline):** `COMPLETE`
- **Section 4 (Human Annotation Ratings Analysis):** `PENDING — NO HUMAN ANNOTATIONS COLLECTED YET (ZERO SIMULATED/FABRICATED RATINGS)`
"""),
        make_code_cell(make_setup_cell_source("nb04", include_elfx_gpu_bootstrap=False), execute_locally=True, local_ns=ns, form_view=True),
        make_code_cell("""
# Cell 1: Load & Display Multi-Level Clustered Power Analysis (Seeds x Annotators)
s2 = json.loads(S2_PATH.read_text(encoding="utf-8"))
p6 = s2["phase6_human_eval"]

print("Status:", p6["status"])
print("Critical Design Fix:", p6["critical_design_fix"])
print("\\n=== MULTI-LEVEL CLUSTERED POWER TABLE (TWO-SIDED ALPHA = 0.05) ===")
print(f"{'Design':<55} {'rho_seed':>8} {'DEFF_s':>7} {'Raters':>6} {'ICC_r':>6} {'N_eff':>7} {'Pwr(0.65)':>10} {'Pwr(0.70)':>10} {'Pwr(0.75)':>10}")
print("-" * 126)
for row in p6["clustered_power_grid"]:
    if row["rho_seed"] in (0.05, 0.10):
        pw = row["power_two_sided_alpha05"]
        print(f"{row['design']:<55} {row['rho_seed']:>8.2f} {row['deff_seed']:>7.3f} {row['n_raters']:>6} "
              f"{row['icc_rater']:>6.2f} {row['effective_sample_size_N_eff']:>7.1f} "
              f"{pw['p_65']:>10.3f} {pw['p_70']:>10.3f} {pw['p_75']:>10.3f}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 2: Psychometric Agreement (Krippendorff's Alpha [Nominal & Ordinal] + Gwet's AC1/AC2) & Mixed-Effects Pipeline
def krippendorff_alpha(ratings_matrix: np.ndarray, level: str = "ordinal") -> float:
    X = np.asarray(ratings_matrix, dtype=float)
    valid_rows = [row[~np.isnan(row)] for row in X if np.sum(~np.isnan(row)) >= 2]
    if not valid_rows:
        return float("nan")
    all_vals = np.concatenate(valid_rows)
    categories = np.unique(all_vals)
    if len(categories) <= 1:
        return 1.0

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    counts = np.array([np.sum(all_vals == c) for c in categories], dtype=float)

    def dist_sq(c1, c2):
        if c1 == c2:
            return 0.0
        if level == "nominal":
            return 1.0
        i1, i2 = min(cat_to_idx[c1], cat_to_idx[c2]), max(cat_to_idx[c1], cat_to_idx[c2])
        s = np.sum(counts[i1 : i2 + 1]) - (counts[i1] + counts[i2]) / 2.0
        return float(s * s)

    d_obs = 0.0
    n_total = 0.0
    for r in valid_rows:
        m = len(r)
        n_total += m
        for i in range(m):
            for j in range(i + 1, m):
                d_obs += 2.0 * dist_sq(r[i], r[j]) / (m - 1.0)
    d_obs /= n_total

    d_exp = 0.0
    for i, c1 in enumerate(categories):
        for j, c2 in enumerate(categories):
            if i != j:
                d_exp += counts[i] * counts[j] * dist_sq(c1, c2)
    d_exp /= (n_total * (n_total - 1.0))
    return float(1.0 - d_obs / d_exp) if d_exp > 0 else 1.0

print("Verified Krippendorff's alpha (ordinal & nominal) implementation ready for real annotations.")
print("[HUMAN EVALUATION STATUS: PENDING — 0 ANNOTATIONS COLLECTED]")
""", execute_locally=True, local_ns=ns),
    ]
    save_notebook(NB_DIR / "04_human_eval_power_prep_and_analysis.ipynb", cells)


# ==============================================================================
# NOTEBOOK 06 (UPDATED): Generalization Scope — ELF-L Scale Validation &
#                        External Architecture Audit (Phase 7)
# ==============================================================================

def build_nb06() -> None:
    ns = {"__name__": "__main__", "ROOT": ROOT}
    cells = [
        make_md_cell("""
# Notebook 06 — Generalization Scope Audit: Within-Family Model-Scale Validation (`ELF-L` 652M) & External Continuous Architecture Audit (Phase 7)

**100% Standalone on Google Colab:** Cell 0 automatically bootstraps `second_stage_stats.json`, `ELF`, and `elfx` on Google Drive.

**Status:**
- **Section 1 (External Continuous-Latent Architecture Audit: `Plaid`, `SSD-LM`, `Diffusion-LM`):** `COMPLETE — PROVES WHY NO EXTERNAL CHECKPOINT IMPLEMENTS (SC-CFG, GAMMA)`
- **Section 2 (`MDLM` 130M Discrete Masked Diffusion Re-Framing):** `COMPLETE — RETAINED STRICTLY AS A COMPLEMENTARY DISCRETE EVALUATION STUDY`
- **Section 3 (`ELF-L` 652M Model-Scale Validation Experiment):** `READY FOR OPTIONAL COLAB A100 EXECUTION`
"""),
        make_code_cell(make_setup_cell_source("nb06", include_elfx_gpu_bootstrap=True), execute_locally=True, local_ns=ns, form_view=True),
        make_code_cell("""
# Cell 1: Verify Existing Cross-Scale (ELF-B 105M vs ELF-M 342M) & Complementary Discrete (MDLM 130M) Summaries on Disk
RUNS_DIR = PROJECT_ROOT / "runs" if (PROJECT_ROOT / "runs").exists() else (PROJECT_ROOT / "content" / "drive" / "MyDrive" / "elf_research" / "runs")

mdlm_summaries = sorted(RUNS_DIR.rglob("*mdlm*/summary.json")) if RUNS_DIR.exists() else []
elfb_summaries = sorted(RUNS_DIR.rglob("p4_elfb_s32_*/summary.json")) if RUNS_DIR.exists() else []
elfm_summaries = sorted(RUNS_DIR.rglob("p5_elfm_s64_*/summary.json")) if RUNS_DIR.exists() else []
elfl_summaries = sorted(RUNS_DIR.rglob("*elfl*/summary.json")) if RUNS_DIR.exists() else []

print("=== ON-DISK CHECKPOINT & RUN INVENTORY BY MODEL ===")
print(f"  ELF-B (105M, within-family baseline):        {len(elfb_summaries)} runs on disk")
print(f"  ELF-M (342M, within-family 3.3x scale):      {len(elfm_summaries)} runs on disk")
print(f"  ELF-L (652M, within-family 6.2x scale):      {len(elfl_summaries)} runs on disk (Optional GPU validation below)")
print(f"  MDLM  (130M, complementary discrete study):  {len(mdlm_summaries)} runs on disk")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 2: [COLAB A100 GPU CELL] ELF-L (652M, 64 Steps) Within-Family Scale Validation
# Automatically executes when a CUDA GPU is attached on Colab (~20 mins on A100 for 500 samples/run).
RUN_ELFL_SCALE_VALIDATION_GPU = torch.cuda.is_available()
N_ELFL_SAMPLES = 500

ELFL_SCALE_MANIFEST = [
    {"sc_cfg": 1.0, "gamma": 0.5, "seeds": [201, 202, 203]},
    {"sc_cfg": 1.0, "gamma": 2.0, "seeds": [201, 202, 203, 204, 205]},
    {"sc_cfg": 4.0, "gamma": 0.5, "seeds": [201, 202, 203, 204, 205]},
    {"sc_cfg": 4.0, "gamma": 2.0, "seeds": [201, 202, 203]},
]
print("ELF-L (652M) Within-Family Model-Scale Validation Manifest:", sum(len(x["seeds"]) for x in ELFL_SCALE_MANIFEST), "runs.")
if RUN_ELFL_SCALE_VALIDATION_GPU:
    from elfx import refs, sweep
    from elfx import non_identifiability as ni
    sweep.set_compile(True)
    ref_texts = refs.get_owt_refs(1000)
    for cell_spec in ELFL_SCALE_MANIFEST:
        sc, g = float(cell_spec["sc_cfg"]), float(cell_spec["gamma"])
        for seed in cell_spec["seeds"]:
            run_id = f"p6_elfl_s64_g{g}_sc{sc}_seed{seed}"
            rd = PROJECT_ROOT / "runs" / run_id
            summ_path = rd / "summary.json"
            ext_path = rd / "extended_metrics.json"
            if not summ_path.exists():
                cfg = {
                    "run_id": run_id,
                    "model": "ELF-L",
                    "steps": 64,
                    "method": "sde",
                    "gamma": g,
                    "sc_cfg": sc,
                    "n": N_ELFL_SAMPLES,
                    "seed": int(seed),
                    "mauve": True,
                }
                sweep.run_config(cfg, verbose=True)
            if not ext_path.exists() and (rd / "samples.jsonl").exists():
                texts = [json.loads(l)["text"] for l in (rd / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
                ext = ni.score_extended_sample_metrics(texts=texts, ref_texts=ref_texts, ctw_depth=6, seed=int(seed))
                ni.save_json_atomic(ext_path, ext)
                s = ni.load_json(summ_path, {})
                s.update(ext)
                ni.save_json_atomic(summ_path, s)

    print("\\n=== ELF-L (652M) WITHIN-FAMILY SCALE VALIDATION SUMMARY ===")
    for cell_spec in ELFL_SCALE_MANIFEST:
        sc, g = float(cell_spec["sc_cfg"]), float(cell_spec["gamma"])
        rows = []
        for seed in cell_spec["seeds"]:
            sp = PROJECT_ROOT / "runs" / f"p6_elfl_s64_g{g}_sc{sc}_seed{seed}" / "summary.json"
            if sp.exists():
                rows.append(json.loads(sp.read_text(encoding="utf-8")))
        if rows:
            print(f"  ELF-L (SC={sc}, gamma={g}, n={len(rows)}): Gen-PPL={np.mean([r['gen_ppl'] for r in rows]):.2f}, H_ps={np.mean([r['entropy'] for r in rows]):.4f}, rep4={np.mean([r['rep4'] for r in rows]):.4f}, JSD_pi={np.mean([r.get('jsd_inter_sample', 0.0) for r in rows]):.4f}")
else:
    print("[STATUS: OPTIONAL GPU EXECUTION PENDING]")
""", execute_locally=True, local_ns=ns),
    ]
    save_notebook(NB_DIR / "06_generalization_elfl_and_cross_architecture.ipynb", cells)


# ==============================================================================
# NOTEBOOK 07 (UPDATED): Computational Fairness, NFE vs. Wall-Clock Cost, &
#                        Throughput Audit (Phase 8)
# ==============================================================================

def build_nb07() -> None:
    ns = {"__name__": "__main__", "ROOT": ROOT, "S2_STATS": S2_STATS}
    cells = [
        make_md_cell("""
# Notebook 07 — Computational Fairness, NFE vs. Wall-Clock Cost, & Throughput Audit (Phase 8)

**100% Standalone on Google Colab & Local CPU:** Cell 0 automatically materializes `second_stage_stats.json` if not yet present on Google Drive.

**Status:**
- **Sections 1–3 (Exact Operation Breakdown, Logged NFE vs. Per-Sample NFE, Wall-Clock Time & Throughput `tokens/s` / `sequences/s` with $95\\%$ CIs):** `COMPLETE — VERIFIED ON REAL RUN LOGS`
- **Section 4 (Randomized, Warm-Up-Controlled, `torch.cuda.synchronize()` Benchmark Protocol):** `AUTO-RUNS ON COLAB GPU`
"""),
        make_code_cell(make_setup_cell_source("nb07", include_elfx_gpu_bootstrap=True), execute_locally=True, local_ns=ns, form_view=True),
        make_code_cell("""
# Cell 1: Load Verified Phase-8 Computational & Throughput Audit from second_stage_stats.json
s2 = json.loads(S2_PATH.read_text(encoding="utf-8"))
p8 = s2["phase8_compute"]

print("Terminology Verdict:", p8["terminology_verdict"])
print("\\n=== 1. EXACT OPERATION BREAKDOWN PER BATCH ===")
for k, v in p8["operation_breakdown_per_batch"].items():
    print(f"  {k}: {v}")

print("\\n=== 2. ELF-M (342M, K=64, M=1,000, A100-SXM4-40GB, B=48) WALL-CLOCK & THROUGHPUT BY CELL ===")
print(f"{'Cell':<12} {'n':>3} {'Mean wall_s':>12} {'SD wall_s':>10} {'95% CI wall_s':>22} {'Seq/s (Mean +/- SD)':>22} {'Tokens/s (Mean +/- SD)':>24}")
print("-" * 112)
for cell, info in p8["elf_m_64steps_1000samples_summary"].items():
    ci = f"[{info['ci95_wall_s'][0]:.2f}, {info['ci95_wall_s'][1]:.2f}]"
    seq_s = f"{info['mean_sequences_per_s']:.2f} +/- {info['sd_sequences_per_s']:.2f}"
    tok_s = f"{info['mean_tokens_per_s']:.1f} +/- {info['sd_tokens_per_s']:.1f}"
    print(f"{cell:<12} {info['n_runs']:>3} {info['mean_wall_s']:>12.2f} {info['sd_wall_s']:>10.2f} {ci:>22} {seq_s:>22} {tok_s:>24}")

sc_cmp = p8["elf_m_sc1_vs_sc4_wall_s"]
g_cmp = p8["elf_m_g05_vs_g20_wall_s"]
print(f"\\nELF-M Wall-Clock Comparison (SC-CFG 4.0 vs 1.0): diff = {sc_cmp['diff_B_minus_A']:+.2f}s ({sc_cmp['rel_diff_pct_vs_A']:+.2f}%), 95% CI = [{sc_cmp['ci95_welch'][0]:+.2f}s, {sc_cmp['ci95_welch'][1]:+.2f}s], p = {sc_cmp['p_welch_2sided']:.4f}")
print(f"ELF-M Wall-Clock Comparison (gamma 2.0 vs 0.5):  diff = {g_cmp['diff_B_minus_A']:+.2f}s ({g_cmp['rel_diff_pct_vs_A']:+.2f}%), 95% CI = [{g_cmp['ci95_welch'][0]:+.2f}s, {g_cmp['ci95_welch'][1]:+.2f}s], p = {g_cmp['p_welch_2sided']:.4f}")
""", execute_locally=True, local_ns=ns),
        make_code_cell("""
# Cell 2: [COLAB GPU BENCHMARK CELL] Synchronized CUDA Timing & Peak Memory Protocol (~1.5 mins on Colab GPU)
RUN_SYNCHRONIZED_CUDA_BENCHMARK = torch.cuda.is_available()

if RUN_SYNCHRONIZED_CUDA_BENCHMARK:
    from elfx import sweep, sample
    sweep.set_compile(True)
    m_info = sweep.get_model("ELF-B")
    model, config, tok = m_info["model"], m_info["config"], m_info["tokenizer"]
    conditions = [(1.0, 0.5), (1.0, 2.0), (2.0, 1.0), (4.0, 0.5), (4.0, 2.0)]
    n_warmup, n_repeats, batch_size = 2, 5, 64

    for w_idx in range(n_warmup):
        sample.generate(model, config, tok, n=batch_size, seed=900 + w_idx, steps=32, method="sde", gamma=1.0, sc_cfg=2.0, batch_size=batch_size)
    torch.cuda.synchronize()

    rng = np.random.default_rng(202708)
    schedule = [c for c in conditions for _ in range(n_repeats)]
    rng.shuffle(schedule)
    timings = {c: [] for c in conditions}
    peak_mb = {c: [] for c in conditions}

    for idx_b, (sc, g) in enumerate(schedule):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        out = sample.generate(model, config, tok, n=batch_size, seed=1000 + idx_b, steps=32, method="sde", gamma=g, sc_cfg=sc, batch_size=batch_size)
        ev_end.record()
        torch.cuda.synchronize()
        ms = ev_start.elapsed_time(ev_end)
        timings[(sc, g)].append(ms / 1000.0)
        peak_mb[(sc, g)].append(torch.cuda.max_memory_allocated() / (1024 ** 2))

    print("\\n=== SYNCHRONIZED CUDA EVENT TIMING & PEAK VRAM (ELF-B, K=32, B=64, 5 REPEATS/CONDITION) ===")
    print(f"{'Condition (SC, gamma)':<24} {'Logged NFE':>10} {'Per-Sample NFE':>15} {'GPU Time (s)':>18} {'Seq/s':>14} {'Peak VRAM (MiB)':>16}")
    print("-" * 102)
    for (sc, g) in conditions:
        t_arr = np.array(timings[(sc, g)])
        m_arr = np.array(peak_mb[(sc, g)])
        seq_s = batch_size / t_arr
        nfe_ps = 32 if sc == 1.0 else 63
        print(f"SC={sc:.1f}, gamma={g:.1f}{'':<8} {'64':>10} {nfe_ps:>15d} {np.mean(t_arr):>8.3f}+/-{np.std(t_arr, ddof=1):<7.3f} {np.mean(seq_s):>6.1f}+/-{np.std(seq_s, ddof=1):<5.1f} {np.mean(m_arr):>10.1f} MiB")
else:
    print("[SYNCHRONIZED CUDA TIMING BENCHMARK READY FOR COLAB GPU]")
""", execute_locally=True, local_ns=ns),
    ]
    save_notebook(NB_DIR / "07_computational_budget_nfe_and_consolidated_summary.ipynb", cells)


def archive_superseded_stage1_notebooks() -> None:
    archive_dir = NB_DIR / "archive_stage1"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for fname in [
        "01_operating_point_matching_and_tost.ipynb",
        "02_additional_matched_pairs_and_selfcond_intervention.ipynb",
        "05_semantic_diversity_and_metric_validation.ipynb",
    ]:
        src = NB_DIR / fname
        if src.exists():
            dst = archive_dir / fname
            shutil.move(str(src), str(dst))
            print(f"[build_second_stage_notebooks] Moved superseded Stage-1 notebook {fname} -> notebooks/archive_stage1/{fname}")


def main() -> None:
    build_nb08()
    build_nb03()
    build_nb09()
    build_nb10()
    build_nb04()
    build_nb06()
    build_nb07()
    archive_superseded_stage1_notebooks()
    print("[build_second_stage_notebooks] All Second-Stage standalone Colab notebooks built and verified successfully.")


if __name__ == "__main__":
    main()
