#!/usr/bin/env python3
"""ACL 2027 Main-Track Statistical Reanalysis & Raw Artifact Audit.

Rebuilds all quantitative claims directly from raw run artifacts (`summary.json`,
`extended_metrics.json`, `config.json`, `gen_meta.json`), computes operating-point
match diagnostics (within-seed bootstrap CIs, seed-level Satterthwaite Welch and
pooled-t CIs, and tolerance-grid sensitivity analyses without post-hoc TOST claims),
separates design-based factorial ANOVA from conditional response-surface regressions
(linear and quadratic, cell-means and run-level), applies Holm-Bonferroni and
Benjamini-Hochberg multiplicity corrections, and generates
`analysis/out/reanalysis_stats.json` and `run_manifest.json`.
"""
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "content" / "drive" / "MyDrive" / "elf_research" / "runs"
OUT_DIR = ROOT / "analysis" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Step-down Holm-Bonferroni adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adj = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = p_values[idx] * (m - rank)
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Step-up Benjamini-Hochberg FDR q-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adj = np.zeros(m, dtype=float)
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p_values[idx] * m / (rank + 1)
        running_min = min(running_min, val)
        adj[idx] = min(1.0, running_min)
    return adj.tolist()


def load_all_raw_runs() -> Dict[str, dict]:
    """Load all raw summary.json and extended_metrics.json files."""
    runs = {}
    for summary_path in sorted(RUNS_DIR.glob("*/summary.json")):
        run_dir = summary_path.parent
        run_id = run_dir.name
        with open(summary_path) as f:
            data = json.load(f)
        ext_path = run_dir / "extended_metrics.json"
        if ext_path.exists():
            with open(ext_path) as f:
                ext = json.load(f)
            data["_extended_metrics_file"] = str(ext_path.relative_to(ROOT))
            for k, v in ext.items():
                data[k] = v
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                data["_config"] = json.load(f)
        meta_path = run_dir / "gen_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                data["_gen_meta"] = json.load(f)
        data["_summary_file"] = str(summary_path.relative_to(ROOT))
        runs[run_id] = data
    return runs


def audit_definitions_and_nfe(runs: Dict[str, dict]) -> dict:
    """Audit exact metric definitions, JSD weighting, and NFE logging across all runs."""
    ext_runs = [r for r in runs.values() if "h_ps_exact" in r]
    h_ps_diffs = []
    jsd_unweighted_vs_weighted = []
    rep4_diffs = []
    jcd_diffs = []

    for r in ext_runs:
        # 1. entropy (summary.json) vs h_ps_exact (extended_metrics.json)
        h_sum = float(r["entropy"])
        h_ex = float(r["h_ps_exact"])
        h_wt = float(r["h_ps_weighted"])
        h_pl = float(r["h_pool"])
        jsd = float(r["jsd_inter_sample"])
        h_ps_diffs.append(abs(h_sum - h_ex))
        jsd_unweighted_vs_weighted.append(
            {
                "run_id": r["run_id"],
                "h_ps_exact": h_ex,
                "h_ps_weighted": h_wt,
                "h_pool": h_pl,
                "jsd_inter_weighted": jsd,
                "h_pool_minus_h_ps_exact": h_pl - h_ex,
                "weighting_gap_nats": (h_pl - h_ex) - jsd,
            }
        )
        # 2. rep4 vs 1 - within_distinct4 vs rep4_check
        r4 = float(r["rep4"])
        r4_chk = float(r["rep4_check"])
        w_d4 = float(r["within_distinct4"])
        rep4_diffs.append(max(abs(r4 - r4_chk), abs(r4 - (1.0 - w_d4))))

        # 3. jcd_bits_per_byte vs 0.5 * (kl_gen_to_ref_bpb + kl_ref_to_gen_bpb)
        jcd = float(r["jcd_bits_per_byte"])
        kl_fwd = float(r["kl_gen_to_ref_bpb"])
        kl_rev = float(r["kl_ref_to_gen_bpb"])
        jcd_diffs.append(abs(jcd - 0.5 * (kl_fwd + kl_rev)))

    # 4. NFE audit across ELF-B and ELF-M
    nfe_records = []
    for rid, r in sorted(runs.items()):
        if rid.startswith(("p4_elfb_s32_", "p4_elfm_s64_", "p5_elfm_s64_")):
            steps = int(r.get("steps", 0))
            bs = int(r.get("batch_size", 256))
            n_samples = int(r.get("n", 1000))
            n_batches = math.ceil(n_samples / bs)
            decode_chunks_per_batch = math.ceil(bs / 32)
            expected_nfe_logged = steps + decode_chunks_per_batch
            per_sample_nfe = steps + 1  # steps Euler/SDE + 1 DLM decode pass
            nfe_records.append(
                {
                    "run_id": rid,
                    "steps": steps,
                    "batch_size": bs,
                    "n_batches": n_batches,
                    "logged_n_elf_fwd": r.get("n_elf_fwd"),
                    "logged_nfe_elf": r.get("nfe_elf"),
                    "expected_logged_nfe_elf": float(expected_nfe_logged),
                    "exact_formula_match": abs(float(r.get("nfe_elf", 0)) - expected_nfe_logged) < 1e-6,
                    "true_per_sample_nfe": per_sample_nfe,
                }
            )

    gaps = [x["weighting_gap_nats"] for x in jsd_unweighted_vs_weighted]
    return {
        "n_extended_runs": len(ext_runs),
        "max_abs_diff_entropy_vs_h_ps_exact": float(max(h_ps_diffs)),
        "max_abs_diff_rep4_vs_1_minus_within_distinct4": float(max(rep4_diffs)),
        "max_abs_diff_jcd_vs_half_kl_sum": float(max(jcd_diffs)),
        "weighting_gap_h_ps_exact_minus_weighted": {
            "mean_nats": float(np.mean(gaps)),
            "min_nats": float(np.min(gaps)),
            "max_nats": float(np.max(gaps)),
            "max_abs_nats": float(np.max(np.abs(gaps))),
            "explanation": (
                "jsd_inter_sample is computed as H_pool - H_ps_weighted, the exact token-length-weighted "
                "Jensen-Shannon divergence with weights pi_m = T_m / sum_m' T_m'. H_ps_exact (unweighted mean "
                "across M=1000 samples) differs from H_ps_weighted by at most 0.0020 nats."
            ),
            "per_run": jsd_unweighted_vs_weighted,
        },
        "nfe_audit": {
            "n_runs_checked": len(nfe_records),
            "all_match_formula_steps_plus_ceil_bs_div_32": all(x["exact_formula_match"] for x in nfe_records),
            "explanation": (
                "In elfx/sample.py, CounterWrapper counts batched calls to model(). Rollout makes `steps` calls "
                "per batch (1 per step because ELF uses guidance-distilled self_cond_cfg_scale tokens via "
                "num_self_cond_cfg_tokens > 0), and decode_latents chunks each batch into slices of 32 tokens, "
                "adding ceil(batch_size / 32) calls per batch. True per-sample NFE is identically `steps + 1` "
                "(33 for 32-step ELF-B, 65 for 64-step ELF-M) across all SC-CFG and gamma settings."
            ),
            "sample_records": nfe_records[:10],
        },
    }


def compute_pair_tost_and_stats(runs_a: List[dict], runs_b: List[dict], label: str) -> dict:
    """Compute detailed matched-pair statistics, TOST equivalence margins, and seed-level / bootstrap CIs."""
    metrics = [
        "gen_ppl",
        "entropy",
        "h_ps_weighted",
        "h_pool",
        "jsd_inter_sample",
        "rep4",
        "within_distinct2",
        "within_distinct4",
        "distinct1",
        "distinct2",
        "distinct3",
        "cross_distinct4",
        "self_bleu",
        "jcd_bits_per_byte",
        "kl_gen_to_ref_bpb",
        "kl_ref_to_gen_bpb",
        "mauve",
        "pos_entropy_q4_minus_q1",
        "pos_entropy_spearman_rho",
        "mean_gpt2_len",
        "p10",
        "p50",
        "p90",
    ]
    out = {
        "label": label,
        "config_a": {"sc_cfg": runs_a[0]["sc_cfg"], "gamma": runs_a[0]["gamma"], "n_seeds": len(runs_a)},
        "config_b": {"sc_cfg": runs_b[0]["sc_cfg"], "gamma": runs_b[0]["gamma"], "n_seeds": len(runs_b)},
        "metrics": {},
    }

    for m in metrics:
        va = [float(r[m]) for r in runs_a if m in r and r[m] is not None and not np.isnan(float(r[m]))]
        vb = [float(r[m]) for r in runs_b if m in r and r[m] is not None and not np.isnan(float(r[m]))]
        if not va or not vb:
            continue
        mean_a = float(np.mean(va))
        mean_b = float(np.mean(vb))
        diff = mean_b - mean_a
        rel_pct = 100.0 * diff / mean_a if mean_a != 0 else float("nan")
        sd_a = float(np.std(va, ddof=1)) if len(va) > 1 else 0.0
        sd_b = float(np.std(vb, ddof=1)) if len(vb) > 1 else 0.0
        se_diff = math.sqrt((sd_a ** 2) / len(va) + (sd_b ** 2) / len(vb)) if (len(va) > 1 and len(vb) > 1) else float("nan")

        # Paired by seed index (seed 0, seed 1)
        paired_diffs = [vb[i] - va[i] for i in range(min(len(va), len(vb)))]
        se_paired = float(np.std(paired_diffs, ddof=1) / math.sqrt(len(paired_diffs))) if len(paired_diffs) > 1 else float("nan")

        # Welch t-test (with exact Satterthwaite df) and equal-variance Student t-test (df_pool = n_A + n_B - 2)
        if len(va) > 1 and len(vb) > 1 and se_diff > 0:
            t_welch, p_welch = stats.ttest_ind(vb, va, equal_var=False)
            t_student, p_student = stats.ttest_ind(vb, va, equal_var=True)
            v1 = (sd_a ** 2) / len(va)
            v2 = (sd_b ** 2) / len(vb)
            df_satt = float(((v1 + v2) ** 2) / ((v1 ** 2) / (len(va) - 1) + (v2 ** 2) / (len(vb) - 1)))
            df_pool = len(va) + len(vb) - 2
            t_crit_90 = float(stats.t.ppf(0.95, df=df_pool))
            t_crit_95 = float(stats.t.ppf(0.975, df=df_pool))
            t_crit_95_satt = float(stats.t.ppf(0.975, df=df_satt))
            ci90_lo, ci90_hi = diff - t_crit_90 * se_diff, diff + t_crit_90 * se_diff
            ci95_lo, ci95_hi = diff - t_crit_95 * se_diff, diff + t_crit_95 * se_diff
            ci95_satt_lo, ci95_satt_hi = diff - t_crit_95_satt * se_diff, diff + t_crit_95_satt * se_diff
            tost_min_margin_t = max(abs(ci90_lo), abs(ci90_hi))
        else:
            t_welch = p_welch = t_student = p_student = df_satt = float("nan")
            ci90_lo = ci90_hi = ci95_lo = ci95_hi = ci95_satt_lo = ci95_satt_hi = tost_min_margin_t = float("nan")

        out["metrics"][m] = {
            "seeds_a": va,
            "seeds_b": vb,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "sd_a": sd_a,
            "sd_b": sd_b,
            "diff_b_minus_a": diff,
            "rel_diff_pct": rel_pct,
            "se_unpaired": se_diff,
            "se_paired": se_paired,
            "paired_seed_diffs": paired_diffs,
            "t_welch": float(t_welch),
            "df_satt": float(df_satt),
            "p_welch_2seed": float(p_welch),
            "t_student_pool": float(t_student),
            "df_pool": int(len(va) + len(vb) - 2) if (len(va) > 1 and len(vb) > 1) else 0,
            "p_student_2seed": float(p_student),
            "ci90_seed_t": [ci90_lo, ci90_hi],
            "ci95_seed_t": [ci95_lo, ci95_hi],
            "ci95_welch_satt": [ci95_satt_lo, ci95_satt_hi],
            "tost_min_margin_seed_t": tost_min_margin_t,
        }

    # Add within-seed sequence-bootstrap SE for Gen-PPL from logged 95% bootstrap CIs in summary.json
    # Each run logs ppl_ci_lo, ppl_ci_hi from B=10,000 resamples of M=1,000 samples.
    se_boot_a = [0.5 * (float(r["ppl_ci_hi"]) - float(r["ppl_ci_lo"])) / 1.95996 for r in runs_a]
    se_boot_b = [0.5 * (float(r["ppl_ci_hi"]) - float(r["ppl_ci_lo"])) / 1.95996 for r in runs_b]
    var_boot_mean_a = sum(s ** 2 for s in se_boot_a) / (len(se_boot_a) ** 2)
    var_boot_mean_b = sum(s ** 2 for s in se_boot_b) / (len(se_boot_b) ** 2)
    se_boot_diff = math.sqrt(var_boot_mean_a + var_boot_mean_b)
    se_seed_ppl = out["metrics"]["gen_ppl"]["se_unpaired"]
    se_total_ppl = math.sqrt(se_boot_diff ** 2 + (se_seed_ppl ** 2 if not math.isnan(se_seed_ppl) else 0.0))
    ppl_diff = out["metrics"]["gen_ppl"]["diff_b_minus_a"]
    ppl_mean = 0.5 * (out["metrics"]["gen_ppl"]["mean_a"] + out["metrics"]["gen_ppl"]["mean_b"])
    ci90_boot = [ppl_diff - 1.64485 * se_boot_diff, ppl_diff + 1.64485 * se_boot_diff]
    ci95_boot = [ppl_diff - 1.95996 * se_boot_diff, ppl_diff + 1.95996 * se_boot_diff]
    tost_boot_margin_ppl = max(abs(ci90_boot[0]), abs(ci90_boot[1]))
    tost_boot_margin_rel_pct = 100.0 * tost_boot_margin_ppl / ppl_mean

    # Sample-level SE for Self-BLEU from self_bleu_std (M=1,000 samples per seed -> 2,000 samples across 2 seeds)
    if "self_bleu" in out["metrics"]:
        sb_vars_a = [(float(r["self_bleu_std"]) ** 2) / float(r.get("n", 1000)) for r in runs_a]
        sb_vars_b = [(float(r["self_bleu_std"]) ** 2) / float(r.get("n", 1000)) for r in runs_b]
        se_sb_sample = math.sqrt(sum(sb_vars_a) / (len(sb_vars_a) ** 2) + sum(sb_vars_b) / (len(sb_vars_b) ** 2))
        sb_diff = out["metrics"]["self_bleu"]["diff_b_minus_a"]
        z_sb = sb_diff / se_sb_sample
        p_sb_sample = float(2.0 * stats.norm.sf(abs(z_sb)))
        out["metrics"]["self_bleu"]["se_sample_level"] = se_sb_sample
        out["metrics"]["self_bleu"]["z_sample_level"] = z_sb
        out["metrics"]["self_bleu"]["p_sample_level"] = p_sb_sample

    out["operating_point_match_diagnostics"] = {
        "methodology_note": (
            "No equivalence margin was pre-registered independently of the observed data; therefore we report "
            "approximate operating-point matches using exact point differences, within-seed 95% bootstrap CIs, "
            "seed-level 95% t-CIs, and the tolerance-grid sensitivity analysis rather than claiming formal TOST equivalence."
        ),
        "gen_ppl": {
            "mean_a": out["metrics"]["gen_ppl"]["mean_a"],
            "mean_b": out["metrics"]["gen_ppl"]["mean_b"],
            "point_diff_ppl": ppl_diff,
            "rel_diff_vs_a_pct": out["metrics"]["gen_ppl"]["rel_diff_pct"],
            "rel_diff_midpoint_pct": 100.0 * abs(ppl_diff) / ppl_mean,
            "se_within_seed_bootstrap": se_boot_diff,
            "se_between_seed": se_seed_ppl,
            "se_total_combined": se_total_ppl,
            "ci95_within_seed_bootstrap_ppl": ci95_boot,
            "ci95_seed_t_df2_ppl": out["metrics"]["gen_ppl"]["ci95_seed_t"],
            "ci95_welch_satt_ppl": out["metrics"]["gen_ppl"]["ci95_welch_satt"],
        },
        "entropy": {
            "mean_a": out["metrics"]["entropy"]["mean_a"],
            "mean_b": out["metrics"]["entropy"]["mean_b"],
            "point_diff_nats": out["metrics"]["entropy"]["diff_b_minus_a"],
            "abs_point_diff_nats": abs(out["metrics"]["entropy"]["diff_b_minus_a"]),
            "se_between_seed": out["metrics"]["entropy"]["se_unpaired"],
            "ci95_seed_t_df2_nats": out["metrics"]["entropy"]["ci95_seed_t"],
            "ci95_welch_satt_nats": out["metrics"]["entropy"]["ci95_welch_satt"],
        },
    }
    out["equivalence_tost_gen_ppl"] = {
        "point_diff_ppl": ppl_diff,
        "point_rel_diff_pct": 100.0 * abs(ppl_diff) / ppl_mean,
        "se_bootstrap_combined": se_boot_diff,
        "ci90_bootstrap_ppl": ci90_boot,
        "ci95_bootstrap_ppl": ci95_boot,
        "tost_min_margin_bootstrap_ppl": tost_boot_margin_ppl,
        "tost_min_margin_bootstrap_rel_pct": tost_boot_margin_rel_pct,
        "tost_min_margin_seed_t_ppl": out["metrics"]["gen_ppl"]["tost_min_margin_seed_t"],
        "tost_min_margin_seed_t_rel_pct": 100.0 * out["metrics"]["gen_ppl"]["tost_min_margin_seed_t"] / ppl_mean,
    }
    h_diff = out["metrics"]["entropy"]["diff_b_minus_a"]
    out["equivalence_tost_entropy"] = {
        "point_diff_nats": h_diff,
        "abs_point_diff_nats": abs(h_diff),
        "ci90_seed_t_nats": out["metrics"]["entropy"]["ci90_seed_t"],
        "tost_min_margin_seed_t_nats": out["metrics"]["entropy"]["tost_min_margin_seed_t"],
        "within_0_01_nats_point": abs(h_diff) <= 0.01,
        "within_0_02_nats_point": abs(h_diff) <= 0.02,
        "within_0_031_nats_point": abs(h_diff) <= 0.031,
    }
    return out


def _seed_of(rid: str, r: dict) -> int:
    if "seed" in r and r["seed"] is not None:
        return int(r["seed"])
    if "seed" in rid:
        tail = rid.split("seed")[-1]
        if tail.isdigit():
            return int(tail)
    return 0


def sensitivity_matching_tolerances(runs: Dict[str, dict]) -> dict:
    """Sweep matching tolerances (Delta H_ps, relative Delta Gen-PPL) across ELF-B and ELF-M cells."""
    # Group ELF-B (including E3) by (sc_cfg, gamma)
    elfb_cells_2seed = {}
    elfb_cells_all = {}
    elfm_cells_9 = {}
    elfm_cells_11 = {}
    orig_9_elfm = {
        (1.0, 1.5), (1.0, 2.0), (4.0, 0.5), (4.0, 1.0),
        (3.0, 0.0), (3.0, 0.5), (3.0, 1.0), (3.0, 1.5), (3.0, 2.0),
    }
    for rid, r in runs.items():
        seed_val = _seed_of(rid, r)
        if rid.startswith("p4_elfb_s32_"):
            key = (float(r["sc_cfg"]), float(r["gamma"]))
            elfb_cells_all.setdefault(key, []).append(r)
            if seed_val in (0, 1):
                elfb_cells_2seed.setdefault(key, []).append(r)
        elif rid.startswith(("p4_elfm_s64_", "p5_elfm_s64_")):
            key = (float(r["sc_cfg"]), float(r["gamma"]))
            elfm_cells_11.setdefault(key, []).append(r)
            if key in orig_9_elfm and seed_val in (0, 1):
                elfm_cells_9.setdefault(key, []).append(r)

    def summarize_cells(cell_map):
        summary = {}
        for k, rlist in sorted(cell_map.items()):
            summary[k] = {
                "sc_cfg": k[0],
                "gamma": k[1],
                "n_seeds": len(rlist),
                "gen_ppl": float(np.mean([x["gen_ppl"] for x in rlist])),
                "entropy": float(np.mean([x["entropy"] for x in rlist])),
                "rep4": float(np.mean([x["rep4"] for x in rlist])),
                "distinct1": float(np.mean([x["distinct1"] for x in rlist])),
                "distinct2": float(np.mean([x["distinct2"] for x in rlist])),
                "mauve": float(np.mean([x["mauve"] for x in rlist])),
                "h_pool": float(np.mean([x["h_pool"] for x in rlist])) if "h_pool" in rlist[0] else None,
                "jsd_inter": float(np.mean([x["jsd_inter_sample"] for x in rlist])) if "jsd_inter_sample" in rlist[0] else None,
                "self_bleu": float(np.mean([x["self_bleu"] for x in rlist])) if "self_bleu" in rlist[0] else None,
                "jcd": float(np.mean([x["jcd_bits_per_byte"] for x in rlist])) if "jcd_bits_per_byte" in rlist[0] else None,
            }
        return summary

    b_sum = summarize_cells(elfb_cells_2seed)
    b_sum_all = summarize_cells(elfb_cells_all)
    m_sum = summarize_cells(elfm_cells_9)
    m_sum_11 = summarize_cells(elfm_cells_11)

    h_tols = [0.005, 0.010, 0.015, 0.020, 0.031, 0.050]
    ppl_tols_pct = [1.5, 2.5, 3.5, 5.0, 7.5, 10.0]

    def sweep_grid(cell_summary):
        keys = sorted(cell_summary.keys())
        all_cross_sc_pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ca, cb = cell_summary[keys[i]], cell_summary[keys[j]]
                if ca["sc_cfg"] >= cb["sc_cfg"]:
                    continue
                # Ensure cb has higher sc_cfg
                mean_ppl = 0.5 * (ca["gen_ppl"] + cb["gen_ppl"])
                rel_ppl_pct = 100.0 * abs(cb["gen_ppl"] - ca["gen_ppl"]) / mean_ppl
                abs_dh = abs(cb["entropy"] - ca["entropy"])
                all_cross_sc_pairs.append(
                    {
                        "cell_a": f"SC={ca['sc_cfg']},g={ca['gamma']}",
                        "cell_b": f"SC={cb['sc_cfg']},g={cb['gamma']}",
                        "sc_a": ca["sc_cfg"],
                        "gamma_a": ca["gamma"],
                        "sc_b": cb["sc_cfg"],
                        "gamma_b": cb["gamma"],
                        "ppl_a": ca["gen_ppl"],
                        "ppl_b": cb["gen_ppl"],
                        "rel_ppl_pct": rel_ppl_pct,
                        "h_a": ca["entropy"],
                        "h_b": cb["entropy"],
                        "abs_dh": abs_dh,
                        "d_rep4_pct": 100.0 * (cb["rep4"] - ca["rep4"]),
                        "d_distinct2": cb["distinct2"] - ca["distinct2"],
                        "d_mauve": cb["mauve"] - ca["mauve"],
                        "d_h_pool": (cb["h_pool"] - ca["h_pool"]) if (ca["h_pool"] is not None and cb["h_pool"] is not None) else None,
                        "d_self_bleu": (cb["self_bleu"] - ca["self_bleu"]) if (ca["self_bleu"] is not None and cb["self_bleu"] is not None) else None,
                        "d_jcd": (cb["jcd"] - ca["jcd"]) if (ca["jcd"] is not None and cb["jcd"] is not None) else None,
                    }
                )
        tol_table = []
        for h_tol in h_tols:
            for p_tol in ppl_tols_pct:
                matched = [p for p in all_cross_sc_pairs if p["abs_dh"] <= h_tol and p["rel_ppl_pct"] <= p_tol]
                if not matched:
                    tol_table.append(
                        {
                            "h_tol_nats": h_tol,
                            "ppl_tol_pct": p_tol,
                            "n_pairs": 0,
                        }
                    )
                    continue
                n_rep4_pos = sum(1 for p in matched if p["d_rep4_pct"] > 0)
                n_d2_pos = sum(1 for p in matched if p["d_distinct2"] > 0)
                tol_table.append(
                    {
                        "h_tol_nats": h_tol,
                        "ppl_tol_pct": p_tol,
                        "n_pairs": len(matched),
                        "frac_rep4_higher_at_high_sc": n_rep4_pos / len(matched),
                        "frac_distinct2_higher_at_high_sc": n_d2_pos / len(matched),
                        "mean_d_rep4_pct_pts": float(np.mean([p["d_rep4_pct"] for p in matched])),
                        "mean_d_distinct2": float(np.mean([p["d_distinct2"] for p in matched])),
                        "mean_d_mauve": float(np.mean([p["d_mauve"] for p in matched])),
                        "pairs": [f"{p['cell_a']} vs {p['cell_b']}" for p in matched],
                    }
                )
        return {"all_pairs_sorted": sorted(all_cross_sc_pairs, key=lambda x: (x["abs_dh"], x["rel_ppl_pct"]))[:25], "tolerance_grid": tol_table}

    return {
        "elfb_30cells": sweep_grid(b_sum),
        "elfb_30cells_allseeds": sweep_grid(b_sum_all),
        "elfm_9cells": sweep_grid(m_sum),
        "elfm_11cells": sweep_grid(m_sum_11),
    }


def design_based_factorial_and_surface_models(runs: Dict[str, dict]) -> dict:
    """Compute pure design-based two-way ANOVA on the balanced 4x7x2 ELF-B factorial grid,
    balanced 2x2x5 ELF-M factorial grid, balanced 2x2 sub-designs, and conditional response-surface
    regressions (linear + quadratic, cell-means + cluster-robust)."""
    # 1. Pure 4 x 7 x 2 balanced ELF-B grid (exclude E3 g1.25 and g1.75 and seeds >= 2 to preserve exact balance)
    base_gammas = {0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0}
    base_scs = {1.0, 2.0, 3.0, 4.0}
    grid_runs = [
        r for rid, r in runs.items()
        if rid.startswith("p4_elfb_s32_")
        and float(r["gamma"]) in base_gammas
        and float(r["sc_cfg"]) in base_scs
        and _seed_of(rid, r) in (0, 1)
    ]
    assert len(grid_runs) == 56, f"Expected 56 balanced ELF-B runs, got {len(grid_runs)}"

    outcomes = ["gen_ppl", "entropy", "rep4", "distinct1", "distinct2", "distinct3", "mauve", "mean_gpt2_len"]

    def compute_two_way_anova(sub_runs, sc_set, g_set, n_rep: int = 2, outcome_list=outcomes):
        n_sc = len(sc_set)
        n_g = len(g_set)
        res = {}
        for y_name in outcome_list:
            vals = np.array([float(r[y_name]) for r in sub_runs if y_name in r and r[y_name] is not None])
            if len(vals) != n_sc * n_g * n_rep:
                continue
            grand_mean = float(np.mean(vals))
            ss_total = float(np.sum((vals - grand_mean) ** 2))

            sc_means = {sc: np.mean([float(r[y_name]) for r in sub_runs if float(r["sc_cfg"]) == sc]) for sc in sc_set}
            g_means = {g: np.mean([float(r[y_name]) for r in sub_runs if float(r["gamma"]) == g]) for g in g_set}
            cell_means = {
                (sc, g): np.mean([float(r[y_name]) for r in sub_runs if float(r["sc_cfg"]) == sc and float(r["gamma"]) == g])
                for sc in sc_set for g in g_set
            }

            ss_sc = float(sum((n_rep * n_g) * ((sc_means[sc] - grand_mean) ** 2) for sc in sc_set))
            ss_gamma = float(sum((n_rep * n_sc) * ((g_means[g] - grand_mean) ** 2) for g in g_set))
            ss_cells = float(sum(n_rep * ((cell_means[(sc, g)] - grand_mean) ** 2) for sc in sc_set for g in g_set))
            ss_inter = ss_cells - ss_sc - ss_gamma
            ss_err = ss_total - ss_cells

            df_sc, df_gamma, df_inter, df_err = n_sc - 1, n_g - 1, (n_sc - 1) * (n_g - 1), n_sc * n_g * (n_rep - 1)
            ms_sc = ss_sc / df_sc
            ms_gamma = ss_gamma / df_gamma
            ms_inter = ss_inter / df_inter
            ms_err = ss_err / df_err

            f_sc = ms_sc / ms_err
            f_gamma = ms_gamma / ms_err
            f_inter = ms_inter / ms_err

            entry = {
                "grand_mean": grand_mean,
                "ss_total": ss_total,
                "sc_cfg": {
                    "df": df_sc,
                    "ss": ss_sc,
                    "F": float(f_sc),
                    "p": float(stats.f.sf(f_sc, df_sc, df_err)),
                    "partial_eta2": float(ss_sc / (ss_sc + ss_err)),
                    "share_of_total_ss": float(ss_sc / ss_total),
                },
                "gamma": {
                    "df": df_gamma,
                    "ss": ss_gamma,
                    "F": float(f_gamma),
                    "p": float(stats.f.sf(f_gamma, df_gamma, df_err)),
                    "partial_eta2": float(ss_gamma / (ss_gamma + ss_err)),
                    "share_of_total_ss": float(ss_gamma / ss_total),
                },
                "interaction": {
                    "df": df_inter,
                    "ss": ss_inter,
                    "F": float(f_inter),
                    "p": float(stats.f.sf(f_inter, df_inter, df_err)),
                    "partial_eta2": float(ss_inter / (ss_inter + ss_err)),
                    "share_of_total_ss": float(ss_inter / ss_total),
                },
                "pure_seed_error": {
                    "df": df_err,
                    "ss": ss_err,
                    "ms": ms_err,
                    "rmse_between_seeds": math.sqrt(ms_err),
                    "share_of_total_ss": float(ss_err / ss_total),
                },
            }
            if n_sc == 2 and n_g == 2:
                sc_lo, sc_hi = sorted(sc_set)
                g_lo, g_hi = sorted(g_set)
                y00, y01 = float(cell_means[(sc_lo, g_lo)]), float(cell_means[(sc_lo, g_hi)])
                y10, y11 = float(cell_means[(sc_hi, g_lo)]), float(cell_means[(sc_hi, g_hi)])
                entry["cell_means"] = {
                    f"sc{sc_lo:g}_g{g_lo:g}": y00,
                    f"sc{sc_lo:g}_g{g_hi:g}": y01,
                    f"sc{sc_hi:g}_g{g_lo:g}": y10,
                    f"sc{sc_hi:g}_g{g_hi:g}": y11,
                }
                entry["main_effect_sc"] = 0.5 * ((y10 - y00) + (y11 - y01))
                entry["main_effect_gamma"] = 0.5 * ((y01 - y00) + (y11 - y10))
                entry["interaction_effect"] = 0.5 * ((y11 - y10) - (y01 - y00))
                entry["diagonal_matched_contrast_sc_hi_g_lo_minus_sc_lo_g_hi"] = y10 - y01
            res[y_name] = entry
        return res

    anova_results = compute_two_way_anova(grid_runs, base_scs, base_gammas, n_rep=2)
    noncollapsed_gammas = {0.0, 0.25, 0.5, 1.0, 1.5, 2.0}
    noncollapsed_runs = [r for r in grid_runs if float(r["gamma"]) in noncollapsed_gammas]
    anova_4x6_results = compute_two_way_anova(noncollapsed_runs, base_scs, noncollapsed_gammas, n_rep=2)

    # 1b. Balanced 2 x 2 x 5 ELF-M (342M, 64 steps) Two-Way Factorial ANOVA (N=20 runs, 5 seeds/cell, df=1,1,1,16)
    elfm_2x2_runs = [
        r for rid, r in runs.items()
        if rid.startswith("p5_elfm_s64_")
        and float(r["sc_cfg"]) in {1.0, 4.0}
        and float(r["gamma"]) in {0.5, 2.0}
        and _seed_of(rid, r) in {0, 1, 2, 3, 4}
    ]
    elfm_2x2_outcomes = [
        "gen_ppl", "entropy", "rep4", "within_distinct2", "within_distinct4",
        "distinct1", "distinct2", "distinct3", "cross_distinct4",
        "h_pool", "jsd_inter_sample", "self_bleu", "jcd_bits_per_byte", "mauve", "mean_gpt2_len",
    ]
    anova_elfm_2x2x5 = (
        compute_two_way_anova(elfm_2x2_runs, {1.0, 4.0}, {0.5, 2.0}, n_rep=5, outcome_list=elfm_2x2_outcomes)
        if len(elfm_2x2_runs) == 20
        else {}
    )

    # 2. Balanced 2x2 Sub-Designs on ELF-B (pure design-based contrasts without covariate conditioning)
    def analyze_2x2_subgrid(sc_lo, sc_hi, g_lo, g_hi):
        cells = {}
        for sc in (sc_lo, sc_hi):
            for g in (g_lo, g_hi):
                cells[(sc, g)] = [r for r in grid_runs if float(r["sc_cfg"]) == sc and float(r["gamma"]) == g]
        sub_out = {"sc_levels": [sc_lo, sc_hi], "gamma_levels": [g_lo, g_hi], "metrics": {}}
        for m in ["gen_ppl", "entropy", "rep4", "distinct1", "distinct2", "mauve"]:
            y00 = np.mean([float(x[m]) for x in cells[(sc_lo, g_lo)]])
            y01 = np.mean([float(x[m]) for x in cells[(sc_lo, g_hi)]])
            y10 = np.mean([float(x[m]) for x in cells[(sc_hi, g_lo)]])
            y11 = np.mean([float(x[m]) for x in cells[(sc_hi, g_hi)]])
            main_sc = 0.5 * ((y10 - y00) + (y11 - y01))
            main_gamma = 0.5 * ((y01 - y00) + (y11 - y10))
            inter = 0.5 * ((y11 - y10) - (y01 - y00))
            # Diagonal trade-off contrast: (sc_hi, g_lo) vs (sc_lo, g_hi)
            diag_diff = y10 - y01
            sub_out["metrics"][m] = {
                "cell_sc_lo_g_lo": float(y00),
                "cell_sc_lo_g_hi": float(y01),
                "cell_sc_hi_g_lo": float(y10),
                "cell_sc_hi_g_hi": float(y11),
                "main_effect_sc": float(main_sc),
                "main_effect_gamma": float(main_gamma),
                "interaction_sc_x_gamma": float(inter),
                "diagonal_matched_contrast_sc_hi_g_lo_minus_sc_lo_g_hi": float(diag_diff),
            }
        return sub_out

    subgrids_2x2 = {
        "sc1_vs_4_gamma0.5_vs_2.0": analyze_2x2_subgrid(1.0, 4.0, 0.5, 2.0),
        "sc1_vs_3_gamma0.5_vs_1.5": analyze_2x2_subgrid(1.0, 3.0, 0.5, 1.5),
        "sc2_vs_4_gamma0.5_vs_1.5": analyze_2x2_subgrid(2.0, 4.0, 0.5, 1.5),
    }

    # 3. Conditional Response-Surface Regressions (Observational ANCOVA):
    # Compare run-level OLS vs Cell-Means OLS (eliminating seed pseudo-replication)
    # and Linear vs Quadratic response surface in (log Gen-PPL, H_ps).
    def fit_ols_hc3(X: np.ndarray, y: np.ndarray):
        n, k = X.shape
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        df = n - k
        s2 = float(np.sum(resid ** 2) / df)
        xtx_inv = np.linalg.inv(X.T @ X)
        se_ols = np.sqrt(np.diag(s2 * xtx_inv))
        # HC3 robust variance
        hat_diag = np.sum((X @ xtx_inv) * X, axis=1)
        u_hc3 = resid / np.maximum(1.0 - hat_diag, 1e-6)
        meat = X.T @ np.diag(u_hc3 ** 2) @ X
        vcov_hc3 = xtx_inv @ meat @ xtx_inv
        se_hc3 = np.sqrt(np.diag(vcov_hc3))
        t_ols = beta / se_ols
        p_ols = 2.0 * stats.t.sf(np.abs(t_ols), df=df)
        t_hc3 = beta / se_hc3
        p_hc3 = 2.0 * stats.t.sf(np.abs(t_hc3), df=df)
        r2 = 1.0 - float(np.sum(resid ** 2) / np.sum((y - np.mean(y)) ** 2))
        return {
            "n": n,
            "df": df,
            "r2": r2,
            "beta_sc": float(beta[1]),
            "se_ols_sc": float(se_ols[1]),
            "p_ols_sc": float(p_ols[1]),
            "se_hc3_sc": float(se_hc3[1]),
            "p_hc3_sc": float(p_hc3[1]),
        }

    def run_surface_suite(run_list: List[dict]):
        # Aggregate to cell means first
        cells = {}
        for r in run_list:
            key = (float(r["sc_cfg"]), float(r["gamma"]))
            cells.setdefault(key, []).append(r)

        cell_rows = []
        for (sc, g), rlist in sorted(cells.items()):
            row = {"sc_cfg": sc, "gamma": g}
            for m in ["gen_ppl", "entropy", "rep4", "distinct1", "distinct2", "distinct3", "mauve", "h_pool", "jsd_inter_sample", "self_bleu", "jcd_bits_per_byte"]:
                vals = [float(x[m]) for x in rlist if m in x and x[m] is not None and not np.isnan(float(x[m]))]
                row[m] = float(np.mean(vals)) if vals else None
            cell_rows.append(row)

        suite = {}
        for m in ["rep4", "distinct1", "distinct2", "distinct3", "mauve", "h_pool", "jsd_inter_sample", "self_bleu", "jcd_bits_per_byte"]:
            valid_runs = [r for r in run_list if m in r and r[m] is not None and not np.isnan(float(r[m]))]
            valid_cells = [c for c in cell_rows if c[m] is not None]
            if len(valid_cells) < 6:
                continue

            # Run-level linear
            sc_r = np.array([float(r["sc_cfg"]) for r in valid_runs])
            lp_r = np.log(np.array([float(r["gen_ppl"]) for r in valid_runs]))
            h_r = np.array([float(r["entropy"]) for r in valid_runs])
            y_r = np.array([float(r[m]) for r in valid_runs])
            X_lin_r = np.column_stack([np.ones_like(sc_r), sc_r, lp_r, h_r])
            res_lin_run = fit_ols_hc3(X_lin_r, y_r)

            # Cell-means linear (1 observation per (sc_cfg, gamma) cell -> zero seed pseudo-replication!)
            sc_c = np.array([float(c["sc_cfg"]) for c in valid_cells])
            lp_c = np.log(np.array([float(c["gen_ppl"]) for c in valid_cells]))
            h_c = np.array([float(c["entropy"]) for c in valid_cells])
            y_c = np.array([float(c[m]) for c in valid_cells])
            X_lin_c = np.column_stack([np.ones_like(sc_c), sc_c, lp_c, h_c])
            res_lin_cell = fit_ols_hc3(X_lin_c, y_c)

            # Cell-means quadratic surface (controls for curved frontier!)
            res_quad_cell = None
            if len(valid_cells) >= 10:
                lp_c_z = lp_c - np.mean(lp_c)
                h_c_z = h_c - np.mean(h_c)
                X_quad_c = np.column_stack([
                    np.ones_like(sc_c), sc_c, lp_c_z, h_c_z,
                    lp_c_z ** 2, h_c_z ** 2, lp_c_z * h_c_z
                ])
                res_quad_cell = fit_ols_hc3(X_quad_c, y_c)

            suite[m] = {
                "run_level_linear": res_lin_run,
                "cell_means_linear": res_lin_cell,
                "cell_means_quadratic": res_quad_cell,
            }
        return suite

    orig_9_elfm = {
        (1.0, 1.5), (1.0, 2.0), (4.0, 0.5), (4.0, 1.0),
        (3.0, 0.0), (3.0, 0.5), (3.0, 1.0), (3.0, 1.5), (3.0, 2.0),
    }
    all_elfb_runs_2seed = [
        r for rid, r in runs.items()
        if rid.startswith("p4_elfb_s32_") and _seed_of(rid, r) in (0, 1)
    ]
    all_elfb_runs = [r for rid, r in runs.items() if rid.startswith("p4_elfb_s32_")]
    orig_elfm_runs = [
        r for rid, r in runs.items()
        if rid.startswith(("p4_elfm_s64_", "p5_elfm_s64_"))
        and (float(r["sc_cfg"]), float(r["gamma"])) in orig_9_elfm
        and _seed_of(rid, r) in (0, 1)
    ]
    all_elfm_runs = [r for rid, r in runs.items() if rid.startswith(("p4_elfm_s64_", "p5_elfm_s64_"))]

    surface_elfb = run_surface_suite(all_elfb_runs_2seed)
    surface_elfb_all = run_surface_suite(all_elfb_runs)
    surface_elfm = run_surface_suite(orig_elfm_runs)
    surface_elfm_all = run_surface_suite(all_elfm_runs)

    # Apply Holm-Bonferroni and BH-FDR across the primary outcome family on cell-means linear regressions
    primary_family = ["rep4", "distinct1", "distinct2", "distinct3", "mauve"]
    for surf in (surface_elfb, surface_elfb_all, surface_elfm, surface_elfm_all):
        p_raw_cell = [surf[m]["cell_means_linear"]["p_ols_sc"] for m in primary_family if m in surf]
        p_holm_cell = holm_bonferroni(p_raw_cell)
        p_bh_cell = benjamini_hochberg(p_raw_cell)
        for idx, m in enumerate([m for m in primary_family if m in surf]):
            surf[m]["cell_means_linear"]["p_holm_5metrics"] = p_holm_cell[idx]
            surf[m]["cell_means_linear"]["q_bh_5metrics"] = p_bh_cell[idx]

    return {
        "elfb_balanced_4x7x2_anova": anova_results,
        "elfb_noncollapsed_4x6x2_anova": anova_4x6_results,
        "elfm_balanced_2x2x5_anova": anova_elfm_2x2x5,
        "elfb_2x2_subgrids": subgrids_2x2,
        "conditional_surface_elfb_60runs_30cells": surface_elfb,
        "conditional_surface_elfb_78runs_30cells": surface_elfb_all,
        "conditional_surface_elfm_13runs_9cells": surface_elfm,
        "conditional_surface_elfm_29runs_11cells": surface_elfm_all,
    }


def main():
    runs = load_all_raw_runs()
    print(f"[statistical_reanalysis] Loaded {len(runs)} raw run directories.")

    # 1. Audit definitions, JSD weighting, and NFE
    def_audit = audit_definitions_and_nfe(runs)

    # 2. Detailed matched-pair comparisons (2-seed baseline and 5-seed P6 expansion)
    def get_seeds(prefix: str, seeds=(0, 1)) -> List[dict]:
        return [runs[f"{prefix}_seed{s}"] for s in seeds if f"{prefix}_seed{s}" in runs]

    pair_prefixes = [
        ("ELF-M_M1_SC1g2.0_vs_SC4g0.5", "p5_elfm_s64_g2.0_sc1.0", "p5_elfm_s64_g0.5_sc4.0"),
        ("ELF-B_S1_strict_SC1g2.0_vs_SC4g1.5", "p4_elfb_s32_g2.0_sc1.0", "p4_elfb_s32_g1.5_sc4.0"),
        ("ELF-B_S2_strict_SC1g1.5_vs_SC3g1.0", "p4_elfb_s32_g1.5_sc1.0", "p4_elfb_s32_g1.0_sc3.0"),
        ("ELF-B_S3_strict_SC2g1.0_vs_SC4g0.5", "p4_elfb_s32_g1.0_sc2.0", "p4_elfb_s32_g0.5_sc4.0"),
        ("ELF-B_E3a_refined_SC1g1.25_vs_SC3g0.5", "p4_elfb_s32_g1.25_sc1.0", "p4_elfb_s32_g0.5_sc3.0"),
        ("ELF-B_E3b_refined_SC1g1.75_vs_SC4g1.0", "p4_elfb_s32_g1.75_sc1.0", "p4_elfb_s32_g1.0_sc4.0"),
        ("ELF-B_C1_close_SC1g1.5_vs_SC4g1.0", "p4_elfb_s32_g1.5_sc1.0", "p4_elfb_s32_g1.0_sc4.0"),
        ("ELF-B_C2_close_SC1g1.0_vs_SC3g0.5", "p4_elfb_s32_g1.0_sc1.0", "p4_elfb_s32_g0.5_sc3.0"),
        ("ELF-B_C3_close_SC1g1.0_vs_SC4g0.5", "p4_elfb_s32_g1.0_sc1.0", "p4_elfb_s32_g0.5_sc4.0"),
        ("ELF-B_C4_close_SC2g1.0_vs_SC3g0.5", "p4_elfb_s32_g1.0_sc2.0", "p4_elfb_s32_g0.5_sc3.0"),
        ("NEGATIVE_CONTROL_same_SC1_g1.0_vs_g2.0", "p4_elfb_s32_g1.0_sc1.0", "p4_elfb_s32_g2.0_sc1.0"),
        ("UNMATCHED_CONTROL_same_g1.0_SC1_vs_SC4", "p4_elfb_s32_g1.0_sc1.0", "p4_elfb_s32_g1.0_sc4.0"),
    ]

    pair_results = {}
    pair_results_5seed = {}
    for label, pa, pb in pair_prefixes:
        ra_2 = get_seeds(pa, (0, 1))
        rb_2 = get_seeds(pb, (0, 1))
        pair_results[label] = compute_pair_tost_and_stats(ra_2, rb_2, label)

        ra_all = get_seeds(pa, (0, 1, 2, 3, 4))
        rb_all = get_seeds(pb, (0, 1, 2, 3, 4))
        n_common = min(len(ra_all), len(rb_all))
        pair_results_5seed[label] = compute_pair_tost_and_stats(ra_all[:n_common], rb_all[:n_common], label)
        if n_common > 2:
            pair_results[label]["five_seed_expansion"] = pair_results_5seed[label]

    primary_pairs = [
        "ELF-M_M1_SC1g2.0_vs_SC4g0.5",
        "ELF-B_S1_strict_SC1g2.0_vs_SC4g1.5",
        "ELF-B_S2_strict_SC1g1.5_vs_SC3g1.0",
        "ELF-B_S3_strict_SC2g1.0_vs_SC4g0.5",
        "ELF-B_E3a_refined_SC1g1.25_vs_SC3g0.5",
        "ELF-B_E3b_refined_SC1g1.75_vs_SC4g1.0",
    ]
    five_seed_pairs = [
        "ELF-M_M1_SC1g2.0_vs_SC4g0.5",
        "ELF-B_S1_strict_SC1g2.0_vs_SC4g1.5",
        "ELF-B_E3a_refined_SC1g1.25_vs_SC3g0.5",
        "ELF-B_E3b_refined_SC1g1.75_vs_SC4g1.0",
    ]
    for m in ["rep4", "h_pool", "jsd_inter_sample", "within_distinct2", "distinct2", "self_bleu", "jcd_bits_per_byte", "mauve"]:
        # 2-seed Holm/BH across 6 primary pairs
        p_raw = [pair_results[pk]["metrics"][m]["p_welch_2seed"] for pk in primary_pairs]
        p_holm = holm_bonferroni(p_raw)
        p_bh = benjamini_hochberg(p_raw)
        for idx, pk in enumerate(primary_pairs):
            pair_results[pk]["metrics"][m]["p_holm_6pairs"] = p_holm[idx]
            pair_results[pk]["metrics"][m]["q_bh_6pairs"] = p_bh[idx]

        # 5-seed expanded Holm/BH across all 6 primary pairs (4 with n=5, 2 with n=2)
        p_raw_5 = [pair_results_5seed[pk]["metrics"][m]["p_welch_2seed"] for pk in primary_pairs]
        p_holm_5 = holm_bonferroni(p_raw_5)
        p_bh_5 = benjamini_hochberg(p_raw_5)
        for idx, pk in enumerate(primary_pairs):
            pair_results_5seed[pk]["metrics"][m]["p_holm_6pairs"] = p_holm_5[idx]
            pair_results_5seed[pk]["metrics"][m]["q_bh_6pairs"] = p_bh_5[idx]

        # 5-seed Holm/BH across the 4 five-seed pairs
        p_raw_4 = [pair_results_5seed[pk]["metrics"][m]["p_welch_2seed"] for pk in five_seed_pairs]
        p_holm_4 = holm_bonferroni(p_raw_4)
        for idx, pk in enumerate(five_seed_pairs):
            pair_results_5seed[pk]["metrics"][m]["p_holm_4pairs_5seed"] = p_holm_4[idx]

    # 3. Tolerance sensitivity
    tol_sens = sensitivity_matching_tolerances(runs)

    # 4. Factorial ANOVA and surface regressions
    fact_and_surf = design_based_factorial_and_surface_models(runs)

    # 5. Reference corpus statistics (R0_owt_ref)
    owt_ref = runs.get("R0_owt_ref", {})

    full_report = {
        "definition_and_nfe_audit": def_audit,
        "matched_pairs_and_controls": pair_results,
        "matched_pairs_5seed_expanded": pair_results_5seed,
        "tolerance_sensitivity": tol_sens,
        "factorial_and_surface_models": fact_and_surf,
        "owt_reference_summary": owt_ref,
    }

    out_path = OUT_DIR / "reanalysis_stats.json"
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"[statistical_reanalysis] Wrote {out_path}")

    # Build machine-readable run_manifest.json at repository root
    manifest = {
        "project": "ELF Continuous-Latent Sampler-Control Evaluation Study",
        "target_venue": "ACL 2027 Main Conference (ARR Double-Blind)",
        "hierarchy": "model/checkpoint -> experiment configuration -> independent run/seed -> generated samples -> metrics -> statistical analysis -> table/figure -> manuscript claim",
        "models": {
            "ELF-S": {"params_M": 43.3, "checkpoint_step": 350000, "backbone": "DiT + frozen T5-Base encoder (d=768)"},
            "ELF-B": {"params_M": 105.0, "checkpoint_step": 182985, "backbone": "DiT + frozen T5-Base encoder (d=768)"},
            "ELF-M": {"params_M": 342.5, "checkpoint_step": 152140, "backbone": "DiT + frozen T5-Large encoder (d=1024)"},
            "MDLM": {"params_M": 169.6, "checkpoint": "kuleshov-group/mdlm-owt", "backbone": "Absorbing-state discrete diffusion"},
            "GPT-2": {"params_M": 124.4, "checkpoint": "gpt2", "backbone": "Autoregressive Transformer"},
            "Evaluator": {"params_M": 774.0, "checkpoint": "gpt2-large", "precision": "bfloat16 on CUDA, context=1024"},
        },
        "primary_experiment_families": {
            "P4_ELFB_Factorial": {
                "description": "Balanced 4 x 7 x 2 factorial grid on ELF-B (32 steps, SC-CFG in {1,2,3,4}, gamma in {0,0.25,0.5,1.0,1.5,2.0,3.0}, 2 seeds/cell, 1000 samples/seed)",
                "n_runs": 56,
                "n_cells": 28,
                "per_sample_nfe": 33,
            },
            "E3_ELFB_Refinement": {
                "description": "Targeted midpoint refinement on ELF-B (32 steps, SC-CFG=1.0, gamma in {1.25, 1.75}, 2 baseline seeds + 3 P6 expansion seeds = 5 seeds/cell, 1000 samples/seed)",
                "n_runs": 10,
                "n_cells": 2,
                "per_sample_nfe": 33,
            },
            "P6_ELFB_5Seed_Expansion": {
                "description": "5-seed replication expansion (seeds 2,3,4, 1000 samples/seed) on ELF-B matched-pair cells S1, Refined-1 (E3a), and Refined-2 (E3b)",
                "n_new_runs": 18,
                "n_cells_expanded_to_5_seeds": 6,
                "per_sample_nfe": 33,
            },
            "E1_P5_P6_ELFM_Factorial": {
                "description": "ELF-M 64-step evaluation across 11 (SC-CFG, gamma) cells (29 runs total), including the complete balanced 2 x 2 x 5 factorial design on SC-CFG in {1.0, 4.0} x gamma in {0.5, 2.0} (20 runs, 5 seeds/cell, 1000 samples/seed), SC=1.0 gamma=1.5 and SC=4.0 gamma=1.0 (2 seeds/cell), and SC=3.0 gamma in {0,0.5,1.0,1.5,2.0} (1 seed/cell)",
                "n_runs": 29,
                "n_cells": 11,
                "per_sample_nfe": 65,
            },
            "E2_P6_Extended_Rescore": {
                "description": "Full extended metric rescoring (H_pool, JSD_inter, within/cross distinct-1..4, 4-gram Self-BLEU, depth-6 CTW forward/reverse/Jeffreys coding divergence, position-wise entropy quartiles) on 64 runs across ELF-M and ELF-B",
                "n_runs": 64,
            },
        },
        "total_logged_runs_in_db": len(runs),
        "reproduction_commands": {
            "1_integrate_raw_artifacts": "python3 analysis/integrate_followup_runs.py",
            "2_statistical_reanalysis": "python3 analysis/statistical_reanalysis.py",
            "3_check_manuscript_claims": "python3 analysis/check_manuscript.py",
            "4_run_unit_tests": "python3 tests/run_tests.py",
            "5_compile_manuscript": "pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex",
        },
    }
    manifest_path = ROOT / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[statistical_reanalysis] Wrote {manifest_path}")


if __name__ == "__main__":
    main()
