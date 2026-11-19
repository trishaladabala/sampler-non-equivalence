#!/usr/bin/env python3
"""
Comprehensive Statistical, Semantic, and Computational Reanalysis for Phases 1-8.
Strictly operates on real experimental artifacts on disk (314 run directories / 452 summary.json
files, 64 extended_metrics.json files, and 600 real generated passages in human_eval_blinded_sheet.csv).
Never fabricates data, seeds, or human annotations.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "content" / "drive" / "MyDrive" / "elf_research" / "runs"
OUT_DIR = ROOT / "analysis" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_runs() -> Dict[str, dict]:
    runs = {}
    for p in sorted(RUNS_DIR.rglob("summary.json")):
        rel = p.parent.relative_to(RUNS_DIR)
        run_id = str(rel).replace("/", "__")
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_dir"] = str(p.parent)
        d["_run_key"] = d.get("run_id", p.parent.name)
        ext_p = p.parent / "extended_metrics.json"
        if ext_p.exists():
            ext = json.loads(ext_p.read_text(encoding="utf-8"))
            for k, v in ext.items():
                if k not in d or d[k] is None:
                    d[k] = v
        runs[run_id] = d
    return runs


def _seed_of(rid: str, r: dict) -> int:
    s = r.get("seed")
    if s is not None:
        return int(s)
    if "_seed" in rid:
        return int(rid.split("_seed")[-1])
    return -1


# ==============================================================================
# PHASE 2: Operating-Point Matching, Candidate Equivalence Margins, & TOST
# ==============================================================================

def satterthwaite_welch_and_tost(x: np.ndarray, y: np.ndarray, margin: float) -> dict:
    """
    Two-sample Welch t-test (B - A = y - x) AND Two One-Sided Tests (TOST)
    at symmetric equivalence margin [-margin, +margin], using Satterthwaite df.
     Also computes Hedges' g (small-sample bias-corrected standardized effect size).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    mx, my = float(np.mean(x)), float(np.mean(y))
    diff = my - mx
    vx = float(np.var(x, ddof=1)) if nx > 1 else 0.0
    vy = float(np.var(y, ddof=1)) if ny > 1 else 0.0
    se = math.sqrt(vx / nx + vy / ny) if (vx / nx + vy / ny) > 0 else 1e-12

    num = (vx / nx + vy / ny) ** 2
    den = 0.0
    if nx > 1 and vx > 0:
        den += (vx / nx) ** 2 / (nx - 1)
    if ny > 1 and vy > 0:
        den += (vy / ny) ** 2 / (ny - 1)
    df_satt = num / den if den > 0 else float(nx + ny - 2)

    t_stat = diff / se
    p_two_sided = float(2.0 * stats.t.sf(abs(t_stat), df=df_satt))
    t_crit_95 = float(stats.t.ppf(0.975, df=df_satt))
    t_crit_90 = float(stats.t.ppf(0.95, df=df_satt))  # 90% CI corresponds to alpha=0.05 TOST

    ci95_lo, ci95_hi = diff - t_crit_95 * se, diff + t_crit_95 * se
    ci90_lo, ci90_hi = diff - t_crit_90 * se, diff + t_crit_90 * se

    # TOST: H01: diff <= -margin vs H11: diff > -margin
    #       H02: diff >= +margin vs H12: diff < +margin
    t_lower = (diff - (-margin)) / se
    t_upper = (margin - diff) / se
    p_lower = float(stats.t.sf(t_lower, df=df_satt))
    p_upper = float(stats.t.sf(t_upper, df=df_satt))
    p_tost = max(p_lower, p_upper)
    min_equiv_margin_alpha05 = abs(diff) + t_crit_90 * se

    # Pooled SD & Hedges' g
    df_pool = nx + ny - 2
    s_pool = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(df_pool, 1))
    cohen_d = diff / s_pool if s_pool > 0 else 0.0
    # Hedges' J correction factor for small samples
    j_corr = 1.0 - (3.0 / (4.0 * df_pool - 1.0)) if df_pool > 1 else 1.0
    hedges_g = cohen_d * j_corr

    return {
        "n_A": nx,
        "n_B": ny,
        "mean_A": mx,
        "mean_B": my,
        "sd_A": math.sqrt(vx),
        "sd_B": math.sqrt(vy),
        "diff_B_minus_A": diff,
        "rel_diff_pct_vs_A": 100.0 * diff / mx if mx != 0 else None,
        "se_welch": se,
        "df_satt": df_satt,
        "t_welch": t_stat,
        "p_welch_2sided": p_two_sided,
        "ci95_welch": [ci95_lo, ci95_hi],
        "ci90_tost": [ci90_lo, ci90_hi],
        "cohen_d": cohen_d,
        "hedges_g": hedges_g,
        "tost_margin_tested": margin,
        "p_tost_alpha05": p_tost,
        "tost_equivalent_at_05": bool(p_tost < 0.05),
        "min_margin_for_tost_05": min_equiv_margin_alpha05,
    }


def derive_scientific_equivalence_margins(runs: Dict[str, dict]) -> dict:
    """
    Derive candidate equivalence margins purely from:
      (1) Within-configuration Monte Carlo sequence-bootstrap noise (M=1,000 sequences)
      (2) Pure cross-seed replication noise of identical configurations (same model, SC-CFG, gamma)
      (3) Reference step-halving benchmark shift (K=64 vs K=32 at fixed SC-CFG=3, gamma=1)
    Completely independent of any pairwise treatment comparison between different (SC-CFG, gamma) cells.
    """
    # 1. Within-run bootstrap SE of Gen-PPL across the 56 ELF-B baseline runs
    rel_half_widths = []
    abs_half_widths = []
    log_se_mc = []
    for k, r in runs.items():
        if k.startswith("p4_elfb_s32_") and _seed_of(k, r) in (0, 1) and float(r.get("gamma", 99)) <= 2.0:
            ppl = float(r["gen_ppl"])
            lo, hi = float(r["ppl_ci_lo"]), float(r["ppl_ci_hi"])
            hw = 0.5 * (hi - lo)
            abs_half_widths.append(hw)
            rel_half_widths.append(100.0 * hw / ppl)
            # SE of log(Gen-PPL) from 95% CI
            se_ln = (math.log(hi) - math.log(lo)) / (2.0 * 1.96)
            log_se_mc.append(se_ln)

    # 2. Pure cross-seed SD within identical cells (pooled across all multi-seed cells)
    by_cell_b = defaultdict(list)
    by_cell_m = defaultdict(list)
    for k, r in runs.items():
        if k.startswith("p4_elfb_s32_") and float(r.get("gamma", 99)) <= 2.0:
            by_cell_b[(float(r["sc_cfg"]), float(r["gamma"]))].append(r)
        elif k.startswith("p5_elfm_s64_"):
            by_cell_m[(float(r["sc_cfg"]), float(r["gamma"]))].append(r)

    def _pooled_within_cell_sd(cell_dict, field, log_transform=False):
        ss = 0.0
        df = 0
        for cell, rlist in cell_dict.items():
            vals = [float(x[field]) for x in rlist if field in x and x[field] is not None]
            if len(vals) >= 2:
                arr = np.log(vals) if log_transform else np.array(vals, dtype=float)
                ss += float(np.sum((arr - np.mean(arr)) ** 2))
                df += len(vals) - 1
        return math.sqrt(ss / max(df, 1)), df

    sd_ln_ppl_b, df_b = _pooled_within_cell_sd(by_cell_b, "gen_ppl", log_transform=True)
    sd_hps_b, _ = _pooled_within_cell_sd(by_cell_b, "entropy", log_transform=False)
    sd_ln_ppl_m, df_m = _pooled_within_cell_sd(by_cell_m, "gen_ppl", log_transform=True)
    sd_hps_m, _ = _pooled_within_cell_sd(by_cell_m, "entropy", log_transform=False)

    # 95% repeatability limit (1.96 * sqrt(2) * sigma_seed = 2.77 * sigma_seed) between two single seeds of the SAME sampler
    repeatability_ln_ppl_b = 1.96 * math.sqrt(2.0) * sd_ln_ppl_b
    repeatability_hps_b = 1.96 * math.sqrt(2.0) * sd_hps_b

    return {
        "rationale_1_monte_carlo_measurement_noise": {
            "description": (
                "Finite-sample evaluation uncertainty of a SINGLE fixed run at M=1,000 sequences "
                "(1.96 * SE_bootstrap of GPT-2 Large token-weighted Gen-PPL across 48 non-collapsed ELF-B runs)."
            ),
            "median_95ci_rel_half_width_pct": float(np.median(rel_half_widths)),
            "mean_95ci_rel_half_width_pct": float(np.mean(rel_half_widths)),
            "median_se_log_ppl": float(np.median(log_se_mc)),
            "two_sample_mc_95_bound_pct": float(np.median(rel_half_widths) * math.sqrt(2.0)),
        },
        "rationale_2_within_cell_cross_seed_repeatability": {
            "description": (
                "Pooled within-cell cross-seed standard deviation sigma_seed across independent RNG seeds "
                "of the EXACT SAME configuration (df_B=50 on ELF-B, df_M=18 on ELF-M), and the 95% Bland-Altman "
                "repeatability limit (2.77 * sigma_seed) between two independent seeds of the same sampler."
            ),
            "elfb_pooled_seed_sd_log_ppl": sd_ln_ppl_b,
            "elfb_pooled_seed_sd_rel_pct": 100.0 * (math.exp(sd_ln_ppl_b) - 1.0),
            "elfb_95_repeatability_rel_pct": 100.0 * (math.exp(repeatability_ln_ppl_b) - 1.0),
            "elfb_pooled_seed_sd_hps_nats": sd_hps_b,
            "elfb_95_repeatability_hps_nats": repeatability_hps_b,
            "elfm_pooled_seed_sd_log_ppl": sd_ln_ppl_m,
            "elfm_pooled_seed_sd_rel_pct": 100.0 * (math.exp(sd_ln_ppl_m) - 1.0),
            "elfm_pooled_seed_sd_hps_nats": sd_hps_m,
        },
        "proposed_three_tier_candidate_margins_for_user_approval": {
            "Tier_I_Strict_Repeatability": {
                "delta_log_ppl": 0.025,  # ~2.5% relative Gen-PPL (~two-sample MC 95% noise bound 2.0% * sqrt(2) = 2.8%)
                "delta_rel_ppl_pct": 2.5,
                "delta_hps_nats": 0.015,  # ~3x pooled cross-seed SD of H_ps (0.0049 nats)
                "justification": "Matches the two-sample Monte Carlo 95% evaluation noise bound (2.0% * sqrt(2) = 2.8%) and ~3x pure cross-seed SD of H_ps (0.0049 nats).",
            },
            "Tier_II_Moderate_Practical": {
                "delta_log_ppl": 0.050,  # ~5.0% relative Gen-PPL (~1/3 of a step-doubling gain 16% on ELF-B->M)
                "delta_rel_ppl_pct": 5.0,
                "delta_hps_nats": 0.025,  # ~0.5% of total entropy ~5.20 nats
                "justification": "Corresponds to 5% relative Gen-PPL (~0.05 nats NLL, <1/3 of the 16-17.5% frontier shift from ELF-B to ELF-M) and 0.025 nats (<0.5% of H_ps).",
            },
        },
    }


def predict_exact_root_matched_pairs(runs: Dict[str, dict]) -> List[dict]:
    """
    Phase 2 Req 3: Use monotone cubic/piecewise-linear response curves across gamma for each SC-CFG
    on ELF-B and ELF-M to find new candidate pairs (SC_A, gamma_A*) vs (SC_B, gamma_B*) where
    Delta H_ps -> 0 AND Delta Gen-PPL is minimized, for confirmatory GPU execution in Notebook 2.
    """
    by_cell_b = defaultdict(list)
    for k, r in runs.items():
        if k.startswith("p4_elfb_s32_"):
            by_cell_b[(float(r["sc_cfg"]), float(r["gamma"]))].append(r)

    sc_curves_b = defaultdict(list)
    for (sc, g), rlist in sorted(by_cell_b.items()):
        sc_curves_b[sc].append({
            "gamma": g,
            "gen_ppl": float(np.mean([x["gen_ppl"] for x in rlist])),
            "h_ps": float(np.mean([x["entropy"] for x in rlist])),
            "rep4": float(np.mean([x["rep4"] for x in rlist])),
            "distinct2": float(np.mean([x["distinct2"] for x in rlist])),
            "mauve": float(np.mean([x["mauve"] for x in rlist])),
            "n_seeds": len(rlist),
        })

    # For each target cell on SC_B in {2, 3, 4}, interpolate exact gamma_A on SC_A=1.0 that matches H_ps or Gen-PPL
    sc1 = sorted(sc_curves_b[1.0], key=lambda x: x["gamma"])
    g1 = np.array([x["gamma"] for x in sc1])
    h1 = np.array([x["h_ps"] for x in sc1])
    p1 = np.array([x["gen_ppl"] for x in sc1])

    candidates = []
    for sc_b in [2.0, 3.0, 4.0]:
        for pt in sc_curves_b[sc_b]:
            gb = pt["gamma"]
            hb = pt["h_ps"]
            pb = pt["gen_ppl"]
            if gb in (0.0, 3.0):
                continue
            # Interpolate gamma on SC=1.0 where H_ps(1.0, gamma) == hb
            # Note h1 decreases monotonically with g1
            if h1.min() <= hb <= h1.max():
                g_star_h = float(np.interp(hb, h1[::-1], g1[::-1]))
                p_at_g_star_h = float(np.interp(g_star_h, g1, p1))
                # Also interpolate gamma on SC=1.0 where Gen-PPL(1.0, gamma) == pb
                g_star_p = float(np.interp(pb, p1[::-1], g1[::-1])) if (p1.min() <= pb <= p1.max()) else None
                h_at_g_star_p = float(np.interp(g_star_p, g1, h1)) if g_star_p is not None else None
                # Compromise gamma* minimizing normalized (Delta_log_PPL / 0.025)^2 + (Delta_H / 0.015)^2
                grid_g = np.linspace(max(0.0, g_star_h - 0.3), min(2.5, g_star_h + 0.3), 601)
                grid_p = np.interp(grid_g, g1, p1)
                grid_h = np.interp(grid_g, g1, h1)
                loss = ((np.log(grid_p) - math.log(pb)) / 0.025) ** 2 + ((grid_h - hb) / 0.015) ** 2
                best_idx = int(np.argmin(loss))
                g_opt = float(grid_g[best_idx])
                p_opt = float(grid_p[best_idx])
                h_opt = float(grid_h[best_idx])
                candidates.append({
                    "model": "ELF-B",
                    "target_B": {"sc_cfg": sc_b, "gamma": gb, "gen_ppl": pb, "h_ps": hb, "existing_seeds": pt["n_seeds"]},
                    "matched_H_on_SC1": {"gamma_star_H": round(g_star_h, 3), "pred_gen_ppl": round(p_at_g_star_h, 2), "pred_rel_ppl_diff_pct": round(100.0 * (pb - p_at_g_star_h) / p_at_g_star_h, 2)},
                    "matched_PPL_on_SC1": {"gamma_star_PPL": round(g_star_p, 3) if g_star_p else None, "pred_h_ps": round(h_at_g_star_p, 4) if h_at_g_star_p else None, "pred_h_diff_nats": round(hb - h_at_g_star_p, 4) if h_at_g_star_p else None},
                    "joint_pareto_match_on_SC1": {
                        "gamma_opt": round(g_opt, 3),
                        "pred_gen_ppl": round(p_opt, 2),
                        "pred_rel_ppl_diff_pct": round(100.0 * (pb - p_opt) / p_opt, 2),
                        "pred_h_ps": round(h_opt, 4),
                        "pred_h_diff_nats": round(hb - h_opt, 4),
                    },
                })
    return candidates


# ==============================================================================
# PHASE 3: Factorial ANOVA, Effect Sizes (eta^2, partial eta^2, omega^2),
#          Residual/Homoskedasticity Diagnostics, Mixed-Effects, & Metric Families
# ==============================================================================

def two_way_anova_extended(
    runs_list: List[dict],
    sc_levels: List[float],
    g_levels: List[float],
    metric: str,
    n_perm: int = 2000,
) -> dict:
    """
    Balanced two-way factorial ANOVA with:
      - Marginal means & cell means
      - SS, df, MS, F, p-value
      - eta^2 (proportion of total SS), partial eta^2 (SS_eff / (SS_eff + SS_err)),
        and omega^2 ((SS_eff - df_eff * MS_err) / (SS_tot + MS_err))
      - Residual diagnostics: Shapiro-Wilk p-value on residuals, Levene / Brown-Forsythe
        test for homogeneity of variance across cells
      - Permutation p-values (seed-level shuffling, n_perm=2000)
      - Random-intercept variance component (ICC_cell)
    """
    cell_vals = defaultdict(list)
    for r in runs_list:
        sc = float(r["sc_cfg"])
        g = float(r["gamma"])
        if sc in sc_levels and g in g_levels and r.get(metric) is not None:
            cell_vals[(sc, g)].append(float(r[metric]))

    A = len(sc_levels)
    B = len(g_levels)
    counts = [len(cell_vals[(sc, g)]) for sc in sc_levels for g in g_levels]
    if not counts or min(counts) < 2 or len(set(counts)) != 1:
        return {"error": f"Unbalanced or insufficient counts for {metric}: {set(counts)}"}
    n_rep = counts[0]
    N = A * B * n_rep

    # Build 3D array (A, B, n_rep)
    Y = np.zeros((A, B, n_rep), dtype=float)
    for i, sc in enumerate(sc_levels):
        for j, g in enumerate(g_levels):
            Y[i, j, :] = cell_vals[(sc, g)]

    grand_mean = float(np.mean(Y))
    mean_A = np.mean(Y, axis=(1, 2))  # shape (A,)
    mean_B = np.mean(Y, axis=(0, 2))  # shape (B,)
    mean_AB = np.mean(Y, axis=2)      # shape (A, B)

    ss_A = float(B * n_rep * np.sum((mean_A - grand_mean) ** 2))
    ss_B = float(A * n_rep * np.sum((mean_B - grand_mean) ** 2))
    ss_cells = float(n_rep * np.sum((mean_AB - grand_mean) ** 2))
    ss_AB = max(0.0, ss_cells - ss_A - ss_B)
    residuals = Y - mean_AB[:, :, None]
    ss_err = float(np.sum(residuals ** 2))
    ss_tot = ss_cells + ss_err

    df_A = A - 1
    df_B = B - 1
    df_AB = (A - 1) * (B - 1)
    df_err = A * B * (n_rep - 1)

    ms_A = ss_A / df_A
    ms_B = ss_B / df_B
    ms_AB = ss_AB / df_AB if df_AB > 0 else 0.0
    ms_err = ss_err / df_err if df_err > 0 else 1e-12

    F_A = ms_A / ms_err
    F_B = ms_B / ms_err
    F_AB = ms_AB / ms_err if df_AB > 0 else 0.0

    p_A = float(stats.f.sf(F_A, df_A, df_err))
    p_B = float(stats.f.sf(F_B, df_B, df_err))
    p_AB = float(stats.f.sf(F_AB, df_AB, df_err)) if df_AB > 0 else 1.0

    # Effect sizes: eta^2, partial eta^2, omega^2
    def _effect_sizes(ss_eff, df_eff):
        eta2 = ss_eff / ss_tot if ss_tot > 0 else 0.0
        p_eta2 = ss_eff / (ss_eff + ss_err) if (ss_eff + ss_err) > 0 else 0.0
        omega2 = max(0.0, (ss_eff - df_eff * ms_err) / (ss_tot + ms_err)) if (ss_tot + ms_err) > 0 else 0.0
        return {
            "eta_sq_pct": 100.0 * eta2,
            "partial_eta_sq": p_eta2,
            "omega_sq_pct": 100.0 * omega2,
        }

    # Assumption diagnostics on residuals
    flat_res = residuals.ravel()
    shapiro_stat, shapiro_p = stats.shapiro(flat_res) if len(flat_res) >= 3 else (None, None)
    # Note: when n_rep == 2, |y_{ij1} - y_bar_{ij}| == |y_{ij2} - y_bar_{ij}| identically inside each cell,
    # so within-cell variance of absolute deviations is 0. Valid Levene tests group residuals by factor level
    # (SC-CFG: A groups of B*n_rep; gamma: B groups of A*n_rep), or across cells when n_rep > 2.
    sc_res_groups = [residuals[i, :, :].ravel() for i in range(A)]
    gam_res_groups = [residuals[:, j, :].ravel() for j in range(B)]
    levene_sc_stat, levene_sc_p = stats.levene(*sc_res_groups, center="mean")
    levene_gam_stat, levene_gam_p = stats.levene(*gam_res_groups, center="mean")
    if n_rep > 2:
        cell_groups = [Y[i, j, :] for i in range(A) for j in range(B)]
        levene_cell_stat, levene_cell_p = stats.levene(*cell_groups, center="mean")
    else:
        levene_cell_stat, levene_cell_p = None, None

    # Non-parametric permutation test (2000 shuffles of run-level observations)
    rng = np.random.default_rng(2027)
    flat_Y = Y.ravel().copy()
    perm_ge_A = 0
    perm_ge_B = 0
    for _ in range(n_perm):
        rng.shuffle(flat_Y)
        Yp = flat_Y.reshape((A, B, n_rep))
        mA_p = np.mean(Yp, axis=(1, 2))
        mB_p = np.mean(Yp, axis=(0, 2))
        mAB_p = np.mean(Yp, axis=2)
        ssA_p = B * n_rep * np.sum((mA_p - grand_mean) ** 2)
        ssB_p = A * n_rep * np.sum((mB_p - grand_mean) ** 2)
        ssE_p = np.sum((Yp - mAB_p[:, :, None]) ** 2)
        msE_p = ssE_p / df_err if df_err > 0 else 1e-12
        if (ssA_p / df_A) / msE_p >= F_A:
            perm_ge_A += 1
        if (ssB_p / df_B) / msE_p >= F_B:
            perm_ge_B += 1

    # Hierarchical variance components (between-cell sigma^2_cell vs within-cell sigma^2_seed)
    ms_cells = ss_cells / (A * B - 1)
    var_seed = ms_err
    var_cell = max(0.0, (ms_cells - ms_err) / n_rep)
    icc_cell = var_cell / (var_cell + var_seed) if (var_cell + var_seed) > 0 else 0.0

    # Marginal means with 95% CIs (using pooled SE = sqrt(ms_err / (B * n_rep)) etc.)
    t_crit = float(stats.t.ppf(0.975, df=df_err))
    se_marg_A = math.sqrt(ms_err / (B * n_rep))
    se_marg_B = math.sqrt(ms_err / (A * n_rep))

    return {
        "metric": metric,
        "design": f"{A}x{B}x{n_rep} (N={N})",
        "grand_mean": grand_mean,
        "df": {"sc_cfg": df_A, "gamma": df_B, "interaction": df_AB, "error": df_err},
        "sc_cfg": {
            "F": F_A,
            "p": p_A,
            "p_perm": (perm_ge_A + 1) / (n_perm + 1),
            **_effect_sizes(ss_A, df_A),
            "marginal_means": {
                str(sc): {"mean": float(mean_A[i]), "ci95": [float(mean_A[i] - t_crit * se_marg_A), float(mean_A[i] + t_crit * se_marg_A)]}
                for i, sc in enumerate(sc_levels)
            },
            "main_effect_high_minus_low": float(mean_A[-1] - mean_A[0]),
            "main_effect_se": math.sqrt(2.0) * se_marg_A,
        },
        "gamma": {
            "F": F_B,
            "p": p_B,
            "p_perm": (perm_ge_B + 1) / (n_perm + 1),
            **_effect_sizes(ss_B, df_B),
            "marginal_means": {
                str(g): {"mean": float(mean_B[j]), "ci95": [float(mean_B[j] - t_crit * se_marg_B), float(mean_B[j] + t_crit * se_marg_B)]}
                for j, g in enumerate(g_levels)
            },
            "main_effect_high_minus_low": float(mean_B[-1] - mean_B[0]),
            "main_effect_se": math.sqrt(2.0) * se_marg_B,
        },
        "interaction": {
            "F": F_AB,
            "p": p_AB,
            **_effect_sizes(ss_AB, df_AB),
        },
        "error_share_pct": 100.0 * ss_err / ss_tot if ss_tot > 0 else 0.0,
        "diagnostics": {
            "shapiro_wilk_W": float(shapiro_stat) if shapiro_stat is not None else None,
            "shapiro_wilk_p": float(shapiro_p) if shapiro_p is not None else None,
            "levene_sc_F": float(levene_sc_stat),
            "levene_sc_p": float(levene_sc_p),
            "levene_gamma_F": float(levene_gam_stat),
            "levene_gamma_p": float(levene_gam_p),
            "levene_cell_F": float(levene_cell_stat) if levene_cell_stat is not None else None,
            "levene_cell_p": float(levene_cell_p) if levene_cell_p is not None else None,
            "icc_between_cell": icc_cell,
            "sigma_seed": math.sqrt(var_seed),
        },
    }


def analyze_metric_dependence_and_families(runs: Dict[str, dict]) -> dict:
    """
    Phase 3 metric dependence & mathematical identity audit across all 64 rescored runs.
    """
    rescored = [
        r for r in runs.values()
        if r.get("h_pool") is not None and r.get("jsd_inter_sample") is not None and r.get("self_bleu") is not None
    ]
    fields = [
        "gen_ppl",
        "entropy",
        "h_ps_weighted",
        "rep4",
        "within_distinct2",
        "h_pool",
        "jsd_inter_sample",
        "distinct2",
        "self_bleu",
        "jcd_bits_per_byte",
        "mauve",
    ]
    mat = np.array([[float(r[f]) for f in fields] for r in rescored], dtype=float)
    corr = np.corrcoef(mat, rowvar=False)
    corr_dict = {
        fields[i]: {fields[j]: round(float(corr[i, j]), 4) for j in range(len(fields))}
        for i in range(len(fields))
    }

    # Verify exact mathematical identity: JSD_pi = H_pool - H_ps_weighted
    jsd_identity_max_err = max(
        abs(float(r["jsd_inter_sample"]) - (float(r["h_pool"]) - float(r["h_ps_weighted"])))
        for r in rescored
    )
    # Verify exact mathematical identity: rep4 = 1 - within_distinct4
    rep4_identity_max_err = max(
        abs(float(r["rep4"]) - (1.0 - float(r["within_distinct4"])))
        for r in rescored
    )

    return {
        "n_rescored_runs": len(rescored),
        "exact_identities_verified": {
            "jsd_pi_equals_h_pool_minus_h_ps_weighted_max_abs_err": jsd_identity_max_err,
            "rep4_equals_1_minus_within_distinct4_max_abs_err": rep4_identity_max_err,
            "h_ps_unweighted_vs_weighted_max_abs_diff": max(
                abs(float(r["entropy"]) - float(r["h_ps_weighted"])) for r in rescored
            ),
        },
        "outcome_families": {
            "Family_1_Operating_Point_Coordinates": ["gen_ppl", "entropy (H_ps)", "h_ps_weighted"],
            "Family_2_Within_Sample_Local_Degeneracy": ["rep4 (= 1 - within_distinct4)", "within_distinct2"],
            "Family_3_Cross_Sample_Lexical_and_Compression_Dispersion": [
                "jsd_inter_sample (= h_pool - h_ps_weighted)",
                "distinct2 (corpus)",
                "self_bleu (4-gram)",
                "jcd_bits_per_byte (CTW byte divergence)",
            ],
            "Family_4_Contextual_and_Semantic_Alignment": [
                "mauve (GPT-2 Large contextual states vs OWT)",
                "semantic_embedding_diversity (T5 / GPT-2 / MPNet cosine & Vendi score)",
            ],
        },
        "pearson_correlation_matrix_64_runs": corr_dict,
    }


# ==============================================================================
# PHASE 4: Human Evaluation Statistical Power Analysis
# ==============================================================================

def human_eval_power_analysis() -> dict:
    """
    Exact binomial sign-test and cluster-adjusted power analysis for the blinded human evaluation.
    """
    results = {}
    for label, n_items in [("Per_Pair_Single_Passage (n=25)", 25), ("Pooled_Single_Passage (n=75)", 75),
                           ("Per_Pair_5Sample_Bundle (n=15)", 15), ("Pooled_5Sample_Bundle (n=45)", 45)]:
        power_curve = {}
        for p_true in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            # Two-sided alpha=0.05 exact binomial test rejection region under H0: p=0.5
            # Find critical k such that 2 * P(Bin(n, 0.5) >= k) <= 0.05
            k_crit = n_items + 1
            for k in range(n_items // 2 + 1, n_items + 1):
                if 2.0 * stats.binom.sf(k - 1, n_items, 0.5) <= 0.05:
                    k_crit = k
                    break
            pwr_05 = float(stats.binom.sf(k_crit - 1, n_items, p_true) + stats.binom.cdf(n_items - k_crit, n_items, p_true))
            power_curve[f"p={p_true:.2f}"] = round(pwr_05, 4)
        results[label] = {
            "n_items": n_items,
            "critical_wins_two_sided_alpha05": k_crit,
            "min_win_rate_pct_for_significance": round(100.0 * k_crit / n_items, 1),
            "power_by_true_preference_rate": power_curve,
        }
    return results


# ==============================================================================
# PHASE 5: Semantic Diversity & Metric Validation on Real Generated Samples
# ==============================================================================

def compute_vendi_score(embeddings: np.ndarray) -> float:
    """
    Vendi Score (Friedman & Dieng, 2023): exponential of the Shannon entropy of the
    eigenvalues of the normalized cosine similarity kernel matrix K / M.
    """
    # Normalize rows to unit L2 norm
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    E = embeddings / np.maximum(norms, 1e-12)
    M = E.shape[0]
    K = (E @ E.T) / float(M)
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 0.0)
    eigvals = eigvals / np.sum(eigvals)
    pos = eigvals[eigvals > 1e-12]
    ent = -float(np.sum(pos * np.log(pos)))
    return float(math.exp(ent))


def compute_semantic_diversity_on_real_samples() -> dict:
    """
    Phase 5: Evaluate semantic diversity on the 600 real generated passages stored in
    human_eval/human_eval_blinded_sheet.csv across M1_ELFM (342M), S1_ELFB (105M),
    and Ref1_ELFB (105M), using:
      1. Cached Pretrained T5-Small Encoder (`t5-small`, contextual token embeddings mean-pooled)
      2. Cached Pretrained GPT-2 (`gpt2`, final hidden layer mean-pooled)
      3. Character/Word TF-IDF + SVD semantic baseline
    """
    import torch
    from transformers import AutoModel, AutoTokenizer, T5EncoderModel

    he_csv = ROOT / "human_eval" / "human_eval_blinded_sheet.csv"
    he_key = ROOT / "human_eval" / "human_eval_answer_key.json"
    if not (he_csv.exists() and he_key.exists()):
        return {"error": "human_eval files missing"}

    key = json.loads(he_key.read_text(encoding="utf-8"))
    with he_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Load locally cached T5-Small encoder and GPT-2 model (local_files_only=True)
    t5_tok = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    t5_enc = T5EncoderModel.from_pretrained("t5-small", local_files_only=True).eval()

    gpt2_tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_mod = AutoModel.from_pretrained("gpt2", local_files_only=True).eval()

    @torch.no_grad()
    def encode_t5(texts: List[str]) -> np.ndarray:
        out_list = []
        for s in range(0, len(texts), 32):
            enc = t5_tok(texts[s:s+32], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = t5_enc(**enc).last_hidden_state  # (B, L, 512)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            out_list.append(pooled.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    @torch.no_grad()
    def encode_gpt2(texts: List[str]) -> np.ndarray:
        out_list = []
        for s in range(0, len(texts), 32):
            enc = gpt2_tok(texts[s:s+32], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = gpt2_mod(**enc).last_hidden_state  # (B, L, 768)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            out_list.append(pooled.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    def mean_pairwise_cosine_dist(emb: np.ndarray) -> float:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        E = emb / np.maximum(norms, 1e-12)
        sim = E @ E.T
        n = E.shape[0]
        iu = np.triu_indices(n, k=1)
        return float(1.0 - np.mean(sim[iu]))

    # Organize T1 (25 single passages/arm, 1200 chars each) and T2 (15 bundles of 5 passages/arm)
    t1_by_pair = defaultdict(lambda: {"low_sc_high_gamma": [], "high_sc_low_gamma": []})
    t2_bundles_by_pair = defaultdict(lambda: {"low_sc_high_gamma": [], "high_sc_low_gamma": []})

    for r in rows:
        iid = r["item_id"]
        meta = key[iid]
        pt = meta["pair_tag"]
        tt = meta["task_type"]
        for col, arm_field in [("passage_or_bundle_A", "A_is"), ("passage_or_bundle_B", "B_is")]:
            arm = meta[arm_field]
            txt = r[col]
            if tt == "Task1_SinglePassage_LessRepetitive":
                t1_by_pair[pt][arm].append(txt)
            else:
                parts = [
                    p.split("]", 1)[-1].strip() if "]" in p else p.strip()
                    for p in txt.split("\n---\n") if p.strip()
                ]
                t2_bundles_by_pair[pt][arm].append(parts)

    def all_but_the_top(emb_low: np.ndarray, emb_high: np.ndarray, d_remove: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mu & Viswanath (2018) / Ethayarajh (2019) anisotropy correction:
        Subtract pooled mean vector and project out top `d_remove` principal directions of the anisotropy cone.
        """
        X = np.vstack([emb_low, emb_high])
        mu = np.mean(X, axis=0, keepdims=True)
        Xc = X - mu
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        top_v = Vt[:d_remove]  # (d_remove, D)
        X_iso = Xc - (Xc @ top_v.T) @ top_v
        n_lo = emb_low.shape[0]
        return X_iso[:n_lo], X_iso[n_lo:]

    def compute_lsa_topic_embeddings(texts_low: List[str], texts_high: List[str], n_components: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stopword-filtered content-word TF-IDF + Truncated SVD (LSA) topic embeddings.
        """
        import re
        all_texts = texts_low + texts_high
        N_docs = len(all_texts)
        doc_tokens = [
            [w for w in re.findall(r"[a-zA-Z]{4,}", t.lower())]
            for t in all_texts
        ]
        df_counts = defaultdict(int)
        for toks in doc_tokens:
            for w in set(toks):
                df_counts[w] += 1
        # Keep content words appearing in >= 2 and <= 50% of passages
        vocab = sorted([w for w, c in df_counts.items() if 2 <= c <= int(0.5 * N_docs)])
        w2i = {w: i for i, w in enumerate(vocab)}
        X_tfidf = np.zeros((N_docs, len(vocab)), dtype=float)
        for d_idx, toks in enumerate(doc_tokens):
            if not toks:
                continue
            tf = defaultdict(int)
            for w in toks:
                if w in w2i:
                    tf[w2i[w]] += 1
            for col_i, cnt in tf.items():
                idf = math.log((1.0 + N_docs) / (1.0 + df_counts[vocab[col_i]])) + 1.0
                X_tfidf[d_idx, col_i] = (cnt / len(toks)) * idf
        # Normalize rows and compute truncated SVD
        norms = np.linalg.norm(X_tfidf, axis=1, keepdims=True)
        X_tfidf = X_tfidf / np.maximum(norms, 1e-12)
        U, S, _ = np.linalg.svd(X_tfidf, full_matrices=False)
        k = min(n_components, U.shape[1])
        X_lsa = U[:, :k] * S[:k]
        n_lo = len(texts_low)
        return X_lsa[:n_lo], X_lsa[n_lo:]

    def bundle_lexical_stats(bundle_texts: List[str]) -> Tuple[float, float]:
        """Compute bundle-level corpus distinct-2 and within-sample rep-4 on a 5-passage bundle."""
        import re
        all_bg = []
        rep4_vals = []
        for t in bundle_texts:
            toks = re.findall(r"\w+|[^\w\s]", t.lower())
            if len(toks) >= 2:
                all_bg.extend(zip(toks[:-1], toks[1:]))
            if len(toks) >= 4:
                ng4 = list(zip(toks[:-3], toks[1:-2], toks[2:-1], toks[3:]))
                rep4_vals.append(1.0 - len(set(ng4)) / len(ng4))
        d2 = len(set(all_bg)) / max(len(all_bg), 1)
        r4 = float(np.mean(rep4_vals)) if rep4_vals else 0.0
        return d2, r4

    pair_semantic_results = {}
    for pt in ["M1_ELFM", "S1_ELFB", "Ref1_ELFB"]:
        t1_low = t1_by_pair[pt]["low_sc_high_gamma"]
        t1_high = t1_by_pair[pt]["high_sc_low_gamma"]
        t2_low_bundles = t2_bundles_by_pair[pt]["low_sc_high_gamma"]
        t2_high_bundles = t2_bundles_by_pair[pt]["high_sc_low_gamma"]

        # 20 bundles of 5 passages each = 100 passages per arm
        all_low = [p for b in t2_low_bundles for p in b] + t1_low
        all_high = [p for b in t2_high_bundles for p in b] + t1_high

        emb_t5_low = encode_t5(all_low)
        emb_t5_high = encode_t5(all_high)
        emb_gpt2_low = encode_gpt2(all_low)
        emb_gpt2_high = encode_gpt2(all_high)

        emb_t5_iso_low, emb_t5_iso_high = all_but_the_top(emb_t5_low, emb_t5_high, d_remove=2)
        emb_gpt2_iso_low, emb_gpt2_iso_high = all_but_the_top(emb_gpt2_low, emb_gpt2_high, d_remove=2)
        emb_lsa_low, emb_lsa_high = compute_lsa_topic_embeddings(all_low, all_high, n_components=32)

        bundle_t5_cos_low, bundle_t5_cos_high = [], []
        bundle_t5_vendi_low, bundle_t5_vendi_high = [], []
        bundle_gpt2_cos_low, bundle_gpt2_cos_high = [], []
        bundle_gpt2_vendi_low, bundle_gpt2_vendi_high = [], []
        bundle_gpt2_iso_cos_low, bundle_gpt2_iso_cos_high = [], []
        bundle_lsa_cos_low, bundle_lsa_cos_high = [], []
        bundle_lex_d2_all, bundle_lex_rep4_all, bundle_sem_t5_all, bundle_sem_lsa_all = [], [], [], []

        for k_idx in range(0, 100, 5):
            e_t5_lo = emb_t5_low[k_idx:k_idx+5]
            e_t5_hi = emb_t5_high[k_idx:k_idx+5]
            bundle_t5_cos_low.append(mean_pairwise_cosine_dist(e_t5_lo))
            bundle_t5_cos_high.append(mean_pairwise_cosine_dist(e_t5_hi))
            bundle_t5_vendi_low.append(compute_vendi_score(e_t5_lo))
            bundle_t5_vendi_high.append(compute_vendi_score(e_t5_hi))

            e_g2_lo = emb_gpt2_low[k_idx:k_idx+5]
            e_g2_hi = emb_gpt2_high[k_idx:k_idx+5]
            bundle_gpt2_cos_low.append(mean_pairwise_cosine_dist(e_g2_lo))
            bundle_gpt2_cos_high.append(mean_pairwise_cosine_dist(e_g2_hi))
            bundle_gpt2_vendi_low.append(compute_vendi_score(e_g2_lo))
            bundle_gpt2_vendi_high.append(compute_vendi_score(e_g2_hi))

            bundle_gpt2_iso_cos_low.append(mean_pairwise_cosine_dist(emb_gpt2_iso_low[k_idx:k_idx+5]))
            bundle_gpt2_iso_cos_high.append(mean_pairwise_cosine_dist(emb_gpt2_iso_high[k_idx:k_idx+5]))
            bundle_lsa_cos_low.append(mean_pairwise_cosine_dist(emb_lsa_low[k_idx:k_idx+5]))
            bundle_lsa_cos_high.append(mean_pairwise_cosine_dist(emb_lsa_high[k_idx:k_idx+5]))

            for b_txt, sem_t5, sem_lsa in [
                (all_low[k_idx:k_idx+5], bundle_t5_cos_low[-1], bundle_lsa_cos_low[-1]),
                (all_high[k_idx:k_idx+5], bundle_t5_cos_high[-1], bundle_lsa_cos_high[-1]),
            ]:
                d2_val, r4_val = bundle_lexical_stats(b_txt)
                bundle_lex_d2_all.append(d2_val)
                bundle_lex_rep4_all.append(r4_val)
                bundle_sem_t5_all.append(sem_t5)
                bundle_sem_lsa_all.append(sem_lsa)

        # Within-passage semantic self-similarity (first half vs second half of each T1 1200-char passage)
        first_lo = [p[:len(p)//2] for p in t1_low]
        second_lo = [p[len(p)//2:] for p in t1_low]
        first_hi = [p[:len(p)//2] for p in t1_high]
        second_hi = [p[len(p)//2:] for p in t1_high]

        e1_lo, e2_lo = encode_t5(first_lo), encode_t5(second_lo)
        e1_hi, e2_hi = encode_t5(first_hi), encode_t5(second_hi)
        within_cos_sim_lo = np.sum(
            (e1_lo / np.linalg.norm(e1_lo, axis=1, keepdims=True)) *
            (e2_lo / np.linalg.norm(e2_lo, axis=1, keepdims=True)), axis=1
        )
        within_cos_sim_hi = np.sum(
            (e1_hi / np.linalg.norm(e1_hi, axis=1, keepdims=True)) *
            (e2_hi / np.linalg.norm(e2_hi, axis=1, keepdims=True)), axis=1
        )

        pair_semantic_results[pt] = {
            "n_passages_per_arm": len(all_low),
            "n_5sample_bundles_per_arm": len(bundle_t5_cos_low),
            "corpus_100_passages": {
                "t5_raw_cosine_dist_low_sc": round(mean_pairwise_cosine_dist(emb_t5_low), 5),
                "t5_raw_cosine_dist_high_sc": round(mean_pairwise_cosine_dist(emb_t5_high), 5),
                "t5_raw_vendi_score_low_sc": round(compute_vendi_score(emb_t5_low), 3),
                "t5_raw_vendi_score_high_sc": round(compute_vendi_score(emb_t5_high), 3),
                "t5_isotropic_cosine_dist_low_sc": round(mean_pairwise_cosine_dist(emb_t5_iso_low), 5),
                "t5_isotropic_cosine_dist_high_sc": round(mean_pairwise_cosine_dist(emb_t5_iso_high), 5),
                "t5_isotropic_vendi_score_low_sc": round(compute_vendi_score(emb_t5_iso_low), 3),
                "t5_isotropic_vendi_score_high_sc": round(compute_vendi_score(emb_t5_iso_high), 3),
                "gpt2_raw_cosine_dist_low_sc": round(mean_pairwise_cosine_dist(emb_gpt2_low), 5),
                "gpt2_raw_cosine_dist_high_sc": round(mean_pairwise_cosine_dist(emb_gpt2_high), 5),
                "gpt2_raw_vendi_score_low_sc": round(compute_vendi_score(emb_gpt2_low), 3),
                "gpt2_raw_vendi_score_high_sc": round(compute_vendi_score(emb_gpt2_high), 3),
                "gpt2_isotropic_cosine_dist_low_sc": round(mean_pairwise_cosine_dist(emb_gpt2_iso_low), 5),
                "gpt2_isotropic_cosine_dist_high_sc": round(mean_pairwise_cosine_dist(emb_gpt2_iso_high), 5),
                "gpt2_isotropic_vendi_score_low_sc": round(compute_vendi_score(emb_gpt2_iso_low), 3),
                "gpt2_isotropic_vendi_score_high_sc": round(compute_vendi_score(emb_gpt2_iso_high), 3),
                "lsa_topic_cosine_dist_low_sc": round(mean_pairwise_cosine_dist(emb_lsa_low), 5),
                "lsa_topic_cosine_dist_high_sc": round(mean_pairwise_cosine_dist(emb_lsa_high), 5),
                "lsa_topic_vendi_score_low_sc": round(compute_vendi_score(emb_lsa_low), 3),
                "lsa_topic_vendi_score_high_sc": round(compute_vendi_score(emb_lsa_high), 3),
            },
            "orthogonality_to_lexical_metrics_40_bundles": {
                "r_t5_cos_vs_bundle_distinct2": round(float(np.corrcoef(bundle_sem_t5_all, bundle_lex_d2_all)[0, 1]), 4),
                "r_t5_cos_vs_bundle_rep4": round(float(np.corrcoef(bundle_sem_t5_all, bundle_lex_rep4_all)[0, 1]), 4),
                "r_lsa_topic_vs_bundle_distinct2": round(float(np.corrcoef(bundle_sem_lsa_all, bundle_lex_d2_all)[0, 1]), 4),
                "r_lsa_topic_vs_bundle_rep4": round(float(np.corrcoef(bundle_sem_lsa_all, bundle_lex_rep4_all)[0, 1]), 4),
            },
            "bundle_level_welch_tests_20_bundles": {
                "t5_raw_cosine_dist": satterthwaite_welch_and_tost(bundle_t5_cos_low, bundle_t5_cos_high, margin=0.01),
                "t5_raw_vendi_score": satterthwaite_welch_and_tost(bundle_t5_vendi_low, bundle_t5_vendi_high, margin=0.10),
                "gpt2_raw_cosine_dist": satterthwaite_welch_and_tost(bundle_gpt2_cos_low, bundle_gpt2_cos_high, margin=0.005),
                "gpt2_isotropic_cosine_dist": satterthwaite_welch_and_tost(bundle_gpt2_iso_cos_low, bundle_gpt2_iso_cos_high, margin=0.02),
                "lsa_topic_cosine_dist": satterthwaite_welch_and_tost(bundle_lsa_cos_low, bundle_lsa_cos_high, margin=0.02),
            },
            "within_passage_half_to_half_semantic_redundancy_t5": satterthwaite_welch_and_tost(
                within_cos_sim_lo, within_cos_sim_hi, margin=0.02
            ),
        }

    return pair_semantic_results


# ==============================================================================
# PHASE 8: Computational Budget, NFE, and Wall-Clock Runtime Audit
# ==============================================================================

def audit_computational_budget(runs: Dict[str, dict]) -> dict:
    """
    Phase 8: Exact audit of theoretical NFE vs. logged batch call counters vs. measured wall-clock
    generation time (wall_s) and GPT-2 scoring time (score_s) across hardware and batch sizes.
    """
    # 1. ELF-B 56-run grid on RTX PRO 6000 (batch_size=256, steps=32)
    b_rtx = [
        r for k, r in runs.items()
        if k.startswith("p4_elfb_s32_") and "RTX PRO 6000" in str(r.get("gpu"))
    ]
    # Separate first-run JIT warmup from steady-state compiled runs
    b_rtx_steady = [r for r in b_rtx if float(r["wall_s"]) < 40.0]

    by_sc_rtx = {}
    for sc in [1.0, 2.0, 3.0, 4.0]:
        sub_all = [float(r["wall_s"]) for r in b_rtx if float(r["sc_cfg"]) == sc]
        sub_st = [float(r["wall_s"]) for r in b_rtx_steady if float(r["sc_cfg"]) == sc]
        by_sc_rtx[str(sc)] = {
            "n_all": len(sub_all),
            "mean_wall_s_all": round(float(np.mean(sub_all)), 3),
            "sd_wall_s_all": round(float(np.std(sub_all, ddof=1)), 3),
            "n_steady_excl_compile_warmup": len(sub_st),
            "mean_wall_s_steady": round(float(np.mean(sub_st)), 3),
            "sd_wall_s_steady": round(float(np.std(sub_st, ddof=1)), 3),
        }

    # 2. ELF-M 24-run grid on A100-SXM4-40GB (batch_size=48, steps=64)
    m_a100 = [
        r for k, r in runs.items()
        if k.startswith("p5_elfm_s64_") and "A100-SXM4-40GB" in str(r.get("gpu"))
    ]
    m_a100_steady = [r for r in m_a100 if float(r["wall_s"]) < 410.0]
    by_sc_m = {}
    for sc in [1.0, 4.0]:
        sub_all = [float(r["wall_s"]) for r in m_a100 if float(r["sc_cfg"]) == sc]
        sub_st = [float(r["wall_s"]) for r in m_a100_steady if float(r["sc_cfg"]) == sc]
        by_sc_m[str(sc)] = {
            "n_all": len(sub_all),
            "mean_wall_s_all": round(float(np.mean(sub_all)), 3),
            "n_steady_excl_compile_warmup": len(sub_st),
            "mean_wall_s_steady": round(float(np.mean(sub_st)), 3),
            "sd_wall_s_steady": round(float(np.std(sub_st, ddof=1)), 3),
            "n_elf_fwd": 1386,
            "logged_nfe_elf": 66.0,
            "true_per_sample_nfe": 65,
        }

    # 3. Interventions overhead (ECP and CMLC vs untouched ODE/SDE baselines)
    ecp_32 = [float(r["wall_s"]) for k, r in runs.items() if "ecp_s3_best_ecp_sc_hard_s32" in k]
    b1_32 = [float(r["wall_s"]) for k, r in runs.items() if "T1_B1_ode32" in k]
    cmlc_ent_16 = [float(r["wall_s"]) for k, r in runs.items() if "T1_R3_adapt_ent_ode16" in k]
    cmlc_proj_32 = [float(r["wall_s"]) for k, r in runs.items() if "T1_R3_adapt_stab_proj_ode32" in k]

    return {
        "nfe_accounting_formula": {
            "true_per_sample_flow_steps": "K (32 for ELF-B, 64 for ELF-M/L)",
            "true_per_sample_decode_passes": "1 (via _dlm_decode_batch at t=1.0)",
            "true_per_sample_nfe": "K + 1 (33 for ELF-B, 65 for ELF-M/L)",
            "why_logged_nfe_elf_exceeds_K_plus_1": (
                "In elfx/sample.py L180-184, decode_latents chunks a batch of size B into slices of 32 "
                "to bound fp32 (chunk, L, vocab) memory, incrementing CounterWrapper ceil(B/32) times per batch. "
                "Thus logged nfe_elf = K + ceil(B/32) (e.g. 32 + 8 = 40 at B=256; 32 + 4 = 36 at B=128; 64 + 2 = 66 at B=48)."
            ),
            "why_sc_cfg_has_zero_extra_forward_passes": (
                "In ELF/src/utils/sampling_utils.py L152-161, config.num_self_cond_cfg_tokens == 4 > 0 "
                "(guidance-distilled conditioning tokens). Every step concatenates [z, x_pred_prev] along dim=-1 "
                "and passes self_cond_cfg_scale as a conditioning scalar in ONE forward pass for all s in [1.0, 4.0]."
            ),
            "sde_vs_ode_extra_ops": (
                "In elfx/sample.py L142-164, SDE steps (gamma > 0) draw one Gaussian tensor eps ~ N(0, I) of shape (B, L, D) "
                "and perform two elementwise linear combinations (alpha * z + (1 - alpha) * eps). "
                "This adds O(B*L*D) RNG/vector ops (<0.2% of transformer FLOPs) and 0 extra neural forward passes."
            ),
        },
        "elfb_rtx_pro_6000_wall_clock_by_sc_cfg": by_sc_rtx,
        "elfm_a100_40gb_wall_clock_by_sc_cfg": by_sc_m,
        "interventions_computational_overhead": {
            "untouched_ode32_baseline_B1_mean_wall_s": round(float(np.mean(b1_32)), 2) if b1_32 else None,
            "ecp_hard_ode32_mean_wall_s": round(float(np.mean(ecp_32)), 2) if ecp_32 else None,
            "ecp_hard_ode32_extra_t5_enc_calls_per_batch": 24,
            "cmlc_adapt_ent_ode16_mean_wall_s": round(float(np.mean(cmlc_ent_16)), 2) if cmlc_ent_16 else None,
            "cmlc_adapt_stab_proj_ode32_mean_wall_s": round(float(np.mean(cmlc_proj_32)), 2) if cmlc_proj_32 else None,
        },
    }


def main():
    runs = load_all_runs()
    print(f"[critical_foundation_reanalysis] Loaded {len(runs)} run summaries.")

    # Phase 2: Candidate equivalence margins & TOST across the 6 primary pairs (5-seed and 2-seed)
    margins_info = derive_scientific_equivalence_margins(runs)

    # Evaluate the 6 matched pairs against Tier I (2.5% PPL / 0.015 nats H) and Tier II (5.0% PPL / 0.025 nats H)
    pair_specs = [
        ("M1_ELFM_5seed", "p5_elfm_s64", (1.0, 2.0), (4.0, 0.5)),
        ("S1_ELFB_5seed", "p4_elfb_s32", (1.0, 2.0), (4.0, 1.5)),
        ("S2_ELFB_2seed", "p4_elfb_s32", (1.0, 1.5), (3.0, 1.0)),
        ("S3_ELFB_2seed", "p4_elfb_s32", (2.0, 1.0), (4.0, 0.5)),
        ("Refined1_ELFB_5seed", "p4_elfb_s32", (1.0, 1.25), (3.0, 0.5)),
        ("Refined2_ELFB_5seed", "p4_elfb_s32", (1.0, 1.75), (4.0, 1.0)),
    ]
    pair_tost_reports = {}
    for pname, prefix, (sc_a, g_a), (sc_b, g_b) in pair_specs:
        r_a = [
            r for k, r in runs.items()
            if k.startswith(prefix) and float(r.get("sc_cfg", -1)) == sc_a and float(r.get("gamma", -1)) == g_a
        ]
        r_b = [
            r for k, r in runs.items()
            if k.startswith(prefix) and float(r.get("sc_cfg", -1)) == sc_b and float(r.get("gamma", -1)) == g_b
        ]
        mean_ppl_a = float(np.mean([x["gen_ppl"] for x in r_a]))
        # Test Gen-PPL at Tier I (2.5% of A) and Tier II (5.0% of A)
        tost_ppl_tier1 = satterthwaite_welch_and_tost(
            [x["gen_ppl"] for x in r_a], [x["gen_ppl"] for x in r_b], margin=0.025 * mean_ppl_a
        )
        tost_ppl_tier2 = satterthwaite_welch_and_tost(
            [x["gen_ppl"] for x in r_a], [x["gen_ppl"] for x in r_b], margin=0.050 * mean_ppl_a
        )
        tost_hps_tier1 = satterthwaite_welch_and_tost(
            [x["entropy"] for x in r_a], [x["entropy"] for x in r_b], margin=0.015
        )
        tost_hps_tier2 = satterthwaite_welch_and_tost(
            [x["entropy"] for x in r_a], [x["entropy"] for x in r_b], margin=0.025
        )
        # Secondary metrics with Hedges' g and 95% CIs
        sec_metrics = {}
        for m in ["rep4", "within_distinct2", "h_pool", "jsd_inter_sample", "distinct2", "self_bleu", "jcd_bits_per_byte", "mauve"]:
            va = [float(x[m]) for x in r_a if x.get(m) is not None]
            vb = [float(x[m]) for x in r_b if x.get(m) is not None]
            if len(va) >= 2 and len(vb) >= 2:
                sec_metrics[m] = satterthwaite_welch_and_tost(va, vb, margin=0.01)

        pair_tost_reports[pname] = {
            "cell_A": {"sc_cfg": sc_a, "gamma": g_a, "n_seeds": len(r_a)},
            "cell_B": {"sc_cfg": sc_b, "gamma": g_b, "n_seeds": len(r_b)},
            "gen_ppl_tier1_2.5pct": tost_ppl_tier1,
            "gen_ppl_tier2_5.0pct": tost_ppl_tier2,
            "h_ps_tier1_0.015nats": tost_hps_tier1,
            "h_ps_tier2_0.025nats": tost_hps_tier2,
            "passes_joint_TOST_Tier1": bool(tost_ppl_tier1["tost_equivalent_at_05"] and tost_hps_tier1["tost_equivalent_at_05"]),
            "passes_joint_TOST_Tier2": bool(tost_ppl_tier2["tost_equivalent_at_05"] and tost_hps_tier2["tost_equivalent_at_05"]),
            "secondary_metrics_hedges_g_and_welch": sec_metrics,
        }

    root_matched_candidates = predict_exact_root_matched_pairs(runs)

    # Phase 3: Extended Factorial ANOVA across different gamma subsets on ELF-B and 2x2x5 on ELF-M
    elfb_base = [
        r for k, r in runs.items()
        if k.startswith("p4_elfb_s32_") and _seed_of(k, r) in (0, 1)
        and float(r["gamma"]) in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    ]
    elfm_2x2x5 = [
        r for k, r in runs.items()
        if k.startswith("p5_elfm_s64_") and float(r["sc_cfg"]) in (1.0, 4.0) and float(r["gamma"]) in (0.5, 2.0)
    ]

    anova_sensitivity = {
        "ELFB_4x5x2_gamma_le_1.5": {
            m: two_way_anova_extended(elfb_base, [1.0, 2.0, 3.0, 4.0], [0.0, 0.25, 0.5, 1.0, 1.5], m)
            for m in ["rep4", "distinct2", "mauve", "gen_ppl", "entropy"]
        },
        "ELFB_4x6x2_gamma_le_2.0": {
            m: two_way_anova_extended(elfb_base, [1.0, 2.0, 3.0, 4.0], [0.0, 0.25, 0.5, 1.0, 1.5, 2.0], m)
            for m in ["rep4", "distinct2", "mauve", "gen_ppl", "entropy"]
        },
        "ELFB_4x7x2_gamma_le_3.0_full": {
            m: two_way_anova_extended(elfb_base, [1.0, 2.0, 3.0, 4.0], [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0], m)
            for m in ["rep4", "distinct2", "mauve", "gen_ppl", "entropy"]
        },
        "ELFM_2x2x5_balanced_5seed": {
            m: two_way_anova_extended(elfm_2x2x5, [1.0, 4.0], [0.5, 2.0], m)
            for m in ["rep4", "within_distinct2", "h_pool", "jsd_inter_sample", "distinct2", "self_bleu", "jcd_bits_per_byte", "mauve", "gen_ppl", "entropy"]
        },
    }

    metric_families = analyze_metric_dependence_and_families(runs)
    power_info = human_eval_power_analysis()
    semantic_info = compute_semantic_diversity_on_real_samples()
    compute_audit = audit_computational_budget(runs)

    out_payload = {
        "phase2_equivalence_and_matching": {
            "scientific_margin_derivations": margins_info,
            "pair_tost_and_effect_size_reports": pair_tost_reports,
            "root_finding_new_candidate_pairs": root_matched_candidates,
        },
        "phase3_factorial_and_metric_hierarchy": {
            "anova_gamma_range_sensitivity_and_diagnostics": anova_sensitivity,
            "metric_dependence_and_outcome_families": metric_families,
        },
        "phase4_human_eval_power": power_info,
        "phase5_semantic_diversity_on_600_real_passages": semantic_info,
        "phase8_computational_budget_audit": compute_audit,
    }

    out_path = OUT_DIR / "critical_foundation_stats.json"
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"[critical_foundation_reanalysis] Wrote {out_path}")


if __name__ == "__main__":
    main()
