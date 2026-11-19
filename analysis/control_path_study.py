"""Rigorous statistical analysis of sampler-control non-identifiability on the ELF-B 56-run grid,
scale runs (ELF-M, ELF-L), and secondary interventions (ECP, CMLC).

Outputs:
  - analysis/out/control_path_stats.json
  - analysis/out/direct_matched_pairs.csv
  - analysis/out/bracketing_triples.csv
  - analysis/out/ancova_table.csv
  - analysis/out/constrained_decisions.csv
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "analysis" / "results_db.csv"
OUT_DIR = ROOT / "analysis" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Exact OLS via QR/SVD least squares with t-stats and F-stats."""
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    df_resid = n - k
    s2 = ssr / df_resid
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2.0 * stats.t.sf(np.abs(tvals), df=df_resid)
    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t": tvals.tolist(),
        "p": pvals.tolist(),
        "r2": float(r2),
        "ssr": float(ssr),
        "sst": float(sst),
        "df_resid": int(df_resid),
        "n": int(n),
        "k": int(k),
    }


def interp_metric_at_h(pts: list[tuple[float, float]], h_target: float) -> float:
    """Linear interpolation of a metric y as a function of entropy H."""
    pts_s = sorted((float(h), float(y)) for h, y in pts)
    hs = [h for h, _ in pts_s]
    if not pts_s or h_target < hs[0] - 1e-12 or h_target > hs[-1] + 1e-12:
        return float("nan")
    for (h0, y0), (h1, y1) in zip(pts_s[:-1], pts_s[1:]):
        if h0 - 1e-12 <= h_target <= h1 + 1e-12:
            if abs(h1 - h0) < 1e-12:
                return 0.5 * (y0 + y1)
            a = (h_target - h0) / (h1 - h0)
            return float((1.0 - a) * y0 + a * y1)
    return float("nan")


def main() -> None:
    df = pd.read_csv(DB_PATH)
    p4_all = df[df["run_id"].str.startswith("p4_elfb_")].copy()
    p4_all_2seed = p4_all[p4_all["seed"].isin([0, 1])].copy()
    p4 = p4_all_2seed[p4_all_2seed["gamma"].isin([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])].copy()
    assert len(p4) == 56, f"Expected 56 baseline p4_elfb runs, got {len(p4)}"
    assert len(p4_all) in (56, 60, 72, 78), f"Expected 56, 60, 72, or 78 total p4_elfb runs, got {len(p4_all)}"

    p4["log_ppl"] = np.log(p4["gen_ppl"])
    p4["iqr_80"] = p4["p90"] - p4["p10"]
    p4["ratio_90_10"] = p4["p90"] / p4["p10"]
    p4["ci_half"] = 0.5 * (p4["ppl_ci_hi"] - p4["ppl_ci_lo"])
    p4["ci_rel_half"] = p4["ci_half"] / p4["gen_ppl"]

    # 1. Cell-level summary and seed-to-seed noise precision
    cells = []
    for (sc, g), grp in p4.groupby(["sc_cfg", "gamma"]):
        s0 = grp[grp["seed"] == 0].iloc[0]
        s1 = grp[grp["seed"] == 1].iloc[0]
        cells.append({
            "sc_cfg": float(sc),
            "gamma": float(g),
            "gen_ppl": float(grp["gen_ppl"].mean()),
            "ppl_s0": float(s0["gen_ppl"]),
            "ppl_s1": float(s1["gen_ppl"]),
            "ppl_seed_diff": float(abs(s0["gen_ppl"] - s1["gen_ppl"])),
            "ppl_seed_rel_diff": float(abs(s0["gen_ppl"] - s1["gen_ppl"]) / grp["gen_ppl"].mean()),
            "ci_lo_min": float(grp["ppl_ci_lo"].min()),
            "ci_hi_max": float(grp["ppl_ci_hi"].max()),
            "ci_half_mean": float(grp["ci_half"].mean()),
            "ci_rel_half_mean": float(grp["ci_rel_half"].mean()),
            "entropy": float(grp["entropy"].mean()),
            "ent_s0": float(s0["entropy"]),
            "ent_s1": float(s1["entropy"]),
            "ent_seed_diff": float(abs(s0["entropy"] - s1["entropy"])),
            "rep4": float(grp["rep4"].mean()),
            "rep4_s0": float(s0["rep4"]),
            "rep4_s1": float(s1["rep4"]),
            "rep4_seed_diff": float(abs(s0["rep4"] - s1["rep4"])),
            "mauve": float(grp["mauve"].mean()),
            "mauve_s0": float(s0["mauve"]),
            "mauve_s1": float(s1["mauve"]),
            "mauve_seed_diff": float(abs(s0["mauve"] - s1["mauve"])),
            "distinct1": float(grp["distinct1"].mean()),
            "d1_s0": float(s0["distinct1"]),
            "d1_s1": float(s1["distinct1"]),
            "distinct2": float(grp["distinct2"].mean()),
            "d2_s0": float(s0["distinct2"]),
            "d2_s1": float(s1["distinct2"]),
            "distinct3": float(grp["distinct3"].mean()),
            "d3_s0": float(s0["distinct3"]),
            "d3_s1": float(s1["distinct3"]),
            "p10": float(grp["p10"].mean()),
            "p50": float(grp["p50"].mean()),
            "p90": float(grp["p90"].mean()),
            "iqr_80": float(grp["iqr_80"].mean()),
            "ratio_90_10": float(grp["ratio_90_10"].mean()),
            "mean_gpt2_len": float(grp["mean_gpt2_len"].mean()),
            "degen": float(grp["degen"].mean()),
        })
    cdf = pd.DataFrame(cells)

    precision = {
        "n_runs": 56,
        "n_cells": 28,
        "ppl_seed_diff_median": float(cdf["ppl_seed_diff"].median()),
        "ppl_seed_diff_max": float(cdf["ppl_seed_diff"].max()),
        "ppl_seed_rel_diff_median_pct": float(cdf["ppl_seed_rel_diff"].median() * 100),
        "ppl_seed_rel_diff_max_pct": float(cdf["ppl_seed_rel_diff"].max() * 100),
        "ppl_ci_rel_half_median_pct": float(cdf["ci_rel_half_mean"].median() * 100),
        "ppl_ci_rel_half_max_pct": float(cdf["ci_rel_half_mean"].max() * 100),
        "ent_seed_diff_median": float(cdf["ent_seed_diff"].median()),
        "ent_seed_diff_max": float(cdf["ent_seed_diff"].max()),
        "rep4_seed_diff_median_pp": float(cdf["rep4_seed_diff"].median() * 100),
        "rep4_seed_diff_max_pp": float(cdf["rep4_seed_diff"].max() * 100),
        "mauve_seed_diff_median": float(cdf["mauve_seed_diff"].median()),
        "mauve_seed_diff_max": float(cdf["mauve_seed_diff"].max()),
    }

    # 2. Directly matched cross-path observed pairs (no interpolation)
    # Criteria: sc_cfg_1 < sc_cfg_2, gamma_1 > gamma_2 (opposite control paths),
    # 95% bootstrap CIs for Gen-PPL overlap, and |d_ent| <= 0.031 nats.
    direct_pairs = []
    for i in range(len(cdf)):
        for j in range(i + 1, len(cdf)):
            r1, r2 = cdf.iloc[i], cdf.iloc[j]
            if not (r1["sc_cfg"] < r2["sc_cfg"] and r1["gamma"] > r2["gamma"]):
                continue
            ci_ov = min(r1["ci_hi_max"], r2["ci_hi_max"]) - max(r1["ci_lo_min"], r2["ci_lo_min"])
            d_ent = float(r2["entropy"] - r1["entropy"])
            d_ppl = float(r2["gen_ppl"] - r1["gen_ppl"])
            rel_ppl = abs(d_ppl) / (0.5 * (r1["gen_ppl"] + r2["gen_ppl"]))
            if ci_ov > 0 and abs(d_ent) <= 0.031:
                direct_pairs.append({
                    "sc1": int(r1["sc_cfg"]),
                    "g1": float(r1["gamma"]),
                    "sc2": int(r2["sc_cfg"]),
                    "g2": float(r2["gamma"]),
                    "tier": "strict (|dH|<=0.010)" if abs(d_ent) <= 0.010 else "broad (|dH|<=0.031)",
                    "ent1": float(r1["entropy"]),
                    "ent2": float(r2["entropy"]),
                    "d_ent": d_ent,
                    "ppl1": float(r1["gen_ppl"]),
                    "ppl2": float(r2["gen_ppl"]),
                    "d_ppl": d_ppl,
                    "rel_ppl_pct": float(rel_ppl * 100),
                    "ci_overlap_ppl": float(ci_ov),
                    "rep4_1_pct": float(r1["rep4"] * 100),
                    "rep4_2_pct": float(r2["rep4"] * 100),
                    "d_rep4_pp": float((r2["rep4"] - r1["rep4"]) * 100),
                    "d_rep4_s0_pp": float((r2["rep4_s0"] - r1["rep4_s0"]) * 100),
                    "d_rep4_s1_pp": float((r2["rep4_s1"] - r1["rep4_s1"]) * 100),
                    "mauve1": float(r1["mauve"]),
                    "mauve2": float(r2["mauve"]),
                    "d_mauve": float(r2["mauve"] - r1["mauve"]),
                    "d_mauve_s0": float(r2["mauve_s0"] - r1["mauve_s0"]),
                    "d_mauve_s1": float(r2["mauve_s1"] - r1["mauve_s1"]),
                    "d1_1": float(r1["distinct1"]),
                    "d1_2": float(r2["distinct1"]),
                    "d_d1": float(r2["distinct1"] - r1["distinct1"]),
                    "d2_1": float(r1["distinct2"]),
                    "d2_2": float(r2["distinct2"]),
                    "d_d2": float(r2["distinct2"] - r1["distinct2"]),
                    "d_d2_s0": float(r2["d2_s0"] - r1["d2_s0"]),
                    "d_d2_s1": float(r2["d2_s1"] - r1["d2_s1"]),
                    "d3_1": float(r1["distinct3"]),
                    "d3_2": float(r2["distinct3"]),
                    "d_d3": float(r2["distinct3"] - r1["distinct3"]),
                    "p10_1": float(r1["p10"]),
                    "p10_2": float(r2["p10"]),
                    "p50_1": float(r1["p50"]),
                    "p50_2": float(r2["p50"]),
                    "p90_1": float(r1["p90"]),
                    "p90_2": float(r2["p90"]),
                })
    dpdf = pd.DataFrame(direct_pairs).sort_values("d_ent", key= lambda s: s.abs())
    dpdf.to_csv(OUT_DIR / "direct_matched_pairs.csv", index=False)

    # 3. Interpolation-free bracketing triples
    triples = []
    for _, A in cdf.iterrows():
        s1, g1 = A["sc_cfg"], A["gamma"]
        for s2 in sorted(cdf["sc_cfg"].unique()):
            if s2 <= s1:
                continue
            sub = cdf[cdf["sc_cfg"] == s2].sort_values("gamma")
            for k in range(len(sub) - 1):
                B_lo = sub.iloc[k]
                B_hi = sub.iloc[k + 1]
                if B_hi["entropy"] <= A["entropy"] <= B_lo["entropy"] and B_hi["gen_ppl"] <= A["gen_ppl"] <= B_lo["gen_ppl"]:
                    rep4_dom = bool((B_lo["rep4"] > A["rep4"]) and (B_hi["rep4"] > A["rep4"]))
                    mauve_dom = bool((B_lo["mauve"] > A["mauve"]) and (B_hi["mauve"] > A["mauve"]))
                    triples.append({
                        "s1": int(s1),
                        "g1": float(g1),
                        "A_ent": float(A["entropy"]),
                        "A_ppl": float(A["gen_ppl"]),
                        "A_rep4_pct": float(A["rep4"] * 100),
                        "A_mauve": float(A["mauve"]),
                        "s2": int(s2),
                        "g_lo": float(B_lo["gamma"]),
                        "g_hi": float(B_hi["gamma"]),
                        "Blo_ent": float(B_lo["entropy"]),
                        "Bhi_ent": float(B_hi["entropy"]),
                        "Blo_ppl": float(B_lo["gen_ppl"]),
                        "Bhi_ppl": float(B_hi["gen_ppl"]),
                        "Blo_rep4_pct": float(B_lo["rep4"] * 100),
                        "Bhi_rep4_pct": float(B_hi["rep4"] * 100),
                        "min_rep4_diff_pp": float((min(B_lo["rep4"], B_hi["rep4"]) - A["rep4"]) * 100),
                        "Blo_mauve": float(B_lo["mauve"]),
                        "Bhi_mauve": float(B_hi["mauve"]),
                        "min_mauve_diff": float(min(B_lo["mauve"], B_hi["mauve"]) - A["mauve"]),
                        "both_rep4_gt_A": rep4_dom,
                        "both_mauve_gt_A": mauve_dom,
                    })
    tdf = pd.DataFrame(triples)
    tdf.to_csv(OUT_DIR / "bracketing_triples.csv", index=False)

    # 4. Formal ANCOVA / OLS regressions on N=28 cell means and N=56 runs (plus N=24 non-collapsed gamma<=2.0)
    cdf["log_ppl"] = np.log(cdf["gen_ppl"])
    cdf["ent_c"] = cdf["entropy"] - 5.20
    cdf["ent_c2"] = cdf["ent_c"] ** 2
    p4["ent_c"] = p4["entropy"] - 5.20
    p4["ent_c2"] = p4["ent_c"] ** 2

    ancova_rows = []
    for subset_name, sub_cell, sub_run in [
        ("all_28_cells", cdf, p4),
        ("non_collapsed_24_cells_g_le_2", cdf[cdf["gamma"] <= 2.0].copy(), p4[p4["gamma"] <= 2.0].copy()),
    ]:
        X0_c = np.column_stack([np.ones(len(sub_cell)), sub_cell["log_ppl"], sub_cell["ent_c"], sub_cell["ent_c2"]])
        X1_c = np.column_stack([X0_c, sub_cell["sc_cfg"]])
        X2_c = np.column_stack([X0_c, sub_cell["sc_cfg"], sub_cell["gamma"], sub_cell["sc_cfg"] * sub_cell["gamma"]])

        X0_r = np.column_stack([np.ones(len(sub_run)), sub_run["log_ppl"], sub_run["ent_c"], sub_run["ent_c2"], sub_run["seed"]])
        X1_r = np.column_stack([X0_r, sub_run["sc_cfg"]])
        X2_r = np.column_stack([X0_r, sub_run["sc_cfg"], sub_run["gamma"], sub_run["sc_cfg"] * sub_run["gamma"]])

        for outcome in ["rep4", "mauve", "distinct1", "distinct2", "distinct3", "ratio_90_10", "mean_gpt2_len"]:
            yc = sub_cell[outcome].values
            m0c, m1c, m2c = fit_ols(X0_c, yc), fit_ols(X1_c, yc), fit_ols(X2_c, yc)
            pr2_sc_c = (m0c["ssr"] - m1c["ssr"]) / m0c["ssr"]
            pr2_full_c = (m0c["ssr"] - m2c["ssr"]) / m0c["ssr"]
            f_full_c = ((m0c["ssr"] - m2c["ssr"]) / 3.0) / (m2c["ssr"] / m2c["df_resid"])
            p_full_c = float(stats.f.sf(f_full_c, 3, m2c["df_resid"]))

            yr = sub_run[outcome].values
            m0r, m1r, m2r = fit_ols(X0_r, yr), fit_ols(X1_r, yr), fit_ols(X2_r, yr)
            pr2_sc_r = (m0r["ssr"] - m1r["ssr"]) / m0r["ssr"]
            f_full_r = ((m0r["ssr"] - m2r["ssr"]) / 3.0) / (m2r["ssr"] / m2r["df_resid"])
            p_full_r = float(stats.f.sf(f_full_r, 3, m2r["df_resid"]))

            ancova_rows.append({
                "subset": subset_name,
                "outcome": outcome,
                "n_cells": len(sub_cell),
                "r2_frontier_cell": m0c["r2"],
                "r2_plus_sc_cell": m1c["r2"],
                "partial_r2_sc_cell": float(pr2_sc_c),
                "beta_sc_cell": m1c["beta"][4],
                "se_sc_cell": m1c["se"][4],
                "t_sc_cell": m1c["t"][4],
                "p_sc_cell": m1c["p"][4],
                "r2_full_cell": m2c["r2"],
                "partial_r2_full_cell": float(pr2_full_c),
                "F_full_cell": float(f_full_c),
                "p_full_cell": p_full_c,
                "n_runs": len(sub_run),
                "beta_sc_run": m1r["beta"][5],
                "t_sc_run": m1r["t"][5],
                "p_sc_run": m1r["p"][5],
                "partial_r2_sc_run": float(pr2_sc_r),
                "F_full_run": float(f_full_r),
                "p_full_run": p_full_r,
            })
    adf = pd.DataFrame(ancova_rows)
    adf.to_csv(OUT_DIR / "ancova_table.csv", index=False)

    # 5. Constrained decision rules across threshold pairs (tau, h0)
    constraints = [
        (45.0, 5.25, "Moderate PPL (<=45), high diversity (H>=5.25)"),
        (35.0, 5.20, "Standard PPL (<=35), human-band diversity (H>=5.20)"),
        (30.0, 5.20, "Tight PPL (<=30), human-band diversity (H>=5.20)"),
        (25.0, 5.13, "Low PPL (<=25), moderate diversity (H>=5.13)"),
    ]
    decision_rows = []
    for tau, h0, label in constraints:
        feas = cdf[(cdf["gen_ppl"] <= tau) & (cdf["entropy"] >= h0)].copy()
        for obj_name, idx in [
            ("Min rep-4", feas["rep4"].idxmin()),
            ("Max MAUVE", feas["mauve"].idxmax()),
            ("Max distinct-2", feas["distinct2"].idxmax()),
            ("Min Gen-PPL", feas["gen_ppl"].idxmin()),
        ]:
            row = feas.loc[idx]
            decision_rows.append({
                "regime": label,
                "ppl_max": tau,
                "ent_min": h0,
                "n_feasible": len(feas),
                "objective": obj_name,
                "sc_cfg": int(row["sc_cfg"]),
                "gamma": float(row["gamma"]),
                "gen_ppl": float(row["gen_ppl"]),
                "entropy": float(row["entropy"]),
                "rep4_pct": float(row["rep4"] * 100),
                "mauve": float(row["mauve"]),
                "distinct1": float(row["distinct1"]),
                "distinct2": float(row["distinct2"]),
                "distinct3": float(row["distinct3"]),
            })
    ddf = pd.DataFrame(decision_rows)
    ddf.to_csv(OUT_DIR / "constrained_decisions.csv", index=False)

    # 6. Scale comparison summary (ELF-B vs ELF-M vs ELF-L including ecp_s4_elfm and p5_elfm)
    s4 = df[df["run_id"].str.startswith("ecp_s4_elfm_")].copy()
    s4_summary = []
    for prefix in ["ode_s32", "sde_g10_s32", "best_ecp_s32", "second_ecp_s32", "ode_s64", "sde_g10_s64", "best_ecp_s64", "second_ecp_s64"]:
        sub = s4[s4["run_id"].str.contains(prefix)]
        s4_summary.append({
            "config": prefix,
            "steps": int(sub["steps"].iloc[0]),
            "gamma": float(sub["gamma"].iloc[0]),
            "sc_cfg": float(sub["sc_cfg"].iloc[0]),
            "n_seeds": len(sub),
            "gen_ppl": float(sub["gen_ppl"].mean()),
            "entropy": float(sub["entropy"].mean()),
            "rep4_pct": float(sub["rep4"].mean() * 100),
            "mauve": float(sub["mauve"].mean()),
            "distinct1": float(sub["distinct1"].mean()),
            "distinct2": float(sub["distinct2"].mean()),
            "distinct3": float(sub["distinct3"].mean()),
        })

    # 7. E3: ELF-B intermediate gamma refinement (gamma in {1.25, 1.75} at SC-CFG=1.0)
    refinement_pairs = []
    if len(p4_all_2seed) == 60:
        cells_60 = []
        for (sc, g), grp in p4_all_2seed.groupby(["sc_cfg", "gamma"]):
            cells_60.append({
                "sc_cfg": float(sc),
                "gamma": float(g),
                "gen_ppl": float(grp["gen_ppl"].mean()),
                "ci_lo_min": float(grp["ppl_ci_lo"].min()),
                "ci_hi_max": float(grp["ppl_ci_hi"].max()),
                "entropy": float(grp["entropy"].mean()),
                "rep4": float(grp["rep4"].mean()),
                "mauve": float(grp["mauve"].mean()),
                "distinct1": float(grp["distinct1"].mean()),
                "distinct2": float(grp["distinct2"].mean()),
                "distinct3": float(grp["distinct3"].mean()),
                "h_pool": float(grp["h_pool"].mean()) if "h_pool" in grp and grp["h_pool"].notna().any() else float("nan"),
                "jsd_inter_sample": float(grp["jsd_inter_sample"].mean()) if "jsd_inter_sample" in grp and grp["jsd_inter_sample"].notna().any() else float("nan"),
                "self_bleu": float(grp["self_bleu"].mean()) if "self_bleu" in grp and grp["self_bleu"].notna().any() else float("nan"),
                "jcd_bits_per_byte": float(grp["jcd_bits_per_byte"].mean()) if "jcd_bits_per_byte" in grp and grp["jcd_bits_per_byte"].notna().any() else float("nan"),
                "within_distinct2": float(grp["within_distinct2"].mean()) if "within_distinct2" in grp and grp["within_distinct2"].notna().any() else float("nan"),
            })
        cdf60 = pd.DataFrame(cells_60)
        for g1, sc2, g2 in [(1.25, 3.0, 0.50), (1.25, 4.0, 0.50), (1.75, 4.0, 1.00), (1.75, 3.0, 1.00)]:
            r1 = cdf60[(cdf60["sc_cfg"] == 1.0) & (cdf60["gamma"] == g1)].iloc[0]
            r2 = cdf60[(cdf60["sc_cfg"] == sc2) & (cdf60["gamma"] == g2)].iloc[0]
            ci_ov = min(r1["ci_hi_max"], r2["ci_hi_max"]) - max(r1["ci_lo_min"], r2["ci_lo_min"])
            refinement_pairs.append({
                "sc1": 1,
                "g1": float(g1),
                "sc2": int(sc2),
                "g2": float(g2),
                "ent1": float(r1["entropy"]),
                "ent2": float(r2["entropy"]),
                "d_ent": float(r2["entropy"] - r1["entropy"]),
                "ppl1": float(r1["gen_ppl"]),
                "ppl2": float(r2["gen_ppl"]),
                "d_ppl": float(r2["gen_ppl"] - r1["gen_ppl"]),
                "ci_overlap_ppl": float(ci_ov),
                "rep4_1_pct": float(r1["rep4"] * 100),
                "rep4_2_pct": float(r2["rep4"] * 100),
                "d_rep4_pp": float((r2["rep4"] - r1["rep4"]) * 100),
                "mauve1": float(r1["mauve"]),
                "mauve2": float(r2["mauve"]),
                "d_mauve": float(r2["mauve"] - r1["mauve"]),
                "d2_1": float(r1["distinct2"]),
                "d2_2": float(r2["distinct2"]),
                "d_d2": float(r2["distinct2"] - r1["distinct2"]),
                "h_pool_1": float(r1["h_pool"]),
                "h_pool_2": float(r2["h_pool"]),
                "d_h_pool": float(r2["h_pool"] - r1["h_pool"]),
                "jsd_1": float(r1["jsd_inter_sample"]),
                "jsd_2": float(r2["jsd_inter_sample"]),
                "d_jsd": float(r2["jsd_inter_sample"] - r1["jsd_inter_sample"]),
                "self_bleu_1": float(r1["self_bleu"]),
                "self_bleu_2": float(r2["self_bleu"]),
                "d_self_bleu": float(r2["self_bleu"] - r1["self_bleu"]),
                "jcd_1": float(r1["jcd_bits_per_byte"]),
                "jcd_2": float(r2["jcd_bits_per_byte"]),
                "d_jcd_bpb": float(r2["jcd_bits_per_byte"] - r1["jcd_bits_per_byte"]),
            })

    # 8. E1: ELF-M (342M, 64 steps) scale-transfer factorial analysis
    elfm_df = df[df["run_id"].str.startswith(("p4_elfm_s64_", "p5_elfm_s64_"))].copy()
    elfm_ancova = []
    if len(elfm_df) >= 9:
        elfm_df["log_ppl"] = np.log(elfm_df["gen_ppl"])
        elfm_df["ent_c"] = elfm_df["entropy"] - 5.20
        X0_m = np.column_stack([np.ones(len(elfm_df)), elfm_df["log_ppl"], elfm_df["ent_c"]])
        X1_m = np.column_stack([X0_m, elfm_df["sc_cfg"]])
        for outcome in ["rep4", "mauve", "distinct1", "distinct2", "distinct3"]:
            ym = elfm_df[outcome].values
            m0m, m1m = fit_ols(X0_m, ym), fit_ols(X1_m, ym)
            pr2_sc_m = (m0m["ssr"] - m1m["ssr"]) / m0m["ssr"]
            elfm_ancova.append({
                "outcome": outcome,
                "n_runs": len(elfm_df),
                "r2_frontier": float(m0m["r2"]),
                "r2_plus_sc": float(m1m["r2"]),
                "partial_r2_sc": float(pr2_sc_m),
                "beta_sc": float(m1m["beta"][3]),
                "se_sc": float(m1m["se"][3]),
                "t_sc": float(m1m["t"][3]),
                "p_sc": float(m1m["p"][3]),
            })

    out_payload = {
        "precision": precision,
        "direct_matched_pairs": direct_pairs,
        "bracketing_triples": triples,
        "ancova": ancova_rows,
        "constrained_decisions": decision_rows,
        "elfm_s4_summary": s4_summary,
        "elfb_refinement_pairs": refinement_pairs,
        "elfm_ancova": elfm_ancova,
    }
    with open(OUT_DIR / "control_path_stats.json", "w") as f:
        json.dump(out_payload, f, indent=2)
    print("Saved control_path_stats.json and CSVs to", OUT_DIR)


if __name__ == "__main__":
    main()
