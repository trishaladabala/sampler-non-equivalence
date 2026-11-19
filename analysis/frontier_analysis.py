"""Frontier analysis from existing artifacts only (no generation, no GPU).

Usage: python3 analysis/collect_runs.py && python3 analysis/frontier_analysis.py
Inputs : analysis/results_db.csv (from collect_runs.py)
         content/drive/MyDrive/elf_research/runs/mdlm_stochastic/_report.md (P8 tables)
Outputs: analysis/out/key_numbers.json  -- every number quoted in the manuscript
         analysis/out/*.csv             -- full tables
         tables/*.tex                   -- LaTeX tables \\input by the manuscript
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import metrics_ref as M  # noqa: E402

RUNS = os.path.join(ROOT, "content", "drive", "MyDrive", "elf_research", "runs")
OUT = os.path.join(HERE, "out")
TAB = os.path.join(ROOT, "tables")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
KEY: dict = {}


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


rows = list(csv.DictReader(open(os.path.join(HERE, "results_db.csv"))))
by_id = {r["run_id"]: r for r in rows}


# ----------------------------------------------------------------------------- stats helpers

def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def pearson(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


def spearman(a, b):
    """Spearman rho with average ranks; p-value via t-approximation (two-sided)."""
    rho = pearson(rankdata(a), rankdata(b))
    n = len(a)
    t = rho * math.sqrt((n - 2) / max(1e-300, 1 - rho ** 2))
    # two-sided p from Student t with n-2 dof via regularized incomplete beta
    p = _t_sf(abs(t), n - 2) * 2
    return rho, p, n


def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        de = d * c; h *= de
        if abs(de - 1) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def _t_sf(t, df):
    return 0.5 * _betai(df / 2, 0.5, df / (df + t * t))


# ============================================================================= Study 1: ELF γ frontier
p4_all = [r for r in rows if r["run_id"].startswith("p4_elfb_s32_")]
p4 = [r for r in p4_all if f(r["gamma"]) in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0) and int(f(r["seed"])) in (0, 1)]
p4_refinement = [r for r in p4_all if f(r["gamma"]) in (1.25, 1.75) and int(f(r["seed"])) in (0, 1)]
p4_followup_seeds = [r for r in p4_all if int(f(r["seed"])) >= 2]
assert len(p4) == 56, f"expected 56 baseline ELF-B P4 runs, found {len(p4)}"
assert len(p4_all) in (56, 60, 72, 78), f"expected 56, 60, 72, or 78 total ELF-B P4 runs, found {len(p4_all)}"
KEY["p4_elfb_runs"] = len(p4)
KEY["p4_elfb_refinement_runs"] = len(p4_refinement)
KEY["p4_elfb_followup_seed_runs"] = len(p4_followup_seeds)
KEY["p4_elfb_total_runs"] = len(p4_all)
KEY["p4_elfm_runs"] = sum(r["run_id"].startswith("p4_elfm_") for r in rows)
KEY["p5_elfm_runs"] = sum(r["run_id"].startswith("p5_elfm_") for r in rows)
KEY["p4_elfl_runs"] = sum(r["run_id"].startswith("p4_elfl_") for r in rows)
KEY["p4_ar_runs"] = sum(r["run_id"].startswith("p4_ar_") for r in rows)

g = [f(r["gamma"]) for r in p4]
rho_rep, p_rep, n_ = spearman(g, [f(r["rep4"]) for r in p4])
rho_H, p_H, _ = spearman(g, [f(r["entropy"]) for r in p4])
rho_ppl, p_ppl, _ = spearman(g, [f(r["gen_ppl"]) for r in p4])
rho_mv, p_mv, _ = spearman(g, [f(r["mauve"]) for r in p4])
KEY["spearman_gamma"] = {"n": n_, "rep4": [rho_rep, p_rep], "entropy": [rho_H, p_H],
                         "gen_ppl": [rho_ppl, p_ppl], "mauve": [rho_mv, p_mv]}

# seed-averaged frontier per SC-CFG
grid = defaultdict(list)
for r in p4:
    grid[(f(r["sc_cfg"]), f(r["gamma"]))].append(r)
front_rows = []
for (sc, gm), rs in sorted(grid.items()):
    front_rows.append(dict(sc_cfg=sc, gamma=gm, n_seeds=len(rs),
                           ppl=mean(f(x["gen_ppl"]) for x in rs),
                           ppl_min=min(f(x["gen_ppl"]) for x in rs), ppl_max=max(f(x["gen_ppl"]) for x in rs),
                           H=mean(f(x["entropy"]) for x in rs), rep4=mean(f(x["rep4"]) for x in rs),
                           mauve=mean(f(x["mauve"]) for x in rs),
                           distinct2=mean(f(x["distinct2"]) for x in rs)))
with open(os.path.join(OUT, "elfb_frontier.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(front_rows[0])); w.writeheader(); w.writerows(front_rows)


def ppl_at(points, h):
    try:
        return M.frontier_ppl_at(points, h)
    except M.ExtrapolationError:
        return None


ppl_at_h = {}
for sc in (1.0, 2.0, 3.0, 4.0):
    pts = [(x["H"], x["ppl"]) for x in front_rows if x["sc_cfg"] == sc]
    ppl_at_h[f"ELF-B_sc{sc:g}"] = {"monotone": M.is_monotone(pts),
                                   "H_range": [min(p[0] for p in pts), max(p[0] for p in pts)],
                                   **{f"{h:.2f}": ppl_at(pts, h) for h in (5.15, 5.20, 5.24)}}
for tag, prefix in (("ELF-M", "p4_elfm_"), ("ELF-L", "p4_elfl_")):
    rs = [r for r in rows if r["run_id"].startswith(prefix)]
    pts = [(f(r["entropy"]), f(r["gen_ppl"])) for r in rs]
    ppl_at_h[tag] = {"monotone": M.is_monotone(pts), "n_points": len(pts),
                     "H_range": [min(p[0] for p in pts), max(p[0] for p in pts)],
                     **{f"{h:.2f}": ppl_at(pts, h) for h in (5.15, 5.20, 5.24)}}
KEY["ppl_at_H"] = ppl_at_h


# ---- Collapse test: is log Gen-PPL a function of entropy alone across all 56 (γ, SC-CFG, seed) runs?
def polyfit(xs, ys, deg):
    """Least squares polynomial fit via normal equations (small deg, numerically fine here)."""
    n = deg + 1
    A = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum((x ** i) * y for x, y in zip(xs, ys)) for i in range(n)]
    for c in range(n):  # Gaussian elimination
        piv = max(range(c, n), key=lambda r_: abs(A[r_][c]))
        A[c], A[piv] = A[piv], A[c]; b[c], b[piv] = b[piv], b[c]
        for r_ in range(c + 1, n):
            fac = A[r_][c] / A[c][c]
            for k in range(c, n):
                A[r_][k] -= fac * A[c][k]
            b[r_] -= fac * b[c]
    coef = [0.0] * n
    for r_ in range(n - 1, -1, -1):
        coef[r_] = (b[r_] - sum(A[r_][k] * coef[k] for k in range(r_ + 1, n))) / A[r_][r_]
    return coef


def r2(xs, ys, coef):
    pred = [sum(c * x ** i for i, c in enumerate(coef)) for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    my = mean(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    return 1 - ss_res / ss_tot, [y - p for y, p in zip(ys, pred)]


Hs = [f(r["entropy"]) for r in p4]
lP = [math.log(f(r["gen_ppl"])) for r in p4]
MV = [f(r["mauve"]) for r in p4]
SC = [f(r["sc_cfg"]) for r in p4]
GM = [f(r["gamma"]) for r in p4]
collapse = {}
for deg in (1, 2, 3):
    c = polyfit(Hs, lP, deg)
    R2, res_ = r2(Hs, lP, c)
    collapse[f"logppl_on_H_deg{deg}"] = {"R2": R2, "resid_sd_log": (sum(e * e for e in res_) / (len(res_) - deg - 1)) ** 0.5,
                                         "max_abs_resid_pct": 100 * (math.exp(max(abs(e) for e in res_)) - 1)}
# raw spread across guidance at fixed gamma (shows guidance looks like a big quality knob in raw PPL)
raw_g0 = {sc: x["ppl"] for x in front_rows for sc in [x["sc_cfg"]] if x["gamma"] == 0.0}
collapse["raw_ppl_gamma0_by_sc"] = raw_g0
collapse["raw_ratio_sc1_over_sc4_gamma0"] = raw_g0[1.0] / raw_g0[4.0]
vals515 = [ppl_at_h[f"ELF-B_sc{s:g}"]["5.15"] for s in (1.0, 2.0, 3.0, 4.0)]
collapse["ppl_at_5.15_spread_pct"] = 100 * (max(vals515) / min(vals515) - 1)
# MAUVE: does it ALSO collapse onto entropy? residual vs SC-CFG
for deg in (2, 3):
    c = polyfit(Hs, MV, deg)
    R2m, resm = r2(Hs, MV, c)
    rho_sc, p_sc, _ = spearman(SC, resm)
    collapse[f"mauve_on_H_deg{deg}"] = {"R2": R2m, "resid_vs_sc_spearman": [rho_sc, p_sc]}
c2 = polyfit(Hs, lP, 2)
_, resP = r2(Hs, lP, c2)
rho_scP, p_scP, _ = spearman(SC, resP)
collapse["logppl_resid_vs_sc_spearman"] = [rho_scP, p_scP]
# seed-to-seed MAUVE noise (same config, 2 seeds) for scale
pairs = defaultdict(list)
for r in p4:
    pairs[(f(r["sc_cfg"]), f(r["gamma"]))].append(f(r["mauve"]))
collapse["mauve_seed_abs_diff_mean"] = mean(abs(v[0] - v[1]) for v in pairs.values() if len(v) == 2)
collapse["mauve_seed_abs_diff_max"] = max(abs(v[0] - v[1]) for v in pairs.values() if len(v) == 2)
# matched-entropy MAUVE comparison across guidance: interpolate MAUVE at H=5.20 per SC (linear in H)
mv_at = {}
for sc in (1.0, 2.0, 3.0, 4.0):
    pts = sorted((x["H"], x["mauve"]) for x in front_rows if x["sc_cfg"] == sc)
    for (h0, m0), (h1, m1) in zip(pts[:-1], pts[1:]):
        if h0 <= 5.20 <= h1:
            mv_at[sc] = m0 + (5.20 - h0) / (h1 - h0) * (m1 - m0)
collapse["mauve_at_H5.20_by_sc"] = mv_at


def lin_at(sc, key, h=5.20):
    pts = sorted((x["H"], x[key]) for x in front_rows if x["sc_cfg"] == sc)
    for (h0, m0), (h1, m1) in zip(pts[:-1], pts[1:]):
        if h0 <= h <= h1:
            return m0 + (h - h0) / (h1 - h0) * (m1 - m0)
    return None


collapse["rep4_at_H5.20_by_sc"] = {sc: lin_at(sc, "rep4") for sc in (1.0, 2.0, 3.0, 4.0)}
collapse["distinct2_at_H5.20_by_sc"] = {sc: lin_at(sc, "distinct2") for sc in (1.0, 2.0, 3.0, 4.0)}
collapse["ppl_at_H5.20_by_sc"] = {sc: ppl_at_h[f"ELF-B_sc{sc:g}"]["5.20"] for sc in (1.0, 2.0, 3.0, 4.0)}
# seed noise of rep4 for scale
collapse["rep4_seed_abs_diff_max"] = max(abs(a - b) for a, b in
                                         (tuple(f(x["rep4"]) for x in rs) for rs in grid.values() if len(rs) == 2))


# γ and mean generation length at which each guidance scale reaches H=5.20 (linear in H between bracketing γ points)
def _lin_at_from_p4(sc, key, h=5.20):
    by_g = defaultdict(list)
    for r in p4:
        if f(r["sc_cfg"]) == sc:
            by_g[f(r["gamma"])].append(r)
    pts = sorted((mean(f(x["entropy"]) for x in rs), (g if key == "gamma" else mean(f(x[key]) for x in rs)))
                 for g, rs in by_g.items())
    for (h0, m0), (h1, m1) in zip(pts[:-1], pts[1:]):
        if h0 <= h <= h1:
            return m0 + (h - h0) / (h1 - h0) * (m1 - m0)
    return None


collapse["gamma_at_H5.20_by_sc"] = {sc: _lin_at_from_p4(sc, "gamma") for sc in (1.0, 2.0, 3.0, 4.0)}
collapse["len_at_H5.20_by_sc"] = {sc: _lin_at_from_p4(sc, "mean_gpt2_len") for sc in (1.0, 2.0, 3.0, 4.0)}
# per-run 95% bootstrap CI on corpus Gen-PPL stored by the harness (B=10,000 resamples of samples; elf_p0_harness.ipynb score_texts)
_hw = [100 * (f(r["ppl_ci_hi"]) - f(r["ppl_ci_lo"])) / 2 / f(r["gen_ppl"]) for r in p4]
collapse["per_run_ppl_ci_halfwidth_pct"] = {"median": sorted(_hw)[len(_hw) // 2] if len(_hw) % 2 else
                                            0.5 * (sorted(_hw)[len(_hw) // 2 - 1] + sorted(_hw)[len(_hw) // 2]),
                                            "min": min(_hw), "max": max(_hw), "n_runs": len(_hw)}
collapse["mauve_sc4_minus_sc1_at_H5.20"] = mv_at[4.0] - mv_at[1.0]
collapse["rep4_sc4_over_sc1_at_H5.20_pct"] = 100 * (collapse["rep4_at_H5.20_by_sc"][4.0] / collapse["rep4_at_H5.20_by_sc"][1.0] - 1)
collapse["distinct2_sc4_over_sc1_at_H5.20_pct"] = 100 * (collapse["distinct2_at_H5.20_by_sc"][4.0] / collapse["distinct2_at_H5.20_by_sc"][1.0] - 1)
KEY["collapse"] = collapse

# real gamma=3 vs the fabricated Table-2 value
r3 = by_id["p4_elfb_s32_g3.0_sc3.0_seed0"]
KEY["gamma3_sc3_seed0"] = {k: f(r3[k]) for k in ("gen_ppl", "entropy", "rep4", "mauve", "distinct2", "degen")}
r3b = by_id["p4_elfb_s32_g3.0_sc3.0_seed1"]
KEY["gamma3_sc3_seed1"] = {k: f(r3b[k]) for k in ("gen_ppl", "entropy", "rep4", "mauve", "distinct2", "degen")}
KEY["owt_reference"] = {k: f(by_id["R0_owt_ref"][k]) for k in ("gen_ppl", "entropy", "rep4", "distinct2")}

# GPT-2 anchors: validity check only
ar = [r for r in rows if r["run_id"].startswith("p4_ar_")]
KEY["gpt2_anchor_check"] = {"max_entropy": max(f(r["entropy"]) for r in ar),
                            "max_mauve": max(f(r["mauve"]) for r in ar),
                            "verdict": "EXCLUDED: no AR run reaches H=5.15, MAUVE<=0.04, and T=1/top-p=1 Gen-PPL is below the human reference; cause not identified (generation starts from a single BOS with full attention mask, so padding is not the explanation)"}

# ============================================================================= Study 3: interventions vs γ frontier

def frontier_from(run_ids):
    acc = defaultdict(list)
    for rid in run_ids:
        r = by_id[rid]
        acc[f(r["gamma"])].append((f(r["entropy"]), f(r["gen_ppl"]), f(r["mauve"]), int(f(r["n"]))))
    return {gm: (mean(v[0] for v in vs), mean(v[1] for v in vs), mean(v[2] for v in vs if v[2] is not None) if any(v[2] is not None for v in vs) else None, len(vs), vs[0][3])
            for gm, vs in acc.items()}


F16 = frontier_from([f"ecp_s3_b1_ode_s16_seed{s}" for s in range(3)] +
                    [f"ecp_s3_b3_sde_g10_s16_seed{s}" for s in range(3)] +
                    [f"ecp_s3_b2_sde_g15_s16_seed{s}" for s in range(3)])
F32 = frontier_from([f"ecp_s3_b1_ode_s32_seed{s}" for s in range(3)] +
                    ["ecp_s3_frontier_sde_g5_s32"] +
                    [f"ecp_s3_b3_sde_g10_s32_seed{s}" for s in range(3)] +
                    [f"ecp_s3_b2_sde_g15_s32_seed{s}" for s in range(3)] +
                    ["ecp_s3_frontier_sde_g20_s32"])
KEY["baseline_frontier_16"] = {str(k): v for k, v in sorted(F16.items())}
KEY["baseline_frontier_32"] = {str(k): v for k, v in sorted(F32.items())}


def nfe_total(r):
    return (f(r["nfe_elf"]) or 0.0), (f(r["nfe_enc"]) or 0.0)


def assess(rid, frontier, label, base_gamma=0.0):
    """rid may be a single run id or a list of seed run ids (values are averaged)."""
    ids = rid if isinstance(rid, list) else [rid]
    rs = [by_id[i] for i in ids]
    r = rs[0]
    ppl = mean(f(x["gen_ppl"]) for x in rs)
    H = mean(f(x["entropy"]) for x in rs)
    mvs = [f(x["mauve"]) for x in rs if f(x["mauve"]) is not None]
    mv = mean(mvs) if mvs else None
    n = int(f(r["n"]))
    pts = [(v[0], v[1]) for v in frontier.values()]
    base = frontier[base_gamma]
    out = dict(label=label, run_id=";".join(ids), n_seeds=len(ids), n=n, ppl=ppl, H=H, mauve=mv,
               nfe_elf=nfe_total(r)[0], nfe_enc=nfe_total(r)[1], base_gamma=base_gamma,
               base_ppl=base[1], base_H=base[0], base_mauve=base[2], base_n=base[4],
               raw_ppl_change_pct=M.relative_change(base[1], ppl),
               dH=H - base[0])
    try:
        fp = M.frontier_ppl_at(pts, H)
        out["frontier_ppl_at_H"] = fp
        out["residual_log"] = math.log(ppl) - math.log(fp)
        tot = math.log(ppl) - math.log(base[1])
        out["share_explained_by_entropy"] = (math.log(fp) - math.log(base[1])) / tot if tot != 0 else None
        out["status"] = "interpolated"
    except M.ExtrapolationError:
        out["frontier_ppl_at_H"] = None
        out["residual_log"] = None
        out["share_explained_by_entropy"] = None
        # Pareto dominance check against observed baseline points
        dom = [v for v in frontier.values() if v[0] >= H and v[1] <= ppl]
        out["status"] = ("pareto-dominated by gamma-tuned baseline (higher H and lower PPL)" if dom
                         else "outside frontier range; not assessable without new runs")
    return out


interventions = [
    ("ecp_s2_sc_hard_l100_ton30_ode_g0_s16", F16, "ECP hard (SC), t_on=0.30, ODE-16"),
    ("ecp_s1_sc_soft_l100_ton30", F16, "ECP soft (SC), t_on=0.30, ODE-16"),
    ("ecp_s2_sc_hard_l100_ton30_ode_g0_s32", F32, "ECP hard (SC), t_on=0.30, ODE-32"),
    ("ecp_s2_sc_hard_ode_s32_k2", F32, "ECP hard (SC), t_on=0.30, k=2, ODE-32"),
    ("ecp_s2_sc_hard_l100_ton30_sde_g15_s32", None, "ECP hard (SC), t_on=0.30, SDE-32 g=1.5"),
    ("ecp_s2_z_weak_hard_l100_ton30_sde_g15_s32", None, "ECP hard (z-weak), t_on=0.30, SDE-32 g=1.5"),
]
res = []
for rid, fr, lab in interventions:
    if rid not in by_id:
        print(f"[warn] missing {rid}")
        continue
    if fr is None:  # SDE γ=1.5 intervention: raw change vs the SDE γ=1.5 baseline; residual vs frontier
        res.append(assess(rid, F32, lab, base_gamma=1.5))
    else:
        res.append(assess(rid, fr, lab))

# CMLC (fine-tuned R3 model), 3 seeds each, run ids carry the cmlc_eval/ prefix
cmlc_ids = {"cmlc_eval/T1_R3_adapt_ent_ode16": ("CMLC R3 adaptive-ent (ODE-16)", 0.0),
            "cmlc_eval/T1_R3_adapt_stab_sde16_g1.0": ("CMLC R3 adaptive-stab (SDE-16, g=1.0)", 1.0),
            "cmlc_eval/T1_R3_adapt_stab_proj_ode16": ("CMLC R3 adaptive-stab + commit (ODE-16)", 0.0),
            "cmlc_eval/T1_R3_adapt_stab_ode16": ("CMLC R3 adaptive-stab (ODE-16)", 0.0)}
for stem, (lab, bg) in cmlc_ids.items():
    ids = [f"{stem}_s{s}" for s in range(3) if f"{stem}_s{s}" in by_id]
    if ids:
        res.append(assess(ids, F16, lab, base_gamma=bg))
    else:
        print(f"[warn] CMLC run {stem} not found in results_db")


# Stage-3 finals (elf_p1a_ecp_sweep.ipynb, sweep S3): 1,000 samples x 3 seeds, activation window read from summary.json
def _t_on(rid):
    d = json.load(open(by_id[rid]["source_path"]))
    return (d.get("hook_kwargs") or {}).get("t_on")


s3_stems = [("ecp_s3_best_ecp_sc_hard", "ECP hard (SC)"),
            ("ecp_s3_second_ecp_sc_zweak", "ECP hard (SC+z-weak)"),
            ("ecp_s3_a5_rand", "Control A5: matched-norm random"),
            ("ecp_s3_a6_nc", "Control A6: non-contextual re-encode"),
            ("ecp_s3_b6_restart", "Baseline B6: restart")]
s3_ton = set()
for stem, lab in s3_stems:
    for steps, fr in ((16, F16), (32, F32)):
        ids = [f"{stem}_s{steps}_seed{s}" for s in range(3) if f"{stem}_s{steps}_seed{s}" in by_id]
        if len(ids) != 3:
            print(f"[warn] stage-3 {stem}_s{steps}: {len(ids)} seeds")
            continue
        tons = {_t_on(i) for i in ids}
        s3_ton |= tons
        ton = tons.pop() if len(tons) == 1 else None
        res.append(assess(ids, fr, f"{lab}, t_on={ton:.2f}, ODE-{steps} [S3]"))
KEY["stage3_t_on"] = sorted(s3_ton)
KEY["interventions"] = res
with open(os.path.join(OUT, "interventions.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for r in res for k in r})); w.writeheader(); w.writerows(res)

# Compute-matched untouched ODE baselines from S3 (steps = ceil(1.7 x {16, 32}))
cm = {}
for steps in (28, 55):
    ids = [f"ecp_s3_b4_ode_flopmatched_s{steps}_seed{s}" for s in range(3)]
    rs = [by_id[i] for i in ids if i in by_id]
    cm[str(steps)] = {"n_seeds": len(rs), "ppl": mean(f(x["gen_ppl"]) for x in rs), "H": mean(f(x["entropy"]) for x in rs),
                      "mauve": mean(f(x["mauve"]) for x in rs), "nfe_elf": f(rs[0]["nfe_elf"])}
KEY["compute_matched_ode"] = cm

# CMLC fine-tune degradation check (cmlc_eval T1 block, 3 seeds): untouched checkpoint vs global-clock control fine-tune
KEY["cmlc_finetune_check"] = {
    stem: mean(f(by_id[f"cmlc_eval/{stem}_s{s}"]["gen_ppl"]) for s in range(3))
    for stem in ("T1_B1_ode16", "T1_B2_R1_global_ode16", "T1_B3_R3_global_ode16")}

# A5 / A6 controls (S3) -- activation window recorded in KEY["stage3_t_on"]
ctrl = defaultdict(list)
for r in rows:
    m = re.match(r"ecp_s3_(a5_rand|a6_nc|b1_ode|best_ecp_sc_hard|b6_restart)_s(16|32)_seed(\d)", r["run_id"])
    if m:
        ctrl[(m.group(1), m.group(2))].append((f(r["gen_ppl"]), f(r["entropy"]), f(r["mauve"])))
KEY["controls_s3"] = {f"{k[0]}_s{k[1]}": {"n_seeds": len(v), "ppl": mean(x[0] for x in v),
                                          "H": mean(x[1] for x in v), "mauve": mean(x[2] for x in v),
                                          "mauve_min": min(x[2] for x in v), "mauve_max": max(x[2] for x in v)}
                      for k, v in sorted(ctrl.items())}

# ============================================================================= Study 2: P8 cross-paradigm
rep = open(os.path.join(RUNS, "mdlm_stochastic", "_report.md")).read()


def parse_table(section_title):
    sec = rep.split(section_title, 1)[1]
    lines = [l for l in sec.splitlines() if l.startswith("|")]
    out = []
    for l in lines[2:]:
        cells = [c.strip() for c in l.strip("|").split("|")]
        if not cells or not re.match(r"^\d", cells[0]):
            break
        out.append(cells)
    return out


mdlm = [dict(N=int(c[0]), tau=float(c[1]), ppl=float(c[2]), dedup=float(c[3]), rep4=float(c[5]), H=float(c[6]), degen=float(c[7]))
        for c in parse_table("Part A")]
elf8 = [dict(N=int(c[0]), gamma=float(c[1]), ppl=float(c[3]), dedup=float(c[4]), rep4=float(c[6]), H=float(c[7]), degen=float(c[8]))
        for c in parse_table("Part C.1")]
assert len(mdlm) == 35 and len(elf8) == 15, (len(mdlm), len(elf8))
KEY["p8_cells"] = {"mdlm": len(mdlm), "elf": len(elf8)}

xp = []
for N in (4, 8, 16, 32, 64):
    mf = [(m["H"], m["ppl"]) for m in mdlm if m["N"] == N and m["tau"] > 0]
    md = [(m["H"], m["dedup"]) for m in mdlm if m["N"] == N and m["tau"] > 0]
    best_raw = min((m for m in mdlm if m["N"] == N and m["degen"] < 0.5), key=lambda m: m["dedup"])
    for e in (x for x in elf8 if x["N"] == N):
        row = dict(N=N, elf_gamma=e["gamma"], elf_ppl=e["ppl"], elf_dedup=e["dedup"], elf_H=e["H"], elf_rep4=e["rep4"],
                   elf_degen=e["degen"], mdlm_monotone=M.is_monotone(mf),
                   mdlm_best_dedup_tau=best_raw["tau"], mdlm_best_dedup=best_raw["dedup"], mdlm_best_H=best_raw["H"],
                   mdlm_best_rep4=best_raw["rep4"])
        try:
            row["mdlm_ppl_at_elfH"] = M.frontier_ppl_at(mf, e["H"])
            row["mdlm_dedup_at_elfH"] = M.frontier_ppl_at(md, e["H"])
            row["ratio_ppl_mdlm_over_elf"] = row["mdlm_ppl_at_elfH"] / e["ppl"]
            row["ratio_dedup_mdlm_over_elf"] = row["mdlm_dedup_at_elfH"] / e["dedup"]
        except M.ExtrapolationError:
            row["mdlm_ppl_at_elfH"] = row["mdlm_dedup_at_elfH"] = None
            row["ratio_ppl_mdlm_over_elf"] = row["ratio_dedup_mdlm_over_elf"] = None
        xp.append(row)
KEY["cross_paradigm"] = xp
with open(os.path.join(OUT, "cross_paradigm.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(xp[0])); w.writeheader(); w.writerows(xp)

# Pipeline consistency check: P8 ELF numbers vs P0/P4 harness for the same nominal config
KEY["p8_vs_p4_consistency"] = {
    "elf_ode32": {"p8_ppl": next(e["ppl"] for e in elf8 if e["N"] == 32 and e["gamma"] == 0.0),
                  "p4_ppl": F32[0.0][1], "p8_H_pooled": next(e["H"] for e in elf8 if e["N"] == 32 and e["gamma"] == 0.0),
                  "p4_H_per_sample": F32[0.0][0]},
    "elf_sde32_g1.5": {"p8_ppl": next(e["ppl"] for e in elf8 if e["N"] == 32 and e["gamma"] == 1.5),
                       "p4_ppl": F32[1.5][1]},
    "note": "P8 used its own ELF sampler/scorer (256 samples, pooled entropy). P8 ELF numbers are only comparable within P8."}

json.dump(KEY, open(os.path.join(OUT, "key_numbers.json"), "w"), indent=2, default=float)
print(json.dumps({k: KEY[k] for k in ("spearman_gamma", "ppl_at_H", "gamma3_sc3_seed0", "p8_vs_p4_consistency")}, indent=1, default=float))
print("\nINTERVENTIONS")
for r in res:
    print(f"  {r['label']:<45} ppl {r['ppl']:7.2f} (base {r['base_ppl']:6.2f}, {r['raw_ppl_change_pct']:+5.1f}%)  "
          f"dH {r['dH']:+.3f}  frontier@H {r['frontier_ppl_at_H'] if r['frontier_ppl_at_H'] is None else round(r['frontier_ppl_at_H'],2)}  "
          f"resid {None if r['residual_log'] is None else round(r['residual_log'],3)}  share {None if r['share_explained_by_entropy'] is None else round(r['share_explained_by_entropy'],2)}  "
          f"n={r['n']}/{r['base_n']} mauve {r['mauve']} vs {None if r['base_mauve'] is None else round(r['base_mauve'],3)}  nfe {r['nfe_elf']}/{r['nfe_enc']}  [{r['status']}]")
print("\nCONTROLS S3", json.dumps(KEY["controls_s3"], indent=1))
print("\nCROSS-PARADIGM (P8, pooled entropy)")
for r in xp:
    print(f"  N={r['N']:<3} g={r['elf_gamma']}: ELF ppl {r['elf_ppl']:8.2f} dedup {r['elf_dedup']:8.2f} H {r['elf_H']:.2f} rep4 {r['elf_rep4']:.3f} | "
          f"MDLM@H ppl {r['mdlm_ppl_at_elfH']} dedup {r['mdlm_dedup_at_elfH']} | ratio ppl {r['ratio_ppl_mdlm_over_elf']} dedup {r['ratio_dedup_mdlm_over_elf']} | "
          f"MDLM best-dedup tau {r['mdlm_best_dedup_tau']} {r['mdlm_best_dedup']} H {r['mdlm_best_H']} rep4 {r['mdlm_best_rep4']}")
