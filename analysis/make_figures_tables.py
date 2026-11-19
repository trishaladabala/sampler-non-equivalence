"""Generate all manuscript figures and LaTeX tables from analysis outputs (no hand-entered numbers).

Usage: python3 analysis/frontier_analysis.py && python3 analysis/make_figures_tables.py
Inputs : analysis/out/{key_numbers.json, elfb_frontier.csv, interventions.csv, cross_paradigm.csv},
         analysis/results_db.csv
Outputs: paper_figures/fig_{collapse,matched_entropy,interventions,cross_paradigm}.pdf
         tables/*.tex
"""
import csv
import json
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
FIG = os.path.join(ROOT, "paper_figures")
TAB = os.path.join(ROOT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
K = json.load(open(os.path.join(OUT, "key_numbers.json")))
front = list(csv.DictReader(open(os.path.join(OUT, "elfb_frontier.csv"))))
db = list(csv.DictReader(open(os.path.join(HERE, "results_db.csv"))))
inter = list(csv.DictReader(open(os.path.join(OUT, "interventions.csv"))))
xp = list(csv.DictReader(open(os.path.join(OUT, "cross_paradigm.csv"))))
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
SC_COLORS = {1.0: "#1b9e77", 2.0: "#d95f02", 3.0: "#7570b3", 4.0: "#e7298a"}


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ Figure 1: collapse
p4 = [r for r in db if r["run_id"].startswith("p4_elfb_s32_") and fl(r["gamma"]) in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0) and int(fl(r["seed"])) in (0, 1)]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axes[0]
for sc, col in SC_COLORS.items():
    rs = [r for r in p4 if fl(r["sc_cfg"]) == sc]
    ax.scatter([fl(r["gamma"]) for r in rs], [fl(r["gen_ppl"]) for r in rs], s=12, color=col, label=f"SC-CFG={sc:g}")
ax.set_yscale("log"); ax.set_xlabel(r"back-noising $\gamma$"); ax.set_ylabel("Gen-PPL (GPT-2 Large)")
ax.set_title("(a) Raw Gen-PPL vs. sampler knob"); ax.legend(frameon=False)
ax = axes[1]
for sc, col in SC_COLORS.items():
    rs = [r for r in p4 if fl(r["sc_cfg"]) == sc]
    ax.scatter([fl(r["entropy"]) for r in rs], [fl(r["gen_ppl"]) for r in rs], s=12, color=col)
ref = K["owt_reference"]
ax.axvline(ref["entropy"], color="grey", ls=":", lw=1)
ax.text(ref["entropy"], 18, " OWT ref.", color="grey", fontsize=7, rotation=90, va="bottom")
r2 = K["collapse"]["logppl_on_H_deg2"]["R2"]
ax.set_yscale("log"); ax.set_xlabel("per-sample unigram entropy $H$ (nats)")
ax.set_title(f"(b) Same 56 runs vs. entropy (quadratic $R^2$={r2:.3f})")
fig.savefig(os.path.join(FIG, "fig_collapse.pdf")); fig.savefig(os.path.join(FIG, "fig_collapse.png"))
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: matched-entropy higher-order metrics
C = K["collapse"]
scs = [1.0, 2.0, 3.0, 4.0]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9)); fig.subplots_adjust(wspace=0.55)
for ax, (key, lab) in zip(axes, [("ppl_at_H5.20_by_sc", "Gen-PPL"), ("rep4_at_H5.20_by_sc", "rep-4"),
                                 ("distinct2_at_H5.20_by_sc", "corpus distinct-2"), ("mauve_at_H5.20_by_sc", "MAUVE")]):
    vals = [C[key][f"{s}"] if f"{s}" in C[key] else C[key][str(s)] for s in scs]
    ax.plot(scs, vals, "o-", color="k", ms=3)
    ax.set_xticks(scs); ax.set_xlabel("SC-CFG"); ax.set_title(lab)
    if key == "ppl_at_H5.20_by_sc":
        ax.set_ylim(0, max(vals) * 1.3)
fig.suptitle("ELF-B, 32 steps, all quantities interpolated to matched entropy $H$=5.20 (2 seeds)", y=1.08, fontsize=8.5)
fig.savefig(os.path.join(FIG, "fig_matched_entropy.pdf")); fig.savefig(os.path.join(FIG, "fig_matched_entropy.png"))
plt.close(fig)

# ------------------------------------------------------------------ Figure 3: interventions vs γ frontier
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
for ax, steps, fkey in ((axes[0], 16, "baseline_frontier_16"), (axes[1], 32, "baseline_frontier_32")):
    F = sorted((v[0], v[1], float(g)) for g, v in K[fkey].items())
    ax.plot([p[0] for p in F], [p[1] for p in F], "k-o", ms=3, label=r"baseline, $\gamma$ sweep (ODE/SDE)")
    for h, p, g in F:
        ax.annotate(f"$\\gamma$={g:g}", (h, p), fontsize=6, xytext=(2, 2), textcoords="offset points")
    seen = set()
    for r in inter:
        lab = r["label"]
        if f"-{steps}" not in lab:
            continue
        s3 = "[S3]" in lab
        if lab.startswith("ECP") and not s3:
            mk, col, grp = "s", "#d95f02", r"ECP, $t_{\mathrm{on}}$=0.30 (1 seed, $n\leq$512)"
        elif lab.startswith("ECP"):
            mk, col, grp = "D", "#e7298a", r"ECP, $t_{\mathrm{on}}$=0.50 (3 seeds)"
        elif lab.startswith("Control"):
            mk, col, grp = "x", "#666666", r"controls A5/A6, $t_{\mathrm{on}}$=0.50 (3 seeds)"
        elif lab.startswith("Baseline B6"):
            mk, col, grp = "P", "#1f78b4", r"restart, $t_{\mathrm{on}}$=0.50 (3 seeds)"
        else:
            mk, col, grp = "^", "#7570b3", "CMLC R3 (fine-tuned, 3 seeds)"
        ax.scatter(fl(r["H"]), fl(r["ppl"]), marker=mk, color=col, s=22, zorder=3,
                   label=grp if grp not in seen else None)
        seen.add(grp)
    ax.set_yscale("log"); ax.set_xlabel("per-sample unigram entropy $H$ (nats)")
    ax.set_title(f"({'a' if steps == 16 else 'b'}) {steps} sampling steps")
axes[0].set_ylabel("Gen-PPL")
_h, _l = {}, []
for ax in axes:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in _h:
            _h[ll] = hh; _l.append(ll)
fig.legend([_h[x] for x in _l], _l, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
fig.subplots_adjust(bottom=0.32, wspace=0.3)
fig.savefig(os.path.join(FIG, "fig_interventions.pdf")); fig.savefig(os.path.join(FIG, "fig_interventions.png"))
plt.close(fig)

# ------------------------------------------------------------------ Figure 4: P8 cross-paradigm (pooled entropy)
mdlm_rows = []
import re  # noqa: E402
rep = open(os.path.join(ROOT, "content/drive/MyDrive/elf_research/runs/mdlm_stochastic/_report.md")).read()
sec = rep.split("Part A", 1)[1]
for l in sec.splitlines():
    if l.startswith("|") and re.match(r"\|\s*\d", l):
        c = [x.strip() for x in l.strip("|").split("|")]
        mdlm_rows.append(dict(N=int(c[0]), tau=float(c[1]), ppl=float(c[2]), H=float(c[6])))
    elif mdlm_rows and not l.startswith("|"):
        break
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
fig.subplots_adjust(wspace=0.14)
custom_offsets = {
    (8, 0.0): (-30, -8),
    (8, 1.0): (-30, 6),
    (8, 1.5): (8, 4),
    (32, 0.0): (-32, -8),
    (32, 1.0): (-32, -12),
    (32, 1.5): (-34, 7),
    (64, 0.0): (-32, 7),
    (64, 1.0): (-28, -13),
    (64, 1.5): (-36, 4),
}
for ax, N in zip(axes, (8, 32, 64)):
    m = sorted((r["H"], r["ppl"], r["tau"]) for r in mdlm_rows if r["N"] == N and r["tau"] > 0)
    ax.plot([x[0] for x in m], [x[1] for x in m], "o-", color="#1b9e77", ms=3.5, lw=1.2, label=r"MDLM-130M ($\tau$ sweep)")
    e = [r for r in xp if int(r["N"]) == N]
    ax.scatter([fl(r["elf_H"]) for r in e], [fl(r["elf_ppl"]) for r in e], marker="s", color="#d95f02", s=24, zorder=4,
               label=r"ELF-B-105M ($\gamma\in\{0,1,1.5\}$)")
    for r in e:
        g_val = fl(r["elf_gamma"])
        dx, dy = custom_offsets.get((N, g_val), (-25, 4))
        ax.annotate(
            f"$\\gamma$={g_val:g}",
            (fl(r["elf_H"]), fl(r["elf_ppl"])),
            fontsize=6.5,
            xytext=(dx, dy),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="#d95f02", lw=0.5, alpha=0.75, shrinkA=0, shrinkB=2),
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
        )
    ax.set_yscale("log")
    ax.set_xlim(1.3, 8.5)
    ax.set_title(f"$N={N}$ steps")
    ax.set_xlabel("Pooled entropy $H_{\\mathrm{pool}}$ (nats)")
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
axes[0].set_ylabel("Gen-PPL (GPT-2 Large)")
axes[0].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, fontsize=6.5, loc="upper left")
fig.savefig(os.path.join(FIG, "fig_cross_paradigm.pdf"))
fig.savefig(os.path.join(FIG, "fig_cross_paradigm.png"))
plt.close(fig)


# ------------------------------------------------------------------ Tables
def w(name, s):
    open(os.path.join(TAB, name), "w").write("% AUTO-GENERATED by analysis/make_figures_tables.py -- do not edit\n" + s)


# T1: ELF-B frontier at SC-CFG=3 (and summary over SC)
lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"$\gamma$ & Gen-PPL & $H$ (nats) & rep-4 & distinct-2 & MAUVE \\", r"\midrule"]
for r in front:
    if fl(r["sc_cfg"]) == 3.0:
        lines.append(f"{fl(r['gamma']):g} & {fl(r['ppl']):.2f} & {fl(r['H']):.3f} & {fl(r['rep4']):.3f} & {fl(r['distinct2']):.3f} & {fl(r['mauve']):.3f} \\\\")
o = K["owt_reference"]
lines += [r"\midrule", f"OWT ref. & {o['gen_ppl']:.2f} & {o['entropy']:.3f} & {o['rep4']:.3f} & {o['distinct2']:.3f} & -- \\\\",
          r"\bottomrule", r"\end{tabular}"]
w("tab_gamma_sweep.tex", "\n".join(lines) + "\n")

# T2: matched-entropy table across guidance
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"SC-CFG & 1 & 2 & 3 & 4 \\", r"\midrule"]
raw = C["raw_ppl_gamma0_by_sc"]
lines.append("Gen-PPL at $\\gamma{=}0$ (raw) & " + " & ".join(f"{raw[str(s)]:.1f}" for s in scs) + r" \\")
lines.append(r"\midrule \multicolumn{5}{l}{\emph{interpolated to matched entropy $H=5.15$}} \\")
lines.append("Gen-PPL & " + " & ".join(f"{K['ppl_at_H'][f'ELF-B_sc{s:g}']['5.15']:.2f}" for s in scs) + r" \\")
lines.append(r"\multicolumn{5}{l}{\emph{interpolated to matched entropy $H=5.20$}} \\")
for key, lab, fmt in (("ppl_at_H5.20_by_sc", "Gen-PPL", "{:.2f}"), ("rep4_at_H5.20_by_sc", "rep-4", "{:.3f}"),
                      ("distinct2_at_H5.20_by_sc", "corpus distinct-2", "{:.3f}"), ("mauve_at_H5.20_by_sc", "MAUVE", "{:.3f}")):
    lines.append(lab + " & " + " & ".join(fmt.format(C[key][str(s)]) for s in scs) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
w("tab_matched_entropy.tex", "\n".join(lines) + "\n")

# T3: interventions
lines = [r"\begin{tabular}{lrrrrrrrl}", r"\toprule",
         r"Configuration & seeds & $n$ & NFE (flow/enc) & Gen-PPL & $\Delta H$ & MAUVE & baseline PPL at $H$ & entropy-adjusted change \\", r"\midrule"]
groups = [(r"(a) ECP screening runs, $t_{\mathrm{on}}{=}0.30$, single seed", lambda L: L.startswith("ECP") and "[S3]" not in L),
          (r"(b) ECP finals with controls, $t_{\mathrm{on}}{=}0.50$, three seeds", lambda L: "[S3]" in L),
          ("(c) CMLC, fine-tuned R3 model, three seeds", lambda L: L.startswith("CMLC"))]
ordered = []
for gi, (gname, pred) in enumerate(groups):
    if gi:
        ordered.append(None)
    ordered.append(gname)
    ordered += [r for r in inter if pred(r["label"])]
assert sum(isinstance(x, dict) for x in ordered) == len(inter), "every intervention row must belong to a group"
for r in ordered:
    if r is None:
        lines.append(r"\midrule"); continue
    if isinstance(r, str):
        lines.append(r"\multicolumn{9}{l}{\emph{" + r + r"}} \\"); continue
    resid = r["residual_log"]
    res_s = f"{100 * (math.exp(fl(resid)) - 1):+.1f}\\%" if resid not in ("", None) else r["status"].split(" (")[0].replace("pareto-dominated by gamma-tuned baseline", "Pareto-dominated").replace("outside frontier range; not assessable without new runs", "out of range")
    fp = r["frontier_ppl_at_H"]
    fp_s = f"{fl(fp):.2f}" if fp not in ("", None) else "--"
    lab = (r["label"].replace(" [S3]", "").replace(", t_on=0.30", "").replace(", t_on=0.50", "")
           .replace("g=", "$\\gamma{=}$").replace("k=2", "$k{=}2$"))
    mv_s = f"{fl(r['mauve']):.3f}" if r["mauve"] not in ("", None) else "--"
    lines.append(f"{lab} & {r['n_seeds']} & {r['n']} & {fl(r['nfe_elf']):.0f}/{fl(r['nfe_enc']):.0f} & {fl(r['ppl']):.2f} & {fl(r['dH']):+.3f} & {mv_s} & {fp_s} & {res_s} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
w("tab_interventions.tex", "\n".join(lines) + "\n")

# T4: cross-paradigm at matched pooled entropy
lines = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
         r"$N$ & ELF $\gamma$ & ELF Gen-PPL & ELF dedup-PPL & ELF rep-4 & pooled $H$ & MDLM Gen-PPL at $H$ & MDLM dedup-PPL at $H$ \\", r"\midrule"]
for r in xp:
    if r["mdlm_ppl_at_elfH"] in ("", None):
        mp = md = "out of range"
    else:
        mp, md = f"{fl(r['mdlm_ppl_at_elfH']):.1f}", f"{fl(r['mdlm_dedup_at_elfH']):.1f}"
    lines.append(f"{r['N']} & {fl(r['elf_gamma']):g} & {fl(r['elf_ppl']):.2f} & {fl(r['elf_dedup']):.2f} & {fl(r['elf_rep4']):.3f} & {fl(r['elf_H']):.2f} & {mp} & {md} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
w("tab_cross_paradigm.tex", "\n".join(lines) + "\n")

# T5: PPL@H scaling
lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model (steps) & observed $H$ range & PPL@5.15 & PPL@5.20 & PPL@5.24 \\", r"\midrule"]
for tag, lab in (("ELF-B_sc3", "ELF-B 105M (32)"), ("ELF-M", "ELF-M 342M (64)"), ("ELF-L", "ELF-L 652M (64)")):
    d = K["ppl_at_H"][tag]
    cell = lambda v: f"{v:.2f}" if v is not None else "n/r"  # noqa: E731
    lines.append(f"{lab} & [{d['H_range'][0]:.3f}, {d['H_range'][1]:.3f}] & {cell(d['5.15'])} & {cell(d['5.20'])} & {cell(d['5.24'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
w("tab_scale.tex", "\n".join(lines) + "\n")

# T6, T7, T8 from control_path_stats.json
cp_path = os.path.join(OUT, "control_path_stats.json")
if os.path.exists(cp_path):
    CP = json.load(open(cp_path))
    # T6: Directly matched observed pairs (no interpolation)
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Pair $(s_1,\gamma_1)$ vs.\ $(s_2,\gamma_2)$ & Tier & $\Delta H$ (nats) & $\text{PPL}_1 / \text{PPL}_2$ ($\Delta\%$) & $\text{rep-4}_1 / \text{rep-4}_2$ & $\Delta\text{rep-4}$ (s0/s1) & $\Delta\text{distinct-2}$ & $\text{MAUVE}_1 / \text{MAUVE}_2$ & $\Delta\text{MAUVE}$ (s0/s1) \\",
        r"\midrule",
    ]
    # Sort strict first, then by abs(d_ent)
    dp_sorted = sorted(CP["direct_matched_pairs"], key=lambda r: (0 if "strict" in r["tier"] else 1, abs(r["d_ent"])))
    for r in dp_sorted:
        tier_s = "Strict" if "strict" in r["tier"] else "Broad"
        pair_s = f"$({r['sc1']},{r['g1']:g})$ vs.\\ $({r['sc2']},{r['g2']:g})$"
        dh_s = f"${r['d_ent']:+.4f}$"
        ppl_s = f"{r['ppl1']:.2f}/{r['ppl2']:.2f} ({r['rel_ppl_pct']:.1f}\\%)"
        rep_s = f"{r['rep4_1_pct']:.2f}\\%/{r['rep4_2_pct']:.2f}\\%"
        drep_s = f"${r['d_rep4_pp']:+.2f}$ (${r['d_rep4_s0_pp']:+.2f}/{r['d_rep4_s1_pp']:+.2f}$)"
        dd2_s = f"${r['d_d2']:+.4f}$"
        mv_s = f"{r['mauve1']:.3f}/{r['mauve2']:.3f}"
        dmv_s = f"${r['d_mauve']:+.3f}$ (${r['d_mauve_s0']:+.2f}/{r['d_mauve_s1']:+.2f}$)"
        lines.append(f"{pair_s} & {tier_s} & {dh_s} & {ppl_s} & {rep_s} & {drep_s} & {dd2_s} & {mv_s} & {dmv_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tab_direct_pairs.tex", "\n".join(lines) + "\n")

    # T7: Cell-level ANCOVA table (non-collapsed N=24 cells, gamma<=2.0)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Outcome $Y$ ($N{=}24$ cells, $\gamma\le 2$) & $R^2(\mathcal{M}_0)$ & $R^2(\mathcal{M}_1)$ & Partial $R^2(\text{SC})$ & $\hat{\beta}_{\text{sc}}$ ($t$-stat) & $p(\text{SC-CFG})$ & Full-path $F_{3,17}$ ($p$) \\",
        r"\midrule",
    ]
    label_map = {
        "rep4": "Within-sample rep-4",
        "distinct1": "Corpus distinct-1",
        "distinct2": "Corpus distinct-2",
        "distinct3": "Corpus distinct-3",
        "mauve": "MAUVE ($n{=}1{,}000$)",
        "ratio_90_10": "Per-sample PPL tail $p_{90}/p_{10}$",
        "mean_gpt2_len": "Mean GPT-2 length",
    }
    for r in CP["ancova"]:
        if r["subset"] != "non_collapsed_24_cells_g_le_2":
            continue
        ylab = label_map.get(r["outcome"], r["outcome"])
        b_fmt = f"${r['beta_sc_cell']:+.5f}$ (${r['t_sc_cell']:+.2f}$)"
        p_sc = f"${r['p_sc_cell']:.1e}$"
        f_full = f"${r['F_full_cell']:.1f}$ (${r['p_full_cell']:.1e}$)"
        lines.append(
            f"{ylab} & {r['r2_frontier_cell']:.4f} & {r['r2_plus_sc_cell']:.4f} & {r['partial_r2_sc_cell']:.3f} & {b_fmt} & {p_sc} & {f_full} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tab_ancova.tex", "\n".join(lines) + "\n")

    # T8: Constrained sampler selection reversal
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Constraint $(\text{PPL}\le\tau,\ H\ge h_0)$ & Objective & Selected $(s,\gamma)$ & Gen-PPL & $H$ (nats) & rep-4 & distinct-2 & MAUVE \\",
        r"\midrule",
    ]
    prev_reg = None
    for r in CP["constrained_decisions"]:
        reg = f"$\\text{{PPL}}\\le {int(r['ppl_max'])},\\ H\\ge {r['ent_min']:.2f}$ ({r['n_feasible']} cells)"
        if prev_reg is not None and reg != prev_reg:
            lines.append(r"\midrule")
        reg_disp = reg if reg != prev_reg else ""
        prev_reg = reg
        lines.append(
            f"{reg_disp} & {r['objective']} & $({r['sc_cfg']},{r['gamma']:.2f})$ & {r['gen_ppl']:.2f} & {r['entropy']:.3f} & {r['rep4_pct']:.2f}\\% & {r['distinct2']:.4f} & {r['mauve']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tab_decisions.tex", "\n".join(lines) + "\n")

print("figures:", sorted(x for x in os.listdir(FIG) if x.startswith("fig_") and x.endswith(".pdf")))
print("tables:", sorted(os.listdir(TAB)))
