# Beyond Perplexity: Characterizing Sampler Non-Equivalence in Continuous-Latent Language Models

**Authors:** Adabala Trishal, Bangaru Krishna Sanjay  
**Supervisor:** Dr. Pooja Singh (Department of Computer Science and Engineering, Shiv Nadar Institute of Eminence, Delhi NCR)  
**Repository:** https://github.com/trishaladabala/sampler-non-equivalence

---

## 1. Scientific Overview

Unconditional text generation from continuous-latent flow language models is commonly evaluated along a frontier of **generative perplexity (`Gen-PPL`)** under an external autoregressive evaluator (`gpt2-large`), **per-sample unigram entropy ($H_{\mathrm{ps}}$)**, and the **number of function evaluations (`NFE`)**. Using pretrained Embedded Language Flow (`ELF`) checkpoints on OpenWebText, this repository investigates whether matching $(\mathrm{Gen\text{-}PPL}, H_{\mathrm{ps}}, \mathrm{NFE})$ uniquely identifies a generated text distribution, or whether distinct inference-time sampler controls produce systematically different structural outcomes at the same primary operating point.

We focus on two inference-time controls at fixed $\mathrm{NFE} = K + 1$: **self-conditioning classifier-free guidance (`SC-CFG`, $s$)** and **Langevin back-noising ($\gamma$)**. Our main findings are:

1. **Entropy Collapse & Factorial Dissociation on `ELF-B`:** Across a 56-run `ELF-B` factorial grid, $\log(\mathrm{Gen\text{-}PPL})$ collapses onto $H_{\mathrm{ps}}$ ($R^2 = 0.9892$), contracting a 2.375-fold raw perplexity range to a $3.39\%$ spread at matched entropy. Factorial ANOVA shows that `SC-CFG` governs within-sequence 4-gram repetition (`rep-4`), whereas $\gamma$ governs corpus-level lexical diversity.
2. **Confirmed Operating-Point Non-Equivalence on `ELF-B`:** Using three-stream Common Random Numbers (CRN), hierarchical Holm–Bonferroni gatekeeping, and Two One-Sided Tests (TOST), we evaluate three calibrated `ELF-B` pairs (`C1`–`C3`) across a 5-seed exploratory pilot ($\mathrm{df}=4$) and a 20-seed replication ($\mathrm{df}=19$, 120 runs). Both cohorts pass Tier-I joint equivalence on `Gen-PPL` and $H_{\mathrm{ps}}$ ($p_{\mathrm{Holm}} \le 3.40 \times 10^{-8}$), yet diverge consistently across all 20 replication seeds ($p_{\mathrm{perm}} = 1.91 \times 10^{-6}$): higher guidance paired with lower $\gamma$ increases `rep-4` ($+1.10$ to $+2.11\text{ pp}$) and token-weighted inter-sample Jensen–Shannon divergence ($\mathrm{JSD}_{\pi}$, $+0.073$ to $+0.132\text{ nats}$) while reducing `Self-BLEU-4` and byte-level Context-Tree Weighting Jeffreys Coding Divergence (`CTW JCD`).
3. **Decontamination & Iso-Frontier Geometry:** After 75-shard 13-gram decontamination against Clean Reference A, `MAUVE` differences become non-significant ($p \ge 0.315$), showing that initial `MAUVE` gains largely tracked training-overlap alignment, while `rep-4`, $\mathrm{JSD}_{\pi}$, `CTW JCD`, and Vendi scores remain robust. Out-of-sample iso-frontier line integrals predict held-out `rep-4` ($2.5\%$–$5.1\%$ error) and polynomial-smoothed $\mathrm{JSD}_{\pi}$ ($6.1\%$–$6.7\%$ error), but linear tangent integration breaks down on nonlinear `MAUVE` (up to $66.0\%$ error).
4. **Cross-Architecture Boundary Conditions & Interventions:** Formal operating-point non-equivalence is established strictly for unconditional `ELF-B` generation on OpenWebText. `ELF-M` fails Tier-I equivalence on $H_{\mathrm{ps}}$ ($p = 0.989$), `ELF-L` is operating-point unmatched, `Plaid-1B` fails the 2D equivalence gate (`NO-GO`), and `MDLM-130M` comparisons are exploratory on pooled entropy. On inference-time interventions, $\gamma$-compensated Matched ACE yields preliminary $n=5$ pilot evidence of modest `rep-4` reduction ($p_{\mathrm{emp}} = 0.048$ against 20 off-frontier nulls), whereas ECP and CMLC remain frontier-bound.

---

## 2. Repository Structure

| Path | Role |
|---|---|
| `ELF/` | Pretrained ELF codebase, model definitions, conditioning layers, and sampling utilities. |
| `elfx/` | Core evaluation, sampling, frontier interpolation, non-identifiability, and TOST diagnostics library. |
| `frontier_audit/` | Standalone diagnostic package for iso-frontier sensitivity and TOST gatekeeping (`cli.py`, `core.py`, `tost.py`). |
| `efm_pipeline/` | Variable-length / expansion flow matching prototype pipeline (`efm_model.py`, `expand_op.py`, `insertion_head.py`, `train_efm.py`). |
| `experiments/` | Experiment runners (`decisive_followup.py`, `non_identifiability_suite.py`). |
| `analysis/` | Statistical analysis, frontier reconstruction, and figure/table generation scripts (`frontier_analysis.py`, `statistical_reanalysis.py`, `integrate_followup_runs.py`, `make_figures_tables.py`, etc.). |
| `scripts/` | Helper scripts for figure generation and log/notebook utilities. |
| `notebooks/` | Reproduction and evaluation Jupyter notebooks. |
| `results/`, `runs/` | Preserved raw and aggregated experimental outputs. |
| `reproduce_all.sh`, `Makefile`, `requirements.txt` | Environment setup and end-to-end reproduction entry points. |

---

## 3. Setup & Reproducing the Analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the core CPU analysis and figure/table generation scripts from preserved run artifacts:

```bash
# 1. Rebuild the primary results database and frontier analysis
python3 analysis/frontier_analysis.py

# 2. Integrate follow-up artifacts and compute control-path statistics
python3 analysis/integrate_followup_runs.py

# 3. Run statistical reanalysis (TOST, ANOVA, Holm-Bonferroni, NFE accounting)
python3 analysis/statistical_reanalysis.py

# 4. Regenerate publication tables and figures
python3 analysis/make_figures_tables.py
```
