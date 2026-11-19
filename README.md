# Same Perplexity, Same Entropy, Different Text: Operating-Point Non-Equivalence of Sampler Controls in a Continuous-Latent Flow Language Model

**Target Venue:** ACL 2027 Main Conference (ACL Rolling Review / ARR Double-Blind Submission)  
**Compiled Manuscript:** `main.pdf` (`main.tex`, `acl.sty`, `references.bib`)  
**Traceability Matrix:** `RESULT_MAP.csv` (143 verified claims, tables, and figures: 115 numerical claims, 23 tables, 5 figures) & `run_manifest.json` (280 raw run directories)

---

## 1. Scientific Overview

Unconditional generation from diffusion and continuous-latent flow language models is commonly evaluated along a two-dimensional frontier of **generative perplexity (Gen-PPL)** under an external autoregressive evaluator (`gpt2-large`) and **mean within-sample unigram entropy ($H_{\text{ps}}$)**. Using the pretrained Embedded Language Flow (ELF) family (ELF-B 105M, ELF-M 342M, and ELF-L 652M) alongside discrete Masked Diffusion Language Models (MDLM 170M), this repository investigates whether a $(\text{Gen-PPL}, H_{\text{ps}}, \text{NFE})$ operating point uniquely identifies the generated text distribution.

Across 280 experiment run directories (including a balanced $4 \times 7 \times 2$ factorial grid on ELF-B, targeted intermediate-$\gamma$ runs, and a 13-run multi-seed factorial on ELF-M), we establish three core findings:

1. **Operating-Point Non-Equivalence of $(\text{SC-CFG}, \gamma)$:** Matching $\text{Gen-PPL}$, $H_{\text{ps}}$, and exact per-sample function evaluations ($\text{NFE}_{\text{sample}} = K + 1$) between low-guidance/high-back-noising ($\text{SC-CFG}=1$, high $\gamma$) and high-guidance/low-back-noising ($\text{SC-CFG}\in\{3,4\}$, low $\gamma$) leaves a highly reproducible trade-off between within-sequence repetition and cross-sequence diversity. Across all 6 strictly matched pairs (`M1`, `S1`–`S3`, `Refined-1/2`), higher $\text{SC-CFG}$ paired with lower $\gamma$ significantly increases within-sample 4-gram repetition (`rep-4`, seed-level $df=2$ Welch $t$-test $p_{\text{Holm}} \le 0.0385$ in 6/6 pairs) while simultaneously increasing pooled corpus entropy ($H_{\text{pool}}$), token-weighted inter-sample Jensen–Shannon divergence ($\mathrm{JSD}_{\pi} = H_{\text{pool}} - H_{\text{ps,weighted}}$), corpus `distinct-2`, and `MAUVE`, and reducing 4-gram `Self-BLEU` and byte-level Context-Tree Weighting Jeffreys Coding Divergence (`CTW JCD`).
2. **Mechanistic Resolution via Design-Based Two-Way Factorial ANOVA:** In the balanced non-collapsed $4 \times 6 \times 2$ ELF-B grid ($\gamma \le 2.0$), $\text{SC-CFG}$ accounts for **$66.1\%$** of total variance in within-sample `rep-4`, whereas $\gamma$ accounts for **$85.2\%$** of total variance in corpus `distinct-2` (**$91.1\%$** across all 56 runs) and **$92.5\%$** in `MAUVE`. Although increasing $\text{SC-CFG}$ at fixed $\gamma$ mildly reduces corpus diversity (main effect $-0.029$), $\gamma$ contracts cross-sample mode coverage over $3\times$ more steeply per unit of $(\text{Gen-PPL}, H_{\text{ps}})$ reduction, explaining why trading $\gamma$ for $\text{SC-CFG}$ along an iso-$(\text{Gen-PPL}, H_{\text{ps}})$ contour preserves corpus diversity while amplifying local phrase loops.
3. **Diagnostic Audit of Inference-Time Interventions (ECP & CMLC):** Evaluating Encoder-Conditioned Projection (ECP) and Confidence-Modulated Latent Caching (CMLC) at matched entropy shows that apparent raw Gen-PPL reductions are artifacts of entropy contraction ($\le 0.2\%$ residual at matched $H_{\text{ps}}$ in 3-seed finals).

---

## 2. Repository Hierarchy & Traceability

```text
model/checkpoint -> experiment configuration -> independent run/seed -> generated samples -> metrics -> statistical analysis -> table/figure -> manuscript claim
```

| Path | Role |
|---|---|
| `main.tex`, `main.pdf`, `acl.sty`, `references.bib` | ACL 2027 double-blind anonymized two-column manuscript ($\le 8$ pages main body + Limitations, Ethical Considerations, References, and 4 Appendices). |
| `RESULT_MAP.csv` | Automated claim-to-artifact traceability map (115 numerical claims `R01`–`R115`, 23 tables `TAB01`–`TAB23`, 5 figures `FIG1`–`FIG5`). |
| `run_manifest.json` | Machine-readable inventory of all 280 raw run directories, models, checkpoints, and exact NFE accounting. |
| `analysis/statistical_reanalysis.py` | End-to-end statistical reanalysis script: verifies metric identities (`JSD_pi`, `rep-4`, `CTW JCD`, `NFE`), computes TOST 90% CI equivalence margins, seed-level ($df=2$) Holm–Bonferroni & BH-FDR $p$-values, design-based $4 \times 7 \times 2$ & $4 \times 6 \times 2$ two-way ANOVA, and cell-means linear/quadratic response surfaces (`analysis/out/reanalysis_stats.json`). |
| `analysis/frontier_analysis.py` | Rebuilds `results_db.csv` and `analysis/out/key_numbers.json` from raw `summary.json` files. |
| `analysis/integrate_followup_runs.py` | Integrates E1 (ELF-M factorial), E2 (ELF-B extended corpus metrics), and E3 (ELF-B intermediate-$\gamma$ runs) into `analysis/out/control_path_stats.json` and generates `tables/tab_elfm_factorial.tex` and `tables/tab_jcd_selfbleu_pairs.tex`. |
| `analysis/make_figures_tables.py` | Generates all remaining LaTeX tables (`tables/*.tex`) and publication figures (`paper_figures/*.pdf`). |
| `analysis/check_manuscript.py` | Strict verification gate checking all 115 numerical claims, 27 acceptance guards, 23 tables, 5 figures, positioning statements, and banned legacy strings. |
| `experiments/decisive_followup.py` | Ready-to-run script for optional GPU follow-up experiments (`--mode audit` on CPU; `--mode gpu` on CUDA for 5-seed matched-pair expansion, second evaluator `Llama-3.2-1B` / `gpt2-xl`, and blinded human-eval pair export). |
| `human_eval/PROTOCOL.md`, `human_eval/analyze_human_eval.py` | Double-blind human evaluation protocol and exact binomial/McNemar/mixed-effects analysis script. |
| `ELF/` | Pretrained ELF codebase, model definitions, conditioning layers, and sampling utilities. |
| `elfx/` | Core evaluation, sampling, frontier interpolation, and TOST diagnostics library. |
| `frontier_audit/` | Standalone diagnostic package for iso-frontier sensitivity and TOST gatekeeping. |
| `notebooks/` | Standalone reproduction and evaluation Jupyter notebooks (03–21). |
| `tests/` | Comprehensive CPU test suite (170 CPU tests: 148 unittests + 22 estimator tests). |

---

## 3. Reproducing the Analysis, Tables, Figures, and Manuscript (CPU)

All statistical analyses, tables, figures, unit tests, and PDF compilation run deterministically on CPU from the preserved raw experiment artifacts in `content/drive/MyDrive/elf_research/runs/` and `followup_artifacts/`:

```bash
# 1. Regenerate the primary results database and frontier analysis
python3 analysis/frontier_analysis.py

# 2. Integrate E1/E2/E3 follow-up artifacts and regenerate control-path statistics
python3 analysis/integrate_followup_runs.py

# 3. Run the comprehensive ACL 2027 statistical reanalysis (TOST, ANOVA, Holm-Bonferroni, NFE audit)
python3 analysis/statistical_reanalysis.py

# 4. Regenerate all LaTeX tables (tables/*.tex) and figures (paper_figures/*.pdf)
python3 analysis/make_figures_tables.py

# 5. Run the complete CPU test suite (170 CPU tests: 148 unittest + 22 estimator tests)
python3 -m unittest discover -s tests
python3 tests/run_tests.py

# 6. Verify manuscript consistency and regenerate RESULT_MAP.csv (115 numerical claims, 27 acceptance guards, 23 tables, 5 figures, 0 errors)
python3 analysis/check_manuscript.py

# 7. Compile the anonymized ACL 2027 two-column PDF
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
