# Reproducibility (`REPRODUCIBILITY.md`)

**Revised:** 2026-10-29. All commands run from `the repository root directory` with the system `python3` (NumPy and Matplotlib only; no GPU, no network).

## 1. Regenerate every number, table and figure

```bash
python3 analysis/collect_runs.py          # 406 summary.json -> analysis/results_db.csv
python3 -m unittest discover -s tests     # 148 unit tests (expected: 148 passed, 0 failed)
python3 tests/run_tests.py                # 22 estimator tests (expected: 22 passed, 0 failed; 170 CPU tests total)
python3 analysis/frontier_analysis.py     # -> analysis/out/key_numbers.json and CSVs
python3 analysis/make_figures_tables.py   # -> tables/*.tex, paper_figures/*.{pdf,png}
python3 analysis/check_manuscript.py      # 115 numeric claims + 27 acceptance guards -> RESULT_MAP.csv (143 rows)
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

`check_manuscript.py` fails in any of these cases:

- a quoted number drifts from `key_numbers.json`;
- a table or figure is not included;
- a banned fabricated value reappears (22.15, 0.892, 43.4%, 98.4%, …);
- the manuscript claims the frontier as its contribution ("frontier residual", "novel frontier", …);
- an obsolete or incorrect statement reappears (`t_on=0.60`, "right-padding", "0.06 nats apart", seed-0 value 151.31).

## 2. Estimators

These are defined in `analysis/metrics_ref.py` and tested in `tests/test_ppl_estimators.py`.

- **Corpus perplexity:** token-weighted, with EOS/pad masking. This is the harness definition; see `score_texts` in `elf_p0_harness.ipynb`.
- **Per-sample unigram entropy:** averaged over samples. Pooled entropy differs from it, and a dedicated test shows this.
- **Entropy-matched values:** linear interpolation of log-PPL in H between bracketing operating points. There is no extrapolation (`ExtrapolationError`).
- **Per-run 95% CI on Gen-PPL:** taken from `ppl_ci_lo`/`ppl_ci_hi` in each `summary.json`. These come from a bootstrap with B=10,000 resampling of samples, implemented in the harness `bootstrap_ci`.

## 3. What cannot be reproduced from this repository

The harness writes `samples.jsonl` and `tokens.npz` into each run directory (`_save_samples` in `elf_p0_harness.ipynb`). **No such files are present in this repository copy** (`find content -name samples.jsonl` returns nothing). If they still exist on the original Google Drive (`/content/drive/MyDrive/elf_research/runs/<run_id>/`), the following become CPU- or scoring-only tasks rather than generation tasks:

- intervals on interpolated matched-entropy values and on rep-4/MAUVE differences;
- pooled entropy for the main sweep;
- self-BLEU and other diversity metrics;
- rescoring with a second LM;
- a MAUVE sample-size calibration.

Otherwise these require regeneration (`EXPERIMENTS_REQUIRED.md`).

Not reproducible here at all: any new generation; the GPT-2 anchor anomaly (cause unknown); the Phase-1 5.68 value (samples not stored).
