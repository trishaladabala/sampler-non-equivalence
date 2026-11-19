# Phase 2 At-Scale Report — 130M MDLM vs. 105M ELF-B Stochastic vs. Deterministic Sampling

## 1. Critical Codebase Bugs Fixed in `phase2_mdlm/`
1. **`mdlm_experiment.py` L71 (`causal_conv1d` & Hydra `+sampling.temperature` Crash)**: Replaced the fragile external subprocess call with pure-PyTorch `causal_conv1d` compatibility shims and micro-batched PyTorch ancestral / DDPM-caching unmasking (`sample_mdlm_ancestral`).
2. **`score_mdlm_results.py` L185 (`D26` Survivor Bias)**: Removed the `if not d` filter that dropped degenerate samples before computing perplexity, reporting `corpus_gen_ppl_all` over 100% of samples alongside 4-gram deduplicated `dedup_ppl` (`D6`).
3. **`score_mdlm_results.py` L186 (`D25` Arithmetic Mean PPL)**: Replaced the arithmetic mean of per-sample exponentials with token-weighted corpus perplexity $\exp(\sum_i \text{NLL}_i / \sum_i T_i)$.
4. **Cross-Paradigm Matched-NFE Evaluation**: Evaluated `130M kuleshov-group/mdlm-owt` ($\tau \in \{0.0, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2\}$) and `105M embedded-language-flows/ELF-B-owt-torch` (ODE $\gamma=0.0$ vs SDE $\gamma \in \{1.0, 1.5\}$) at matched step counts $N \in \{4, 8, 16, 32, 64\}$.

## 2. Part B — Audit of Pre-Existing `phase2_mdlm/results/**/*.json`
- **`samples_N16_T0.0.json`**: `{"rep_4": 0.9940944881889764, "distinct_4": 0.005905511811023622, "unigram_entropy": 0.028314716575101546, "degeneracy_rate": 1.0}`
- **`samples_N16_T0.5.json`**: `{"rep_4": 0.30962169309926846, "distinct_4": 0.6903783069007315, "unigram_entropy": 3.722727489447825, "degeneracy_rate": 0.4609375}`

## 3. Part A — 130M `kuleshov-group/mdlm-owt` Temperature x Steps Sweep
| Steps (N) | Temp | Corpus Gen-PPL (All) | Dedup-PPL | Legacy Survivor-Biased PPL | Rep-4 | Unigram Entropy | Degeneracy Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0 | 1.26 | 1.59 | inf | 0.916 | 0.19 | 1.00 |
| 4 | 0.5 | 13.71 | 16.33 | 14.01 | 0.036 | 3.52 | 0.07 |
| 4 | 0.7 | 48.93 | 57.52 | 49.59 | 0.003 | 5.47 | 0.00 |
| 4 | 0.8 | 152.40 | 181.86 | 154.68 | 0.000 | 6.71 | 0.00 |
| 4 | 0.9 | 546.15 | 631.08 | 555.09 | 0.000 | 7.74 | 0.00 |
| 4 | 1.0 | 1966.18 | 2155.15 | 2002.41 | 0.000 | 8.65 | 0.00 |
| 4 | 1.2 | 11964.02 | 12235.69 | 12060.68 | 0.000 | 9.94 | 0.00 |
| 8 | 0.0 | 1.08 | 1.08 | inf | 0.000 | -0.00 | 1.00 |
| 8 | 0.5 | 6.91 | 12.45 | 7.41 | 0.149 | 3.42 | 0.42 |
| 8 | 0.7 | 24.77 | 37.60 | 25.38 | 0.014 | 5.65 | 0.00 |
| 8 | 0.8 | 74.68 | 104.26 | 76.14 | 0.001 | 6.90 | 0.00 |
| 8 | 0.9 | 223.51 | 276.75 | 228.59 | 0.000 | 7.70 | 0.00 |
| 8 | 1.0 | 810.21 | 934.13 | 847.08 | 0.000 | 8.47 | 0.00 |
| 8 | 1.2 | 8958.63 | 9312.34 | 9163.76 | 0.000 | 9.94 | 0.00 |
| 16 | 0.0 | 1.04 | 602.64 | inf | 0.994 | 0.03 | 1.00 |
| 16 | 0.5 | 4.43 | 10.46 | 4.90 | 0.310 | 3.72 | 0.46 |
| 16 | 0.7 | 16.01 | 27.44 | 16.91 | 0.032 | 6.26 | 0.04 |
| 16 | 0.8 | 41.56 | 56.55 | 42.41 | 0.003 | 7.12 | 0.00 |
| 16 | 0.9 | 104.83 | 127.36 | 106.98 | 0.001 | 7.69 | 0.00 |
| 16 | 1.0 | 343.70 | 401.29 | 360.10 | 0.000 | 8.30 | 0.00 |
| 16 | 1.2 | 5988.86 | 6337.81 | 6287.56 | 0.000 | 9.82 | 0.00 |
| 32 | 0.0 | 1.04 | 602.64 | inf | 0.994 | 0.03 | 1.00 |
| 32 | 0.5 | 3.41 | 9.55 | 4.07 | 0.455 | 4.27 | 0.64 |
| 32 | 0.7 | 12.20 | 19.76 | 12.83 | 0.049 | 6.53 | 0.02 |
| 32 | 0.8 | 28.01 | 35.84 | 28.74 | 0.005 | 7.14 | 0.00 |
| 32 | 0.9 | 65.97 | 78.63 | 67.63 | 0.001 | 7.63 | 0.00 |
| 32 | 1.0 | 194.25 | 227.47 | 203.70 | 0.000 | 8.20 | 0.00 |
| 32 | 1.2 | 3844.26 | 4148.53 | 4217.99 | 0.000 | 9.67 | 0.00 |
| 64 | 0.0 | 1.04 | 602.64 | inf | 0.994 | 0.03 | 1.00 |
| 64 | 0.5 | 2.90 | 8.86 | 3.53 | 0.560 | 4.67 | 0.71 |
| 64 | 0.7 | 10.55 | 16.94 | 11.41 | 0.056 | 6.58 | 0.03 |
| 64 | 0.8 | 22.99 | 28.72 | 23.66 | 0.007 | 7.15 | 0.00 |
| 64 | 0.9 | 50.14 | 59.38 | 51.41 | 0.001 | 7.58 | 0.00 |
| 64 | 1.0 | 139.04 | 163.72 | 146.40 | 0.000 | 8.13 | 0.00 |
| 64 | 1.2 | 2842.62 | 3084.75 | 3167.03 | 0.000 | 9.55 | 0.00 |

## 4. Part C.1 — 105M `ELF-B-owt` ODE vs. SDE Matched-NFE Sweep
| Steps (N) | Gamma | Mode | Corpus Gen-PPL (All) | Dedup-PPL | Legacy Survivor-Biased PPL | Rep-4 | Unigram Entropy | Degeneracy Rate |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 4 | 0.0 | ODE | 351.35 | 346.91 | 363.70 | 0.000 | 7.93 | 0.00 |
| 4 | 1.0 | SDE(gamma=1.0) | 11.34 | 13.49 | 12.11 | 0.101 | 4.31 | 0.04 |
| 4 | 1.5 | SDE(gamma=1.5) | 2.82 | 6.17 | inf | 0.644 | 1.81 | 1.00 |
| 8 | 0.0 | ODE | 181.37 | 180.79 | 189.11 | 0.000 | 7.86 | 0.00 |
| 8 | 1.0 | SDE(gamma=1.0) | 36.44 | 38.60 | 40.08 | 0.023 | 6.40 | 0.00 |
| 8 | 1.5 | SDE(gamma=1.5) | 2.99 | 5.30 | 3.74 | 0.501 | 2.13 | 0.86 |
| 16 | 0.0 | ODE | 79.64 | 82.30 | 83.62 | 0.004 | 7.76 | 0.00 |
| 16 | 1.0 | SDE(gamma=1.0) | 44.59 | 47.49 | 46.65 | 0.008 | 7.40 | 0.00 |
| 16 | 1.5 | SDE(gamma=1.5) | 8.98 | 14.58 | 10.96 | 0.247 | 3.92 | 0.20 |
| 32 | 0.0 | ODE | 41.93 | 49.11 | 43.95 | 0.022 | 7.64 | 0.00 |
| 32 | 1.0 | SDE(gamma=1.0) | 26.03 | 32.31 | 27.28 | 0.031 | 7.39 | 0.00 |
| 32 | 1.5 | SDE(gamma=1.5) | 49.91 | 59.85 | 51.95 | 0.032 | 7.26 | 0.00 |
| 64 | 0.0 | ODE | 26.37 | 40.49 | 27.88 | 0.073 | 7.57 | 0.00 |
| 64 | 1.0 | SDE(gamma=1.0) | 18.40 | 29.79 | 19.20 | 0.088 | 7.40 | 0.00 |
| 64 | 1.5 | SDE(gamma=1.5) | 19.04 | 31.75 | 19.96 | 0.092 | 7.34 | 0.00 |

## 5. Part C.2 — Cross-Paradigm Comparison (130M MDLM vs. 105M ELF-B at Matched NFEs)
| Steps (N) | MDLM Det (tau=0) PPL | MDLM Stoch (tau=0.9) PPL | MDLM Stoch (tau=1.0) PPL | ELF-B ODE (gamma=0) PPL | ELF-B SDE (gamma=1.0) PPL | ELF-B SDE (gamma=1.5) PPL |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.26 | 546.15 | 1966.18 | 351.35 | 11.34 | 2.82 |
| 8 | 1.08 | 223.51 | 810.21 | 181.37 | 36.44 | 2.99 |
| 16 | 1.04 | 104.83 | 343.70 | 79.64 | 44.59 | 8.98 |
| 32 | 1.04 | 65.97 | 194.25 | 41.93 | 26.03 | 49.91 |
| 64 | 1.04 | 50.14 | 139.04 | 26.37 | 18.40 | 19.04 |
