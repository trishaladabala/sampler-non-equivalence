# Prompt 5 — Clean Consistency vs. Progressive Distillation Report

## 1. Methodological Fixes Applied
- **Disjoint Splits (`D26`)**: Enforced zero overlap between `DISTILL` (WMT14 train), `SELECT` (newstest2013), and `TEST` (newstest2014).
- **In-Domain Exposure Control (`D-S`)**: Trained with standard flow matching on identical `DISTILL` `(z_0, z_1^ODE)` pairs and step budget.
- **Valid-Token Masking (`D22`)**: All consistency, endpoint, and decoder KL losses exclude source condition tokens and post-EOS padding.

## 2. Seed-0 Evaluation Matrix
| Method | 1-step BLEU | 2-step BLEU | 4-step BLEU | 8-step BLEU | 32-step BLEU |
|---|---:|---:|---:|---:|---:|
| D-T | 0.16 | 0.00 | 0.14 | 0.77 | 3.82 |
| D-C | 0.19 | 0.31 | 0.36 | 0.19 | 0.73 |
| D-P | 0.39 | 0.37 | 0.37 | 0.36 | 0.53 |
| D-S | 0.38 | 0.12 | 0.13 | 0.49 | 0.31 |
| gate_g_distill | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |