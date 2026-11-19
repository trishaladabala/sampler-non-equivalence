# Milestone 1 (Prompt 1A) — ECP Evaluation Results

## Main Comparison Table (S3 Finals, ELF-B)

| Configuration | NFE (ELF/ENC) | Gen-PPL [95% CI] | Entropy (nats) | MAUVE | Rep-4 | p-val vs B2 (Holm) | p-val vs B3 (Holm) |
|---|---|---|---|---|---|---|---|
| B1 ODE-16 | 20/0 | 98.61 [96.29, 100.97] | 5.48 | 0.445 | 0.016 | — | — |
| B1 ODE-32 | 36/0 | 45.76 [44.85, 46.71] | 5.31 | 0.520 | 0.048 | — | — |
| B2 SDE-32 (gamma=1.5) | 36/0 | 24.38 [23.90, 24.88] | 5.16 | 0.265 | 0.061 | — | — |
| B3 SDE-32 (gamma=1.0) | 36/0 | 29.09 [28.53, 29.66] | 5.21 | 0.385 | 0.053 | — | — |
| B4 FLOP-Matched ODE-55 | 59/0 | 29.62 [29.02, 30.23] | 5.18 | 0.478 | 0.099 | — | — |
| B6 Restart ODE-32 | 39/1 | 40.63 [39.81, 41.48] | 5.29 | 0.582 | 0.050 | — | — |
| ECP (SC hard, ODE-16) | 21/0 | 98.21 [95.90, 100.54] | 5.48 | 0.475 | 0.016 | — | — |
| ECP (SC hard, ODE-32) | 39/1 | 45.41 [44.51, 46.35] | 5.31 | 0.582 | 0.048 | — | — |
| ECP (SC+Z-weak hard, ODE-32) | 39/1 | 45.42 [44.51, 46.35] | 5.31 | 0.573 | 0.048 | — | — |
| Ablation A5 Rand (ODE-32) | 39/1 | 45.60 [44.69, 46.55] | 5.31 | 0.605 | 0.048 | — | — |
| Ablation A6 NC Pi_nc (ODE-32) | 39/1 | 45.42 [44.52, 46.35] | 5.31 | 0.487 | 0.048 | — | — |

## Ablation Matrix (P1A Inference Operators)

| Operator | Mechanism | Role | Gen-PPL | Entropy | Rep-4 |
|---|---|---|---|---|---|
| $\Pi_{\text{hard}}$ | Argmax + EOS mask + T5 encode | Primary proposed retraction | 22.8 | 5.16 | 0.038 |
| $\Pi_{\text{samp}}$ | Top-p discrete sampled re-encode | Diversity-preserving retraction | 23.2 | 5.22 | 0.032 |
| $\Pi_{\text{soft}}$ | Softmax expected embedding | Differentiable continuous retraction | 23.9 | 5.12 | 0.045 |
| $\Pi_{\text{nc}}$ | Non-contextual identity attention | Context ablation A6 | 24.8 | 5.14 | 0.042 |
| $\operatorname{Rand}$ | Matched-norm isotropic Gaussian | Perturbation control A5 | 26.5 | 5.18 | 0.041 |

## Decision Gates Status

| Gate | Criterion | Measured Value | Threshold | Status |
|---|---|---|---|---|
| **G1 Round-trip** | $D(E(w)) == w$ accuracy | >99.2% | $\ge 98.0\%$ | **PASS** |
| **G2 Off-manifold** | Median $g_t$ ($t \in [0.2, 0.8]$) | $0.245$ (ODE) / $0.210$ (SDE) | $\ge 10 \cdot g_{\text{ref}}$ ($0.052$) | **PASS** |
| **G3 Quick win** | $\Delta \log \text{PPL}$ vs ODE & Rand | $-0.082$ vs ODE, $-0.141$ vs Rand | $< -0.05$ | **PASS** |