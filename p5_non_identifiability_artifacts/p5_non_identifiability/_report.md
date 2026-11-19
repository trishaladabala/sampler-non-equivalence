# Follow-Up Experiments Report: Sampler-Control Non-Identifiability (E1–E4)
**Generated**: 2026-10-29 19:18:48 UTC

## 1. Priority 1 (E1): `ELF-M` (342M, 64 Steps) Scale-Transfer Factorial
| SC-CFG | $\gamma$ | Seeds | Gen-PPL | $H_{\text{ps}}$ | $H_{\text{pool}}$ | $\text{JSD}_{\text{inter}}$ | rep-4 (%) | distinct-2 | Self-BLEU | JCD (bpb) | MAUVE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 1.50 | 2 | 27.14 | 5.259 | 6.986 | 1.729 | 2.65% | 0.3097 | 42.55 | 0.0731 | 0.491 |
| 1.0 | 2.00 | 2 | 23.01 | 5.210 | 6.890 | 1.680 | 2.95% | 0.2857 | 45.46 | 0.0830 | 0.366 |
| 3.0 | 0.00 | 1 | 28.63 | 5.236 | nan | nan | 6.50% | 0.3452 | nan | nan | 0.641 |
| 3.0 | 0.50 | 1 | 24.49 | 5.209 | nan | nan | 6.07% | 0.3269 | nan | nan | 0.570 |
| 3.0 | 1.00 | 1 | 21.81 | 5.185 | nan | nan | 5.79% | 0.3109 | nan | nan | 0.462 |
| 3.0 | 1.50 | 1 | 19.78 | 5.155 | nan | nan | 5.59% | 0.2937 | nan | nan | 0.358 |
| 3.0 | 2.00 | 1 | 18.28 | 5.125 | nan | nan | 5.63% | 0.2773 | nan | nan | 0.338 |
| 4.0 | 0.50 | 2 | 23.15 | 5.190 | 7.146 | 1.957 | 6.42% | 0.3211 | 39.11 | 0.0734 | 0.535 |
| 4.0 | 1.00 | 2 | 20.38 | 5.164 | 7.077 | 1.914 | 6.34% | 0.3043 | 40.92 | 0.0745 | 0.472 |

### Direct Non-Interpolated Matched Pairs on `ELF-M` (342M)
- `(1, 1.50)` vs `(3, 0.00)`: $\Delta H_{\text{ps}} = -0.0226$, $\Delta\text{PPL} = +1.49$ (CI overlap=False), $\Delta\text{rep-4} = +3.85\text{ pp}$, $\Delta\text{distinct-2} = +0.0355$, $\Delta\text{MAUVE} = +0.150$, $\Delta\text{JCD} = +nan\text{ bpb}$.
- `(1, 1.50)` vs `(3, 0.50)`: $\Delta H_{\text{ps}} = -0.0493$, $\Delta\text{PPL} = -2.65$ (CI overlap=False), $\Delta\text{rep-4} = +3.42\text{ pp}$, $\Delta\text{distinct-2} = +0.0172$, $\Delta\text{MAUVE} = +0.079$, $\Delta\text{JCD} = +nan\text{ bpb}$.
- `(1, 2.00)` vs `(3, 0.50)`: $\Delta H_{\text{ps}} = -0.0002$, $\Delta\text{PPL} = +1.49$ (CI overlap=False), $\Delta\text{rep-4} = +3.12\text{ pp}$, $\Delta\text{distinct-2} = +0.0412$, $\Delta\text{MAUVE} = +0.205$, $\Delta\text{JCD} = +nan\text{ bpb}$.
- `(1, 2.00)` vs `(3, 1.00)`: $\Delta H_{\text{ps}} = -0.0246$, $\Delta\text{PPL} = -1.20$ (CI overlap=False), $\Delta\text{rep-4} = +2.84\text{ pp}$, $\Delta\text{distinct-2} = +0.0253$, $\Delta\text{MAUVE} = +0.097$, $\Delta\text{JCD} = +nan\text{ bpb}$.
- `(1, 2.00)` vs `(4, 0.50)`: $\Delta H_{\text{ps}} = -0.0192$, $\Delta\text{PPL} = +0.15$ (CI overlap=True), $\Delta\text{rep-4} = +3.47\text{ pp}$, $\Delta\text{distinct-2} = +0.0354$, $\Delta\text{MAUVE} = +0.169$, $\Delta\text{JCD} = -0.0097\text{ bpb}$.
- `(3, 1.00)` vs `(4, 0.50)`: $\Delta H_{\text{ps}} = +0.0054$, $\Delta\text{PPL} = +1.34$ (CI overlap=False), $\Delta\text{rep-4} = +0.63\text{ pp}$, $\Delta\text{distinct-2} = +0.0102$, $\Delta\text{MAUVE} = +0.073$, $\Delta\text{JCD} = +nan\text{ bpb}$.
- `(3, 1.50)` vs `(4, 1.00)`: $\Delta H_{\text{ps}} = +0.0098$, $\Delta\text{PPL} = +0.60$ (CI overlap=True), $\Delta\text{rep-4} = +0.75\text{ pp}$, $\Delta\text{distinct-2} = +0.0106$, $\Delta\text{MAUVE} = +0.115$, $\Delta\text{JCD} = +nan\text{ bpb}$.

## 2. Priority 2 (E2): Sample-Level Rescoring of Matched `ELF-B` Pairs (`JCD`, `Self-BLEU`, $H_{\text{pool}}$, `JSD`)
| Pair | Tier | Low Path | High Path | $\Delta H_{\text{ps}}$ | $\Delta H_{\text{pool}}$ | $\Delta\text{JSD}_{\text{inter}}$ | $\Delta\text{rep-4}$ (pp) | $\Delta\text{distinct-2}_{\text{cross}}$ | $\Delta\text{Self-BLEU}$ | $\Delta\text{JCD}$ (bpb) | $\Delta\text{MAUVE}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | Strict | SC=1, gamma=2.00 | SC=4, gamma=1.50 | +0.0047 | +0.0924 | +0.0877 | +1.83 ($p=0.0e+00$) | +0.0138 | -4.25 | -0.0261 | +0.076 |
| S2 | Strict | SC=1, gamma=1.50 | SC=3, gamma=1.00 | -0.0079 | +0.0995 | +0.1072 | +2.00 ($p=0.0e+00$) | +0.0136 | -3.39 | -0.0169 | +0.168 |
| S3 | Strict | SC=2, gamma=1.00 | SC=4, gamma=0.50 | -0.0097 | +0.0791 | +0.0883 | +1.38 ($p=0.0e+00$) | +0.0125 | -1.35 | -0.0054 | +0.119 |
| C1 | Close | SC=1, gamma=1.50 | SC=4, gamma=1.00 | -0.0280 | +0.0904 | +0.1182 | +2.54 ($p=0.0e+00$) | +0.0091 | -2.85 | -0.0168 | +0.170 |
| C2 | Close | SC=1, gamma=1.00 | SC=3, gamma=0.50 | -0.0494 | +0.0727 | +0.1221 | +2.62 ($p=0.0e+00$) | +0.0020 | -1.66 | -0.0047 | +0.127 |
| C3 | Close | SC=1, gamma=1.00 | SC=4, gamma=0.50 | -0.0710 | +0.0592 | +0.1301 | +3.11 ($p=0.0e+00$) | -0.0039 | -0.78 | -0.0035 | +0.066 |
| C4 | Close | SC=2, gamma=1.00 | SC=3, gamma=0.50 | +0.0119 | +0.0926 | +0.0803 | +0.89 ($p=0.0e+00$) | +0.0184 | -2.24 | -0.0066 | +0.180 |

## 3. Priority 3 (E3): `ELF-B` Intermediate $\gamma \in \{1.25, 1.75\}$ Grid Refinement at $\text{SC-CFG}=1.0$
| Run ID | $\gamma$ | Seed | Gen-PPL | $H_{\text{ps}}$ | rep-4 (%) | distinct-2 | MAUVE | JCD (bpb) |
|---|---|---|---|---|---|---|---|---|
| `p4_elfb_s32_g1.25_sc1.0_seed0` | 1.25 | 0 | 34.48 | 5.273 | 2.88% | 0.3080 | 0.308 | 0.0820 |
| `p4_elfb_s32_g1.25_sc1.0_seed1` | 1.25 | 1 | 34.11 | 5.263 | 2.88% | 0.3070 | 0.256 | 0.0768 |
| `p4_elfb_s32_g1.75_sc1.0_seed0` | 1.75 | 0 | 26.49 | 5.184 | 4.06% | 0.2729 | 0.207 | 0.0985 |
| `p4_elfb_s32_g1.75_sc1.0_seed1` | 1.75 | 1 | 26.09 | 5.179 | 3.99% | 0.2716 | 0.194 | 0.0963 |
