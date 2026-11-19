# OSF Preregistration: Confirmatory Replication of Operating-Point Non-Equivalence in Continuous-Latent Flow Language Models

**Protocol Identifier:** `PREREG-ELF-ARR2027-C1C3-REP20`  
**Registration Type:** Open Science Framework (OSF) Standard Preregistration  
**Timestamp:** assigned by OSF on registration  
**Target Venue:** ACL 2027 (ARR Cycle)  
**Authorship:** Anonymized for Double-Blind Peer Review  

---

## 1. Study Information

### 1.1 Title
Same Perplexity, Same Entropy, Different Text: Operating-Point Non-Equivalence of Sampler Controls in Continuous-Latent Flow Language Models (Confirmatory Multi-Seed Replication Protocol)

### 1.2 Research Questions & Core Problem
Generative flow and continuous-latent diffusion models for text generate discrete sequences by rolling out continuous trajectories over latent representations. Practitioners modulate generation dynamics through trajectory controls such as Langevin back-noising rate ($\gamma \ge 0$) and self-conditioning classifier-free guidance ($\text{SC-CFG}$, denoted $s$). Previous evaluation paradigms benchmark sampling controls at single operating points, or compare models along a 2D Pareto frontier of generative perplexity ($\text{Gen-PPL}$) versus per-sample unigram entropy ($H_{\mathrm{ps}}$).

This study addresses the fundamental question of **identifiability and operating-point equivalence**:
> *Does matching generative perplexity ($\text{Gen-PPL}$) and per-sample entropy ($H_{\mathrm{ps}}$) across distinct sampler control configurations guarantee structural equivalence of the generated text distribution?*

We hypothesize that operating points on the 2D frontier are **non-equivalent**: distinct control pairs calibrated to have identical $\text{Gen-PPL}$ and $H_{\mathrm{ps}}$ produce text distributions that differ systematically and significantly across within-sequence repetition, cross-sequence vocabulary diversity, and distributional reference alignment.

### 1.3 Hypotheses
1. **Operating-Point Equivalence Gate (Family 1):** Across three calibration-frozen operating pairs ($C1, C2, C3$), Arm A ($\text{SC}=1.0, \gamma^*$) and Arm B ($\text{SC} \in \{2.0, 3.0, 4.0\}, \gamma=1.0$) are statistically equivalent in generative perplexity ($\text{Gen-PPL}$) within $\pm 2.5\%$ relative margin and per-sample entropy ($H_{\mathrm{ps}}$) within $\pm 0.015$ nats margin under paired Common Random Numbers (CRN) Two One-Sided Tests (TOST) at family-wise error rate $\alpha = 0.05$.
2. **Structural Divergence (Family 2):** Conditional on Family 1 gate satisfaction, Arm B exhibits statistically significant structural differences from Arm A across three orthogonal structural dimensions:
   - Higher within-sequence 4-gram repetition ($\Delta \text{rep-4} > 0$).
   - Higher cross-sample vocabulary diversity ($\Delta \mathrm{JSD}_\pi > 0$).
   - Higher distributional similarity to pretraining-excluded reference text ($\Delta \mathrm{MAUVE} > 0$).

---

## 2. Model Specifications & Frozen Operating Points

### 2.1 Model Architecture & Parameter Accounting
All primary confirmatory tests are conducted on the official checkpoint of **ELF-B** (continuous-latent language flow model):
- Architecture: 12-layer continuous diffusion/flow Transformer backbone ($d_{\mathrm{model}} = 768$, $n_{\mathrm{heads}} = 12$, sequence length $L = 256$).
- Parameter Count: **104,579,940** trainable parameters in `eval_model` (ELF backbone).
- **Mandatory Parameter Accounting Note:** This parameter count **strictly excludes** the frozen T5-small text encoder ($\approx 35\text{M}$ parameters, initialized separately via `elfx/load.py` lines 109–112 with `requires_grad=False`).
- Integrator: Distilled Euler–Maruyama SDE rollout with hybrid Langevin back-noising (`method="sde"` in `elfx/sample.py`, matching `ELF/src/utils/sampling_utils.py:_sde_step`) with fixed step count $K = 32$ plus 1 terminal decode evaluation at $t=1.0$ via `_dlm_decode_batch` ($NFE_{\mathrm{sample}} = K + 1 = 33$).

### 2.2 Frozen Calibrated Operating Points
The operating-point control roots $\gamma^*$ were calibrated on exploratory Stage 1 (seeds 0 through 4) using 1D cubic spline interpolation to match the operating point of Arm B at $\gamma = 1.000$. These operating points are **strictly frozen** prior to replication:

| Pair ID | Arm A (Baseline SC) | Arm B (Elevated SC) | Target Operating Point (Stage 1 Est.) | Tier-I Margin ($\Delta_{\mathrm{PPL}}, \Delta_H$) |
|:---|:---|:---|:---|:---|
| **C1** | $\text{SC}=1.0,\; \gamma^* = 1.346$ | $\text{SC}=2.0,\; \gamma = 1.000$ | $\text{Gen-PPL} \approx 32.4$, $H_{\mathrm{ps}} \approx 5.25\text{ nats}$ | $\pm 0.81$ PPL ($\pm 2.5\%$), $\pm 0.015\text{ nats}$ |
| **C2** | $\text{SC}=1.0,\; \gamma^* = 1.547$ | $\text{SC}=3.0,\; \gamma = 1.000$ | $\text{Gen-PPL} \approx 28.9$, $H_{\mathrm{ps}} \approx 5.21\text{ nats}$ | $\pm 0.72$ PPL ($\pm 2.5\%$), $\pm 0.015\text{ nats}$ |
| **C3** | $\text{SC}=1.0,\; \gamma^* = 1.651$ | $\text{SC}=4.0,\; \gamma = 1.000$ | $\text{Gen-PPL} \approx 27.5$, $H_{\mathrm{ps}} \approx 5.19\text{ nats}$ | $\pm 0.69$ PPL ($\pm 2.5\%$), $\pm 0.015\text{ nats}$ |

---

## 3. Sampling Plan & Replication Data Collection

### 3.1 Replication Seeds & Sample Size
- **Replication Seeds:** Exactly 20 fresh, previously unexamined seeds:
  $$\mathcal{S}_{\mathrm{rep}} = \{401, 402, 403, \dots, 420\}.$$
- **Sample Size per Seed:** $M = 1,000$ independent sequences per seed ($L = 256$ tokens).
- **Total Sequences Generated:**
  - Per arm per pair: $20 \text{ seeds} \times 1,000 = 20,000$ sequences.
  - Total across study: $3 \text{ pairs} \times 2 \text{ arms} \times 20 \text{ seeds} = 120 \text{ Colab GPU runs}$ ($120,000$ sequences).
- **Power Analysis:** With $n = 20$ paired seed observations, paired $t$-tests achieve $> 99\%$ statistical power ($\alpha = 0.05$) to detect secondary effect sizes of $|d| \ge 1.0$. For reference, pilot seed-level effect sizes observed on seeds 201–205 ranged $d_{\mathrm{pilot}} = 2.0\text{--}8.0$ (`analysis/out/c1_c3_per_seed_recovered.json`). Exact paired sign-flip permutation tests have a minimum achievable p-value of $2/2^{20} \approx 1.9 \times 10^{-6}$, ensuring robust non-parametric significance under multi-test correction.

### 3.2 Seeding Integrity & Three-Stream Common Random Numbers (CRN)
To eliminate sampling variance from between-arm comparisons, Arm A and Arm B share bit-identical pseudorandom sequences across all three stochastic sampling streams for every sequence $j \in \{0, \dots, M-1\}$ under replication seed $s \in \mathcal{S}_{\mathrm{rep}}$:
1. **Initial Latents $\mathbf{z}_0^{(j)}$:** Seeded deterministically via $\mathrm{seed} \times 100003 + j$.
2. **Discretization Schedule $t^{(j)}$:** Seeded via cryptographic hash integer $\mathrm{seed\_int}(s, j, \text{"sched"})$.
3. **Langevin SDE Brownian Noise $\boldsymbol{\epsilon}_i^{(j)}$:** Seeded via cryptographic hash integer $\mathrm{seed\_int}(s, j, \text{"sde"})$.

Prior to trajectory rollout, all three tensors are audited for bit-identical SHA-256 digests across Arm A and Arm B:
$$\mathrm{SHA256}(\mathbf{z}_{0, A}) == \mathrm{SHA256}(\mathbf{z}_{0, B}), \quad \mathrm{SHA256}(\mathbf{t}_A) == \mathrm{SHA256}(\mathbf{t}_B), \quad \mathrm{SHA256}(\boldsymbol{\epsilon}_A) == \mathrm{SHA256}(\boldsymbol{\epsilon}_B),$$
replicating the empirical verification protocol established in Notebook 11 (`crn_empirical_digest_audit.json`).
All raw generated sequence tokens and decoded strings are saved in JSONL format (`samples.jsonl`) on persistent storage with SHA-256 manifests.

---

## 4. Variables & Evaluation Measures

### 4.1 Primary Gate Variables
1. **Generative Perplexity ($\text{Gen-PPL}$):** Evaluated under a frozen autoregressive scorer model (`gpt2-large`, 774M parameters, executed in `bfloat16` precision on CUDA via `elfx/metrics.py:score_gpt2`). Generated text samples are retokenized using the GPT-2 tokenizer with context truncation at $L_{\max} = 1024$ tokens; empty samples are dropped, the first end-of-sequence token is counted, and padding tokens are masked. Perplexity is computed as corpus token-weighted exponentiated negative log-likelihood:
   $$\text{Gen-PPL} = \exp\left( \frac{\sum_{i=1}^M \mathrm{NLL}_i}{\sum_{i=1}^M \mathrm{count}_i} \right) = \mathrm{corpus\_ppl}(\mathbf{nll}, \mathbf{count}),$$
   where $\mathrm{NLL}_i = \sum_{j=1}^{L_i} -\log P_{\mathrm{GPT2}}(x_{i,j} \mid x_{i,<j})$ is the cumulative negative log-likelihood over valid tokens of sample $i$, and $\mathrm{count}_i$ is the number of valid tokens in sample $i$.
2. **Per-Sample Unigram Entropy ($H_{\mathrm{ps}}$):** Mean across sequences of the empirical unigram Shannon entropy (in nats):
   $$H_{\mathrm{ps}} = \frac{1}{M} \sum_{i=1}^M \left( -\sum_{w \in \mathcal{V}} \hat{p}_i(w) \log \hat{p}_i(w) \right).$$

### 4.2 Confirmatory Secondary Metrics (Family 2)
Selected a priori from three orthogonal structural clusters to avoid pseudo-replication:
1. **`rep-4` (Local Repetition):** Fraction of 4-grams within a sequence that are duplicates:
   $$\text{rep-4} = 1 - \frac{|\text{unique 4-grams}|}{|\text{total 4-grams}|}.$$
2. **$\mathrm{JSD}_\pi$ (Cross-Sequence Diversity / Inter-Sample JSD):** Jensen–Shannon divergence among individual sequence empirical unigram token distributions, implemented exactly as `jsd_inter_sample` in `elfx/non_identifiability.py` lines 304–311:
   $$\mathrm{JSD}_\pi = \max\left(0,\; H_{\mathrm{pool}} - H_{\mathrm{ps, weighted}}\right),$$
   where for $M$ non-empty samples with valid token counts $L_i$ and total token count $N_{\mathrm{tok}} = \sum_{i=1}^M L_i$:
   $$H_{\mathrm{ps, weighted}} = \frac{1}{N_{\mathrm{tok}}} \sum_{i=1}^M L_i \cdot H_{\mathrm{ps}, i}, \quad \bar{p}(w) = \frac{1}{N_{\mathrm{tok}}} \sum_{i=1}^M c_i(w), \quad H_{\mathrm{pool}} = -\sum_{w \in \mathcal{V}} \bar{p}(w) \log \bar{p}(w).$$
3. **$\mathrm{MAUVE}_{\mathrm{heldout}}$ (Distributional Alignment):** MAUVE divergence between generated sample distributions and pretraining-excluded reference text, specified a priori as follows:
   - **Featurizer:** Last-token representation from the final hidden layer (post `ln_f`) of frozen `gpt2-large` (774M), extracted via `elfx.metrics.gpt2_features` with batch size 16 and truncation at 1024 tokens.
   - **Reference Corpus:** Reference A (`webtext.test.jsonl` from OpenAI `gpt-2-output-dataset`), verified to have 0 matching 13-grams against `openwebtext-t5` training shards, subsampled to $N_{\mathrm{ref}} = M = 1,000$ documents matching sample length (the reference size must be matched to generated sample count $M$, so $N_{\mathrm{ref}} = M = 1,000$).
   - **Algorithm Settings:** Computed via official `mauve-text` through the exact call matching `elfx/metrics.py:mauve_from_features` (`mauve.compute_mauve(p_features=ref_feats, q_features=gen_feats, verbose=False, seed=25)`). This adheres strictly to the library defaults: `num_buckets = 'auto'` ($k = \lfloor \min(|P|, |Q|) / 10 \rfloor$, yielding $k = 100$ at $M = 1{,}000$ and $k = 50$ at $M = 500$ as specified in the manuscript), default scaling factor $c = 5.0$ (`mauve_scaling_factor = 5.0`), and fixed random seed `seed = 25`.
   - **Sample Count:** Evaluated over $M = 1,000$ passages per arm per replication seed with reference size matched to $N_{\mathrm{ref}} = M = 1,000$.

### 4.3 Exploratory & Diagnostic Variables (Family 3)
Reported descriptively with point estimates, standard errors, and nominal 95% bootstrap CIs without formal confirmatory claims:
- `within_distinct2`, `H_pool`, `distinct-2`, `Self-BLEU`, byte-level context-tree weighting compression divergence (`CTW JCD`), Sentence-BERT embedding Vendi scores (`all-mpnet-base-v2`, `all-MiniLM-L6-v2`, `bge-base-en-v1.5`), and PPL quantile ratios ($p_{90}/p_{10}$).

---

## 5. Statistical Analysis Plan & Hierarchical Gatekeeping

### 5.1 Equivalence Margin Definition & Justification
- **Tier-I Equivalence Margins:**
  - $\Delta_{\mathrm{PPL}} = \pm 2.5\%$ relative to Arm A mean ($[-0.025 \cdot \bar{Y}_A,\; +0.025 \cdot \bar{Y}_A]$).
  - $\Delta_{H_{\mathrm{ps}}} = \pm 0.015$ nats ($[-0.015,\; +0.015]$ nats).
- **Scientific Justification of Tier-I Equivalence Margins:**
  Tier-I equivalence bounds ($\Delta_{\mathrm{PPL}} = \pm 2.5\%$ relative, $\Delta_{H_{\mathrm{ps}}} = \pm 0.015$ nats) are grounded in empirical pilot dispersion across held-out seeds 201–205 (`analysis/out/c1_c3_per_seed_recovered.json`):
  1. *Pilot Seed Dispersion:* Across 5 pilot seeds ($M=1,000$ sequences/seed), seed-to-seed standard deviation for Gen-PPL was $\mathrm{SD} = 0.12\text{--}0.18$ points on Arm A and $0.23\text{--}0.28$ points on Arm B ($0.4\%\text{--}1.0\%$ relative). For unigram entropy $H_{\mathrm{ps}}$, seed standard deviation was $\mathrm{SD} = 0.004\text{--}0.007$ nats.
  2. *Margin Ratio:* The preregistered bounds ($\pm 0.81$ PPL on C1, $\pm 0.72$ PPL on C2, $\pm 0.69$ PPL on C3; $\pm 0.015$ nats on $H_{\mathrm{ps}}$) represent approximately $2.5\text{--}4.5 \times \mathrm{SD}_{\mathrm{pilot}}$. This establishes a stringent equivalence region that excludes macro-level operating point shifts while maintaining statistical power to confirm equivalence under paired CRN design.

### 5.2 Primary Equivalence Test Design: Paired CRN TOST
Because Arm A and Arm B share identical initial latent samples $\mathbf{z}_0$, schedules, and Brownian increments under CRN, within-seed responses exhibit positive correlation. The paired correlation coefficient $r_{AB} = \mathrm{Corr}(Y_{A,s}, Y_{B,s})$ will be estimated directly from the replication data rather than assumed. For context, observed pilot correlations across seeds 201–205 were $r_{\mathrm{PPL}} = 0.561$ (C1), $0.847$ (C2), $0.565$ (C3) for Gen-PPL, and $r_H = 0.870\text{--}0.903$ for $H_{\mathrm{ps}}$.
For each primary metric $Y \in \{\text{Gen-PPL}, H_{\mathrm{ps}}\}$:
- Paired differences: $D_s = Y_{B,s} - Y_{A,s}$ for $s = 1, \dots, n$ ($n = 20$).
- Mean difference: $\bar{D} = \frac{1}{n}\sum_{s=1}^n D_s$; Sample standard error: $\mathrm{SE}_{\bar{D}} = \frac{s_D}{\sqrt{n}}$, where $s_D = \sqrt{\frac{1}{n-1}\sum_{s=1}^n (D_s - \bar{D})^2}$.
- Lower one-sided test ($H_{01}: \mu_D \le \Delta_{\mathrm{lower}}$ vs $H_{11}: \mu_D > \Delta_{\mathrm{lower}}$):
  $$t_1 = \frac{\bar{D} - \Delta_{\mathrm{lower}}}{\mathrm{SE}_{\bar{D}}}, \quad p_1 = 1 - F_{t(19)}(t_1).$$
- Upper one-sided test ($H_{02}: \mu_D \ge \Delta_{\mathrm{upper}}$ vs $H_{12}: \mu_D < \Delta_{\mathrm{upper}}$):
  $$t_2 = \frac{\Delta_{\mathrm{upper}} - \bar{D}}{\mathrm{SE}_{\bar{D}}}, \quad p_2 = 1 - F_{t(19)}(t_2).$$
- TOST Equivalence p-value:
  $$p_{\mathrm{TOST}} = \max(p_1, p_2).$$
- Joint TOST on operating pair $k \in \{C1, C2, C3\}$:
  $$p_{\mathrm{joint}, k} = \max\left( p_{\mathrm{TOST}, k}^{(\mathrm{PPL})},\; p_{\mathrm{TOST}, k}^{(H_{\mathrm{ps}})} \right).$$
- **Sensitivity Check:** Unpaired Welch TOST ($df \approx 38$) is reported alongside paired TOST to verify robustness against pairing assumptions.

### 5.3 Hierarchical Multiplicity Control Protocol
To guarantee strong Family-Wise Error Rate ($\mathrm{FWER} \le 0.05$) control, we enforce strict **Hierarchical Gatekeeping**:

```
                       ┌────────────────────────────────────────┐
                       │ Family 1: Operating-Point Gate (m = 3) │
                       │ Joint TOST on (Gen-PPL, H_ps) for C1-C3│
                       │ Holm-Bonferroni correction at α = 0.05 │
                       └───────────────────┬────────────────────┘
                                           │
                     All 3 pairs pass?     │
                        ┌──────────────────┴──────────────────┐
                        │ YES                                 │ NO
                        ▼                                     ▼
     ┌────────────────────────────────────┐       ┌───────────────────────────────┐
     │ Family 2: Confirmatory (m = 9)     │       │ GATE CLOSED                   │
     │ 3 metrics (rep-4, JSD, MAUVE)      │       │ Secondary tests cannot make   │
     │ across C1, C2, C3                  │       │ confirmatory claims.          │
     │ Step-down Holm at α = 0.05         │       │ Document gate violation.      │
     └──────────────────┬─────────────────┘       └───────────────────────────────┘
                        │
                        ▼
     ┌────────────────────────────────────┐
     │ Family 3: Exploratory & Diagnostic │
     │ Descriptive point estimates & CIs  │
     │ No formal hypothesis testing claims│
     └────────────────────────────────────┘
```

#### Family 1 Gate Specification:
- Tested hypotheses: $H_{0, k}: \text{Pair } k \text{ is non-equivalent}$ for $k \in \{C1, C2, C3\}$ ($m_1 = 3$).
- The 3 joint p-values $p_{\mathrm{joint}, (1)} \le p_{\mathrm{joint}, (2)} \le p_{\mathrm{joint}, (3)}$ are adjusted via step-down Holm:
  $$p_{\mathrm{Holm}, (i)} = \min\left(1.0,\; \max_{j \le i} (m_1 - j + 1) p_{\mathrm{joint}, (j)}\right).$$
- **Gate Condition:** The gate is declared **OPEN** if and only if $p_{\mathrm{Holm}, k} < 0.05$ for all $k \in \{C1, C2, C3\}$. If any pair fails to demonstrate equivalence, Family 2 is not tested and all confirmatory claims are rejected.

#### Gate Closure & Reporting Contingency:
If any of the three confirmatory pairs fails the Family 1 equivalence test at step-down Holm-adjusted $\alpha = 0.05$, the Family 1 gate is declared **CLOSED**. In this event:
1. Family 2 hypothesis tests will not make formal confirmatory claims.
2. All secondary metrics across all seeds will nevertheless be computed and tabulated in full, accompanied by explicit labels designating them as *non-confirmatory / gate-closed exploratory analyses*.
3. The manuscript will report the observed bounds and document the precise nature of the operating-point divergence.

#### Family 2 Confirmatory Secondary Specification:
- Tested hypotheses: $3 \text{ pairs} \times 3 \text{ metrics} = 9$ hypotheses ($m_2 = 9$).
- For each comparison, test paired difference $D_{s} = Y_{B,s} - Y_{A,s}$ via two-sided paired Student's $t$-test ($df = 19$):
  $$t = \frac{\bar{D}}{\mathrm{SE}_{\bar{D}}}, \quad p_{\mathrm{raw}} = 2 \cdot \left(1 - F_{t(19)}(|t|)\right).$$
- Multiplicity adjustment: Step-down Holm-Bonferroni over the $m_2 = 9$ tests at $\alpha = 0.05$.
- Formal Reporting Notation: Every adjusted p-value must explicitly state the family, number of tests, and alpha level:
  $$p_{\mathrm{Holm}} (\text{Family 2}, m=9, \alpha=0.05).$$

#### Family 3 Exploratory Specification:
- All remaining secondary metrics (`within_distinct2`, `H_pool`, `distinct-2`, `Self-BLEU`, `CTW JCD`, embedding Vendi scores, PPL quantile ratios) are reported purely descriptively with bootstrap 95% CIs and leave-one-seed-out (LOSO) ranges. No confirmatory significance is claimed.

---

## 6. Iso-Frontier Sensitivity Diagnostic ($\kappa_m$, Family 3 Exploratory)

To complement discrete hypothesis testing with a continuous sensitivity diagnostic along the operating curve, we define the **Iso-Frontier Arc-Length Sensitivity Metric** $\kappa_m$:
- Let $\theta = (s, \gamma) \in \mathbb{R}^2$ be the sampler control vector.
- Let $\mathcal{C}$ be the empirical iso-frontier level-set curve along which $\text{Gen-PPL}(\theta)$ and $H_{\mathrm{ps}}(\theta)$ remain matched, parameterized by arc length $\ell$ with unit tangent vector $\hat{\mathbf{u}}_{\mathcal{C}}(\theta)$.
- For any secondary metric $m(\theta)$, the arc-length rate of change is:
  $$\kappa_m(\theta) = \left. \frac{dm}{d\ell} \right|_{\mathcal{C}} = \nabla m(\theta)^\top \hat{\mathbf{u}}_{\mathcal{C}}(\theta).$$
- **Empirical Estimation Specification:**
  - **Surface Data:** Estimated from the existing Stage-1 empirical response-surface grid across $\text{SC-CFG} \in [1.0, 4.0]$ and $\gamma \in [0.8, 2.2]$ (`analysis/results_db.csv`).
  - **Smoother:** Bivariate spline interpolation (`scipy.interpolate.RectBivariateSpline` with degree $k_x, k_y \ge 2$).
  - **Uncertainty Estimation:** 95% confidence intervals constructed via 10,000 bootstrap resamples across calibration seeds.
- **Diagnostic Interpretation:** A non-zero sensitivity $\kappa_m \neq 0$ demonstrates that the secondary metric $m$ varies systematically along the iso-frontier contour within this sampler family. This diagnostic is exploratory (Family 3) and does not constitute a confirmatory hypothesis test.

---

## 7. Data Exclusion, Quality Checks & Missing Data Rules

1. **Fail-Closed Execution:** Any GPU run encountering an out-of-memory error, NaN/Inf evaluation, or incomplete sequence count ($M < 1,000$) will fail execution. No imputation, synthetic backfill, or selective replacement is permitted.
2. **Cryptographic Three-Stream CRN Verification:** A pre-rollout integrity assertion verifies that initial latent noise, discretization time schedules, and Brownian increments exhibit bit-identical SHA-256 tensor digests between paired arms. Any run exhibiting a digest mismatch fails closed immediately.
3. **Pretraining Contamination Exclusion:** Any sequence in the evaluation reference exhibiting $\ge 13$-gram verbatim overlap with the `openwebtext-t5` training corpus is excluded prior to MAUVE and reference evaluation.
4. **Reproducibility Guarantee:** All analysis scripts and packages (`frontier_audit/`, `analysis/iso_frontier_sensitivity.py`) must pass CPU unit test discovery under Python `-W error` with zero warnings, zero errors, and zero failures.

### 7.3 Reproducibility Freeze, Deviations Policy & Prior Data Disclosure
1. **Prior Data Declaration:** Operating-point roots $\gamma^*$ were calibrated on Stage-1 exploratory seeds 0–4. Seeds 201–205 constitute pilot data observed prior to registration. Seeds 401–420 are completely unexamined, ungenerated fresh replication seeds.
2. **Analysis Pipeline Freeze:** Primary statistical testing will be conducted via `analysis/final_consolidation_audit.py` and `frontier_audit/tost.py`. Cryptographic SHA-256 digests of these analysis scripts will be recorded upon OSF upload.
3. **Protocol Deviations Logging:** Any departure from this protocol will be logged with timestamp, technical justification, and sensitivity audit in `reports/PREREGISTRATION_DEVIATIONS.md`.
