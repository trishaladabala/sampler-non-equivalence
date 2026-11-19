# Standby Human Evaluation Protocol: Operating-Point Structural Divergence

**Protocol ID:** `ELF-HUMAN-STANDBY-2027`  
**Target Venue:** ACL 2027 (ARR Cycle)  
**Status:** **STANDBY PROTOCOL (Turn-Key Readiness)**  
**Note:** Due to absence of human raters and IRB funding at the initial submission freeze, primary empirical results in the manuscript rely on the **validated automated (LLM) judgment panel** (see `LLM_JUDGE_VALIDATION_PROTOCOL.md`). This protocol is maintained in a fully verified, turn-key state to allow rapid execution if crowd or expert raters become available prior to final publication.

---

## 1. Study Overview & Research Questions

The study evaluates whether matched operating points on the 2D Pareto frontier ($\text{Gen-PPL}, H_{\mathrm{ps}}$) of continuous-latent flow language models (ELF-B, ELF-M) produce perceptually indistinguishable text distributions to human evaluators.

### Primary Hypotheses
1. **Task 1 (Local Repetition & Fluency):** High-guidance generation (Arm B: elevated $\text{SC-CFG}$, $\gamma=1.0$) will be perceived as containing more local repetitive artifacts than baseline generation (Arm A: $\text{SC}=1.0$, elevated $\gamma^*$), despite identical generative perplexity.
2. **Task 2 (Bundle Diversity):** When observing sets of 5 generated passages, human evaluators will identify higher cross-passage lexical and topical variety in Arm B compared to Arm A.

---

## 2. Human Subjects & Ethics

- **Participant Recruitment:** Prolific or MTurk platform; native English speakers, approval rate $\ge 98\%$, minimum 500 completed submissions.
- **Compensation:** Target rate $\ge \$15.00/\text{hour}$. Estimated time per HIT: 8 minutes for 10 trials ($\approx \$2.50$ per HIT).
- **Ethics & IRB:** Exemption or IRB protocol under standard benign behavioral research involving non-sensitive text evaluation.
- **Informed Consent:** Explicit consent screen explaining anonymous text evaluation, right to withdraw at any time without penalty.

---

## 3. Experimental Design

### 3.1 Task 1: Single-Passage Pairwise Comparison
- **Stimuli:** Two generated passages (Passage A and Passage B) of length 256 tokens ($\approx 200\text{--}250$ words), generated from the same seed $s \in \{401, \dots, 420\}$ under Arm A vs Arm B.
- **Randomization:** Order of presentation $(A, B)$ vs $(B, A)$ is balanced $50/50$ across items using cryptographic PRNG. Evaluators are double-blinded.
- **Evaluation Prompts:**
  1. *Repetition:* "Which passage exhibits less unnecessary repetition or looping phrasing?" (Options: Passage 1, Passage 2, No noticeable difference).
  2. *Fluency & Naturalness:* "Which passage reads more naturally as coherent English prose?" (Options: Passage 1, Passage 2, No noticeable difference).
  3. *Overall Preference:* "Which passage is higher quality overall?"

### 3.2 Task 2: 5-Passage Bundle Diversity Comparison
- **Stimuli:** Bundle 1 (5 passages from Arm A) vs Bundle 2 (5 passages from Arm B) generated from the same seed.
- **Evaluation Question:**
  "Considering all 5 passages in each group, which group exhibits greater variety in vocabulary, sentence structures, and subject matter?" (Options: Group 1, Group 2, Equally diverse).

### 3.3 Attention & Quality Control Checks (Catch Trials)
- 10% of trials are synthetic catch trials:
  - An obviously degenerate passage (e.g., "the the the the" repeated 50 times) paired with coherent text.
  - Participants who fail $\ge 1$ catch trial are discarded and replaced.

---

## 4. Power Analysis & Clustered Design Effect

With $R=3$ independent human ratings per item and intra-cluster correlation $\rho_{\mathrm{item}} \approx 0.20$:
$$\mathrm{DEFF} = 1 + (R - 1)\rho_{\mathrm{item}} = 1 + (3 - 1)(0.20) = 1.40.$$
- Effective sample size: $N_{\mathrm{eff}} = N_{\mathrm{items}} \times R / \mathrm{DEFF} = 300 \times 3 / 1.40 \approx 643$ trials.
- Power $> 0.95$ to detect a win-rate difference $|\pi - 0.50| \ge 0.08$ at $\alpha = 0.05$.

---

## 5. Statistical Analysis & Equivalence Testing

1. **Mixed-Effects Logistic Regression:**
   $$\text{logit}(P(Y_{ijk} = \text{Arm B})) = \beta_0 + u_j + v_k,$$
   where $u_j \sim \mathcal{N}(0, \sigma_{\mathrm{item}}^2)$ and $v_k \sim \mathcal{N}(0, \sigma_{\mathrm{rater}}^2)$.
2. **Equivalence TOST on Preference Rate:**
   Equivalence interval: $\pi \in [0.45, 0.55]$ ($\pm 5\%$ margin around parity).
3. **Inter-Annotator Agreement:**
   Krippendorff's $\alpha$ (nominal) and Fleiss' $\kappa$.

---

## 6. Execution Runbook

1. Generate blinded assignment CSV via `ITEM_STRATIFICATION_SPEC.md`.
2. Deploy HITs to Prolific with answer keys in `human_eval_answer_key.json`.
3. Ingest collected responses with `human_eval/analyze_human_eval.py`.
4. Merge results into `analysis/out/final_consolidation_stats.json`.
