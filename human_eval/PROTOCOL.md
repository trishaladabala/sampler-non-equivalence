# Blinded Human Evaluation Protocol: Within-Sample Repetition vs. Cross-Sample Diversity at Matched $(\text{Gen-PPL}, H_{\text{ps}})$

## 1. Objective and Research Question

This protocol tests whether human raters perceive the structural differences measured by automatic metrics between two inference configurations of **ELF-M** (342M parameters, 64 steps) that operate at a matched generative perplexity and per-sample unigram entropy point:

- **Configuration A (Low Guidance, High Back-Noising):** $\text{SC-CFG} = 1.0, \gamma = 2.0$ ($\text{Gen-PPL} = 23.01$, $H_{\text{ps}} = 5.210$ nats)
- **Configuration B (High Guidance, Low Back-Noising):** $\text{SC-CFG} = 4.0, \gamma = 0.5$ ($\text{Gen-PPL} = 23.15$, $H_{\text{ps}} = 5.190$ nats)

Automatic evaluation indicates two simultaneous shifts from A to B:
1. **Higher within-sample repetition in B:** 4-gram repetition rises from $2.95\%$ to $6.42\%$ ($+3.47\text{ pp}$), and within-sample Distinct-2 drops from $0.7985$ to $0.7554$.
2. **Higher inter-sample topic/lexical diversity in B:** Pooled unigram entropy rises from $6.890$ to $7.146$ nats, inter-sample JSD rises from $1.680$ to $1.957$ nats, and 4-gram Self-BLEU drops from $45.46$ to $39.11$.

---

## 2. Experimental Design (Two-Task Double-Blind Design)

Because within-sample repetition is a **single-document** property whereas inter-sample diversity is a **set-level (multi-document)** property, evaluating single passages in isolation cannot test both dimensions simultaneously. We therefore use a two-part blinded design:

### Task 1 — Single-Passage Fluency & Local Repetition (100 Paired Trials)
- **Unit of evaluation:** A paired trial $i \in \{1, \dots, 100\}$ matching sample index $j_i$ from Configuration A and sample index $j_i$ from Configuration B (both generated from the same initial latent noise $z_0^{(j_i)}$ under seed 0, truncated cleanly at 250 words for uniform reading length).
- **Blinding & randomization:** In each trial, the two passages are labeled `(Passage X, Passage Y)` with order randomized by a Bernoulli($0.5$) coin flip recorded only in `human_eval/key_secret.json`.
- **Rater questions per trial:**
  1. **Local Repetition (Forced Choice + Tie):** *"Which passage exhibits more noticeable phrase, clause, or sentence-level repetition?"* (`Passage X`, `Passage Y`, `Indistinguishable`).
  2. **Local Coherence & Grammaticality (5-point Likert per passage):** Rate each passage from 1 (severely incoherent/broken) to 5 (natural, coherent English prose).

### Task 2 — Multi-Passage Bundle Diversity (50 Bundle-Pair Trials)
- **Unit of evaluation:** Each trial presents **Bundle X** (4 independent samples drawn without replacement from one configuration) and **Bundle Y** (4 independent samples drawn without replacement from the other configuration, truncated to 120 words each).
- **Blinding & randomization:** Bundle labels (`Bundle X`, `Bundle Y`) and sample order within each bundle are randomized.
- **Rater question per trial:**
  1. **Inter-Sample Diversity (Forced Choice + Tie):** *"Which bundle of 4 texts covers a broader range of topics, styles, and vocabulary across its 4 passages?"* (`Bundle X`, `Bundle Y`, `Indistinguishable`).

---

## 3. Annotator Requirements & Statistical Analysis Plan

- **Annotators:** 3 independent fluent English annotators (blinded to model identity, configuration parameters, and study hypothesis).
- **Inter-Annotator Agreement:** Krippendorff's $\alpha$ (ordinal/nominal) and Fleiss' $\kappa$ across the 3 annotators on both tasks.
- **Primary Hypothesis Tests (Pre-registered):**
  - **H1 (Within-Sample Repetition):** Two-sided exact binomial test (and mixed-effects logistic regression with random intercepts for rater and item) testing whether Configuration B is selected as more repetitive significantly more often than Configuration A ($p_0 = 0.5$ among non-tie preferences).With $N = 100$ pairs, a true preference rate of $68\%$ achieves $>90\%$ power at $\alpha = 0.05$.
  - **H2 (Inter-Sample Bundle Diversity):** Two-sided exact binomial test on the $N = 50$ bundle pairs testing whether Configuration B bundles are selected as more diverse across samples than Configuration A bundles.
- **Scoring Script:** Run `python3 human_eval/analyze_human_eval.py --ratings human_eval/ratings.csv --key human_eval/key_secret.json` to compute win rates, Wilson 95% confidence intervals, exact binomial $p$-values, and Fleiss' $\kappa$.
