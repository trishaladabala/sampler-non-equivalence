# Multi-Seed Stratified Sampling Specification for Automated Judgments

**Specification ID:** `ELF-ITEM-STRAT-REP20`  
**Governing Document:** `PREREG-ELF-ARR2027-C1C3-REP20`  
**Application:** WS3' Automated Judge Panel & Standby Human Evaluation  

---

## 1. Objectives

Previous exploratory surveys noted that evaluating items from only a single seed ($s=201$) risks sampling bias. To ensure population validity across all 20 confirmatory replication seeds:
1. Every evaluation trial is stratified uniformly across seeds $\mathcal{S}_{\mathrm{rep}} = \{401, 402, \dots, 420\}$.
2. Paired items maintain Common Random Number (CRN) alignment: Passage A and Passage B within a pair share the identical sample index $j \in \{0, \dots, 999\}$ and initial latent vector $\mathbf{z}_0^{(s, j)}$.

---

## 2. Sampling Matrix

| Operating Pair | Seed Pool | Samples / Seed | Total Item Pairs | Presentation Pairs (with Swap) |
|---|---|---|---|---|
| **C3** (High contrast: $\text{SC}=4.0$ vs $\text{SC}=1.0$) | Seeds 401–420 ($N=20$) | 15 pairs / seed | 300 pairs | 600 presentations |
| **C1** (Low contrast: $\text{SC}=2.0$ vs $\text{SC}=1.0$) | Seeds 401–420 ($N=20$) | 15 pairs / seed | 300 pairs | 600 presentations |
| **Total** | | | **600 pairs** | **1,200 presentations** |

### Stratified Index Selection
For each seed $s \in \{401, \dots, 420\}$:
- Pseudo-random deterministic indices are selected using:
  $$\text{RNG} = \text{RandomState}(\text{seed} = 20270400 + s).$$
- Indices $\mathcal{I}_s \subset \{0, \dots, 999\}$ of size $|\mathcal{I}_s| = 15$ are sampled without replacement.

---

## 3. Blinding & Presentation Key

- For each sampled pair $k = (s, j)$:
  - Coin flip $C_{s, j} \in \{0, 1\}$ determines initial assignment:
    - If $C_{s, j} = 0$: Presentation 1 is `(Arm A, Arm B)`; Presentation 2 is `(Arm B, Arm A)`.
    - If $C_{s, j} = 1$: Presentation 1 is `(Arm B, Arm A)`; Presentation 2 is `(Arm A, Arm B)`.
- The secret assignment map is stored in:
  `/content/drive/MyDrive/elf_research/runs/judge_eval/judge_secret_key.json`
- All evaluations are completed before the secret key is unblinded for statistical aggregation.
