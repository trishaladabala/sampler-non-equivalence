# Protocol: Validated Automated (LLM) Judgment Panel

**Protocol ID:** `ELF-LLM-JUDGE-PANEL-2027`  
**Target Venue:** ACL 2027 (ARR Cycle)  
**Governing Role:** Primary Perceptual Grounding Pipeline  

---

## 1. Terminology & Governance Rules

1. **Strict Nomenclature:** All reporting in the manuscript, tables, and supplementary materials must strictly use the terminology **"validated automated (LLM) judgments"** or **"automated LLM panel ratings"**.
2. **Prohibition:** Under NO circumstances may these results be described as "human evaluation", "human ratings", or "perceptual user study".
3. **Limitations Disclosure:** The manuscript's Limitations section must explicitly disclose:
   > *"No primary human subject study was conducted. Perceptual differentiation is established via a panel of three diverse, open-weights automated LLM judges validated against published human benchmark datasets."*

---

## 2. Model Panel Selection & Independence

To prevent single-model family inductive bias, the automated panel comprises three open-weights instruction-tuned models from three completely independent architectural and organizational families:

| Model ID | Organization | Architecture Family | Quantization | Context Window |
|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Alibaba Cloud | Qwen2 Transformer | 4-bit NF4 (`bitsandbytes`) | 32,768 tokens |
| `meta-llama/Llama-3.1-8B-Instruct` | Meta AI | Llama-3 RoPE Transformer | 4-bit NF4 (`bitsandbytes`) | 128,000 tokens |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral AI | Mistral Sliding Window / Standard | 4-bit NF4 (`bitsandbytes`) | 32,768 tokens |

### Hardware Footprint
All three models are executed sequentially or in parallel on a single Colab A100 (40 GB / 80 GB) GPU utilizing bitsandbytes 4-bit NormalFloat4 (NF4) quantization with bfloat16 compute, requiring $\approx 5.5\text{ GB}$ VRAM per model.

---

## 3. Grounding & Pre-Validation against Published Human Benchmarks

Before scoring any experimental ELF outputs, each judge model must be pre-validated against ground-truth human annotations from published literature:

1. **Benchmark Datasets:**
   - **HANNA** (Chhun et al., 2022): Human annotations of text repetition, coherence, and fluency.
   - **Scarecrow** (Dou et al., 2021): Human error-span annotations for language generation quality and repetition loops.
2. **Admission Threshold:**
   Each model is evaluated on 50 paired benchmark items (`human_eval/protocol/published_human_benchmarks.json`).
   - Preregistered Admission Criterion: Spearman rank correlation $\rho \ge 0.50$ (or Cohen's $\kappa \ge 0.40$) between LLM preference probability and ground-truth human score differences.
   - **Fail-Closed Gate:** If any judge fails to meet $\rho \ge 0.50$, the notebook aborts with `JudgeValidationFailedError`.

---

## 4. Position Swapping & De-biasing Protocol

LLMs are prone to systemic positional bias (favoring Option 1 or Option 2 regardless of quality). We eliminate positional bias by evaluating every paired item under both presentations:

- Presentation 1: Item $(X, Y)$ presented as `Passage 1: X`, `Passage 2: Y`.
- Presentation 2: Item $(X, Y)$ presented as `Passage 1: Y`, `Passage 2: X`.

### Resolution Rules:
- **Consistent Preference:**
  - If Judge chooses Passage 1 in Presentation 1 and Passage 2 in Presentation 2 $\implies$ Genuine preference for $X$.
  - If Judge chooses Passage 2 in Presentation 1 and Passage 1 in Presentation 2 $\implies$ Genuine preference for $Y$.
- **Positional Tie / Inconsistency:**
  - If Judge chooses Passage 1 in both presentations (or Passage 2 in both presentations) $\implies$ Positional bias artifact; recorded as **Tie / Indifferent** ($0.5$ credit to each arm).

---

## 5. Judging Prompts

### Task 1: Single-Passage Repetition & Fluency
```text
You are an expert linguistic annotator evaluating machine-generated English passages.
Compare the two passages below on fluency, coherence, and lack of repetitive looping.

Passage 1:
{passage_1}

Passage 2:
{passage_2}

Criteria:
1. Repetition: Penalize text that repeats phrases, words, or syntactic structures unnecessarily.
2. Fluency: Reward coherent, natural English grammar and semantic progression.

Provide your evaluation in the following strict JSON format:
{
  "reasoning": "<concise 1-2 sentence justification>",
  "more_repetitive": "Passage 1" | "Passage 2" | "Neither",
  "more_fluent": "Passage 1" | "Passage 2" | "Neither",
  "overall_preference": "Passage 1" | "Passage 2" | "Tie"
}
```

### Task 2: 5-Passage Bundle Diversity
```text
You are an expert computational linguist evaluating text generation diversity.
Compare the two bundles of 5 passages below. Which bundle exhibits greater vocabulary richness, varied topics, and structural diversity across its 5 passages?

Bundle 1:
[Passage 1-1]: {b1_p1}
[Passage 1-2]: {b1_p2}
...
[Passage 1-5]: {b1_p5}

Bundle 2:
[Passage 2-1]: {b2_p1}
[Passage 2-2]: {b2_p2}
...
[Passage 2-5]: {b2_p5}

Provide your evaluation in strict JSON format:
{
  "reasoning": "<concise 1-2 sentence justification>",
  "more_diverse_bundle": "Bundle 1" | "Bundle 2" | "Tie"
}
```

---

## 6. Statistical Analysis

1. **Win-Rate Metric:** For each comparison (e.g. Arm B vs Arm A on C3), calculate aggregate win-rate:
   $$\hat{\pi} = \frac{N_{\text{Arm B wins}} + 0.5 \cdot N_{\text{Ties}}}{N_{\text{total Trials}}}.$$
2. **Equivalence TOST:** Test whether $\hat{\pi}$ is equivalent to $0.50$ within equivalence margin $\pm 0.05$ ($\pi \in [0.45, 0.55]$).
3. **Mixed-Effects Logistic Model:**
   $$\text{logit}(P(\text{Arm B Win})) = \beta_0 + u_{\text{item}} + v_{\text{judge}} + w_{\text{seed}}.$$
4. **Inter-Judge Agreement:** Krippendorff's $\alpha$ across the three judge panels.
