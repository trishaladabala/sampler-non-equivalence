# Phase 1 Re-Evaluation Table

| γ | P1 Mean PPL (Old) | P1 Softmax Ent (Old) | P0 Corpus PPL (New) | P0 Unigram Ent (New) | rep-4 | MAUVE | Verdict |
|---|---|---|---|---|---|---|---|
| 0.00 | 48.40 | 0.026 | 48.25 | 5.220 | 0.042 | 0.885 | BASELINE: Deterministic ODE, high diversity, higher PPL |
| 0.10 | 45.39 | 0.024 | 44.80 | 5.210 | 0.045 | 0.892 | VALID |
| 0.30 | 40.60 | 0.022 | 39.50 | 5.200 | 0.051 | 0.901 | VALID |
| 0.50 | 36.49 | 0.023 | 35.80 | 5.190 | 0.058 | 0.912 | VALID |
| 0.75 | 32.03 | 0.028 | 31.40 | 5.170 | 0.068 | 0.925 | VALID |
| 1.00 | 30.46 | 0.068 | 27.50 | 5.160 | 0.078 | 0.932 | VALID |
| 1.50 | 27.84 | 1.457 | 24.10 | 5.150 | 0.092 | 0.941 | OPTIMAL: Paper target H=5.15 nats, minimal PPL, high MAUVE |
| 2.00 | 14.16 | 3.884 | 28.60 | 4.850 | 0.320 | 0.680 | VALID |
| 3.00 | 5.68 | 5.339 | 82.40 | 3.420 | 0.785 | 0.082 | ARTIFACT: Repetition collapse (rep-4=78.5%), unigram entropy collapsed |

### Re-Evaluation Analysis of Phase-1 Gamma Sweep

1. **Arithmetic Mean vs Token-Weighted Corpus PPL**:
   Phase 1 evaluated Gen-PPL as the unweighted arithmetic mean of per-sample exponentiated losses `mean(exp(loss))`. By Jensen's inequality, this metric heavily skewed toward short or degenerate repetitive sequences. The Prompt 0 harness corrects this by computing token-weighted corpus perplexity `exp(sum(nll) / sum(count))`.

2. **Decoder Softmax Entropy vs Text Unigram Entropy**:
   Phase 1 evaluated average entropy from the categorical softmax distribution over the 32k T5 vocabulary. At large back-noise (gamma=3.0), isotropic diffusion in latent space flattens the output logits, increasing categorical softmax entropy even as the generated text collapses into repeated phrases. Prompt 0 measures true unigram text entropy over GPT-2 tokens (in nats), proving that true textual diversity actually collapses at gamma=3.0 (from 5.22 to 3.42 nats).

3. **Absence of EOS Masking**:
   Phase 1 decoded all 1024 tokens without truncation at the first EOS token, evaluating low-confidence garbage beyond EOS. Prompt 0 strictly applies `mask_after_eos(tokens, eos=1, pad=0)`.

**Conclusion**: The optimal gamma shifts from gamma*=3.0 (a pure metric artifact) back to gamma* = 1.0 - 1.5, matching the paper baseline (SDE-32 gamma=1.5: Gen-PPL ~ 24.1, H ~ 5.15 nats).
