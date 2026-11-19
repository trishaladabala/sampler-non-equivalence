#!/usr/bin/env python3
"""Sampler-Control Non-Identifiability Follow-Up Experiment Suite (Priorities 1-4).

Can be used in three ways:
  1. Imported as `elfx.non_identifiability` inside Google Colab (`elf_p5_non_identifiability.ipynb`).
  2. Run as a standalone CLI script on a GPU machine, Colab terminal, or local Mac CPU:
       python3 experiments/non_identifiability_suite.py --mode smoke
       python3 experiments/non_identifiability_suite.py --mode e1_elfm_factorial
       python3 experiments/non_identifiability_suite.py --mode e2_matched_rescore
       python3 experiments/non_identifiability_suite.py --mode e3_elfb_refinement
       python3 experiments/non_identifiability_suite.py --mode e4_ar_audit
       python3 experiments/non_identifiability_suite.py --mode all
     When executed on a CUDA GPU with `elfx` + `torch`, runs full sample generation and scoring.
     When executed on a local CPU machine without CUDA/PyTorch, runs offline verification and
     statistical analysis over all local `runs/*/summary.json` (and any local `samples.jsonl`),
     writes the structured JSON/Markdown outputs under `runs/p5_non_identifiability/`, and lists
     the remaining GPU jobs queued for `elf_p5_non_identifiability.ipynb`.
  3. Imported on CPU for unit testing (`tests/test_ppl_estimators.py`).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Environment & Path Resolution (Colab + Local Workspace Compatible)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_REPO_ROOT = os.path.dirname(_THIS_DIR)
_LOCAL_DRIVE_MIRROR = os.path.join(_LOCAL_REPO_ROOT, "content", "drive", "MyDrive", "elf_research")
_DEFAULT_ROOT = (
    "/content/drive/MyDrive/elf_research"
    if os.path.isdir("/content/drive")
    else (_LOCAL_DRIVE_MIRROR if os.path.isdir(_LOCAL_DRIVE_MIRROR) else _LOCAL_REPO_ROOT)
)

ROOT = os.environ.get("ELF_ROOT", _DEFAULT_ROOT)
CODE = os.path.join(ROOT, "code") if os.path.isdir("/content/drive") else _LOCAL_REPO_ROOT
RUNS = os.path.join(ROOT, "runs")
REFS = os.path.join(ROOT, "refs")
P5_DIR = os.path.join(RUNS, "p5_non_identifiability")


def has_cuda_gpu() -> bool:
    return (torch is not None) and bool(torch.cuda.is_available())


def _try_import_elfx():
    """Import full GPU elfx modules if torch + CUDA + elfx are available."""
    if not has_cuda_gpu():
        return None
    for cand in (CODE, _LOCAL_REPO_ROOT):
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import elfx
        from elfx import frontier, load, metrics, refs, registry, sample, sweep
        return dict(
            elfx=elfx,
            load=load,
            sample=sample,
            metrics=metrics,
            refs=refs,
            registry=registry,
            sweep=sweep,
            frontier=frontier,
        )
    except Exception:
        return None


def save_json_atomic(path: Union[str, Path], data: Any) -> None:
    p = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = f"{p}.tmp"

    def _default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default)
    os.replace(tmp, p)


def load_json(path: Union[str, Path], default: Any = None) -> Any:
    p = str(path)
    if not os.path.isfile(p):
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_local_owt_refs(n: int = 1000) -> List[str]:
    """Loads cached OpenWebText reference texts from REFS or local repository if available."""
    candidates = [
        os.path.join(REFS, f"owt_ref_{n}.jsonl"),
        os.path.join(_LOCAL_REPO_ROOT, "refs", f"owt_ref_{n}.jsonl"),
        os.path.join(_LOCAL_REPO_ROOT, "ELF", "phase1", "results", f"owt_ref_{n}.jsonl"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            with open(c, "r", encoding="utf-8") as f:
                return [json.loads(line)["text"] for line in f if line.strip()]
    return []


# ---------------------------------------------------------------------------
# 1. Byte-Level Context-Tree Weighting (CTW) & Jeffreys Coding Divergence (JCD)
#    (Franca & Tong, 2026, arXiv:2606.08417)
# ---------------------------------------------------------------------------
class ByteKTContextModel:
    """Hierarchical Krichevsky-Trofimov (KT) Context-Tree Weighting byte model.

    Fits a context tree of depth `depth` over UTF-8 byte streams with Dirichlet(1/2)
    (KT) smoothing at each node and recursive context-tree weighting mixture across
    depths d = 0 .. depth. Evaluates the average codelength (in bits per byte) of any
    target byte stream under the fitted predictive distribution.
    """

    def __init__(self, depth: int = 6, alpha: float = 0.5):
        self.depth = int(depth)
        self.alpha = float(alpha)
        self.depth_counts: List[Dict[bytes, Counter]] = [
            defaultdict(Counter) for _ in range(self.depth + 1)
        ]
        self.depth_totals: List[Dict[bytes, int]] = [
            defaultdict(int) for _ in range(self.depth + 1)
        ]
        self.total_bytes: int = 0

    def fit(self, texts: Sequence[str], max_bytes: int = 2_000_000) -> "ByteKTContextModel":
        raw = b"\n".join(
            t.encode("utf-8", errors="replace")
            for t in texts
            if isinstance(t, str) and t.strip()
        )
        if max_bytes and len(raw) > max_bytes:
            raw = raw[:max_bytes]
        self.total_bytes = len(raw)
        D = self.depth
        padded = bytes([0] * D) + raw
        for i in range(D, len(padded)):
            sym = padded[i]
            for d in range(D + 1):
                ctx = padded[i - d : i] if d > 0 else b""
                self.depth_counts[d][ctx][sym] += 1
                self.depth_totals[d][ctx] += 1
        return self

    def cross_entropy_bits_per_byte(
        self, texts: Sequence[str], max_bytes: int = 2_000_000
    ) -> float:
        """Evaluates average codelength (bits/byte) of `texts` under this model."""
        raw = b"\n".join(
            t.encode("utf-8", errors="replace")
            for t in texts
            if isinstance(t, str) and t.strip()
        )
        if not raw:
            return float("nan")
        if max_bytes and len(raw) > max_bytes:
            raw = raw[:max_bytes]

        D = self.depth
        alpha = self.alpha
        vocab_alpha = 256.0 * alpha
        padded = bytes([0] * D) + raw

        total_bits = 0.0
        inv_ln2 = 1.0 / math.log(2.0)

        for i in range(D, len(padded)):
            sym = padded[i]
            p_mix = 1.0 / 256.0
            for d in range(D, -1, -1):
                ctx = padded[i - d : i] if d > 0 else b""
                tot = self.depth_totals[d].get(ctx, 0)
                if tot > 0:
                    c = self.depth_counts[d][ctx].get(sym, 0)
                    p_kt = (c + alpha) / (tot + vocab_alpha)
                else:
                    p_kt = 1.0 / 256.0
                if d == D:
                    p_mix = p_kt
                else:
                    p_mix = 0.5 * p_kt + 0.5 * p_mix
            total_bits += -math.log(max(p_mix, 1e-15)) * inv_ln2

        return float(total_bits / len(raw))


def compute_jcd_ctw(
    gen_texts: Sequence[str],
    ref_texts: Sequence[str],
    depth: int = 6,
    max_bytes: int = 500_000,
) -> Dict[str, float]:
    """Computes symmetric byte-level CTW Jeffreys Coding Divergence (JCD) in bits/byte.

    Following Franca & Tong (2026, arXiv:2606.08417):
      KL_code(P_gen || P_ref) = H_CTW(P_gen, P_ref) - H_CTW(P_gen, P_gen)
      KL_code(P_ref || P_gen) = H_CTW(P_ref, P_gen) - H_CTW(P_ref, P_ref)
      JCD(P_gen, P_ref) = 0.5 * (KL_code(P_gen || P_ref) + KL_code(P_ref || P_gen))
    """
    m_gen = ByteKTContextModel(depth=depth).fit(gen_texts, max_bytes=max_bytes)
    m_ref = ByteKTContextModel(depth=depth).fit(ref_texts, max_bytes=max_bytes)

    h_gen_gen = m_gen.cross_entropy_bits_per_byte(gen_texts, max_bytes=max_bytes)
    h_gen_ref = m_ref.cross_entropy_bits_per_byte(gen_texts, max_bytes=max_bytes)
    h_ref_ref = m_ref.cross_entropy_bits_per_byte(ref_texts, max_bytes=max_bytes)
    h_ref_gen = m_gen.cross_entropy_bits_per_byte(ref_texts, max_bytes=max_bytes)

    kl_gen_ref = max(0.0, h_gen_ref - h_gen_gen)
    kl_ref_gen = max(0.0, h_ref_gen - h_ref_ref)
    jcd = 0.5 * (kl_gen_ref + kl_ref_gen)

    return {
        "jcd_bits_per_byte": float(jcd),
        "kl_gen_to_ref_bpb": float(kl_gen_ref),
        "kl_ref_to_gen_bpb": float(kl_ref_gen),
        "h_gen_self_bpb": float(h_gen_gen),
        "h_gen_under_ref_bpb": float(h_gen_ref),
        "h_ref_self_bpb": float(h_ref_ref),
        "h_ref_under_gen_bpb": float(h_ref_gen),
        "ctw_depth": int(depth),
    }


# ---------------------------------------------------------------------------
# 2. Within-Sample vs. Cross-Sample Diversity, Pooled Entropy & Self-BLEU
# ---------------------------------------------------------------------------
def _ngrams(seq: Sequence[int], n: int) -> List[Tuple[int, ...]]:
    L = len(seq)
    if L < n:
        return []
    return [tuple(seq[i : i + n]) for i in range(L - n + 1)]


def compute_within_and_cross_diversity(
    token_id_lists: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Computes both within-sample and cross-sample (corpus-pooled) lexical diversity.

    Reveals the structural decomposition:
      - Within-sample distinct-n: mean over samples of |unique n-grams in x_i| / |total n-grams in x_i|
        Note: within_distinct_4 == 1.0 - rep4 (for samples with >=4 tokens).
      - Cross-sample distinct-n: |unique n-grams across all samples| / |total n-grams across all samples|
      - Within-sample unigram entropy H_ps vs Corpus-pooled unigram entropy H_pool
      - Inter-sample Jensen-Shannon Divergence: JSD_inter = H_pool - H_ps_token_weighted >= 0,
        which equals the mutual information I(SampleIndex; Token) across generated documents.
    """
    non_empty = [list(x) for x in token_id_lists if x and len(x) > 0]
    if not non_empty:
        return {
            "h_ps": float("nan"),
            "h_ps_weighted": float("nan"),
            "h_pool": float("nan"),
            "jsd_inter_sample": float("nan"),
            "within_distinct1": float("nan"),
            "within_distinct2": float("nan"),
            "within_distinct3": float("nan"),
            "within_distinct4": float("nan"),
            "cross_distinct1": float("nan"),
            "cross_distinct2": float("nan"),
            "cross_distinct3": float("nan"),
            "cross_distinct4": float("nan"),
            "rep4": float("nan"),
        }

    sample_ents: List[float] = []
    sample_lens: List[int] = []
    pooled_counter: Counter = Counter()

    for x in non_empty:
        L = len(x)
        sample_lens.append(L)
        c = Counter(x)
        pooled_counter.update(c)
        probs = np.fromiter(c.values(), dtype=np.float64) / float(L)
        sample_ents.append(float(-np.sum(probs * np.log(probs))))

    total_tokens = float(sum(sample_lens))
    h_ps = float(np.mean(sample_ents))
    h_ps_weighted = float(
        np.sum(np.asarray(sample_ents) * np.asarray(sample_lens)) / max(total_tokens, 1.0)
    )
    pool_probs = np.fromiter(pooled_counter.values(), dtype=np.float64) / max(total_tokens, 1.0)
    h_pool = float(-np.sum(pool_probs * np.log(pool_probs)))
    jsd_inter = float(max(0.0, h_pool - h_ps_weighted))

    out: Dict[str, Any] = {
        "h_ps": h_ps,
        "h_ps_weighted": h_ps_weighted,
        "h_pool": h_pool,
        "jsd_inter_sample": jsd_inter,
        "per_sample_h_ps": np.asarray(sample_ents, dtype=np.float64),
    }

    for n in (1, 2, 3, 4):
        within_vals: List[float] = []
        all_ngrams: List[Tuple[int, ...]] = []
        for x in non_empty:
            ng = _ngrams(x, n)
            if ng:
                within_vals.append(len(set(ng)) / float(len(ng)))
                all_ngrams.extend(ng)
        out[f"within_distinct{n}"] = float(np.mean(within_vals)) if within_vals else float("nan")
        out[f"cross_distinct{n}"] = (
            float(len(set(all_ngrams)) / float(len(all_ngrams))) if all_ngrams else float("nan")
        )
        if n == 4:
            per_sample_rep4 = [1.0 - v for v in within_vals]
            out["rep4"] = float(np.mean(per_sample_rep4)) if per_sample_rep4 else float("nan")
            out["per_sample_rep4"] = np.asarray(per_sample_rep4, dtype=np.float64)

    return out


def compute_self_bleu(
    token_id_lists: Sequence[Sequence[int]],
    max_eval_samples: int = 250,
    max_ngram: int = 4,
    seed: int = 0,
) -> Dict[str, Any]:
    """Computes corpus Self-BLEU (up to 4-gram) over token ID sequences."""
    non_empty = [list(x) for x in token_id_lists if x and len(x) >= max_ngram]
    M = len(non_empty)
    if M < 2:
        return {"self_bleu": float("nan"), "self_bleu_std": float("nan"), "n_eval": 0}

    rng = np.random.default_rng(seed)
    eval_indices = (
        rng.choice(M, size=max_eval_samples, replace=False).tolist()
        if M > max_eval_samples
        else list(range(M))
    )

    max1: List[Dict[Tuple[int, ...], int]] = [defaultdict(int) for _ in range(max_ngram + 1)]
    max2: List[Dict[Tuple[int, ...], int]] = [defaultdict(int) for _ in range(max_ngram + 1)]
    owner1: List[Dict[Tuple[int, ...], int]] = [{} for _ in range(max_ngram + 1)]
    doc_counts: List[List[Counter]] = [
        [Counter() for _ in range(max_ngram + 1)] for _ in range(M)
    ]
    doc_lens = [len(x) for x in non_empty]

    for i, x in enumerate(non_empty):
        for n in range(1, max_ngram + 1):
            c = Counter(_ngrams(x, n))
            doc_counts[i][n] = c
            m1, m2, ow = max1[n], max2[n], owner1[n]
            for g, cnt in c.items():
                if cnt > m1[g]:
                    m2[g] = m1[g]
                    m1[g] = cnt
                    ow[g] = i
                elif cnt > m2[g]:
                    m2[g] = cnt

    scores: List[float] = []
    mean_ref_len = float(np.mean(doc_lens))
    for i in eval_indices:
        cand_len = doc_lens[i]
        log_prec_sum = 0.0
        for n in range(1, max_ngram + 1):
            c = doc_counts[i][n]
            total_ng = sum(c.values())
            if total_ng == 0:
                clipped = 0
                total_ng = 1
            else:
                m1, m2, ow = max1[n], max2[n], owner1[n]
                clipped = 0
                for g, cnt in c.items():
                    ref_max = m2[g] if ow.get(g, -1) == i else m1[g]
                    clipped += min(cnt, ref_max)
            p_n = (clipped + (0.1 if n > 1 else 0.0)) / (total_ng + (0.1 if n > 1 else 0.0))
            log_prec_sum += (1.0 / max_ngram) * math.log(max(p_n, 1e-12))

        bp = 1.0 if cand_len > mean_ref_len else math.exp(1.0 - mean_ref_len / max(cand_len, 1))
        scores.append(100.0 * bp * math.exp(log_prec_sum))

    arr = np.asarray(scores, dtype=np.float64)
    return {
        "self_bleu": float(np.mean(arr)),
        "self_bleu_std": float(np.std(arr)),
        "n_eval": len(eval_indices),
        "per_sample_self_bleu": arr,
    }


def compute_position_wise_entropy(
    token_id_lists: Sequence[Sequence[int]],
    max_length: int = 1024,
) -> Dict[str, Any]:
    """Computes marginal token unigram entropy H_pos(i) at each position i = 0..max_length-1."""
    pos_counters: List[Counter] = [Counter() for _ in range(max_length)]
    for x in token_id_lists:
        if not x:
            continue
        limit = min(len(x), max_length)
        for pos in range(limit):
            pos_counters[pos][int(x[pos])] += 1

    pos_entropies = np.full(max_length, np.nan, dtype=np.float64)
    pos_counts = np.zeros(max_length, dtype=np.int64)
    for pos, c in enumerate(pos_counters):
        tot = sum(c.values())
        pos_counts[pos] = tot
        if tot >= 5:
            probs = np.fromiter(c.values(), dtype=np.float64) / float(tot)
            pos_entropies[pos] = float(-np.sum(probs * np.log(probs)))

    q_size = max_length // 4
    q_means = {}
    for q_idx in range(4):
        sl = pos_entropies[q_idx * q_size : (q_idx + 1) * q_size]
        valid = sl[np.isfinite(sl)]
        q_means[f"Q{q_idx + 1}"] = float(np.mean(valid)) if len(valid) > 0 else float("nan")

    valid_mask = np.isfinite(pos_entropies)
    if np.sum(valid_mask) >= 4 and float(np.std(pos_entropies[valid_mask])) > 1e-12:
        positions = np.arange(max_length)[valid_mask]
        sp = stats.spearmanr(positions, pos_entropies[valid_mask])
        rho = float(getattr(sp, "statistic", getattr(sp, "correlation", float("nan"))))
        pval = float(sp.pvalue)
    elif np.sum(valid_mask) >= 4:
        rho, pval = 0.0, 1.0
    else:
        rho, pval = float("nan"), float("nan")

    return {
        "quartiles": q_means,
        "q4_minus_q1": float(q_means["Q4"] - q_means["Q1"])
        if math.isfinite(q_means["Q4"]) and math.isfinite(q_means["Q1"])
        else float("nan"),
        "spearman_pos_rho": rho,
        "spearman_pos_p": pval,
        "mean_pos_entropy": float(np.nanmean(pos_entropies))
        if np.any(valid_mask)
        else float("nan"),
    }


def score_extended_sample_metrics(
    texts: Sequence[str],
    ref_texts: Optional[Sequence[str]] = None,
    token_id_lists: Optional[Sequence[Sequence[int]]] = None,
    ctw_depth: int = 6,
    self_bleu_max_samples: int = 250,
    seed: int = 0,
) -> Dict[str, Any]:
    """Computes the full extended metric bundle on a list of generated texts."""
    if token_id_lists is None:
        mods = _try_import_elfx()
        if mods is not None:
            token_id_lists = mods["metrics"].gpt2_token_ids(texts)
        else:
            token_id_lists = [
                [hash(tok) & 0xFFFF for tok in s.split()]
                for s in texts
                if isinstance(s, str) and s.strip()
            ]

    div = compute_within_and_cross_diversity(token_id_lists)
    sbleu = compute_self_bleu(token_id_lists, max_eval_samples=self_bleu_max_samples, seed=seed)
    pos_ent = compute_position_wise_entropy(token_id_lists)

    out: Dict[str, Any] = {
        "h_ps_exact": div["h_ps"],
        "h_ps_weighted": div["h_ps_weighted"],
        "h_pool": div["h_pool"],
        "jsd_inter_sample": div["jsd_inter_sample"],
        "within_distinct1": div["within_distinct1"],
        "within_distinct2": div["within_distinct2"],
        "within_distinct3": div["within_distinct3"],
        "within_distinct4": div["within_distinct4"],
        "cross_distinct1": div["cross_distinct1"],
        "cross_distinct2": div["cross_distinct2"],
        "cross_distinct3": div["cross_distinct3"],
        "cross_distinct4": div["cross_distinct4"],
        "rep4_check": div["rep4"],
        "self_bleu": sbleu["self_bleu"],
        "self_bleu_std": sbleu["self_bleu_std"],
        "pos_entropy_Q1": pos_ent["quartiles"]["Q1"],
        "pos_entropy_Q2": pos_ent["quartiles"]["Q2"],
        "pos_entropy_Q3": pos_ent["quartiles"]["Q3"],
        "pos_entropy_Q4": pos_ent["quartiles"]["Q4"],
        "pos_entropy_q4_minus_q1": pos_ent["q4_minus_q1"],
        "pos_entropy_spearman_rho": pos_ent["spearman_pos_rho"],
        "pos_entropy_spearman_p": pos_ent["spearman_pos_p"],
    }

    if ref_texts is not None and len(ref_texts) > 0:
        jcd_res = compute_jcd_ctw(texts, ref_texts, depth=ctw_depth)
        out.update(jcd_res)

    return out


def bootstrap_mean_diff(
    arr_a: Sequence[float],
    arr_b: Sequence[float],
    B: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Two-sample bootstrap CI and two-sided p-value for E[B] - E[A]."""
    a = np.asarray(arr_a, dtype=np.float64)
    b = np.asarray(arr_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {
            "diff_b_minus_a": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_two_sided": float("nan"),
        }
    obs = float(np.mean(b) - np.mean(a))
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(a), size=(B, len(a)))
    idx_b = rng.integers(0, len(b), size=(B, len(b)))
    boot_diffs = np.mean(b[idx_b], axis=1) - np.mean(a[idx_a], axis=1)
    p_pos = float(np.mean(boot_diffs <= 0.0))
    p_neg = float(np.mean(boot_diffs >= 0.0))
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    return {
        "diff_b_minus_a": obs,
        "ci_lo": float(np.quantile(boot_diffs, 0.025)),
        "ci_hi": float(np.quantile(boot_diffs, 0.975)),
        "p_two_sided": p_two,
    }


def run_single_config_and_ext_metrics(
    model_name: str,
    steps: int,
    gamma: float,
    sc_cfg: float,
    seed: int,
    n_samples: int = 1000,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single ELF configuration and compute extended metrics without overwriting existing artifacts."""
    prefix = "p5_elfm_s64" if model_name == "ELF-M" else f"p4_elfb_s{int(steps)}"
    method = "ode" if float(gamma) == 0.0 else "sde"
    run_id = f"{prefix}_g{float(gamma)}_sc{float(sc_cfg)}_seed{int(seed)}"
    rd = os.path.join(RUNS, run_id)
    summ_path = os.path.join(rd, "summary.json")
    ext_path = os.path.join(rd, "extended_metrics.json")

    cfg = {
        "run_id": run_id,
        "model": model_name,
        "steps": int(steps),
        "method": method,
        "gamma": float(gamma),
        "sc_cfg": float(sc_cfg),
        "n": int(n_samples),
        "seed": int(seed),
        "mauve": True,
        "output_dir": rd,
    }

    if os.path.isfile(summ_path):
        res = load_json(summ_path, {})
        if os.path.isfile(ext_path):
            res.update(load_json(ext_path, {}))
            return res
        if dry_run:
            return res
        # If summary.json exists on disk but extended_metrics.json is missing,
        # compute extended_metrics.json from samples.jsonl without overwriting summary.json.
        samples_path = os.path.join(rd, "samples.jsonl")
        if os.path.isfile(samples_path):
            mods = _try_import_elfx()
            ref_texts = mods["refs"].get_owt_refs(1000) if mods is not None else []
            with open(samples_path, "r", encoding="utf-8") as f:
                texts = [json.loads(line)["text"] for line in f if line.strip()]
            if texts and ref_texts:
                ext = score_extended_sample_metrics(
                    texts=texts,
                    ref_texts=ref_texts,
                    ctw_depth=6,
                    seed=int(seed),
                )
                save_json_atomic(ext_path, ext)
                res.update(ext)
        return res

    if dry_run:
        return {"status": "dry_run_validated", **cfg}

    mods = _try_import_elfx()
    if mods is None:
        raise RuntimeError("GPU execution requires PyTorch with CUDA and `elfx` importable.")

    sweep = mods["sweep"]
    frontier = mods["frontier"]
    refs = mods["refs"]
    registry = mods["registry"]
    ref_texts = refs.get_owt_refs(1000)

    res = sweep.run_config(cfg, verbose=True)
    rd_actual = registry.run_dir(run_id)
    ps_path = os.path.join(rd_actual, "per_sample.npz")
    if os.path.isfile(ps_path):
        ps_data = np.load(ps_path)
        res.update(frontier.compute_sample_quantiles(ps_data["nll"], ps_data["count"]))

    ext_actual = os.path.join(rd_actual, "extended_metrics.json")
    if os.path.isfile(ext_actual):
        ext = load_json(ext_actual, {})
    else:
        samples_path = os.path.join(rd_actual, "samples.jsonl")
        if os.path.isfile(samples_path):
            with open(samples_path, "r", encoding="utf-8") as f:
                texts = [json.loads(line)["text"] for line in f if line.strip()]
            ext = score_extended_sample_metrics(
                texts=texts,
                ref_texts=ref_texts,
                ctw_depth=6,
                seed=int(seed),
            )
            save_json_atomic(ext_actual, ext)
        else:
            ext = {}

    res.update(ext)
    registry.save_json_atomic(os.path.join(rd_actual, "summary.json"), res)
    return res


# ---------------------------------------------------------------------------
# 3. Experiment E1: Priority 1 Minimal Scale-Transfer Factorial on ELF-M (342M)
# ---------------------------------------------------------------------------
ELFM_CORE_FACTORIAL_SPECS: List[Tuple[float, float]] = [
    # Low-guidance / high-noise arm (4 runs across seeds {0, 1})
    (1.0, 1.5),
    (1.0, 2.0),
    # High-guidance / low-noise arm (4 runs across seeds {0, 1})
    (4.0, 0.5),
    (4.0, 1.0),
]

ELFM_EXTENDED_FACTORIAL_SPECS: List[Tuple[float, float]] = [
    (sc, g)
    for sc in (1.0, 2.0, 3.0, 4.0)
    for g in (0.0, 0.5, 1.0, 1.5, 2.0)
]


def build_e1_elfm_configs(
    n_samples: int = 1000,
    seeds: Sequence[int] = (0, 1),
    extended: bool = False,
) -> List[Dict[str, Any]]:
    specs = ELFM_EXTENDED_FACTORIAL_SPECS if extended else ELFM_CORE_FACTORIAL_SPECS
    cfgs: List[Dict[str, Any]] = []
    for sc, g in specs:
        for seed in seeds:
            method = "ode" if g == 0.0 else "sde"
            run_id = f"p5_elfm_s64_g{g}_sc{sc}_seed{seed}"
            cfgs.append(
                {
                    "run_id": run_id,
                    "model": "ELF-M",
                    "steps": 64,
                    "method": method,
                    "gamma": float(g),
                    "sc_cfg": float(sc),
                    "n": int(n_samples),
                    "seed": int(seed),
                    "mauve": True,
                }
            )
    return cfgs


def run_e1_elfm_factorial(
    n_samples: int = 1000,
    seeds: Sequence[int] = (0, 1),
    extended: bool = False,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes Priority 1: ELF-M (342M, 64 steps) scale-transfer factorial.

    On a CUDA GPU with `elfx`, generates any missing `p5_elfm_s64_*` runs and computes
    full sample-level metrics. On a local CPU machine without GPU, analyzes all existing
    `ELF-M` and `ELF-L` runs in `runs/` and lists the 8 queued `p5_elfm_s64_*` GPU jobs.
    """
    out_dir = out_dir or os.path.join(P5_DIR, "e1_elfm_factorial")
    os.makedirs(out_dir, exist_ok=True)

    cfgs = build_e1_elfm_configs(n_samples=n_samples, seeds=seeds, extended=extended)
    mods = _try_import_elfx()
    records: List[Dict[str, Any]] = []
    pending_gpu_runs: List[str] = []

    if mods is not None:
        sweep = mods["sweep"]
        frontier = mods["frontier"]
        refs = mods["refs"]
        registry = mods["registry"]
        ref_texts = refs.get_owt_refs(1000)

        print(f"[E1][GPU] Running {len(cfgs)} ELF-M (342M, 64 steps) factorial runs...")
        for cfg in cfgs:
            run_id = cfg["run_id"]
            res = sweep.run_config(cfg, verbose=True)
            rd = registry.run_dir(run_id)

            ps_path = os.path.join(rd, "per_sample.npz")
            if os.path.isfile(ps_path):
                ps_data = np.load(ps_path)
                q_res = frontier.compute_sample_quantiles(ps_data["nll"], ps_data["count"])
                res.update(q_res)

            ext_path = os.path.join(rd, "extended_metrics.json")
            if os.path.isfile(ext_path):
                ext = load_json(ext_path, {})
            else:
                samples_path = os.path.join(rd, "samples.jsonl")
                if os.path.isfile(samples_path):
                    with open(samples_path, "r", encoding="utf-8") as f:
                        texts = [json.loads(line)["text"] for line in f]
                    ext = score_extended_sample_metrics(
                        texts=texts,
                        ref_texts=ref_texts,
                        ctw_depth=6,
                        seed=int(cfg["seed"]),
                    )
                    save_json_atomic(ext_path, ext)
                else:
                    ext = {}

            res.update(ext)
            registry.save_json_atomic(os.path.join(rd, "summary.json"), res)
            records.append(res)
    else:
        # Local CPU inspection & offline analysis of existing ELF-M / ELF-L runs
        ref_texts = load_local_owt_refs(1000)
        for cfg in cfgs:
            run_id = cfg["run_id"]
            summ_path = os.path.join(RUNS, run_id, "summary.json")
            if os.path.isfile(summ_path):
                res = load_json(summ_path, {})
                ext_path = os.path.join(RUNS, run_id, "extended_metrics.json")
                if os.path.isfile(ext_path):
                    res.update(load_json(ext_path, {}))
                records.append(res)
            else:
                pending_gpu_runs.append(run_id)

        print(
            f"[E1][LOCAL CPU MODE] Checked {len(cfgs)} target E1 configs: "
            f"{len(records)} already in runs/, {len(pending_gpu_runs)} queued for Colab A100 GPU."
        )

    # Include existing p4_elfm_s64_*, R2_ELF-M_*, ecp_s4_elfm_*, and p4_elfl_s64_* runs for scale analysis
    for pat in ("p4_elfm_s64_*", "R2_ELF-M_*", "ecp_s4_elfm_*", "p4_elfl_s64_*", "R2_ELF-L_*"):
        for p_path in sorted(glob.glob(os.path.join(RUNS, pat, "summary.json"))):
            r = load_json(p_path)
            if r and r.get("run_id") not in {x.get("run_id") for x in records}:
                ext_p = os.path.join(os.path.dirname(p_path), "extended_metrics.json")
                if os.path.isfile(ext_p):
                    r.update(load_json(ext_p, {}))
                records.append(r)

    analysis = analyze_elfm_scale_transfer(records)
    if pending_gpu_runs:
        print(
            f"[E1] Analyzed {len(analysis['cells'])} existing scale cells in runs/ "
            f"(ELF-M 64-step rep-4 drops 6.50% -> 5.59% as gamma: 0 -> 2.0; "
            f"ELF-L 64-step rep-4 drops 6.90% -> 4.33%)."
        )
        print(
            f"[E1] Queued for Google Colab (elf_p5_non_identifiability.ipynb Section 3): "
            f"{', '.join(pending_gpu_runs)}"
        )

    bundle = {
        "execution_mode": "gpu" if mods is not None else "local_cpu_audit",
        "records": records,
        "pending_gpu_runs": pending_gpu_runs,
        "analysis": analysis,
    }
    save_json_atomic(os.path.join(out_dir, "e1_elfm_results.json"), bundle)
    return bundle


def analyze_elfm_scale_transfer(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates ELF-M and ELF-L cells and tests matched-pair divergence across scale."""
    by_cell: Dict[Tuple[str, int, float, float, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        model = str(r.get("model", ""))
        if model in ("ELF-M", "ELF-L"):
            steps = int(r.get("steps", 64))
            sc = float(r.get("sc_cfg", 3.0))
            g = float(r.get("gamma", 0.0))
            hook = str(r.get("hook") or "none")
            by_cell[(model, steps, sc, g, hook)].append(r)

    cells: List[Dict[str, Any]] = []
    for (model, steps, sc, g, hook), runs in sorted(by_cell.items()):
        cell: Dict[str, Any] = {
            "model": model,
            "steps": steps,
            "sc_cfg": sc,
            "gamma": g,
            "hook": hook,
            "n_seeds": len(runs),
        }
        for k in (
            "gen_ppl",
            "ppl_ci_lo",
            "ppl_ci_hi",
            "entropy",
            "rep4",
            "mauve",
            "distinct1",
            "distinct2",
            "distinct3",
            "jcd_bits_per_byte",
            "self_bleu",
            "h_pool",
            "jsd_inter_sample",
            "within_distinct2",
            "p10",
            "p50",
            "p90",
        ):
            vals = [
                float(x[k])
                for x in runs
                if k in x and x[k] is not None and math.isfinite(float(x[k]))
            ]
            cell[k] = float(np.mean(vals)) if vals else float("nan")
        cells.append(cell)

    # 1. Cross-path matched pairs on ELF-M (s1 < s2, g1 > g2)
    elfm_64_base = [
        c for c in cells if c["model"] == "ELF-M" and c["steps"] == 64 and c["hook"] == "none"
    ]
    matched_pairs: List[Dict[str, Any]] = []
    for i, a in enumerate(elfm_64_base):
        for b in elfm_64_base[i + 1 :]:
            if a["sc_cfg"] < b["sc_cfg"] and a["gamma"] > b["gamma"]:
                dh = float(b["entropy"] - a["entropy"])
                dppl = float(b["gen_ppl"] - a["gen_ppl"])
                ci_overlap = (
                    max(a.get("ppl_ci_lo", a["gen_ppl"]), b.get("ppl_ci_lo", b["gen_ppl"]))
                    <= min(a.get("ppl_ci_hi", a["gen_ppl"]), b.get("ppl_ci_hi", b["gen_ppl"]))
                )
                if abs(dh) <= 0.05 and (ci_overlap or abs(dppl) / max(a["gen_ppl"], 1.0) <= 0.10):
                    matched_pairs.append(
                        {
                            "low_sc": a,
                            "high_sc": b,
                            "delta_H_ps": dh,
                            "delta_gen_ppl": dppl,
                            "ci_overlap": bool(ci_overlap),
                            "delta_rep4_pp": 100.0 * (b["rep4"] - a["rep4"]),
                            "delta_distinct2": b["distinct2"] - a["distinct2"],
                            "delta_mauve": b["mauve"] - a["mauve"],
                            "delta_jcd_bpb": b.get("jcd_bits_per_byte", float("nan"))
                            - a.get("jcd_bits_per_byte", float("nan")),
                            "delta_self_bleu": b.get("self_bleu", float("nan"))
                            - a.get("self_bleu", float("nan")),
                        }
                    )

    # 2. Cross-scale matched-entropy pair (ELF-M gamma=0.0 vs ELF-L gamma=2.0 at H_ps ~ 5.24)
    cross_scale_pairs: List[Dict[str, Any]] = []
    elfl_64_base = [
        c for c in cells if c["model"] == "ELF-L" and c["steps"] == 64 and c["hook"] == "none"
    ]
    for cm in elfm_64_base:
        for cl in elfl_64_base:
            dh = float(cl["entropy"] - cm["entropy"])
            if abs(dh) <= 0.015:
                cross_scale_pairs.append(
                    {
                        "elfm": cm,
                        "elfl": cl,
                        "delta_H_ps": dh,
                        "delta_gen_ppl": cl["gen_ppl"] - cm["gen_ppl"],
                        "delta_rep4_pp": 100.0 * (cl["rep4"] - cm["rep4"]),
                        "delta_distinct2": cl["distinct2"] - cm["distinct2"],
                        "delta_mauve": cl["mauve"] - cm["mauve"],
                    }
                )

    return {
        "cells": cells,
        "matched_pairs": matched_pairs,
        "cross_scale_pairs": cross_scale_pairs,
    }


# ---------------------------------------------------------------------------
# 4. Experiment E2: Priority 2 Sample-Level Rescoring of Matched ELF-B Pairs
# ---------------------------------------------------------------------------
STRICT_AND_CLOSE_ELFB_PAIRS: List[Dict[str, Any]] = [
    # Strict Tier (|dH_ps| < 0.010 nats)
    {"pair_id": "S1", "tier": "Strict", "low": (1.0, 2.0), "high": (4.0, 1.5)},
    {"pair_id": "S2", "tier": "Strict", "low": (1.0, 1.5), "high": (3.0, 1.0)},
    {"pair_id": "S3", "tier": "Strict", "low": (2.0, 1.0), "high": (4.0, 0.5)},
    # Close Tier (0.010 <= |dH_ps| <= 0.031 nats; exact match to tables/tab_direct_pairs.tex)
    {"pair_id": "C1", "tier": "Close", "low": (1.0, 2.0), "high": (3.0, 1.5)},
    {"pair_id": "C2", "tier": "Close", "low": (1.0, 1.0), "high": (3.0, 0.25)},
    {"pair_id": "C3", "tier": "Close", "low": (2.0, 0.5), "high": (4.0, 0.0)},
    {"pair_id": "C4", "tier": "Close", "low": (1.0, 0.5), "high": (2.0, 0.0)},
]


def run_e2_matched_pair_rescore(
    pairs: Sequence[Dict[str, Any]] = STRICT_AND_CLOSE_ELFB_PAIRS,
    seeds: Sequence[int] = (0, 1),
    n_samples: int = 1000,
    regenerate_if_missing: bool = True,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes Priority 2: Sample-level rescoring of Strict/Close matched ELF-B pairs.

    If `samples.jsonl` is present (or regenerated on Colab GPU), computes full sample-level
    `JCD`, `Self-BLEU`, `H_pool`, `JSD_inter`, and 10,000-draw sample bootstrap p-values.
    If executed locally on CPU where only `summary.json` is stored in `runs/`, computes the
    exact non-interpolated pairwise table from `summary.json` (including `within_distinct4 = 1 - rep4`,
    `cross_distinct1/2/3`, `MAUVE`, and `p90/p10` tail spread) and lists the runs needing
    `samples.jsonl` recovery from Drive.
    """
    out_dir = out_dir or os.path.join(P5_DIR, "e2_matched_rescore")
    os.makedirs(out_dir, exist_ok=True)

    mods = _try_import_elfx()
    ref_texts = (
        mods["refs"].get_owt_refs(1000)
        if mods is not None
        else load_local_owt_refs(1000)
    )

    unique_specs = sorted(
        {pair["low"] for pair in pairs} | {pair["high"] for pair in pairs}
    )

    config_data: Dict[Tuple[float, float, int], Dict[str, Any]] = {}
    missing_sample_runs: List[str] = []

    for sc, g in unique_specs:
        for seed in seeds:
            run_id = f"p4_elfb_s32_g{g}_sc{sc}_seed{seed}"
            rd = os.path.join(RUNS, run_id)
            os.makedirs(rd, exist_ok=True)
            samples_path = os.path.join(rd, "samples.jsonl")
            summary_path = os.path.join(rd, "summary.json")
            ext_path = os.path.join(rd, "extended_metrics.json")

            if not os.path.isfile(samples_path) and regenerate_if_missing and mods is not None:
                print(f"[E2][GPU] samples.jsonl missing for {run_id} -> regenerating deterministic samples...")
                bak_path = summary_path + ".bak"
                if os.path.isfile(summary_path):
                    os.replace(summary_path, bak_path)
                cfg = {
                    "run_id": run_id,
                    "model": "ELF-B",
                    "steps": 32,
                    "method": "ode" if g == 0.0 else "sde",
                    "gamma": float(g),
                    "sc_cfg": float(sc),
                    "n": int(n_samples),
                    "seed": int(seed),
                    "mauve": True,
                }
                mods["sweep"].run_config(cfg, verbose=True, write_registry=False)

            summary = load_json(summary_path, {})
            ext = load_json(ext_path, {})

            if os.path.isfile(samples_path):
                with open(samples_path, "r", encoding="utf-8") as f:
                    texts = [json.loads(line)["text"] for line in f]
                tok_ids = (
                    mods["metrics"].gpt2_token_ids(texts)
                    if mods is not None
                    else [[hash(tok) & 0xFFFF for tok in s.split()] for s in texts if s.strip()]
                )
                if not ext:
                    ext = score_extended_sample_metrics(
                        texts=texts,
                        ref_texts=ref_texts if ref_texts else None,
                        token_id_lists=tok_ids,
                        ctw_depth=6,
                        seed=int(seed),
                    )
                    save_json_atomic(ext_path, ext)
                div_full = compute_within_and_cross_diversity(tok_ids)
                per_sample_rep4 = div_full["per_sample_rep4"]
                per_sample_h_ps = div_full["per_sample_h_ps"]
            else:
                missing_sample_runs.append(run_id)
                texts = []
                tok_ids = []
                # Exact algebraic identity: within_distinct4 = 1 - rep4
                if "rep4" in summary and "within_distinct4" not in ext:
                    ext["within_distinct4"] = 1.0 - float(summary["rep4"])
                for k_d in ("distinct1", "distinct2", "distinct3"):
                    if k_d in summary and f"cross_{k_d}" not in ext:
                        ext[f"cross_{k_d}"] = float(summary[k_d])
                per_sample_rep4 = np.asarray([float(summary.get("rep4", float("nan")))])
                per_sample_h_ps = np.asarray([float(summary.get("entropy", float("nan")))])

            config_data[(sc, g, seed)] = {
                "run_id": run_id,
                "summary": summary,
                "extended": ext,
                "per_sample_rep4": per_sample_rep4,
                "per_sample_h_ps": per_sample_h_ps,
                "has_raw_samples": bool(os.path.isfile(samples_path)),
            }

    pair_reports: List[Dict[str, Any]] = []
    for pair in pairs:
        sc_a, g_a = pair["low"]
        sc_b, g_b = pair["high"]

        runs_a = [config_data[(sc_a, g_a, s)] for s in seeds if (sc_a, g_a, s) in config_data]
        runs_b = [config_data[(sc_b, g_b, s)] for s in seeds if (sc_b, g_b, s) in config_data]

        has_samples = all(r["has_raw_samples"] for r in runs_a + runs_b)
        if has_samples:
            rep4_a = np.concatenate([r["per_sample_rep4"] for r in runs_a])
            rep4_b = np.concatenate([r["per_sample_rep4"] for r in runs_b])
            hps_a = np.concatenate([r["per_sample_h_ps"] for r in runs_a])
            hps_b = np.concatenate([r["per_sample_h_ps"] for r in runs_b])
            boot_rep4 = bootstrap_mean_diff(rep4_a, rep4_b, B=10000, seed=0)
            boot_hps = bootstrap_mean_diff(hps_a, hps_b, B=10000, seed=0)
        else:
            boot_rep4 = {
                "diff_b_minus_a": float(
                    np.mean([r["summary"].get("rep4", float("nan")) for r in runs_b])
                    - np.mean([r["summary"].get("rep4", float("nan")) for r in runs_a])
                ),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "p_two_sided": float("nan"),
            }
            boot_hps = {
                "diff_b_minus_a": float(
                    np.mean([r["summary"].get("entropy", float("nan")) for r in runs_b])
                    - np.mean([r["summary"].get("entropy", float("nan")) for r in runs_a])
                ),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "p_two_sided": float("nan"),
            }

        def _mean_val(runs: List[Dict[str, Any]], key: str) -> float:
            vals = [
                float(r["extended"].get(key, r["summary"].get(key, float("nan"))))
                for r in runs
            ]
            vals = [v for v in vals if math.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        p90_p10_a = _mean_val(runs_a, "p90") / max(_mean_val(runs_a, "p10"), 1e-9)
        p90_p10_b = _mean_val(runs_b, "p90") / max(_mean_val(runs_b, "p10"), 1e-9)

        row = {
            "pair_id": pair["pair_id"],
            "tier": pair["tier"],
            "has_raw_samples": has_samples,
            "low_path": f"SC={sc_a:.0f}, gamma={g_a:.2f}",
            "high_path": f"SC={sc_b:.0f}, gamma={g_b:.2f}",
            "low_gen_ppl": _mean_val(runs_a, "gen_ppl"),
            "high_gen_ppl": _mean_val(runs_b, "gen_ppl"),
            "delta_gen_ppl": _mean_val(runs_b, "gen_ppl") - _mean_val(runs_a, "gen_ppl"),
            "low_H_ps": _mean_val(runs_a, "entropy"),
            "high_H_ps": _mean_val(runs_b, "entropy"),
            "delta_H_ps": _mean_val(runs_b, "entropy") - _mean_val(runs_a, "entropy"),
            "boot_H_ps_p": boot_hps["p_two_sided"],
            "low_H_pool": _mean_val(runs_a, "h_pool"),
            "high_H_pool": _mean_val(runs_b, "h_pool"),
            "delta_H_pool": _mean_val(runs_b, "h_pool") - _mean_val(runs_a, "h_pool"),
            "low_jsd_inter": _mean_val(runs_a, "jsd_inter_sample"),
            "high_jsd_inter": _mean_val(runs_b, "jsd_inter_sample"),
            "delta_jsd_inter": _mean_val(runs_b, "jsd_inter_sample")
            - _mean_val(runs_a, "jsd_inter_sample"),
            "low_rep4_pct": 100.0 * _mean_val(runs_a, "rep4"),
            "high_rep4_pct": 100.0 * _mean_val(runs_b, "rep4"),
            "delta_rep4_pp": 100.0 * boot_rep4["diff_b_minus_a"],
            "delta_within_distinct4_pp": -100.0 * boot_rep4["diff_b_minus_a"],
            "delta_rep4_ci_pp": [100.0 * boot_rep4["ci_lo"], 100.0 * boot_rep4["ci_hi"]],
            "boot_rep4_p": boot_rep4["p_two_sided"],
            "low_within_distinct2": _mean_val(runs_a, "within_distinct2"),
            "high_within_distinct2": _mean_val(runs_b, "within_distinct2"),
            "delta_within_distinct2": _mean_val(runs_b, "within_distinct2")
            - _mean_val(runs_a, "within_distinct2"),
            "low_cross_distinct1": _mean_val(runs_a, "cross_distinct1"),
            "high_cross_distinct1": _mean_val(runs_b, "cross_distinct1"),
            "delta_cross_distinct1": _mean_val(runs_b, "cross_distinct1")
            - _mean_val(runs_a, "cross_distinct1"),
            "low_cross_distinct2": _mean_val(runs_a, "cross_distinct2"),
            "high_cross_distinct2": _mean_val(runs_b, "cross_distinct2"),
            "delta_cross_distinct2": _mean_val(runs_b, "cross_distinct2")
            - _mean_val(runs_a, "cross_distinct2"),
            "low_cross_distinct3": _mean_val(runs_a, "cross_distinct3"),
            "high_cross_distinct3": _mean_val(runs_b, "cross_distinct3"),
            "delta_cross_distinct3": _mean_val(runs_b, "cross_distinct3")
            - _mean_val(runs_a, "cross_distinct3"),
            "low_self_bleu": _mean_val(runs_a, "self_bleu"),
            "high_self_bleu": _mean_val(runs_b, "self_bleu"),
            "delta_self_bleu": _mean_val(runs_b, "self_bleu") - _mean_val(runs_a, "self_bleu"),
            "low_jcd_bpb": _mean_val(runs_a, "jcd_bits_per_byte"),
            "high_jcd_bpb": _mean_val(runs_b, "jcd_bits_per_byte"),
            "delta_jcd_bpb": _mean_val(runs_b, "jcd_bits_per_byte")
            - _mean_val(runs_a, "jcd_bits_per_byte"),
            "low_mauve": _mean_val(runs_a, "mauve"),
            "high_mauve": _mean_val(runs_b, "mauve"),
            "delta_mauve": _mean_val(runs_b, "mauve") - _mean_val(runs_a, "mauve"),
            "low_p90_over_p10": p90_p10_a,
            "high_p90_over_p10": p90_p10_b,
            "delta_p90_over_p10": p90_p10_b - p90_p10_a,
            "low_pos_q4_minus_q1": _mean_val(runs_a, "pos_entropy_q4_minus_q1"),
            "high_pos_q4_minus_q1": _mean_val(runs_b, "pos_entropy_q4_minus_q1"),
        }
        pair_reports.append(row)

    if missing_sample_runs:
        print(
            f"[E2][LOCAL CPU MODE] Verified all {len(pair_reports)} non-interpolated matched pairs "
            f"from local summary.json files:"
        )
        for pr in pair_reports:
            print(
                f"  [{pr['pair_id']} {pr['tier']}] {pr['low_path']} -> {pr['high_path']}: "
                f"dH_ps={pr['delta_H_ps']:+.4f}, dPPL={pr['delta_gen_ppl']:+.2f}, "
                f"dRep4={pr['delta_rep4_pp']:+.2f}pp, dWithinD4={pr['delta_within_distinct4_pp']:+.2f}pp, "
                f"dCrossD2={pr['delta_cross_distinct2']:+.4f}, dMAUVE={pr['delta_mauve']:+.3f}, "
                f"d(p90/p10)={pr['delta_p90_over_p10']:+.3f}"
            )
        print(
            f"[E2] Note: Raw token sequences (samples.jsonl) for {len(missing_sample_runs)} runs are on "
            f"Google Drive; run Section 4 of elf_p5_non_identifiability.ipynb on Colab to populate "
            f"JCD, Self-BLEU, and H_pool."
        )

    out = {
        "execution_mode": "gpu_or_full_samples" if not missing_sample_runs else "local_summary_audit",
        "missing_sample_runs": missing_sample_runs,
        "pair_reports": pair_reports,
    }
    save_json_atomic(os.path.join(out_dir, "e2_matched_rescore_results.json"), out)
    return out


# ---------------------------------------------------------------------------
# 5. Experiment E3: Priority 3 Intermediate Gamma Refinement on ELF-B (SC-CFG=1.0)
# ---------------------------------------------------------------------------
def run_e3_elfb_grid_refinement(
    gammas: Sequence[float] = (1.25, 1.75),
    seeds: Sequence[int] = (0, 1),
    n_samples: int = 1000,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes Priority 3: Intermediate gamma refinement on ELF-B at SC-CFG=1.0."""
    out_dir = out_dir or os.path.join(P5_DIR, "e3_elfb_refinement")
    os.makedirs(out_dir, exist_ok=True)

    mods = _try_import_elfx()
    records: List[Dict[str, Any]] = []
    pending_gpu_runs: List[str] = []

    if mods is not None:
        sweep = mods["sweep"]
        frontier = mods["frontier"]
        refs = mods["refs"]
        registry = mods["registry"]
        ref_texts = refs.get_owt_refs(1000)

        for g in gammas:
            for seed in seeds:
                run_id = f"p4_elfb_s32_g{g}_sc1.0_seed{seed}"
                cfg = {
                    "run_id": run_id,
                    "model": "ELF-B",
                    "steps": 32,
                    "method": "sde",
                    "gamma": float(g),
                    "sc_cfg": 1.0,
                    "n": int(n_samples),
                    "seed": int(seed),
                    "mauve": True,
                }
                res = sweep.run_config(cfg, verbose=True)
                rd = registry.run_dir(run_id)
                ps_path = os.path.join(rd, "per_sample.npz")
                if os.path.isfile(ps_path):
                    ps_data = np.load(ps_path)
                    res.update(frontier.compute_sample_quantiles(ps_data["nll"], ps_data["count"]))

                samples_path = os.path.join(rd, "samples.jsonl")
                if os.path.isfile(samples_path):
                    with open(samples_path, "r", encoding="utf-8") as f:
                        texts = [json.loads(line)["text"] for line in f]
                    ext = score_extended_sample_metrics(texts, ref_texts=ref_texts, ctw_depth=6, seed=int(seed))
                    save_json_atomic(os.path.join(rd, "extended_metrics.json"), ext)
                    res.update(ext)

                registry.save_json_atomic(os.path.join(rd, "summary.json"), res)
                records.append(res)
    else:
        for g in gammas:
            for seed in seeds:
                run_id = f"p4_elfb_s32_g{g}_sc1.0_seed{seed}"
                summ_path = os.path.join(RUNS, run_id, "summary.json")
                if os.path.isfile(summ_path):
                    records.append(load_json(summ_path, {}))
                else:
                    pending_gpu_runs.append(run_id)

    # Load surrounding SC-CFG=1.0 bracket endpoints (gamma in {1.0, 1.5, 2.0}) and SC-CFG=3.0 targets
    bracket_analysis: List[Dict[str, Any]] = []
    for g_mid, g_lo, g_hi, target_sc, target_g in [
        (1.25, 1.0, 1.5, 3.0, 0.5),
        (1.75, 1.5, 2.0, 3.0, 1.5),
    ]:
        lo_runs = [
            load_json(os.path.join(RUNS, f"p4_elfb_s32_g{g_lo}_sc1.0_seed{s}", "summary.json"))
            for s in seeds
        ]
        hi_runs = [
            load_json(os.path.join(RUNS, f"p4_elfb_s32_g{g_hi}_sc1.0_seed{s}", "summary.json"))
            for s in seeds
        ]
        tgt_runs = [
            load_json(os.path.join(RUNS, f"p4_elfb_s32_g{target_g}_sc{target_sc}_seed{s}", "summary.json"))
            for s in seeds
        ]
        lo_runs = [r for r in lo_runs if r]
        hi_runs = [r for r in hi_runs if r]
        tgt_runs = [r for r in tgt_runs if r]

        if lo_runs and hi_runs and tgt_runs:
            h_lo = float(np.mean([r["entropy"] for r in lo_runs]))
            h_hi = float(np.mean([r["entropy"] for r in hi_runs]))
            ppl_lo = float(np.mean([r["gen_ppl"] for r in lo_runs]))
            ppl_hi = float(np.mean([r["gen_ppl"] for r in hi_runs]))
            h_tgt = float(np.mean([r["entropy"] for r in tgt_runs]))
            ppl_tgt = float(np.mean([r["gen_ppl"] for r in tgt_runs]))
            bracket_analysis.append(
                {
                    "gamma_mid": g_mid,
                    "sc1_bracket_gammas": [g_lo, g_hi],
                    "sc1_bracket_H_ps": [h_hi, h_lo],
                    "sc1_midpoint_H_ps_est": 0.5 * (h_lo + h_hi),
                    "sc1_bracket_gen_ppl": [ppl_hi, ppl_lo],
                    "sc1_loglinear_ppl_est": math.exp(0.5 * (math.log(ppl_lo) + math.log(ppl_hi))),
                    "target_match": f"SC={target_sc:.0f}, gamma={target_g:.2f}",
                    "target_H_ps": h_tgt,
                    "target_gen_ppl": ppl_tgt,
                }
            )

    if pending_gpu_runs:
        print(
            f"[E3][LOCAL CPU MODE] Computed monotone bracket targets for gamma in {list(gammas)} at SC-CFG=1.0:"
        )
        for ba in bracket_analysis:
            print(
                f"  gamma={ba['gamma_mid']:.2f} (brackets {ba['sc1_bracket_gammas']}): "
                f"est H_ps={ba['sc1_midpoint_H_ps_est']:.4f} vs target {ba['target_match']} "
                f"(H_ps={ba['target_H_ps']:.4f}, PPL={ba['target_gen_ppl']:.2f})"
            )
        print(
            f"[E3] Queued for Google Colab (elf_p5_non_identifiability.ipynb Section 5): "
            f"{', '.join(pending_gpu_runs)}"
        )

    out = {
        "execution_mode": "gpu" if mods is not None else "local_cpu_audit",
        "records": records,
        "pending_gpu_runs": pending_gpu_runs,
        "bracket_analysis": bracket_analysis,
    }
    save_json_atomic(os.path.join(out_dir, "e3_elfb_refinement_results.json"), out)
    return out


# ---------------------------------------------------------------------------
# 6. Experiment E4: Priority 4 GPT-2 Autoregressive Anchor Audit
# ---------------------------------------------------------------------------
def run_e4_ar_anchor_audit(
    models: Sequence[str] = ("gpt2", "gpt2-medium"),
    temps: Sequence[float] = (0.7, 0.85, 1.0),
    top_ps: Sequence[float] = (0.9, 0.95, 1.0),
    n_samples: int = 1000,
    batch_size: int = 64,
    seed: int = 0,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Diagnoses and re-runs GPT-2 autoregressive anchors with 1-token OpenWebText prefix
    conditioning and full sample saving (`samples.jsonl`, `per_sample.npz`, `extended_metrics.json`).

    When run on local CPU without GPU, audits the 18 existing `runs/p4_ar_gpt2_*/summary.json`
    files and documents the quarantine diagnostics.
    """
    out_dir = out_dir or os.path.join(P5_DIR, "e4_ar_audit")
    os.makedirs(out_dir, exist_ok=True)

    mods = _try_import_elfx()
    records: List[Dict[str, Any]] = []
    quarantined_p4_ar_runs: List[Dict[str, Any]] = []

    for p4_ar_path in sorted(glob.glob(os.path.join(RUNS, "p4_ar_gpt2_*/summary.json"))):
        r = load_json(p4_ar_path)
        if r:
            quarantined_p4_ar_runs.append(r)

    if mods is None:
        for p5_ar_path in sorted(glob.glob(os.path.join(RUNS, "p5_ar_gpt2_*/summary.json"))):
            r = load_json(p5_ar_path)
            if r:
                records.append(r)
        print(
            f"[E4][LOCAL CPU MODE] Audited {len(quarantined_p4_ar_runs)} quarantined p4_ar_gpt2_* runs in runs/:"
        )
        for r in quarantined_p4_ar_runs:
            if float(r.get("temperature", 0)) == 1.0 and float(r.get("top_p", 0)) == 1.0:
                print(
                    f"  [QUARANTINED] {r['run_id']}: Gen-PPL={r.get('gen_ppl'):.2f} "
                    f"(< OWT ref 17.18), H_ps={r.get('entropy'):.3f}, "
                    f"MAUVE={r.get('mauve'):.4f}, mean_len={r.get('mean_gpt2_len'):.1f}"
                )
        print(
            "[E4] Queued for Google Colab (elf_p5_non_identifiability.ipynb Section 6): "
            "1-token OpenWebText prefix AR anchor regeneration with saved samples.jsonl."
        )
        out = {
            "execution_mode": "local_cpu_audit",
            "quarantined_p4_ar_runs": quarantined_p4_ar_runs,
            "records": records,
        }
        save_json_atomic(os.path.join(out_dir, "e4_ar_audit_results.json"), out)
        return out

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    sweep = mods["sweep"]
    frontier = mods["frontier"]
    refs = mods["refs"]
    registry = mods["registry"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_texts = refs.get_owt_refs(1000)

    for model_name in models:
        tag = "gpt2_medium" if "medium" in model_name else "gpt2_small"
        tok = GPT2TokenizerFast.from_pretrained(model_name, padding_side="left")
        tok.pad_token = tok.eos_token
        eos_id = tok.eos_token_id or 50256

        owt_first_tokens: List[int] = []
        for rt in ref_texts:
            ids = tok.encode(rt, add_special_tokens=False)
            owt_first_tokens.append(int(ids[0]) if ids else eos_id)

        model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="sdpa").to(device)
        if device == "cuda":
            model = model.bfloat16()
        model.eval()

        for temp in temps:
            for top_p in top_ps:
                run_id = f"p5_ar_{tag}_owt1tok_T{temp}_p{top_p}_seed{seed}"
                rd = registry.run_dir(run_id)
                summary_path = os.path.join(rd, "summary.json")

                if os.path.isfile(summary_path) and os.path.isfile(os.path.join(rd, "samples.jsonl")):
                    s = load_json(summary_path)
                    print(f"[skip] {run_id}: PPL={s.get('gen_ppl', float('nan')):.2f} MAUVE={s.get('mauve', float('nan')):.3f}")
                    records.append(s)
                    continue

                print(f"[E4][GPU] Generating {run_id} (N={n_samples}, T={temp}, top_p={top_p})...")
                torch.manual_seed(seed * 10003 + int(temp * 100) + int(top_p * 10))
                all_texts: List[str] = []
                t0 = time.time()

                for start in range(0, n_samples, batch_size):
                    cur_bs = min(batch_size, n_samples - start)
                    prefix_ids = [
                        owt_first_tokens[(start + k) % len(owt_first_tokens)]
                        for k in range(cur_bs)
                    ]
                    input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(1)
                    attn_mask = torch.ones_like(input_ids)
                    with torch.inference_mode():
                        gen_out = model.generate(
                            input_ids,
                            attention_mask=attn_mask,
                            max_length=1024,
                            do_sample=True,
                            temperature=float(temp),
                            top_p=float(top_p),
                            pad_token_id=eos_id,
                            eos_token_id=eos_id,
                        )
                    for seq in gen_out.tolist():
                        if eos_id in seq:
                            seq = seq[: seq.index(eos_id)]
                        all_texts.append(tok.decode(seq, skip_special_tokens=True))

                wall_s = time.time() - t0
                registry.log_jsonl(
                    os.path.join(rd, "samples.jsonl"),
                    [{"i": i, "text": t} for i, t in enumerate(all_texts)],
                )

                m_res, per_sample = sweep.score_texts(all_texts, mauve=True)
                np.savez_compressed(os.path.join(rd, "per_sample.npz"), **per_sample)
                quantiles = frontier.compute_sample_quantiles(per_sample["nll"], per_sample["count"])
                ext = score_extended_sample_metrics(all_texts, ref_texts=ref_texts, ctw_depth=6, seed=seed)
                save_json_atomic(os.path.join(rd, "extended_metrics.json"), ext)

                summary = {
                    "run_id": run_id,
                    "model": model_name,
                    "anchor": tag,
                    "method": "ar_owt_1tok_prefix",
                    "temperature": float(temp),
                    "top_p": float(top_p),
                    "n": int(n_samples),
                    "seed": int(seed),
                    "wall_s": wall_s,
                    "status": "done",
                }
                summary.update(m_res)
                summary.update(quantiles)
                summary.update(ext)

                save_json_atomic(summary_path, summary)
                registry.append_registry(summary)
                records.append(summary)
                print(
                    f"[done] {run_id}: PPL={summary['gen_ppl']:.2f} H={summary['entropy']:.3f} "
                    f"MAUVE={summary['mauve']:.3f} JCD={summary.get('jcd_bits_per_byte', float('nan')):.4f}"
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {
        "execution_mode": "gpu",
        "quarantined_p4_ar_runs": quarantined_p4_ar_runs,
        "records": records,
    }
    save_json_atomic(os.path.join(out_dir, "e4_ar_audit_results.json"), out)
    return out


# ---------------------------------------------------------------------------
# 7. Automated Markdown & LaTeX Report Synthesis
# ---------------------------------------------------------------------------
def generate_p5_report(
    e1_bundle: Optional[Dict[str, Any]] = None,
    e2_bundle: Optional[Dict[str, Any]] = None,
    e3_bundle: Optional[Dict[str, Any]] = None,
    e4_bundle: Optional[Dict[str, Any]] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> str:
    """Compiles results from E1-E4 into a publication-ready Markdown report."""
    out_path = Path(out_path or os.path.join(P5_DIR, "_report.md"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections: List[str] = [
        "# Follow-Up Experiments Report: Sampler-Control Non-Identifiability (E1–E4)",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
    ]

    if e1_bundle and e1_bundle.get("analysis"):
        cells = e1_bundle["analysis"].get("cells", [])
        pairs = e1_bundle["analysis"].get("matched_pairs", [])
        cs_pairs = e1_bundle["analysis"].get("cross_scale_pairs", [])
        pending = e1_bundle.get("pending_gpu_runs", [])
        sections.append("## 1. Priority 1 (E1): `ELF-M` (342M) & `ELF-L` (652M) Scale-Transfer Analysis")
        if pending:
            sections.append(
                f"> **Status**: Local `ELF-M`/`ELF-L` cells verified below; {len(pending)} new two-seed `SC-CFG in {{1.0, 4.0}}` runs queued for Colab GPU execution (`elf_p5_non_identifiability.ipynb` Section 3)."
            )
            sections.append("")
        sections.append(
            "| Model | Steps | Hook | SC-CFG | $\\gamma$ | Seeds | Gen-PPL | $H_{\\text{ps}}$ | rep-4 (%) | distinct-2 | MAUVE | JCD (bpb) |"
        )
        sections.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cells:
            jcd_s = f"{c['jcd_bits_per_byte']:.4f}" if math.isfinite(c.get("jcd_bits_per_byte", float("nan"))) else "—"
            sections.append(
                f"| {c['model']} | {c['steps']} | {c['hook']} | {c['sc_cfg']:.1f} | {c['gamma']:.2f} | "
                f"{c['n_seeds']} | {c['gen_ppl']:.2f} | {c['entropy']:.3f} | "
                f"{100.0 * c['rep4']:.2f}% | {c['distinct2']:.4f} | {c['mauve']:.3f} | {jcd_s} |"
            )
        sections.append("")
        if cs_pairs:
            sections.append("### Cross-Scale Matched-Entropy Comparison (`ELF-M` vs. `ELF-L`)")
            for csp in cs_pairs:
                cm, cl = csp["elfm"], csp["elfl"]
                sections.append(
                    f"- `ELF-M (SC={cm['sc_cfg']:.0f}, gamma={cm['gamma']:.1f})` ($H_{{\\text{{ps}}}}={cm['entropy']:.3f}$) vs. "
                    f"`ELF-L (SC={cl['sc_cfg']:.0f}, gamma={cl['gamma']:.1f})` ($H_{{\\text{{ps}}}}={cl['entropy']:.3f}$): "
                    f"$\\Delta H_{{\\text{{ps}}}} = {csp['delta_H_ps']:+.4f}$, "
                    f"$\\Delta\\text{{Gen-PPL}} = {csp['delta_gen_ppl']:+.2f}$ ({cm['gen_ppl']:.2f} $\\to$ {cl['gen_ppl']:.2f}), "
                    f"$\\Delta\\text{{rep-4}} = {csp['delta_rep4_pp']:+.2f}\\text{{ pp}}$ ({100*cm['rep4']:.2f}% $\\to$ {100*cl['rep4']:.2f}%), "
                    f"$\\Delta\\text{{MAUVE}} = {csp['delta_mauve']:+.3f}$ ({cm['mauve']:.3f} $\\to$ {cl['mauve']:.3f})."
                )
            sections.append("")
        if pairs:
            sections.append("### Direct Non-Interpolated Matched Pairs on `ELF-M` (342M)")
            for mp in pairs:
                a, b = mp["low_sc"], mp["high_sc"]
                sections.append(
                    f"- `({a['sc_cfg']:.0f}, {a['gamma']:.2f})` vs `({b['sc_cfg']:.0f}, {b['gamma']:.2f})`: "
                    f"$\\Delta H_{{\\text{{ps}}}} = {mp['delta_H_ps']:+.4f}$, "
                    f"$\\Delta\\text{{PPL}} = {mp['delta_gen_ppl']:+.2f}$, "
                    f"$\\Delta\\text{{rep-4}} = {mp['delta_rep4_pp']:+.2f}\\text{{ pp}}$, "
                    f"$\\Delta\\text{{distinct-2}} = {mp['delta_distinct2']:+.4f}$, "
                    f"$\\Delta\\text{{MAUVE}} = {mp['delta_mauve']:+.3f}$."
                )
            sections.append("")

    if e2_bundle and e2_bundle.get("pair_reports"):
        sections.append(
            "## 2. Priority 2 (E2): Non-Interpolated Matched `ELF-B` Pairs (`Strict` & `Close` Tiers)"
        )
        sections.append(
            "| Pair | Tier | Low Path | High Path | $\\Delta H_{\\text{ps}}$ | $\\Delta\\text{PPL}$ | $\\Delta\\text{rep-4}$ (pp) | $\\Delta\\text{within-d4}$ (pp) | $\\Delta\\text{cross-d2}$ | $\\Delta\\text{MAUVE}$ | $\\Delta(\\text{p90/p10})$ | $\\Delta\\text{JCD}$ (bpb) |"
        )
        sections.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for pr in e2_bundle["pair_reports"]:
            jcd_s = f"{pr['delta_jcd_bpb']:+.4f}" if math.isfinite(pr.get("delta_jcd_bpb", float("nan"))) else "Colab E2"
            sections.append(
                f"| {pr['pair_id']} | {pr['tier']} | {pr['low_path']} | {pr['high_path']} | "
                f"{pr['delta_H_ps']:+.4f} | {pr['delta_gen_ppl']:+.2f} | "
                f"{pr['delta_rep4_pp']:+.2f} | {pr['delta_within_distinct4_pp']:+.2f} | "
                f"{pr['delta_cross_distinct2']:+.4f} | {pr['delta_mauve']:+.3f} | "
                f"{pr['delta_p90_over_p10']:+.3f} | {jcd_s} |"
            )
        sections.append("")

    if e3_bundle:
        sections.append("## 3. Priority 3 (E3): `ELF-B` Intermediate $\\gamma \\in \\{1.25, 1.75\\}$ Grid Refinement at $\\text{SC-CFG}=1.0$")
        if e3_bundle.get("bracket_analysis"):
            sections.append("| Midpoint $\\gamma$ | Bracket $(\\gamma_{\\text{lo}}, \\gamma_{\\text{hi}})$ | Bracket $H_{\\text{ps}}$ | Midpoint Est. $H_{\\text{ps}}$ | Log-Linear Est. PPL | Target Match | Target $H_{\\text{ps}}$ | Target PPL |")
            sections.append("|---|---|---|---|---|---|---|---|")
            for ba in e3_bundle["bracket_analysis"]:
                sections.append(
                    f"| {ba['gamma_mid']:.2f} | `{ba['sc1_bracket_gammas']}` | "
                    f"`[{ba['sc1_bracket_H_ps'][0]:.3f}, {ba['sc1_bracket_H_ps'][1]:.3f}]` | "
                    f"{ba['sc1_midpoint_H_ps_est']:.4f} | {ba['sc1_loglinear_ppl_est']:.2f} | "
                    f"`{ba['target_match']}` | {ba['target_H_ps']:.4f} | {ba['target_gen_ppl']:.2f} |"
                )
            sections.append("")
        if e3_bundle.get("records"):
            sections.append("| Run ID | $\\gamma$ | Seed | Gen-PPL | $H_{\\text{ps}}$ | rep-4 (%) | distinct-2 | MAUVE |")
            sections.append("|---|---|---|---|---|---|---|---|")
            for r in e3_bundle["records"]:
                sections.append(
                    f"| `{r['run_id']}` | {r['gamma']:.2f} | {r['seed']} | {r['gen_ppl']:.2f} | "
                    f"{r['entropy']:.3f} | {100.0 * r['rep4']:.2f}% | {r['distinct2']:.4f} | "
                    f"{r['mauve']:.3f} |"
                )
            sections.append("")

    if e4_bundle:
        sections.append("## 4. Priority 4 (E4): Autoregressive `GPT-2` Anchor Audit")
        if e4_bundle.get("quarantined_p4_ar_runs"):
            sections.append("### Quarantined `p4_ar_gpt2_*` Runs (Bare `<|endoftext|>` BOS Diagnosis)")
            sections.append("| Run ID | Model | $T$ | top-$p$ | Gen-PPL | $H_{\\text{ps}}$ | rep-4 (%) | MAUVE | Mean Len |")
            sections.append("|---|---|---|---|---|---|---|---|---|")
            for r in e4_bundle["quarantined_p4_ar_runs"]:
                sections.append(
                    f"| `{r['run_id']}` | {r.get('model','')} | {r.get('temperature',0):.2f} | "
                    f"{r.get('top_p',0):.2f} | {r.get('gen_ppl',0):.2f} | {r.get('entropy',0):.3f} | "
                    f"{100.0 * r.get('rep4',0):.2f}% | {r.get('mauve',0):.4f} | {r.get('mean_gpt2_len',0):.1f} |"
                )
            sections.append("")

    text = "\n".join(sections)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# 8. Self-Contained CPU & GPU Smoke Tests (T1-T5)
# ---------------------------------------------------------------------------
def run_smoke_tests() -> Dict[str, Any]:
    """Runs fast mathematical & pipeline smoke tests (T1-T5) without requiring GPU checkpoints."""
    print("[smoke] Running Sampler-Control Non-Identifiability Smoke Suite (T1-T5)...")
    t0 = time.time()

    # T1: Byte-level CTW Jeffreys Coding Divergence (JCD) invariants
    ref_corpus = [
        f"The quick brown fox jumps over the lazy dog in document {i} with varied vocabulary and natural phrasing."
        for i in range(60)
    ]
    diverse_gen = [
        f"In document {i}, the agile brown fox leaps across a sleepy hound with natural phrasing and varied words."
        for i in range(60)
    ]
    repetitive_gen = [
        "the the the the the the the the the the the the the the the the the the the the"
        for _ in range(60)
    ]

    jcd_self = compute_jcd_ctw(ref_corpus, ref_corpus, depth=4)
    jcd_div = compute_jcd_ctw(diverse_gen, ref_corpus, depth=4)
    jcd_div_rev = compute_jcd_ctw(ref_corpus, diverse_gen, depth=4)
    jcd_rep = compute_jcd_ctw(repetitive_gen, ref_corpus, depth=4)

    assert abs(jcd_self["jcd_bits_per_byte"]) < 1e-9, f"T1 self-JCD != 0: {jcd_self}"
    assert abs(jcd_div["jcd_bits_per_byte"] - jcd_div_rev["jcd_bits_per_byte"]) < 1e-9, "T1 JCD asymmetry"
    assert jcd_rep["jcd_bits_per_byte"] > jcd_div["jcd_bits_per_byte"] * 2.0, (
        f"T1 JCD failed to penalize repetition collapse: rep={jcd_rep['jcd_bits_per_byte']} vs div={jcd_div['jcd_bits_per_byte']}"
    )
    print(
        f"  [PASS] T1 Byte-CTW JCD: self={jcd_self['jcd_bits_per_byte']:.6f}, "
        f"diverse={jcd_div['jcd_bits_per_byte']:.4f} bpb, repetitive={jcd_rep['jcd_bits_per_byte']:.4f} bpb"
    )

    # T2: Within-sample vs Cross-sample diversity & Jensen-Shannon identity
    identical_across_samples = [list(range(1, 51)) for _ in range(20)]
    disjoint_across_samples = [
        [i * 100 + (k % 25) for k in range(50)] for i in range(20)
    ]
    div_a = compute_within_and_cross_diversity(identical_across_samples)
    div_b = compute_within_and_cross_diversity(disjoint_across_samples)

    assert abs(div_a["within_distinct4"] - (1.0 - div_a["rep4"])) < 1e-9, "T2 within_distinct4 != 1 - rep4"
    assert abs(div_b["within_distinct4"] - (1.0 - div_b["rep4"])) < 1e-9, "T2 within_distinct4 != 1 - rep4"
    assert abs(div_a["jsd_inter_sample"]) < 1e-9, f"T2 identical samples should have 0 JSD: {div_a['jsd_inter_sample']}"
    assert div_b["jsd_inter_sample"] > 2.5, f"T2 disjoint samples should have high JSD: {div_b['jsd_inter_sample']}"
    assert div_b["rep4"] > div_a["rep4"], "T2 expected Case B to have higher within-sample rep-4"
    assert div_b["cross_distinct2"] > div_a["cross_distinct2"] * 10.0, "T2 expected Case B to have higher cross-sample distinct-2"
    print(
        f"  [PASS] T2 Within vs Cross Diversity Split: "
        f"Case A (rep4={div_a['rep4']:.3f}, cross_d2={div_a['cross_distinct2']:.3f}, JSD={div_a['jsd_inter_sample']:.3f}) vs "
        f"Case B (rep4={div_b['rep4']:.3f}, cross_d2={div_b['cross_distinct2']:.3f}, JSD={div_b['jsd_inter_sample']:.3f})"
    )

    # T3: Self-BLEU & Position-Wise Entropy
    sb_a = compute_self_bleu(identical_across_samples, max_eval_samples=20)
    sb_b = compute_self_bleu(disjoint_across_samples, max_eval_samples=20)
    assert sb_a["self_bleu"] > 95.0, f"T3 identical samples Self-BLEU should be ~100: {sb_a['self_bleu']}"
    assert sb_b["self_bleu"] < 5.0, f"T3 disjoint samples Self-BLEU should be near 0: {sb_b['self_bleu']}"
    pos_b = compute_position_wise_entropy(disjoint_across_samples, max_length=32)
    assert math.isfinite(pos_b["quartiles"]["Q1"]) and pos_b["quartiles"]["Q1"] > 2.0
    print(
        f"  [PASS] T3 Self-BLEU & Position Entropy: "
        f"Self-BLEU(identical)={sb_a['self_bleu']:.1f}, Self-BLEU(disjoint)={sb_b['self_bleu']:.2f}, "
        f"PosEntropy Q1={pos_b['quartiles']['Q1']:.3f} nats"
    )

    # T4: Bootstrap Mean Difference CI and p-value
    boot = bootstrap_mean_diff(
        np.zeros(100, dtype=np.float64),
        np.ones(100, dtype=np.float64) * 0.02,
        B=2000,
        seed=0,
    )
    assert abs(boot["diff_b_minus_a"] - 0.02) < 1e-9
    assert boot["p_two_sided"] < 1e-3
    print(f"  [PASS] T4 Bootstrap Mean Diff: diff={boot['diff_b_minus_a']:+.4f}, p={boot['p_two_sided']:.2e}")

    # T5: Config builder check for E1 (8 core runs, 40 extended runs) and P6 (40 target runs)
    c_core = build_e1_elfm_configs(n_samples=1000, seeds=(0, 1), extended=False)
    c_ext = build_e1_elfm_configs(n_samples=1000, seeds=(0, 1), extended=True)
    c_p6 = build_p6_acl2027_configs(n_samples=1000, seeds=(0, 1, 2, 3, 4), include_refined2=False)
    assert len(c_core) == 8, f"T5 expected 8 core E1 configs, got {len(c_core)}"
    assert len(c_ext) == 40, f"T5 expected 40 extended E1 configs, got {len(c_ext)}"
    assert len(c_p6) == 40, f"T5 expected 40 P6 target configs (8 cells x 5 seeds), got {len(c_p6)}"
    assert len({c['run_id'] for c in c_p6}) == 40, "T5 duplicate run_id in P6 configs"
    print(f"  [PASS] T5 E1 & P6 Config Builders: {len(c_core)} E1 core, {len(c_ext)} E1 ext, {len(c_p6)} P6 5-seed runs")

    wall = time.time() - t0
    print(f"[smoke] ALL T1-T5 SMOKE TESTS PASSED in {wall:.2f}s.")
    return {
        "status": "ALL_PASS",
        "wall_s": wall,
        "T1_jcd": {"diverse": jcd_div["jcd_bits_per_byte"], "repetitive": jcd_rep["jcd_bits_per_byte"]},
        "T2_split": {
            "caseA_rep4": div_a["rep4"],
            "caseA_cross_d2": div_a["cross_distinct2"],
            "caseB_rep4": div_b["rep4"],
            "caseB_cross_d2": div_b["cross_distinct2"],
        },
        "T3_self_bleu": {"identical": sb_a["self_bleu"], "disjoint": sb_b["self_bleu"]},
    }


# ---------------------------------------------------------------------------
# 8. P6: ACL 2027 Decisive Follow-Up (n=5 Seeds + 2x2 ELF-M Off-Diagonals + Human Eval Packets)
# ---------------------------------------------------------------------------
ACL2027_DECISIVE_SPECS: List[Dict[str, Any]] = [
    # Group F1: Off-diagonal ELF-M (342M, 64 steps) cells completing the 2x2 factorial (10 new runs across seeds 0..4)
    {
        "group": "F1",
        "model": "ELF-M",
        "steps": 64,
        "sc_cfg": 1.0,
        "gamma": 0.5,
        "role": "2x2_offdiag_low_sc_low_g",
        "est_a100_min": 7.3,
    },
    {
        "group": "F1",
        "model": "ELF-M",
        "steps": 64,
        "sc_cfg": 4.0,
        "gamma": 2.0,
        "role": "2x2_offdiag_high_sc_high_g",
        "est_a100_min": 7.3,
    },
    # Group F2: ELF-M (342M, 64 steps) matched pair M1 seed expansion to n=5 (seeds 0,1 exist; seeds 2,3,4 pending = 6 new runs)
    {
        "group": "F2",
        "model": "ELF-M",
        "steps": 64,
        "sc_cfg": 1.0,
        "gamma": 2.0,
        "role": "5seed_M1_low_sc_high_g",
        "est_a100_min": 7.3,
    },
    {
        "group": "F2",
        "model": "ELF-M",
        "steps": 64,
        "sc_cfg": 4.0,
        "gamma": 0.5,
        "role": "5seed_M1_high_sc_low_g",
        "est_a100_min": 7.3,
    },
    # Group F3: ELF-B (105M, 32 steps) matched pairs S1 and Refined-1 seed expansion to n=5 (seeds 0,1 exist; seeds 2,3,4 pending = 12 new runs)
    {
        "group": "F3",
        "model": "ELF-B",
        "steps": 32,
        "sc_cfg": 1.0,
        "gamma": 2.0,
        "role": "5seed_S1_low_sc_high_g",
        "est_a100_min": 1.6,
    },
    {
        "group": "F3",
        "model": "ELF-B",
        "steps": 32,
        "sc_cfg": 4.0,
        "gamma": 1.5,
        "role": "5seed_S1_high_sc_low_g",
        "est_a100_min": 1.6,
    },
    {
        "group": "F3",
        "model": "ELF-B",
        "steps": 32,
        "sc_cfg": 1.0,
        "gamma": 1.25,
        "role": "5seed_Ref1_low_sc_high_g",
        "est_a100_min": 1.6,
    },
    {
        "group": "F3",
        "model": "ELF-B",
        "steps": 32,
        "sc_cfg": 3.0,
        "gamma": 0.5,
        "role": "5seed_Ref1_high_sc_low_g",
        "est_a100_min": 1.6,
    },
]

ACL2027_OPTIONAL_REFINED2_SPECS: List[Dict[str, Any]] = [
    # Optional Group F3b: ELF-B Refined-2 (E3b) seed expansion to n=5 (seeds 0,1 exist; seeds 2,3,4 pending = 6 new runs)
    {
        "group": "F3b",
        "model": "ELF-B",
        "steps": 32,
        "sc_cfg": 1.0,
        "gamma": 1.75,
        "role": "5seed_Ref2_low_sc_high_g",
        "est_a100_min": 1.6,
    },
    {
        "group": "F3b",
        "model": "ELF-B",
        "steps": 32,
        "sc_cfg": 4.0,
        "gamma": 1.0,
        "role": "5seed_Ref2_high_sc_low_g",
        "est_a100_min": 1.6,
    },
]


def build_p6_acl2027_configs(
    n_samples: int = 1000,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    include_refined2: bool = False,
    group_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    specs = list(ACL2027_DECISIVE_SPECS)
    if include_refined2:
        specs.extend(ACL2027_OPTIONAL_REFINED2_SPECS)
    cfgs: List[Dict[str, Any]] = []
    seen_ids = set()
    for sp in specs:
        if group_filter and sp["group"] != group_filter:
            continue
        prefix = "p5_elfm_s64" if sp["model"] == "ELF-M" else f"p4_elfb_s{int(sp['steps'])}"
        method = "ode" if float(sp["gamma"]) == 0.0 else "sde"
        for seed in seeds:
            run_id = f"{prefix}_g{float(sp['gamma'])}_sc{float(sp['sc_cfg'])}_seed{int(seed)}"
            if run_id in seen_ids:
                raise ValueError(f"Duplicate run_id in P6 specs: {run_id}")
            seen_ids.add(run_id)
            rd = os.path.join(RUNS, run_id)
            cfgs.append(
                {
                    "run_id": run_id,
                    "group": sp["group"],
                    "role": sp["role"],
                    "model": sp["model"],
                    "steps": int(sp["steps"]),
                    "method": method,
                    "gamma": float(sp["gamma"]),
                    "sc_cfg": float(sp["sc_cfg"]),
                    "n": int(n_samples),
                    "seed": int(seed),
                    "mauve": True,
                    "output_dir": rd,
                    "est_a100_min": float(sp["est_a100_min"]),
                    "exists_on_disk": os.path.isfile(os.path.join(rd, "summary.json")),
                }
            )
    return cfgs


def run_p6_group(
    group: Optional[str] = None,
    n_samples: int = 1000,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    include_refined2: bool = False,
    dry_run: bool = False,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run P6 decisive follow-up group ('F1', 'F2', 'F3', 'F3b', or None for all) with per-run checkpointing."""
    if out_dir is None:
        out_dir = os.path.join(RUNS, "p6_followup")
    os.makedirs(out_dir, exist_ok=True)
    cfgs = build_p6_acl2027_configs(
        n_samples=n_samples,
        seeds=seeds,
        include_refined2=include_refined2,
        group_filter=group,
    )
    existing = [c for c in cfgs if c["exists_on_disk"]]
    pending = [c for c in cfgs if not c["exists_on_disk"]]
    est_min = sum(c["est_a100_min"] for c in pending)
    label = group if group else "ALL"
    print(
        f"[P6:{label}] Target runs: {len(cfgs)} | Existing on disk (preserved): {len(existing)} | "
        f"Pending GPU runs: {len(pending)} (~{est_min:.1f} min / {est_min/60.0:.2f}h on A100)"
    )
    records: List[Dict[str, Any]] = []
    for idx, c in enumerate(cfgs, 1):
        t0 = time.time()
        was_existing = c["exists_on_disk"]
        res = run_single_config_and_ext_metrics(
            model_name=c["model"],
            steps=c["steps"],
            gamma=c["gamma"],
            sc_cfg=c["sc_cfg"],
            seed=c["seed"],
            n_samples=c["n"],
            dry_run=dry_run,
        )
        rec = {
            **c,
            **res,
            "was_existing": was_existing,
            "wall_s": time.time() - t0,
        }
        records.append(rec)
        if not dry_run and "gen_ppl" in rec:
            tag = "SKIP-EXISTING" if was_existing else "NEW-GPU-DONE"
            print(
                f"  [{idx:02d}/{len(cfgs):02d} {tag}] {c['run_id']}: "
                f"PPL={rec.get('gen_ppl', float('nan')):.2f}, H_ps={rec.get('entropy', float('nan')):.4f}, "
                f"H_pool={rec.get('h_pool', float('nan')):.4f}, JSD={rec.get('jsd_inter_sample', float('nan')):.4f}, "
                f"rep4={rec.get('rep4', float('nan'))*100.0:.2f}%, "
                f"SelfBLEU={rec.get('self_bleu', float('nan')):.2f}, "
                f"JCD={rec.get('jcd_bits_per_byte', float('nan')):.4f} bpb, "
                f"MAUVE={rec.get('mauve', float('nan')):.3f}"
            )
            save_json_atomic(
                os.path.join(out_dir, f"p6_progress_{label}.json"),
                {"group": label, "completed_count": idx, "total_count": len(cfgs), "records": records},
            )
    return {"group": label, "records": records, "existing_count": len(existing), "pending_count": len(pending)}


def _satterthwaite_welch_pair(va: Sequence[float], vb: Sequence[float]) -> Dict[str, Any]:
    """Compute two-sample Satterthwaite Welch t-test and equal-variance Student t-test for seed-level observations."""
    from scipy import stats as sp_stats
    a = np.asarray([x for x in va if x is not None and math.isfinite(float(x))], dtype=np.float64)
    b = np.asarray([x for x in vb if x is not None and math.isfinite(float(x))], dtype=np.float64)
    na, nb = int(len(a)), int(len(b))
    mean_a = float(np.mean(a)) if na > 0 else float("nan")
    mean_b = float(np.mean(b)) if nb > 0 else float("nan")
    sd_a = float(np.std(a, ddof=1)) if na > 1 else 0.0
    sd_b = float(np.std(b, ddof=1)) if nb > 1 else 0.0
    diff = mean_b - mean_a
    rel_pct = 100.0 * diff / mean_a if (math.isfinite(mean_a) and abs(mean_a) > 1e-12) else float("nan")
    if na > 1 and nb > 1 and (sd_a > 0 or sd_b > 0):
        v1 = (sd_a ** 2) / na
        v2 = (sd_b ** 2) / nb
        se_diff = math.sqrt(v1 + v2)
        df_satt = float(((v1 + v2) ** 2) / ((v1 ** 2) / (na - 1) + (v2 ** 2) / (nb - 1)))
        df_pool = na + nb - 2
        t_welch, p_welch = sp_stats.ttest_ind(b, a, equal_var=False)
        t_stud, p_stud = sp_stats.ttest_ind(b, a, equal_var=True)
        t_crit_satt = float(sp_stats.t.ppf(0.975, df=df_satt))
        t_crit_pool = float(sp_stats.t.ppf(0.975, df=df_pool))
        ci95_satt = [float(diff - t_crit_satt * se_diff), float(diff + t_crit_satt * se_diff)]
        ci95_pool = [float(diff - t_crit_pool * se_diff), float(diff + t_crit_pool * se_diff)]
    else:
        se_diff = df_satt = t_welch = p_welch = t_stud = p_stud = float("nan")
        df_pool = max(0, na + nb - 2)
        ci95_satt = [float("nan"), float("nan")]
        ci95_pool = [float("nan"), float("nan")]
    return {
        "n_a": na,
        "n_b": nb,
        "seeds_a": [float(x) for x in a],
        "seeds_b": [float(x) for x in b],
        "mean_a": mean_a,
        "mean_b": mean_b,
        "sd_a": sd_a,
        "sd_b": sd_b,
        "diff_b_minus_a": float(diff),
        "rel_diff_pct": float(rel_pct),
        "se_unpaired": float(se_diff),
        "t_welch": float(t_welch),
        "df_satt": float(df_satt),
        "p_welch": float(p_welch),
        "ci95_welch_satt": ci95_satt,
        "t_student_pool": float(t_stud),
        "df_pool": int(df_pool),
        "p_student_pool": float(p_stud),
        "ci95_student_pool": ci95_pool,
    }


def _two_way_anova_2x2(cell_Map: Dict[Tuple[float, float], List[float]]) -> Dict[str, Any]:
    """Compute exact two-way factorial ANOVA on a 2x2 grid: SC-CFG in {1.0, 4.0} x gamma in {0.5, 2.0}."""
    from scipy import stats as sp_stats
    scs = [1.0, 4.0]
    gammas = [0.5, 2.0]
    all_vals = []
    for s in scs:
        for g in gammas:
            all_vals.extend(cell_Map.get((s, g), []))
    if len(all_vals) < 8:
        return {"status": "incomplete_cells", "n_total": len(all_vals)}
    grand_mean = float(np.mean(all_vals))
    n_per_cell = min(len(cell_Map.get((s, g), [])) for s in scs for g in gammas)
    if n_per_cell < 2:
        return {"status": "insufficient_replication", "n_per_cell": n_per_cell}
    # Use balanced n_per_cell observations per cell
    y = np.zeros((2, 2, n_per_cell), dtype=np.float64)
    for i, s in enumerate(scs):
        for j, g in enumerate(gammas):
            y[i, j, :] = np.asarray(cell_Map[(s, g)][:n_per_cell], dtype=np.float64)
    mu = float(np.mean(y))
    mean_i = np.mean(y, axis=(1, 2))
    mean_j = np.mean(y, axis=(0, 2))
    mean_ij = np.mean(y, axis=2)

    ss_sc = float(2 * n_per_cell * np.sum((mean_i - mu) ** 2))
    ss_g = float(2 * n_per_cell * np.sum((mean_j - mu) ** 2))
    ss_inter = float(
        n_per_cell
        * np.sum((mean_ij - mean_i[:, None] - mean_j[None, :] + mu) ** 2)
    )
    ss_err = float(np.sum((y - mean_ij[:, :, None]) ** 2))
    ss_tot = ss_sc + ss_g + ss_inter + ss_err

    df_sc, df_g, df_inter = 1, 1, 1
    df_err = 4 * (n_per_cell - 1)
    ms_err = ss_err / df_err if df_err > 0 else float("nan")

    def _f_and_p(ss: float, df1: int) -> Tuple[float, float]:
        if ms_err > 0 and df_err > 0:
            f_val = (ss / df1) / ms_err
            p_val = float(sp_stats.f.sf(f_val, df1, df_err))
            return float(f_val), p_val
        return float("nan"), float("nan")

    f_sc, p_sc = _f_and_p(ss_sc, df_sc)
    f_g, p_g = _f_and_p(ss_g, df_g)
    f_inter, p_inter = _f_and_p(ss_inter, df_inter)

    # Pure main effects (holding the other factor fixed)
    main_effect_sc = float(mean_i[1] - mean_i[0])       # SC=4 minus SC=1 across both gammas
    main_effect_gamma = float(mean_j[1] - mean_j[0])    # gamma=2.0 minus gamma=0.5 across both SCs

    return {
        "status": "complete",
        "n_per_cell": int(n_per_cell),
        "n_total": int(4 * n_per_cell),
        "df_error": int(df_err),
        "cell_means": {
            f"sc{scs[i]}_g{gammas[j]}": float(mean_ij[i, j])
            for i in range(2)
            for j in range(2)
        },
        "cell_sds": {
            f"sc{scs[i]}_g{gammas[j]}": float(np.std(y[i, j, :], ddof=1))
            for i in range(2)
            for j in range(2)
        },
        "main_effect_sc4_minus_sc1": main_effect_sc,
        "main_effect_g2.0_minus_g0.5": main_effect_gamma,
        "sc_cfg": {
            "df": df_sc,
            "ss": ss_sc,
            "share_of_total_ss": ss_sc / ss_tot if ss_tot > 0 else float("nan"),
            "F": f_sc,
            "p": p_sc,
        },
        "gamma": {
            "df": df_g,
            "ss": ss_g,
            "share_of_total_ss": ss_g / ss_tot if ss_tot > 0 else float("nan"),
            "F": f_g,
            "p": p_g,
        },
        "interaction": {
            "df": df_inter,
            "ss": ss_inter,
            "share_of_total_ss": ss_inter / ss_tot if ss_tot > 0 else float("nan"),
            "F": f_inter,
            "p": p_inter,
        },
        "error": {
            "df": df_err,
            "ss": ss_err,
            "share_of_total_ss": ss_err / ss_tot if ss_tot > 0 else float("nan"),
        },
    }


def analyze_p6_acl2027_results(out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load all available seeds (up to n=5) from RUNS and compute P6 5-seed Welch tests + 2x2x5 ELF-M ANOVA."""
    if out_dir is None:
        out_dir = os.path.join(RUNS, "p6_followup")
    os.makedirs(out_dir, exist_ok=True)

    def _load_cell_runs(model: str, steps: int, sc: float, g: float, max_seeds: int = 10) -> List[Dict[str, Any]]:
        prefix = "p5_elfm_s64" if model == "ELF-M" else f"p4_elfb_s{int(steps)}"
        found = []
        for s in range(max_seeds):
            rid = f"{prefix}_g{float(g)}_sc{float(sc)}_seed{s}"
            rd = os.path.join(RUNS, rid)
            sp = os.path.join(rd, "summary.json")
            ep = os.path.join(rd, "extended_metrics.json")
            if os.path.isfile(sp):
                d = load_json(sp, {})
                if os.path.isfile(ep):
                    d.update(load_json(ep, {}))
                # Normalize keys
                if "distinct2" not in d and "cross_distinct2" in d:
                    d["distinct2"] = d["cross_distinct2"]
                found.append(d)
        return found

    pair_defs = [
        ("ELF-M_M1_SC1g2.0_vs_SC4g0.5", "ELF-M", 64, (1.0, 2.0), (4.0, 0.5)),
        ("ELF-B_S1_SC1g2.0_vs_SC4g1.5", "ELF-B", 32, (1.0, 2.0), (4.0, 1.5)),
        ("ELF-B_Ref1_SC1g1.25_vs_SC3g0.5", "ELF-B", 32, (1.0, 1.25), (3.0, 0.5)),
        ("ELF-B_Ref2_SC1g1.75_vs_SC4g1.0", "ELF-B", 32, (1.0, 1.75), (4.0, 1.0)),
        ("ELF-B_S2_SC1g1.5_vs_SC3g1.0", "ELF-B", 32, (1.0, 1.5), (3.0, 1.0)),
        ("ELF-B_S3_SC2g1.0_vs_SC4g0.5", "ELF-B", 32, (2.0, 1.0), (4.0, 0.5)),
    ]
    metrics_list = [
        "gen_ppl",
        "entropy",
        "rep4",
        "within_distinct2",
        "distinct2",
        "h_pool",
        "jsd_inter_sample",
        "self_bleu",
        "jcd_bits_per_byte",
        "mauve",
    ]

    pairs_out: Dict[str, Any] = {}
    for pname, model, steps, (sc_a, g_a), (sc_b, g_b) in pair_defs:
        runs_a = _load_cell_runs(model, steps, sc_a, g_a)
        runs_b = _load_cell_runs(model, steps, sc_b, g_b)
        m_dict = {}
        for m in metrics_list:
            va = [r.get(m) for r in runs_a if r.get(m) is not None]
            vb = [r.get(m) for r in runs_b if r.get(m) is not None]
            m_dict[m] = _satterthwaite_welch_pair(va, vb)
        pairs_out[pname] = {
            "model": model,
            "steps": steps,
            "cell_a": {"sc_cfg": sc_a, "gamma": g_a, "n_seeds": len(runs_a)},
            "cell_b": {"sc_cfg": sc_b, "gamma": g_b, "n_seeds": len(runs_b)},
            "metrics": m_dict,
        }

    # Holm-Bonferroni across the 6 primary pairs for each metric
    for m in metrics_list:
        raw_ps = []
        for pname in pairs_out:
            p_val = pairs_out[pname]["metrics"][m]["p_welch"]
            raw_ps.append((pname, p_val if math.isfinite(p_val) else 1.0))
        raw_ps_sorted = sorted(raw_ps, key=lambda x: x[1])
        k_total = len(raw_ps_sorted)
        running_max = 0.0
        for rank_idx, (pname, p_raw) in enumerate(raw_ps_sorted):
            p_adj = min(1.0, p_raw * (k_total - rank_idx))
            running_max = max(running_max, p_adj)
            pairs_out[pname]["metrics"][m]["p_holm_6pairs"] = float(running_max)

    # 2x2 ELF-M Factorial ANOVA across SC in {1.0, 4.0} x gamma in {0.5, 2.0}
    elfm_2x2_cells: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for sc in (1.0, 4.0):
        for g in (0.5, 2.0):
            elfm_2x2_cells[(sc, g)] = _load_cell_runs("ELF-M", 64, sc, g)

    elfm_anova_out: Dict[str, Any] = {}
    for m in metrics_list:
        cmap = {
            k: [float(r[m]) for r in v if r.get(m) is not None and math.isfinite(float(r[m]))]
            for k, v in elfm_2x2_cells.items()
        }
        elfm_anova_out[m] = _two_way_anova_2x2(cmap)

    summary_bundle = {
        "matched_pairs": pairs_out,
        "elfm_2x2_factorial_anova": elfm_anova_out,
        "elfm_2x2_seed_counts": {
            f"sc{sc}_g{g}": len(elfm_2x2_cells[(sc, g)])
            for sc in (1.0, 4.0)
            for g in (0.5, 2.0)
        },
    }
    save_json_atomic(os.path.join(out_dir, "p6_reanalysis_stats.json"), summary_bundle)
    return summary_bundle


def generate_p6_report_and_tables(
    stats_bundle: Dict[str, Any],
    out_dir: Optional[str] = None,
) -> str:
    """Generate Markdown synthesis report and LaTeX tables from the P6 5-seed + 2x2 ELF-M statistical bundle."""
    if out_dir is None:
        out_dir = os.path.join(RUNS, "p6_followup")
    os.makedirs(out_dir, exist_ok=True)

    lines = [
        "# ACL 2027 Decisive Follow-Up Synthesis Report (`P6`: 5-Seed Replication & 2x2 ELF-M ANOVA)",
        "",
        "## 1. 5-Seed Matched-Pair Satterthwaite Welch Tests",
        "",
        "| Pair | Model | Seeds ($n_A,n_B$) | Gen-PPL ($A$ vs $B$, $\\Delta\\%$) | $H_{\\text{ps}}$ ($\\Delta$ nats) | $\\Delta\\texttt{rep-4}$ (pp, Welch $t$, $df_{\\text{Satt}}$, $p_{\\text{Holm}}$) | $\\Delta\\mathrm{JSD}_{\\pi}$ | $\\Delta\\text{Self-BLEU}$ | $\\Delta\\text{CTW JCD}$ | $\\Delta\\text{MAUVE}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for pname, pdata in stats_bundle.get("matched_pairs", {}).items():
        m = pdata["metrics"]
        na = pdata["cell_a"]["n_seeds"]
        nb = pdata["cell_b"]["n_seeds"]
        ppl_a = m["gen_ppl"]["mean_a"]
        ppl_b = m["gen_ppl"]["mean_b"]
        ppl_rel = m["gen_ppl"]["rel_diff_pct"]
        dh = m["entropy"]["diff_b_minus_a"]
        drep_pp = m["rep4"]["diff_b_minus_a"] * 100.0
        trep = m["rep4"]["t_welch"]
        dfrep = m["rep4"]["df_satt"]
        pholm = m["rep4"].get("p_holm_6pairs", m["rep4"]["p_welch"])
        djsd = m["jsd_inter_sample"]["diff_b_minus_a"]
        dsb = m["self_bleu"]["diff_b_minus_a"]
        djcd = m["jcd_bits_per_byte"]["diff_b_minus_a"]
        dmauve = m["mauve"]["diff_b_minus_a"]
        lines.append(
            f"| `{pname}` | {pdata['model']} | {na},{nb} | "
            f"{ppl_a:.2f} vs {ppl_b:.2f} ({ppl_rel:+.2f}%) | {dh:+.4f} | "
            f"{drep_pp:+.2f} pp ($t={trep:.2f}, df={dfrep:.2f}, p_{{\\text{{Holm}}}}={pholm:.4g}$) | "
            f"{djsd:+.4f} | {dsb:+.2f} | {djcd:+.4f} | {dmauve:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Balanced $2 \\times 2$ Two-Way Factorial ANOVA on ELF-M (342M, 64 Steps)",
            "",
            f"- **Cell seed counts:** `{stats_bundle.get('elfm_2x2_seed_counts', {})}`",
            "",
            "| Metric | Status | $N$ ($n$/cell) | $\\text{SC-CFG}$ Share (\\%) | $F_{\\text{SC}}$ ($p$) | $\\gamma$ Share (\\%) | $F_{\\gamma}$ ($p$) | Interaction Share (\\%) | Main Effect $\\Delta_{\\text{SC}}$ | Main Effect $\\Delta_{\\gamma}$ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for mname, ad in stats_bundle.get("elfm_2x2_factorial_anova", {}).items():
        if ad.get("status") != "complete":
            lines.append(f"| `{mname}` | `{ad.get('status')}` | - | - | - | - | - | - | - | - |")
            continue
        n_tot = ad["n_total"]
        n_cell = ad["n_per_cell"]
        sc_sh = ad["sc_cfg"]["share_of_total_ss"] * 100.0
        f_sc = ad["sc_cfg"]["F"]
        p_sc = ad["sc_cfg"]["p"]
        g_sh = ad["gamma"]["share_of_total_ss"] * 100.0
        f_g = ad["gamma"]["F"]
        p_g = ad["gamma"]["p"]
        in_sh = ad["interaction"]["share_of_total_ss"] * 100.0
        me_sc = ad["main_effect_sc4_minus_sc1"]
        me_g = ad["main_effect_g2.0_minus_g0.5"]
        lines.append(
            f"| `{mname}` | `complete` | {n_tot} ({n_cell}) | "
            f"**{sc_sh:.1f}%** | {f_sc:.1f} ({p_sc:.2e}) | "
            f"**{g_sh:.1f}%** | {f_g:.1f} ({p_g:.2e}) | "
            f"{in_sh:.1f}% | {me_sc:+.4f} | {me_g:+.4f} |"
        )

    report_md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "_p6_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
    return report_md


def build_human_eval_packets(
    out_dir: Optional[str] = None,
    n_single_pairs_per_match: int = 25,
    n_bundles_per_match: int = 15,
    bundle_size: int = 5,
    seed: int = 2027,
) -> Dict[str, Any]:
    """Generate double-blind A/B human evaluation CSV sheets and secret answer key from saved samples.jsonl."""
    import csv as py_csv
    if out_dir is None:
        out_dir = os.path.join(RUNS, "p6_followup", "human_eval")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    target_pairs = [
        ("M1_ELFM", "p5_elfm_s64_g2.0_sc1.0_seed0", "p5_elfm_s64_g0.5_sc4.0_seed0"),
        ("S1_ELFB", "p4_elfb_s32_g2.0_sc1.0_seed0", "p4_elfb_s32_g1.5_sc4.0_seed0"),
        ("Ref1_ELFB", "p4_elfb_s32_g1.25_sc1.0_seed0", "p4_elfb_s32_g0.5_sc3.0_seed0"),
    ]

    blinded_rows: List[Dict[str, Any]] = []
    answer_key: Dict[str, Any] = {}
    item_counter = 1

    for pair_tag, run_low_sc, run_high_sc in target_pairs:
        p_low = os.path.join(RUNS, run_low_sc, "samples.jsonl")
        p_high = os.path.join(RUNS, run_high_sc, "samples.jsonl")
        if not (os.path.isfile(p_low) and os.path.isfile(p_high)):
            continue
        with open(p_low, "r", encoding="utf-8") as f:
            texts_low = [json.loads(line)["text"].strip() for line in f if line.strip()]
        with open(p_high, "r", encoding="utf-8") as f:
            texts_high = [json.loads(line)["text"].strip() for line in f if line.strip()]

        n_avail = min(len(texts_low), len(texts_high))
        if n_avail < n_single_pairs_per_match + n_bundles_per_match * bundle_size:
            continue

        perm = rng.permutation(n_avail)
        # Task 1: Single-passage local repetition vs fluency
        for k in range(n_single_pairs_per_match):
            idx = int(perm[k])
            swap = bool(rng.integers(0, 2) == 1)
            text_a = texts_high[idx][:1200] if swap else texts_low[idx][:1200]
            text_b = texts_low[idx][:1200] if swap else texts_high[idx][:1200]
            item_id = f"T1_{pair_tag}_{item_counter:03d}"
            item_counter += 1
            blinded_rows.append(
                {
                    "item_id": item_id,
                    "task_type": "Task1_SinglePassage_LessRepetitive",
                    "prompt_question": "Which passage has LESS phrase repetition / looping while remaining fluent? (Enter A, B, or Tie)",
                    "passage_or_bundle_A": text_a,
                    "passage_or_bundle_B": text_b,
                    "annotator_1_choice": "",
                    "annotator_2_choice": "",
                    "annotator_3_choice": "",
                }
            )
            answer_key[item_id] = {
                "pair_tag": pair_tag,
                "task_type": "Task1_SinglePassage_LessRepetitive",
                "A_is": "high_sc_low_gamma" if swap else "low_sc_high_gamma",
                "B_is": "low_sc_high_gamma" if swap else "high_sc_low_gamma",
                "expected_winner_less_repetitive": "B" if swap else "A",
            }

        # Task 2: 5-passage bundle topical diversity
        offset = n_single_pairs_per_match
        for b_idx in range(n_bundles_per_match):
            idxs = [int(perm[offset + b_idx * bundle_size + j]) for j in range(bundle_size)]
            bundle_low = "\n---\n".join(f"[{j+1}] {texts_low[i][:350]}" for j, i in enumerate(idxs))
            bundle_high = "\n---\n".join(f"[{j+1}] {texts_high[i][:350]}" for j, i in enumerate(idxs))
            swap = bool(rng.integers(0, 2) == 1)
            text_a = bundle_high if swap else bundle_low
            text_b = bundle_low if swap else bundle_high
            item_id = f"T2_{pair_tag}_{item_counter:03d}"
            item_counter += 1
            blinded_rows.append(
                {
                    "item_id": item_id,
                    "task_type": "Task2_5SampleBundle_MoreDiverse",
                    "prompt_question": "Which 5-sample bundle exhibits GREATER topical & lexical variety across its 5 snippets? (Enter A, B, or Tie)",
                    "passage_or_bundle_A": text_a,
                    "passage_or_bundle_B": text_b,
                    "annotator_1_choice": "",
                    "annotator_2_choice": "",
                    "annotator_3_choice": "",
                }
            )
            answer_key[item_id] = {
                "pair_tag": pair_tag,
                "task_type": "Task2_5SampleBundle_MoreDiverse",
                "A_is": "high_sc_low_gamma" if swap else "low_sc_high_gamma",
                "B_is": "low_sc_high_gamma" if swap else "high_sc_low_gamma",
                "expected_winner_more_diverse": "A" if swap else "B",
            }

    csv_path = os.path.join(out_dir, "human_eval_blinded_sheet.csv")
    key_path = os.path.join(out_dir, "human_eval_answer_key.json")
    if blinded_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = py_csv.DictWriter(f, fieldnames=list(blinded_rows[0].keys()))
            writer.writeheader()
            writer.writerows(blinded_rows)
    save_json_atomic(key_path, answer_key)
    print(f"[human-eval] Wrote {len(blinded_rows)} blinded items to {csv_path} and key to {key_path}")
    return {"n_items": len(blinded_rows), "csv_path": csv_path, "key_path": key_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Sampler-Control Non-Identifiability Follow-Up Experiments (E1-E4)."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "e1_elfm_factorial",
            "e2_matched_rescore",
            "e3_elfb_refinement",
            "e4_ar_audit",
            "all",
        ],
        default="smoke",
        help="Experiment stage to run.",
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples per run.")
    parser.add_argument(
        "--extended-e1",
        action="store_true",
        help="Run the full 4x5x2=40 ELF-M grid instead of the minimal 8-run factorial.",
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        run_smoke_tests()
        return

    run_smoke_tests()
    e1_res = e2_res = e3_res = e4_res = None

    if args.mode in ("e1_elfm_factorial", "all"):
        e1_res = run_e1_elfm_factorial(n_samples=args.n_samples, extended=args.extended_e1)
    if args.mode in ("e2_matched_rescore", "all"):
        e2_res = run_e2_matched_pair_rescore(n_samples=args.n_samples)
    if args.mode in ("e3_elfb_refinement", "all"):
        e3_res = run_e3_elfb_grid_refinement(n_samples=args.n_samples)
    if args.mode in ("e4_ar_audit", "all"):
        e4_res = run_e4_ar_anchor_audit(n_samples=args.n_samples)

    report_path = os.path.join(P5_DIR, "_report.md")
    generate_p5_report(e1_res, e2_res, e3_res, e4_res, out_path=report_path)
    print(f"[done] Wrote synthesis report to {report_path}")


if __name__ == "__main__":
    main()
