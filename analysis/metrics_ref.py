"""Reference implementations of the evaluation quantities used in the paper.

All functions are pure numpy and are unit-tested in tests/test_ppl_estimators.py.

Notation. Sample m has valid (pre-EOS, non-pad) token count T_m and summed negative
log-likelihood S_m under the evaluator LM; L_m = S_m / T_m is its per-token loss.

  mean_seq_ppl     = (1/M) sum_m exp(L_m)              (legacy "mean of sample PPLs")
  exp_mean_seq_ppl = exp((1/M) sum_m L_m)              (sequence-weighted)
  corpus_ppl       = exp(sum_m S_m / sum_m T_m)        (token-weighted; used by the ELF harness)

Jensen (exp convex):    mean_seq_ppl >= exp_mean_seq_ppl, equality iff all L_m equal.
corpus_ppl vs exp_mean_seq_ppl: corpus_ppl = exp(sum_m w_m L_m), w_m = T_m / sum T, so the two
differ only through length weighting; either can be larger. corpus_ppl <= mean_seq_ppl is NOT
guaranteed in general (it holds when lengths are equal, by Jensen).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


class ExtrapolationError(ValueError):
    """Raised when PPL@H would require extrapolating beyond the observed entropy range."""


# --------------------------------------------------------------------------- masking

def valid_mask(token_ids: Sequence[int], eos_id: int, pad_id: int, count_eos: bool = True) -> np.ndarray:
    """Boolean mask of positions scored: everything up to (and optionally including) the first EOS,
    excluding pad. Positions after the first EOS are never scored."""
    ids = np.asarray(token_ids)
    mask = ids != pad_id
    eos_pos = np.where(ids == eos_id)[0]
    if eos_pos.size:
        first = int(eos_pos[0])
        mask[first + (1 if count_eos else 0):] = False
    return mask


def sequence_stats(token_nll: Sequence[float], mask: Sequence[bool]) -> tuple[float, int]:
    """Return (summed NLL S_m, valid token count T_m) over masked positions."""
    nll = np.asarray(token_nll, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    return float(nll[m].sum()), int(m.sum())


# --------------------------------------------------------------------------- PPL estimators

def _check(S: Sequence[float], T: Sequence[int]):
    S = np.asarray(S, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if S.shape != T.shape or S.ndim != 1 or S.size == 0:
        raise ValueError("S and T must be non-empty 1-D arrays of equal length")
    if np.any(T <= 0):
        raise ValueError("every sample must have at least one valid token (drop empty samples explicitly)")
    return S, T


def mean_seq_ppl(S, T) -> float:
    S, T = _check(S, T)
    return float(np.mean(np.exp(S / T)))


def exp_mean_seq_ppl(S, T) -> float:
    S, T = _check(S, T)
    return float(np.exp(np.mean(S / T)))


def corpus_ppl(S, T) -> float:
    S, T = _check(S, T)
    return float(np.exp(S.sum() / T.sum()))


# --------------------------------------------------------------------------- diversity

def per_sample_unigram_entropy(ids_list: Iterable[Sequence[int]]) -> float:
    """ELF harness definition: entropy (nats) of each sample's own unigram distribution,
    averaged over non-empty samples. Bounded above by log(len(sample))."""
    vals = []
    for ids in ids_list:
        ids = np.asarray(ids)
        if ids.size == 0:
            continue
        _, c = np.unique(ids, return_counts=True)
        p = c / c.sum()
        vals.append(float(-(p * np.log(p)).sum()))
    return float(np.mean(vals)) if vals else float("nan")


def pooled_unigram_entropy(ids_list: Iterable[Sequence[int]]) -> float:
    """Corpus definition: entropy (nats) of the unigram distribution pooled over all samples.
    Not comparable to per_sample_unigram_entropy (it is typically much larger)."""
    allids = [i for ids in ids_list for i in ids]
    if not allids:
        return float("nan")
    _, c = np.unique(np.asarray(allids), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def rep_n(ids: Sequence[int], n: int = 4) -> float:
    """Fraction of n-grams in a sample that are repeats: 1 - unique/total."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return float("nan")
    return 1.0 - len(set(grams)) / len(grams)


# --------------------------------------------------------------------------- frontier

def frontier_ppl_at(points: Sequence[tuple[float, float]], h_target: float) -> float:
    """PPL at entropy h_target by linear interpolation of log(PPL) in H between the two
    observed operating points that bracket h_target. points = [(H, PPL), ...].

    Raises ExtrapolationError if h_target lies outside [min H, max H].
    If several bracketing pairs exist (non-monotone frontier) the pair whose H values are
    closest to h_target is used; callers should check monotonicity separately."""
    pts = sorted((float(h), float(p)) for h, p in points)
    hs = [h for h, _ in pts]
    if not pts or h_target < hs[0] - 1e-12 or h_target > hs[-1] + 1e-12:
        raise ExtrapolationError(f"H={h_target} outside observed range [{hs[0] if hs else None}, {hs[-1] if hs else None}]")
    best = None
    for (h0, p0), (h1, p1) in zip(pts[:-1], pts[1:]):
        if h0 - 1e-12 <= h_target <= h1 + 1e-12:
            width = h1 - h0
            if best is None or width < best[0]:
                best = (width, h0, p0, h1, p1)
    if best is None:  # exact single point
        for h, p in pts:
            if abs(h - h_target) < 1e-12:
                return p
        raise ExtrapolationError("no bracketing pair")
    _, h0, p0, h1, p1 = best
    if h1 == h0:
        return math.exp(0.5 * (math.log(p0) + math.log(p1)))
    a = (h_target - h0) / (h1 - h0)
    return math.exp((1 - a) * math.log(p0) + a * math.log(p1))


def is_monotone(points: Sequence[tuple[float, float]]) -> bool:
    """True if log PPL is non-decreasing in H along the sorted frontier."""
    pts = sorted(points)
    return all(p1 >= p0 for (_, p0), (_, p1) in zip(pts[:-1], pts[1:]))


def frontier_residual(points, h_new: float, ppl_new: float) -> float:
    """log(PPL_new) - log(frontier PPL at H_new). Negative = better than the baseline frontier
    at matched entropy. Raises ExtrapolationError if H_new is outside the frontier."""
    return math.log(ppl_new) - math.log(frontier_ppl_at(points, h_new))


def relative_change(baseline: float, new: float) -> float:
    """100 * (baseline - new) / baseline  (positive = reduction)."""
    return 100.0 * (baseline - new) / baseline
