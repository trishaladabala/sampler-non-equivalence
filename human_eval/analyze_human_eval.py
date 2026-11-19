#!/usr/bin/env python3
"""Scoring and Inter-Annotator Agreement Script for Blinded Human Evaluation.

Computes win rates, Wilson score 95% confidence intervals, exact two-sided
binomial p-values, and Fleiss' kappa for Task 1 (Single-Passage Repetition)
and Task 2 (Multi-Passage Bundle Diversity).
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, z: float = 1.95996) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + ((z ** 2) / (4.0 * (n ** 2)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa for N items x C categories where each row sums to R raters."""
    N, C = ratings_matrix.shape
    R = int(ratings_matrix[0].sum())
    if R <= 1 or N == 0:
        return float("nan")
    p_j = ratings_matrix.sum(axis=0) / (N * R)
    P_e = float(np.sum(p_j ** 2))
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - R) / (R * (R - 1))
    P_bar = float(np.mean(P_i))
    if abs(1.0 - P_e) < 1e-12:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def main():
    parser = argparse.ArgumentParser(description="Analyze blinded human evaluation ratings")
    parser.add_argument("--ratings", type=Path, help="Path to CSV of rater annotations")
    parser.add_argument("--key", type=Path, help="Path to secret randomization key JSON")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic self-test of agreement/CI functions")
    args = parser.parse_args()

    if args.self_test:
        lo, hi = wilson_ci(35, 50)
        assert 0.55 < lo < 0.70 < hi < 0.85, f"Unexpected Wilson CI: {(lo, hi)}"
        perfect = np.array([[3, 0, 0], [0, 3, 0], [3, 0, 0], [0, 3, 0]], dtype=float)
        assert abs(fleiss_kappa(perfect) - 1.0) < 1e-9, "Expected Fleiss kappa = 1.0 on unanimous agreement"
        print("[analyze_human_eval] Self-test PASSED (Wilson CI and Fleiss' kappa verified).")
        return

    if not args.ratings or not args.ratings.exists():
        print("[analyze_human_eval] No ratings file provided or found yet. Protocol is ready for annotation.")
        return

    with open(args.key) as f:
        key_map = json.load(f)

    # Aggregate majority votes and rater matrices
    by_item: Dict[str, List[str]] = {}
    with open(args.ratings) as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row["item_id"]
            choice = row["choice"]  # 'X', 'Y', or 'Tie'
            # Map 'X'/'Y' back to 'A' (SC=1,g=2.0) or 'B' (SC=4,g=0.5)
            mapping = key_map[item_id]
            decoded = mapping.get(choice, "Tie")
            by_item.setdefault(item_id, []).append(decoded)

    wins_a = wins_b = ties = 0
    mat = []
    for item_id, votes in sorted(by_item.items()):
        ca = votes.count("A")
        cb = votes.count("B")
        ct = votes.count("Tie")
        mat.append([ca, cb, ct])
        if cb > ca and cb >= ct:
            wins_b += 1
        elif ca > cb and ca >= ct:
            wins_a += 1
        else:
            ties += 1

    n_decisive = wins_a + wins_b
    binom_p = float(stats.binomtest(wins_b, n_decisive, p=0.5).pvalue) if n_decisive > 0 else float("nan")
    ci_lo, ci_hi = wilson_ci(wins_b, n_decisive)
    kappa = fleiss_kappa(np.array(mat, dtype=float)) if mat else float("nan")

    print(f"Items: {len(by_item)} | B wins: {wins_b} | A wins: {wins_a} | Ties: {ties}")
    print(f"B decisive win rate: {wins_b / max(n_decisive, 1):.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}] (p = {binom_p:.4g})")
    print(f"Fleiss' kappa: {kappa:.3f}")


if __name__ == "__main__":
    main()
