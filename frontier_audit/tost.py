"""
frontier_audit.tost: Statistical testing module for equivalence and gatekeeping.

Implements:
1. Two One-Sided Tests (TOST) under Common Random Numbers (paired Student's t, df=n-1).
2. Sensitivity TOST under unpaired Welch's t-test (Satterthwaite df).
3. Joint TOST on multiple primary equivalence metrics (e.g. Gen-PPL and Entropy).
4. Step-down Holm-Bonferroni Family-Wise Error Rate (FWER) multiplicity control.
5. Strict Hierarchical Gatekeeping across multi-family hypothesis hierarchies.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats


@dataclass
class TOSTResult:
    test_type: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    diff: float  # mean_b - mean_a
    se: float
    df: float
    low_bound: float
    high_bound: float
    t1_lower: float
    p1_lower: float
    t2_upper: float
    p2_upper: float
    p_tost: float
    is_equivalent: bool
    ci_level: float
    ci_diff_lower: float
    ci_diff_upper: float
    p_two_sided_diff: float
    correlation: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def tost_paired(
    a: Union[List[float], np.ndarray],
    b: Union[List[float], np.ndarray],
    low: float,
    high: float,
    alpha: float = 0.05,
) -> TOSTResult:
    """
    Paired Two One-Sided Tests (TOST) for equivalence under Common Random Numbers (CRN).

    Tests:
      H01: mu_d <= low   vs  H11: mu_d > low
      H02: mu_d >= high  vs  H12: mu_d < high
    where d = b - a.

    Equivalence is declared at level alpha if p_tost = max(p1, p2) < alpha,
    which is mathematically equivalent to the 100*(1 - 2*alpha)% confidence interval
    for the mean difference lying entirely within (low, high).

    Args:
        a: Observations for Arm A (e.g., baseline sampler).
        b: Observations for Arm B (e.g., elevated guidance sampler).
        low: Lower bound of equivalence interval (typically -margin).
        high: Upper bound of equivalence interval (typically +margin).
        alpha: Nominal significance level (default: 0.05).

    Returns:
        TOSTResult with test statistics, degrees of freedom, and equivalence decision.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Arrays must have identical shapes for paired TOST: {arr_a.shape} vs {arr_b.shape}")
    if len(arr_a) < 2:
        raise ValueError("At least 2 paired observations required.")
    if low >= high:
        raise ValueError(f"Equivalence bounds invalid: low ({low}) must be strictly less than high ({high})")

    diffs = arr_b - arr_a
    n = len(diffs)
    df = float(n - 1)
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    mean_diff = float(np.mean(diffs))

    sd_diff = float(np.std(diffs, ddof=1))
    se = sd_diff / np.sqrt(n)

    if se < 1e-15:
        se = 1e-15

    # Lower test: H01: mu_d <= low vs H11: mu_d > low
    t1 = (mean_diff - low) / se
    p1 = float(1.0 - stats.t.cdf(t1, df=df))

    # Upper test: H02: mu_d >= high vs H12: mu_d < high
    t2 = (high - mean_diff) / se
    p2 = float(1.0 - stats.t.cdf(t2, df=df))

    p_tost = float(max(p1, p2))
    is_equiv = bool(p_tost < alpha)

    # 1 - 2*alpha CI corresponds to TOST rejection region
    t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
    ci_lower = float(mean_diff - t_crit * se)
    ci_upper = float(mean_diff + t_crit * se)

    # Standard two-sided test for difference from zero
    t_zero = mean_diff / se
    p_two_sided = float(2.0 * stats.t.sf(abs(t_zero), df=df))

    corr = None
    if np.std(arr_a) > 1e-15 and np.std(arr_b) > 1e-15:
        corr = float(np.corrcoef(arr_a, arr_b)[0, 1])

    return TOSTResult(
        test_type="paired_crn_tost",
        n_a=n,
        n_b=n,
        mean_a=mean_a,
        mean_b=mean_b,
        diff=mean_diff,
        se=se,
        df=df,
        low_bound=float(low),
        high_bound=float(high),
        t1_lower=float(t1),
        p1_lower=p1,
        t2_upper=float(t2),
        p2_upper=p2,
        p_tost=p_tost,
        is_equivalent=is_equiv,
        ci_level=float(1.0 - 2.0 * alpha),
        ci_diff_lower=ci_lower,
        ci_diff_upper=ci_upper,
        p_two_sided_diff=p_two_sided,
        correlation=corr,
    )


def tost_welch(
    a: Union[List[float], np.ndarray],
    b: Union[List[float], np.ndarray],
    low: float,
    high: float,
    alpha: float = 0.05,
) -> TOSTResult:
    """
    Unpaired Welch TOST for equivalence (sensitivity check ignoring CRN pairing).

    Uses Welch-Satterthwaite approximation for degrees of freedom.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    n_a = len(arr_a)
    n_b = len(arr_b)

    if n_a < 2 or n_b < 2:
        raise ValueError("At least 2 observations required in each arm.")
    if low >= high:
        raise ValueError(f"Equivalence bounds invalid: low ({low}) must be strictly less than high ({high})")

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    mean_diff = float(mean_b - mean_a)

    var_a = float(np.var(arr_a, ddof=1))
    var_b = float(np.var(arr_b, ddof=1))

    se_a = var_a / n_a
    se_b = var_b / n_b
    se = np.sqrt(se_a + se_b)

    if se < 1e-15:
        se = 1e-15

    # Welch-Satterthwaite degrees of freedom
    denom = (se_a**2 / (n_a - 1)) + (se_b**2 / (n_b - 1))
    df = float((se_a + se_b) ** 2 / denom) if denom > 1e-15 else float(n_a + n_b - 2)

    t1 = (mean_diff - low) / se
    p1 = float(1.0 - stats.t.cdf(t1, df=df))

    t2 = (high - mean_diff) / se
    p2 = float(1.0 - stats.t.cdf(t2, df=df))

    p_tost = float(max(p1, p2))
    is_equiv = bool(p_tost < alpha)

    t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
    ci_lower = float(mean_diff - t_crit * se)
    ci_upper = float(mean_diff + t_crit * se)

    t_zero = mean_diff / se
    p_two_sided = float(2.0 * (1.0 - stats.t.cdf(abs(t_zero), df=df)))

    return TOSTResult(
        test_type="unpaired_welch_tost",
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        diff=mean_diff,
        se=se,
        df=df,
        low_bound=float(low),
        high_bound=float(high),
        t1_lower=float(t1),
        p1_lower=p1,
        t2_upper=float(t2),
        p2_upper=p2,
        p_tost=p_tost,
        is_equivalent=is_equiv,
        ci_level=float(1.0 - 2.0 * alpha),
        ci_diff_lower=ci_lower,
        ci_diff_upper=ci_upper,
        p_two_sided_diff=p_two_sided,
        correlation=None,
    )


def joint_tost(results: List[TOSTResult]) -> Dict[str, Union[float, bool, List[dict]]]:
    """
    Joint TOST across multiple primary metrics (Intersection-Union Principle).

    For an operating point to be declared equivalent across both Gen-PPL and Entropy,
    both must be statistically equivalent within their respective margins:
        p_joint = max(p_TOST_1, p_TOST_2, ...)
    """
    if not results:
        raise ValueError("Empty results list for joint TOST.")

    p_joint = max(r.p_tost for r in results)
    is_equiv = all(r.is_equivalent for r in results)

    return {
        "p_joint": float(p_joint),
        "is_equivalent": bool(is_equiv),
        "individual_results": [r.to_dict() for r in results],
    }


def holm_bonferroni(
    p_values: Union[List[float], np.ndarray, Dict[str, float]],
    alpha: float = 0.05,
) -> Dict[str, Union[np.ndarray, Dict[str, float], Dict[str, bool]]]:
    """
    Step-down Holm-Bonferroni correction for strong Family-Wise Error Rate (FWER) control.

    Args:
        p_values: Array, list, or dict of raw p-values.
        alpha: Nominal FWER threshold (default: 0.05).

    Returns:
        Dictionary containing:
        - 'adjusted_p_values': dict or ndarray of adjusted p-values
        - 'significant': dict or ndarray of boolean rejection decisions
        - 'm': number of tested hypotheses
        - 'alpha': nominal alpha level
    """
    is_dict = isinstance(p_values, dict)
    if is_dict:
        keys = list(p_values.keys())
        raw_p = np.array([p_values[k] for k in keys], dtype=np.float64)
    else:
        keys = None
        raw_p = np.asarray(p_values, dtype=np.float64)

    m = len(raw_p)
    if m == 0:
        return {
            "adjusted_p_values": {} if is_dict else np.array([]),
            "significant": {} if is_dict else np.array([], dtype=bool),
            "m": 0,
            "alpha": alpha,
        }

    # Step-down procedure: sort ascending
    sort_idx = np.argsort(raw_p)
    sorted_p = raw_p[sort_idx]

    adjusted_sorted = np.zeros(m, dtype=np.float64)
    running_max = 0.0

    for rank, p in enumerate(sorted_p):
        multiplier = m - rank
        adj = p * multiplier
        running_max = max(running_max, adj)
        adjusted_sorted[rank] = min(1.0, running_max)

    # Restore original ordering
    adjusted_p = np.zeros(m, dtype=np.float64)
    adjusted_p[sort_idx] = adjusted_sorted
    sig = adjusted_p < alpha

    if is_dict:
        adj_dict = {keys[i]: float(adjusted_p[i]) for i in range(m)}
        sig_dict = {keys[i]: bool(sig[i]) for i in range(m)}
        return {
            "adjusted_p_values": adj_dict,
            "significant": sig_dict,
            "m": m,
            "alpha": alpha,
        }

    return {
        "adjusted_p_values": adjusted_p,
        "significant": sig,
        "m": m,
        "alpha": alpha,
    }


def hierarchical_gatekeeping(
    family1_p_values: Dict[str, float],
    family2_p_values: Dict[str, float],
    alpha: float = 0.05,
) -> dict:
    """
    Strict Hierarchical Gatekeeping procedure across Family 1 (Gate) and Family 2 (Confirmatory).

    Gate Condition:
      All hypotheses in Family 1 must be rejected (pass equivalence) under Holm-Bonferroni
      correction at level alpha.
      If ANY hypothesis in Family 1 fails, the gate is CLOSED: Family 2 hypotheses are not
      confirmatory and cannot be rejected (all significant = False).
      If all Family 1 hypotheses pass, the gate is OPEN: Family 2 hypotheses are tested
      under Holm-Bonferroni correction at level alpha.

    Args:
        family1_p_values: Dict of test name -> raw joint TOST p-value for Family 1.
        family2_p_values: Dict of test name -> raw p-value for Family 2 secondary metrics.
        alpha: FWER threshold (default: 0.05).

    Returns:
        Structured dictionary with gate status, adjusted p-values, and formal decisions.
    """
    # 1. Evaluate Family 1 under Holm
    f1_holm = holm_bonferroni(family1_p_values, alpha=alpha)
    f1_passed = all(f1_holm["significant"].values())

    # 2. Evaluate Family 2 conditionally
    if f1_passed:
        gate_status = "OPEN"
        f2_holm = holm_bonferroni(family2_p_values, alpha=alpha)
        f2_adj = f2_holm["adjusted_p_values"]
        f2_sig = f2_holm["significant"]
    else:
        gate_status = "CLOSED"
        # When gate is closed, no confirmatory rejections are permitted
        f2_adj = {k: 1.0 for k in family2_p_values}
        f2_sig = {k: False for k in family2_p_values}

    return {
        "gate_status": gate_status,
        "family1": {
            "name": "Family 1: Operating-Point Equivalence Gate",
            "m": len(family1_p_values),
            "alpha": alpha,
            "raw_p_values": family1_p_values,
            "adjusted_p_values": f1_holm["adjusted_p_values"],
            "significant": f1_holm["significant"],
            "all_passed": f1_passed,
        },
        "family2": {
            "name": "Family 2: Confirmatory Secondary Metrics",
            "m": len(family2_p_values),
            "alpha": alpha,
            "raw_p_values": family2_p_values,
            "adjusted_p_values": f2_adj,
            "significant": f2_sig,
            "n_significant": sum(1 for v in f2_sig.values() if v),
        },
    }


def format_multiplicity_p(
    p_adj: float,
    family_name: str = "Family 2",
    m: int = 9,
    alpha: float = 0.05,
    style: str = "text",
) -> str:
    r"""
    Format adjusted p-value with explicit family, hypothesis count m, and alpha level.

    Examples:
      style="text":  "p_Holm = 0.0084 (Family 2, m=9, alpha=0.05)"
      style="latex": "$p_{\mathrm{Holm}} (\mathrm{Fam}\,2, m{=}9)$"
    """
    if style == "latex":
        fam_abbr = "\\mathrm{Fam}\\,1" if "1" in family_name else "\\mathrm{Fam}\\,2"
        if p_adj < 0.001:
            p_str = f"< 0.001"
        else:
            p_str = f"= {p_adj:.4f}"
        return f"$p_{{\\mathrm{{Holm}}}} ({fam_abbr}, m{{=}}{m}) {p_str}$"

    if p_adj < 0.0001:
        return f"p_Holm < 0.0001 ({family_name}, m={m}, alpha={alpha})"
    return f"p_Holm = {p_adj:.4f} ({family_name}, m={m}, alpha={alpha})"
