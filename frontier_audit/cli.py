"""
frontier_audit.cli: Command-line interface for the frontier_audit diagnostic toolkit.

Usage examples:
  python -m frontier_audit.cli demo
  python -m frontier_audit.cli tost --a 32.3,32.4,32.5 --b 32.4,32.5,32.6 --low -0.5 --high 0.5 --paired
  python -m frontier_audit.cli gatekeeping --family1 fam1.json --family2 fam2.json
"""

import argparse
import json
import sys
from typing import List
import numpy as np

from .core import IsoFrontierAudit
from .tost import (
    tost_paired,
    tost_welch,
    hierarchical_gatekeeping,
    format_multiplicity_p,
)


def _parse_float_list(arg: str) -> List[float]:
    if "," in arg:
        return [float(x.strip()) for x in arg.split(",") if x.strip()]
    # Check if file path
    try:
        with open(arg, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if "," in content:
                return [float(x.strip()) for x in content.split(",") if x.strip()]
            return [float(line.strip()) for line in content.splitlines() if line.strip()]
    except Exception:
        return [float(arg)]


def run_demo() -> int:
    """Run an automated synthetic verification demo of frontier_audit."""
    print("=" * 70)
    print("frontier_audit: Verification Demo on Synthetic Response Surface")
    print("=" * 70)

    # 1. Construct regular grid over s in [1.0, 4.0], gamma in [1.0, 2.5]
    s_grid = np.linspace(1.0, 4.0, 7)
    g_grid = np.linspace(1.0, 2.5, 7)
    S, G = np.meshgrid(s_grid, g_grid, indexing="ij")

    # Primary metrics: PPL and Entropy (strongly collinear)
    P = 35.0 - 5.0 * (G - 1.0) + 2.0 * (S - 1.0)
    H = 5.30 - 0.10 * (G - 1.0) + 0.04 * (S - 1.0)

    # Secondary metrics:
    # 1) Null metric: m_null = 0.05 * (P / 35.0) (strictly dependent on PPL)
    # 2) Divergent metric: m_div = 0.04 + 0.015 * (S - 1.0) (diverges along iso-PPL)
    M_null = 0.05 * (P / 35.0)
    M_div = 0.04 + 0.015 * (S - 1.0)

    audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m_null": M_null, "m_div": M_div})

    p_target = 33.0
    print(f"Tracing iso-PPL level curve at P_target = {p_target}...")
    curve = audit.trace_iso_curve(p_target, s_range=(1.0, 3.5), n_pts=25)
    print(f"Traced {len(curve.s)} points. Total arc length: {curve.total_arc_length:.4f}")

    # Evaluate Jacobian test at midpoint
    s_mid = float(curve.s[len(curve.s) // 2])
    g_mid = float(curve.g[len(curve.g) // 2])
    print(f"\nEvaluating midpoint (s={s_mid:.2f}, gamma={g_mid:.2f}):")

    res_null = audit.jacobian_rank_test(s_mid, g_mid, "m_null")
    print(f"  [Null Metric]: ortho_ratio={res_null.ortho_ratio:.4f}, kappa={res_null.kappa_m:.4f}, is_null={res_null.is_null}")

    res_div = audit.jacobian_rank_test(s_mid, g_mid, "m_div")
    print(f"  [Divergent Metric]: ortho_ratio={res_div.ortho_ratio:.4f}, kappa={res_div.kappa_m:.4f}, is_null={res_div.is_null}")

    assert res_null.is_null, "Null metric should be detected as null"
    assert not res_div.is_null, "Divergent metric should be detected as non-null"

    # Multi-seed bootstrap demo
    print("\nRunning multi-seed bootstrap sensitivity CI...")
    rng = np.random.RandomState(42)
    seed_grids = []
    for _ in range(10):
        noise = rng.normal(0.0, 0.01, size=P.shape)
        seed_grids.append({
            "ppl": P + noise,
            "h": H + noise * 0.01,
            "m_div": M_div + noise * 0.001,
        })

    ci_res = audit.bootstrap_kappa_ci(seed_grids, s_grid, g_grid, p_target, s_mid, "m_div", n_boot=100)
    print(f"  Bootstrap kappa 95% CI: [{ci_res['ci_lower']:.4f}, {ci_res['ci_upper']:.4f}], point: {ci_res['kappa_point']:.4f}")
    assert ci_res["ci_lower"] > 0, "Divergent metric 95% CI should exclude zero"

    # Gatekeeping demo
    print("\nDemonstrating Hierarchical Gatekeeping (Family 1 Gate -> Family 2 Confirmatory):")
    fam1 = {"C1_joint": 0.0011, "C2_joint": 0.0004, "C3_joint": 0.0030}
    fam2 = {
        "C1_rep4": 4.89e-8, "C1_jsd": 1.31e-5, "C1_mauve": 0.00838,
        "C2_rep4": 2.59e-8, "C2_jsd": 7.01e-7, "C2_mauve": 0.000165,
        "C3_rep4": 2.63e-9, "C3_jsd": 1.61e-7, "C3_mauve": 0.000165,
    }
    gate_res = hierarchical_gatekeeping(fam1, fam2, alpha=0.05)
    print(f"  Gate Status: {gate_res['gate_status']}")
    print(f"  Family 1 (m=3): All passed = {gate_res['family1']['all_passed']}")
    print(f"  Family 2 (m=9): Rejections = {gate_res['family2']['n_significant']} / {gate_res['family2']['m']}")
    for k, p_adj in gate_res["family2"]["adjusted_p_values"].items():
        print(f"    {k:12s}: {format_multiplicity_p(p_adj, 'Family 2', m=9)}")

    print("\nDemo completed successfully. All assertions passed!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="frontier-audit",
        description="Diagnostic toolkit for continuous-latent flow sampler equivalence & iso-frontier sensitivity.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # Subcommand: demo
    subparsers.add_parser("demo", help="Run synthetic demonstration and self-test")

    # Subcommand: tost
    tost_parser = subparsers.add_parser("tost", help="Run Two One-Sided Tests (TOST)")
    tost_parser.add_argument("--a", required=True, help="Arm A observations (comma-separated or file)")
    tost_parser.add_argument("--b", required=True, help="Arm B observations (comma-separated or file)")
    tost_parser.add_argument("--low", type=float, required=True, help="Lower equivalence bound")
    tost_parser.add_argument("--high", type=float, required=True, help="Upper equivalence bound")
    tost_parser.add_argument("--paired", action="store_true", default=True, help="Paired CRN test (default)")
    tost_parser.add_argument("--unpaired", action="store_true", help="Unpaired Welch test")
    tost_parser.add_argument("--alpha", type=float, default=0.05, help="Alpha level (default: 0.05)")

    # Subcommand: gatekeeping
    gate_parser = subparsers.add_parser("gatekeeping", help="Run Hierarchical Gatekeeping")
    gate_parser.add_argument("--family1", required=True, help="Path to Family 1 p-values JSON")
    gate_parser.add_argument("--family2", required=True, help="Path to Family 2 p-values JSON")
    gate_parser.add_argument("--alpha", type=float, default=0.05, help="Alpha level (default: 0.05)")

    args = parser.parse_args()

    if args.subcommand == "demo" or args.subcommand is None:
        return run_demo()

    if args.subcommand == "tost":
        arr_a = _parse_float_list(args.a)
        arr_b = _parse_float_list(args.b)
        if args.unpaired:
            res = tost_welch(arr_a, arr_b, args.low, args.high, alpha=args.alpha)
        else:
            res = tost_paired(arr_a, arr_b, args.low, args.high, alpha=args.alpha)
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    if args.subcommand == "gatekeeping":
        with open(args.family1, "r", encoding="utf-8") as f:
            fam1 = json.load(f)
        with open(args.family2, "r", encoding="utf-8") as f:
            fam2 = json.load(f)
        res = hierarchical_gatekeeping(fam1, fam2, alpha=args.alpha)
        print(json.dumps(res, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
