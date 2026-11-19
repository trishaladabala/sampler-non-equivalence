"""
tests/test_adversarial_tost.py: Adversarial challenge suite for frontier_audit/tost.py.

Stress-tests:
1. Degenerate arrays: zero-variance arrays, constant differences inside/outside/boundary, N=2 minimal sample size, invalid shapes and bounds.
2. Extreme correlations: exact r = 1.0, r = -1.0, and r = 0.0 under CRN pairing vs unpaired Welch.
3. Multiplicity edge cases in Holm-Bonferroni: all 0s, all 1s, exact ties, m=1, empty list/dict, and NaN behavior.
4. Hierarchical gatekeeping 4-quadrant matrix: all-pass, mixed Family 1, mixed Family 2, all-fail Family 1.
5. Numerical scale invariance: extreme scale ($10^8$ vs $10^{-8}$) and boundary behavior.
"""

import math
import os
import sys
import unittest
import numpy as np
import scipy.stats as stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier_audit.tost import (
    tost_paired,
    tost_welch,
    joint_tost,
    holm_bonferroni,
    hierarchical_gatekeeping,
    format_multiplicity_p,
    TOSTResult,
)


class TestAdversarialTOST(unittest.TestCase):
    """Adversarial stress-test suite for statistical equivalence and gatekeeping."""

    def test_degenerate_zero_variance_identical_arrays(self):
        """Zero-variance: Arm A and Arm B identical, clamping SE to 1e-15."""
        a = [5.0, 5.0, 5.0, 5.0]
        b = [5.0, 5.0, 5.0, 5.0]
        low, high = -1.0, 1.0

        # Paired TOST
        res_p = tost_paired(a, b, low=low, high=high, alpha=0.05)
        self.assertEqual(res_p.diff, 0.0)
        self.assertAlmostEqual(res_p.se, 1e-15, places=14)
        self.assertEqual(res_p.p_tost, 0.0)
        self.assertTrue(res_p.is_equivalent)
        self.assertIsNone(res_p.correlation, "Constant array must have correlation=None")
        self.assertAlmostEqual(res_p.ci_diff_lower, 0.0, places=10)
        self.assertAlmostEqual(res_p.ci_diff_upper, 0.0, places=10)

        # Welch TOST
        res_w = tost_welch(a, b, low=low, high=high, alpha=0.05)
        self.assertEqual(res_w.diff, 0.0)
        self.assertAlmostEqual(res_w.se, 1e-15, places=14)
        self.assertEqual(res_w.p_tost, 0.0)
        self.assertTrue(res_w.is_equivalent)
        self.assertEqual(res_w.df, 6.0, "Welch df must fallback to n_a + n_b - 2 = 6")

    def test_degenerate_constant_differences_inside_outside_boundary(self):
        """Constant non-zero difference: test equivalence inside, outside, and on boundary."""
        a = [1.0, 2.0, 3.0, 4.0]
        low, high = -0.5, 0.5

        # 1. Constant diff strictly inside (-0.2 in [-0.5, 0.5])
        b_inside = [0.8, 1.8, 2.8, 3.8]  # diff = -0.2
        res_in = tost_paired(a, b_inside, low=low, high=high, alpha=0.05)
        self.assertAlmostEqual(res_in.diff, -0.2)
        self.assertAlmostEqual(res_in.se, 1e-15, places=14)
        self.assertEqual(res_in.p_tost, 0.0)
        self.assertTrue(res_in.is_equivalent)

        # 2. Constant diff strictly outside (+1.0 not in [-0.5, 0.5])
        b_outside = [2.0, 3.0, 4.0, 5.0]  # diff = +1.0
        res_out = tost_paired(a, b_outside, low=low, high=high, alpha=0.05)
        self.assertAlmostEqual(res_out.diff, 1.0)
        self.assertEqual(res_out.p_tost, 1.0)
        self.assertFalse(res_out.is_equivalent)

        # 3. Constant diff exactly on upper boundary (+0.5)
        b_boundary = [1.5, 2.5, 3.5, 4.5]  # diff = +0.5 == high
        res_bound = tost_paired(a, b_boundary, low=low, high=high, alpha=0.05)
        self.assertAlmostEqual(res_bound.diff, 0.5)
        self.assertAlmostEqual(res_bound.p_tost, 0.5, places=5)
        self.assertFalse(res_bound.is_equivalent, "Boundary difference must not be declared equivalent (p=0.5 >= alpha)")

    def test_degenerate_minimum_sample_size_n2(self):
        """Minimal sample size N=2: heavy-tailed Cauchy (df=1) distribution."""
        a = [10.0, 20.0]
        b = [10.1, 20.2]  # diff = [0.1, 0.2], mean = 0.15, sd = 0.07071..., se = 0.05
        low, high = -0.5, 0.5

        res_p = tost_paired(a, b, low=low, high=high, alpha=0.05)
        self.assertEqual(res_p.df, 1.0)
        self.assertAlmostEqual(res_p.diff, 0.15, places=10)
        self.assertAlmostEqual(res_p.se, 0.05, places=10)
        # For df=1, alpha=0.05, t_crit = tan(0.45*pi) = 6.313751514675045
        t_crit = math.tan(0.45 * math.pi)
        self.assertAlmostEqual(res_p.ci_diff_lower, 0.15 - t_crit * 0.05, places=8)
        self.assertAlmostEqual(res_p.ci_diff_upper, 0.15 + t_crit * 0.05, places=8)
        self.assertTrue(res_p.is_equivalent)

        # Welch with n_a=2, n_b=2
        res_w = tost_welch(a, b, low=low, high=high, alpha=0.05)
        self.assertTrue(1.0 <= res_w.df <= 2.0)
        self.assertFalse(res_w.is_equivalent, "Welch without pairing variance reduction should not declare equivalence")

    def test_degenerate_invalid_sample_size_and_bounds(self):
        """Input validation: enforce n >= 2 and low < high."""
        # N=1
        with self.assertRaises(ValueError):
            tost_paired([1.0], [1.0], -1.0, 1.0)
        with self.assertRaises(ValueError):
            tost_welch([1.0], [1.0], -1.0, 1.0)

        # Shape mismatch in paired
        with self.assertRaises(ValueError):
            tost_paired([1.0, 2.0], [1.0, 2.0, 3.0], -1.0, 1.0)

        # Bound ordering violations
        with self.assertRaises(ValueError):
            tost_paired([1.0, 2.0], [1.0, 2.0], 1.0, 1.0)  # low == high
        with self.assertRaises(ValueError):
            tost_paired([1.0, 2.0], [1.0, 2.0], 2.0, 1.0)  # low > high
        with self.assertRaises(ValueError):
            tost_welch([1.0, 2.0], [1.0, 2.0], 1.0, 1.0)

    def test_extreme_correlation_positive_one(self):
        """Extreme correlation r = +1.0: perfect pairing gives massive SE reduction."""
        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # b has exact linear relationship: r = +1.0
        b = a + 0.05

        res_p = tost_paired(a, b, low=-0.2, high=0.2, alpha=0.05)
        res_w = tost_welch(a, b, low=-0.2, high=0.2, alpha=0.05)

        self.assertAlmostEqual(res_p.correlation, 1.0, places=12)
        # Paired SE is clamped to 1e-15, Welch SE is ~7.07
        self.assertLess(res_p.se, 1e-14)
        self.assertGreater(res_w.se, 5.0)
        self.assertTrue(res_p.is_equivalent)
        self.assertFalse(res_w.is_equivalent)

    def test_extreme_correlation_negative_one(self):
        """Extreme correlation r = -1.0: anti-pairing inflates paired SE beyond Welch SE."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # b = -a + 6.0: perfect negative correlation r = -1.0
        b = -a + 6.0

        res_p = tost_paired(a, b, low=-10.0, high=10.0, alpha=0.05)
        res_w = tost_welch(a, b, low=-10.0, high=10.0, alpha=0.05)

        self.assertAlmostEqual(res_p.correlation, -1.0, places=12)
        # Var(b - a) = Var(-2a) = 4 Var(a). SE_paired = 2 * sd(a) / sqrt(5).
        # Var(a) + Var(b) = 2 Var(a). SE_welch = sqrt(2) * sd(a) / sqrt(5).
        # Therefore SE_paired should be sqrt(2) times larger than SE_welch!
        self.assertAlmostEqual(res_p.se / res_w.se, math.sqrt(2.0), places=10)
        self.assertGreater(res_p.se, res_w.se)

    def test_extreme_correlation_zero(self):
        """Exact correlation r = 0.0: paired SE equals Welch SE."""
        # Construct orthogonal vectors: sum((a - mean_a) * (b - mean_b)) = 0
        a = np.array([1.0, 2.0, 3.0])  # centered: [-1, 0, 1]
        b = np.array([1.0, -2.0, 1.0])  # centered: [1, -2, 1]
        # dot product = -1*1 + 0*-2 + 1*1 = 0

        res_p = tost_paired(a, b, low=-10.0, high=10.0, alpha=0.05)
        res_w = tost_welch(a, b, low=-10.0, high=10.0, alpha=0.05)

        self.assertAlmostEqual(res_p.correlation, 0.0, places=14)
        self.assertAlmostEqual(res_p.se, res_w.se, places=12)

    def test_holm_bonferroni_all_zero_and_all_one(self):
        """Holm-Bonferroni boundary p-values: all 0.0 and all 1.0."""
        # All 0.0
        res_0 = holm_bonferroni([0.0, 0.0, 0.0, 0.0], alpha=0.05)
        np.testing.assert_array_equal(res_0["adjusted_p_values"], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(res_0["significant"], [True, True, True, True])

        # All 1.0
        res_1 = holm_bonferroni([1.0, 1.0, 1.0], alpha=0.05)
        np.testing.assert_array_equal(res_1["adjusted_p_values"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(res_1["significant"], [False, False, False])

        # Single hypothesis m=1
        res_single = holm_bonferroni([0.03], alpha=0.05)
        self.assertEqual(res_single["adjusted_p_values"][0], 0.03)
        self.assertTrue(res_single["significant"][0])

    def test_holm_bonferroni_ties_and_monotonicity(self):
        """Holm-Bonferroni ties must receive identical adjusted p-values and maintain step-down monotonicity."""
        # Unsorted inputs with multiple ties
        raw_p = [0.04, 0.01, 0.01, 0.04, 0.02]
        # m = 5
        # Sorted:
        # rank 0: 0.01 * 5 = 0.05
        # rank 1: 0.01 * 4 = 0.04 -> running max = 0.05
        # rank 2: 0.02 * 3 = 0.06 -> running max = 0.06
        # rank 3: 0.04 * 2 = 0.08 -> running max = 0.08
        # rank 4: 0.04 * 1 = 0.04 -> running max = 0.08
        # Adjusted for tied 0.01s: both must be 0.05
        # Adjusted for tied 0.04s: both must be 0.08
        res = holm_bonferroni(raw_p, alpha=0.05)
        adj = res["adjusted_p_values"]

        # Check tie equality
        self.assertEqual(adj[1], adj[2], "Tied 0.01s must have identical adjusted p-values")
        self.assertEqual(adj[0], adj[3], "Tied 0.04s must have identical adjusted p-values")
        self.assertAlmostEqual(adj[1], 0.05, places=10)
        self.assertAlmostEqual(adj[0], 0.08, places=10)

        # Monotonicity check
        sorted_indices = np.argsort(raw_p)
        sorted_adj = adj[sorted_indices]
        for i in range(len(sorted_adj) - 1):
            self.assertLessEqual(sorted_adj[i], sorted_adj[i + 1])

    def test_holm_bonferroni_empty_inputs(self):
        """Empty input structures must return well-typed empty containers without error."""
        # Empty list
        res_list = holm_bonferroni([])
        self.assertEqual(res_list["m"], 0)
        self.assertEqual(len(res_list["adjusted_p_values"]), 0)
        self.assertEqual(len(res_list["significant"]), 0)

        # Empty dict
        res_dict = holm_bonferroni({})
        self.assertEqual(res_dict["m"], 0)
        self.assertEqual(res_dict["adjusted_p_values"], {})
        self.assertEqual(res_dict["significant"], {})

    def test_hierarchical_gatekeeping_four_quadrant_matrix(self):
        """Hierarchical gatekeeping matrix across Family 1 and Family 2 states."""
        f1_all_pass = {"C1": 0.001, "C2": 0.002, "C3": 0.003}
        f1_mixed = {"C1": 0.001, "C2": 0.060, "C3": 0.002}  # C2 fails
        f1_all_fail = {"C1": 0.100, "C2": 0.200, "C3": 0.300}

        f2_all_pass = {"rep4": 0.001, "jsd": 0.002, "mauve": 0.003}
        f2_mixed = {"rep4": 0.001, "jsd": 0.030, "mauve": 0.040}  # for m=3: 0.03*2=0.06 -> fails

        # Quadrant 1: F1 PASS, F2 PASS -> OPEN, all 3 significant
        q1 = hierarchical_gatekeeping(f1_all_pass, f2_all_pass, alpha=0.05)
        self.assertEqual(q1["gate_status"], "OPEN")
        self.assertTrue(q1["family1"]["all_passed"])
        self.assertEqual(q1["family2"]["n_significant"], 3)

        # Quadrant 2: F1 PASS, F2 MIXED -> OPEN, only rep4 significant
        q2 = hierarchical_gatekeeping(f1_all_pass, f2_mixed, alpha=0.05)
        self.assertEqual(q2["gate_status"], "OPEN")
        self.assertTrue(q2["family1"]["all_passed"])
        self.assertEqual(q2["family2"]["n_significant"], 1)
        self.assertTrue(q2["family2"]["significant"]["rep4"])
        self.assertFalse(q2["family2"]["significant"]["jsd"])
        self.assertFalse(q2["family2"]["significant"]["mauve"])

        # Quadrant 3: F1 MIXED, F2 PASS -> CLOSED, 0 significant, all adj p = 1.0
        q3 = hierarchical_gatekeeping(f1_mixed, f2_all_pass, alpha=0.05)
        self.assertEqual(q3["gate_status"], "CLOSED")
        self.assertFalse(q3["family1"]["all_passed"])
        self.assertEqual(q3["family2"]["n_significant"], 0)
        self.assertTrue(all(v == 1.0 for v in q3["family2"]["adjusted_p_values"].values()))
        self.assertTrue(all(v is False for v in q3["family2"]["significant"].values()))

        # Quadrant 4: F1 FAIL, F2 PASS -> CLOSED, 0 significant
        q4 = hierarchical_gatekeeping(f1_all_fail, f2_all_pass, alpha=0.05)
        self.assertEqual(q4["gate_status"], "CLOSED")
        self.assertFalse(q4["family1"]["all_passed"])
        self.assertEqual(q4["family2"]["n_significant"], 0)

    def test_joint_tost_intersection_union_multitest(self):
        """Joint TOST requires every single constituent test to establish equivalence."""
        pass_res = [tost_paired([1, 2, 3], [1.01, 2.01, 3.01], -0.5, 0.5) for _ in range(5)]
        fail_res = tost_paired([1, 2, 3], [5, 6, 7], -0.5, 0.5)

        # 5 passing metrics
        joint_pass = joint_tost(pass_res)
        self.assertTrue(joint_pass["is_equivalent"])
        self.assertEqual(len(joint_pass["individual_results"]), 5)

        # 5 passing + 1 failing
        joint_fail = joint_tost(pass_res + [fail_res])
        self.assertFalse(joint_fail["is_equivalent"])
        self.assertEqual(joint_fail["p_joint"], fail_res.p_tost)

        # Empty list error
        with self.assertRaises(ValueError):
            joint_tost([])

    def test_numerical_scale_invariance(self):
        """Test numeric stability across extreme scales: 10^8 and 10^-8."""
        # High magnitude (10^8)
        scale_high = 1e8
        a_high = np.array([1.0, 2.0, 3.0, 4.0]) * scale_high
        b_high = a_high + 0.01 * scale_high
        res_high = tost_paired(a_high, b_high, low=-0.05 * scale_high, high=0.05 * scale_high)
        self.assertTrue(res_high.is_equivalent)
        self.assertFalse(math.isnan(res_high.p_tost))

        # Low magnitude (10^-8)
        scale_low = 1e-8
        a_low = np.array([1.0, 2.0, 3.0, 4.0]) * scale_low
        b_low = a_low + 0.01 * scale_low
        res_low = tost_paired(a_low, b_low, low=-0.05 * scale_low, high=0.05 * scale_low)
        self.assertTrue(res_low.is_equivalent)
        self.assertFalse(math.isnan(res_low.p_tost))


if __name__ == "__main__":
    unittest.main()
