"""
tests/test_gatekeeping.py: Unit tests for TOST equivalence, Holm adjustment, and hierarchical gatekeeping.

Verifies:
1. Paired vs unpaired Welch TOST test statistics against exact scipy t-distributions.
2. Variance reduction and increased power under Common Random Numbers (CRN).
3. Step-down Holm-Bonferroni multiplicity correction properties (monotonicity, capping).
4. Hierarchical gatekeeping behavior:
   - Gate OPEN when all Family 1 tests pass under Holm alpha=0.05.
   - Gate CLOSED when any Family 1 test fails (barring all Family 2 confirmatory rejections).
5. Exact paper replication case: C1-C3 Family 1 gate + Family 2 9-way confirmatory tests.
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

from frontier_audit import (
    tost_paired,
    tost_welch,
    joint_tost,
    holm_bonferroni,
    hierarchical_gatekeeping,
    format_multiplicity_p,
)


class TestGatekeepingAndTOST(unittest.TestCase):
    """Test suite for TOST equivalence testing, Holm correction, and gatekeeping."""

    def test_paired_tost_against_scipy_analytical(self):
        """Verify paired TOST test statistics against manual analytical formulas."""
        # Fixed observations
        a = np.array([32.1, 32.4, 32.3, 32.5, 32.2])
        b = np.array([32.3, 32.5, 32.4, 32.7, 32.3])
        d = b - a  # [0.2, 0.1, 0.1, 0.2, 0.1]
        n = 5
        df = n - 1

        mean_d = float(np.mean(d))
        sd_d = float(np.std(d, ddof=1))
        se_d = sd_d / math.sqrt(n)

        low = -0.50
        high = +0.50

        # Analytical t-statistics
        t1_expected = (mean_d - low) / se_d
        p1_expected = float(1.0 - stats.t.cdf(t1_expected, df=df))

        t2_expected = (high - mean_d) / se_d
        p2_expected = float(1.0 - stats.t.cdf(t2_expected, df=df))

        p_tost_expected = max(p1_expected, p2_expected)

        res = tost_paired(a, b, low=low, high=high, alpha=0.05)

        self.assertAlmostEqual(res.diff, mean_d, places=10)
        self.assertAlmostEqual(res.se, se_d, places=10)
        self.assertEqual(res.df, 4.0)
        self.assertAlmostEqual(res.t1_lower, t1_expected, places=10)
        self.assertAlmostEqual(res.p1_lower, p1_expected, places=10)
        self.assertAlmostEqual(res.t2_upper, t2_expected, places=10)
        self.assertAlmostEqual(res.p2_upper, p2_expected, places=10)
        self.assertAlmostEqual(res.p_tost, p_tost_expected, places=10)
        self.assertTrue(res.is_equivalent)

    def test_paired_crn_variance_reduction(self):
        """Verify that positive CRN correlation reduces SE and increases TOST power."""
        rng = np.random.RandomState(42)
        n = 20
        # Correlated Gaussian random variables (r ~ 0.8)
        base = rng.normal(30.0, 1.0, size=n)
        a = base + rng.normal(0.0, 0.2, size=n)
        b = base + rng.normal(0.05, 0.2, size=n)  # mean diff ~ 0.05

        margin = 0.30
        res_paired = tost_paired(a, b, low=-margin, high=margin, alpha=0.05)
        res_welch = tost_welch(a, b, low=-margin, high=margin, alpha=0.05)

        # SE of paired differences should be much smaller than unpaired Welch SE
        self.assertLess(res_paired.se, res_welch.se)
        self.assertGreater(res_paired.correlation, 0.70)
        # Smaller SE translates to lower p-value (greater power to declare equivalence)
        self.assertLess(res_paired.p_tost, res_welch.p_tost)

    def test_tost_non_equivalence_when_outside_margin(self):
        """Verify that TOST rejects equivalence when difference is outside margin."""
        a = np.array([30.0, 30.1, 30.2, 30.0, 30.1])
        b = np.array([31.5, 31.6, 31.4, 31.5, 31.6])  # mean diff ~ 1.44

        margin = 0.50  # margin [-0.5, 0.5] does not contain 1.44
        res = tost_paired(a, b, low=-margin, high=margin, alpha=0.05)

        self.assertFalse(res.is_equivalent, "TOST must not declare equivalence when difference is outside margin")
        self.assertGreater(res.p_tost, 0.50, "p_tost must be large when difference exceeds upper margin")

    def test_joint_tost_intersection_union(self):
        """Verify joint TOST requires all constituent metrics to demonstrate equivalence."""
        res_pass1 = tost_paired([1, 2, 3], [1.1, 2.05, 3.05], low=-0.5, high=0.5, alpha=0.05)
        res_pass2 = tost_paired([5, 6, 7], [5.05, 6.02, 7.03], low=-0.5, high=0.5, alpha=0.05)
        res_fail = tost_paired([1, 2, 3], [3, 4, 5], low=-0.5, high=0.5, alpha=0.05)

        # Both pass => joint passes
        joint_pass = joint_tost([res_pass1, res_pass2])
        self.assertTrue(joint_pass["is_equivalent"])
        self.assertEqual(joint_pass["p_joint"], max(res_pass1.p_tost, res_pass2.p_tost))

        # One fails => joint fails
        joint_fail = joint_tost([res_pass1, res_fail])
        self.assertFalse(joint_fail["is_equivalent"])
        self.assertEqual(joint_fail["p_joint"], max(res_pass1.p_tost, res_fail.p_tost))

    def test_holm_bonferroni_step_down_properties(self):
        """Verify step-down Holm-Bonferroni correction on known analytical cases."""
        raw_p = [0.01, 0.02, 0.03, 0.05]
        # m = 4
        # Sorted:
        # rank 0: 0.01 * 4 = 0.04
        # rank 1: 0.02 * 3 = 0.06
        # rank 2: 0.03 * 2 = 0.06
        # rank 3: 0.05 * 1 = 0.05 => running max = 0.06
        # Adjusted: [0.04, 0.06, 0.06, 0.06]

        res = holm_bonferroni(raw_p, alpha=0.05)
        adj_p = res["adjusted_p_values"]

        np.testing.assert_allclose(adj_p, [0.04, 0.06, 0.06, 0.06], atol=1e-10)
        # At alpha=0.05, only the first test rejects
        np.testing.assert_array_equal(res["significant"], [True, False, False, False])

        # Test capping at 1.0
        res_cap = holm_bonferroni([0.6, 0.8], alpha=0.05)
        np.testing.assert_allclose(res_cap["adjusted_p_values"], [1.0, 1.0])

        # Test dict interface
        d_p = {"test_a": 0.001, "test_b": 0.04}
        res_d = holm_bonferroni(d_p, alpha=0.05)
        self.assertIn("test_a", res_d["adjusted_p_values"])
        self.assertAlmostEqual(res_d["adjusted_p_values"]["test_a"], 0.002)

    def test_hierarchical_gatekeeping_gate_open_and_closed(self):
        """Verify hierarchical gatekeeping properly controls family transitions."""
        # Case A: Family 1 Gate PASSES
        f1_pass = {"C1_joint": 0.001, "C2_joint": 0.002, "C3_joint": 0.005}
        f2_data = {"test1": 0.001, "test2": 0.004, "test3": 0.010}

        gate_open = hierarchical_gatekeeping(f1_pass, f2_data, alpha=0.05)
        self.assertEqual(gate_open["gate_status"], "OPEN")
        self.assertTrue(gate_open["family1"]["all_passed"])
        # With gate open, significant results can be declared
        self.assertGreater(gate_open["family2"]["n_significant"], 0)

        # Case B: Family 1 Gate FAILS (C3 joint p-value fails Holm)
        f1_fail = {"C1_joint": 0.001, "C2_joint": 0.002, "C3_joint": 0.060}
        gate_closed = hierarchical_gatekeeping(f1_fail, f2_data, alpha=0.05)
        self.assertEqual(gate_closed["gate_status"], "CLOSED")
        self.assertFalse(gate_closed["family1"]["all_passed"])
        # All Family 2 tests must be rejected = False when gate is closed
        self.assertEqual(gate_closed["family2"]["n_significant"], 0)
        for sig_val in gate_closed["family2"]["significant"].values():
            self.assertFalse(sig_val)

    def test_paper_confirmatory_reproducibility(self):
        """
        Verify the exact paper revision specification:
        Family 1 (3 pairs joint TOST) passes Holm alpha=0.05.
        Family 2 (9 tests across C1-C3 x rep4, JSD, MAUVE) all pass Holm alpha=0.05.
        """
        # Primary Joint TOST p-values from calibration/pilot data
        f1_paper = {
            "C1_Joint_PPL_H": 0.0021,
            "C2_Joint_PPL_H": 0.0022,
            "C3_Joint_PPL_H": 0.0123,
        }

        # Secondary 9-way confirmatory p-values
        f2_paper = {
            "C1_rep4": 4.89e-8,
            "C1_JSD": 1.31e-5,
            "C1_MAUVE": 0.00838,
            "C2_rep4": 2.59e-8,
            "C2_JSD": 7.01e-7,
            "C2_MAUVE": 0.000165,
            "C3_rep4": 2.63e-9,
            "C3_JSD": 1.61e-7,
            "C3_MAUVE": 0.000165,
        }

        result = hierarchical_gatekeeping(f1_paper, f2_paper, alpha=0.05)

        # Verify Gate is Open
        self.assertEqual(result["gate_status"], "OPEN")
        self.assertTrue(result["family1"]["all_passed"])

        # In Family 1 (m=3):
        # C3_Joint: 0.0123 * 1 = 0.0123 < 0.05
        # C2_Joint: 0.0022 * 2 = 0.0044 < 0.05
        # C1_Joint: 0.0021 * 3 = 0.0063 < 0.05
        for p_adj in result["family1"]["adjusted_p_values"].values():
            self.assertLess(p_adj, 0.05)

        # In Family 2 (m=9):
        # All 9 secondary comparisons must survive Holm correction at alpha=0.05!
        self.assertEqual(result["family2"]["n_significant"], 9)

        # Specifically verify C1 MAUVE (the largest p-value in the 9-test family):
        # Rank 8 (largest of 9): multiplier is 9 - 8 = 1.
        # Adjusted p = 0.00838 * 1 = 0.00838 < 0.05.
        c1_mauve_adj = result["family2"]["adjusted_p_values"]["C1_MAUVE"]
        self.assertAlmostEqual(c1_mauve_adj, 0.00838, places=5)
        self.assertLess(c1_mauve_adj, 0.05)

        # Verify formatting function
        text_str = format_multiplicity_p(c1_mauve_adj, "Family 2", m=9, alpha=0.05, style="text")
        self.assertIn("p_Holm = 0.0084 (Family 2, m=9, alpha=0.05)", text_str)

        latex_str = format_multiplicity_p(c1_mauve_adj, "Family 2", m=9, alpha=0.05, style="latex")
        self.assertIn("$p_{\\mathrm{Holm}} (\\mathrm{Fam}\\,2, m{=}9) = 0.0084$", latex_str)


if __name__ == "__main__":
    unittest.main()
