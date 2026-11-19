"""
tests/test_reaudit_m1.py: Independent adversarial re-audit verification suite for Milestone M1.

Rigorously verifies resolution of all 5 differential geometric, numerical, and boundary vulnerabilities:
1. Minimal grid dimension validation fail-closed (no FITPACK parder code 10 crash).
2. Moore-Penrose pseudo-inverse projection in jacobian_rank_test (exact subspace decomposition).
3. Traversal-consistent tangent orientation and sign conservation across forward/reverse curves.
4. Multi-branch continuity tracking preserving FTC across saddle point crossings.
5. Disconnected gap detection failing closed on domain exits and jump discontinuities.
"""

import math
import os
import sys
import unittest
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier_audit.core import IsoFrontierAudit, IsoCurveResult, JacobianTestResult


class TestReauditMinimalGridValidation(unittest.TestCase):
    """Deep verification of minimal grid dimension and numerical validation."""

    def test_grid_dimension_boundary_conditions(self):
        """Verify fail-closed behavior for all minimal grid dimension edge cases."""
        # Case 1: 2x3 grid
        with self.assertRaises(ValueError) as ctx:
            IsoFrontierAudit([1.0, 2.0], [1.0, 2.0, 3.0], np.ones((2, 3)), np.ones((2, 3)))
        self.assertIn("at least 3 points", str(ctx.exception))

        # Case 2: 3x2 grid
        with self.assertRaises(ValueError) as ctx:
            IsoFrontierAudit([1.0, 2.0, 3.0], [1.0, 2.0], np.ones((3, 2)), np.ones((3, 2)))
        self.assertIn("at least 3 points", str(ctx.exception))

        # Case 3: 2x2 grid
        with self.assertRaises(ValueError) as ctx:
            IsoFrontierAudit([1.0, 2.0], [1.0, 2.0], np.ones((2, 2)), np.ones((2, 2)))
        self.assertIn("at least 3 points", str(ctx.exception))

        # Case 4: 3x3 grid with kx=1
        with self.assertRaises(ValueError) as ctx:
            IsoFrontierAudit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], np.ones((3, 3)), np.ones((3, 3)), kx=1)
        self.assertIn("degree kx, ky >= 2", str(ctx.exception))

        # Case 5: 3x3 grid with ky=1
        with self.assertRaises(ValueError) as ctx:
            IsoFrontierAudit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], np.ones((3, 3)), np.ones((3, 3)), ky=1)
        self.assertIn("degree kx, ky >= 2", str(ctx.exception))

    def test_minimal_admissible_3x3_grid_gradient_evaluation(self):
        """Verify that the minimal admissible grid (3x3 with kx=2, ky=2) evaluates partial derivatives cleanly without FITPACK parder code 10 crash."""
        s_grid = [1.0, 2.0, 3.0]
        g_grid = [1.0, 2.0, 3.0]
        P = np.array([
            [10.0, 12.0, 15.0],
            [12.0, 15.0, 19.0],
            [15.0, 19.0, 24.0]
        ])
        H = np.array([
            [4.0, 4.2, 4.5],
            [4.2, 4.5, 4.9],
            [4.5, 4.9, 5.4]
        ])
        M = np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5]
        ])
        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m": M}, kx=2, ky=2)
        grad_p, grad_h = audit.get_gradients(2.0, 2.0)
        grad_m = audit.get_secondary_gradient(2.0, 2.0, "m")

        self.assertEqual(len(grad_p), 2)
        self.assertEqual(len(grad_h), 2)
        self.assertEqual(len(grad_m), 2)
        self.assertTrue(np.all(np.isfinite(grad_p)))
        self.assertTrue(np.all(np.isfinite(grad_h)))
        self.assertTrue(np.all(np.isfinite(grad_m)))

    def test_nan_inf_fail_closed_coverage(self):
        """Verify that all inputs fail closed when contaminated with NaN or Inf."""
        clean_3 = [1.0, 2.0, 3.0]
        ones_33 = np.ones((3, 3))

        # NaN in s_grid
        with self.assertRaises(ValueError):
            IsoFrontierAudit([1.0, np.nan, 3.0], clean_3, ones_33, ones_33)

        # Inf in g_grid
        with self.assertRaises(ValueError):
            IsoFrontierAudit(clean_3, [1.0, np.inf, 3.0], ones_33, ones_33)

        # Non-increasing s_grid
        with self.assertRaises(ValueError):
            IsoFrontierAudit([1.0, 3.0, 2.0], clean_3, ones_33, ones_33)

        # NaN in add_metric_grid
        audit = IsoFrontierAudit(clean_3, clean_3, ones_33, ones_33)
        bad_m = np.ones((3, 3))
        bad_m[1, 1] = np.nan
        with self.assertRaises(ValueError) as ctx:
            audit.add_metric_grid("m_bad", bad_m)
        self.assertIn("contains NaN or Inf", str(ctx.exception))


class TestReauditMoorePenroseProjection(unittest.TestCase):
    """Deep verification of Moore-Penrose pseudo-inverse row-space projection in jacobian_rank_test."""

    def test_flat_primary_with_divergent_secondary(self):
        """
        Verify that when primary surfaces have exactly zero gradient,
        Moore-Penrose projection gives ortho_ratio = 1.0 and is_null = False.
        """
        s_grid = np.linspace(1.0, 4.0, 5)
        g_grid = np.linspace(1.0, 4.0, 5)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = np.full_like(S, 20.0)
        H = np.full_like(S, 5.0)
        M = 2.0 * S + 3.0 * G

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m": M}, kx=2, ky=2)
        audit.get_gradients = lambda s, g: (np.array([0.0, 0.0]), np.array([0.0, 0.0]))

        res = audit.jacobian_rank_test(2.0, 2.0, "m")
        self.assertEqual(res.ortho_ratio, 1.0)
        self.assertFalse(res.is_null)

    def test_zero_grad_p_with_active_grad_h(self):
        """
        Verify that when grad_p = 0 but grad_h != 0,
        Moore-Penrose projection projects onto row(grad_h).
        If grad_m is orthogonal to grad_h, ortho_ratio = 1.0.
        If grad_m is parallel to grad_h, ortho_ratio = 0.0.
        """
        s_grid = np.linspace(1.0, 4.0, 5)
        g_grid = np.linspace(1.0, 4.0, 5)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = np.full_like(S, 20.0)
        H = 4.0 * S  # grad_h = [4.0, 0.0]

        # Secondary metric 1: orthogonal to grad_h -> M_ortho = 2.0 * G (grad_m = [0.0, 2.0])
        # Secondary metric 2: parallel to grad_h -> M_par = 3.0 * S (grad_m = [3.0, 0.0])
        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m_ortho": 2.0 * G, "m_par": 3.0 * S}, kx=2, ky=2)
        audit.get_gradients = lambda s, g: (np.array([0.0, 0.0]), np.array([4.0, 0.0]))

        res_ortho = audit.jacobian_rank_test(2.0, 2.0, "m_ortho")
        self.assertEqual(res_ortho.ortho_ratio, 1.0)
        self.assertFalse(res_ortho.is_null)

        res_par = audit.jacobian_rank_test(2.0, 2.0, "m_par")
        self.assertAlmostEqual(res_par.ortho_ratio, 0.0, places=6)
        self.assertTrue(res_par.is_null)

    def test_exact_subspace_orthogonal_decomposition(self):
        """
        Verify exact theoretical decomposition:
        grad_p = [1, 2], grad_h = [2, 4] (rank 1 manifold along u = [1, 2]/sqrt(5)).
        grad_m = [0, 5].
        Parallel component: (grad_m . u) * u = (10/sqrt(5)) * [1, 2]/sqrt(5) = [2, 4].
        Perpendicular component: [0, 5] - [2, 4] = [-2, 1].
        norm(grad_m_perp) = sqrt(5).
        norm(grad_m) = 5.
        ortho_ratio = sqrt(5) / 5 = 1 / sqrt(5) ~ 0.4472136.
        """
        s_grid = np.linspace(1.0, 4.0, 5)
        g_grid = np.linspace(1.0, 4.0, 5)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = 1.0 * S + 2.0 * G
        H = 2.0 * S + 4.0 * G
        M = 5.0 * G

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m": M}, kx=2, ky=2)
        res = audit.jacobian_rank_test(2.0, 2.0, "m")

        expected_ortho_ratio = 1.0 / np.sqrt(5.0)
        self.assertAlmostEqual(res.ortho_ratio, expected_ortho_ratio, places=5)
        self.assertFalse(res.is_null)


class TestReauditTangentSignsAndFTC(unittest.TestCase):
    """Deep verification of traversal-consistent tangent signs and Fundamental Theorem of Calculus."""

    def test_forward_and_reverse_curve_ftc_identity(self):
        """
        Verify that for both forward (s_min -> s_max) and reverse (s_max -> s_min) paths,
        integral(kappa_m dl) == m(end) - m(start).
        Specifically, delta_m_reverse == -delta_m_forward.
        """
        s_grid = np.linspace(1.0, 5.0, 15)
        g_grid = np.linspace(1.0, 5.0, 15)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")

        # P(s, g) = 20 + 2*S - 3*G  => level curve g(s) = (20 + 2s - P)/3
        # Secondary metric m(s, g) = 0.5 * S^2 + 0.25 * G^2
        P = 20.0 + 2.0 * S - 3.0 * G
        H = 5.0 + 0.1 * S - 0.15 * G
        M = 0.5 * S**2 + 0.25 * G**2

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m": M}, kx=3, ky=3)

        # Forward traversal: s from 2.0 to 4.0
        curve_fwd = audit.trace_iso_curve(20.0, s_range=(2.0, 4.0), n_pts=25)
        prof_fwd = audit.compute_kappa_along_curve(curve_fwd, "m")
        direct_fwd = audit.eval_metric(curve_fwd.s[-1], curve_fwd.g[-1], "m") - audit.eval_metric(curve_fwd.s[0], curve_fwd.g[0], "m")

        self.assertAlmostEqual(prof_fwd["delta_m_integrated"], direct_fwd, places=4)

        # Reverse traversal: s from 4.0 to 2.0
        curve_rev = audit.trace_iso_curve(20.0, s_range=(4.0, 2.0), n_pts=25)
        prof_rev = audit.compute_kappa_along_curve(curve_rev, "m")
        direct_rev = audit.eval_metric(curve_rev.s[-1], curve_rev.g[-1], "m") - audit.eval_metric(curve_rev.s[0], curve_rev.g[0], "m")

        self.assertAlmostEqual(prof_rev["delta_m_integrated"], direct_rev, places=4)
        self.assertAlmostEqual(direct_rev, -direct_fwd, places=4)
        self.assertAlmostEqual(prof_rev["delta_m_integrated"], -prof_fwd["delta_m_integrated"], places=4)

    def test_vertical_level_curve_tangents(self):
        """
        Verify unit tangent computation when dP/dg = 0 (vertical level curve where tangent is along g).
        """
        s_grid = np.linspace(1.0, 4.0, 5)
        g_grid = np.linspace(1.0, 4.0, 5)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = 3.0 * S  # dP/ds = 3, dP/dg = 0
        H = 1.0 * S
        audit = IsoFrontierAudit(s_grid, g_grid, P, H, kx=2, ky=2)

        u_fwd = audit.compute_unit_tangent(2.0, 2.0, forward_s=True)
        u_rev = audit.compute_unit_tangent(2.0, 2.0, forward_s=False)

        # Tangent is orthogonal to [3, 0], so it must be [0, 1] or [0, -1]
        self.assertAlmostEqual(u_fwd[0], 0.0, places=10)
        self.assertAlmostEqual(abs(u_fwd[1]), 1.0, places=10)
        self.assertAlmostEqual(u_rev[0], 0.0, places=10)
        self.assertAlmostEqual(abs(u_rev[1]), 1.0, places=10)
        # Orientation tie-breaker: forward_s gives u_g >= 0, reverse gives u_g <= 0
        self.assertGreaterEqual(u_fwd[1], 0.0)
        self.assertLessEqual(u_rev[1], 0.0)


class TestReauditSaddlePointContinuity(unittest.TestCase):
    """Deep verification of multi-branch continuity tracking at saddle points."""

    def test_saddle_point_no_branch_hopping(self):
        """
        Construct a saddle surface P(s, g) = 25.0 - (s - 2.5)^2 + (g - 2.5)^2.
        Level set P = 25.0 corresponds to (g - 2.5)^2 = (s - 2.5)^2 => g = 2.5 +/- (s - 2.5).
        Branch 1: g = s (slope +1).
        Branch 2: g = 5.0 - s (slope -1).
        Tracing starting at s=1.5 on Branch 1 (g=1.5) must follow g=s through s=2.5 to s=3.5 (g=3.5),
        never jumping to Branch 2 (which would jump to g=1.5 at s=3.5).
        """
        s_grid = np.linspace(1.0, 4.0, 25)
        g_grid = np.linspace(1.0, 4.0, 25)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = 25.0 - (S - 2.5)**2 + (G - 2.5)**2
        H = 5.0 + 0.05 * P
        M = 0.2 * S + 0.3 * G

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m": M}, kx=3, ky=3)

        curve = audit.trace_iso_curve(25.0, s_range=(1.5, 3.5), n_pts=21)

        # Check endpoints and monotonic progression along branch 1: g(s) should be approximately s
        self.assertAlmostEqual(curve.g[0], 1.5, delta=0.05)
        self.assertAlmostEqual(curve.g[-1], 3.5, delta=0.05)

        # Step-to-step diffs must be strictly smooth (no sudden hop)
        g_diffs = np.abs(np.diff(curve.g))
        self.assertTrue(np.all(g_diffs < 0.15), f"Max diff was {np.max(g_diffs)}")

        # Check FTC
        profile = audit.compute_kappa_along_curve(curve, "m")
        delta_m_integrated = profile["delta_m_integrated"]
        delta_m_direct = audit.eval_metric(curve.s[-1], curve.g[-1], "m") - audit.eval_metric(curve.s[0], curve.g[0], "m")
        self.assertLess(abs(delta_m_integrated - delta_m_direct), 0.005)


class TestReauditDisconnectedGapDetection(unittest.TestCase):
    """Deep verification of fail-closed disconnected gap and domain exit detection."""

    def test_domain_exit_raises_runtime_error(self):
        """
        When a level curve exits the g bounds [g_low, g_high] for an intermediate s,
        trace_iso_curve must immediately raise RuntimeError, never silently skipping or bridging.
        """
        s_grid = np.linspace(1.0, 5.0, 15)
        g_grid = np.linspace(1.0, 4.0, 15)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")

        # P(s, g) has a steep ridge at s in [2.5, 3.5] where minimum P is > 30.
        P = 15.0 + 2.0 * G + 30.0 * np.exp(-((S - 3.0) / 0.3)**2)
        H = 5.0 + 0.1 * S
        audit = IsoFrontierAudit(s_grid, g_grid, P, H)

        # Target P = 18.0 exists for s < 2.3 and s > 3.7 within G in [1, 4],
        # but does NOT exist in between.
        with self.assertRaises(RuntimeError) as ctx:
            audit.trace_iso_curve(18.0, s_range=(1.5, 4.5), n_pts=25)
        self.assertIn("disconnected or exits domain", str(ctx.exception))

    def test_target_completely_outside_range(self):
        """When p_target does not exist anywhere on the grid, fail closed."""
        s_grid = np.linspace(1.0, 4.0, 5)
        g_grid = np.linspace(1.0, 4.0, 5)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = 20.0 + S + G
        H = 5.0 + S
        audit = IsoFrontierAudit(s_grid, g_grid, P, H, kx=2, ky=2)

        with self.assertRaises(RuntimeError):
            audit.trace_iso_curve(100.0, s_range=(1.5, 3.5), n_pts=10)


if __name__ == "__main__":
    unittest.main()
