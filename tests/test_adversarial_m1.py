"""
tests/test_adversarial_m1.py: Adversarial stress tests for differential geometry and numerical algorithms
in frontier_audit/core.py and analysis/iso_frontier_sensitivity.py.

Evaluates:
1. Degenerate surfaces: flat surfaces, collinear/anti-parallel gradients, zero gradients, saddle points.
2. Boundary & extreme coordinates outside grid: extrapolation, reverse tracing, disconnected curves.
3. Numerical stability: SVD singular values, Moore-Penrose pseudo-inverse vs naive projection, float underflow.
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


class TestAdversarialDegenerateSurfaces(unittest.TestCase):
    """Stress tests for degenerate surfaces, critical points, and rank deficiencies."""

    def test_minimal_grid_spline_degree_fitpack_crash(self):
        """
        Adversarial Test 1.1: Minimal grid (n_s=2 or n_g=2) or degree kx < 2, ky < 2
        must fail closed in __init__ with ValueError to prevent FITPACK parder crash (ier=10).
        """
        s_grid = [1.0, 2.0]
        g_grid = [1.0, 2.0, 3.0]
        P = np.array([[20.0, 21.0, 22.0], [22.0, 23.0, 24.0]])
        H = np.array([[5.0, 5.1, 5.2], [5.2, 5.3, 5.4]])

        # Fail closed on minimal 2-point grid
        with self.assertRaises(ValueError) as ctx:
            IsoFrontierAudit(s_grid, g_grid, P, H)
        self.assertIn("at least 3 points along each axis", str(ctx.exception))

        # Fail closed on degree kx < 2 or ky < 2
        s_grid3 = [1.0, 2.0, 3.0]
        g_grid3 = [1.0, 2.0, 3.0]
        P3 = np.ones((3, 3))
        H3 = np.ones((3, 3))
        with self.assertRaises(ValueError) as ctx2:
            IsoFrontierAudit(s_grid3, g_grid3, P3, H3, kx=1)
        self.assertIn("degree kx, ky >= 2", str(ctx2.exception))

    def test_flat_surface_zero_gradients_falsely_flags_divergent_metric_as_null(self):
        """
        Adversarial Test 1.2: When P and H have zero gradients, Moore-Penrose pseudo-inverse
        projection correctly recognizes that grad_m is orthogonal to row(J_prim) and
        yields ortho_ratio = 1.0, is_null = False.
        """
        s_grid = np.linspace(1.0, 4.0, 5)
        g_grid = np.linspace(1.0, 4.0, 5)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = np.full_like(S, 20.0)  # completely flat P
        H = np.full_like(S, 5.0)   # completely flat H
        M = 0.5 * S                # secondary metric clearly depends on s (grad_m = [0.5, 0])

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m": M}, kx=2, ky=2)
        # Mock get_gradients to return exact zero vector
        audit.get_gradients = lambda s, g: (np.array([0.0, 0.0]), np.array([0.0, 0.0]))

        res = audit.jacobian_rank_test(2.0, 2.0, "m")
        self.assertAlmostEqual(res.kappa_m, 0.5, places=5)
        self.assertEqual(res.ortho_ratio, 1.0, "Moore-Penrose projection preserves orthogonal variation on flat surface")
        self.assertFalse(res.is_null, "Divergent secondary metric correctly identified as non-null (is_null=False)")

    def test_collinear_primary_gradients_svd_rank(self):
        """
        Adversarial Test 1.3: Verify SVD behavior for collinear and anti-parallel primary gradients.
        """
        s_grid = np.linspace(1.0, 4.0, 5)
        g_grid = np.linspace(1.0, 4.0, 5)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")

        # Perfectly anti-parallel gradients: P = 2*S - 3*G, H = -4*S + 6*G
        P = 2.0 * S - 3.0 * G
        H = -4.0 * S + 6.0 * G
        M_null = 1.0 * S - 1.5 * G  # collinear with P and H
        M_div = 1.0 * S + 1.0 * G   # non-collinear

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m_null": M_null, "m_div": M_div}, kx=2, ky=2)

        res_null = audit.jacobian_rank_test(2.0, 2.0, "m_null")
        res_div = audit.jacobian_rank_test(2.0, 2.0, "m_div")

        # Primary singular values: should be rank 1 (sigma_2 ~ 0)
        s_prim = res_null.singular_values_primary
        self.assertGreater(s_prim[0], 1.0)
        self.assertLess(s_prim[1], 1e-12, "Primary manifold must have rank 1 for collinear gradients")

        # For null metric, full J singular values must also have rank 1
        s_J_null = res_null.singular_values_J
        self.assertLess(s_J_null[1], 1e-12, "Full J for collinear metric must have sigma_2 ~ 0")

        # For divergent metric, full J must have rank 2
        s_J_div = res_div.singular_values_J
        self.assertGreater(s_J_div[1], 0.1, "Full J for divergent metric must have sigma_2 > 0 (rank 2)")

    def test_saddle_point_branch_discontinuity_and_ftc_breakdown(self):
        """
        Adversarial Test 1.4: Saddle point P(s, g) = 20 + (s - 2.5)^2 - (g - 2.0)^2.
        Level set through saddle point has two intersecting branches: g = 2.0 +/- (s - 2.5).
        Continuity tracking and trajectory prediction select the consistent continuous branch,
        preventing branch hopping and preserving the Fundamental Theorem of Calculus.
        """
        s_grid = np.linspace(1.0, 4.0, 15)
        g_grid = np.linspace(1.0, 4.0, 15)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = 20.0 + (S - 2.5)**2 - (G - 2.0)**2
        H = 5.0 + 0.1 * P
        M = 0.1 * S + 0.05 * G

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m": M}, kx=3, ky=3)

        curve = audit.trace_iso_curve(20.0, s_range=(1.5, 3.5), n_pts=21)
        profile = audit.compute_kappa_along_curve(curve, "m")

        # Verify no branch hopping: step-to-step |diff(g)| is small (<= 0.2)
        g_jumps = np.abs(np.diff(curve.g))
        self.assertFalse(np.any(g_jumps > 0.2), "Continuity tracking successfully prevents branch hopping")

        # Verify FTC holds: integrated kappa matches direct endpoint evaluation
        delta_m_integrated = profile["delta_m_integrated"]
        delta_m_direct = audit.eval_metric(curve.s[-1], curve.g[-1], "m") - audit.eval_metric(curve.s[0], curve.g[0], "m")

        discrepancy = abs(delta_m_integrated - delta_m_direct)
        self.assertLess(
            discrepancy, 0.01,
            f"FTC holds with continuity tracking: discrepancy {discrepancy:.4f} < 0.01"
        )


class TestAdversarialBoundaryAndExtrapolation(unittest.TestCase):
    """Stress tests for boundary conditions, extrapolation, and curve orientation."""

    def test_reverse_direction_tangent_sign_inversion(self):
        """
        Adversarial Test 2.1: Reverse tracing (s_range=(s_high, s_low)).
        compute_unit_tangent orients tangent vectors consistent with progression
        (u_s <= 0 when s_end < s_start), ensuring integral(kappa dl) == delta_m.
        """
        s_grid = np.linspace(1.0, 4.0, 10)
        g_grid = np.linspace(1.0, 3.0, 10)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = 20.0 + 2.0 * S - 3.0 * G
        H = 5.0 + 0.1 * S - 0.15 * G
        M = 0.05 * S
        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m": M})

        # Forward trace
        curve_fwd = audit.trace_iso_curve(20.0, s_range=(1.5, 3.0), n_pts=10)
        prof_fwd = audit.compute_kappa_along_curve(curve_fwd, "m")
        direct_fwd = audit.eval_metric(curve_fwd.s[-1], curve_fwd.g[-1], "m") - audit.eval_metric(curve_fwd.s[0], curve_fwd.g[0], "m")
        self.assertAlmostEqual(prof_fwd["delta_m_integrated"], direct_fwd, places=4)

        # Reverse trace
        curve_rev = audit.trace_iso_curve(20.0, s_range=(3.0, 1.5), n_pts=10)
        prof_rev = audit.compute_kappa_along_curve(curve_rev, "m")
        direct_rev = audit.eval_metric(curve_rev.s[-1], curve_rev.g[-1], "m") - audit.eval_metric(curve_rev.s[0], curve_rev.g[0], "m")

        # In reverse trace, direct_rev is negative (-0.075), and prof_rev['delta_m_integrated'] matches
        # because tangents are oriented along the traversal direction.
        self.assertAlmostEqual(direct_rev, -direct_fwd, places=4)
        self.assertAlmostEqual(prof_rev["delta_m_integrated"], direct_rev, places=4,
                               msg="Traversal-consistent tangent correctly matches direct delta in reverse tracing")

    def test_extrapolation_spline_clamping_vs_derivative_mismatch(self):
        """
        Adversarial Test 2.2: Extreme boundary coordinates outside grid.
        FITPACK RectBivariateSpline clamps function evaluation at boundaries while
        evaluating derivatives using boundary polynomials, causing inconsistent derivatives.
        """
        s_grid = np.linspace(1.0, 4.0, 10)
        g_grid = np.linspace(1.0, 3.0, 10)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = 20.0 + np.exp(S / 2.0) - np.log(G)
        H = 5.0 + 0.1 * S - 0.15 * G
        audit = IsoFrontierAudit(s_grid, g_grid, P, H, kx=3, ky=3)

        # Evaluate at s=4.0 (boundary) vs s=10.0 (far outside boundary)
        p_bound = audit.eval_ppl(4.0, 2.0)
        p_far = audit.eval_ppl(10.0, 2.0)
        # FITPACK clamps function evaluation to boundary value
        self.assertAlmostEqual(p_bound, p_far, places=4, msg="FITPACK clamps value outside boundary")

        # But gradient evaluates non-zero boundary derivative
        grad_far, _ = audit.get_gradients(10.0, 2.0)
        self.assertGreater(abs(grad_far[0]), 1.0, "Gradient evaluates non-zero polynomial derivative")

    def test_disconnected_curve_gap_stitching(self):
        """
        Adversarial Test 2.3: When a level set temporarily exits the domain,
        trace_iso_curve fails closed and raises RuntimeError rather than silently bridging gaps.
        """
        s_grid = np.linspace(1.0, 5.0, 15)
        g_grid = np.linspace(1.0, 3.0, 15)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        # In the middle (s ~ 3.0), P has a huge hump (min P > 40).
        P = 20.0 - 2.0 * G + 30.0 * np.exp(-((S - 3.0) / 0.3)**2)
        H = 5.0 + 0.1 * S
        audit = IsoFrontierAudit(s_grid, g_grid, P, H)

        # Target P = 16.0 cannot be satisfied near s = 3.0 within G in [1, 3].
        # trace_iso_curve must fail closed with RuntimeError rather than silently stitching across the gap.
        with self.assertRaises(RuntimeError) as ctx:
            audit.trace_iso_curve(16.0, s_range=(1.0, 5.0), n_pts=20)
        self.assertIn("disconnected or exits domain", str(ctx.exception))


class TestAdversarialLinearAlgebraAndPinv(unittest.TestCase):
    """Stress tests for SVD numerical stability, Moore-Penrose pseudo-inverse, and rank tests."""

    def test_moore_penrose_projection_vs_naive_projection_under_rank_deficiency(self):
        """
        Adversarial Test 3.1: Compare Moore-Penrose pseudo-inverse projection
        P_parallel = J_prim^+ @ J_prim vs naive projection onto grad_p.
        When grad_p is zero or ill-conditioned, Moore-Penrose projection gracefully
        extracts the orthogonal subspace from grad_h, whereas naive projection fails completely.
        """
        u_prim = np.array([1.0, 2.0]) / np.sqrt(5.0)
        u_ortho = np.array([-2.0, 1.0]) / np.sqrt(5.0)

        # Case A: grad_p = 0, grad_h != 0
        grad_p_zero = np.array([0.0, 0.0])
        grad_h_valid = 1.5 * u_prim
        grad_m = 0.5 * u_prim + 0.2 * u_ortho

        # Moore-Penrose pseudo-inverse
        J_prim = np.vstack([grad_p_zero, grad_h_valid])
        pinv_J = np.linalg.pinv(J_prim)
        P_proj = pinv_J @ J_prim
        m_perp_pinv = grad_m - (P_proj @ grad_m)
        ortho_ratio_pinv = np.linalg.norm(m_perp_pinv) / np.linalg.norm(grad_m)

        # Moore-Penrose extracts exact 0.2 orthogonal component
        self.assertAlmostEqual(np.linalg.norm(m_perp_pinv), 0.2, places=7)
        self.assertAlmostEqual(ortho_ratio_pinv, 0.2 / np.linalg.norm(grad_m), places=7)

        # Current core.py naive projection
        norm_p = np.linalg.norm(grad_p_zero)
        norm_m = np.linalg.norm(grad_m)
        if norm_m > 1e-15 and norm_p > 1e-15:
            m_perp_naive = grad_m - (np.dot(grad_m, grad_p_zero) / (norm_p**2)) * grad_p_zero
            ortho_ratio_naive = np.linalg.norm(m_perp_naive) / norm_m
        else:
            ortho_ratio_naive = 0.0

        # Naive projection completely misses the 0.2 orthogonal component and outputs 0.0
        self.assertEqual(ortho_ratio_naive, 0.0, "Vulnerability confirmed: naive projection returns 0.0")

    def test_svd_rank2_detection_sensitivity_limit(self):
        """
        Adversarial Test 3.2: SVD singular value sigma_2(J) scaling with perturbation amplitude.
        Evaluates machine precision limits: when perturbation epsilon < 1e-15, SVD cannot distinguish
        perturbation from numerical noise.
        """
        u_prim = np.array([1.0, 2.0]) / np.sqrt(5.0)
        u_ortho = np.array([-2.0, 1.0]) / np.sqrt(5.0)

        grad_p = 3.0 * u_prim
        grad_h = 1.5 * u_prim

        epsilons = [1e-1, 1e-3, 1e-6, 1e-10, 1e-14, 1e-16]
        sigma2_values = []

        for eps in epsilons:
            grad_m = 0.5 * u_prim + eps * u_ortho
            J = np.vstack([grad_p, grad_h, grad_m])
            s_vals = np.linalg.svd(J, compute_uv=False)
            sigma2_values.append(s_vals[1])

        # For eps >= 1e-10, sigma_2 scales linearly with eps
        for i in range(len(epsilons) - 2):
            ratio = sigma2_values[i] / epsilons[i]
            self.assertAlmostEqual(ratio, 1.0, delta=0.05,
                                   msg=f"Linear scaling of sigma_2 with eps={epsilons[i]}")

        # At 1e-16 (below float64 machine epsilon relative to norm ~ 3.0), SVD noise floor is reached
        self.assertLess(sigma2_values[-1], 1e-15, "Noise floor reached at eps=1e-16")

    def test_nan_input_lack_of_fail_closed_validation(self):
        """
        Adversarial Test 3.3: IsoFrontierAudit rejects NaN and Inf grids fail-closed,
        enforcing data integrity.
        """
        s_grid = [1.0, 2.0, 3.0, 4.0]
        g_grid = [1.0, 2.0, 3.0, 4.0]
        P = np.ones((4, 4))
        P[1, 1] = np.nan
        H = np.ones((4, 4))

        # Must raise ValueError on NaN in ppl_grid
        with self.assertRaises(ValueError) as ctx:
            IsoFrontierAudit(s_grid, g_grid, P, H)
        self.assertIn("ppl_grid contains NaN or Inf values", str(ctx.exception))

        # Must raise ValueError on Inf in h_grid
        P_clean = np.ones((4, 4))
        H_inf = np.ones((4, 4))
        H_inf[2, 2] = np.inf
        with self.assertRaises(ValueError) as ctx2:
            IsoFrontierAudit(s_grid, g_grid, P_clean, H_inf)
        self.assertIn("h_grid contains NaN or Inf values", str(ctx2.exception))

        # Must raise ValueError on NaN in secondary metric grid
        M_nan = np.ones((4, 4))
        M_nan[0, 0] = np.nan
        with self.assertRaises(ValueError) as ctx3:
            IsoFrontierAudit(s_grid, g_grid, P_clean, np.ones((4, 4)), m_grids={"m": M_nan})
        self.assertIn("m_grid['m'] contains NaN or Inf values", str(ctx3.exception))


if __name__ == "__main__":
    unittest.main()
