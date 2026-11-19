"""
tests/test_iso_frontier.py: Unit tests for IsoFrontierAudit and differential geometry tools.

Verifies:
1. Bivariate spline evaluation and analytical partial derivatives.
2. Analytically known level curves and unit tangent vectors.
3. Jacobian rank-2 test for null (functionally dependent) vs divergent secondary metrics.
4. Fundamental theorem of calculus along the iso-curve (integrated kappa matches delta m).
5. Multi-seed bootstrap confidence interval coverage on synthetic samplers.
"""

import math
import os
import sys
import unittest
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier_audit import IsoFrontierAudit, IsoCurveResult, JacobianTestResult


class TestIsoFrontierAudit(unittest.TestCase):
    """Test suite for IsoFrontierAudit numerical algorithms."""

    def setUp(self):
        # Set fixed seed for reproducibility
        self.rng = np.random.RandomState(1337)

    def test_rect_bivariate_spline_accuracy(self):
        """Verify spline fitting and exact gradient evaluation against analytical polynomials."""
        # Domain: s in [1.0, 4.0], gamma in [1.0, 3.0]
        s_grid = np.linspace(1.0, 4.0, 9)
        g_grid = np.linspace(1.0, 3.0, 9)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")

        # Analytical quadratic polynomial surface
        # P(s, g) = 2*s^2 - 3*g^2 + s*g + 15
        # dP/ds = 4*s + g
        # dP/dg = -6*g + s
        P = 2.0 * S**2 - 3.0 * G**2 + S * G + 15.0
        H = 0.5 * P + 2.0

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, kx=3, ky=3)

        # Test points inside the domain
        test_pts = [(1.5, 1.8), (2.2, 2.5), (3.1, 2.0)]
        for s_t, g_t in test_pts:
            p_eval = audit.eval_ppl(s_t, g_t)
            p_true = 2.0 * s_t**2 - 3.0 * g_t**2 + s_t * g_t + 15.0
            self.assertAlmostEqual(p_eval, p_true, places=4, msg="PPL evaluation error")

            grad_p, _ = audit.get_gradients(s_t, g_t)
            dp_ds_true = 4.0 * s_t + g_t
            dp_dg_true = -6.0 * g_t + s_t

            self.assertAlmostEqual(grad_p[0], dp_ds_true, places=3, msg="dP/ds gradient error")
            self.assertAlmostEqual(grad_p[1], dp_dg_true, places=3, msg="dP/dg gradient error")

    def test_analytical_level_curve_tracing(self):
        """Verify level curve tracing against analytical hyperbolic level set P(s, g) = s * g = C."""
        s_grid = np.linspace(1.0, 4.0, 15)
        g_grid = np.linspace(1.0, 4.0, 15)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")

        # P(s, g) = s * g
        P = S * G
        H = np.log(P) + 4.0

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, kx=3, ky=3)

        # Trace level set P = 4.0 => g(s) = 4.0 / s
        p_target = 4.0
        curve = audit.trace_iso_curve(p_target, s_range=(1.5, 3.5), n_pts=20)

        for s_i, g_i in zip(curve.s, curve.g):
            g_expected = 4.0 / s_i
            self.assertAlmostEqual(g_i, g_expected, places=4, msg=f"Mismatch on level set at s={s_i}")

        # Verify unit tangents: along g = C/s, dg/ds = -C/s^2 = -g/s
        # Unit tangent vector: [1, -g/s] / sqrt(1 + (g/s)^2)
        for s_i, g_i, u_i in zip(curve.s, curve.g, curve.tangents):
            slope = -g_i / s_i
            norm = math.sqrt(1.0 + slope**2)
            u_expected = np.array([1.0 / norm, slope / norm])
            np.testing.assert_allclose(u_i, u_expected, atol=1e-3, err_msg="Tangent vector error")

    def test_jacobian_rank_test_null_vs_divergent(self):
        """
        Verify that Jacobian rank test and kappa detect:
        - Null secondary metric (collinear with primary response): kappa = 0, is_null = True
        - Divergent secondary metric (varies along level set): kappa != 0, is_null = False
        """
        s_grid = np.linspace(1.0, 4.0, 11)
        g_grid = np.linspace(1.0, 3.0, 11)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")

        # Linear response surface:
        # P(s, g) = 20.0 + 3.0 * s - 4.0 * g
        # grad_P = [3.0, -4.0]
        # Tangent along level curve is orthogonal to grad_P:
        # u_hat = [4.0/5.0, 3.0/5.0] = [0.8, 0.6]
        P = 20.0 + 3.0 * S - 4.0 * G
        H = 5.0 + 0.3 * S - 0.4 * G  # perfectly collinear with P

        # Metric 1 (Null): m_null(s, g) = (P(s, g) / 20.0)^2
        # grad_m_null is strictly parallel to grad_P.
        # Directional derivative along tangent must be zero.
        M_null = (P / 20.0) ** 2

        # Metric 2 (Divergent): m_div(s, g) = 0.05 * s
        # grad_m_div = [0.05, 0.0]
        # kappa = grad_m_div . u_hat = 0.05 * 0.8 = 0.040
        M_div = 0.05 * S

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m_null": M_null, "m_div": M_div}, kx=2, ky=2)

        s_eval = 2.5
        g_eval = 2.0

        # Evaluate Null Metric
        res_null = audit.jacobian_rank_test(s_eval, g_eval, "m_null")
        self.assertTrue(res_null.is_null, "Null metric must be flagged as is_null=True")
        self.assertAlmostEqual(res_null.ortho_ratio, 0.0, places=3, msg="Null metric ortho_ratio must be ~0")
        self.assertAlmostEqual(res_null.kappa_m, 0.0, places=3, msg="Null metric kappa must be ~0")

        # Evaluate Divergent Metric
        res_div = audit.jacobian_rank_test(s_eval, g_eval, "m_div")
        self.assertFalse(res_div.is_null, "Divergent metric must be flagged as is_null=False")
        self.assertGreater(res_div.ortho_ratio, 0.5, "Divergent metric ortho_ratio must be substantial")
        self.assertAlmostEqual(res_div.kappa_m, 0.040, places=3, msg="Divergent metric kappa must match 0.040")

    def test_curve_integration_matches_total_delta(self):
        """Verify Fundamental Theorem of Calculus: integral(kappa dl) == m(end) - m(start)."""
        s_grid = np.linspace(1.0, 4.0, 15)
        g_grid = np.linspace(1.0, 3.0, 15)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")

        P = 30.0 + 2.0 * S - 3.0 * G
        H = 5.20 + 0.1 * S - 0.15 * G
        M = 0.04 + 0.02 * S + 0.01 * G**2

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"metric": M}, kx=3, ky=3)

        curve = audit.trace_iso_curve(30.0, s_range=(1.5, 3.5), n_pts=50)
        profile = audit.compute_kappa_along_curve(curve, "metric")

        # Delta m computed from direct evaluations at endpoints
        m_start = audit.eval_metric(curve.s[0], curve.g[0], "metric")
        m_end = audit.eval_metric(curve.s[-1], curve.g[-1], "metric")
        delta_m_direct = m_end - m_start

        delta_m_integrated = profile["delta_m_integrated"]
        self.assertAlmostEqual(
            delta_m_integrated,
            delta_m_direct,
            places=3,
            msg=f"Integrated kappa ({delta_m_integrated}) does not match endpoint delta ({delta_m_direct})",
        )

    def test_bootstrap_ci_coverage(self):
        """Verify that seed bootstrap confidence intervals accurately cover true analytical kappa."""
        s_grid = np.linspace(1.0, 4.0, 9)
        g_grid = np.linspace(1.0, 3.0, 9)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")

        # True base surface
        P_base = 25.0 + 2.0 * S - 4.0 * G
        H_base = 5.20 + 0.1 * S - 0.2 * G
        # u_hat = [4.0, 2.0] / sqrt(20) = [2/sqrt(5), 1/sqrt(5)] = [0.8944, 0.4472]
        # Metric: M = 0.05 * S  => grad_M = [0.05, 0] => kappa = 0.05 * 0.8944 = 0.04472
        M_base = 0.05 * S
        true_kappa = 0.05 * (4.0 / math.sqrt(20.0))

        # Generate 20 synthetic seed grids with realistic noise
        n_seeds = 20
        seed_grids = []
        for _ in range(n_seeds):
            p_noise = self.rng.normal(0.0, 0.05, size=P_base.shape)
            h_noise = self.rng.normal(0.0, 0.002, size=H_base.shape)
            m_noise = self.rng.normal(0.0, 0.0005, size=M_base.shape)
            seed_grids.append({
                "ppl": P_base + p_noise,
                "h": H_base + h_noise,
                "m": M_base + m_noise,
            })

        p_target = 23.0
        s_eval = 2.0
        ci_res = IsoFrontierAudit.bootstrap_kappa_ci(
            seed_grids,
            s_grid,
            g_grid,
            p_target=p_target,
            s_eval=s_eval,
            metric_name="m",
            n_boot=150,
            alpha=0.05,
            random_state=42,
        )

        self.assertLess(ci_res["ci_lower"], true_kappa, "95% CI lower bound must be below true kappa")
        self.assertGreater(ci_res["ci_upper"], true_kappa, "95% CI upper bound must be above true kappa")
        self.assertGreater(ci_res["ci_lower"], 0.0, "95% CI for divergent metric must exclude zero")
        self.assertAlmostEqual(ci_res["kappa_point"], true_kappa, delta=0.002, msg="Point estimate must match analytical within noise tolerance")


if __name__ == "__main__":
    unittest.main()
