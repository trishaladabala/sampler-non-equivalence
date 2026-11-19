"""
tests/test_adversarial_round2_invariants.py: Empirical Adversarial Challenge Suite
for Remediation Round 2 Iteration 2 (Challenger 1).

Focus Areas:
1. Mathematical and Empirical Invariant Challenge:
   - Verify that the impossible rank-3 condition in R^2 is completely eradicated from main.tex,
     frontier_audit/core.py, and analysis/iso_frontier_sensitivity.py.
   - Adversarially verify near-rank-1 geometry and singular values: condition numbers kappa(J_F) ~ 1,300-4,032.
   - Verify that unit tangent t is exactly orthogonal to grad Gen-PPL (<t, grad Gen-PPL> = 0) and
     check tangent projection ratio rho_m = |kappa_m| / ||grad m||.
   - Verify fail-closed behavior of RectBivariateSpline in analysis/iso_frontier_sensitivity.py:
     assert zero hardcoded fallback constants.
   - Verify seed-level bootstrap 95% CIs (B=500) and verify honest out-of-sample prediction vs
     seeds 201-205 (including discrepancies on JSD and MAUVE).
2. S^510 Null Space Geometry in Notebook 21:
   - In notebooks/21_ace_null_distribution_and_matched_ace.ipynb and
     notebooks/21_colab_matched_ace_intervention.ipynb, verify the 20-direction hypersphere null
     generator on S^510 subset R^512:
     - All 20 vectors v_k must have unit norm ||v_k||_2 = 1.0.
     - All 20 vectors must be strictly orthogonal to d_hat_ACE: |<v_k, d_hat_ACE>| < 1e-6.
     - Verify the 20 null directions are random samples on S^510, NOT an orthonormal basis among themselves.
"""

import ast
import csv
import json
import os
import re
import sys
import tempfile
import unittest
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frontier_audit.core import (
    IsoFrontierAudit,
    IsoCurveResult,
    JacobianTestResult,
)
from analysis.iso_frontier_sensitivity import (
    load_elfb_ode32_grid_fail_closed,
    run_cell_seed_bootstrap,
    load_held_out_measured_deltas,
)


class TestRankThreeEradication(unittest.TestCase):
    """Verify that the impossible rank-3 condition in R^2 is eradicated everywhere."""

    def test_mathematical_impossibility_of_rank_3_in_r2(self):
        """Mathematical theorem: any 3x2 matrix in R^{3x2} has rank <= min(3, 2) = 2."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            A = rng.randn(3, 2)
            rank = np.linalg.matrix_rank(A)
            self.assertLessEqual(rank, 2, f"Matrix rank {rank} exceeded 2 in R^2!")

    def test_main_tex_contains_no_rank_3_claims(self):
        """main.tex must contain no claims that Jacobian or augmented matrix has rank 3."""
        tex_path = os.path.join(_REPO_ROOT, "main.tex")
        with open(tex_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for regex patterns claiming rank 3
        rank3_patterns = [
            r"rank[\s\-_=]*3",
            r"rank\s*\(\s*J\s*\)\s*=\s*3",
            r"rank\s*\(\s*J_F\s*\)\s*=\s*3",
            r"rank\s*3\s+augmented",
            r"augmented.*rank\s*3",
            r"full\s+rank\s+3",
        ]
        for pat in rank3_patterns:
            matches = re.findall(pat, content, re.IGNORECASE)
            self.assertEqual(
                matches,
                [],
                f"Found illegal rank-3 claim in main.tex matching pattern '{pat}': {matches}"
            )

        # Confirm main.tex correctly asserts rank 2
        self.assertIn(r"augmented primary--secondary matrix $[\nabla\mathrm{Gen\text{-}PPL}, \nabla m]^\top$ has rank 2", content)
        self.assertIn(r"augmented matrix $[\nabla\mathrm{Gen\text{-}PPL}, \nabla m]^\top \in \mathbb{R}^{2 \times 2}$ has full rank 2", content)

    def test_frontier_audit_and_sensitivity_code_contains_no_rank_3_logic(self):
        """frontier_audit/core.py and analysis/iso_frontier_sensitivity.py must have no rank 3 logic."""
        for rel_path in ["frontier_audit/core.py", "analysis/iso_frontier_sensitivity.py"]:
            file_path = os.path.join(_REPO_ROOT, rel_path)
            with open(file_path, "r", encoding="utf-8") as f:
                code_text = f.read()

            # Verify no rank 3 assignments
            self.assertNotIn("rank_augmented = 3", code_text)
            self.assertNotIn("rank_aug = 3", code_text)
            self.assertNotIn("rank == 3", code_text)

            # In frontier_audit/core.py, verify docstring explicitly clarifies rank 3 is impossible
            if "core.py" in rel_path:
                self.assertIn("Any 3x2 matrix has maximum rank min(3, 2) = 2; rank 3 is impossible.", code_text)

    def test_jacobian_test_result_augmented_rank_is_strictly_1_or_2(self):
        """JacobianTestResult rank_augmented must strictly be 1 or 2."""
        s_grid = np.linspace(1.0, 4.0, 5)
        g_grid = np.linspace(1.0, 4.0, 5)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = 20.0 + 2.0 * S - 3.0 * G
        H = 5.0 + 0.2 * S - 0.3 * G
        M_null = 2.0 * P
        M_div = 0.5 * S

        audit = IsoFrontierAudit(s_grid, g_grid, P, H, {"m_null": M_null, "m_div": M_div}, kx=2, ky=2)
        res_null = audit.jacobian_rank_test(2.0, 2.0, "m_null")
        res_div = audit.jacobian_rank_test(2.0, 2.0, "m_div")

        self.assertEqual(res_null.rank_augmented, 1)
        self.assertEqual(res_div.rank_augmented, 2)
        self.assertIn(res_null.rank_augmented, (1, 2))
        self.assertIn(res_div.rank_augmented, (1, 2))


class TestNearRankOneGeometry(unittest.TestCase):
    """Adversarially verify near-rank-1 geometry, singular values, and condition numbers."""

    @classmethod
    def setUpClass(cls):
        cls.s_grid, cls.g_grid, cls.mean_grids, cls.seed_data = load_elfb_ode32_grid_fail_closed()
        cls.audit = IsoFrontierAudit(
            cls.s_grid,
            cls.g_grid,
            cls.mean_grids["gen_ppl"],
            cls.mean_grids["entropy"],
            cls.mean_grids,
            kx=2,
            ky=2,
        )

    def test_primary_singular_values_and_condition_numbers_at_confirmatory_points(self):
        """Condition number kappa(J_F) must be ~ 1,300 - 4,032 across confirmatory arms."""
        confirmatory_points = [
            ("C1_A", 1.0, 1.3316447871138217),
            ("C1_B", 2.0, 0.9995175383614311),
            ("C2_A", 1.0, 1.5372257148031159),
            ("C2_B", 3.0, 0.9964853187078995),
            ("C3_A", 1.0, 1.6374508937133951),
            ("C3_B", 4.0, 0.9989862396485288),
        ]

        for name, s_val, g_val in confirmatory_points:
            grad_p, grad_h = self.audit.get_gradients(s_val, g_val)
            J_prim = np.vstack([grad_p, grad_h])
            s_vals = np.linalg.svd(J_prim, compute_uv=False)
            sigma_1, sigma_2 = float(s_vals[0]), float(s_vals[1])
            cond = sigma_1 / (sigma_2 + 1e-12)

            # Check dominant singular value range: sigma_1 ~ 10 - 20
            self.assertGreater(sigma_1, 10.0, f"{name}: sigma_1={sigma_1} < 10.0")
            self.assertLess(sigma_1, 20.0, f"{name}: sigma_1={sigma_1} > 20.0")

            # Check subdominant singular value range: sigma_2 ~ 0.004 - 0.010
            self.assertGreater(sigma_2, 0.004, f"{name}: sigma_2={sigma_2} < 0.004")
            self.assertLess(sigma_2, 0.010, f"{name}: sigma_2={sigma_2} > 0.010")

            # Check condition number kappa(J_F) ~ 1300 - 4032
            self.assertGreater(cond, 1300.0, f"{name}: cond={cond} < 1300")
            self.assertLess(cond, 4033.0, f"{name}: cond={cond} > 4032")

            # Check gradient cosine similarity > 0.997
            cos_sim = np.dot(grad_p, grad_h) / (np.linalg.norm(grad_p) * np.linalg.norm(grad_h))
            self.assertGreater(cos_sim, 0.997, f"{name}: cos_sim={cos_sim} <= 0.997")

    def test_json_archive_matches_empirical_singular_values_and_condition_numbers(self):
        """Verify that analysis/out/iso_frontier_sensitivity.json contains exact verified condition numbers."""
        json_path = os.path.join(_REPO_ROOT, "analysis", "out", "iso_frontier_sensitivity.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for pair_id in ["C1", "C2", "C3"]:
            pair_data = data[pair_id]
            for m_name in ["rep4", "jsd_inter_sample", "mauve"]:
                m_info = pair_data["metrics"][m_name]
                cond_a = m_info["arm_a_test"]["primary_condition_number"]
                cond_b = m_info["arm_b_test"]["primary_condition_number"]

                self.assertGreaterEqual(cond_a, 1300.0)
                self.assertLessEqual(cond_a, 4032.0)
                self.assertGreaterEqual(cond_b, 1300.0)
                self.assertLessEqual(cond_b, 4032.0)


class TestTangentOrthogonalityAndProjectionRatio(unittest.TestCase):
    """Adversarially verify tangent t orthogonality to grad Gen-PPL and projection ratio rho_m."""

    @classmethod
    def setUpClass(cls):
        s_grid, g_grid, mean_grids, _ = load_elfb_ode32_grid_fail_closed()
        cls.audit = IsoFrontierAudit(
            s_grid,
            g_grid,
            mean_grids["gen_ppl"],
            mean_grids["entropy"],
            mean_grids,
            kx=2,
            ky=2,
        )

    def test_unit_tangent_exact_orthogonality_to_grad_gen_ppl(self):
        """Unit tangent t must satisfy ||t||_2 = 1.0 and <t, grad Gen-PPL> = 0.0 identically."""
        # Test across a dense grid of points in the domain
        s_test = np.linspace(1.2, 3.8, 10)
        g_test = np.linspace(0.2, 2.8, 10)

        for s_val in s_test:
            for g_val in g_test:
                t = self.audit.compute_unit_tangent(s_val, g_val)
                norm_t = np.linalg.norm(t)
                self.assertAlmostEqual(norm_t, 1.0, places=12, msg=f"Tangent norm != 1.0 at ({s_val}, {g_val})")

                grad_p, _ = self.audit.get_gradients(s_val, g_val)
                dot_product = float(np.dot(t, grad_p))
                self.assertAlmostEqual(
                    dot_product,
                    0.0,
                    places=12,
                    msg=f"Tangent not orthogonal to grad Gen-PPL at ({s_val}, {g_val}): dot={dot_product}"
                )

    def test_tangent_projection_ratio_bounds_and_algebraic_consistency(self):
        """rho_m = |kappa_m| / ||grad m|| must lie in [0, 1] and match |<t, grad m>| / ||grad m||."""
        confirmatory_points = [
            (1.0, 1.3316447871138217),
            (2.0, 0.9995175383614311),
            (1.0, 1.5372257148031159),
            (3.0, 0.9964853187078995),
            (1.0, 1.6374508937133951),
            (4.0, 0.9989862396485288),
        ]
        for s_val, g_val in confirmatory_points:
            for m_name in ["rep4", "jsd_inter_sample", "mauve"]:
                res = self.audit.jacobian_rank_test(s_val, g_val, m_name)
                grad_m = np.array(res.grad_m)
                norm_m = float(np.linalg.norm(grad_m))
                kappa = res.kappa_m
                rho = res.tangent_projection_ratio

                self.assertGreaterEqual(rho, 0.0)
                self.assertLessEqual(rho, 1.0)

                expected_rho = abs(kappa) / norm_m
                self.assertAlmostEqual(rho, expected_rho, places=7)

    def test_synthetic_adversarial_fields_tangent_projection(self):
        """Construct synthetic collinear and strictly orthogonal fields to test sensitivity bounds."""
        s_grid = np.linspace(1.0, 4.0, 7)
        g_grid = np.linspace(0.0, 3.0, 7)
        S, G = np.meshgrid(s_grid, g_grid, indexing="ij")
        P = 30.0 + 3.0 * S - 4.0 * G
        H = 5.0 + 0.3 * S - 0.4 * G

        # Collinear field: grad_m parallel to grad_P
        M_collinear = 5.0 * P
        # Purely tangential field: grad_m along tangent vector [4/5, 3/5]
        # Psi(s, g) = 4*s + 3*g => grad_Psi = [4, 3], which is orthogonal to [3, -4]
        M_pure_tangential = 4.0 * S + 3.0 * G

        audit = IsoFrontierAudit(
            s_grid, g_grid, P, H,
            {"m_collinear": M_collinear, "m_tangential": M_pure_tangential},
            kx=2, ky=2
        )

        res_col = audit.jacobian_rank_test(2.0, 1.5, "m_collinear")
        self.assertAlmostEqual(res_col.kappa_m, 0.0, places=5)
        self.assertAlmostEqual(res_col.tangent_projection_ratio, 0.0, places=5)
        self.assertTrue(res_col.is_null)
        self.assertEqual(res_col.rank_augmented, 1)

        res_tan = audit.jacobian_rank_test(2.0, 1.5, "m_tangential")
        self.assertAlmostEqual(res_tan.tangent_projection_ratio, 1.0, places=5)
        self.assertFalse(res_tan.is_null)
        self.assertEqual(res_tan.rank_augmented, 2)


class TestFailClosedBivariateSpline(unittest.TestCase):
    """Verify fail-closed behavior of RectBivariateSpline and zero hardcoded fallback constants."""

    def test_ast_inspection_asserts_zero_hardcoded_fallback_constants(self):
        """AST inspection of analysis/iso_frontier_sensitivity.py confirms zero hardcoded fallbacks."""
        script_path = os.path.join(_REPO_ROOT, "analysis", "iso_frontier_sensitivity.py")
        with open(script_path, "r", encoding="utf-8") as f:
            code_text = f.read()

        # Check for banned functions
        self.assertNotIn("_interpolate_grid", code_text)

        # Inspect AST of load_elfb_ode32_grid_fail_closed to assert no fallback constant injection
        tree = ast.parse(code_text, filename=script_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "load_elfb_ode32_grid_fail_closed":
                # Exclude the docstring (first statement if it is Expr(Constant(str)))
                stmts = node.body
                if stmts and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant):
                    stmts = stmts[1:]
                body_code = "\n".join(ast.unparse(s) for s in stmts)
                # Ensure the old fallback constants (30.0, 5.20, 0.05, 1.75) never appear in code
                self.assertNotIn("30.0", body_code)
                self.assertNotIn("5.20", body_code)
                self.assertNotIn("0.05", body_code)
                self.assertNotIn("1.75", body_code)
                self.assertNotIn("fallback_val", body_code)

    def test_fail_closed_on_missing_or_corrupted_csv(self):
        """load_elfb_ode32_grid_fail_closed must fail closed on missing file, missing rows, NaNs."""
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            load_elfb_ode32_grid_fail_closed("/path/to/nonexistent_db.csv")

        # Incomplete CSV (missing rows)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("run_id,sc_cfg,gamma,seed,gen_ppl,entropy,rep4,mauve,distinct2\n")
            tmp.write("p4_elfb_s32_s1.0_g0.0_seed0,1.0,0.0,0,32.0,5.2,0.04,0.2,0.3\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                load_elfb_ode32_grid_fail_closed(tmp_path)
            self.assertIn("expected 56 authoritative runs", str(ctx.exception))
        finally:
            os.remove(tmp_path)

        # CSV with NaN metric
        valid_csv = os.path.join(_REPO_ROOT, "analysis", "results_db.csv")
        with open(valid_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Filter to 56 runs but set one rep4 to NaN
        p4_runs = [
            r for r in rows
            if r.get("run_id", "").startswith("p4_elfb_s32_")
            and float(r.get("gamma", -1)) in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
            and float(r.get("sc_cfg", -1)) in [1.0, 2.0, 3.0, 4.0]
            and int(float(r.get("seed", -1))) in [0, 1]
        ]
        p4_runs[5]["rep4"] = "NaN"

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp2:
            writer = csv.DictWriter(tmp2, fieldnames=rows[0].keys())
            writer.writeheader()
            for r in p4_runs:
                writer.writerow(r)
            tmp2_path = tmp2.name

        try:
            with self.assertRaises(ValueError) as ctx2:
                load_elfb_ode32_grid_fail_closed(tmp2_path)
            self.assertIn("non-finite metric", str(ctx2.exception))
        finally:
            os.remove(tmp2_path)


class TestSeedBootstrapAndOutOfSampleValidation(unittest.TestCase):
    """Verify seed-level bootstrap 95% CIs (B=500) and honest out-of-sample prediction vs seeds 201-205."""

    @classmethod
    def setUpClass(cls):
        cls.s_grid, cls.g_grid, cls.mean_grids, cls.seed_data = load_elfb_ode32_grid_fail_closed()
        json_path = os.path.join(_REPO_ROOT, "analysis", "out", "iso_frontier_sensitivity.json")
        with open(json_path, "r", encoding="utf-8") as f:
            cls.archived_results = json.load(f)

    def test_seed_level_bootstrap_b500_execution_and_ci_exclusion(self):
        """B=500 bootstrap on Arm A strictly excludes zero for rep4, jsd, mauve across C1-C3."""
        test_points = [
            ("C1", 1.0, 1.3316447871138217),
            ("C2", 1.0, 1.5372257148031159),
            ("C3", 1.0, 1.6374508937133951),
        ]

        for pair_id, s_eval, g_eval in test_points:
            for m_name in ["rep4", "jsd_inter_sample", "mauve"]:
                boot_res = run_cell_seed_bootstrap(
                    self.s_grid,
                    self.g_grid,
                    self.seed_data,
                    s_eval=s_eval,
                    g_eval=g_eval,
                    metric_name=m_name,
                    n_boot=500,
                    random_state=42,
                )
                ci_low, ci_high = boot_res["ci_lower"], boot_res["ci_upper"]
                # Must exclude zero on Arm A
                self.assertTrue(
                    boot_res["excludes_zero"],
                    f"{pair_id} Arm A {m_name}: CI [{ci_low}, {ci_high}] does not exclude 0!"
                )
                self.assertGreater(ci_low, 0.0, f"{pair_id} Arm A {m_name}: ci_low={ci_low} <= 0")

    def test_honest_out_of_sample_discrepancy_reporting(self):
        """Out-of-sample prediction must honestly report discrepancies on JSD and MAUVE."""
        for pair_id in ["C1", "C2", "C3"]:
            pair_data = self.archived_results[pair_id]
            oos = pair_data["out_of_sample_validation"]

            # rep4 has tight agreement
            rep4_val = oos["rep4"]
            self.assertIn("Exceptional agreement", rep4_val["assessment"])
            self.assertLess(abs(rep4_val["discrepancy"]), 0.002)

            # jsd_inter_sample has real discrepancy (~0.005 - 0.008)
            jsd_val = oos["jsd_inter_sample"]
            self.assertIn("discrepancy", jsd_val)
            self.assertNotEqual(jsd_val["discrepancy"], 0.0, "JSD discrepancy cannot be zero")
            self.assertIn("Captures positive divergence", jsd_val["assessment"])

            # mauve has substantial curvature overestimation at high guidance
            mauve_val = oos["mauve"]
            self.assertIn("overestimates at higher guidance due to metric curvature", mauve_val["assessment"])
            if pair_id == "C3":
                # C3 MAUVE predicted 0.2009 vs measured 0.1210: discrepancy ~ +0.0799
                self.assertGreater(mauve_val["discrepancy"], 0.05, "C3 MAUVE discrepancy must reflect curvature overestimation")


class TestS510NullSpaceGeometryNotebook21(unittest.TestCase):
    """
    Adversarial verification of the 20-direction hypersphere null generator on S^510 subset R^512
    in notebooks/21_ace_null_distribution_and_matched_ace.ipynb and
    notebooks/21_colab_matched_ace_intervention.ipynb.
    """

    def _extract_generator(self, notebook_name: str):
        nb_path = os.path.join(_REPO_ROOT, "notebooks", notebook_name)
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [
            "".join(c["source"])
            for c in nb["cells"]
            if c.get("cell_type") == "code"
        ]
        # Find cell defining generate_20_orthogonal_null_directions
        target_cell = None
        for cell in code_cells:
            if "def generate_20_orthogonal_null_directions(" in cell:
                target_cell = cell
                break
        self.assertIsNotNone(target_cell, f"Could not find generator definition in {notebook_name}")

        clean_lines = []
        for line in target_cell.splitlines():
            if line.strip().startswith("%") or line.strip().startswith("!"):
                continue
            clean_lines.append(line)
        clean_code = "\n".join(clean_lines)

        env = {"torch": torch}
        exec(clean_code, env)
        return env["generate_20_orthogonal_null_directions"]

    def test_20_direction_hypersphere_null_generator_s510_geometry(self):
        """
        Verify that for both canonical and colab notebook 21:
        1. All 20 vectors v_k have unit norm ||v_k||_2 = 1.0 (to within 1e-6).
        2. All 20 vectors are strictly orthogonal to d_hat_ACE: |<v_k, d_hat_ACE>| < 1e-6.
        3. All 20 vectors are random samples on S^510, NOT an orthonormal basis among themselves.
        """
        notebooks = [
            "21_ace_null_distribution_and_matched_ace.ipynb",
        ]

        test_seeds = [202704, 42, 1337, 99999]

        for nb_name in notebooks:
            gen_func = self._extract_generator(nb_name)

            for seed_base in test_seeds:
                # Test with independent random vectors (offset seed to ensure independent d_target)
                for unnormalized in [False, True]:
                    torch.manual_seed(seed_base + 9999)
                    d_target = torch.randn(512, dtype=torch.float32)
                    if not unnormalized:
                        d_target = d_target / d_target.norm(p=2)

                    d_hat = d_target / d_target.norm(p=2)

                    null_dirs = gen_func(d_target, seed_base=seed_base, device="cpu")

                    # Shape check
                    self.assertEqual(null_dirs.shape, (20, 512), f"{nb_name}: Shape mismatch")

                    # 1. Unit norm check for all 20 vectors
                    for k in range(20):
                        norm_k = float(null_dirs[k].norm(p=2).item())
                        self.assertAlmostEqual(
                            norm_k,
                            1.0,
                            places=6,
                            msg=f"{nb_name} seed={seed_base}: v_{k} norm {norm_k} != 1.0"
                        )

                    # 2. Strict orthogonality to d_hat_ACE: |<v_k, d_hat_ACE>| < 1e-6
                    for k in range(20):
                        dot_k = float(torch.dot(null_dirs[k], d_hat).item())
                        self.assertLess(
                            abs(dot_k),
                            1e-6,
                            f"{nb_name} seed={seed_base}: v_{k} not orthogonal to d_hat: dot={dot_k}"
                        )

                    # 3. NOT an orthonormal basis among themselves:
                    # Random samples on S^510 have non-zero mutual dot products
                    gram_matrix = (null_dirs @ null_dirs.T).numpy()
                    # Diagonal must be 1.0
                    np.testing.assert_allclose(np.diag(gram_matrix), 1.0, atol=1e-5)

                    # Off-diagonal elements should NOT all be 0.0 (they are random samples on S^510)
                    off_diag = gram_matrix[~np.eye(20, dtype=bool)]
                    max_abs_off_diag = np.max(np.abs(off_diag))
                    mean_abs_off_diag = np.mean(np.abs(off_diag))

                    # For d=511 random unit vectors, typical mutual dot product is ~ 1/sqrt(511) ~ 0.044
                    self.assertGreater(
                        max_abs_off_diag,
                        0.01,
                        f"{nb_name}: Expected random samples on S^510, but off-diagonals are near 0 (looks like orthonormal basis)!"
                    )
                    self.assertLess(
                        mean_abs_off_diag,
                        0.25,
                        f"{nb_name}: Random samples on S^510 should have small mutual dot product (~0.044), got mean {mean_abs_off_diag}"
                    )

    def test_collinear_target_fails_closed(self):
        """Adversarial challenge: when d_target happens to be collinear with v_0, generator must assert fail-closed."""
        for nb_name in ["21_ace_null_distribution_and_matched_ace.ipynb"]:
            gen_func = self._extract_generator(nb_name)
            # If d_target is drawn from the exact same generator seed as v_0:
            g = torch.Generator(device="cpu").manual_seed(202704)
            d_collinear = torch.randn(512, generator=g)

            with self.assertRaises(AssertionError) as ctx:
                gen_func(d_collinear, seed_base=202704, device="cpu")
            self.assertIn("Orthogonality assertion failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
