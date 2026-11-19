"""
tests/test_prereg_nb16_consistency.py
Automated cross-check asserting exact consistency between prereg/PREREGISTRATION.md
and notebooks/16_preregistered_c1_c3_replication.ipynb.
"""
import json
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPreregNb16Consistency(unittest.TestCase):
    def test_constants_and_hierarchy_match(self):
        prereg_text = (REPO_ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
        nb16_data = json.loads((REPO_ROOT / "notebooks" / "16_preregistered_c1_c3_replication.ipynb").read_text(encoding="utf-8"))
        
        # 1. Assert header timestamp is OSF assigned
        self.assertIn("**Timestamp:** assigned by OSF on registration", prereg_text)
        self.assertNotIn("2026-11-05T15:00:00Z", prereg_text)
        
        # 2. Assert seeds 401-420
        self.assertIn("401, 402, 403", prereg_text)
        self.assertIn("420", prereg_text)
        
        # 3. Assert K=32 and NFE=33
        self.assertIn(r"NFE_{\mathrm{sample}} = K + 1 = 33", prereg_text)
        self.assertIn("K = 32", prereg_text)
        
        # 4. Assert scorer is gpt2-large
        self.assertIn("gpt2-large", prereg_text)
        self.assertNotIn("GPT-2 XL (1.5B)", prereg_text)
        
        # 5. Extract FROZEN_OPERATING_POINTS from Notebook 16
        code_cells = [c["source"] for c in nb16_data["cells"] if c["cell_type"] == "code"]
        full_code = "\n".join("".join(c) for c in code_cells)
        self.assertIn("1.346", full_code)
        self.assertIn("1.547", full_code)
        self.assertIn("1.651", full_code)

        # 6. Assert operating point constants match in prereg
        self.assertIn("1.346", prereg_text)
        self.assertIn("1.547", prereg_text)
        self.assertIn("1.651", prereg_text)
        self.assertIn("32.4", prereg_text)
        self.assertIn("28.9", prereg_text)
        self.assertIn("27.5", prereg_text)

        # 7. Assert margins match
        self.assertIn("0.81", prereg_text)
        self.assertIn("0.72", prereg_text)
        self.assertIn("0.69", prereg_text)
        self.assertIn("0.015", prereg_text)
        self.assertIn("0.81", full_code)
        self.assertIn("0.72", full_code)
        self.assertIn("0.69", full_code)
        self.assertIn("0.015", full_code)

        # 8. Assert seeds 401-420 in notebook 16
        self.assertIn("401", full_code)
        self.assertIn("421", full_code)  # range(401, 421)

        # 9. Assert M=1000 in both
        self.assertIn("1000", full_code)
        self.assertIn("1,000", prereg_text)

        # 10. Assert three-stream CRN in prereg
        self.assertIn("Three-Stream Common Random Numbers", prereg_text)
        self.assertIn("Cryptographic Three-Stream CRN Verification", prereg_text)

        # 11. Assert Family 1 Gate Closure and Family 3 exploratory kappa
        self.assertIn("Gate Closure & Reporting Contingency", prereg_text)
        self.assertIn("Family 3 Exploratory", prereg_text)

    def test_mauve_algorithm_settings_and_code_alignment(self):
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import inspect
        import elfx.metrics

        prereg_text = (REPO_ROOT / "prereg" / "PREREGISTRATION.md").read_text(encoding="utf-8")
        
        # 1. Assert erroneous values are absent
        self.assertNotIn("num_buckets = 500", prereg_text)
        self.assertNotIn("c = 1.0", prereg_text)
        self.assertNotIn("scaling parameter c = 1.0", prereg_text)
        
        # 2. Assert exact call and defaults are documented in preregistration
        self.assertIn("mauve.compute_mauve(p_features=ref_feats, q_features=gen_feats, verbose=False, seed=25)", prereg_text)
        self.assertIn("num_buckets = 'auto'", prereg_text)
        self.assertIn("mauve_scaling_factor = 5.0", prereg_text)
        self.assertIn("seed = 25", prereg_text)
        self.assertIn("N_{\\mathrm{ref}} = M = 1,000", prereg_text)
        
        # 3. Assert alignment with elfx.metrics.mauve_from_features implementation
        sig = inspect.signature(elfx.metrics.mauve_from_features)
        self.assertEqual(sig.parameters["seed"].default, 25)
        metrics_source = inspect.getsource(elfx.metrics.mauve_from_features)
        self.assertIn("compute_mauve", metrics_source)
        self.assertIn("p_features=ref_feats", metrics_source)
        self.assertIn("q_features=gen_feats", metrics_source)
        self.assertIn("seed=seed", metrics_source)


if __name__ == "__main__":
    unittest.main()
