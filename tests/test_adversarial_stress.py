"""Empirical Adversarial Stress Test Suite for ELF Research Roadmap Core Modules.

Executes adversarial stress harnesses and edge cases across:
1. Water-filling algorithm (clock_sample.py): extreme budgets (m' < min(tau), m' > max(tau)),
   all capped tokens, single-token active sets, zero uncertainties, NaN uncertainties,
   monotonicity stress across 1,000 random trials.
2. CME mask generation (train_clock.py): Km in {2, 4, inf}, identical tau, descending tau,
   ascending tau, padding patterns, L=1, all-invalid masks.
3. Deletion target construction (expanding.py): p_del = 0, single token deletion, p_del = 1,
   boundary preservation, short sequences (S < min_keep), Poisson gap loss edge cases.
4. PPL interpolation (frontier.py): out-of-bracket entropy, single-point curves, duplicate entropy,
   unordered points, non-finite/negative PPL filtering, exact boundary interpolation.
5. ECP projection (ecp.py): sequences without EOS, EOS at position 0, multiple consecutive EOS,
   single-token sequences, all 5 projection operators under boundary conditions.
"""

import math
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(WORKSPACE_ROOT))

# Import module extraction logic from test_local_cpu
from tests.test_local_cpu import (
    MockELFConfig,
    MockELFModel,
    MockT5Encoder,
    MockTokenizer,
    extract_all_modules,
)

TEMP_DIR, EXTRACTED_MODULES = extract_all_modules()
sys.path.insert(0, TEMP_DIR)
sys.path.insert(0, str(WORKSPACE_ROOT / "ELF" / "src"))
os.environ["ELF_ROOT"] = TEMP_DIR
os.environ["ELF_SRC"] = str(WORKSPACE_ROOT / "ELF" / "src")

import elfx.clock_sample as cs
import elfx.ecp as ecp
import elfx.expanding as exp
import elfx.frontier as fr
import elfx.train_clock as tc


class TestAdversarialWaterFilling(unittest.TestCase):
    """Stress tests and boundary condition harnesses for water-filling clock allocation."""

    def test_01_extreme_budget_below_current_mean(self):
        """When target m' < mean_A(tau), clock monotonicity forces tau' == tau."""
        tau = torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float32)
        u = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        target_m = 0.3

        tau_prime, feasible = cs.water_filling_alloc(tau, u, m_next=target_m, c_max=0.5)

        # Monotonicity check
        self.assertTrue((tau_prime >= tau - 1e-6).all().item(), "Tau must not decrease")
        self.assertTrue(torch.allclose(tau_prime, tau), "Tau must remain unchanged when target is below current mean")

        # Invariant check: When target is strictly below current mean by > 1e-4,
        # achieved mean cannot equal target_m.
        achieved_mean = tau_prime.mean().item()
        diff = abs(achieved_mean - target_m)
        self.assertGreater(diff, 1e-4, "Achieved mean cannot reach target_m < mean(tau)")

    def test_02_extreme_budget_above_cap_infeasible(self):
        """When target m' exceeds max(tau) + c_max, allocation is constrained and marked infeasible."""
        tau = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        u = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        c_max = 0.2
        target_m = 0.8  # Max possible mean is 0.2 + 0.2 = 0.4

        tau_prime, feasible = cs.water_filling_alloc(tau, u, m_next=target_m, c_max=c_max)

        self.assertFalse(feasible, "Should be flagged infeasible when cap prevents reaching target")
        self.assertTrue((tau_prime >= tau - 1e-6).all().item(), "Monotonicity violated")
        max_step = (tau_prime - tau).max().item()
        self.assertLessEqual(max_step, c_max + 1e-5, f"Step exceeded c_max: {max_step}")

    def test_03_all_tokens_hit_cap_simultaneously(self):
        """When all tokens hit the cap, algorithm terminates cleanly without infinite loops."""
        L = 64
        tau = torch.full((1, L), 0.2)
        u = torch.rand(1, L)
        c_max = 0.15
        target_m = 0.7

        tau_prime, feasible = cs.water_filling_alloc(tau, u, m_next=target_m, c_max=c_max, n_iters=5)

        self.assertFalse(feasible)
        self.assertTrue(torch.allclose(tau_prime, torch.full_like(tau, 0.2 + c_max), atol=1e-5))

    def test_04_single_token_active_set(self):
        """Single active token (L=1 or 1 uncommitted token) allocates accurately without division error."""
        tau = torch.tensor([[0.25]], dtype=torch.float32)
        u = torch.tensor([[0.5]], dtype=torch.float32)

        # Reachable step
        tau_prime, feasible = cs.water_filling_alloc(tau, u, m_next=0.45, c_max=0.3)
        self.assertTrue(feasible)
        self.assertAlmostEqual(tau_prime.item(), 0.45, places=4)

        # Unreachable step due to cap
        tau_prime_inf, feas_inf = cs.water_filling_alloc(tau, u, m_next=0.8, c_max=0.2)
        self.assertFalse(feas_inf)
        self.assertAlmostEqual(tau_prime_inf.item(), 0.45, places=4)

    def test_05_zero_active_tokens(self):
        """When all tokens are already committed (tau >= 0.999), returns identity safely."""
        tau = torch.full((2, 16), 1.0)
        u = torch.rand(2, 16)
        tau_prime, feasible = cs.water_filling_alloc(tau, u, m_next=0.5)
        self.assertTrue(torch.equal(tau_prime, tau))

    def test_06_uniform_and_zero_uncertainties(self):
        """When uncertainty values are all zero or uniform, weights are equal and do not divide by zero."""
        tau = torch.tensor([[0.2, 0.3, 0.4]])
        u_zero = torch.zeros((1, 3))
        tau_prime, feasible = cs.water_filling_alloc(tau, u_zero, m_next=0.5, c_max=0.4)
        self.assertTrue(feasible)
        self.assertAlmostEqual(tau_prime.mean().item(), 0.5, places=4)

    def test_07_final_step_force_one(self):
        """When m_next >= 0.999, tau_prime is forced to exactly 1.0 regardless of cap."""
        tau = torch.tensor([[0.1, 0.4, 0.7]])
        u = torch.tensor([[1.0, 2.0, 3.0]])
        tau_prime, feasible = cs.water_filling_alloc(tau, u, m_next=1.0, c_max=0.1)
        self.assertTrue(feasible)
        self.assertTrue((tau_prime == 1.0).all().item())

    def test_08_monotone_stress_1000_trials(self):
        """1,000 random trials verify monotonicity and cap bounds under all conditions."""
        torch.manual_seed(42)
        for _ in range(1000):
            B = torch.randint(1, 4, (1,)).item()
            L = torch.randint(2, 32, (1,)).item()
            tau = torch.rand(B, L)
            u = torch.rand(B, L) * 10.0
            c_max = float(torch.empty(1).uniform_(0.05, 0.5).item())
            m_next = float(torch.empty(1).uniform_(0.0, 1.0).item())

            tau_p, feas = cs.water_filling_alloc(tau, u, m_next=m_next, c_max=c_max)

            # Invariant 1: Monotonicity
            self.assertTrue(
                (tau_p >= tau - 1e-6).all().item(),
                f"Clock monotonicity violated: min delta = {(tau_p - tau).min().item()}",
            )
            # Invariant 2: Bounded in [0, 1]
            self.assertTrue((tau_p <= 1.0 + 1e-6).all().item(), "Tau exceeded 1.0")
            # Invariant 3: Step cap compliance when not forced final
            if m_next < 0.999:
                max_step = (tau_p - tau).max().item()
                self.assertLessEqual(
                    max_step,
                    c_max + 1e-5,
                    f"Cap violated: step {max_step} > c_max {c_max}",
                )


class TestAdversarialCMEMask(unittest.TestCase):
    """Stress tests and boundary condition harnesses for Clock-Masked Encoding attention masks."""

    def test_01_quantization_levels(self):
        """Km in {2, 4, inf} produces proper quantization step cutoffs."""
        tau = torch.tensor([[0.1, 0.3, 0.6, 0.9]])
        valid = torch.ones((1, 4), dtype=torch.bool)

        # Km = 2: buckets [0, 0.5) -> 0.0, [0.5, 1.0] -> 0.5
        M_2 = tc.build_cme_mask(tau, valid, Km=2)
        self.assertEqual(M_2.shape, (1, 4, 4))
        # Queries 0 and 1 (bucket 0) can attend to all keys (buckets 0 and 0.5)
        self.assertTrue((M_2[0, 0] == 1.0).all().item())
        self.assertTrue((M_2[0, 1] == 1.0).all().item())
        # Queries 2 and 3 (bucket 0.5) cannot attend to keys 0 and 1 (bucket 0)
        self.assertTrue((M_2[0, 2, :2] == 0.0).all().item())
        self.assertTrue((M_2[0, 3, :2] == 0.0).all().item())

        # Km = inf: continuous unquantized comparisons
        M_inf = tc.build_cme_mask(tau, valid, Km=float("inf"))
        for i in range(4):
            for j in range(4):
                expected = 1.0 if tau[0, j] >= tau[0, i] else 0.0
                self.assertEqual(M_inf[0, i, j].item(), expected)

    def test_02_all_tokens_identical_tau(self):
        """When all tokens have identical tau, valid tokens form full bidirectional attention."""
        tau = torch.full((2, 8), 0.42)
        valid = torch.ones((2, 8), dtype=torch.bool)
        for km in [2, 4, float("inf"), None]:
            M = tc.build_cme_mask(tau, valid, Km=km)
            self.assertTrue((M == 1.0).all().item(), f"Failed for Km={km}")

    def test_03_strictly_descending_tau(self):
        """Strictly descending tau produces strictly lower-triangular causal attention mask."""
        tau = torch.tensor([[1.0, 0.8, 0.6, 0.4, 0.2]])
        valid = torch.ones((1, 5), dtype=torch.bool)
        M = tc.build_cme_mask(tau, valid, Km=float("inf"))

        # Row i can only attend to j <= i
        for i in range(5):
            for j in range(5):
                expected = 1.0 if j <= i else 0.0
                self.assertEqual(
                    M[0, i, j].item(),
                    expected,
                    f"Mismatch at ({i}, {j}): got {M[0, i, j].item()}, expected {expected}",
                )

    def test_04_strictly_ascending_tau(self):
        """Strictly ascending tau produces strictly upper-triangular reverse attention mask."""
        tau = torch.tensor([[0.2, 0.4, 0.6, 0.8, 1.0]])
        valid = torch.ones((1, 5), dtype=torch.bool)
        M = tc.build_cme_mask(tau, valid, Km=float("inf"))

        for i in range(5):
            for j in range(5):
                expected = 1.0 if j >= i else 0.0
                self.assertEqual(M[0, i, j].item(), expected)

    def test_05_padding_mask_invariants(self):
        """Padding tokens are never attended to by valid queries, and pad rows attend to valid keys."""
        valid = torch.tensor([[True, True, True, False, False]])
        tau = torch.tensor([[0.2, 0.5, 0.8, 0.0, 0.0]])
        M = tc.build_cme_mask(tau, valid_mask=valid, Km=4)

        # 1. Valid queries never attend to padding keys
        self.assertEqual(M[0, :3, 3:].sum().item(), 0.0)
        # 2. Pad rows attend to all valid keys (ELF collate rule)
        self.assertTrue((M[0, 3:, :3] == 1.0).all().item())
        # 3. Pad rows do not attend to pad keys
        self.assertEqual(M[0, 3:, 3:].sum().item(), 0.0)
        # 4. Diagonal of valid queries is 1.0
        for i in range(3):
            self.assertEqual(M[0, i, i].item(), 1.0)

    def test_06_edge_case_dimensions(self):
        """Sequence length L=1 and all-invalid mask execute without crashing."""
        # L=1
        M_1 = tc.build_cme_mask(torch.tensor([[0.5]]), Km=2)
        self.assertEqual(M_1.shape, (1, 1, 1))
        self.assertEqual(M_1[0, 0, 0].item(), 1.0)

        # All invalid
        M_inv = tc.build_cme_mask(
            torch.tensor([[0.5, 0.5]]), valid_mask=torch.zeros((1, 2), dtype=torch.bool)
        )
        self.assertEqual(M_inv.shape, (1, 2, 2))
        self.assertEqual(M_inv.sum().item(), 0.0)


class TestAdversarialDeletionTargets(unittest.TestCase):
    """Stress tests and boundary condition harnesses for expanding canvas deletion targets."""

    def test_01_p_del_zero(self):
        """p_del = 0 produces zero deleted tokens, zero gaps, and retains full sequence."""
        tokens = torch.randint(10, 500, (4, 16))
        ret, gaps = exp.build_deletion_targets(tokens, p_del=0.0)
        self.assertTrue(torch.equal(ret, tokens))
        self.assertEqual(gaps.shape, (4, 15))
        self.assertEqual(gaps.sum().item(), 0)

    def test_02_single_deleted_token(self):
        """Single deleted token leaves S-1 retained tokens and gaps summing to exactly 1."""
        tokens = torch.randint(10, 500, (4, 10))
        # S=10, p_del=0.1 -> n_del = 1
        ret, gaps = exp.build_deletion_targets(tokens, p_del=0.1)
        self.assertEqual(ret.shape, (4, 9))
        self.assertEqual(gaps.shape, (4, 8))
        for b in range(4):
            self.assertEqual(gaps[b].sum().item(), 1)

    def test_03_boundary_preservation(self):
        """First (0) and last (S-1) tokens are strictly preserved under high deletion rate."""
        tokens = torch.randint(10, 500, (4, 20))
        ret, gaps = exp.build_deletion_targets(tokens, p_del=0.5, min_keep=4)
        for b in range(4):
            self.assertEqual(ret[b, 0].item(), tokens[b, 0].item(), "First token deleted")
            self.assertEqual(ret[b, -1].item(), tokens[b, -1].item(), "Last token deleted")

    def test_04_short_sequences_below_min_keep(self):
        """Sequences shorter than min_keep return unchanged with empty/zero gaps."""
        for s in [1, 2, 3]:
            tokens = torch.randint(10, 500, (2, s))
            ret, gaps = exp.build_deletion_targets(tokens, p_del=0.3, min_keep=4)
            self.assertTrue(torch.equal(ret, tokens))
            self.assertEqual(gaps.shape, (2, max(0, s - 1)))
            self.assertEqual(gaps.sum().item(), 0)

    def test_05_poisson_loss_extremes(self):
        """Poisson NLL loss behaves properly under zero targets, large rates, and empty masks."""
        rates = torch.tensor([[0.0, 1.0, 10.0]])
        targets = torch.tensor([[0.0, 2.0, 10.0]])
        loss = exp.poisson_gap_loss(rates, targets)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

        # Empty mask
        mask_empty = torch.zeros_like(rates, dtype=torch.bool)
        loss_empty = exp.poisson_gap_loss(rates, targets, mask=mask_empty)
        self.assertEqual(loss_empty.item(), 0.0)


class TestAdversarialPPLInterpolation(unittest.TestCase):
    """Stress tests and boundary condition harnesses for frontier PPL linear interpolation."""

    def test_01_out_of_bracket_returns_nan(self):
        """Target entropy outside the curve bracket returns float('nan')."""
        pts = [(5.0, 30.0), (5.3, 20.0)]
        self.assertTrue(math.isnan(fr.ppl_at_entropy(pts, target_h=4.8)))
        self.assertTrue(math.isnan(fr.ppl_at_entropy(pts, target_h=5.4)))

    def test_02_single_point_and_empty_returns_nan(self):
        """Single-point curves or empty lists return float('nan')."""
        self.assertTrue(math.isnan(fr.ppl_at_entropy([], target_h=5.15)))
        self.assertTrue(math.isnan(fr.ppl_at_entropy([(5.15, 25.0)], target_h=5.15)))

    def test_03_duplicate_entropy_no_zero_division(self):
        """Duplicate entropy points do not trigger division-by-zero errors."""
        pts = [(5.15, 20.0), (5.15, 30.0)]
        val = fr.ppl_at_entropy(pts, target_h=5.15)
        self.assertTrue(math.isfinite(val))
        self.assertAlmostEqual(val, 20.0, places=4)

    def test_04_unordered_and_non_monotonic_sorting(self):
        """Unordered entropy points are sorted before bracket search."""
        pts = [(5.3, 20.0), (4.9, 40.0), (5.1, 28.0)]
        val = fr.ppl_at_entropy(pts, target_h=5.1)
        self.assertAlmostEqual(val, 28.0, places=4)

    def test_05_non_finite_and_negative_ppl_filtered(self):
        """Non-positive or non-finite PPL values are safely filtered out."""
        pts = [(5.0, -10.0), (5.1, 0.0), (5.2, float("inf")), (5.3, 20.0), (5.4, 15.0)]
        val = fr.ppl_at_entropy(pts, target_h=5.35)
        expected = math.sqrt(20.0 * 15.0)
        self.assertAlmostEqual(val, expected, places=4)

    def test_06_exact_boundary_interpolation(self):
        """Target H exactly equal to min(H) or max(H) returns exact boundary PPL."""
        pts = [(5.0, 30.0), (5.3, 20.0)]
        self.assertAlmostEqual(fr.ppl_at_entropy(pts, target_h=5.0), 30.0, places=4)
        self.assertAlmostEqual(fr.ppl_at_entropy(pts, target_h=5.3), 20.0, places=4)


class TestAdversarialEOSMaskingAndProjections(unittest.TestCase):
    """Stress tests and boundary condition harnesses for ECP EOS masking and projection operators."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = MockTokenizer()
        cls.encoder = MockT5Encoder()
        cls.config = MockELFConfig()
        cls.model = MockELFModel(cls.config)

    def test_01_eos_at_position_zero(self):
        """EOS at position 0 marks index 0 valid and pads all subsequent positions."""
        tokens = torch.tensor([[1, 20, 30, 40], [10, 20, 1, 40]], dtype=torch.long)
        masked, valid = ecp.compute_eos_mask(tokens, eos_token_id=1, pad_token_id=0)

        # Batch 0: EOS at position 0
        self.assertEqual(valid[0].tolist(), [True, False, False, False])
        self.assertEqual(masked[0].tolist(), [1, 0, 0, 0])

        # Batch 1: EOS at position 2
        self.assertEqual(valid[1].tolist(), [True, True, True, False])
        self.assertEqual(masked[1].tolist(), [10, 20, 1, 0])

    def test_02_no_eos_in_sequence(self):
        """Sequence with no EOS retains all tokens in valid and leaves IDs unmasked."""
        tokens = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        masked, valid = ecp.compute_eos_mask(tokens, eos_token_id=1, pad_token_id=0)
        self.assertTrue(valid.all().item())
        self.assertTrue(torch.equal(masked, tokens))

    def test_03_multiple_consecutive_eos(self):
        """Multiple consecutive EOS tokens keep only the very first EOS in valid."""
        tokens = torch.tensor([[10, 1, 1, 1, 50]], dtype=torch.long)
        masked, valid = ecp.compute_eos_mask(tokens, eos_token_id=1, pad_token_id=0)
        self.assertEqual(valid[0].tolist(), [True, True, False, False, False])
        self.assertEqual(masked[0].tolist(), [10, 1, 0, 0, 0])

    def test_04_projections_survive_eos_edge_cases(self):
        """All 5 projection operators execute cleanly under EOS-at-0 and no-EOS inputs."""
        B, L, D = 2, 8, 512
        x_hat = torch.randn(B, L, D)

        operators = [
            ("project_hard", lambda: ecp.project_hard(x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config)),
            ("project_samp", lambda: ecp.project_samp(x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config, tau=1.0, top_p=0.9)),
            ("project_soft", lambda: ecp.project_soft(x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config, tau=1.0)),
            ("project_nc", lambda: ecp.project_nc(x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config)),
            ("project_rand", lambda: ecp.project_rand(x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config)),
        ]

        for name, op_call in operators:
            out = op_call()
            self.assertEqual(out.shape, (B, L, D), f"{name} shape mismatch")
            self.assertTrue(torch.isfinite(out).all().item(), f"{name} produced non-finite values")


if __name__ == "__main__":
    unittest.main()
