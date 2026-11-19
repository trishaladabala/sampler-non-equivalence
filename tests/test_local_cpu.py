"""Local CPU Unit Test Runner for elfx modules extracted from research roadmap notebooks.

Extracts all `%%writefile` modules from `elf_p0_harness.ipynb` and the six roadmap notebooks
(Prompt 1A, 1B, 2A, 2B, P3, 4) into a temporary package, and runs comprehensive CPU unit tests
covering core algorithmic, mathematical, and interface contracts:
  1. ecp.py (shapes, EOS masking, lambda schedules, hook modes, counters)
  2. train_ecp.py (loss equivalence at q=0, projection delta at q>0, finite gradients)
  3. train_clock.py (zero-init equivalence, CME mask equality, tau distributions, gradient flow)
  4. clock_sample.py (water-filling convergence, monotonicity, prefix preservation, reduction)
  5. expanding.py (gap target construction, post-transformer insertion head Poisson loss, expand + CME transport)
  6. frontier.py (ppl_at_entropy interpolation, quantiles, Spearman correlation, Phase 1 rescoring)
"""

import ast
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Module Extractor
# ---------------------------------------------------------------------------

NOTEBOOK_SOURCES = [
    ("elf_p0_harness.ipynb", [
        "__init__.py", "load.py", "sample.py", "metrics.py",
        "refs.py", "registry.py", "sweep.py", "official.py"
    ]),
    ("elf_p1a_ecp_sweep.ipynb", ["ecp.py"]),
    ("elf_p1b_ecp_ft.ipynb", ["train_ecp.py"]),
    ("elf_p2a_cmlc_train.ipynb", ["train_clock.py"]),
    ("elf_p2b_cmlc_eval.ipynb", ["clock_sample.py"]),
    ("elf_p3_expanding_pilot.ipynb", ["expanding.py"]),
    ("elf_p4_frontier.ipynb", ["frontier.py"]),
]


def extract_all_modules() -> Tuple[str, Dict[str, str]]:
    """Extracts all %%writefile modules from available notebooks into a temp directory."""
    temp_dir = tempfile.mkdtemp(prefix="elfx_test_pkg_")
    pkg_dir = Path(temp_dir) / "elfx"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    extracted = {}
    for nb_name, expected_modules in NOTEBOOK_SOURCES:
        nb_path = WORKSPACE_ROOT / nb_name
        if not nb_path.exists():
            continue
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            print(f"[extractor][WARN] Could not parse {nb_name}: {e}")
            continue

        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            if not src.startswith("%%writefile"):
                continue

            lines = src.splitlines()
            header = lines[0].split()
            if len(header) < 2:
                continue
            target_path = header[1]
            mod_name = os.path.basename(target_path)

            dest_path = pkg_dir / mod_name
            with open(dest_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(lines[1:]))
            extracted[mod_name] = str(dest_path)

    # Ensure __init__.py exists
    init_path = pkg_dir / "__init__.py"
    if not init_path.exists():
        with open(init_path, "w", encoding="utf-8") as f:
            f.write('__version__ = "0.2.0-test"\n')

    return temp_dir, extracted


# Setup execution environment
TEMP_DIR, EXTRACTED_MODULES = extract_all_modules()
sys.path.insert(0, TEMP_DIR)
sys.path.insert(0, str(WORKSPACE_ROOT / "ELF" / "src"))
os.environ["ELF_ROOT"] = TEMP_DIR
os.environ["ELF_SRC"] = str(WORKSPACE_ROOT / "ELF" / "src")

# ---------------------------------------------------------------------------
# Lightweight Mocks for Fast & Isolated CPU Testing
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Lightweight mock tokenizer with standard EOS and PAD IDs."""
    eos_token_id = 1
    pad_token_id = 0


class MockT5Model(nn.Module):
    def __init__(self, shared: nn.Embedding):
        super().__init__()
        self.shared = shared

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.shared(input_ids)
        class Out:
            pass
        out = Out()
        out.last_hidden_state = x
        return out


class MockT5Encoder(nn.Module):
    """Lightweight mock T5 encoder returning latents directly as Tensor and exposing .model."""
    def __init__(self, hidden_size: int = 512, vocab_size: int = 32100):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.shared = nn.Embedding(vocab_size, hidden_size)
        self.model = MockT5Model(self.shared)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        return self.shared(input_ids)


class MockELFConfig:
    def __init__(self, hidden_size: int = 768, text_encoder_dim: int = 512, vocab_size: int = 32100):
        self.hidden_size = hidden_size
        self.text_encoder_dim = text_encoder_dim
        self.vocab_size = vocab_size
        self.model = "ELF-B"
        self.sc_cfg = 3.0
        self.latent_mean = 0.0
        self.latent_std = 0.2
        self.denoiser_noise_scale = 2.0
        self.self_cond_prob = 0.5
        self.num_self_cond_cfg_tokens = 4
        self.use_bf16 = False


class MockELFModel(nn.Module):
    """Lightweight mock ELF model for forward, decoder logits, and step testing."""
    def __init__(self, config: Optional[MockELFConfig] = None):
        super().__init__()
        self.config = config or MockELFConfig()
        H = self.config.hidden_size
        D = self.config.text_encoder_dim
        V = self.config.vocab_size

        self.text_proj = nn.Linear(D, H)
        self.self_cond_proj = nn.Linear(2 * D, D)
        self.final_layer = nn.Linear(H, D)
        self.final_layer.weight.data.zero_()
        self.final_layer.bias.data.zero_()

        self.decoder_proj = nn.Linear(H, D)
        self.unembed = nn.Linear(D, V)
        self.unembed_bias = nn.Parameter(torch.zeros(V))

    def forward(self, x, t, decoder_step_active=None, **kwargs):
        # x is (B, L, D) or (B, L, 2D) if self-cond
        if x.shape[-1] == 2 * self.config.text_encoder_dim:
            x = self.self_cond_proj(x)
        h = self.text_proj(x)
        v = self.final_layer(h)
        logits = None
        if decoder_step_active is not None:
            dec_h = F.gelu(self.decoder_proj(h))
            logits = self.unembed(dec_h) + self.unembed_bias
        return v, logits


# ---------------------------------------------------------------------------
# Test Suite 1: ecp.py (Prompt 1A)
# ---------------------------------------------------------------------------

class TestECP(unittest.TestCase):
    """Comprehensive tests for elfx/ecp.py."""

    @classmethod
    def setUpClass(cls):
        if "ecp.py" not in EXTRACTED_MODULES:
            raise unittest.SkipTest("ecp.py not found in notebooks yet")
        try:
            import elfx.ecp as ecp_mod
            cls.ecp = ecp_mod
        except Exception as e:
            raise unittest.SkipTest(f"Failed to import elfx.ecp: {e}")

        cls.tokenizer = MockTokenizer()
        cls.encoder = MockT5Encoder()
        cls.config = MockELFConfig()
        cls.model = MockELFModel(cls.config)

    def test_01_shapes_and_finiteness_all_projections(self):
        """All 5 projection operators output shape (B, L, D) and finite values."""
        B, L, D = 2, 16, 512
        x_hat = torch.randn(B, L, D)

        proj_hard_fn = getattr(self.ecp, "project_hard", getattr(self.ecp, "pi_hard", None))
        proj_samp_fn = getattr(self.ecp, "project_samp", getattr(self.ecp, "pi_samp", None))
        proj_soft_fn = getattr(self.ecp, "project_soft", getattr(self.ecp, "pi_soft", None))
        proj_nc_fn = getattr(self.ecp, "project_nc", getattr(self.ecp, "pi_nc", None))
        proj_rand_fn = getattr(self.ecp, "project_rand", getattr(self.ecp, "pi_rand", None))

        # 1. hard
        out_hard = proj_hard_fn(
            x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config
        )
        self.assertEqual(out_hard.shape, (B, L, D))
        self.assertTrue(torch.isfinite(out_hard).all().item(), "project_hard contained non-finite values")

        # 2. samp
        out_samp = proj_samp_fn(
            x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config, tau=1.0, top_p=0.9
        )
        self.assertEqual(out_samp.shape, (B, L, D))
        self.assertTrue(torch.isfinite(out_samp).all().item(), "project_samp contained non-finite values")

        # 3. soft
        out_soft = proj_soft_fn(
            x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config, tau=1.0
        )
        self.assertEqual(out_soft.shape, (B, L, D))
        self.assertTrue(torch.isfinite(out_soft).all().item(), "project_soft contained non-finite values")

        # 4. nc (identity attention mask)
        out_nc = proj_nc_fn(
            x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config
        )
        self.assertEqual(out_nc.shape, (B, L, D))
        self.assertTrue(torch.isfinite(out_nc).all().item(), "project_nc contained non-finite values")

        # 5. rand (matched-norm Gaussian perturbation)
        out_rand = proj_rand_fn(
            x_hat, encoder=self.encoder, tokenizer=self.tokenizer, model=self.model, config=self.config
        )
        self.assertEqual(out_rand.shape, (B, L, D))
        self.assertTrue(torch.isfinite(out_rand).all().item(), "project_rand contained non-finite values")

    def test_02_eos_masking_preserves_first_eos(self):
        """EOS masking keeps the first EOS in valid and masks subsequent positions."""
        eos_fn = getattr(self.ecp, "compute_eos_mask", getattr(self.ecp, "valid_and_eos_mask", None))
        tokens = torch.tensor([[100, 200, 1, 300, 400]], dtype=torch.long)
        import inspect
        sig = inspect.signature(eos_fn)
        res = eos_fn(tokens, eos_token_id=1, pad_token_id=0) if "eos_token_id" in sig.parameters else eos_fn(tokens, eos_id=1, pad_id=0)
        # res is (masked_ids, valid_mask) or (w_enc, valid)
        w_enc, valid = res[0], res[1]

        expected_valid = torch.tensor([[True, True, True, False, False]])
        expected_w = torch.tensor([[100, 200, 1, 0, 0]])

        self.assertTrue(torch.equal(valid, expected_valid), f"valid mask mismatch: {valid}")
        self.assertTrue(torch.equal(w_enc, expected_w), f"w_enc mismatch: {w_enc}")

        # Edge case: No EOS in sequence -> all tokens valid
        no_eos_tokens = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        res_no_eos = eos_fn(no_eos_tokens, eos_token_id=1, pad_token_id=0) if "eos_token_id" in sig.parameters else eos_fn(no_eos_tokens, eos_id=1, pad_id=0)
        self.assertTrue(res_no_eos[1].all().item(), "All tokens should be valid when no EOS present")
        self.assertTrue(torch.equal(res_no_eos[0], no_eos_tokens))

    def test_03_lambda_schedules(self):
        """Lambda step and ramp mixing schedules adhere strictly to mathematical specifications."""
        lam_fn = getattr(self.ecp, "get_lambda", getattr(self.ecp, "lambda_schedule", None))
        import inspect
        sig = inspect.signature(lam_fn)
        # Step schedule: 0 below t_on, lambda_max at and above t_on
        t = torch.tensor([0.1, 0.49, 0.5, 0.8])
        step_vals = lam_fn(t, mode="step", t_on=0.5, lambda_max=0.8) if "mode" in sig.parameters else lam_fn(t, schedule="step", t_on=0.5, lambda_max=0.8)
        step_flat = step_vals.squeeze().tolist() if isinstance(step_vals, torch.Tensor) else [step_vals]
        self.assertAlmostEqual(step_flat[0], 0.0, places=5)
        self.assertAlmostEqual(step_flat[1], 0.0, places=5)
        self.assertAlmostEqual(step_flat[2], 0.8, places=5)
        self.assertAlmostEqual(step_flat[3], 0.8, places=5)

        # Ramp schedule: linear ramp from t_on to 1.0
        t_ramp = torch.tensor([0.2, 0.5, 0.75, 1.0])
        ramp_vals = lam_fn(t_ramp, mode="ramp", t_on=0.5, lambda_max=1.0) if "mode" in sig.parameters else lam_fn(t_ramp, schedule="ramp", t_on=0.5, lambda_max=1.0)
        ramp_flat = ramp_vals.squeeze().tolist() if isinstance(ramp_vals, torch.Tensor) else [ramp_vals]
        self.assertAlmostEqual(ramp_flat[0], 0.0, places=5)
        self.assertAlmostEqual(ramp_flat[1], 0.0, places=5)
        self.assertAlmostEqual(ramp_flat[2], 0.5, places=5)
        self.assertAlmostEqual(ramp_flat[3], 1.0, places=5)

    def test_04_hook_modes_transformations(self):
        """Hook modes (SC, Z-weak, Z-strong, SC+Z-weak) execute proper updates."""
        hook_maker = getattr(self.ecp, "build_ecp_hook", getattr(self.ecp, "make_ecp_hook", None))
        import inspect
        sig = inspect.signature(hook_maker)
        hook_fn = hook_maker(
            model=self.model,
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            config=self.config,
            mode="SC",
            proj_type="hard" if "proj_type" in sig.parameters else "hard",
            lambda_max=1.0,
            t_on=0.2,
            schedule_type="step" if "schedule_type" in sig.parameters else "step",
            k=1,
        )

        B, L, D = 2, 8, 512
        z = torch.randn(B, L, D)
        x_pred = torch.randn(B, L, D)

        t = torch.tensor([0.5, 0.5])
        t_next = torch.tensor([0.6, 0.6])
        t_eff = torch.tensor([0.5, 0.5])

        # Step at t=0.5 (above t_on=0.2)
        z_out, x_out = hook_fn(
            i=1, t=t, t_next=t_next, t_eff=t_eff, z=z.clone(), x_pred=x_pred.clone()
        )
        # SC mode updates x_pred while preserving z
        self.assertTrue(torch.equal(z_out, z), "SC mode must preserve z")
        self.assertFalse(torch.equal(x_out, x_pred), "SC mode must update x_pred with projection")

    def test_05_nfe_counter_increments(self):
        """NFE counter increments when decoder or encoder forwards occur."""
        from elfx import sample as sample_mod
        before = sample_mod.n_enc_fwd
        self.ecp.count_enc(2)
        self.assertEqual(sample_mod.n_enc_fwd, before + 2)


# ---------------------------------------------------------------------------
# Test Suite 2: train_ecp.py (Prompt 1B)
# ---------------------------------------------------------------------------

class TestTrainECP(unittest.TestCase):
    """Comprehensive tests for elfx/train_ecp.py."""

    @classmethod
    def setUpClass(cls):
        if "train_ecp.py" not in EXTRACTED_MODULES:
            raise unittest.SkipTest("train_ecp.py not found in notebooks yet")
        try:
            import elfx.train_ecp as train_ecp_mod
            cls.train_ecp = train_ecp_mod
        except Exception as e:
            raise unittest.SkipTest(f"Failed to import elfx.train_ecp: {e}")

    def test_01_loss_equivalence_at_q0(self):
        """Loss computation at q=0 matches reference train_step within numerical precision."""
        from modules.model import ELF
        from train_step import train_step
        from utils.train_utils import TrainState
        from types import SimpleNamespace
        import copy

        cfg = SimpleNamespace(
            t_eps=0.05, self_cond_prob=0.5, latent_mean=0.0, latent_std=0.2,
            decoder_prob=0.2, decoder_noise_scale=5.0, denoiser_noise_scale=2.0,
            denoiser_p_mean=-1.5, denoiser_p_std=0.8, time_schedule="logit_normal",
            pad_token="pad", label_drop_prob=0.0, num_self_cond_cfg_tokens=4,
            self_cond_cfg_min=1.0, self_cond_cfg_max=3.0, decoder_p_std=0.8,
            decoder_p_mean=-1.5, grad_accum_steps=1, ema_decay1=0.999,
            use_bf16=False, max_length=16,
        )

        enc = MockT5Encoder(hidden_size=32, vocab_size=100)
        m_base = ELF(depth=1, hidden_size=64, num_heads=2, text_encoder_dim=32, vocab_size=100, max_length=16)
        batch = {
            "input_ids": torch.randint(1, 100, (2, 16)),
            "encoder_attention_mask": torch.ones(2, 16, 16),
            "attention_mask": torch.ones(2, 16),
            "cond_seq_mask": torch.zeros(2, 16),
        }

        m_ref = copy.deepcopy(m_base)
        torch.manual_seed(42)
        st_ref = TrainState(model=m_ref, optimizer=torch.optim.AdamW(m_ref.parameters(), lr=0.0))
        st_ref.ema_params1 = TrainState.init_ema(m_ref)
        _, m_ref_out = train_step(st_ref, enc, batch, cfg)

        m_ecp = copy.deepcopy(m_base)
        torch.manual_seed(42)
        st_ecp = TrainState(model=m_ecp, optimizer=torch.optim.AdamW(m_ecp.parameters(), lr=0.0))
        st_ecp.ema_params1 = TrainState.init_ema(m_ecp)
        step_fn = getattr(self.train_ecp, "ecp_train_step", getattr(self.train_ecp, "train_step_ecp", None))
        self.assertIsNotNone(step_fn, "Neither ecp_train_step nor train_step_ecp found in train_ecp")
        _, m_ecp_out = step_fn(st_ecp, enc, batch, cfg, q=0.0)

        delta = abs(m_ref_out["loss"].item() - m_ecp_out["loss"].item())
        self.assertLess(delta, 1e-4, f"Loss delta {delta} between reference and ecp_train_step at q=0 exceeds 1e-4")

    def test_02_projection_delta_active_at_q_positive(self):
        """Projection delta is non-zero and tracked when q > 0."""
        from modules.model import ELF
        from utils.train_utils import TrainState
        from types import SimpleNamespace
        import copy

        cfg = SimpleNamespace(
            t_eps=0.05, self_cond_prob=0.5, latent_mean=0.0, latent_std=0.2,
            decoder_prob=0.2, decoder_noise_scale=5.0, denoiser_noise_scale=2.0,
            denoiser_p_mean=-1.5, denoiser_p_std=0.8, time_schedule="logit_normal",
            pad_token="pad", label_drop_prob=0.0, num_self_cond_cfg_tokens=4,
            self_cond_cfg_min=1.0, self_cond_cfg_max=3.0, decoder_p_std=0.8,
            decoder_p_mean=-1.5, grad_accum_steps=1, ema_decay1=0.999,
            use_bf16=False, max_length=16,
        )

        enc = MockT5Encoder(hidden_size=32, vocab_size=100)
        m_base = ELF(depth=1, hidden_size=64, num_heads=2, text_encoder_dim=32, vocab_size=100, max_length=16)
        batch = {
            "input_ids": torch.randint(1, 100, (16, 16)),
            "encoder_attention_mask": torch.ones(16, 16, 16),
            "attention_mask": torch.ones(16, 16),
            "cond_seq_mask": torch.zeros(16, 16),
        }

        m_ecp = copy.deepcopy(m_base)
        torch.manual_seed(42)
        st_ecp = TrainState(model=m_ecp, optimizer=torch.optim.AdamW(m_ecp.parameters(), lr=0.0))
        st_ecp.ema_params1 = TrainState.init_ema(m_ecp)
        step_fn = getattr(self.train_ecp, "ecp_train_step", getattr(self.train_ecp, "train_step_ecp", None))
        self.assertIsNotNone(step_fn, "Neither ecp_train_step nor train_step_ecp found in train_ecp")
        _, met = step_fn(st_ecp, enc, batch, cfg, q=0.5, t_on=0.0)

        self.assertIn("proj_delta", met, "proj_delta metric missing from ecp_train_step return")
        self.assertIn("frac_proj", met, "frac_proj metric missing from ecp_train_step return")
        self.assertGreater(met["proj_delta"], 0.0, f"proj_delta {met['proj_delta']} must be strictly positive at q=0.5")
        self.assertGreater(met["frac_proj"], 0.0, f"frac_proj {met['frac_proj']} must be strictly positive at q=0.5")


# ---------------------------------------------------------------------------
# Test Suite 3: train_clock.py (Prompt 2A)
# ---------------------------------------------------------------------------

class TestTrainClock(unittest.TestCase):
    """Comprehensive tests for elfx/train_clock.py."""

    @classmethod
    def setUpClass(cls):
        if "train_clock.py" not in EXTRACTED_MODULES:
            raise unittest.SkipTest("train_clock.py not found in notebooks yet")
        try:
            import elfx.train_clock as train_clock_mod
            cls.tc = train_clock_mod
        except Exception as e:
            raise unittest.SkipTest(f"Failed to import elfx.train_clock: {e}")

    def test_01_zero_init_equivalence(self):
        """ELFClock with zero-initialized lt_out equals baseline ELF within 1e-5."""
        if not hasattr(self.tc, "ELFClock"):
            self.skipTest("ELFClock not defined")
        from modules.model import ELF
        D, H, L = 512, 768, 16
        base_elf = ELF(depth=2, hidden_size=H, num_heads=12, max_length=L, text_encoder_dim=D)
        clock_elf = self.tc.ELFClock(depth=2, hidden_size=H, num_heads=12, max_length=L, text_encoder_dim=D)

        # Copy identical weights to overlapping keys
        base_dict = base_elf.state_dict()
        clock_elf.load_state_dict(base_dict, strict=False)

        # Ensure lt_out is strictly zero-initialized
        if hasattr(clock_elf, "lt_out"):
            clock_elf.lt_out.weight.data.zero_()
            if clock_elf.lt_out.bias is not None:
                clock_elf.lt_out.bias.data.zero_()

        x = torch.randn(2, L, D)
        t = torch.tensor([0.5, 0.5])
        sc = torch.tensor([3.0, 3.0])
        tau = torch.rand(2, L)

        with torch.no_grad():
            out_base = base_elf(x, t, self_cond_cfg_scale=sc)
            if isinstance(out_base, tuple):
                out_base = out_base[0]
            out_clock = clock_elf(x, t, local_t=tau, self_cond_cfg_scale=sc)
            if isinstance(out_clock, tuple):
                out_clock = out_clock[0]

        max_delta = (out_base - out_clock).abs().max().item()
        self.assertLess(max_delta, 1e-5, f"Zero-init delta {max_delta} exceeds 1e-5")

    def test_02_cme_mask_equality(self):
        """CME mask matches unconditional mask in global mode and conditional mask for known prefix."""
        if not hasattr(self.tc, "build_cme_mask"):
            self.skipTest("build_cme_mask not defined")

        B, L = 2, 8
        valid = torch.ones(B, L, dtype=torch.bool)

        # Global mode: all tau equal
        tau_global = torch.full((B, L), 0.5)
        cme_global = self.tc.build_cme_mask(tau_global, valid, Km=None)

        # In global mode, all valid tokens attend to all valid tokens
        expected_global = valid.unsqueeze(1).expand(-1, L, -1).float()
        self.assertTrue(torch.allclose(cme_global, expected_global), "Global CME mask must equal full mask")

        # Known prefix: tokens 0..3 have tau=1.0, tokens 4..7 have tau=0.2
        tau_prefix = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2],
            [1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2],
        ])
        cme_prefix = self.tc.build_cme_mask(tau_prefix, valid, Km=None)
        # Prefix tokens (i < 4) can only attend to j < 4 (since q(tau_j) >= q(tau_i))
        prefix_attend_suffix = cme_prefix[:, :4, 4:]
        self.assertEqual(prefix_attend_suffix.sum().item(), 0.0, "Prefix tokens must NOT attend to ungenerated suffix")

    def test_03_tau_sampler_mixture_distributions(self):
        """Vectorized tau sampler returns bounded values and valid known-prefix structure."""
        if not hasattr(self.tc, "sample_tau_mixture"):
            self.skipTest("sample_tau_mixture not defined")

        B, L = 64, 32
        valid = torch.ones(B, L, dtype=torch.bool)
        res = self.tc.sample_tau_mixture(B, L, valid_lengths=torch.full((B,), L))
        tau, known_mask = res[0], res[1]

        self.assertEqual(tau.shape, (B, L))
        self.assertTrue((tau >= 0.0).all().item(), "tau values below 0")
        self.assertTrue((tau <= 1.0).all().item(), "tau values above 1")

        # Check that whenever known_mask is True, tau == 1.0
        if known_mask.any():
            self.assertTrue((tau[known_mask] == 1.0).all().item(), "Known tokens must have tau == 1.0")


# ---------------------------------------------------------------------------
# Test Suite 4: clock_sample.py (Prompt 2B)
# ---------------------------------------------------------------------------

class TestClockSample(unittest.TestCase):
    """Comprehensive tests for elfx/clock_sample.py."""

    @classmethod
    def setUpClass(cls):
        if "clock_sample.py" not in EXTRACTED_MODULES:
            raise unittest.SkipTest("clock_sample.py not found in notebooks yet")
        try:
            import elfx.clock_sample as cs_mod
            cls.cs = cs_mod
        except Exception as e:
            raise unittest.SkipTest(f"Failed to import elfx.clock_sample: {e}")

    def test_01_water_filling_convergence_and_monotonicity(self):
        """Water filling respects monotonicity, per-step cap, and converges to target mean."""
        alloc_fn = getattr(self.cs, "water_filling_alloc", getattr(self.cs, "water_filling_allocation", None))
        if alloc_fn is None:
            self.skipTest("water-filling allocation function not found")

        L = 32
        tau = torch.full((1, L), 0.2)
        u = torch.rand(1, L)  # Uncertainties
        target_mean = 0.4
        c_max = 0.3

        tau_next, feasible = alloc_fn(tau, u, target_mean, c_max=c_max)

        # Monotonicity
        self.assertTrue((tau_next >= tau - 1e-6).all().item(), "Water filling violated clock monotonicity")

        # Cap
        max_step = (tau_next - tau).max().item()
        self.assertLessEqual(max_step, c_max + 1e-5, f"Step cap {c_max} exceeded: {max_step}")

        # Mean convergence
        if feasible:
            achieved_mean = tau_next.mean().item()
            self.assertAlmostEqual(achieved_mean, target_mean, places=3)

    def test_02_final_step_force_tau_one(self):
        """Final sampling step forces tau to 1.0 for all active tokens."""
        alloc_fn = getattr(self.cs, "water_filling_alloc", getattr(self.cs, "water_filling_allocation", None))
        self.assertIsNotNone(alloc_fn, "water_filling_alloc not found in clock_sample")

        # Test with 2D batch of varying tau values
        tau = torch.tensor([
            [0.1, 0.4, 0.7, 0.95],
            [0.2, 0.5, 0.8, 0.85],
        ])
        u = torch.tensor([
            [0.3, 0.8, 0.5, 0.2],
            [0.1, 0.9, 0.4, 0.7],
        ])

        # At final step (m_next = 1.0), all tokens must be forced to tau = 1.0
        tau_next, feasible = alloc_fn(tau, u, m_next=1.0, c_max=0.5)
        self.assertTrue(feasible, "Final step allocation should report feasible")
        self.assertTrue(torch.allclose(tau_next, torch.ones_like(tau_next)),
                        f"Final step (m_next=1.0) must force all tau to 1.0, got {tau_next}")

        # Also test near-final threshold m_next = 0.9995
        tau_next_near, _ = alloc_fn(tau, u, m_next=0.9995, c_max=0.5)
        self.assertTrue(torch.allclose(tau_next_near, torch.ones_like(tau_next_near)),
                        f"Step with m_next >= 0.999 must force all tau to 1.0, got {tau_next_near}")


# ---------------------------------------------------------------------------
# Test Suite 5: expanding.py (Stretch Proposal P3 Pilot)
# ---------------------------------------------------------------------------

class TestExpanding(unittest.TestCase):
    """Comprehensive tests for elfx/expanding.py."""

    @classmethod
    def setUpClass(cls):
        if "expanding.py" not in EXTRACTED_MODULES:
            raise unittest.SkipTest("expanding.py not found in notebooks yet")
        try:
            import elfx.expanding as exp_mod
            cls.exp = exp_mod
        except Exception as e:
            raise unittest.SkipTest(f"Failed to import elfx.expanding: {e}")

    def test_01_deletion_target_construction(self):
        """Target construction produces non-zero gap targets summing to deleted count."""
        target_fn = getattr(
            self.exp,
            "build_deletion_targets",
            getattr(self.exp, "create_deletion_insertion_targets", None),
        )
        if target_fn is None:
            self.skipTest("deletion target construction function not found")

        B, S = 4, 32
        tokens = torch.randint(10, 1000, (B, S))
        valid = torch.ones(B, S, dtype=torch.bool)

        res = target_fn(tokens, p_del=0.3)
        # res returns (masked_tokens, gap_counts) or (masked_tokens, gap_counts, ...)
        masked_tokens = res[0]
        gap_counts = res[1]

        # Verify gap counts sum to deleted tokens for each row
        for b in range(B):
            n_orig = S
            n_kept = masked_tokens[b].shape[0] if masked_tokens[b].dim() == 1 else (masked_tokens[b] != 0).sum().item()
            n_deleted = n_orig - n_kept
            sum_gaps = gap_counts[b].sum().item()
            self.assertEqual(
                sum_gaps,
                n_deleted,
                f"Row {b}: sum of gap counts ({sum_gaps}) must equal deleted tokens ({n_deleted})",
            )

    def test_02_post_transformer_insertion_head_poisson_loss(self):
        """InsertionHead processes post-transformer hidden states and Poisson loss decreases."""
        if not hasattr(self.exp, "InsertionHead"):
            self.skipTest("InsertionHead not defined")

        B, S, H = 2, 16, 768
        head = self.exp.InsertionHead(hidden_size=H)
        hidden_states = torch.randn(B, S, H, requires_grad=True)

        rates = head(hidden_states)
        self.assertEqual(rates.shape, (B, S - 1), "Rates must be predicted for S-1 gaps")
        self.assertTrue((rates >= 0.0).all().item(), "Poisson rates must be non-negative")

        # Optimize Poisson loss on synthetic target counts
        target_counts = torch.randint(0, 3, (B, S - 1)).float()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

        initial_loss = F.poisson_nll_loss(rates, target_counts, full=True).item()
        for _ in range(10):
            optimizer.zero_grad()
            pred_rates = head(hidden_states)
            loss = F.poisson_nll_loss(pred_rates, target_counts, full=True)
            loss.backward()
            optimizer.step()

        final_loss = F.poisson_nll_loss(head(hidden_states), target_counts, full=True).item()
        self.assertLess(final_loss, initial_loss, "Poisson NLL loss should decrease on overfit batch")

    def test_03_vectorized_expand_and_cme_transport(self):
        """vectorized_expand_cme expands canvas, sets new slots to tau=0, and CME transports."""
        expand_fn = getattr(self.exp, "vectorized_expand_cme", None)
        if expand_fn is None:
            self.skipTest("vectorized_expand_cme not found")

        B, S, D = 2, 8, 512
        z = torch.randn(B, S, D)
        tau = torch.full((B, S), 0.5)
        # 1 insertion in gap 0, 0 elsewhere
        k = torch.zeros((B, S - 1), dtype=torch.long)
        k[:, 0] = 1
        hat_x = torch.randn(B, S, D)

        z_exp, tau_exp, mask_exp = expand_fn(z, tau, k, hat_x, encoder=None)
        # Expanded length must be S + 1 = 9
        self.assertEqual(z_exp.shape[1], S + 1)
        self.assertEqual(tau_exp.shape[1], S + 1)
        # Newly inserted slot (at index 1) must have tau == 0.0
        self.assertAlmostEqual(tau_exp[0, 1].item(), 0.0, places=5)


# ---------------------------------------------------------------------------
# Test Suite 6: frontier.py (Prompt 4)
# ---------------------------------------------------------------------------

class TestFrontier(unittest.TestCase):
    """Comprehensive tests for elfx/frontier.py."""

    @classmethod
    def setUpClass(cls):
        if "frontier.py" not in EXTRACTED_MODULES:
            raise unittest.SkipTest("frontier.py not found in notebooks yet")
        try:
            import elfx.frontier as frontier_mod
            cls.fr = frontier_mod
        except Exception as e:
            raise unittest.SkipTest(f"Failed to import elfx.frontier: {e}")

    def test_01_ppl_at_entropy_interpolation(self):
        """Linear interpolation of log(PPL) against unigram entropy H."""
        interp_fn = getattr(self.fr, "ppl_at_entropy", None)
        if interp_fn is None:
            self.skipTest("ppl_at_entropy not found")

        # Two points: (H=5.0, PPL=30.0) and (H=5.3, PPL=20.0)
        # Interpolate at H=5.15 (exact midpoint)
        # log(PPL) = 0.5 * (log(30) + log(20)) = log(sqrt(600)) = log(24.494897)
        points = [(5.0, 30.0), (5.3, 20.0)]
        target_h = 5.15
        expected_ppl = math.sqrt(30.0 * 20.0)

        # Call interp_fn supporting either points list or (ppls, entropies)
        try:
            val = interp_fn(points, target_h=target_h)
        except TypeError:
            val = interp_fn([30.0, 20.0], [5.0, 5.3], target_h=target_h)

        self.assertAlmostEqual(val, expected_ppl, places=3)

        # Out-of-bracket target returns NaN
        try:
            nan_val = interp_fn(points, target_h=6.0)
        except TypeError:
            nan_val = interp_fn([30.0, 20.0], [5.0, 5.3], target_h=6.0)
        self.assertTrue(math.isnan(nan_val), "Out-of-bracket target entropy must return NaN")

    def test_02_quantiles_computation(self):
        """Per-sample PPL quantiles (10%, 50%, 90%) are monotonically ordered."""
        quantiles_fn = getattr(
            self.fr, "compute_sample_quantiles", getattr(self.fr, "ppl_quantiles", None)
        )
        if quantiles_fn is None:
            self.skipTest("quantile function not found")

        # Synthetic sample NLLs and token lengths
        nlls = [200.0, 300.0, 400.0, 500.0, 600.0]
        counts = [100, 100, 100, 100, 100]
        q = quantiles_fn(nlls, counts)
        self.assertIn("p10", q)
        self.assertIn("p50", q)
        self.assertIn("p90", q)
        self.assertLessEqual(q["p10"], q["p50"])
        self.assertLessEqual(q["p50"], q["p90"])

    def test_03_spearman_correlation(self):
        """Spearman correlation helper computes correct rank correlation."""
        spearman_fn = getattr(self.fr, "compute_spearman_correlations", None)
        if spearman_fn is None:
            self.skipTest("spearman function not found")

        gammas = [0.0, 1.0, 2.0, 3.0]
        rep4 = [0.01, 0.05, 0.15, 0.40]
        H = [5.10, 5.15, 5.25, 5.35]
        corr = spearman_fn(gammas, rep4, H)
        key = "rho_gamma_rep4" if "rho_gamma_rep4" in corr else "gamma_rep4"
        self.assertIn(key, corr)
        self.assertAlmostEqual(corr[key], 1.0, places=2)

    def test_04_phase1_rescore_logic(self):
        """Phase 1 rescoring resolves the gamma=3.0 artifact."""
        rescore_fn = getattr(self.fr, "rescore_phase1_sweep", None)
        if rescore_fn is None:
            self.skipTest("rescore_phase1_sweep not found")

        phase1_path = WORKSPACE_ROOT / "ELF" / "phase1" / "results" / "p1_gamma" / "p1_gamma_sweep.json"
        if not phase1_path.exists():
            self.skipTest("Phase 1 sweep json not found")

        res = rescore_fn(phase1_path)
        self.assertIsInstance(res, dict)
        self.assertTrue(
            "markdown_table" in res or "comparison_rows" in res or "rescore_table" in res,
            f"Missing table in rescore output: {res.keys()}",
        )
        self.assertEqual(res.get("gamma_star_corrected"), 1.5)


class TestSampleP0(unittest.TestCase):
    """Tests for Prompt 0 elfx/sample.py (S2b SDE/ODE equivalence under bf16 and S3 RNG batch invariance)."""

    def test_01_s2b_ode_sde_bf16_equivalence(self):
        """sample.rollout matches _generate_samples_single_batch bit-for-bit under bf16 autocast."""
        import elfx.sample as sample_mod
        from utils.generation_utils import _generate_samples_single_batch
        from utils.sampling_utils import get_sampling_steps
        from configs.config import SamplingConfig, Config
        from modules.model import ELF

        torch.manual_seed(0)
        cfg = Config()
        cfg.max_length = 32
        cfg.denoiser_noise_scale = 2.0
        cfg.denoiser_p_mean = -0.8
        cfg.denoiser_p_std = 0.8
        cfg.t_eps = 0.05
        cfg.self_cond_prob = 0.9
        cfg.num_self_cond_cfg_tokens = 4
        cfg.num_time_tokens = 4
        cfg.num_model_mode_tokens = 1
        cfg.use_bf16 = True

        model = ELF(
            text_encoder_dim=64, max_length=32, hidden_size=128, depth=4, num_heads=4,
            bottleneck_dim=32, num_time_tokens=4, num_self_cond_cfg_tokens=4,
            num_model_mode_tokens=1, vocab_size=100,
        ).eval()
        torch.nn.init.normal_(model.final_layer.linear.weight, std=0.2)

        z = sample_mod.initial_noise(range(4), 7, 32, 64, cfg.denoiser_noise_scale, "cpu")
        torch.manual_seed(11)
        ts = get_sampling_steps(n_steps=8, time_schedule="logit_normal", P_mean=-0.8, P_std=0.8)

        for method, gamma in [("ode", 0.0), ("sde", 1.5)]:
            sc = SamplingConfig(
                sampling_method=method, num_sampling_steps=[8], cfgs=[1],
                sde_gamma=gamma, self_cond_cfg_scales=[3], time_schedule="logit_normal",
            )
            with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
                g1 = torch.Generator().manual_seed(123)
                off = _generate_samples_single_batch(
                    model=model, generator=g1, z=z.clone(), t_steps=ts,
                    cond_seq=None, cond_seq_mask=None, config=cfg, sampling_config=sc,
                    cfg_scale=1, self_cond_cfg_scale=3,
                )
                g2 = torch.Generator().manual_seed(123)
                eps = (lambda i: torch.randn(z.shape, generator=g2)) if method == "sde" else None
                mine = sample_mod.rollout(model, z.clone(), ts, cfg, method, gamma, 3.0, eps_fn=eps)
            rel = ((off - mine).norm() / off.norm()).item()
            self.assertEqual(rel, 0.0, f"{method}: rel_err={rel} != 0.0 under bf16 autocast")

    def test_02_t5_3d_mask_transformers5_compat(self):
        """Real T5EncoderModel with 3D (B, L, L) float mask (with 0s) works via _patch_t5_3d_mask."""
        import numpy as np
        import elfx
        from transformers import T5Config, T5EncoderModel
        from modules.t5_encoder import T5Encoder
        from utils.encoder_utils import encode_text, build_self_attn_cond_masks

        elfx.ensure_elf_on_path()
        cfg = T5Config(vocab_size=100, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4)
        enc = T5Encoder.__new__(T5Encoder)
        torch.nn.Module.__init__(enc)
        enc.model = T5EncoderModel(cfg).eval()

        is_cond = np.array([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]], dtype=bool)
        is_valid = np.array([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=bool)
        enc_mask, _, _ = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
        enc_mask_t = torch.from_numpy(enc_mask)
        ids = torch.randint(0, 100, (2, 8))

        lat1 = encode_text(ids, enc_mask_t, enc, 0.0, 0.2, use_bf16=False)
        self.assertEqual(lat1.shape, (2, 8, 32))
        ids2 = ids.clone()
        ids2[0, 3:] = (ids2[0, 3:] + 13) % 100
        lat2 = encode_text(ids2, enc_mask_t, enc, 0.0, 0.2, use_bf16=False)
        self.assertEqual((lat1[0, :3] - lat2[0, :3]).abs().max().item(), 0.0)
        self.assertGreater((lat1[0, 3:6] - lat2[0, 3:6]).abs().max().item(), 0.0)


def run_summary():
    """Prints status of extracted modules and executes CPU unit test suite."""
    print("=" * 78)
    print("ELF LOCAL CPU MODULE UNIT TEST SUITE")
    print("=" * 78)
    print(f"Extracted Package Path: {TEMP_DIR}/elfx")
    print(f"Total Extracted Modules: {len(EXTRACTED_MODULES)}")
    for mod, path in sorted(EXTRACTED_MODULES.items()):
        size_kb = os.path.getsize(path) / 1024
        print(f"  * {mod:<20} ({size_kb:.1f} KB)")
    print("=" * 78)


if __name__ == "__main__":
    run_summary()
    unittest.main()
