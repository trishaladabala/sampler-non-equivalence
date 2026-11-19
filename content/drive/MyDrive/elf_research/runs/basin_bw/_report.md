# Exclusive Notebook 6 — 105M Basin-Width Diagnostic & BW-ELF Fine-Tuning Report

## 1. Summary of Codebase Bugs Repaired (`phase3_basin/` & `src/train_step.py`)
1. **`phase3_basin/diagnostic_basin_width.py` (`D21`, `D22`)**: Added strict checkpoint key verification, `bf16` inference, post-EOS valid-token masking (`build_valid_mask`), and trajectory-direction perturbation curves alongside isotropic Gaussian perturbations.
2. **`src/train_step.py` L239/L281 (`D20` Bug 1)**: Fixed `NameError: name 'x_pred' is not defined` by unpacking `v_pred, x_pred = net_out_to_v_x(...)`.
3. **`src/configs/config.py` (`D20` Bug 2)**: Added explicit `BWELFConfig` dataclass fields matching `src/configs/bw_elf_finetune.yaml`.
4. **`src/train_step.py` L250, L281, L301 (`D20` Bug 3)**: Passed `self_cond_cfg_scale=sc_scale` into every `decode_z(...)` call so the 4 SC-CFG conditioning tokens are prepended and RoPE positions are not shifted by 4 tokens.
5. **`src/train_step.py` L322 (`D20` Bug 4)**: Replaced the 8.4 GB fp32 `F.one_hot(input_ids, V)` tensor in `loss_margin` with in-register `logits.scatter(-1, idx, -1e9).max(dim=-1)`.
6. **`src/train_step.py` L281 & L301 (`D20` Bug 5)**: Weighted `loss_int_ce` by $t^2 \mathbf{1}[t \ge 0.3]$ and aligned `loss_robust` perturbations with the ODE trajectory error direction $\hat{x}_1(z_t) - x_0$.

## 2. BW-ELF Fine-Tuning Ablation Summary
| Arm | Total Loss | FM Loss | Int-CE Loss | Robust Loss | Margin Loss | Post-FT Traj Flip (@0.20) |
|---|---:|---:|---:|---:|---:|---:|
| A0_baseline | 0.4567 | 0.4567 | 0.0000 | 0.0000 | 0.0000 | 0.0107 |
| A1_robust_traj | 0.4573 | 0.4573 | 0.0000 | 0.0001 | 0.0000 | 0.0134 |
| A1b_robust_iso | 0.4583 | 0.4583 | 0.0000 | 0.0001 | 0.0000 | 0.0117 |
| A2_margin_only | 0.4581 | 0.4581 | 0.0000 | 0.0000 | 0.0000 | 0.0107 |
| A3_int_ce_weighted | 0.4604 | 0.4590 | 0.0271 | 0.0000 | 0.0000 | 0.0142 |
| A4_full_bw_elf | 0.4627 | 0.4611 | 0.0308 | 0.0002 | 0.0000 | 0.0139 |