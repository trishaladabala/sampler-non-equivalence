# Exclusive Notebook 9 — 105M ELF Decoder-Basin Investigation (Gates 1–5) Report

## 1. Critical Codebase Bugs Fixed across `elf-basin/` Gates 1–5
1. **Gate 1 (`elf-basin/scripts/gate1_verify.py` L174–193, `D22`)**: Restricted basin-entry times $\tau^*(b, i)$ strictly to valid content tokens (`pos <= first_eos_idx` & `cond_mask == 0`) and added within-sequence entry IQR, eliminating the post-EOS `<pad>` dilution that skewed DE-EN median entry time to `0.125`.
2. **Gates 2 & 3 (`elf-basin/scripts/harvest_deen.py` & `phase3_headroom.py` L148–153, `D26`)**: Fixed missing 3D conditional source mask (`build_self_attn_cond_masks`) during reference encoding, fixed undefined `gt_tokens_mmap`, and separated ground-truth reference autoencoding (`84.33 BLEU`) from true trajectory headroom.
3. **Gate 4 (`elf-basin/scripts/train_adapter.py`, `D25`)**: Replaced the sign-flipped `target_x = 2*x_hat - x_ref` target and scratch 32,100-vocab projection head with zero-initialized `ResidualBasinAdapter` variants (`V1`–`V5`) before the frozen 105M decoder.
4. **Gate 5 / Phase 5 (`elf-basin/src/anchored_sampler.py`, `phase5_pilot.py`, `phase5_control.py`, `D25`)**: Fixed `_reencode_tokens` passing a 2D mask instead of 3D `cond_seq_mask`, restored `x_sc` self-conditioning across ODE steps, and moved confidence-selective anchoring *before* the Euler velocity step.
