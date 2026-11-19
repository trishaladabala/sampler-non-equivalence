# Exclusive Notebook 7 — EFM Local-Time Compression (`W=8`), Second Synthetic Tasks, & Scale-Up Report

## 1. Critical Codebase Bugs Fixed in `efm_pipeline/`
1. **`eval_runner.py` L179–193 & `scaleup_matrix.py` L187/214 (`D25`)**: Fixed `quantized` and `lowrank` modes falling through to `else: local_times = t` (global scalar broadcast) during ODE sampling, and fixed `continuous` mode dividing velocity by global `1 - t` instead of per-token `1 - tau_i`.
2. **`train_efm.py` L394 & `synthetic_runner.py` L38–53**: Fixed training and generation ignoring per-sample `wave_ids` and causal wave ordering $\tau_i(t) = \text{clamp}(W \cdot t - w_i, 0, 1)$.
3. **`ground_truth_eval.py` L58 (`D26`)**: Fixed `internal_dependency_accuracy` rewarding constant all-equal sequences (`c1 == c2 == c3 == c4`) with `1.0`.

## 2. Part A — Phase 1 `W x K` Local-Time Compression Sweep
| W | Mode | K | Final Loss | TF Acc | Non-Degenerate Rule Acc | Degeneracy Rate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ELF_global | 1 | 0.4206 | 0.887 | 0.000 | 0.000 |
| 4 | continuous | 8 | 1.1393 | 0.584 | 0.000 | 0.000 |
| 4 | quantized | 1 | 1.2126 | 0.572 | 0.008 | 0.000 |
| 4 | quantized | 2 | 1.1774 | 0.574 | 0.008 | 0.000 |
| 4 | quantized | 4 | 1.1534 | 0.607 | 0.000 | 0.000 |
| 4 | quantized | 8 | 1.1352 | 0.605 | 0.004 | 0.000 |
| 4 | lowrank | 4 | 1.1301 | 0.598 | 0.000 | 0.000 |
| 8 | continuous | 8 | 1.2111 | 0.559 | 0.004 | 0.000 |
| 8 | quantized | 1 | 1.2412 | 0.570 | 0.016 | 0.000 |
| 8 | quantized | 2 | 1.2349 | 0.561 | 0.000 | 0.000 |
| 8 | quantized | 4 | 1.2114 | 0.584 | 0.004 | 0.000 |
| 8 | quantized | 8 | 1.1957 | 0.566 | 0.000 | 0.000 |
| 8 | lowrank | 4 | 1.2094 | 0.570 | 0.004 | 0.000 |
| 16 | continuous | 8 | 1.2508 | 0.549 | 0.000 | 0.000 |
| 16 | quantized | 1 | 1.2834 | 0.541 | 0.004 | 0.000 |
| 16 | quantized | 2 | 1.2567 | 0.543 | 0.023 | 0.000 |
| 16 | quantized | 4 | 1.2326 | 0.549 | 0.016 | 0.000 |
| 16 | quantized | 8 | 1.2304 | 0.555 | 0.008 | 0.000 |
| 16 | lowrank | 4 | 1.2427 | 0.566 | 0.004 | 0.000 |

## 3. Part B — `second_synthetic_task.md` Option A (KV Copy) & Option C (Sorting)
| Task | Variant | W | Mode | K | Primary Metric |
|---|---|---:|---|---:|---:|
| Option A (KV Copy) | ELF_Global_W1 | 1 | none | 1 | KV Copy Acc = 0.168 |
| Option A (KV Copy) | EFM_Quantized_W8_K4 | 8 | quantized | 4 | KV Copy Acc = 1.000 |
| Option A (KV Copy) | EFM_LowRank_W8_K4 | 8 | lowrank | 4 | KV Copy Acc = 0.355 |
| Option A (KV Copy) | EFM_Continuous_W8 | 8 | continuous | 8 | KV Copy Acc = 0.359 |
| Option C (Sorting) | ELF_Global_W1 | 1 | none | 1 | Monotonicity = 0.109 (Elem Acc = 0.223) |
| Option C (Sorting) | EFM_Quantized_W8_K4 | 8 | quantized | 4 | Monotonicity = 0.625 (Elem Acc = 0.570) |
| Option C (Sorting) | EFM_LowRank_W8_K4 | 8 | lowrank | 4 | Monotonicity = 0.609 (Elem Acc = 0.484) |
| Option C (Sorting) | EFM_Continuous_W8 | 8 | continuous | 8 | Monotonicity = 0.625 (Elem Acc = 0.555) |
