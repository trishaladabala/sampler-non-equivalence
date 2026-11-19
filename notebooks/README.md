# Google Colab Execution Runbook: Research Roadmap Revision Suite

**Governing Directives:** ARR 2027 Remediation Round 2 Directive (Items A.1–A.8, E), `RESEARCH_ROADMAP.md`, `prereg/PREREGISTRATION.md`  
**Execution Environment:** Google Colab Pro / Pro+ (NVIDIA A100-SXM4-40GB / 80GB GPU)  
**Persistent Storage:** Google Drive mounted at `/content/drive/MyDrive/elf_research`  
**User Compute Credit Balance:** ~500 Compute Units (CU)  
**Total Plan Estimate:** ~144.7 CU (Full suite including Plaid-1B study) / ~77.9 CU (Core baseline), leaving >350 CU in reserve.

---

## 1. Overview & Execution Workflow

The revision suite consists of six dedicated Jupyter notebooks (`notebooks/15` through `notebooks/21`) designed for turn-key execution in Google Colab. Each notebook implements the production branch end-to-end, writes versioned output manifests with cryptographic SHA-256 digests, supports a ~2-minute GPU preflight mode, and fails closed upon missing dependencies or corrupted checkpoints.

### Quick Reference: Exactly Which Notebook to Open in Colab

To eliminate ambiguity when uploading to Google Drive or Colab, each experimental step corresponds to **exactly one canonical `.ipynb` notebook file**. Former duplicate `_colab_` copies have been moved to `notebooks/archive_superseded/` to avoid any file confusion.

| Execution Step | Roadmap Milestone | Canonical File to Open in Colab | Primary Artifact Emitted on Google Drive |
|---|---|---|---|
| **Step 1 (Pilot)** | WS2 Pretraining-Excluded Split | `notebooks/17_heldout_reference_and_contamination_audit.ipynb` | `runs/ref_eval/pretraining_excluded_eval_manifest.json` |
| **Step 2** | WS3' Validated LLM Judges | `notebooks/15_final_human_evaluation.ipynb` | `runs/judge_eval/validated_llm_judge_manifest.json` |
| **Step 3** | WS1 Confirmatory Replication | `notebooks/16_preregistered_c1_c3_replication.ipynb` | `runs/replication_eval/replication_analysis_report.json` |
| **Step 4** | WS5 Matched ACE Intervention | `notebooks/21_ace_null_distribution_and_matched_ace.ipynb` | `runs/ace_eval/matched_ace_summary.json` |
| **Step 5** | WS4 Plaid Feasibility Spike | `notebooks/18_plaid_feasibility_spike.ipynb` | `runs/plaid_eval/plaid_spike_response_surface.json` |
| **Step 6** | WS4 Plaid Equivalence Study | `notebooks/19_plaid_heldout_equivalence.ipynb` | `runs/plaid_eval/plaid_heldout_equivalence_report.json` |

*Note: Do NOT use symlinks. Open only the canonical files above.*

### Recommended Execution Sequence

```
                        ┌─────────────────────────────────────────────────────────┐
                        │ Step 1: Pre-Execution Validation                        │
                        │ Upload and timestamp `prereg/PREREGISTRATION.md` on OSF │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 │                                                                     │
                 ▼                                                                     ▼
┌─────────────────────────────────┐                                   ┌─────────────────────────────────┐
│ NB17: Pretraining-Excluded Ref  │                                   │ NB15: Validated Automated       │
│ (Pilot Mode: seeds 201–205)     │                                   │       LLM-Judge Panel           │
│ Est: 0.38 h | 4.6 CU            │                                   │ Est: 0.42 h | 5.0 CU            │
└────────────────┬────────────────┘                                   └─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ NB16: Confirmatory Replication  │
│ (Seeds 401–420, 120 runs)       │
│ Est: 2.87 h | 34.4 CU           │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ NB17: Pretraining-Excluded Ref  │
│ (Replication Mode: +120 runs)   │
│ Est: 0.57 h | 6.8 CU            │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ NB21: Matched ACE Intervention  │
│ (20-Dir Null, 4k Calib, TOST)   │
│ Est: 1.49 h | 17.9 CU           │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ NB18: Plaid-1B Feasibility      │
│ (4x4 Grid, Spike Gate Decision) │
│ Est: 0.77 h | 9.2 CU            │
└────────────────┬────────────────┘
                 │
         [GATE == GO?]
        /             \
      YES              NO
      /                 \
     ▼                   ▼
┌─────────────────────┐  ┌─────────────────────────────────┐
│ NB19: Plaid Held-Out│  │ Complete Pipeline.              │
│ Equivalence Study   │  │ Document boundary conditions in │
│ Est: 5.57 h|66.8 CU │  │ manuscript Sec. 4.             │
└─────────────────────┘  └─────────────────────────────────┘
```

> **Important Ordering Note:** NB17 operates in dual mode. It can be executed in **Pilot Mode** first (evaluating existing pilot seeds 201–205 and building contamination-free reference splits in ~23 min / 4.6 CU) *before* NB16. Once NB16 completes generating replication seeds 401–420, NB17 can be re-run in **Replication Mode** (+6.8 CU) to finalize held-out MAUVE on the new replication seeds.

---

## 2. Empirical Hardware Benchmarking & Closed-Form Runtime Equations

All runtime and Compute Unit estimates in this runbook are grounded in synchronized hardware benchmarks executed on NVIDIA A100-SXM4-40GB under CUDA 12.x and PyTorch 2.5 (archived in `notebooks/07_cuda_hardware_timing_benchmark.ipynb` and `reports/FINAL_HARDWARE_AND_NFE_AUDIT.md`).

### 2.1 Measured Throughput Benchmark
- **Target GPU:** NVIDIA A100-SXM4-40GB / 80GB (Colab Pro / Pro+).
- **Consumption Rate:** $11.0\text{--}13.0\text{ Compute Units / Hour}$ (nominal $12.0\text{ CU/h} = 0.20\text{ CU/minute}$).
- **Benchmarked Generation Latency (ELF-B, $K=32$ steps, batch size $B=64$, chunk size 32):**
  Measured across 5 distinct $(\text{SC-CFG}, \gamma)$ conditions with synchronized CUDA events (`torch.cuda.Event(enable_timing=True)` after warmup):
  $$T_{\mathrm{batch}} = 3.384 \pm 0.016\text{ seconds per 64-sequence batch}$$
  $$\text{Throughput} = \frac{64}{3.384} \approx 18.91\text{ sequences / second}$$

### 2.2 Closed-Form Per-Run Runtime Equation for ELF-B ($M = 1,000$ Samples)
Every complete evaluation run on ELF-B at $M = 1,000$ consists of the following deterministic pipeline stages:
1. **Sample Generation ($1,000$ sequences):**
   $$N_{\mathrm{batches}} = \left\lceil \frac{1000}{64} \right\rceil = 16\text{ batches}, \quad T_{\mathrm{gen}} = 16 \times 3.384\text{ s} = 54.14\text{ seconds}$$
2. **Autoregressive Scoring (`score_gpt2`, GPT-2 Large 774M in `bfloat16`):**
   $$T_{\mathrm{score}} \approx 14.0\text{ seconds}$$
3. **Feature Extraction (`gpt2_features`, 774M last hidden state, $B=16$):**
   $$T_{\mathrm{feat}} \approx 11.0\text{ seconds}$$
4. **Secondary Metrics (`rep-4`, $\mathrm{JSD}_\pi$, `distinct-2`):**
   $$T_{\mathrm{lexical}} \approx 2.0\text{ seconds}$$
5. **MAUVE Computation ($1,000 \times 1,280$ embeddings, $k=100$):**
   $$T_{\mathrm{mauve}} \approx 3.0\text{ seconds}$$
6. **I/O & Cryptographic Manifest Writing (`summary.json`, `samples.jsonl`):**
   $$T_{\mathrm{io}} \approx 1.0\text{ second}$$

$$\mathbf{T_{\mathrm{run, ELF-B}}} = 54.14 + 14.0 + 11.0 + 2.0 + 3.0 + 1.0 = \mathbf{85.14\text{ seconds}} \approx \mathbf{1.42\text{ minutes}}$$

---

## 3. Detailed Per-Notebook Specifications & Compute Accounting

#### 3.1 Notebook 16: Preregistered Confirmatory Replication C1–C3
- **Canonical Target:** `notebooks/16_preregistered_c1_c3_replication.ipynb`  
- **Archived Duplicate:** `notebooks/archive_superseded/16_colab_confirmatory_replication.ipynb`  
- **Scientific Objective:** Execute the OSF-preregistered confirmatory replication of operating pairs C1, C2, and C3 on 20 strictly held-out seeds (401–420) under cryptographic Three-Stream Common Random Numbers (CRN).
- **Experimental Workload:**
  - 3 pairs (C1, C2, C3) $\times$ 2 arms (Arm A, Arm B) $\times$ 20 replication seeds (401–420) = **120 complete runs**.
  - Sample size: $M = 1,000$ sequences per run ($120,000$ total sequences) on `ELF-B` ($K=32$ steps, $\mathrm{NFE}=33$).
- **Features & Safety Mechanisms:**
  - **Resume-Safe Checkpointing:** Automatically inspects `/content/drive/MyDrive/elf_research/runs/` and skips any seed whose `summary.json` is already written and valid. Network disconnections in Colab do not waste compute units.
  - **Samples Archival:** Persists full generated text sequences to `samples.jsonl` on Drive with SHA-256 digests.
  - **Three-Stream CRN Verification:** Explicitly audits the bit-identical SHA-256 digests of initial latent states ($\mathbf{z}_0$), time schedules ($\mathbf{t}$), and SDE Brownian increments ($\boldsymbol{\epsilon}$) across Arm A and Arm B before running each pair.
  - **Preregistered Gatekeeping:** Executes hierarchical gatekeeping: Family 1 (3 joint TOSTs on Gen-PPL and $H_{\mathrm{ps}}$ at Holm $\alpha = 0.05$) $\to$ Family 2 (9 tests on `rep-4`, $\mathrm{JSD}_\pi$, $\mathrm{MAUVE}_{\mathrm{heldout}}$ at Holm $\alpha = 0.05$).
- **Wall Time & Compute Units:**
  $$T_{\mathrm{gen+score}} = 120 \times 85.14\text{ s} = 10,217\text{ s} \approx 170.3\text{ minutes}$$
  With startup, model initialization, and resume check overhead: **$2.87\text{ hours} \approx 172\text{ minutes}$**.
  $$\text{CU} = 2.87\text{ h} \times 12.0\text{ CU/h} = \mathbf{34.4\text{ CU}} \quad (\text{safe budget cap: } 40\text{ CU})$$

---

### 3.2 Notebook 17: Pretraining-Excluded Reference Corpus & Contamination Audit
- **Canonical Target:** `notebooks/17_heldout_reference_and_contamination_audit.ipynb`  
- **Archived Duplicate:** `notebooks/archive_superseded/17_colab_pretraining_excluded_eval_split.ipynb`  
- **Scientific Objective:** Eliminate pretraining contamination concerns by building a verified pretraining-excluded reference split, executing a 13-gram streaming MinHash filter against `openwebtext-t5` training shards, and recomputing distributional metrics ($\mathrm{MAUVE}$ and $\mathrm{CTW\ JCD}$).
- **Experimental Workload:**
  1. Download Reference A (`webtext.test.jsonl` from OpenAI `gpt-2-output-dataset`) and CC-News post-cutoff slice: $\sim 4.0\text{ min}$.
  2. 13-gram streaming Bloom filter / hashed set contamination filter against `openwebtext-t5` shards: $\sim 10.0\text{ min}$.
  3. Precompute reference features ($N_{\mathrm{ref}} = 1,000$ texts for Reference A and CC-News): $\approx 0.4\text{ min}$.
  4. Distributional metric evaluation across saved `samples.jsonl`:
     - **Pilot Mode (Seeds 201–205, 30 runs):**
       $$T_{\mathrm{pilot}} = 30 \text{ runs} \times 17\text{ s} = 510\text{ s} = 8.5\text{ minutes}$$
       $$\text{Total Pilot Time} = 4.0 + 10.0 + 0.4 + 8.5 = 22.9\text{ minutes} \approx 0.38\text{ hours}$$
       $$\mathbf{\text{CU}_{\mathrm{pilot}}} = 0.38\text{ h} \times 12.0\text{ CU/h} = \mathbf{4.6\text{ CU}} \quad (\text{safe budget cap: } 6\text{ CU})$$
     - **Replication Mode (Seeds 401–420, +120 runs after NB16):**
       $$T_{\mathrm{repl}} = 120 \text{ runs} \times 17\text{ s} = 2,040\text{ s} = 34.0\text{ minutes} \approx 0.57\text{ hours}$$
       $$\mathbf{\text{CU}_{\mathrm{repl}}} = 0.57\text{ h} \times 12.0\text{ CU/h} = \mathbf{6.8\text{ CU}} \quad (\text{safe budget cap: } 10\text{ CU})$$
- **Total Notebook 17 (Full Suite):** $56.9\text{ minutes} \approx 0.95\text{ hours} \to \mathbf{11.4\text{ CU}}$ (safe cap: 16 CU).

---

#### 3.3 Notebook 15: Validated Automated LLM-Judge Panel
- **Canonical Target:** `notebooks/15_final_human_evaluation.ipynb`  
- **Archived Duplicate:** `notebooks/archive_superseded/15_colab_validated_llm_judge_panel.ipynb`  
- **Scientific Objective:** Replace speculative human evaluation with a preregistered, multi-judge panel of open-weights language models (`Qwen2.5-7B-Instruct`, `Meta-Llama-3.1-8B-Instruct`, and `Mistral-7B-Instruct-v0.3`) loaded in 4-bit NF4 (`bitsandbytes`), validated on published human benchmarks before scoring model samples.
- **Experimental Workload:**
  1. Load and initialize 3 NF4 evaluator models: $3 \times 3.0\text{ min} = 9.0\text{ min}$.
  2. Grounding benchmark validation on published HANNA dataset (50 items $\times$ 3 judges = 150 evaluations): $\approx 2.5\text{ min}$ (enforces preregistered gate: mean agreement $\ge 70\%$).
  3. Position-swapped pairwise evaluations on stratified C1/C3 passages (120 pairs $\times$ 2 swaps $\times$ 3 judges = 720 queries): $\approx 12.0\text{ min}$.
  4. Set-level diversity evaluations (30 sets $\times$ 3 judges = 90 queries): $\approx 1.5\text{ min}$.
  5. Mixed-effects logit regression analysis: $\approx 0.5\text{ min}$.
- **Wall Time & Compute Units:**
  $$\text{Total Time} = 9.0 + 2.5 + 12.0 + 1.5 + 0.5 = 25.5\text{ minutes} \approx 0.42\text{ hours}$$
  $$\text{CU} = 0.42\text{ h} \times 12.0\text{ CU/h} = \mathbf{5.0\text{ CU}} \quad (\text{safe budget cap: } 8\text{ CU})$$

---

### 3.4 Notebook 18: Plaid-1B Cross-Architecture Feasibility Spike
- **Canonical Target:** `notebooks/18_plaid_feasibility_spike.ipynb`  
- **Archived Duplicate:** `notebooks/archive_superseded/18_colab_plaid_1b_feasibility.ipynb`  
- **Scientific Objective:** Investigate whether operating-point non-equivalence generalizes to an independent continuous diffusion architecture (Plaid-1B, 1.3B parameters, Gulrajani & Hashimoto, NeurIPS 2023) across real inference controls: Score Temperature $T_{\mathrm{score}} \in \{0.8, 1.0, 1.2, 1.5\}$ and Discretization Steps $K \in \{32, 64, 128, 256\}$.
- **Experimental Workload:**
  1. Clone `github.com/igul222/plaid`, download official split release weights (`v1.0.0`, 2.4 GB archive), and extract: $\sim 6.0\text{ min}$.
  2. 4 temperatures $\times$ 4 steps $\times$ 2 seeds (101, 102) = **32 evaluated grid points** ($M = 64$ sequences per point).
     On 1.3B Plaid, 64 samples at average $K=120$ takes $\sim 65\text{ s}$ generation $+ 8\text{ s}$ scoring $\approx 73\text{ s}$ per point.
     $$32 \times 73\text{ s} = 2,336\text{ s} \approx 39.0\text{ minutes}$$
  3. Response surface analysis and automated Go/No-Go gate calculation:
     - Iso-contour matching threshold: $|\Delta\text{Gen-PPL}| \le 0.50$ and $|\Delta H_{\mathrm{ps}}| \le 0.020$ nats.
     - Go condition: At least one matched pair satisfies $|\Delta\texttt{rep-4}| \ge 0.010$ ($1.0$ percentage point).
     - Manifest emission: Writes `runs/plaid_eval/plaid_spike_response_surface.json` with `GATE_DECISION: GO` or `NO-GO`.
- **Wall Time & Compute Units:**
  $$\text{Total Time} = 6.0 + 39.0 + 1.0 = 46.0\text{ minutes} \approx 0.77\text{ hours}$$
  $$\text{CU} = 0.77\text{ h} \times 12.0\text{ CU/h} = \mathbf{9.2\text{ CU}} \quad (\text{safe budget cap: } 20\text{ CU})$$

---

### 3.5 Notebook 19: Plaid-1B Held-Out Operating-Point Equivalence Study
- **Canonical Target:** `notebooks/19_plaid_heldout_equivalence.ipynb`  
- **Archived Duplicate:** `notebooks/archive_superseded/19_colab_plaid_operating_point_equivalence.ipynb`  
- **Scientific Objective:** Conditional on `GATE_DECISION == GO` in Notebook 18, execute the full confirmatory TOST study on Plaid-1B across 10 held-out seeds.
- **Precondition Check:** Enforces `verify_plaid_spike_gate()`. Fails closed (`PlaidGateNotPassedError`) if Notebook 18 yielded `NO-GO`.
- **Experimental Workload:**
  1. Calibration on seeds 0–4 (fine-tuning matched pairs): $\approx 22.0\text{ min}$.
  2. 2 pairs $\times$ 2 arms $\times$ 10 held-out seeds (201–210) = **40 runs** ($M = 500$ sequences/run).
     Plaid-1B generation ($M=500, K=64\text{--}128$): $\sim 7.5\text{ min}$ per run.
     Scoring ($M=500$): $\sim 15\text{ s}$ per run.
     $$40 \times 7.75\text{ min} = 310.0\text{ minutes} \approx 5.17\text{ hours}$$
  3. TOST statistical inference and summary generation: $\sim 2.0\text{ min}$.
- **Wall Time & Compute Units:**
  $$\text{Total Time} = 22.0 + 310.0 + 2.0 = 334.0\text{ minutes} \approx 5.57\text{ hours}$$
  $$\text{CU} = 5.57\text{ h} \times 12.0\text{ CU/h} = \mathbf{66.8\text{ CU}} \quad (\text{safe budget cap: } 100\text{ CU})$$

---

### 3.6 Notebook 21: Matched ACE & Null Distribution
- **Canonical Target:** `notebooks/21_ace_null_distribution_and_matched_ace.ipynb`  
- **Archived Duplicate:** `notebooks/archive_superseded/21_colab_matched_ace_intervention.ipynb`  
- **Scientific Objective:** Evaluate the ACE-inspired self-conditioning projection against a 20-direction random orthogonal null distribution, run a published-scale 4,096-trajectory calibration on 5 seeds, and evaluate gamma-compensated matched-operating-point ACE under Joint TOST.
- **Experimental Workload:**
  1. **20-Direction Random Orthogonal Null:**
     20 directions $\times$ 5 seeds = 100 runs ($M=500$ on ELF-B, $K=32$):
     Per run: 8 batches $\times 3.384\text{ s} = 27.1\text{ s}$ gen $+ 12.0\text{ s}$ scoring $= 39.1\text{ s} \approx 0.65\text{ min}$.
     $$100 \times 0.65\text{ min} = 65.0\text{ minutes}$$
  2. **Published-Scale 4,096-Trajectory Calibration:**
     5 seeds $\times$ 4,096 trajectories: 64 batches $\times 3.384\text{ s} = 216.6\text{ s} \approx 3.6\text{ minutes}$.
  3. **Gamma-Compensated Matched ACE Evaluation:**
     Calibrate $\gamma^*$ on 2 seeds ($\approx 6.5\text{ min}$), evaluate on 5 held-out seeds $\times$ 2 arms ($M=1,000$): $10 \text{ runs} \times 1.42\text{ min} = 14.2\text{ min}$.
- **Wall Time & Compute Units:**
  $$\text{Total Time} = 65.0 + 3.6 + 6.5 + 14.2 = 89.3\text{ minutes} \approx 1.49\text{ hours}$$
  $$\text{CU} = 1.49\text{ h} \times 12.0\text{ CU/h} = \mathbf{17.9\text{ CU}} \quad (\text{safe budget cap: } 25\text{ CU})$$

---

## 4. Consolidated Compute Budget Reconciliation

The closed-form calculations above establish an internally consistent, rigorous accounting across all experimental workstreams (all superseded duplicate files are archived in `notebooks/archive_superseded/`):

| Workstream | Canonical Target Notebook | Experimental Scope | Runs / Tasks | Wall Time | Nominal CU (12 CU/h) | Safe Budget Cap | Status |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **WS2 (Pilot)** | `17_heldout_reference_and_contamination_audit.ipynb` | Contamination Filter & Pilot Ref Scoring | 30 runs | 0.38 h | **4.6 CU** | 6 CU | Turn-key Ready |
| **WS3'** | `15_final_human_evaluation.ipynb` | NF4 3-Judge LLM Panel & HANNA Gate | 720 pairs | 0.42 h | **5.0 CU** | 8 CU | Turn-key Ready |
| **WS1** | `16_preregistered_c1_c3_replication.ipynb` | Confirmatory Replication C1–C3 | 120 runs | 2.87 h | **34.4 CU** | 40 CU | Turn-key Ready |
| **WS2 (Repl)** | `17_heldout_reference_and_contamination_audit.ipynb` | Held-Out Scoring on Replication Seeds | +120 runs | +0.57 h | **+6.8 CU** | +10 CU | Turn-key Ready |
| **WS5** | `21_ace_null_distribution_and_matched_ace.ipynb` | 20-Dir Null, 4k Calib, Matched ACE | 110 runs | 1.49 h | **17.9 CU** | 25 CU | Turn-key Ready |
| **WS4 (Spike)**| `18_plaid_feasibility_spike.ipynb` | Plaid-1B 4x4 Grid Feasibility Spike | 32 points | 0.77 h | **9.2 CU** | 20 CU | Turn-key Ready |
| **WS4 (Study)**| `19_plaid_heldout_equivalence.ipynb` | Plaid-1B Held-Out Equivalence (if GO)| 40 runs | 5.57 h | **66.8 CU** | 100 CU | Conditional on NB18 |
| **CORE TOTAL** | *(NB15, NB16, NB17, NB18, NB21)* | **Mandatory Revision Baseline** | **292 runs** | **6.50 h** | **~77.9 CU** | **< 100 CU** | **Turn-key Ready** |
| **FULL TOTAL** | *(All including NB19 Plaid Study)* | **Complete Experimental Suite** | **332 runs** | **12.07 h** | **~144.7 CU** | **< 200 CU** | **Turn-key Ready** |

### Budget Verdict
- **User's Available Colab Balance:** **~500 Compute Units**
- **Full Suite Consumption:** **~144.7 Compute Units**
- **Remaining Balance in Reserve:** **>350 Compute Units**
- **Conclusion:** The experimental workload is fully feasible, highly efficient, and leaves substantial buffer for any potential Colab VM preemptions or re-runs.

---

## 5. Google Colab User Execution Instructions

### Step 0: Google Drive Mounting & Workspace Verification
Open Google Colab with an **A100 GPU** runtime (`Runtime > Change runtime type > GPU > A100`).

In each notebook, Cell 1 mounts Google Drive and sets up the root path:
```python
from google.colab import drive
import os, sys

drive.mount('/content/drive')
RESEARCH_DIR = '/content/drive/MyDrive/elf_research'
os.makedirs(RESEARCH_DIR, exist_ok=True)
sys.path.insert(0, '/content/7semminor')
```

### Step 1: Rapid 2-Minute GPU Preflight Mode
Before executing the full production sweep, set `RUN_SMOKE_CPU = False` and `RUN_PREFLIGHT_GPU = True` in the notebook configuration cell:
```python
RUN_SMOKE_CPU = False      # Activates the real GPU pipeline
RUN_PREFLIGHT_GPU = True   # Executes tiny real test (M=8, K=4, 1 seed) in ~2 minutes
```
Run all cells. Verify that:
1. Checkpoints and weights load into GPU VRAM without error.
2. Tensor operations execute cleanly on CUDA.
3. Scoring and metric functions complete and emit test JSONs.

### Step 2: Full Production Run
Set `RUN_PREFLIGHT_GPU = False`:
```python
RUN_SMOKE_CPU = False
RUN_PREFLIGHT_GPU = False  # Full production run according to preregistered specifications
```
Run all cells. The notebook will automatically:
- Execute all scheduled runs with live progress logging.
- Skip already completed runs if resumed.
- Persist `summary.json`, `extended_metrics.json`, and `samples.jsonl` to `/content/drive/MyDrive/elf_research/runs/`.
- Compute all confirmatory TOST statistics and hierarchical gatekeeping tests.
- Emit final JSON manifests with SHA-256 digests.

### Step 3: Local Integration
Once the Colab run completes:
1. Sync or copy the output JSON manifests from `/content/drive/MyDrive/elf_research/` into the local repository (`analysis/out/` and `runs/`).
2. Run the provenance audit and table rebuild:
   ```bash
   ./ELF/.venv/bin/python3 analysis/final_consolidation_audit.py
   ./ELF/.venv/bin/python3 analysis/generate_revised_tables.py
   pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex main && pdflatex main
   ```
3. The manuscript will seamlessly integrate the new experimental results, replacing the temporary pending execution boxes with the confirmatory tables and figures.
