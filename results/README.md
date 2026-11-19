# Organized Experimental Results Directory (`results/`)

To make browsing and auditing straightforward while preserving exact backwards-compatible paths required by the automated verification scripts (`analysis/` and `tests/`), all experimental outputs from the Google Colab notebooks are organized into four clear tiers:

---

## Directory Structure

```text
results/
├── README.md                                 # This overview and index
├── stage2_colab_audits/                      # Consolidated statistical JSONs from Notebooks 03–14
│   ├── reanalysis_stats.json                    # Master statistical reanalysis across 314 runs
│   ├── authoritative_experiment_inventory.json
│   ├── c1_c3_per_seed_recovered.json         # Notebook 11: 30 per-seed C1–C3 (201–205) summaries & CRN digests
│   ├── control_path_stats.json               # Integrated ELF-B & ELF-M control-path stats
│   ├── final_consolidation_stats.json        # Master stats for Tables 1–23 (Notebooks 03–14)
│   ├── iso_frontier_sensitivity.json         # Stage-1 40-run response-surface predictions & residuals
│   ├── key_numbers.json                      # Key frontier & intervention numbers checked by check_manuscript.py
│   ├── nb12_colab_statistical_reanalysis.json# Notebook 12: Colab paired CRN / Welch TOST & blocked ANCOVA
│   └── second_stage_stats.json               # Notebooks 03–10: ANOVA, MixedLM, ELF-L, NFE timing, ACE, Vendi
│
├── stage3_confirmatory_revision/             # Outputs from Stage-3 Confirmatory Colab Notebooks 15–21 (mirrors runs/)
│   ├── heldout_reference/                    # Notebook 17: 13-gram MinHash contamination audit & held-out WebText/CC-News MAUVE
│   ├── llm_judge_panel/                      # Notebook 15: 3-model NF4 LLM-judge panel (HANNA gate & C1/C3 pairs)
│   ├── p21_matched_ace/                      # Notebook 21: 20-direction random null & matched ACE (gamma*=0.9455)
│   ├── plaid_eval/                           # Notebooks 18–19: Plaid-1B 4x4 response surface & 2D NO-GO gate
│   └── replication_c1_c3/                    # Notebook 16: 20-seed confirmatory replication (seeds 401–420, 120 runs)
│
├── external_exploratory_tracks/              # Outputs from Exploratory P5–P9 Colab Tracks (mirrors external_results/)
│   ├── mdlm_at_scale_live_summary.json       # Notebook P8: MDLM-130M vs. ELF-B cross-paradigm sweep (Table 18, Fig. 5)
│   ├── mdlm_stochastic/_report.md            # Notebook P8: Detailed cross-paradigm step-budget report
│   ├── elf_p5_distill_live_summary.json      # Notebook P5-distill: WMT14 distillation gate (excluded negative gate)
│   ├── bw_elf_105m_live_summary.json         # Notebook P6-basin: 60-step basin-width fine-tune (excluded)
│   ├── efm_p7_live_summary.json              # Notebook P7-efm: Toy synthetic & 3.4M scale-up sweep (excluded)
│   ├── elf_p9_live_summary.json              # Notebook P9-basin-gates: Decoder basin entry & adapter gates (excluded)
│   └── elf_basin_gates/                      # Notebook P9-basin-gates: Detailed report and summary JSON
│
└── elf_interface_audit.json                  # Interface verification summary
```

---

## Where Raw Per-Run `summary.json` Directories Live (`content/drive/MyDrive/elf_research/runs/` & `p5_non_identifiability_artifacts/`)

Because the Google Colab notebooks mounted Google Drive at `/content/drive/MyDrive/elf_research/runs/` and the analysis scripts (`analysis/frontier_analysis.py`, `analysis/statistical_reanalysis.py`, `analysis/make_figures_tables.py`) read directly from that relative path:

- **`content/drive/MyDrive/elf_research/runs/`** preserves all **330+ Stage-1 and follow-up per-run directories** (`R0_owt_ref`, `p4_elfb_*`, `p4_elfm_*`, `p4_elfl_*`, `ecp_s1_*`..`ecp_s5_*`, `cmlc_eval/*`, `mdlm_stochastic/`, `elf_basin_gates/`).
- **`p5_non_identifiability_artifacts/runs/`** and **`p6_followup_artifacts.zip`** preserve the **64 extended-metric (`extended_metrics.json`) and 24 ELF-M $2\times 2$ factorial run directories** from `elf_p5_non_identifiability.ipynb` and `elf_p6_followup.ipynb`.
- **`runs/`** at the repository root holds the canonical Stage-3 confirmatory outputs (`replication_c1_c3/`, `heldout_reference/`, `llm_judge_panel/`, `p21_matched_ace/`, `plaid_eval/`), mirrored in `results/stage3_confirmatory_revision/`.

For a complete notebook-by-notebook mapping from each `.ipynb` file to its results and paper tables/figures, see [`../notebooks/NOTEBOOK_TO_RESULTS_AND_PAPER_MAP.md`](../notebooks/NOTEBOOK_TO_RESULTS_AND_PAPER_MAP.md).
