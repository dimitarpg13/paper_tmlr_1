# Colab Notebooks

All Colab-ready notebooks clone from
`https://github.com/dimitarpg13/paper_tmlr_1.git`, mount Google Drive for
persistent output, and auto-detect the available GPU (CUDA preferred, MPS/CPU
fallback). Both are **dual-mode**: they run unchanged in a local checkout,
writing results under `notebooks/conservative_arch/scripts/results/` instead
of GDrive.

---

## Colab-Ready Notebooks

### r2_jsons_a100_h100

**Path:** `notebooks/conservative_arch/scripts/r2_jsons_a100_h100.ipynb`

Orchestrates the four paper-headline R² fit scripts that generate the
five JSON artefacts cited in §8/§9. Runs trajectory extraction and
curve-fitting on up to three reference architectures (GPT-2 small,
MatchedGPT 8M, leak-free SPLM) and produces:

| Output JSON | Section | Description |
|-------------|---------|-------------|
| `R2_three_way_separator` | §8 | Separability of three architecture families |
| `R2_oracle` | §8 | Oracle attractor fit on SPLM trajectories |
| `R2_capacity_sweep` | §9 | Shared-V capacity vs R² on GPT-2 |
| `R2_token_direction` × 3 | §8 | Per-architecture token-direction alignment |

**Fit scripts invoked:**

| Stage | Script | Notes |
|-------|--------|-------|
| 1a–1c | `extract_gpt2_baseline.py` / `extract_matched_baseline.py` / `trajectory_extraction.py` | Gated by `RUN_GPT2`, `RUN_MATCHED`, `RUN_SPLM` toggles |
| 2 | `shared_potential_fit.py` | Per architecture |
| 3 | `splm_oracle_fit.py` | SPLM only |
| 4 | `sharedV_capacity_sweep.py` | GPT-2 only; 6 capacity configs: `h128_d2`, `h256_d2`, `h512_d2`, `h1024_d2`, `h512_d3`, `h512_d4`; 3000 steps each |
| 5 | `token_direction_fit.py` | Per architecture |
| 6 | `build_r2_three_way_separator.py` | Aggregation |

**Dataset:** `e_init_corpus` — 50 sentences across 5 domains (mathematics,
narrative, scientific, code_description, conversational); `max_len=64`,
`n_test_per_domain=2`, `seed=0`.

**GPU:** A100 (sm_80) / H100 (sm_90) — warns if compute capability < 8.0.
Budget: ≲1 h on H100 with all three architectures.

**GDrive output:** `paper_tmlr_1_r2_jsons/`

---

### pca_symmetry_sweep_a100_h100

**Path:** `notebooks/conservative_arch/scripts/pca_symmetry_sweep_a100_h100.ipynb`

Robustness sweep for the §7.2 local-conservativity (Jacobian-symmetry)
result under higher PCA-k values. Tests whether the symmetry gap holds
across `k ∈ {16, 32}` (optionally 64) on GPT-2 small, Pythia-160M, and
optionally MatchedGPT 8M and leak-free SPLM. Produces a verdict per
architecture: REJECTED (gap ≤ 0.10), PARTIAL (0.10–0.20), or
CONFIRMED (> 0.20).

**Stages:**

| Stage | What it does |
|-------|-------------|
| 0 | Config + Colab bootstrap |
| 1 | TF32 off, seeds, device |
| 2 | `stream_subprocess` helper |
| Extract | Hidden-state extraction per architecture (`RUN_GPT2`, `RUN_PYTHIA`, `RUN_MATCHED`, `RUN_SPLM` toggles) |
| 4 | `jacobian_symmetry.py` for each (architecture × PCA-k) |
| 5 | Copy artefacts to GDrive / results root |
| 6 | `aggregate_pca_sweep.py` |
| 9 | Display verdict table and figure |
| 10 | CLI equivalents for reference |

**Key config:** `SWEEP_PCA_KS = [16, 32]` (add `64` for extended sweep);
same corpus params as `r2_jsons`.

**Dataset:** Same `e_init_corpus` (50 sentences, 5 domains). Models:
GPT-2 small, EleutherAI/pythia-160m, optional MatchedGPT 8M and SPLM
from GDrive / local checkpoints.

**GPU:** A100 (sm_80) / H100 (sm_90). Budget: ≲30 min (GPT-2 + Pythia);
≲1 h with all four architectures on H100.

**GDrive output:** `paper_tmlr_1_pca_sweep/`

---

## Local-Only Notebooks

The following notebooks have no Colab setup (no `REPO_URL`, no
`google.colab` imports). They auto-detect CUDA / MPS / CPU and write
outputs to local `results/` directories.

---

### energy_landscape_validation

**Path:** `notebooks/stp_loss/energy_landscape_validation.ipynb`

Tests whether STP (sequence-to-prediction) loss emerges from a Lagrangian
with a Gaussian-well potential in GPT-2 hidden-state space. Full six-stage
protocol matching the companion document
`docs/STP_Loss_Is_An_Emergent_Property_Of_The_Energy_Landscape_Defined_By_Gaussian_Well_Potential.md`.

**Stages:**

| Stage | What it does |
|-------|-------------|
| 0 | Setup — optional pip installs, imports, device selection |
| 1 | Load model, build 50-sentence corpus, extract hidden states and per-token NTP loss |
| 2 | Energy-proxy plots (NTP loss distribution) |
| 3 | Cosine distance from trajectory centre (final hidden state) |
| 4 | Fit Gaussian-well / harmonic / linear potentials; global + per-domain + restoring-force analysis |
| 5 | Lagrangian reconstruction (angular velocity), total action, correlation with STP loss |
| 6 | Acceleration–STP equivalence on consecutive triplets; tangential acceleration stats; permutation null (100 permutations) |
| — | Save `experiment_results.json` |

**Model:** `gpt2` by default; `meta-llama/Llama-3.2-1B` commented as
alternative. `max_length=256`.

**Dataset:** Inline 50-sentence / 5-domain corpus.

**GPU:** Any (CUDA → MPS → CPU auto-detect; FP16 on CUDA, FP32 otherwise).

---

### pythia_tangential_acceleration

**Path:** `notebooks/cross_model/pythia_tangential_acceleration.ipynb`

Replicates the §13.2 tangential-acceleration statistics on Pythia-160M
alongside GPT-2 small for cross-architecture robustness. Both models must
show systematic deceleration (negative signed tangential acceleration) with
permutation-null z-scores > 3.

**Stages:**

| Stage | What it does |
|-------|-------------|
| 0 | Setup (`DEVICE`, `MODELS_TO_RUN`, `MAX_LENGTH=256`, `SEED=42`, `N_PERMUTATIONS=100`) |
| 1 | Build 50-sentence / 5-domain corpus |
| 2 | Define pipeline helpers (`extract_trajectory`, `decompose_acceleration`, `analyze_trajectories`, `permutation_null`) |
| 3 | Run both models; compute deceleration fractions and z-scores |
| 4 | Side-by-side summary table |
| 5 | Signed `a_parallel` distribution histograms per model |
| 6 | Observed vs null bar chart |
| 7 | Cache results as JSON/NPZ and print headline interpretation |

**Models:** `gpt2`, `EleutherAI/pythia-160m` (last layer only, `MAX_LENGTH=256`).

**Dataset:** Same 50-sentence / 5-domain corpus.

**GPU:** Not GPU-intensive — runs in 2–3 min per model on Apple Silicon
MPS; CUDA also supported.

---

### energy_landscape_validation_executed

**Path:** `notebooks/stp_loss/energy_landscape_validation_executed.ipynb`

Fully-executed snapshot of `energy_landscape_validation.ipynb` with all
cell outputs frozen. Stages 1–6 complete, including the Stage 6 acceleration–
STP equivalence and permutation-null tests. Executed on MPS
(`CUDA available: False`). Serves as a frozen reference for the paper.

---

### energy_landscape_validation_executed_backup

**Path:** `notebooks/stp_loss/energy_landscape_validation_executed_backup.ipynb`

Older executed snapshot (Stages 1–5 only; no Stage 6). Uses Euclidean
distance in Stage 3 instead of cosine. Retained as a backup from an earlier
protocol iteration (papermill run 2026-04-16). Not used as a primary
reference — prefer `energy_landscape_validation_executed.ipynb`.
