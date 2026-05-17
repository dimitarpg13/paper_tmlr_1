# PCA-symmetry sweep — robustness extension to `paper_tmlr_1` §7.2

This directory contains a Colab-friendly experimental harness for the
**§5.1 PCA-symmetry sweep**: a robustness check of the
velocity-aware Jacobian-symmetry local-conservativity finding reported
in `paper_tmlr_1` §7.2, which is conducted at PCA-16 only.

The sweep is **not** part of the paper as written; it is a follow-up
robustness check whose outcome determines a small set of conditional
paper edits before TMLR submission. The decision rule, hypothesis
framework, and paper-update policy are all specified in
[`docs/SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md`](https://github.com/dimitarpg13/semsimula/blob/main/docs/SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md)
§5.1 in the main `semsimula` repository.

## Files

| File | Purpose |
|---|---|
| `pca_symmetry_sweep_a100_h100.ipynb` | Colab notebook (designed for H100, runs on any CUDA / MPS / CPU). Mounts GDrive, clones this repo, extracts hidden-state trajectories for GPT-2 small + Pythia-160M (+ optionally MatchedGPT 8M and SPLM if local checkpoints are available in GDrive), runs the velocity-aware Jacobian-symmetry test at each requested PCA dimension, copies all artefacts to GDrive, runs the aggregator, and displays the verdict inline. |
| `aggregate_pca_sweep.py` | CLI aggregator. Reads every `splm_*_pca*_jacsym_results.npz` under `--in-dir`, builds the per-architecture max-gap comparison table (markdown), the per-layer gap profile figure (PNG), and the REJECTED / PARTIAL / CONFIRMED verdict per architecture and overall (markdown + JSON). |

## Wall-clock budget on H100

| Stage | Architecture(s) | Wall-clock |
|---|---|---|
| Extract GPT-2 small | gpt2 | ~3 min |
| Extract Pythia-160M | EleutherAI/pythia-160m | ~5 min |
| Extract MatchedGPT (if ckpt) | local | ~2 min |
| Extract SPLM (if ckpt) | local | ~2 min |
| Jacsym at PCA-16 + PCA-32 (per arch) | — | ~3–6 min |
| Aggregate + verdict | — | < 30 sec |

**Total: ≲ 30 min for GPT-2 + Pythia only; ≲ 1 h with all four architectures.**

## Decision rule (verbatim from §5.1)

| Outcome | Max-layer TEST gap at largest tested PCA-k | Paper-edit implication |
|---|---|---|
| **H2 REJECTED** | ≤ 0.10 | `paper_tmlr_1` §7.2 stands; optional 1-paragraph robustness footnote before submission. |
| **H2 PARTIAL** | (0.10, 0.20] | One-paragraph scope-of-claim sharpening to §7.2. |
| **H2 CONFIRMED** | > 0.20 | Substantive rewrite of §7.2 framing and the introduction's contribution claims. The §8 three-way separator headline (*R²* = 0.949 / 0.56 / 0.45) is unaffected and in fact *strengthens* under this reading. |

The §5.1 verdict for the paper-edit cycle is determined by the **GPT-2 small** row alone. The remaining architectures, if present, sharpen the diagnosis but do not change the binary classification.

## Why two PCA dimensions?

With ~1,300 triplets per architecture, the symmetric-restricted regression fits a $d_{\mathrm{PCA}} \times d_{\mathrm{PCA}}$ symmetric matrix per layer with $\binom{d_{\mathrm{PCA}} + 1}{2}$ free parameters:

| $d_{\mathrm{PCA}}$ | Symmetric free params per layer | Triplets | Conditioning |
|---:|---:|---:|---|
| 16 | 136 | ~1,300 | Well-conditioned (paper baseline) |
| 32 | 528 | ~1,300 | Borderline; tractable |
| 64 | 2,080 | ~1,300 | Over-parameterised; needs ridge |
| 128 | 8,256 | ~1,300 | Severely over-parameterised |

The notebook defaults to a 2-point sweep (PCA-16 vs PCA-32). The optional PCA-64 third point is enabled by setting `SWEEP_PCA_KS = [16, 32, 64]` at the top of the notebook and relies on the ridge regulariser built into `jacobian_symmetry.py`.

## Running headless (CLI)

```bash
cd notebooks/conservative_arch/
mkdir -p results

# 1. Extract trajectories.
python extract_gpt2_baseline.py --model gpt2 \
  --out results/gpt2_baseline.trajectories.pkl \
  --n_test_per_domain 2 --max_len 64 --seed 0
python extract_gpt2_baseline.py --model EleutherAI/pythia-160m \
  --out results/pythia-160m_baseline.trajectories.pkl \
  --n_test_per_domain 2 --max_len 64 --seed 0

# 2. Run the PCA-symmetry sweep.
for ARCH in gpt2 pythia; do
  for K in 16 32; do
    python jacobian_symmetry.py \
      --traj  results/${ARCH}_baseline.trajectories.pkl \
      --pca_k $K --tag ${ARCH}_pca${K}
  done
done

# 3. Aggregate the verdict.
python scripts/aggregate_pca_sweep.py \
  --in-dir results --out-dir results --pca-ks 16 32
```

Output artefacts:

- `splm_<arch>_pca<k>_jacsym_results.npz` — raw per-layer R² values for full and symmetric fits.
- `splm_<arch>_pca<k>_fig_jacsym.png` — side-by-side position-only vs velocity-aware fit profile.
- `splm_<arch>_pca<k>_jacsym_summary.md` — per-architecture, per-PCA-k summary.
- `pca_sweep_table.md` — aggregate max-gap table across architectures and PCA-k values.
- `pca_sweep_gap_profile.png` — per-layer gap profile as a function of PCA dimension.
- `pca_sweep_verdict.md` — REJECTED / PARTIAL / CONFIRMED classification per architecture and overall.
- `pca_sweep_summary.json` — machine-readable summary of the above.

## On Colab

Open `pca_symmetry_sweep_a100_h100.ipynb` directly from this repository on GitHub in Colab. The first code cell handles the GDrive mount, repo clone, and dependency install; all outputs land under `/content/drive/MyDrive/paper_tmlr_1_pca_sweep/`.

## Executed-run artefacts (committed)

The first executed run of this harness (2026-05-17, H100 / Colab, GPT-2 small + Pythia-160M at PCA-16 and PCA-32) is committed in this repository under [`results/pca_sweep/`](../../../results/pca_sweep/). That directory contains the per-(architecture, PCA-dim) `.npz` / `.png` / `.md` artefacts, the aggregator outputs (`pca_sweep_table.md`, `pca_sweep_verdict.md`, `pca_sweep_summary.json`, `pca_sweep_gap_profile.png`), and a README that ties the headline numbers (GPT-2 0.079 → 0.089; Pythia 0.070 → 0.067; overall REJECTED) back to the §5.1 decision rule and the paper_tmlr_1 §7.2 robustness footnote they support.
