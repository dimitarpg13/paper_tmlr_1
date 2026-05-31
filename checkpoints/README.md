# `checkpoints/` — locally trained model checkpoints

This directory holds the PyTorch checkpoints (`*.pt`) for the two
architectures trained from scratch in the paper:

| Checkpoint | Canonical filename | Paper section | Training script |
|---|---|---|---|
| SPLM positive control (leak-free, $\gamma^{\ast} = 0.10$) | `splm_em_ln_leakfree_gamma0p10_seed0_ckpt_latest.pt` | §7 / §8.2--§8.3 / §A.3 | [`../notebooks/conservative_arch/ln_damping_sweep/train_splm_em_ln.py`](../notebooks/conservative_arch/ln_damping_sweep/train_splm_em_ln.py) |
| Matched-attention 8M-parameter GPT-2-style decoder | `matched_baseline_shakespeare_seed0_ckpt_latest.pt` | §8.4 | [`../notebooks/conservative_arch/train_matched.py`](../notebooks/conservative_arch/train_matched.py) |

The checkpoints themselves are excluded from git (see `.gitignore`,
which excludes `*.pt`, `*.pth`, `*.bin`, `*.safetensors`) because they
are large and exactly reproducible from the training scripts. The two
canonical filenames above are the exact names that
`r2_jsons_a100_h100.ipynb` and `pca_symmetry_sweep_a100_h100.ipynb`
look for (both locally under this `checkpoints/` directory and, on
Colab, under `/MyDrive/paper_tmlr_1_checkpoints/`).

## How to populate

The one-shot orchestrator
[`../notebooks/conservative_arch/make_checkpoints.py`](../notebooks/conservative_arch/make_checkpoints.py)
trains both architectures and writes them here under the canonical
names. It is idempotent (existing checkpoints are skipped unless
`--force`):

```bash
cd ../notebooks/conservative_arch/
python make_checkpoints.py                 # full paper-grade, both, seed 0
# Outputs:
#   ../../checkpoints/splm_em_ln_leakfree_gamma0p10_seed0_ckpt_latest.pt
#   ../../checkpoints/matched_baseline_shakespeare_seed0_ckpt_latest.pt

# Fast, NON-paper-grade end-to-end pipeline check (300 steps each):
python make_checkpoints.py --mode smoke
# Just one architecture / force a retrain:
python make_checkpoints.py --which splm --force
```

Equivalently, you can flip `TRAIN_CHECKPOINTS = True` in the
configuration cell of either headline notebook to (re)create the
checkpoints in-session before extraction.

Under the hood `make_checkpoints.py` calls the two training scripts
directly:

```bash
# SPLM positive control (leak-free, gamma* = 0.10):
cd ln_damping_sweep/
python train_splm_em_ln.py --mode shakespeare --seed 0 \
    --fixed-gamma 0.10 --tag-suffix gamma0p10_seed0 \
    --logfreq-path ../../../data/logfreq_surprisal.npy

# Matched-attention 8M baseline:
cd ..
python train_matched.py --mode shakespeare --seed 0
```

The SPLM control consumes the unigram-surprisal table
`data/logfreq_surprisal.npy` (shipped with the repo); regenerate it
from scratch with
[`../notebooks/conservative_arch/compute_unigram_frequencies.py`](../notebooks/conservative_arch/compute_unigram_frequencies.py)
if needed.

Total training wall-clock on a single Apple MPS / consumer GPU
(`--mode shakespeare`): ≲4 h for the SPLM positive control, ≲2 h for
the matched-attention baseline.

## Provenance manifest

For long-term reproducibility, each checkpoint produced by a training
run is accompanied by a sibling `<checkpoint>.provenance.json` file
recording the SHA256 of the `.pt` file, the git commit at training
time, the random seed, and the full training config. The provenance
manifests *are* tracked in git; the `.pt` files are not.

Pretrained GPT-2 small (124 M parameters) and Pythia-160M are pulled
from the HuggingFace Hub on demand and cached under
`~/.cache/huggingface/`, not in this directory.
