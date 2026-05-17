# `checkpoints/` — locally trained model checkpoints

This directory holds the PyTorch checkpoints (`*.pt`) for the two
architectures trained from scratch in the paper:

| Checkpoint | Paper section | Training script |
|---|---|---|
| SPLM positive control (leak-free, $\gamma^{\ast} = 0.10$) | §7 / §8.2--§8.3 / §A.3 | [`../notebooks/conservative_arch/ln_damping_sweep/train_splm_em_ln.py`](../notebooks/conservative_arch/ln_damping_sweep/train_splm_em_ln.py) |
| Matched-attention 8M-parameter GPT-2-style decoder | §8.4 | [`../notebooks/conservative_arch/train_matched.py`](../notebooks/conservative_arch/train_matched.py) |

The checkpoints themselves are excluded from git (see `.gitignore`,
which excludes `*.pt`, `*.pth`, `*.bin`, `*.safetensors`) because they
are large and exactly reproducible from the training scripts.

## How to populate

```bash
cd ../notebooks/conservative_arch/ln_damping_sweep/
python train_splm_em_ln.py --gamma 0.10 --seed 0
# Output:
#   ../../checkpoints/splm_em_ln_leakfree_seed0.pt

cd ../
python train_matched.py --params 8M --seed 0
# Output:
#   ../checkpoints/matched_attention_8M_seed0.pt
```

Total training wall-clock on a single Apple MPS / consumer GPU:
≲4 h for the SPLM positive control, ≲2 h for the matched-attention
baseline.

## Provenance manifest

For long-term reproducibility, each checkpoint produced by a training
run is accompanied by a sibling `<checkpoint>.provenance.json` file
recording the SHA256 of the `.pt` file, the git commit at training
time, the random seed, and the full training config. The provenance
manifests *are* tracked in git; the `.pt` files are not.

Pretrained GPT-2 small (124 M parameters) and Pythia-160M are pulled
from the HuggingFace Hub on demand and cached under
`~/.cache/huggingface/`, not in this directory.
