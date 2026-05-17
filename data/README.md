# `data/` — corpora used by the experiments

This directory is the canonical location for raw and preprocessed
corpora consumed by the experiments. The contents are excluded from
git (see `.gitignore`) because they are large and regeneratable from
upstream sources.

## Expected layout after a clean run

```
data/
├── tiny_shakespeare/
│   └── input.txt              # ~1.1 MB, public domain
└── README.md                  # this file
```

## How to populate

On first run, [`../notebooks/conservative_arch/data_module.py`](../notebooks/conservative_arch/data_module.py)
downloads the Tiny Shakespeare corpus from
<https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt>
into `tiny_shakespeare/input.txt` and caches it for all subsequent
runs. No manual setup is required.

GPT-2 small (124 M parameters) and Pythia-160M are pulled from the
HuggingFace Hub at the first import in
[`../notebooks/stp_loss/energy_landscape_validation.ipynb`](../notebooks/stp_loss/energy_landscape_validation.ipynb) and
[`../notebooks/cross_model/pythia_tangential_acceleration.ipynb`](../notebooks/cross_model/pythia_tangential_acceleration.ipynb)
respectively, and are cached under the standard HuggingFace cache
directory (typically `~/.cache/huggingface/`), not in this repository.
