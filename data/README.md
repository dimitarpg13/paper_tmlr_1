# `data/` — corpora used by the experiments

This directory is the canonical location for raw and preprocessed
corpora consumed by the experiments. The contents are excluded from
git (see `.gitignore`) because they are large and regeneratable from
upstream sources.

## Expected layout after a clean run

```
data/
├── tiny_shakespeare/
│   └── input.txt              # ~1.1 MB, public domain (auto-downloaded)
├── logfreq_surprisal.npy      # 197 KB, committed (see below)
└── README.md                  # this file
```

## How to populate

### `tiny_shakespeare/input.txt`

On first run, [`../notebooks/conservative_arch/data_module.py`](../notebooks/conservative_arch/data_module.py)
downloads the Tiny Shakespeare corpus from
<https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt>
into `tiny_shakespeare/input.txt` and caches it for all subsequent
runs. No manual setup is required.

### `logfreq_surprisal.npy` (committed; 197 KB)

A frozen vector of per-token unigram surprisal values, one entry per
GPT-2 vocabulary id (50 257 floats, `dtype=float32`). Required by the
`mass_mode="logfreq"` SPLM variant
(`../notebooks/conservative_arch/model_sarf_mass.py`,
`model_ln.py`) used to instantiate the leak-corrected SPLM positive
control reported in paper_tmlr_1 §A.3.

Generated upstream by
`semsimula/notebooks/conservative_arch/sarf_mass_variant/compute_unigram_frequencies.py`
on Tiny Shakespeare. We commit it directly because (i) it is small,
(ii) the SPLM checkpoints we ship hard-code its values via the
embedded `logfreq_path`, and (iii) Colab clones do not have access to
the upstream `semsimula` paths the checkpoint was trained against.

The extractor / oracle loader (`trajectory_extraction.py`,
`splm_oracle_fit.py`) automatically overrides the absolute
`logfreq_path` saved inside the checkpoint with this bundled copy.

### Pretrained HuggingFace models

GPT-2 small (124 M parameters) and Pythia-160M are pulled from the
HuggingFace Hub at the first import in
[`../notebooks/stp_loss/energy_landscape_validation.ipynb`](../notebooks/stp_loss/energy_landscape_validation.ipynb) and
[`../notebooks/cross_model/pythia_tangential_acceleration.ipynb`](../notebooks/cross_model/pythia_tangential_acceleration.ipynb)
respectively, and are cached under the standard HuggingFace cache
directory (typically `~/.cache/huggingface/`), not in this repository.
