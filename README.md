# Code repository for *Locally Conservative, Globally Not: Diagnosing the Lagrangian Structure of Pretrained Transformers*

This is the public code repository accompanying the TMLR paper
**Locally Conservative, Globally Not: Diagnosing the Lagrangian Structure
of Pretrained Transformers** (D. P. Gueorguiev, 2026).

The repository reproduces every numerical claim and figure in the paper
end-to-end from frozen public checkpoints (GPT-2 small,
Pythia-160M) plus a small set of locally-trained checkpoints (the SPLM
positive control of §7--§8 and the matched-attention 8M-parameter
baseline of §8.4). No retraining of the public models is required.

> **Note.** The paper itself lives in a separate repository. This
> repository is code-only: it ships the experimental and diagnostic
> source needed to reproduce the paper's numerical content. The paper's
> theorems and proofs are self-contained and require no code support.

---

## Scope and contents

The repository carries exactly the experiments described in the paper:

| Paper section | Experiment | Code location |
|---|---|---|
| §3 / §4 | STP--acceleration identity, machine-precision verification on GPT-2 small (1,314 triplets) | [`notebooks/stp_loss/`](notebooks/stp_loss/) |
| §5 | Mechanical content: tangential-vs-normal acceleration, deceleration sign, permutation null | [`notebooks/stp_loss/`](notebooks/stp_loss/) |
| §6 | Cross-architecture replication on Pythia-160M | [`notebooks/cross_model/`](notebooks/cross_model/) |
| §7 | Velocity-aware Jacobian-symmetry test at PCA-16; shared-potential regression methodology | [`notebooks/conservative_arch/jacobian_symmetry.py`](notebooks/conservative_arch/jacobian_symmetry.py), [`notebooks/conservative_arch/shared_potential_fit.py`](notebooks/conservative_arch/shared_potential_fit.py) |
| §8 | Three-way architectural separator (SPLM positive control / matched-attention 8M / pretrained GPT-2 small) at *R²* = 0.949 / 0.56 / 0.45 | [`notebooks/conservative_arch/run_full_pipeline.py`](notebooks/conservative_arch/run_full_pipeline.py), [`notebooks/conservative_arch/plot_three_way_comparison.py`](notebooks/conservative_arch/plot_three_way_comparison.py) |
| §8.2--§8.3 | SPLM positive control: model definition and training | [`notebooks/conservative_arch/model.py`](notebooks/conservative_arch/model.py), [`notebooks/conservative_arch/train_splm.py`](notebooks/conservative_arch/train_splm.py) |
| §8.4 | Matched-attention 8M-parameter GPT-2-style decoder baseline | [`notebooks/conservative_arch/matched_baseline_model.py`](notebooks/conservative_arch/matched_baseline_model.py), [`notebooks/conservative_arch/train_matched.py`](notebooks/conservative_arch/train_matched.py) |
| §9.1 | Oracle fit *R²* = 1.0000 (uses positive control's own potential as *V_ψ*) | [`notebooks/conservative_arch/splm_oracle_fit.py`](notebooks/conservative_arch/splm_oracle_fit.py) |
| §9.2 | *V_ψ* capacity sweep over a 16× parameter band | [`notebooks/conservative_arch/sharedV_capacity_sweep.py`](notebooks/conservative_arch/sharedV_capacity_sweep.py) |
| §9.3 | Coordinate-system robustness under token-direction coordinates | [`notebooks/conservative_arch/token_direction_fit.py`](notebooks/conservative_arch/token_direction_fit.py) |
| §A.3 | Leak-corrected re-measurement of the SPLM positive control's *R²* | [`notebooks/conservative_arch/ln_damping_sweep/`](notebooks/conservative_arch/ln_damping_sweep/) + [`notebooks/conservative_arch/causal_probe.py`](notebooks/conservative_arch/causal_probe.py) + [`notebooks/conservative_arch/eval_ppl_under_fix.py`](notebooks/conservative_arch/eval_ppl_under_fix.py) |

The folder layout mirrors the original development repository's
`notebooks/` structure so that the code paths and module imports match
the paper's `\path{...}` references verbatim.

---

## Repository structure

```
paper_tmlr_1/
├── README.md                      # this file
├── LICENSE                        # MIT
├── .gitignore                     # excludes checkpoints, parquets, caches
├── requirements.txt               # pinned dependencies
│
├── notebooks/                     # source code for every experiment
│   ├── stp_loss/                  # §3 / §4 / §5 STP–acceleration identity
│   ├── cross_model/               # §6 Pythia-160M replication
│   └── conservative_arch/         # §7 / §8 / §9 / §A.3
│       ├── model.py                 # SPLM model class
│       ├── matched_baseline_model.py
│       ├── train_splm.py
│       ├── train_matched.py
│       ├── shared_potential_fit.py  # §7 separator regression
│       ├── splm_oracle_fit.py       # §9.1 oracle
│       ├── sharedV_capacity_sweep.py# §9.2 capacity sweep
│       ├── token_direction_fit.py   # §9.3
│       ├── jacobian_symmetry.py     # §7.2 / §9 control
│       ├── causal_probe.py          # §A.3 leak probe
│       ├── eval_ppl_under_fix.py    # §A.3 leak-free PPL
│       ├── trajectory_extraction.py # frozen-checkpoint hidden-state extraction
│       ├── data_module.py           # Tiny Shakespeare loader
│       ├── extract_*.py             # GPT-2 / matched-baseline extractors
│       ├── plot_*.py                # §8.5 / §9 plotting
│       ├── run_full_pipeline.py     # §8 end-to-end runner
│       └── ln_damping_sweep/        # §A.3 leak-free retrain
│
├── data/                          # placeholder; see data/README.md
├── checkpoints/                   # placeholder; see checkpoints/README.md
├── figures/                       # placeholder; see figures/README.md
└── results/                       # placeholder; see results/README.md
```

`data/`, `checkpoints/`, `figures/`, and `results/` are kept as empty
directories with a `README.md` documenting how to regenerate or fetch
their contents. The repository itself ships only source code; large
binary artifacts (PyTorch checkpoints, Parquet datasets, trajectory
pickles) are excluded by `.gitignore` and described in each
sub-directory's README.

---

## Setup

The code targets Python 3.10 or later.

```bash
git clone <repo-url> paper_tmlr_1
cd paper_tmlr_1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hardware: the descriptive experiments of §3--§6 run on CPU in minutes;
the §7--§9 frozen-checkpoint separator regression runs in
≲1 hour on a single Apple MPS / consumer GPU; retraining the SPLM
positive control of §A.3 leak-free retrain takes ≲4 hours on a
single MPS device. No multi-GPU setup is required for any experiment
in this repository.

---

## Data and checkpoints

| Asset | Source | How to obtain |
|---|---|---|
| Tiny Shakespeare corpus | [karpathy/char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) | Auto-downloaded into `data/tiny_shakespeare/` by [`notebooks/conservative_arch/data_module.py`](notebooks/conservative_arch/data_module.py) on first run. |
| GPT-2 small checkpoint | HuggingFace `gpt2` | Auto-downloaded via `transformers.GPT2Model.from_pretrained('gpt2')` on first run of the §4--§5 scripts. |
| Pythia-160M checkpoint | HuggingFace `EleutherAI/pythia-160m` | Auto-downloaded via `transformers.GPTNeoXModel.from_pretrained('EleutherAI/pythia-160m')` on first run of the §6 notebook. |
| SPLM positive-control checkpoint | This repository | Re-train with [`notebooks/conservative_arch/train_splm.py`](notebooks/conservative_arch/train_splm.py); the §A.3 leak-free re-train is in [`notebooks/conservative_arch/ln_damping_sweep/train_splm_em_ln.py`](notebooks/conservative_arch/ln_damping_sweep/train_splm_em_ln.py). Output checkpoints live under `checkpoints/`. |
| Matched-attention 8M baseline | This repository | Re-train with [`notebooks/conservative_arch/train_matched.py`](notebooks/conservative_arch/train_matched.py). |

---

## Reproducing each paper section

### §3 / §4 STP--acceleration identity (≈5 min on CPU)

```bash
cd notebooks/stp_loss/
jupyter nbconvert --to notebook --execute energy_landscape_validation.ipynb
```

The notebook performs the closed-form numerical verification of
Theorem 1: it loads pretrained GPT-2 small, extracts last-layer hidden
states on a 50-sentence corpus, computes the cosine $\mathcal L_{\mathrm{STP}}$ and the
right-hand side of $\mathcal L_{\mathrm{STP}} = 1 - \sqrt{1 - |\vec a_\perp|^2 / \|\vec d_2\|^2}$
for each of 1,314 consecutive triplets, and reports the Pearson
correlation, the max absolute residual, and the mean absolute residual
between the two computations. The reference numerical values in the
paper are $r = 1.000000000$, max residual $1.7 \times 10^{-13}$, mean
residual $4.7 \times 10^{-16}$.

### §5 Mechanical content (≈3 min on CPU)

The same notebook also produces the descriptive measurements of §5:
the tangential and normal acceleration magnitudes per triplet, the
$|a_\parallel|/|\vec a_\perp|$ ratio histogram by domain, the sign of
$a_\parallel$, the permutation-null test with 100 random permutations
per sentence, and the per-sentence trajectory profiles.

### §6 Cross-architecture replication on Pythia-160M (≈10 min on CPU)

```bash
cd notebooks/cross_model/
jupyter nbconvert --to notebook --execute pythia_tangential_acceleration.ipynb
```

The notebook is a near-clone of the §4--§5 GPT-2 protocol, applied
verbatim to Pythia-160M. The reference numerical values in the paper
are 100% deceleration ($a_\parallel < 0$ on every triplet), permutation
*z*-scores $z = -33.2$ on $|a_\parallel|$ and $z = -23.3$ on
$|\vec a_\perp|$, and a $|a_\parallel|/|\vec a_\perp|$ ratio of 1.61.

### §7 Velocity-aware Jacobian-symmetry test at PCA-16

```bash
cd notebooks/conservative_arch/
python jacobian_symmetry.py --pca 16 --architectures splm,matched,gpt2
```

### §8 Three-way shared-potential separator

```bash
cd notebooks/conservative_arch/
python run_full_pipeline.py
python plot_three_way_comparison.py
```

This trains the SPLM positive control and the matched-attention 8M
baseline if their checkpoints are not present in `checkpoints/`, then
runs the off-line OLS-on-frozen-tensors shared-potential regression of
§7 against each of the three architectures, and reports the
*R²* = 0.949 / 0.56 / 0.45 three-way separator.

### §9 Internal-validity controls

```bash
cd notebooks/conservative_arch/
python splm_oracle_fit.py            # §9.1 oracle R² = 1.0000
python sharedV_capacity_sweep.py     # §9.2 16x parameter sweep
python token_direction_fit.py        # §9.3 token-direction coords
```

### §A.3 Leak-corrected re-measurement of the SPLM positive control

```bash
cd notebooks/conservative_arch/
python causal_probe.py               # verifies the closed-loop Jacobian
                                     #   ∂loss_t / ∂h_s vanishes for s > t
cd ln_damping_sweep/
python train_splm_em_ln.py --gamma 0.10 --seed 0   # leak-free retrain
python analyse_sweep.py              # confirms R² = 0.949 on the
                                     #   leak-corrected checkpoint
```

The forensic detail of the leak bug and its fix is documented at
[`docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md`](docs/Causal_Leak_in_SPLM_Integrate_Bug_and_Fix.md) (to be added).

---

## Robustness extensions (not part of the paper)

The following extensions live in
[`notebooks/conservative_arch/scripts/`](notebooks/conservative_arch/scripts/)
and are **not** part of `paper_tmlr_1` as submitted. They are
robustness checks whose outcome determines a small set of conditional
paper edits before TMLR submission. The decision rules and
paper-update policies are specified in companion notes in the main
[`semsimula`](https://github.com/dimitarpg13/semsimula) repository
under `docs/`.

### §5.1 PCA-symmetry sweep (≲30 min on H100; ≲1 h with all four architectures)

Re-runs the §7.2 velocity-aware Jacobian-symmetry test at multiple
PCA dimensions (default: PCA-16 vs PCA-32; optional PCA-64 with
ridge regularisation) on the same frozen-checkpoint hidden-state
trajectories, to test whether the *R²* gap of ≤ 0.079 reported at
PCA-16 is robust under an increase in the projection dimension or is
an artifact of aggressive dimensionality reduction.

```bash
# Colab (preferred for H100 access):
#   open notebooks/conservative_arch/scripts/pca_symmetry_sweep_a100_h100.ipynb
#   in Google Colab; the first cell handles GDrive mount + repo clone.

# Headless / SLURM:
cd notebooks/conservative_arch/
mkdir -p results
python extract_gpt2_baseline.py --model gpt2 \
  --out results/gpt2_baseline.trajectories.pkl
python extract_gpt2_baseline.py --model EleutherAI/pythia-160m \
  --out results/pythia-160m_baseline.trajectories.pkl
for ARCH in gpt2 pythia; do
  for K in 16 32; do
    python jacobian_symmetry.py \
      --traj  results/${ARCH}_baseline.trajectories.pkl \
      --pca_k $K --tag ${ARCH}_pca${K}
  done
done
python scripts/aggregate_pca_sweep.py \
  --in-dir results --out-dir results --pca-ks 16 32
```

The aggregator classifies the outcome as **REJECTED** (max gap ≤ 0.10
at every PCA-k; paper §7.2 stands), **PARTIAL** (gap in (0.10, 0.20]
at the largest PCA-k; scope-of-claim sharpening needed), or
**CONFIRMED** (gap > 0.20; §7.2 framing rewrite needed, §8 three-way
separator headline strengthens). The verdict for the paper-edit cycle
is determined by the GPT-2 row alone; the remaining architectures
sharpen but do not change the classification.

See [`notebooks/conservative_arch/scripts/README.md`](notebooks/conservative_arch/scripts/README.md)
for full documentation.

---

## Reproduction contract

Every numerical claim in the paper has corresponding source code in
this repository. Each script is self-contained, deterministic for a
fixed random seed, and emits its results as a small JSON file under
`results/` (with the config hash, checkpoint SHA256 if applicable, code
commit, and random seed recorded in the file header).

The total wall-clock budget to reproduce every headline number from the
papers's frozen-checkpoint experiments is ≲4 hours on a single Apple
MPS or consumer GPU. Retraining the SPLM positive control and the
matched-attention 8M baseline from scratch adds another ≲6 hours.

---

## Citation

If you use this code, please cite the paper as:

```bibtex
@article{Gueorguiev2026LocallyConservative,
  title  = {Locally Conservative, Globally Not: Diagnosing the
            Lagrangian Structure of Pretrained Transformers},
  author = {Gueorguiev, Dimitar P.},
  year   = {2026},
  journal= {Transactions on Machine Learning Research},
  note   = {Under review.}
}
```

---

## License

This code is released under the MIT License; see [LICENSE](LICENSE).

The Tiny Shakespeare corpus is in the public domain. Pretrained
GPT-2 small is released by OpenAI under the modified MIT license
(see HuggingFace `gpt2` model card). Pretrained Pythia-160M is
released by EleutherAI under the Apache 2.0 license (see HuggingFace
`EleutherAI/pythia-160m` model card).
