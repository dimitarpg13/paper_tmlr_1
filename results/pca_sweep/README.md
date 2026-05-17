# `results/pca_sweep/` — executed §5.1 PCA-symmetry sweep artefacts

This directory contains the as-executed artefacts of the **§5.1
H2 PCA-symmetry sweep** — a robustness check on the velocity-aware
Jacobian-symmetry local-conservativity finding of `paper_tmlr_1`
§7.2, run at a second PCA dimension (PCA-32) on the same frozen
hidden-state trajectories that the paper §7.2 reports at PCA-16.

These artefacts back the one-paragraph **"Robustness under the
PCA-dimension choice"** footnote in `paper_tmlr_1` §7.2 and the
§4.2 falsification block of the `q9e_n` verdict note in the
[`semsimula` repo's `docs/`](https://github.com/dimitarpg13/semsimula/blob/main/docs/SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md)
(mirrored to [`semsimula-paper/companion_notes/`](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md)).

The harness that produced them lives in
[`../../notebooks/conservative_arch/scripts/`](../../notebooks/conservative_arch/scripts/);
the verdict note's §5.1 is the canonical decision rule for the
classification recorded below.

## Provenance

| Field | Value |
|---|---|
| Date executed | 2026-05-17 |
| Hardware | NVIDIA H100 (Google Colab) |
| Harness | [`notebooks/conservative_arch/scripts/pca_symmetry_sweep_a100_h100.ipynb`](../../notebooks/conservative_arch/scripts/pca_symmetry_sweep_a100_h100.ipynb) |
| Aggregator | [`notebooks/conservative_arch/scripts/aggregate_pca_sweep.py`](../../notebooks/conservative_arch/scripts/aggregate_pca_sweep.py) |
| Diagnostic | [`notebooks/conservative_arch/jacobian_symmetry.py`](../../notebooks/conservative_arch/jacobian_symmetry.py) (velocity-aware variant; `paper_tmlr_1` §7.2 protocol) |
| Architectures | `gpt2` (pretrained GPT-2 small, $d = 768$, $L = 12$); `EleutherAI/pythia-160m` (pretrained Pythia-160M, $d = 768$, $L = 12$) |
| PCA dimensions | $\{16, 32\}$ (the well-conditioned-regression regime; PCA-64+ requires ridge and was not pursued — see §5.1 of the verdict note) |
| Sentences | 40 train / 10 test per architecture |
| Random seed | 0 |

## Headline numbers — max-over-layers TEST gap

| Architecture | PCA-16 max gap | PCA-32 max gap | $\Delta$ (PCA-32 − PCA-16) | Verdict |
|---|---:|---:|---:|---|
| `gpt2` (pretrained) | $0.079$ (layer $10$) | $0.089$ (layer $10$) | $+0.010$ | **REJECTED** ($< 0.10$) |
| `pythia` (pretrained) | $0.070$ (layers $5$/$7$) | $0.067$ (layers $5$/$7$) | $-0.003$ | **REJECTED** ($< 0.10$) |

**Overall verdict: REJECTED** — the PCA-16 local-conservativity
finding is robust under the increase to PCA-32 on both tested
architectures; the H2 "PCA-16 artefact" hypothesis (verdict
note §4.2) is falsified.

## Files

### Aggregator outputs

| File | Purpose |
|---|---|
| `pca_sweep_verdict.md` | Per-architecture and overall REJECTED / PARTIAL / CONFIRMED classification against the §5.1 decision rule. |
| `pca_sweep_table.md` | Per-architecture max-over-layers gap (PCA-16 vs PCA-32) and full per-layer gap profile for both architectures. |
| `pca_sweep_summary.json` | Machine-readable summary of the per-architecture max-gap matrix and the verdict per architecture. |
| `pca_sweep_gap_profile.png` | Per-layer gap profile $\Delta_{\mathrm{sym}}(\ell, d_{\mathrm{PCA}})$ as a function of PCA dimension, side-by-side for both architectures. |

### Per-(architecture, PCA-dim) artefacts

Pattern: `splm_<arch>_pca<k>_jacsym_*` for `arch ∈ {gpt2, pythia}`, `k ∈ {16, 32}`.

| File pattern | Purpose |
|---|---|
| `*_jacsym_results.npz` | Raw per-layer position-only and velocity-aware $R^{2}_{\mathrm{full}}$ / $R^{2}_{\mathrm{sym}}$ values, train and test. ~3 KB each. |
| `*_jacsym_summary.md` | Per-architecture, per-PCA-k human-readable summary with the full per-layer fit-quality table and verdict. |
| `*_fig_jacsym.png` | Side-by-side position-only vs velocity-aware fit profile. |

## Decision rule (verbatim from §5.1 of the verdict note)

| Outcome | Max-layer TEST gap at largest tested PCA-k | Paper-edit implication |
|---|---|---|
| **H2 REJECTED** | $\le 0.10$ | `paper_tmlr_1` §7.2 stands; optional 1-paragraph robustness footnote before submission. |
| **H2 PARTIAL** | $(0.10, 0.20]$ | One-paragraph scope-of-claim sharpening to §7.2. |
| **H2 CONFIRMED** | $> 0.20$ | Substantive rewrite of §7.2 framing and the introduction's contribution claims. (The §8 three-way separator headline $R^2 = 0.949 / 0.56 / 0.45$ is unaffected and in fact *strengthens* under this reading.) |

Both `gpt2` and `pythia` land in the REJECTED column, so the
licensed paper edit is the one-paragraph robustness footnote
(landed in `paper_tmlr_1` §7.2 on 2026-05-17, commit
[`7242b0e`](https://github.com/dimitarpg13/semsimula/commit/7242b0e)).

## Why no PCA-64+ in this sweep?

The symmetric-restricted regression fits a
$d_{\mathrm{PCA}} \times d_{\mathrm{PCA}}$ symmetric matrix per
layer with $\binom{d_{\mathrm{PCA}} + 1}{2}$ free parameters
against $\sim 1{,}300$ triplets per architecture:

| $d_{\mathrm{PCA}}$ | Sym. free params per layer | Triplets | Conditioning |
|---:|---:|---:|---|
| 16 | 136 | $\sim 1{,}300$ | Well-conditioned (paper §7.2 baseline) |
| 32 | 528 | $\sim 1{,}300$ | Borderline; tractable (this sweep) |
| 64 | 2,080 | $\sim 1{,}300$ | Over-parameterised; requires ridge |
| 128 | 8,256 | $\sim 1{,}300$ | Severely over-parameterised; requires ridge |

PCA-32 is the largest PCA dimension at which the
symmetric-restricted regression is well-conditioned without
relying on a regulariser whose strength would itself need to be
defended in a reviewer-readable note. The 2-point sweep was
sufficient for an unambiguous REJECTED on both architectures;
the higher arm of the sweep ($d_{\mathrm{PCA}} \ge 64$) is
available as a follow-up via the `SWEEP_PCA_KS` option in the
harness notebook if a reviewer requests it.

## Reproducing this run

See [`../../notebooks/conservative_arch/scripts/README.md`](../../notebooks/conservative_arch/scripts/README.md)
for the full Colab and CLI reproduction recipes. Output filenames
match the schema documented above; the aggregator step emits the
`pca_sweep_*` summary files into the same directory it is pointed
at via `--out-dir`.

## Cross-references

- `paper_tmlr_1` §7.2 — the paragraph "Robustness under the PCA-dimension choice" cites the headline numbers in this directory and the verdict.
- [`SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md`](https://github.com/dimitarpg13/semsimula/blob/main/docs/SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md) §4.2 — full falsification block for the H2 hypothesis; §5.1 — the executed-protocol record and decision-rule application.
- [`semsimula-paper/companion_notes/...`](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md) — mirror of the verdict note.
