# PCA-symmetry sweep — verdict

Implements §5.1 of `docs/SP_HSPLM_Stage_2_q9e_n_verdict_and_structural_reexamination.md` applied to the PCA-dimension sweep on the per-token-internal velocity-aware Jacobian-symmetry test of `paper_tmlr_1` §7.2.

## Per-architecture verdict

- **gpt2**: max gap at PCA-32 = +0.089 → **REJECTED**; PCA-16 reference from paper §7.2 = +0.079.
- **pythia**: max gap at PCA-32 = +0.067 → **REJECTED**.

## Overall verdict: **REJECTED**

The `paper_tmlr_1` §7.2 PCA-16 local-conservativity finding is robust at the largest tested PCA dimension. Paper-edit implication: §7.2 stands as written; an optional one-paragraph robustness footnote can be added before submission noting that we verified the finding at PCA-32 (and PCA-64 if run) with the headline gaps stable.

## Reference: §5.1 decision rule

- **H2 REJECTED** if max-layer gap stays <= 0.10 at every tested PCA-k.
- **H2 PARTIAL** if max-layer gap at largest PCA-k is in (0.10, 0.20].
- **H2 CONFIRMED** if max-layer gap at largest PCA-k is > 0.20.
