# Jacobian-symmetry diagnostic -- scalar-potential LM (pythia_pca16)

Tests whether the per-step linear operator $M_\ell$ is **symmetric**.  A symmetric $M_\ell$ is what a conservative flow on a scalar potential must produce (Hessians of scalars are symmetric).  Skew-symmetric or symmetric-non-Hessian components indicate non-conservative dynamics.

We run TWO variants:

1. **Position-only (§1.5 analogue)**: $x_{\ell+1}-x_\ell \approx M_\ell x_\ell$.  For **damped second-order** dynamics (i.e. both this model and the Failure-doc §1 integrator), this fit is known to be confounded because the single-step transition mixes $x_\ell$ with the hidden velocity $v_\ell \approx x_\ell - x_{\ell-1}$.  The confound can manufacture apparent asymmetry even in genuinely conservative flows.
2. **Velocity-aware**: $x_{\ell+1}-x_\ell \approx A v_\ell + M_\ell x_\ell$ with $v_\ell = x_\ell - x_{\ell-1}$.  Here $M_\ell$ is the clean signal of the per-step spring matrix.  **This is the variant that tests conservativity.**

- Hidden dim `d = 768`, integration steps `L = 12`, PCA `k = 16`
- Train / test sentences: 40 / 10

## Per-layer fit quality

| layer | POS-only $R^{2}_\text{full}$ | POS-only $R^{2}_\text{sym}$ | VEL-aug $R^{2}_\text{full}$ | VEL-aug $R^{2}_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.369 | +0.301 | +nan | +nan | +nan |
| 1 | +0.870 | +0.156 | +0.876 | +0.874 | +0.003 |
| 2 | +0.974 | +0.887 | +0.976 | +0.973 | +0.003 |
| 3 | +0.993 | +0.561 | +0.993 | +0.990 | +0.003 |
| 4 | +0.802 | +0.734 | +0.818 | +0.803 | +0.015 |
| 5 | +0.535 | +0.395 | +0.556 | +0.491 | +0.064 |
| 6 | +0.763 | +0.736 | +0.773 | +0.760 | +0.013 |
| 7 | +0.773 | +0.633 | +0.794 | +0.724 | +0.070 |
| 8 | +0.791 | +0.661 | +0.806 | +0.787 | +0.019 |
| 9 | +0.927 | +0.894 | +0.933 | +0.919 | +0.014 |
| 10 | +0.984 | +0.955 | +0.986 | +0.981 | +0.005 |
| 11 | +0.987 | +0.986 | +0.987 | +0.987 | +0.000 |

## Reference: GPT-2 small (§1.5 of Failure doc)

At matched PCA-$k$, POS-only $R^{2}_\text{full}\in[0.5,0.8]$ but POS-only $R^{2}_\text{sym}<0$ across every layer.  The gap in the **velocity-aware** variant has not yet been re-run for GPT-2 here; that is a required cross-check for v2 to make the comparison apples-to-apples.

## Verdict

- **Position-only (§1.5 analogue)**: max TEST gap = +0.714 (vs. GPT-2 where symmetric $R^2$ was *negative* everywhere).  Already a qualitative improvement.

- **Velocity-aware (the clean test)**: **Velocity-aware: symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.070).  The per-step spring matrix is consistent with a symmetric Hessian, i.e. the dynamics is conservative on h.**

## Artefacts

- `splm_pythia_pca16_jacsym_results.npz`
- `splm_pythia_pca16_fig_jacsym.png`
