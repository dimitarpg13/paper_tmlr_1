# Jacobian-symmetry diagnostic -- scalar-potential LM (pythia_pca32)

Tests whether the per-step linear operator $M_\ell$ is **symmetric**.  A symmetric $M_\ell$ is what a conservative flow on a scalar potential must produce (Hessians of scalars are symmetric).  Skew-symmetric or symmetric-non-Hessian components indicate non-conservative dynamics.

We run TWO variants:

1. **Position-only (§1.5 analogue)**: $x_{\ell+1}-x_\ell \approx M_\ell x_\ell$.  For **damped second-order** dynamics (i.e. both this model and the Failure-doc §1 integrator), this fit is known to be confounded because the single-step transition mixes $x_\ell$ with the hidden velocity $v_\ell \approx x_\ell - x_{\ell-1}$.  The confound can manufacture apparent asymmetry even in genuinely conservative flows.
2. **Velocity-aware**: $x_{\ell+1}-x_\ell \approx A v_\ell + M_\ell x_\ell$ with $v_\ell = x_\ell - x_{\ell-1}$.  Here $M_\ell$ is the clean signal of the per-step spring matrix.  **This is the variant that tests conservativity.**

- Hidden dim `d = 768`, integration steps `L = 12`, PCA `k = 32`
- Train / test sentences: 40 / 10

## Per-layer fit quality

| layer | POS-only $R^{2}_\text{full}$ | POS-only $R^{2}_\text{sym}$ | VEL-aug $R^{2}_\text{full}$ | VEL-aug $R^{2}_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.374 | +0.286 | +nan | +nan | +nan |
| 1 | +0.895 | +0.175 | +0.900 | +0.899 | +0.001 |
| 2 | +0.971 | +0.884 | +0.974 | +0.970 | +0.004 |
| 3 | +0.993 | +0.563 | +0.994 | +0.990 | +0.004 |
| 4 | +0.797 | +0.737 | +0.815 | +0.800 | +0.015 |
| 5 | +0.576 | +0.429 | +0.605 | +0.538 | +0.067 |
| 6 | +0.787 | +0.726 | +0.803 | +0.780 | +0.023 |
| 7 | +0.797 | +0.643 | +0.805 | +0.750 | +0.055 |
| 8 | +0.785 | +0.648 | +0.801 | +0.781 | +0.020 |
| 9 | +0.925 | +0.884 | +0.927 | +0.914 | +0.013 |
| 10 | +0.985 | +0.952 | +0.985 | +0.981 | +0.005 |
| 11 | +0.980 | +0.977 | +0.979 | +0.979 | +0.001 |

## Reference: GPT-2 small (§1.5 of Failure doc)

At matched PCA-$k$, POS-only $R^{2}_\text{full}\in[0.5,0.8]$ but POS-only $R^{2}_\text{sym}<0$ across every layer.  The gap in the **velocity-aware** variant has not yet been re-run for GPT-2 here; that is a required cross-check for v2 to make the comparison apples-to-apples.

## Verdict

- **Position-only (§1.5 analogue)**: max TEST gap = +0.721 (vs. GPT-2 where symmetric $R^2$ was *negative* everywhere).  Already a qualitative improvement.

- **Velocity-aware (the clean test)**: **Velocity-aware: symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.067).  The per-step spring matrix is consistent with a symmetric Hessian, i.e. the dynamics is conservative on h.**

## Artefacts

- `splm_pythia_pca32_jacsym_results.npz`
- `splm_pythia_pca32_fig_jacsym.png`
