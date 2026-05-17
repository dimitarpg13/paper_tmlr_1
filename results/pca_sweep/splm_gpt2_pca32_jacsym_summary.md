# Jacobian-symmetry diagnostic -- scalar-potential LM (gpt2_pca32)

Tests whether the per-step linear operator $M_\ell$ is **symmetric**.  A symmetric $M_\ell$ is what a conservative flow on a scalar potential must produce (Hessians of scalars are symmetric).  Skew-symmetric or symmetric-non-Hessian components indicate non-conservative dynamics.

We run TWO variants:

1. **Position-only (§1.5 analogue)**: $x_{\ell+1}-x_\ell \approx M_\ell x_\ell$.  For **damped second-order** dynamics (i.e. both this model and the Failure-doc §1 integrator), this fit is known to be confounded because the single-step transition mixes $x_\ell$ with the hidden velocity $v_\ell \approx x_\ell - x_{\ell-1}$.  The confound can manufacture apparent asymmetry even in genuinely conservative flows.
2. **Velocity-aware**: $x_{\ell+1}-x_\ell \approx A v_\ell + M_\ell x_\ell$ with $v_\ell = x_\ell - x_{\ell-1}$.  Here $M_\ell$ is the clean signal of the per-step spring matrix.  **This is the variant that tests conservativity.**

- Hidden dim `d = 768`, integration steps `L = 12`, PCA `k = 32`
- Train / test sentences: 40 / 10

## Per-layer fit quality

| layer | POS-only $R^{2}_\text{full}$ | POS-only $R^{2}_\text{sym}$ | VEL-aug $R^{2}_\text{full}$ | VEL-aug $R^{2}_\text{sym}$ | VEL-aug gap |
|--:|--:|--:|--:|--:|--:|
| 0 | +0.822 | +0.661 | +nan | +nan | +nan |
| 1 | +0.997 | +0.994 | +0.997 | +0.997 | +0.000 |
| 2 | +1.000 | +0.997 | +1.000 | +1.000 | +0.000 |
| 3 | +0.977 | +0.975 | +0.977 | +0.976 | +0.001 |
| 4 | +0.968 | +0.964 | +0.970 | +0.968 | +0.002 |
| 5 | +0.920 | +0.914 | +0.922 | +0.920 | +0.002 |
| 6 | +0.780 | +0.759 | +0.783 | +0.778 | +0.006 |
| 7 | +0.549 | +0.485 | +0.548 | +0.528 | +0.020 |
| 8 | +0.389 | +0.324 | +0.400 | +0.388 | +0.011 |
| 9 | +0.552 | +0.397 | +0.563 | +0.523 | +0.040 |
| 10 | +0.615 | +0.442 | +0.632 | +0.542 | +0.089 |
| 11 | +0.999 | +0.998 | +0.999 | +0.999 | +0.000 |

## Reference: GPT-2 small (§1.5 of Failure doc)

At matched PCA-$k$, POS-only $R^{2}_\text{full}\in[0.5,0.8]$ but POS-only $R^{2}_\text{sym}<0$ across every layer.  The gap in the **velocity-aware** variant has not yet been re-run for GPT-2 here; that is a required cross-check for v2 to make the comparison apples-to-apples.

## Verdict

- **Position-only (§1.5 analogue)**: max TEST gap = +0.172 (vs. GPT-2 where symmetric $R^2$ was *negative* everywhere).  Already a qualitative improvement.

- **Velocity-aware (the clean test)**: **Velocity-aware: symmetric-restricted fit tracks the unconstrained fit (max TEST gap = +0.089).  The per-step spring matrix is consistent with a symmetric Hessian, i.e. the dynamics is conservative on h.**

## Artefacts

- `splm_gpt2_pca32_jacsym_results.npz`
- `splm_gpt2_pca32_fig_jacsym.png`
