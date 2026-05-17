# PCA-symmetry sweep: PCA-16 vs PCA-32 (vs PCA-64 if run)

Velocity-aware Jacobian-symmetry test (`paper_tmlr_1` §7.2 protocol) re-run at multiple PCA dimensions on the same frozen-checkpoint hidden-state trajectories. Headline statistic: per-architecture **max-over-layers TEST gap** $\max_\ell |R^2_{\mathrm{full}} - R^2_{\mathrm{sym}}|$. Paper §7.2 reports max gap of 0.079 (GPT-2), 0.070 (matched-attention 8M), 0.040 (SPLM) at PCA-16. This sweep tests whether those values are robust under an increase in the diagnostic's projection dimension.

## Max-over-layers gap by architecture and PCA-k

| architecture | PCA-16 max gap | PCA-32 max gap | verdict |
|---|---:|---:|---|
| gpt2 | +0.079 | +0.089 | REJECTED |
| pythia | +0.070 | +0.067 | REJECTED |

## Per-layer gap, architecture = gpt2

| layer | PCA-16 | PCA-32 |
|---:|---:|---:|
| 0 | n/a | n/a |
| 1 | +0.000 | +0.000 |
| 2 | +0.000 | +0.000 |
| 3 | +0.000 | +0.001 |
| 4 | +0.003 | +0.002 |
| 5 | +0.001 | +0.002 |
| 6 | +0.008 | +0.006 |
| 7 | +0.022 | +0.020 |
| 8 | +0.019 | +0.011 |
| 9 | +0.046 | +0.040 |
| 10 | +0.079 | +0.089 |
| 11 | +0.000 | +0.000 |

## Per-layer gap, architecture = pythia

| layer | PCA-16 | PCA-32 |
|---:|---:|---:|
| 0 | n/a | n/a |
| 1 | +0.003 | +0.001 |
| 2 | +0.003 | +0.004 |
| 3 | +0.003 | +0.004 |
| 4 | +0.015 | +0.015 |
| 5 | +0.064 | +0.067 |
| 6 | +0.013 | +0.023 |
| 7 | +0.070 | +0.055 |
| 8 | +0.019 | +0.020 |
| 9 | +0.014 | +0.013 |
| 10 | +0.005 | +0.005 |
| 11 | +0.000 | +0.001 |
