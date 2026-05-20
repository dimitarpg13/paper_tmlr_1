# Causal Leak in SPLM `integrate()` — Bug and Fix

**Status:** Resolved (v4 leak-free integrator, `cfg.causal_force = True`)  
**Scope:** Affects SPLM positive-control checkpoints trained before the fix. Does *not* affect the GPT-2 descriptive experiments (§3–§6) or the matched-attention 8M baseline (§8.4).  
**Full treatment:** Appendix A.3 of the paper.

---

## The bug

In every per-step `integrate()` site of the v2 SPLM family, the conservative
force $-\nabla_h V_\theta(\xi_t, h_t)$ on hidden state $h_t$ at generation
position $t$ was being computed against **future** positions. The causal
cumulative-mean context pool

$$
\xi_t = \frac{1}{t} \sum_{s \le t} h_s^{(\ell)}
$$

accumulated the hidden states $h_s^{(\ell)}$ **without detaching them from the
autograd graph**. As a result, the loss at position $t' < t$ propagated
gradients back through positions $s = t'+1, \ldots, t$ that appear inside
$\xi_t$. The trained $V_\theta$ silently learned to route prediction signal
through this anti-causal channel, inflating apparent validation perplexity
relative to a causally honest closed-loop integrator.

## The fix

One line in `integrate()`:

```python
# Before (buggy):
xi = h.mean(dim=0, keepdim=True)

# After (fixed):
xi = h.detach().mean(dim=0, keepdim=True)
```

The `.detach()` severs the autograd path from future positions into $\xi_t$,
enforcing strict causal honesty. All checkpoints produced after this fix are
tagged `cfg.causal_force = True` in their provenance JSON header.

## Verification

The regression test `notebooks/conservative_arch/causal_probe.py` re-runs at
every SPLM model registration. It verifies that the closed-loop Jacobian

$$
\frac{\partial \mathrm{loss}_{t'}}{\partial h_s} = 0 \quad \text{for all } s > t'
$$

using both a perturbation test ($\Delta = 0$ at strict `1e-6` tolerance) and
a gradient-Jacobian test (`∂logits_{t'}/∂embed_t ≡ 0` for `t' < t`).

```bash
cd notebooks/conservative_arch/
python causal_probe.py   # should report PASS on every position pair
```

## Why the shared-potential separator is leak-immune

The shared-potential regression of §7 is a **static off-line least-squares fit
on frozen hidden-state tensors** extracted by a single forward pass through the
architecture being measured. It does not back-propagate through the
architecture's integrator, does not re-run any training steps, and does not
depend on any property of the architecture's autograd graph.

Whatever the optimiser did with $V_\theta$ during training (including routing
signal through the leaky $\xi$ channel), the diagnostic asks only whether the
**post-training** $(V_\theta, h^{(\ell)})$ pair is internally consistent under
the prescribed force law. The answer is independent of any training-time
pathology.

## Confirmatory re-measurement

The SPLM positive control was re-trained under the leak-free integrator at
identical architecture and hyperparameters (`leakfree_3seed/gamma0p10/seed0`,
`L=8`, `d=128`, `d_V=512`, `γ* = 0.10`). The leak-corrected shared-potential
separator result:

| Checkpoint | Median per-layer test R² | Layer range |
|---|---|---|
| v2 (buggy) | 0.90 | — |
| **v4 leak-free (headline)** | **0.957** | [0.947, 0.969] |

The ~0.05 improvement is mechanistically consistent: the buggy integrator
routed some prediction signal through the anti-causal $\xi$ channel rather
than through the conservative $-\nabla_h V_\theta$ channel that the regression
measures; under causally honest training, trajectories sit closer to the
regression ansatz.

The GPT-2 and matched-baseline numbers (R² = 0.46 and 0.54) are **unchanged**;
those architectures have no SPLM lineage and are unaffected by this bug.

---

*Reference:* Paper Appendix A.3, `notebooks/conservative_arch/ln_damping_sweep/`
