"""
SPLM variant: LayerNorm after each damped integration step.

This is the standalone-repo port of the upstream semsimula module
`notebooks/conservative_arch/energetic_minima/model_ln.py`. It is
required to load the leak-corrected SPLM positive-control checkpoint
cited in paper_tmlr_1 §A.3, whose saved `variant` tag is
"sarf_mass_ln".

Architectural idea
------------------
The only change from the SARF-faithful SPLM with per-token semantic
mass (the `logfreq` variant from `model_sarf_mass.py`) is that after
each semi-implicit damped integration step

    v_{l+1} = (v_l + dt * f_l / m) / (1 + dt * gamma)
    h_{l+1} = h_l + dt * v_{l+1}

we project h_{l+1} back to the unit-LayerNorm shell

    h_{l+1} <- (h_{l+1} - mu_{l+1}) / (sigma_{l+1} + eps)
               per-token mean/variance.

Rationale: compactness of S^{d-1} (up to the mean-shift) delivers a
finite minimum of any continuous V_theta on the shell by the
extreme-value theorem. This is the cheapest way to buy a finite minimum
of V without changing V's functional form or its loss-side gauge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_sarf_mass import (
    SPLMSARFMassConfig,
    ScalarPotentialLMSARFMass,
    causal_cumulative_mean,
)


@dataclass
class SPLMSARFMassLNConfig(SPLMSARFMassConfig):
    """Extends SPLMSARFMassConfig with LayerNorm-after-step switches."""
    ln_eps: float = 1e-5
    ln_after_step: bool = True
    ln_affine: bool = False


class ScalarPotentialLMSARFMassLN(ScalarPotentialLMSARFMass):
    """SARF-faithful SPLM with mandatory LayerNorm after every damped step."""

    def __init__(self, cfg: SPLMSARFMassLNConfig):
        super().__init__(cfg)
        self.cfg: SPLMSARFMassLNConfig = cfg
        if cfg.ln_affine:
            self.post_ln = nn.LayerNorm(cfg.d, eps=cfg.ln_eps)
        else:
            self.post_ln = None

    def _project(self, h: torch.Tensor) -> torch.Tensor:
        if self.post_ln is not None:
            return self.post_ln(h)
        return F.layer_norm(h, (self.cfg.d,), eps=self.cfg.ln_eps)

    def integrate(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        return_trajectory: bool = False,
        return_xi_trajectory: bool = False,
    ) -> Tuple[torch.Tensor,
               Optional[List[torch.Tensor]],
               Optional[List[torch.Tensor]]]:
        cfg = self.cfg
        h = self._project(emb) if cfg.ln_after_step else emb
        v = torch.zeros_like(h)
        gamma, dt = self.gamma, cfg.dt

        m = self.compute_mass(x, emb)
        m_b = m

        traj_h: Optional[List[torch.Tensor]] = None
        traj_xi: Optional[List[torch.Tensor]] = None
        if return_trajectory:
            traj_h = [h.detach().cpu()]
        if return_xi_trajectory:
            traj_xi = []

        for _ in range(cfg.L):
            xi_input = h.detach() if cfg.causal_force else h
            xi_now = causal_cumulative_mean(xi_input)
            if return_xi_trajectory:
                assert traj_xi is not None
                traj_xi.append(xi_now.detach().cpu())

            h_in = h
            if not h_in.requires_grad:
                h_in = h_in.requires_grad_(True)
            V = self.V_theta(xi_now, h_in).sum()
            grad_V, = torch.autograd.grad(
                V, h_in,
                create_graph=self.training,
                retain_graph=True,
            )
            f = -grad_V
            v = (v + dt * f / m_b) / (1.0 + dt * gamma)
            h_new = h_in + dt * v
            if cfg.ln_after_step:
                h_new = self._project(h_new)
            h = h_new
            if return_trajectory:
                assert traj_h is not None
                traj_h.append(h.detach().cpu())

        return h, traj_h, traj_xi
