"""
SARF-faithful scalar-potential LM with per-token semantic mass.

This is the standalone-repo port of the upstream semsimula module
`notebooks/conservative_arch/sarf_mass_variant/model_sarf_mass.py`.
It is required to load the leak-corrected SPLM positive-control
checkpoint cited in paper_tmlr_1 §A.3, which was trained with a
per-token logfreq-surprisal mass and a fixed damping coefficient.

Three mass parameterisations are supported, selected by `cfg.mass_mode`:

  - "global"     : single learnable scalar (baseline-equivalent null
                   control, kept so this module can reproduce the
                   pre-mass baseline in a single codebase).
  - "embed_head" : m_t = softplus(<w_m, E[x_t]> + b_m) + eps, i.e. a
                   cheap linear head on the token embedding. Learned,
                   position-invariant, content-dependent.
  - "logfreq"    : m_t = 1 + alpha * (-log p_hat(x_t)), a frozen
                   unigram-surprisal prior with a single learnable
                   scale alpha >= 0.

Everything else (SARF-faithful xi re-pooling, shared V_theta, damped
Euler-Lagrange integrator with learnable or fixed gamma, tied-embedding
readout) is identical to the SARF baseline. Mass is computed once per
forward pass from the first-layer input and held fixed across the L
integration steps, matching the framework's per-particle-scalar
prescription (no state-dependence; no layer drift).

Causal-leak fix (`causal_force=True`, default) detaches the autograd
path from xi back to h inside the integration loop; this is the
physics-correct Euler-Lagrange equation
    m * h_tt = -d V(xi_t, h_t) / d h_t
as the per-token dynamics. See `docs/Causal_Leak_in_SPLM_*` in the
upstream repo for the bug-and-fix writeup. The leak-free positive
control reported in paper_tmlr_1 §A.3 was trained with
`causal_force=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SPLMSARFMassConfig:
    vocab_size: int = 50257
    d: int = 128
    max_len: int = 512
    v_hidden: int = 512
    v_depth: int = 3
    L: int = 8
    dt: float = 1.0
    init_m: float = 1.0
    init_gamma: float = 1.0
    learn_mgamma: bool = True

    mass_mode: str = "global"
    logfreq_init_alpha: float = 0.0
    logfreq_path: Optional[str] = None

    fixed_gamma: Optional[float] = None

    causal_force: bool = True


class ScalarPotential(nn.Module):
    """MLP (xi, h) -> scalar energy. Identical to the SARF baseline."""

    def __init__(self, d: int, hidden: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(2 * d, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=0.002)
        nn.init.zeros_(last.bias)

    def forward(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([xi, h], dim=-1))


def causal_cumulative_mean(h: torch.Tensor) -> torch.Tensor:
    T = h.shape[1]
    cumsum = h.cumsum(dim=1)
    denom = torch.arange(1, T + 1, device=h.device, dtype=h.dtype).view(1, T, 1)
    return cumsum / denom


def _raw_from_positive(y: float) -> float:
    import math
    return math.log(math.expm1(max(y, 1e-3)))


class ScalarPotentialLMSARFMass(nn.Module):
    """SARF-faithful SPLM with pluggable per-token mass."""

    def __init__(self, cfg: SPLMSARFMassConfig):
        super().__init__()
        self.cfg = cfg
        self.E = nn.Embedding(cfg.vocab_size, cfg.d)
        self.P = nn.Parameter(torch.zeros(cfg.max_len, cfg.d))
        nn.init.normal_(self.E.weight, std=0.02)
        nn.init.normal_(self.P, std=0.01)

        self.V_theta = ScalarPotential(cfg.d, cfg.v_hidden, cfg.v_depth)

        self.raw_m_bias = nn.Parameter(
            torch.tensor(_raw_from_positive(cfg.init_m)),
            requires_grad=cfg.learn_mgamma,
        )
        if cfg.fixed_gamma is not None:
            self.raw_gamma = nn.Parameter(
                torch.tensor(0.0),
                requires_grad=False,
            )
            self._gamma_value: Optional[float] = float(cfg.fixed_gamma)
        else:
            self.raw_gamma = nn.Parameter(
                torch.tensor(_raw_from_positive(cfg.init_gamma)),
                requires_grad=cfg.learn_mgamma,
            )
            self._gamma_value = None

        if cfg.mass_mode == "global":
            pass
        elif cfg.mass_mode == "embed_head":
            self.mass_head = nn.Linear(cfg.d, 1, bias=True)
            nn.init.zeros_(self.mass_head.weight)
            nn.init.zeros_(self.mass_head.bias)
        elif cfg.mass_mode == "logfreq":
            if cfg.logfreq_path is None:
                raise ValueError(
                    "mass_mode='logfreq' requires cfg.logfreq_path (a .npy "
                    "file with one surprisal value per vocabulary id)."
                )
            surprisal = torch.from_numpy(_load_npy(cfg.logfreq_path)).float()
            if surprisal.numel() != cfg.vocab_size:
                raise ValueError(
                    f"logfreq vector length {surprisal.numel()} != "
                    f"vocab_size {cfg.vocab_size}"
                )
            self.register_buffer("logfreq_surprisal", surprisal)
            self.raw_logfreq_alpha = nn.Parameter(
                torch.tensor(
                    _raw_from_positive(max(cfg.logfreq_init_alpha, 1e-3))
                ),
                requires_grad=True,
            )
        else:
            raise ValueError(f"unknown mass_mode: {cfg.mass_mode!r}")

    @property
    def gamma(self) -> torch.Tensor:
        if self._gamma_value is not None:
            return torch.full(
                (), self._gamma_value,
                device=self.raw_gamma.device, dtype=self.raw_gamma.dtype,
            )
        return F.softplus(self.raw_gamma)

    @property
    def m_global(self) -> torch.Tensor:
        return F.softplus(self.raw_m_bias) + 1e-3

    def compute_mass(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.mass_mode == "global":
            return self.m_global
        if cfg.mass_mode == "embed_head":
            raw = self.mass_head(self.E(x))
            return F.softplus(raw + self.raw_m_bias) + 1e-3
        if cfg.mass_mode == "logfreq":
            surprisal = self.logfreq_surprisal[x]
            alpha = F.softplus(self.raw_logfreq_alpha)
            scaled = alpha * surprisal.unsqueeze(-1)
            return F.softplus(self.raw_m_bias + scaled) + 1e-3
        raise RuntimeError("unreachable")

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = self.P[:T].unsqueeze(0)
        return self.E(x) + pos

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
        h = emb
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
                V, h_in, create_graph=self.training, retain_graph=True,
            )
            f = -grad_V
            v = (v + dt * f / m_b) / (1.0 + dt * gamma)
            h = h_in + dt * v
            if return_trajectory:
                assert traj_h is not None
                traj_h.append(h.detach().cpu())

        return h, traj_h, traj_xi

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        return_xi_trajectory: bool = False,
    ):
        emb = self._embed(x)
        h_L, traj_h, traj_xi = self.integrate(
            x, emb,
            return_trajectory=return_trajectory,
            return_xi_trajectory=return_xi_trajectory,
        )
        logits = h_L @ self.E.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
        out = [logits, loss]
        if return_trajectory:
            out.append(traj_h)
        if return_xi_trajectory:
            out.append(traj_xi)
        return tuple(out) if len(out) > 2 else (out[0], out[1])

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.cfg.max_len:]
            logits, _ = self.forward(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, nxt], dim=1)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _load_npy(path: str):
    import numpy as np
    return np.load(path)
