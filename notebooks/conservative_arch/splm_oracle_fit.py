"""
SPLM oracle reference for the step-2 shared-V_psi fit.

Setup
-----
The step-2 test fit a generic MLP V_psi(h) and measured how well
a single scalar can reproduce per-layer Delta x_l across all
layers and sentences.  For SPLM, the "ideal" such scalar is
SPLM's own learned V_theta(xi, h) -- by construction the model's
dynamics is

    Delta h_l  =  v_{l+1} * dt
               =  (1/(1+dt*gamma)) * [ dt * v_l  -  dt^2/m * grad_h V_theta(xi, h_l) ]

Plugging this oracle V_theta into the step-2 ansatz

    Delta h_l ~ alpha_l * v_l  -  beta_l * grad_h V_theta(xi, h_l)

should fit essentially perfectly (TRAIN R^2 ~ 1, TEST R^2 ~ 1), because
it IS the closed-form integrator up to unit-constants alpha_l, beta_l.
The test therefore quantifies two things:

  A) How much of the step-2 SPLM residual is explained simply by the
     context dependence dropped when V_psi(h) ignores xi.
  B) Whether the step-2 learned V_psi recovers (an approximation of)
     SPLM's true V_theta -- by comparing per-layer R^2 oracle vs learned.

This is the PC-side upper bound for step 2.

Works on raw h-space (not x_ps), since V_theta lives there.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model import ScalarPotentialLM, SPLMConfig
from e_init_corpus import CORPUS
from trajectory_extraction import load_splm_any_variant


SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


def tokenize(sentence: str, max_len: int) -> np.ndarray:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok.encode(sentence)
    ids = ids[:max_len]
    return np.asarray(ids, dtype=np.int64)


def _extract_oracle_tuples_plain(
    model: ScalarPotentialLM, sentence: str, device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plain SPLM (model.py) oracle re-integration.

    Dynamics:
       h_{l+1} = h_l + dt * v_{l+1}
       v_{l+1} = (v_l + dt*f_l/m) / (1 + dt*gamma)
       f_l     = -grad_h V_theta(xi, h_l)        # xi fixed at h_0 pool

    Returns (H, GV) with shapes (L+1, T, d) and (L, T, d).
    """
    max_len = model.cfg.max_len
    ids = tokenize(sentence, max_len)
    x = torch.from_numpy(ids).unsqueeze(0).to(device)

    with torch.enable_grad():
        emb, xi = model._embed_and_pool(x)
        h = emb
        v = torch.zeros_like(h)
        dt, m, gamma = model.cfg.dt, model.m, model.gamma

        H  = [h.detach().cpu().numpy()[0]]
        GV = []
        for _ in range(model.cfg.L):
            h_in = h.detach().requires_grad_(True)
            V_out = model.V_theta(xi, h_in).sum()
            grad_V, = torch.autograd.grad(V_out, h_in)
            GV.append(grad_V.detach().cpu().numpy()[0])
            f = -grad_V
            v = (v + dt * f / m) / (1.0 + dt * gamma)
            h = h_in + dt * v
            H.append(h.detach().cpu().numpy()[0])

    return (
        np.stack(H, axis=0).astype(np.float32),
        np.stack(GV, axis=0).astype(np.float32),
    )


def _extract_oracle_tuples_sarf_mass_ln(
    model, sentence: str, device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """SARFMassLN SPLM (model_ln.py) oracle re-integration.

    Mirrors ScalarPotentialLMSARFMassLN.integrate(...) faithfully:
      - per-token mass m = compute_mass(x, emb_initial)
      - per-step xi = causal_cumulative_mean(h_l.detach()) if causal_force
      - post-step LN projection of h_{l+1} (if ln_after_step)
      - gamma is the fixed-or-learned scalar exposed by model.gamma

    Returns (H, GV) with shapes (L+1, T, d) and (L, T, d). With LN
    folded into the dynamics, Delta h_l is no longer exactly affine in
    (v_l, grad_V), so the downstream linear oracle fit reports R^2 < 1
    in general; that residual is precisely the part of the SARFMassLN
    dynamics that LN-after-step distorts (and is the headline §9.1
    quantity).
    """
    from model_sarf_mass import causal_cumulative_mean
    import torch.nn.functional as F

    cfg = model.cfg
    max_len = cfg.max_len
    ids = tokenize(sentence, max_len)
    x = torch.from_numpy(ids).unsqueeze(0).to(device)

    with torch.enable_grad():
        emb = model._embed(x)
        h = model._project(emb) if cfg.ln_after_step else emb
        v = torch.zeros_like(h)
        dt, gamma = cfg.dt, model.gamma
        m_b = model.compute_mass(x, emb)

        H = [h.detach().cpu().numpy()[0]]
        GV = []
        for _ in range(cfg.L):
            xi_input = h.detach() if cfg.causal_force else h
            xi_now = causal_cumulative_mean(xi_input)

            h_in = h.detach().requires_grad_(True)
            V_out = model.V_theta(xi_now, h_in).sum()
            grad_V, = torch.autograd.grad(V_out, h_in)
            GV.append(grad_V.detach().cpu().numpy()[0])
            f = -grad_V
            v = (v + dt * f / m_b) / (1.0 + dt * gamma)
            h_new = h_in + dt * v
            if cfg.ln_after_step:
                h_new = model._project(h_new)
            h = h_new
            H.append(h.detach().cpu().numpy()[0])

    return (
        np.stack(H, axis=0).astype(np.float32),
        np.stack(GV, axis=0).astype(np.float32),
    )


def extract_oracle_tuples(model, sentence: str, device: str, variant: str):
    """Dispatch the per-sentence oracle re-integration by checkpoint variant."""
    if variant == "sarf_mass_ln":
        return _extract_oracle_tuples_sarf_mass_ln(model, sentence, device)
    return _extract_oracle_tuples_plain(model, sentence, device)


# ---------------------------------------------------------------------------
def build_pools(model, sentences: List[Dict], device: str, variant: str):
    """For each (split, layer l >= 1) build pooled (v, dh, grad_V_theta, grad_V_theta_h_aligned).

    We fit on

        Delta h_l  ~=  alpha_l * v_l  -  beta_l * grad_V_theta(xi, h_l)

    where v_l := h_l - h_{l-1}.  Note: v_l at layer l>=1 requires h_{l-1}.
    """
    train_V, train_dH, train_G, train_Lay = [], [], [], []
    test_V,  test_dH,  test_G,  test_Lay  = [], [], [], []

    for s in sentences:
        H, GV = extract_oracle_tuples(model, s["sentence"], device, variant)
        L = H.shape[0] - 1
        for ell in range(1, L):
            v_l   = H[ell]   - H[ell - 1]
            dh_l  = H[ell+1] - H[ell]
            g_l   = GV[ell]               # grad V_theta at h_l -- for l in 1..L-1
            layer = np.full((H.shape[1],), ell, dtype=np.int64)
            bucket = (train_V, train_dH, train_G, train_Lay) \
                if s["split"] == "train" \
                else (test_V, test_dH, test_G, test_Lay)
            bucket[0].append(v_l);  bucket[1].append(dh_l)
            bucket[2].append(g_l);  bucket[3].append(layer)

    def cat(lst):
        return np.concatenate(lst, axis=0) if lst else np.zeros((0, 1), dtype=np.float32)

    return (cat(train_V),  cat(train_dH),  cat(train_G),  cat(train_Lay),
            cat(test_V),   cat(test_dH),   cat(test_G),   cat(test_Lay))


# ---------------------------------------------------------------------------
def fit_scalars(V: np.ndarray, G: np.ndarray, Y: np.ndarray, LAY: np.ndarray):
    """Per-layer least-squares: [alpha_l, beta_l] = argmin || Y - alpha V - beta*(-G) ||^2."""
    alpha = {}; beta = {}
    Y_pred = np.zeros_like(Y)
    for ell in np.unique(LAY):
        m = LAY == ell
        v = V[m].reshape(-1); g = (-G[m]).reshape(-1); y = Y[m].reshape(-1)
        A = np.stack([v, g], axis=1)  # (N*d, 2)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coef
        alpha[int(ell)] = float(a); beta[int(ell)] = float(b)
        Y_pred[m] = a * V[m] + b * (-G[m])
    return Y_pred, alpha, beta


def r2_per_layer(Y: np.ndarray, Yp: np.ndarray, LAY: np.ndarray) -> Dict[int, float]:
    out = {}
    for ell in np.unique(LAY):
        m = LAY == ell
        num = float(np.sum((Y[m] - Yp[m]) ** 2))
        den = float(np.sum((Y[m] - Y[m].mean(0, keepdims=True)) ** 2))
        out[int(ell)] = 1.0 - num / (den + 1e-12)
    return out


def predict(V, G, LAY, alpha, beta):
    Yp = np.zeros_like(V)
    for ell in np.unique(LAY):
        m = LAY == ell
        Yp[m] = alpha[int(ell)] * V[m] + beta[int(ell)] * (-G[m])
    return Yp


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the train/test sentence split.")
    ap.add_argument(
        "--emit-json", default=None,
        help="If set, also write a paper-headline JSON with provenance + "
             "per-layer oracle R^2 at this path.",
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                              else ("mps" if torch.backends.mps.is_available() else "cpu"))

    ckpt_path = Path(args.ckpt)
    tag = args.tag or ckpt_path.stem.replace(".ckpt_latest", "").replace("splm_", "")
    print(f"[splm-oracle] ckpt={ckpt_path.name}  tag={tag}  device={device}")

    model, model_cfg, variant, _ck = load_splm_any_variant(ckpt_path, device)
    L = model_cfg.L
    d = model_cfg.d
    print(f"[splm-oracle] variant={variant}  d={d}  L={L}")

    sentences = []
    rng = np.random.default_rng(args.seed)
    for domain, arr in CORPUS.items():
        idx = rng.permutation(len(arr))
        train_idx = idx[:int(0.8 * len(arr))]
        test_idx  = idx[int(0.8 * len(arr)):]
        for i, s in enumerate(arr):
            split = "train" if i in set(train_idx.tolist()) else "test"
            sentences.append(dict(sentence=s, domain=domain, split=split))
    print(f"[splm-oracle] sentences: train="
          f"{sum(1 for s in sentences if s['split']=='train')}, "
          f"test={sum(1 for s in sentences if s['split']=='test')}")

    (V_tr, Y_tr, G_tr, Ltr,
     V_te, Y_te, G_te, Lte) = build_pools(model, sentences, device, variant)
    print(f"[splm-oracle] samples: train={V_tr.shape[0]:,}  test={V_te.shape[0]:,}")

    Y_pred_tr, alpha, beta = fit_scalars(V_tr, G_tr, Y_tr, Ltr)
    Y_pred_te = predict(V_te, G_te, Lte, alpha, beta)
    r2_tr = r2_per_layer(Y_tr, Y_pred_tr, Ltr)
    r2_te = r2_per_layer(Y_te, Y_pred_te, Lte)

    print("\n[splm-oracle] per-layer TEST R^2 (oracle V_theta):")
    for ell in sorted(r2_te):
        print(f"[splm-oracle]   l={ell:2d}   train {r2_tr[ell]:+.4f}   "
              f"test {r2_te[ell]:+.4f}   alpha={alpha[ell]:+.4f}   "
              f"beta={beta[ell]:+.4f}")

    # Compare to step-2 learned V_psi fit (on same SPLM).
    step2_npz = RESULTS_DIR / "sharedV_shakespeare_ckpt_latest_results.npz"
    cmp_layers, cmp_shv = None, None
    if step2_npz.exists():
        z = np.load(step2_npz)
        cmp_layers = z["layers"].tolist()
        cmp_shv    = z["r2_shv_test"].tolist()

    # ---- Save ----
    out_npz = RESULTS_DIR / f"splm_oracle_{tag}_results.npz"
    np.savez(
        out_npz,
        layers=np.array(sorted(r2_te.keys())),
        r2_train=np.array([r2_tr[e] for e in sorted(r2_tr)]),
        r2_test =np.array([r2_te[e] for e in sorted(r2_te)]),
        alpha=np.array([alpha[e] for e in sorted(alpha)]),
        beta =np.array([beta[e]  for e in sorted(beta)]),
    )
    print(f"[splm-oracle] saved -> {out_npz}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 4.3))
    layers = sorted(r2_te.keys())
    y_or_te = [r2_te[e] for e in layers]
    y_or_tr = [r2_tr[e] for e in layers]
    ax.plot(layers, y_or_tr, marker="o", linewidth=2.0, color="tab:blue",
            label=f"oracle $V_\\theta$ TRAIN")
    ax.plot(layers, y_or_te, marker="o", linewidth=2.0, color="tab:cyan",
            label=f"oracle $V_\\theta$ TEST")
    if cmp_shv is not None:
        ax.plot(cmp_layers, cmp_shv, marker="s", linewidth=1.5, color="tab:green",
                linestyle="--", label="learned $V_\\psi$ TEST (step-2)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(1, color="gray", linewidth=0.3, linestyle=":")
    ax.set_xlabel("layer $\\ell$"); ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"SPLM oracle $V_\\theta$ vs. learned $V_\\psi$ -- {tag}")
    fig.tight_layout()
    fig_path = RESULTS_DIR / f"splm_oracle_{tag}_fig.png"
    fig.savefig(fig_path, dpi=140); plt.close(fig)
    print(f"[splm-oracle] saved -> {fig_path}")

    # ---- Markdown ----
    md = RESULTS_DIR / f"splm_oracle_{tag}_summary.md"
    with md.open("w") as f:
        f.write(f"# SPLM oracle fit -- {tag}\n\n")
        f.write("**Purpose.**  Upper-bound reference for the step-2 "
                "shared-$V_\\psi$ fit on SPLM.  Replaces the learned "
                "$V_\\psi(h)$ with SPLM's own $V_\\theta(\\xi, h)$ and "
                "keeps the same per-layer $\\alpha_\\ell, \\beta_\\ell$ "
                "fitting procedure.  Numerical mismatch from 1.0 is then "
                "purely due to integrator constants and numerical precision.\n\n")
        f.write("## Per-layer $R^2$  (oracle $V_\\theta$)\n\n")
        f.write("| layer | TRAIN | TEST | $\\alpha_\\ell$ | $\\beta_\\ell$ |\n")
        f.write("|--:|--:|--:|--:|--:|\n")
        for ell in layers:
            f.write(f"| {ell} | {r2_tr[ell]:+.4f} | {r2_te[ell]:+.4f} | "
                    f"{alpha[ell]:+.4f} | {beta[ell]:+.4f} |\n")
        if cmp_shv is not None:
            f.write("\n## Oracle vs. learned $V_\\psi$ (step-2)\n\n")
            f.write("| layer | oracle TEST | learned $V_\\psi$ TEST | gap |\n")
            f.write("|--:|--:|--:|--:|\n")
            for ell, shv in zip(cmp_layers, cmp_shv):
                orc = r2_te.get(int(ell))
                if orc is None:
                    continue
                f.write(f"| {ell} | {orc:+.4f} | {shv:+.4f} | {orc - shv:+.4f} |\n")
        f.write("\n![fig](splm_oracle_"
                f"{tag}_fig.png)\n")
    print(f"[splm-oracle] saved -> {md}")

    if args.emit_json is not None:
        from provenance import make_provenance, write_paper_json

        per_layer_test = [float(r2_te[ell]) for ell in layers]
        per_layer_train = [float(r2_tr[ell]) for ell in layers]
        median_test = float(np.median(per_layer_test))
        mean_test = float(np.mean(per_layer_test))
        min_test = float(np.min(per_layer_test))
        max_test = float(np.max(per_layer_test))

        config = {
            "fit": "splm_oracle_fit",
            "checkpoint": ckpt_path.name,
            "checkpoint_variant": variant,
            "d": int(d),
            "L": int(L),
            "method": "per-layer closed-form LS on (alpha_l, beta_l) "
                      "with oracle V_theta",
        }
        provenance = make_provenance(
            script_path=Path(__file__),
            config=config,
            random_seed=int(args.seed),
            checkpoint_path=ckpt_path,
        )
        payload = {
            "section": "paper_tmlr_1 \u00a79.1 (SPLM oracle reference "
                       "for the shared-V_psi fit)",
            "tag": tag,
            "config": config,
            "n_samples": {
                "train": int(V_tr.shape[0]),
                "test": int(V_te.shape[0]),
            },
            "headline_r2": {
                "median_per_layer_test": median_test,
                "mean_per_layer_test": mean_test,
                "min_per_layer_test": min_test,
                "max_per_layer_test": max_test,
            },
            "per_layer_r2": {
                "layers": [int(ell) for ell in layers],
                "oracle_test": per_layer_test,
                "oracle_train": per_layer_train,
            },
            "learned_scalars": {
                "alpha": [float(alpha[ell]) for ell in layers],
                "beta": [float(beta[ell]) for ell in layers],
            },
        }
        write_paper_json(Path(args.emit_json), provenance, payload)
        print(f"[splm-oracle] saved -> {args.emit_json}")


if __name__ == "__main__":
    main()
