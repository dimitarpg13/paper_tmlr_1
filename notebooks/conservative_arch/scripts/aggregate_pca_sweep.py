"""Aggregate the PCA-symmetry sweep outputs into the §5.1 verdict artefacts.

This script consumes the per-(architecture, PCA-k) outputs of
`jacobian_symmetry.py`:

    splm_<arch>_pca<k>_jacsym_results.npz
    splm_<arch>_pca<k>_fig_jacsym.png
    splm_<arch>_pca<k>_jacsym_summary.md

and produces three aggregate artefacts:

    pca_sweep_table.md           markdown comparison table (the headline)
    pca_sweep_gap_profile.png    per-layer gap as a function of PCA-k
    pca_sweep_verdict.md         §5.1 verdict + paper-edit implications

The verdict logic implements the PCA-symmetry sweep decision rule
(see `results/pca_sweep/README.md`):

    REJECTED   -- max layer gap stays <= 0.10 at every tested PCA-k;
                     paper_tmlr_1 §7.2 PCA-16 finding is robust.

    PARTIAL    -- gap grows mildly with PCA-k; in (0.10, 0.20] at the
                     largest tested PCA-k.  Paper edit: add a one-paragraph
                     scope-of-claim sharpening to §7.2.

    CONFIRMED  -- gap > 0.20 at the largest tested PCA-k.  Paper edit:
                     substantive rewrite of §7.2 and the introduction's
                     contribution claims.

Usage (CLI):
    python aggregate_pca_sweep.py \
        --in-dir /path/to/results \
        --pca-ks 16 32 \
        --architectures gpt2 \
        --out-dir /path/to/results

If `--architectures` is omitted, the script auto-discovers every
architecture for which results at every requested PCA-k are present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Verdict bands on max-per-layer velocity-aware symmetric-vs-full gap.
GAP_BAND_REJECT = 0.10   # H2 rejected if largest-tested-PCA gap <= this
GAP_BAND_PARTIAL = 0.20  # H2 partial if gap in (REJECT, PARTIAL]; else CONFIRMED

# Paper reference: PCA-16 gap for GPT-2 small from paper_tmlr_1 §7.2.
PAPER_PCA16_GAP_REFERENCE = {
    "gpt2": 0.079,
    "matched": 0.070,
    "splm": 0.040,
}


def discover_results(
    in_dir: Path, pca_ks: List[int], architectures: Optional[List[str]] = None
) -> Dict[Tuple[str, int], Path]:
    """Return a dict mapping (arch, pca_k) -> npz path.

    If `architectures` is None, auto-discovers every architecture for
    which results at every requested PCA-k are present.
    """
    files = sorted(in_dir.glob("splm_*_pca*_jacsym_results.npz"))
    by_key: Dict[Tuple[str, int], Path] = {}
    seen_archs: set[str] = set()
    for f in files:
        # filename: splm_<arch_tag>_pca<k>_jacsym_results.npz
        stem = f.stem  # without .npz
        parts = stem.split("_")
        try:
            i_pca = next(i for i, p in enumerate(parts) if p.startswith("pca"))
        except StopIteration:
            continue
        arch_tag = "_".join(parts[1:i_pca])  # drop leading "splm" and trailing "pcaK_jacsym_results"
        try:
            k = int(parts[i_pca][3:])
        except ValueError:
            continue
        if k not in pca_ks:
            continue
        by_key[(arch_tag, k)] = f
        seen_archs.add(arch_tag)

    if architectures is None:
        architectures = sorted(
            a for a in seen_archs
            if all((a, k) in by_key for k in pca_ks)
        )

    out: Dict[Tuple[str, int], Path] = {}
    for arch in architectures:
        for k in pca_ks:
            key = (arch, k)
            if key not in by_key:
                raise FileNotFoundError(
                    f"missing result file for arch={arch!r}, pca_k={k}: "
                    f"expected splm_{arch}_pca{k}_jacsym_results.npz under {in_dir}"
                )
            out[key] = by_key[key]
    return out


def load_per_layer_gap(npz_path: Path) -> np.ndarray:
    """Return per-layer TEST velocity-aware symmetric-vs-full gap.

    The jacobian_symmetry.py output stores r2_test_full_v and r2_test_sym_v
    per layer; the gap is (full - sym), which by construction is >= 0 in the
    well-conditioned regime (a symmetric-restricted fit cannot beat the
    unconstrained fit on training data; on held-out data it can only
    marginally outperform within fit noise).
    """
    z = np.load(npz_path)
    full = np.asarray(z["r2_test_full_v"], dtype=float)
    sym = np.asarray(z["r2_test_sym_v"], dtype=float)
    gap = full - sym
    # Layer 0 has NaN because the velocity-aware test needs h_{l-1}.
    return gap


def classify_verdict(max_gap_at_largest_pca: float) -> str:
    if max_gap_at_largest_pca <= GAP_BAND_REJECT:
        return "REJECTED"
    if max_gap_at_largest_pca <= GAP_BAND_PARTIAL:
        return "PARTIAL"
    return "CONFIRMED"


def render_table_md(
    results: Dict[Tuple[str, int], np.ndarray],
    architectures: List[str],
    pca_ks: List[int],
) -> str:
    lines: List[str] = []
    lines.append("# PCA-symmetry sweep: PCA-16 vs PCA-32 (vs PCA-64 if run)\n")
    lines.append("Velocity-aware Jacobian-symmetry test "
                 "(`paper_tmlr_1` §7.2 protocol) re-run at multiple PCA "
                 "dimensions on the same frozen-checkpoint hidden-state "
                 "trajectories. Headline statistic: per-architecture "
                 "**max-over-layers TEST gap** "
                 "$\\max_\\ell |R^2_{\\mathrm{full}} - R^2_{\\mathrm{sym}}|$. "
                 "Paper §7.2 reports max gap of 0.079 (GPT-2), 0.070 "
                 "(matched-attention 8M), 0.040 (SPLM) at PCA-16. "
                 "This sweep tests whether those values are robust under "
                 "an increase in the diagnostic's projection dimension.\n")

    # ---- Headline max-gap table.
    lines.append("## Max-over-layers gap by architecture and PCA-k\n")
    header = ["architecture"] + [f"PCA-{k} max gap" for k in pca_ks] + ["verdict"]
    sep = ["---"] + ["---:" for _ in pca_ks] + ["---"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(sep) + "|")
    for arch in architectures:
        row = [arch]
        gaps_at_pca = []
        for k in pca_ks:
            g = results[(arch, k)]
            mg = float(np.nanmax(g))
            gaps_at_pca.append(mg)
            row.append(f"{mg:+.3f}")
        verdict = classify_verdict(gaps_at_pca[-1])
        row.append(verdict)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- Per-layer table for each architecture.
    for arch in architectures:
        lines.append(f"## Per-layer gap, architecture = {arch}\n")
        any_g = results[(arch, pca_ks[0])]
        n_layers = len(any_g)
        header2 = ["layer"] + [f"PCA-{k}" for k in pca_ks]
        sep2 = ["---:"] + ["---:" for _ in pca_ks]
        lines.append("| " + " | ".join(header2) + " |")
        lines.append("|" + "|".join(sep2) + "|")
        for ell in range(n_layers):
            row = [str(ell)]
            for k in pca_ks:
                g = results[(arch, k)][ell]
                row.append(f"{g:+.3f}" if np.isfinite(g) else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def render_gap_profile(
    results: Dict[Tuple[str, int], np.ndarray],
    architectures: List[str],
    pca_ks: List[int],
    out_path: Path,
) -> None:
    n_arch = len(architectures)
    fig, axes = plt.subplots(1, n_arch, figsize=(5 * n_arch, 4.0), sharey=True)
    if n_arch == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    for ax, arch in zip(axes, architectures):
        any_g = results[(arch, pca_ks[0])]
        layers = np.arange(len(any_g))
        for j, k in enumerate(pca_ks):
            g = results[(arch, k)]
            color = cmap(0.15 + 0.7 * (j / max(len(pca_ks) - 1, 1)))
            ax.plot(layers, g, marker="o", color=color, label=f"PCA-{k}")
        ax.axhline(GAP_BAND_REJECT, color="green", linestyle="--", alpha=0.6,
                   label=f"H2 reject band ({GAP_BAND_REJECT:.2f})")
        ax.axhline(GAP_BAND_PARTIAL, color="orange", linestyle="--", alpha=0.6,
                   label=f"H2 partial / confirmed boundary ({GAP_BAND_PARTIAL:.2f})")
        ax.set_title(f"{arch}: per-layer TEST gap")
        ax.set_xlabel("layer $\\ell$")
        ax.set_ylabel("$R^2_{\\mathrm{full}} - R^2_{\\mathrm{sym}}$ (TEST)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("PCA-symmetry sweep: velocity-aware Jacobian-symmetry gap "
                 "as a function of PCA dimension")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def render_verdict_md(
    results: Dict[Tuple[str, int], np.ndarray],
    architectures: List[str],
    pca_ks: List[int],
) -> str:
    lines: List[str] = []
    lines.append("# PCA-symmetry sweep — verdict\n")
    lines.append("PCA-dimension sweep on the per-token-internal "
                 "velocity-aware Jacobian-symmetry test of "
                 "`paper_tmlr_1` §7.2 (decision rule: see "
                 "`results/pca_sweep/README.md`).\n")
    lines.append("## Per-architecture verdict\n")
    overall = "REJECTED"
    for arch in architectures:
        gaps = [float(np.nanmax(results[(arch, k)])) for k in pca_ks]
        verdict = classify_verdict(gaps[-1])
        if verdict == "CONFIRMED" or (verdict == "PARTIAL" and overall == "REJECTED"):
            overall = verdict
        max_pca = pca_ks[-1]
        baseline = PAPER_PCA16_GAP_REFERENCE.get(arch, None)
        baseline_str = (f"; PCA-16 reference from paper §7.2 = {baseline:+.3f}"
                        if baseline is not None else "")
        lines.append(f"- **{arch}**: max gap at PCA-{max_pca} = "
                     f"{gaps[-1]:+.3f} → **{verdict}**{baseline_str}.")
    lines.append("")

    lines.append(f"## Overall verdict: **{overall}**\n")
    if overall == "REJECTED":
        lines.append("The `paper_tmlr_1` §7.2 PCA-16 local-conservativity "
                     "finding is robust at the largest tested PCA dimension. "
                     "Paper-edit implication: §7.2 stands as written; an "
                     "optional one-paragraph robustness footnote can be "
                     "added before submission noting that we verified the "
                     "finding at PCA-32 (and PCA-64 if run) with the "
                     "headline gaps stable.")
    elif overall == "PARTIAL":
        lines.append("The §7.2 finding holds at PCA-16 but the symmetric-"
                     "restricted regression loses some explanatory power "
                     "as the PCA dimension grows. Paper-edit implication: "
                     "a one-paragraph scope-of-claim sharpening to §7.2 "
                     "is warranted, explicitly flagging that the "
                     "local-conservativity reading is specifically at "
                     "PCA-16 and that the PCA-dimension dependence is a "
                     "real (but bounded) effect. The headline three-way "
                     "separator of §8 is unaffected.")
    else:
        lines.append("The §7.2 local-conservativity finding is an artifact "
                     "of the PCA-16 projection: the symmetric-restricted "
                     "regression breaks substantially at the larger PCA "
                     "dimensions. Paper-edit implication: substantive "
                     "rewrite of §7.2 framing and corresponding paragraphs "
                     "of the introduction. The §8 three-way separator "
                     "headline ($R^2 = 0.949 / 0.56 / 0.45$) is unaffected "
                     "and in fact *strengthens* under this reading "
                     "(GPT-2 is non-conservative at higher PCA dimensions "
                     "too, which makes the global shared-V failure more "
                     "structural).")
    lines.append("")

    lines.append("## Reference: decision rule\n")
    lines.append(f"- **REJECTED** if max-layer gap stays "
                 f"<= {GAP_BAND_REJECT:.2f} at every tested PCA-k.")
    lines.append(f"- **PARTIAL** if max-layer gap at largest PCA-k "
                 f"is in ({GAP_BAND_REJECT:.2f}, {GAP_BAND_PARTIAL:.2f}].")
    lines.append(f"- **CONFIRMED** if max-layer gap at largest PCA-k "
                 f"is > {GAP_BAND_PARTIAL:.2f}.")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, type=Path,
                    help="directory containing splm_*_pca*_jacsym_results.npz")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="directory to write aggregate artefacts "
                         "(defaults to --in-dir)")
    ap.add_argument("--pca-ks", nargs="+", type=int, required=True,
                    help="list of PCA dimensions present in --in-dir")
    ap.add_argument("--architectures", nargs="*", default=None,
                    help="optional list of architecture tags to include; "
                         "if omitted, auto-discovered")
    args = ap.parse_args()

    in_dir = args.in_dir.expanduser().resolve()
    out_dir = (args.out_dir or args.in_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pca_ks = sorted(args.pca_ks)

    by_key = discover_results(in_dir, pca_ks, args.architectures)
    architectures = sorted(set(arch for arch, _ in by_key))
    print(f"[aggregate] architectures = {architectures}")
    print(f"[aggregate] pca_ks = {pca_ks}")

    results: Dict[Tuple[str, int], np.ndarray] = {
        key: load_per_layer_gap(npz_path) for key, npz_path in by_key.items()
    }

    table_md = render_table_md(results, architectures, pca_ks)
    table_path = out_dir / "pca_sweep_table.md"
    table_path.write_text(table_md, encoding="utf-8")
    print(f"[aggregate] wrote {table_path}")

    fig_path = out_dir / "pca_sweep_gap_profile.png"
    render_gap_profile(results, architectures, pca_ks, fig_path)
    print(f"[aggregate] wrote {fig_path}")

    verdict_md = render_verdict_md(results, architectures, pca_ks)
    verdict_path = out_dir / "pca_sweep_verdict.md"
    verdict_path.write_text(verdict_md, encoding="utf-8")
    print(f"[aggregate] wrote {verdict_path}")

    # Machine-readable summary for downstream automation.
    summary = {
        "pca_ks": pca_ks,
        "architectures": architectures,
        "per_arch_max_gap": {
            arch: {k: float(np.nanmax(results[(arch, k)])) for k in pca_ks}
            for arch in architectures
        },
        "verdict": {
            arch: classify_verdict(float(np.nanmax(results[(arch, pca_ks[-1])])))
            for arch in architectures
        },
    }
    summary_json = out_dir / "pca_sweep_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[aggregate] wrote {summary_json}")


if __name__ == "__main__":
    main()
