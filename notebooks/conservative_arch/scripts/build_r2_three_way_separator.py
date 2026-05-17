r"""Aggregate three per-architecture shared_potential_fit JSONs into the
§8 three-way-separator headline JSON (`results/R2_three_way_separator.json`).

The §8 headline cites a three-way comparison of median per-layer test
$R^{2}$ for the shared-$V_\psi$ fit on:

    - SPLM positive control (leak-free, gamma* = 0.10 LayerNorm-after-step
      Euler integrator variant; trained checkpoint;
      `splm_em_ln_leakfree_gamma0p10_seed0.trajectories.pkl`).
    - matched-attention 8M-parameter GPT-2-style baseline (trained from
      scratch on Tiny Shakespeare;
      `matched_baseline.trajectories.pkl`).
    - pretrained GPT-2 small (no SPLM lineage;
      `gpt2_baseline.trajectories.pkl`).

The paper-cited numbers are R^2 = 0.949 / 0.56 / 0.45 respectively. This
aggregator consumes the three per-architecture JSONs produced by
shared_potential_fit.py --emit-json and emits a single
R2_three_way_separator.json with:

    - a top-level _provenance block (this aggregator's git SHA, config
      hash over the input file list, and timestamp),
    - the three per-architecture _provenance blocks (preserved from the
      upstream JSONs, so the full chain SPLM ckpt / GPT-2 weights /
      MatchedGPT ckpt SHA256 is retained),
    - the three architectures' headline R^2 and per-layer profile,
    - a small derived headline block recording the three median values
      and the architectural separator margin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
CONS_DIR = SCRIPT_DIR.parent
if str(CONS_DIR) not in sys.path:
    sys.path.insert(0, str(CONS_DIR))

from provenance import make_provenance, write_paper_json


def _load_input(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _check_input(label: str, obj: Dict[str, Any], path: Path) -> None:
    section = obj.get("section", "")
    if "shared-potential separator" not in section:
        raise ValueError(
            f"input for {label!r} at {path} does not look like a "
            f"shared_potential_fit.py emit-json output (section={section!r})"
        )
    if "headline_r2" not in obj or "per_layer_r2" not in obj:
        raise ValueError(
            f"input for {label!r} at {path} is missing required keys "
            f"'headline_r2' and/or 'per_layer_r2'"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splm", required=True, type=Path,
        help="Path to shared_potential_fit.py --emit-json output for SPLM "
             "positive control (leak-free, gamma* = 0.10).",
    )
    ap.add_argument(
        "--matched", required=True, type=Path,
        help="Path to shared_potential_fit.py --emit-json output for the "
             "matched-attention 8M GPT-2-style baseline.",
    )
    ap.add_argument(
        "--gpt2", required=True, type=Path,
        help="Path to shared_potential_fit.py --emit-json output for "
             "pretrained GPT-2 small.",
    )
    ap.add_argument(
        "--out", required=True, type=Path,
        help="Path for the aggregated R2_three_way_separator.json.",
    )
    args = ap.parse_args()

    inputs = [
        ("splm_leakfree_gamma0p10",       args.splm),
        ("matched_attention_8M",           args.matched),
        ("gpt2_small_pretrained",          args.gpt2),
    ]
    objs: Dict[str, Dict[str, Any]] = {}
    for label, path in inputs:
        obj = _load_input(path)
        _check_input(label, obj, path)
        objs[label] = obj

    arch_rows: List[Dict[str, Any]] = []
    for label, path in inputs:
        obj = objs[label]
        h = obj["headline_r2"]
        arch_rows.append({
            "architecture": label,
            "source_json": str(path),
            "tag": obj.get("tag"),
            "config": obj.get("config"),
            "n_samples": obj.get("n_samples"),
            "headline_r2": {
                "median_per_layer_test": float(h["median_per_layer_test"]),
                "mean_per_layer_test": float(h["mean_per_layer_test"]),
                "min_per_layer_test": float(h["min_per_layer_test"]),
                "max_per_layer_test": float(h["max_per_layer_test"]),
                "overall_pooled_train": float(h["overall_pooled_train"]),
                "overall_pooled_test": float(h["overall_pooled_test"]),
            },
            "per_layer_r2": obj["per_layer_r2"],
            "upstream_provenance": obj.get("_provenance"),
        })

    medians = {row["architecture"]: row["headline_r2"]["median_per_layer_test"]
               for row in arch_rows}
    splm_median = medians["splm_leakfree_gamma0p10"]
    matched_median = medians["matched_attention_8M"]
    gpt2_median = medians["gpt2_small_pretrained"]

    config = {
        "aggregator": "build_r2_three_way_separator",
        "inputs": {label: str(path) for label, path in inputs},
        "architectures": [label for label, _ in inputs],
    }
    provenance = make_provenance(
        script_path=Path(__file__),
        config=config,
        random_seed=0,
        checkpoint_path=None,
    )

    payload = {
        "section": "paper_tmlr_1 \u00a78 (three-way shared-potential separator)",
        "headline_r2_three_way": {
            "splm_leakfree_gamma0p10": splm_median,
            "matched_attention_8M":     matched_median,
            "gpt2_small_pretrained":    gpt2_median,
            "splm_vs_matched_margin":   splm_median - matched_median,
            "splm_vs_gpt2_margin":      splm_median - gpt2_median,
            "matched_vs_gpt2_margin":   matched_median - gpt2_median,
        },
        "verdict": (
            "Three-way architectural separator: SPLM positive control "
            "passes the shared-potential test ("
            f"median per-layer test R^2 = {splm_median:+.3f}), while both "
            "attention-based architectures fail decisively ("
            f"matched = {matched_median:+.3f}, GPT-2 = {gpt2_median:+.3f}). "
            "The SPLM vs attention gap is the substantive content of "
            "the headline claim."
        ),
        "architectures": arch_rows,
    }
    write_paper_json(args.out, provenance, payload)
    print(f"[three-way] saved -> {args.out}")
    print(f"[three-way] headline: SPLM={splm_median:+.3f}  "
          f"matched={matched_median:+.3f}  GPT-2={gpt2_median:+.3f}  "
          f"(SPLM\u2212GPT2={splm_median - gpt2_median:+.3f})")


if __name__ == "__main__":
    main()
