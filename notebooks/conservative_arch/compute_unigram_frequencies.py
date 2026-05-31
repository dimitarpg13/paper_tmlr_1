"""
Precompute unigram surprisal -log p_hat(v) for every GPT-2 BPE token id, using
the Tiny Shakespeare train split as the corpus.

This regenerates the `mass_mode='logfreq'` table consumed by the SPLM
positive-control trainer (`ln_damping_sweep/train_splm_em_ln.py`) and by
trajectory extraction. A precomputed copy ships with the repository at
`<repo>/data/logfreq_surprisal.npy`, so this script is only needed if you want
to regenerate that file from scratch (e.g. for an end-to-end provenance audit).

Smoothing:
  p_hat(v) = (c_v + 1) / (N + V) -- add-one Laplace smoothing, so every
  vocabulary id gets a finite surprisal even when unseen in the corpus.
Surprisal:
  s(v) = -log p_hat(v), with natural log (units: nats). The scale is learned
  via the softplus-ed alpha in the model, so the choice of log base does not
  matter for training.

Output: a float32 vector of shape (50257,), ready to be passed to
SPLMSARFMassConfig(mass_mode='logfreq', logfreq_path=...).

Usage:
    python3 compute_unigram_frequencies.py
    python3 compute_unigram_frequencies.py --out /path/to/logfreq_surprisal.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent           # notebooks/conservative_arch
REPO_ROOT = SCRIPT_DIR.parent.parent                   # <repo>
DEFAULT_OUT = REPO_ROOT / "data" / "logfreq_surprisal.npy"

sys.path.insert(0, str(SCRIPT_DIR))
from data_module import load_tiny_shakespeare  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help=f"Output .npy path (default: {DEFAULT_OUT}).",
    )
    args = ap.parse_args()

    vocab_size = 50257
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = load_tiny_shakespeare()
    N = len(train_ids)
    print(f"[freq] corpus tokens: train={N:,}  val={len(val_ids):,}  "
          f"vocab_size={vocab_size}")

    counts = np.bincount(train_ids, minlength=vocab_size).astype(np.int64)
    nz = int((counts > 0).sum())
    print(f"[freq] unique types seen: {nz:,} / {vocab_size:,} "
          f"({100 * nz / vocab_size:.1f}%)")

    p = (counts + 1.0) / (N + vocab_size)
    surprisal = -np.log(p).astype(np.float32)
    print(f"[freq] surprisal  min={surprisal.min():.3f}  "
          f"max={surprisal.max():.3f}  "
          f"mean={surprisal.mean():.3f}  median={np.median(surprisal):.3f}")

    seen = surprisal[counts > 0]
    print(f"[freq] seen-only  min={seen.min():.3f}  "
          f"max={seen.max():.3f}  mean={seen.mean():.3f}")

    np.save(out, surprisal)
    print(f"[freq] saved -> {out}  "
          f"shape={surprisal.shape}  dtype={surprisal.dtype}")

    meta = out.with_suffix(".meta.txt")
    with meta.open("w") as f:
        f.write("Unigram surprisal -log p_hat(v) for GPT-2 BPE vocabulary\n")
        f.write("Corpus: Tiny Shakespeare (train split)\n")
        f.write(f"Tokens: {N:,}\n")
        f.write(f"Vocab:  {vocab_size:,}\n")
        f.write(f"Unique seen: {nz:,}\n")
        f.write("Smoothing: add-one Laplace, p = (c+1)/(N+V)\n")
        f.write(f"surprisal: min={surprisal.min():.3f}  "
                f"max={surprisal.max():.3f}  "
                f"mean={surprisal.mean():.3f}  "
                f"median={np.median(surprisal):.3f}\n")
    print(f"[freq] metadata -> {meta}")


if __name__ == "__main__":
    main()
