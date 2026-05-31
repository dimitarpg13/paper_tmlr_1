"""
(Re)create the two locally-trained checkpoints that the paper-headline
notebooks consume, directly inside this repository:

  1. SPLM positive control (leak-free, gamma* = 0.10)
       -> checkpoints/splm_em_ln_leakfree_gamma0p10_seed<SEED>_ckpt_latest.pt
       produced by  ln_damping_sweep/train_splm_em_ln.py
  2. Matched-attention 8M GPT-2-style decoder baseline
       -> checkpoints/matched_baseline_shakespeare_seed<SEED>_ckpt_latest.pt
       produced by  train_matched.py

These are the exact filenames expected (locally, and under
`/MyDrive/paper_tmlr_1_checkpoints/`) by:
  - notebooks/conservative_arch/scripts/r2_jsons_a100_h100.ipynb
  - notebooks/conservative_arch/scripts/pca_symmetry_sweep_a100_h100.ipynb

The script is idempotent: a checkpoint that already exists is left untouched
unless --force is given.

Cost (single GPU): `--mode shakespeare` (the paper-grade setting, 4000 steps
each) is ~4 h for the SPLM control + ~2 h for the matched baseline. Use
`--mode smoke` (300 steps) for a fast, NON-paper-grade end-to-end pipeline
check; the smoke checkpoints are written under the SAME canonical names with a
loud warning so the downstream notebooks pick them up.

Usage:
    python3 make_checkpoints.py                       # full, both, seed 0
    python3 make_checkpoints.py --mode smoke           # fast pipeline check
    python3 make_checkpoints.py --which splm --force   # just the SPLM control
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent            # notebooks/conservative_arch
LN_DIR = SCRIPT_DIR / "ln_damping_sweep"
REPO_ROOT = SCRIPT_DIR.parent.parent                    # <repo>
DEFAULT_CKPT_DIR = REPO_ROOT / "checkpoints"
DEFAULT_LOGFREQ = REPO_ROOT / "data" / "logfreq_surprisal.npy"


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ (cd {cwd} && {' '.join(cmd)})", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[make-ckpt] {src.name}  ->  {dst}", flush=True)
    # carry over a provenance sidecar if the trainer emitted one
    prov = src.with_suffix(src.suffix + ".provenance.json")
    if not prov.exists():
        prov = src.parent / (src.stem + ".provenance.json")
    if prov.exists():
        shutil.copy2(prov, dst.parent / (dst.stem + ".provenance.json"))
        print(f"[make-ckpt] provenance  ->  {dst.parent / (dst.stem + '.provenance.json')}",
              flush=True)


def ensure_logfreq(logfreq: Path) -> None:
    if logfreq.exists():
        print(f"[make-ckpt] logfreq table present: {logfreq}", flush=True)
        return
    print(f"[make-ckpt] logfreq table absent at {logfreq} -- regenerating "
          f"via compute_unigram_frequencies.py", flush=True)
    _run([sys.executable, "compute_unigram_frequencies.py", "--out", str(logfreq)],
         cwd=SCRIPT_DIR)
    if not logfreq.exists():
        raise RuntimeError(f"failed to produce logfreq table at {logfreq}")


def make_splm(mode: str, seed: int, ckpt_dir: Path, logfreq: Path,
              device: str | None, force: bool) -> None:
    target = ckpt_dir / f"splm_em_ln_leakfree_gamma0p10_seed{seed}_ckpt_latest.pt"
    if target.exists() and not force:
        print(f"[make-ckpt] SPLM checkpoint already present (skip): {target}", flush=True)
        return
    ensure_logfreq(logfreq)
    suffix = f"gamma0p10_seed{seed}"
    with tempfile.TemporaryDirectory(prefix="make_splm_") as td:
        build = Path(td)
        cmd = [sys.executable, "train_splm_em_ln.py",
               "--mode", mode, "--seed", str(seed),
               "--fixed-gamma", "0.10",
               "--tag-suffix", suffix,
               "--logfreq-path", str(logfreq),
               "--results-dir", str(build)]
        if device:
            cmd += ["--device", device]
        _run(cmd, cwd=LN_DIR)
        produced = build / f"splm_em_ln_{mode}_{suffix}_ckpt_latest.pt"
        if not produced.exists():
            raise RuntimeError(f"SPLM trainer did not produce {produced}")
        if mode == "smoke":
            print("[make-ckpt] WARNING: --mode smoke -> this SPLM checkpoint is a "
                  "300-step NON-paper-grade artifact (pipeline check only).", flush=True)
        _copy(produced, target)


def make_matched(mode: str, seed: int, ckpt_dir: Path,
                 device: str | None, force: bool) -> None:
    target = ckpt_dir / f"matched_baseline_shakespeare_seed{seed}_ckpt_latest.pt"
    if target.exists() and not force:
        print(f"[make-ckpt] matched checkpoint already present (skip): {target}", flush=True)
        return
    cmd = [sys.executable, "train_matched.py", "--mode", mode, "--seed", str(seed)]
    if device:
        cmd += ["--device", device]
    _run(cmd, cwd=SCRIPT_DIR)
    produced = SCRIPT_DIR / "results" / f"matched_{mode}_ckpt_latest.pt"
    if not produced.exists():
        raise RuntimeError(f"matched trainer did not produce {produced}")
    if mode == "smoke":
        print("[make-ckpt] WARNING: --mode smoke -> this matched checkpoint is a "
              "300-step NON-paper-grade artifact (pipeline check only).", flush=True)
    _copy(produced, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["smoke", "shakespeare"], default="shakespeare",
                    help="shakespeare = paper-grade (4000 steps, ~6 h total); "
                         "smoke = fast pipeline check (300 steps, NOT paper-grade).")
    ap.add_argument("--which", choices=["all", "splm", "matched"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", dest="out_dir", default=str(DEFAULT_CKPT_DIR),
                    help=f"Checkpoint output directory (default: {DEFAULT_CKPT_DIR}).")
    ap.add_argument("--logfreq-path", dest="logfreq_path", default=str(DEFAULT_LOGFREQ),
                    help=f"Unigram-surprisal table for the SPLM control "
                         f"(default: {DEFAULT_LOGFREQ}; regenerated if absent).")
    ap.add_argument("--device", default=None, help="torch device override (cuda/mps/cpu).")
    ap.add_argument("--force", action="store_true",
                    help="Retrain even if the target checkpoint already exists.")
    args = ap.parse_args()

    ckpt_dir = Path(args.out_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logfreq = Path(args.logfreq_path).expanduser().resolve()

    print(f"[make-ckpt] mode={args.mode}  which={args.which}  seed={args.seed}  "
          f"out_dir={ckpt_dir}", flush=True)
    if args.mode == "smoke":
        print("[make-ckpt] *** SMOKE MODE: checkpoints are NOT paper-grade ***", flush=True)

    if args.which in ("all", "splm"):
        make_splm(args.mode, args.seed, ckpt_dir, logfreq, args.device, args.force)
    if args.which in ("all", "matched"):
        make_matched(args.mode, args.seed, ckpt_dir, args.device, args.force)

    print("\n[make-ckpt] done. Checkpoints in:", ckpt_dir, flush=True)
    for p in sorted(ckpt_dir.glob("*.pt")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
