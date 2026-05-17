"""Provenance helper for paper-headline JSON results.

Implements the `_provenance` block convention documented in
[`results/README.md`](../../results/README.md):

```
{
  "_provenance": {
    "git_commit":       "<40-char SHA, or '(not a git checkout)'>",
    "code_path":        "notebooks/.../<script>.py",
    "config_hash":      "<sha256 of resolved config>",
    "checkpoint_sha256": "<sha256 of consumed .pt, if any, else null>",
    "random_seed":      0,
    "timestamp":        "YYYY-MM-DDTHH:MM:SSZ",
    "host":             "<machine identifier>"
  },
  "...": "..."
}
```

Used by the §7 / §8 / §9 fit scripts in
`notebooks/conservative_arch/` to emit the paper-headline JSON results
named in `results/README.md`.

Public API:

- `make_provenance(script_path, config, random_seed, checkpoint_path=None)`
  -> `dict` ready to be embedded under the `_provenance` key.
- `write_paper_json(out_path, provenance, payload)` -> writes
  `{ "_provenance": <provenance>, **payload }` as pretty-printed JSON.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _git_commit(cwd: Optional[Path] = None) -> str:
    """Resolve the current git commit SHA, or a sentinel if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "(not a git checkout)"


def _repo_root() -> Optional[Path]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        )
        return Path(out.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _code_path(script_path: Path) -> str:
    """Relativise `script_path` against the repo root, fall back to basename."""
    script_path = Path(script_path).resolve()
    root = _repo_root()
    if root is None:
        return script_path.name
    try:
        return str(script_path.relative_to(root))
    except ValueError:
        return script_path.name


def _config_hash(config: Dict[str, Any]) -> str:
    """SHA256 of the JSON-serialised, key-sorted config dict.

    Non-JSON-native values (e.g. `Path`, `numpy.float64`) are coerced via
    `default=str` so the hash is stable across runs of the same config.
    """
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()


def _checkpoint_sha256(ckpt_path: Optional[Path]) -> Optional[str]:
    if ckpt_path is None:
        return None
    p = Path(ckpt_path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_provenance(
    script_path: Path,
    config: Dict[str, Any],
    random_seed: int,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the `_provenance` block for a paper-headline JSON.

    Parameters
    ----------
    script_path:
        Path to the script emitting the JSON; typically `Path(__file__)`.
    config:
        JSON-serialisable dict capturing the *resolved* config that
        produced the numbers in `payload`. The SHA256 of this dict is
        recorded under `config_hash` so reviewers can match a cited number
        to the exact configuration that produced it.
    random_seed:
        The integer seed passed to numpy / torch RNGs at fit time.
    checkpoint_path:
        If the script consumed a model checkpoint, the path to that file.
        Its SHA256 is recorded so a cited number is bound to the exact
        weights that produced it. Pass `None` for scripts that operate
        only on trajectory pickles (the trajectory's provenance is then
        the upstream extractor's responsibility).
    """
    return {
        "git_commit": _git_commit(),
        "code_path": _code_path(script_path),
        "config_hash": _config_hash(config),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "random_seed": int(random_seed),
        "timestamp": (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
        "host": platform.node(),
    }


def write_paper_json(
    out_path: Path,
    provenance: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    """Write `{ "_provenance": provenance, **payload }` to `out_path`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"_provenance": provenance, **payload}
    with out_path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")
