"""Seeding and run manifests: the reproducibility contract.

Same ``run_manifest.json`` (seed, thread count, library versions, env
description, resolved hyperparameters) => identical weights. See
tests/test_reproducibility.py.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["set_global_seed", "collect_versions", "git_commit", "write_run_manifest", "read_run_manifest"]

MANIFEST_VERSION = 1


def set_global_seed(seed: Optional[int], deterministic_torch: bool = True) -> None:
    """Seed random / numpy / torch and request deterministic torch kernels."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except ImportError:
        pass


def collect_versions() -> Dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "stable_baselines3", "gymnasium", "numpy", "matplotlib"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "?"))
        except Exception:
            out[name] = "not installed"
    try:
        import torch
        out["torch_num_threads"] = str(torch.get_num_threads())
        out["torch_deterministic"] = str(torch.are_deterministic_algorithms_enabled())
    except Exception:
        pass
    return out


def git_commit(root: Optional[Path] = None) -> Optional[str]:
    try:
        cwd = str(root) if root else None
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd, timeout=5)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, cwd=cwd, timeout=5)
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += "-dirty"
            return sha
    except Exception:
        return None
    return None


def write_run_manifest(
    path: Path,
    *,
    seed: Optional[int],
    algorithm: str,
    env_description: Dict[str, Any],
    resolved_hyperparams: Dict[str, Any],
    cli_args: Optional[Dict[str, Any]] = None,
    eval_standard: Optional[Dict[str, Any]] = None,
    curriculum: Optional[Dict[str, Any]] = None,
    lineage: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "algorithm": str(algorithm).upper(),
        "git_commit": git_commit(project_root),
        "versions": collect_versions(),
        "command": " ".join(sys.argv),
        "cli_args": cli_args or {},
        "env": env_description,
        "hyperparameters": resolved_hyperparams,
        "eval_standard": eval_standard or {},
        "curriculum": curriculum or {},
        "lineage": lineage or [],
        "run_dir": str(path.parent.resolve()),
    }
    if extra:
        manifest.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=_default)
    return manifest


def read_run_manifest(path: Path) -> Dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if callable(o):
        return f"<callable {getattr(o, '__name__', type(o).__name__)}>"
    return str(o)
