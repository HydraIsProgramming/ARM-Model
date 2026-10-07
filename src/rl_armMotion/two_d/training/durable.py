"""
Crash-safe file writes for training checkpoints.

The machine these models train on has blue-screened mid-run, and each time the files being
written at that moment came back the right size but filled with zeros - the operating system had
recorded the new length before the data reached the disk. A run whose checkpoint is zero-filled
cannot be resumed.

`durable_write` writes through a temporary file next to the target, forces that file to disk, and
only then swaps it in with an atomic rename. A crash at any point leaves either the previous
complete file or the new complete file, never a partial one. The temporary name starts with
".partial-", so globs for package files ("*.zip", "step_*") never pick up a leftover.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Union

PARTIAL_PREFIX = ".partial-"


def durable_write(final: Union[str, Path], write: Callable[[Path], object]) -> Path:
    """Call ``write(tmp_path)`` to produce the file, then make it ``final`` atomically.

    ``final`` must carry its real extension (``.zip``, ``.pkl``, ``.json``): the temporary file
    keeps it, so writers that add a missing extension (Stable-Baselines3's ``save``) leave the
    name alone.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(PARTIAL_PREFIX + final.name)
    write(tmp)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final)
    return final


def durable_text(final: Union[str, Path], text: str) -> Path:
    return durable_write(final, lambda p: Path(p).write_text(text, encoding="utf-8"))
