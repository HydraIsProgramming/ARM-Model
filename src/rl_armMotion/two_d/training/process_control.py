"""
Per-process CPU controls for training on a machine with an unstable CPU.

The training PC (Intel i7-13700K) logs corrected machine-check errors on two P-cores and has
blue-screened under sustained training load. Nothing here touches a system setting; each call
changes only the calling process (and, for affinity and priority, the processes it starts):

* ``set_cpus`` - which logical processors the process may run on (an allow-list), e.g. only the
  low-voltage E-cores.
* ``efficiency_mode`` - Windows "efficiency mode" for this process: EcoQoS power throttling (the
  scheduler prefers efficient cores and runs them at efficient clocks and voltage) and a
  below-normal priority, the same thing Task Manager's "Efficiency mode" applies.

On other operating systems both are no-ops.
"""

from __future__ import annotations

import os
from typing import Iterable, List

ECO_ENV = "ARM_EFFICIENCY_MODE"     # set by run_campaign; train_fischer_session applies it to itself


def parse_cpus(spec: str) -> List[int]:
    """'16-23' or '0-11,16-23' -> sorted logical processor numbers."""
    out = set()
    for part in (spec or "").split(","):
        if part.strip():
            lo, _, hi = part.strip().partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def _kernel32():
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE      # a 64-bit pseudo-handle; the default int truncates it
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    return ctypes, k32


def set_cpus(cpus: Iterable[int]) -> int:
    """Restrict this process (and the processes it starts) to the given logical processors."""
    cpus = sorted(set(int(c) for c in cpus))
    if os.name != "nt" or not cpus:
        return 0
    n = os.cpu_count() or 1
    bad = [c for c in cpus if not 0 <= c < n]
    if bad:
        raise ValueError(f"no logical processor(s) {bad} on this machine (0-{n - 1})")
    mask = sum(1 << c for c in cpus)
    ctypes, k32 = _kernel32()
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
        raise OSError(f"could not set the processor affinity (error {ctypes.get_last_error()})")
    return mask


def efficiency_mode() -> bool:
    """EcoQoS + below-normal priority for this process. Returns True if both were applied."""
    if os.name != "nt":
        return False
    ctypes, k32 = _kernel32()

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    PROCESS_POWER_THROTTLING = 4                         # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    st = PowerThrottlingState(PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
                              PROCESS_POWER_THROTTLING_EXECUTION_SPEED)
    h = k32.GetCurrentProcess()
    eco = bool(k32.SetProcessInformation(h, PROCESS_POWER_THROTTLING, ctypes.byref(st), ctypes.sizeof(st)))
    pri = bool(k32.SetPriorityClass(h, BELOW_NORMAL_PRIORITY_CLASS))
    return eco and pri


def apply_from_env() -> None:
    """Called at the top of a training run: honour the efficiency mode its launcher asked for."""
    if os.environ.get(ECO_ENV) == "1":
        efficiency_mode()
