"""
Watch the CPU while training runs, and back off before it crashes.

    python scripts/hw_watchdog.py                       # alongside run_campaign.py; exits when training ends

The training PC's CPU (i7-13700K) logs corrected machine-check errors (Windows "WHEA-Logger"
events) and has blue-screened under sustained training load and during sleep transitions. This
watchdog changes no system setting; it only acts on the training processes and on itself:

* While any training run is alive it asks Windows not to sleep (SetThreadExecutionState, the call
  a video player makes; the screen may still turn off). Two of the crashes happened going to sleep.
* Every --poll seconds it counts WHEA-Logger events in the System log. If --max-errors or more
  arrive within --window minutes, it PAUSES every training run (suspends the processes - nothing is
  lost, they continue exactly where they stopped) and resumes them once --cooldown minutes pass
  with no new errors.
* If it has had to pause --max-pauses times within three hours, it leaves the runs paused and says
  so: the errors are not settling and the machine needs its BIOS fix before training continues.

Everything it does is logged with a timestamp (default C:/arm-model-runs/hw_watchdog.log).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import psutil

TRAIN_MARK = "train_fischer_session.py"
WHEA_QUERY = "*[System[Provider[@Name='Microsoft-Windows-WHEA-Logger'] and TimeCreated[timediff(@SystemTime) <= {ms}]]]"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--poll", type=float, default=30.0, help="seconds between checks")
    p.add_argument("--max-errors", type=int, default=3, help="WHEA events within --window that trigger a pause")
    p.add_argument("--window", type=float, default=10.0, help="minutes")
    p.add_argument("--cooldown", type=float, default=20.0, help="error-free minutes before resuming")
    p.add_argument("--max-pauses", type=int, default=3, help="pauses within 3 h after which runs stay paused")
    p.add_argument("--grace", type=float, default=10.0,
                   help="minutes to wait for training to appear (and to linger after it ends) before exiting")
    p.add_argument("--log", default="C:/arm-model-runs/hw_watchdog.log")
    p.add_argument("--resume-now", action="store_true", help="resume every paused training run and exit")
    return p.parse_args()


class Log:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def whea_events(minutes: float) -> int:
    """WHEA-Logger events in the System log within the last `minutes`."""
    q = WHEA_QUERY.format(ms=int(minutes * 60_000))
    try:
        out = subprocess.run(["wevtutil", "qe", "System", f"/q:{q}", "/f:xml", "/c:500"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return out.count("<Event ")


def training_processes():
    procs = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            if any(TRAIN_MARK in (a or "") for a in (p.info["cmdline"] or [])):
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return procs


def keep_awake(on: bool) -> None:
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def act(procs, suspend: bool) -> int:
    n = 0
    for p in procs:
        try:
            p.suspend() if suspend else p.resume()
            n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return n


def main():
    a = parse_args()
    log = Log(a.log)
    if a.resume_now:
        log(f"resumed {act(training_processes(), suspend=False)} training run(s) by hand")
        return 0
    log(f"watchdog started: pause at {a.max_errors}+ WHEA errors in {a.window:g} min, resume after "
        f"{a.cooldown:g} error-free min; keeping the PC awake while training runs")
    paused, staying, pauses, gone_since, awake = False, False, [], None, False
    seen_training = False
    start = time.time()
    last_beat = 0.0
    try:
        while True:
            procs = training_processes()
            now = time.time()
            if procs:
                seen_training, gone_since = True, None
                if not awake:
                    keep_awake(True)
                    awake = True
            else:
                gone_since = gone_since or now
                if awake:
                    keep_awake(False)
                    awake = False
                limit = a.grace * 60
                if (seen_training and now - gone_since > limit) or (not seen_training and now - start > limit):
                    log("no training running - watchdog exiting")
                    return 0

            recent = whea_events(a.window)
            if not paused and recent >= a.max_errors and procs:
                n = act(procs, suspend=True)
                paused = True
                pauses = [t for t in pauses if now - t < 3 * 3600] + [now]
                log(f"PAUSED {n} training run(s): {recent} hardware (WHEA) errors in the last {a.window:g} min")
            elif paused and whea_events(a.cooldown) == 0:
                if len(pauses) < a.max_pauses:
                    n = act(procs, suspend=False)
                    paused = False
                    log(f"RESUMED {n} training run(s) after {a.cooldown:g} min without hardware errors")
                elif not staying:
                    staying = True
                    log(f"STAYING PAUSED: {len(pauses)} pauses within 3 h - the hardware errors keep coming back. "
                        f"The BIOS fix is needed before training continues; to resume anyway run "
                        f"python scripts/hw_watchdog.py --resume-now")
            if now - last_beat > 600:
                last_beat = now
                log(f"ok: {len(procs)} training run(s){' (paused)' if paused else ''}, "
                    f"{whea_events(60)} WHEA error(s) in the last hour")
            time.sleep(a.poll)
    finally:
        if paused and not staying:          # never leave runs frozen because the watchdog itself stopped
            act(training_processes(), suspend=False)
        keep_awake(False)


if __name__ == "__main__":
    sys.exit(main())
