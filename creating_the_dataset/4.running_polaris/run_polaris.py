#!/usr/bin/env python3
"""Run POLARIS (temperature + RAT alignment, then dust emission) on every box of grids_64.

Each box runs in its own directory under OUT (default /scratch, /home is full): grid.dat, polaris.cmd and
summary.json are copied there, POLARIS runs at nice 19 with its stdout filtered (it prints one
"Wrong cell border" line per escaping photon, GBs per box), and the result is validated:
114 detector images, the temperature grid, finite I/Q/U, I > 0 and nonzero Q/U at >= 50 um,
plausible dust temperatures. A box ends with DONE.json or FAILED.json (reason, log tail).

Sharing the machine: every 15 s the runner measures the CPU used by everything else on the machine (all
processes outside its own jobs). It runs as many boxes as fit in the CPUs nobody else wants, keeping
HEADROOM_CPUS free: target = (CPUs - others' peak use over the last 10 min - headroom) / threads per box,
clamped to [min_jobs, max_jobs]. When others need CPUs the newest jobs are suspended (SIGSTOP: frozen in
place, no work lost, CPU freed at once); once the demand has been gone for 10 min they resume, then new
boxes start. Running jobs are never killed. Low free memory also blocks starting or resuming boxes.

Control while running (files in OUT, read every 15 s):
  max_jobs   most boxes at once (default 16 = all 128 CPUs at 8 threads); `echo 8 > OUT/max_jobs`
  min_jobs   boxes kept running however busy the machine is (default 2; 0 = give everything back)
  pause      if present, no new boxes are started (running ones continue)
  stop       if present, the runner exits once the running (and suspended) boxes are done
Rerunning the script resumes: boxes with DONE.json are skipped (FAILED.json too, unless --retry-failed).
Boxes with stars go first (most stars first), then the rest, so problems with stars show up early.

Usage: run_polaris.py [--out DIR] [--max-jobs 16] [--limit N] [--only snapshot_500/core_000873 ...]
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
GRIDS = HERE.parent / "3.preparing_polaris_grids" / "grids_64"
sys.path.insert(0, str(GRIDS.parent))
from polaris_octree_reader import read_octree_grid

POLARIS = "/home/ilion/0/sjafaritabar/STARFORGE/M2e5_R30_Res585/POLARIS_experimental/bin/polaris"
N_DETECTORS = 114
MIN_FREE_GB = 20                    # stop starting boxes below this on OUT or on / (POLARIS temp files)
HEADROOM_CPUS = 8                   # logical CPUs always left for others on top of what they use
WINDOW_S = 600                      # others' demand = their peak over this window (quick down, slow up)
MIN_AVAILABLE_GB = 30               # no starting/resuming boxes below this much available memory
POLL_S = 15
FILTER = ("tr '\\r' '\\n' | grep -v -e 'Wrong cell border' -e '\\[%\\]' | awk '!seen[$0]++'")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(out, msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(out / "runner.log", "a") as f:
        f.write(line + "\n")


def orphans(out):
    """POLARIS processes still working inside OUT (e.g. left behind by a killed runner)."""
    pids = []
    for proc in Path("/proc").iterdir():
        try:
            if proc.name.isdigit() and Path(os.readlink(proc / "cwd")).is_relative_to(out) \
                    and "polaris" in (proc / "cmdline").read_bytes().decode(errors="replace"):
                pids.append(int(proc.name))
        except OSError:
            continue
    return pids


def boxes(only):
    found = []
    for s in sorted(GRIDS.glob("snapshot_*/core_*/summary.json")):
        key = f"{s.parent.parent.name}/{s.parent.name}"
        if only and key not in only:
            continue
        d = json.loads(s.read_text())
        found.append((key, d["n_stellar_sources"]))
    return [k for k, _ in sorted(found, key=lambda b: (-b[1], b[0]))]


def wavelengths_um(cmd_text):
    return [float(l.split(">", 1)[1].split()[0]) * 1e6 for l in cmd_text.splitlines()
            if l.strip().startswith("<detector_dust")]


def validate(run):
    """Return None if the box output is complete and sane, else the reason."""
    tgrid = run / "temp" / "grid_temp_rat.dat"
    if not tgrid.exists():
        return "no temp/grid_temp_rat.dat"
    T = read_octree_grid(tgrid)["dust_temperature_K"]
    if not np.all(np.isfinite(T)) or T.min() < 2.0 or T.max() > 3000.0:
        return f"dust temperature out of range ({np.nanmin(T):.3g}-{np.nanmax(T):.3g} K)"
    lam = wavelengths_um((run / "polaris.cmd").read_text())          # one entry per detector (3 views)
    if len(lam) != N_DETECTORS:
        return f"{len(lam)} detectors in polaris.cmd"
    for k in range(1, N_DETECTORS + 1):
        f = run / "em" / "data" / f"polaris_detector_nr{k:04d}.fits.gz"
        if not f.exists():
            return f"missing {f.name}"
        iqu = fits.getdata(f)[:3, 0].astype(float)
        if not np.all(np.isfinite(iqu)):
            return f"non-finite I/Q/U in {f.name}"
        if lam[k - 1] >= 50 and (iqu[0].min() <= 0 or not np.any(iqu[1:])):
            return f"I <= 0 or Q = U = 0 in {f.name}"
    return None


def start(key, out):
    src, run = GRIDS / key, out / key
    if run.exists():
        shutil.rmtree(run)                                     # leftover of an interrupted run
    run.mkdir(parents=True)
    for name in ("grid.dat", "polaris.cmd", "summary.json"):
        shutil.copy2(src / name, run / name)
    (run / "temp").mkdir(); (run / "em").mkdir()
    sh = f"nice -n 19 {POLARIS} polaris.cmd 2>&1 | {FILTER} > run_summary.log; exit ${{PIPESTATUS[0]}}"
    p = subprocess.Popen(["bash", "-c", sh], cwd=run, stdin=subprocess.DEVNULL, start_new_session=True)
    return dict(proc=p, t0=time.time(), started=now(), suspended_at=None, suspended_s=0.0)


def finish(key, job, out):
    run = out / key
    code, secs = job["proc"].returncode, time.time() - job["t0"] - job["suspended_s"]
    try:
        reason = f"POLARIS exit code {code}" if code else validate(run)
    except Exception as e:                                     # a broken output must not stop the runner
        reason = f"validation error: {type(e).__name__}: {e}"
    rec = dict(box=key, started_utc=job["started"], finished_utc=now(), seconds=round(secs, 1),
               suspended_seconds=round(job["suspended_s"], 1), exit_code=code,
               host=os.uname().nodename, version=json.loads((run / "summary.json").read_text()).get("version"))
    if reason:
        tail = (run / "run_summary.log").read_text(errors="replace").splitlines()[-30:]
        (run / "FAILED.json").write_text(json.dumps(dict(rec, reason=reason, log_tail=tail), indent=2) + "\n")
    else:
        (run / "grid.dat").unlink()                            # the original stays in grids_64
        (run / "DONE.json").write_text(json.dumps(rec, indent=2) + "\n")
    return reason, secs


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def available_mem_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return float("inf")


class OthersCPU:
    """Logical CPUs used by all processes outside our jobs' process groups, from /proc CPU-time deltas."""

    def __init__(self):
        self.prev, self.t, self.history = {}, None, []

    def sample(self, own_pgids):
        cur, t = {}, time.time()
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                f = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if int(f[2]) not in own_pgids:                     # f[2] = process group
                cur[proc.name] = int(f[11]) + int(f[12])        # utime + stime of all threads, clock ticks
        used = None
        if self.t is not None:
            ticks = sum(v - self.prev[k] for k, v in cur.items() if k in self.prev and v >= self.prev[k])
            used = ticks / os.sysconf("SC_CLK_TCK") / (t - self.t)
            self.history.append((t, used))
            self.history = [(ht, u) for ht, u in self.history if t - ht <= WINDOW_S]
        self.prev, self.t = cur, t
        return used, max((u for _, u in self.history), default=None)


def read_int(path, default):
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return default


def signal_job(job, sig):
    try:
        os.killpg(job["proc"].pid, sig)                         # bash, polaris and the log filter
    except ProcessLookupError:
        pass


def main():
    global GRIDS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/scratch/sjafaritabar/polaris_runs"))
    ap.add_argument("--max-jobs", type=int, default=16, help="initial value of OUT/max_jobs")
    ap.add_argument("--min-jobs", type=int, default=2, help="initial value of OUT/min_jobs")
    ap.add_argument("--threads", type=int, default=8, help="POLARIS threads per box (as in polaris.cmd)")
    ap.add_argument("--limit", type=int, help="run at most this many boxes")
    ap.add_argument("--only", nargs="*", help="box keys like snapshot_500/core_000873")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--grids", type=Path, default=GRIDS, help="grid folder (for tests)")
    args = ap.parse_args()
    GRIDS, out = args.grids, args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if pids := orphans(out):
        sys.exit(f"POLARIS still running in {out} (pids {pids}); kill them or wait before restarting")
    (out / "max_jobs").write_text(f"{args.max_jobs}\n")
    (out / "min_jobs").write_text(f"{args.min_jobs}\n")
    for f in ("pause", "stop"):
        (out / f).unlink(missing_ok=True)

    todo = []
    for key in boxes(set(args.only or [])):
        if (out / key / "DONE.json").exists():
            continue
        if (out / key / "FAILED.json").exists() and not args.retry_failed:
            continue
        todo.append(key)
    todo = todo[:args.limit] if args.limit else todo
    log(out, f"runner pid {os.getpid()}: {len(todo)} boxes to run, output {out}")

    running = {}

    def resume_all_and_exit(signum, frame):                    # never leave frozen jobs behind
        for job in running.values():
            signal_job(job, signal.SIGCONT)
        log(out, f"runner got signal {signum}: resumed suspended jobs, exiting (POLARIS jobs keep running)")
        sys.exit(1)
    signal.signal(signal.SIGTERM, resume_all_and_exit)
    signal.signal(signal.SIGINT, resume_all_and_exit)

    ncpu, others = os.cpu_count(), OthersCPU()
    n_done, n_failed, busy_s, last_target, last_status = 0, 0, 0.0, None, 0.0
    while todo or running:
        for key, job in list(running.items()):
            if job["proc"].poll() is not None:
                del running[key]
                reason, secs = finish(key, job, out)
                busy_s += secs
                if reason:
                    n_failed += 1
                    log(out, f"FAILED {key} after {secs / 60:.1f} min: {reason}")
                else:
                    n_done += 1
                    left = len(todo) + len(running)
                    eta_h = busy_s / (n_done + n_failed) * left / max(last_target or 1, 1) / 3600
                    log(out, f"done {key} {secs / 60:.1f} min | {n_done} ok, {n_failed} failed, {left} left, "
                             f"~{eta_h:.0f} h at {last_target} boxes")

        max_jobs, min_jobs = read_int(out / "max_jobs", args.max_jobs), read_int(out / "min_jobs", args.min_jobs)
        used_now, demand = others.sample({job["proc"].pid for job in running.values()})
        if demand is None:                                     # first sample: no rate yet
            time.sleep(1)
            continue
        target = max(min_jobs, min(max_jobs, int((ncpu - demand - HEADROOM_CPUS) // args.threads)))
        if target != last_target:
            log(out, f"target {target} boxes (others use {used_now:.0f} CPUs now, peak {demand:.0f} in 10 min)")
            last_target = target
        if (out / "stop").exists():
            todo = []

        active = sorted((k for k, j in running.items() if j["suspended_at"] is None), key=lambda k: running[k]["t0"])
        frozen = sorted((k for k, j in running.items() if j["suspended_at"] is not None), key=lambda k: running[k]["t0"])
        while len(active) > target:                            # free CPUs: freeze the newest jobs
            key = active.pop()
            signal_job(running[key], signal.SIGSTOP)
            running[key]["suspended_at"] = time.time()
            log(out, f"suspend {key} ({len(active)} active, {len(frozen) + 1} suspended)")
            frozen.insert(0, key)
        roomy = available_mem_gb() > MIN_AVAILABLE_GB
        while len(active) < target and frozen and roomy:      # oldest frozen first: they finish soonest
            key = frozen.pop(0)
            job = running[key]
            signal_job(job, signal.SIGCONT)
            job["suspended_s"] += time.time() - job["suspended_at"]
            job["suspended_at"] = None
            active.append(key)
            log(out, f"resume {key} ({len(active)} active, {len(frozen)} suspended)")
        low = min(free_gb(out), free_gb("/"))
        while len(active) < target and todo and roomy and not (out / "pause").exists():
            if low < MIN_FREE_GB:
                log(out, f"only {low:.0f} GB free, not starting new boxes")
                break
            key = todo.pop(0)
            running[key] = start(key, out)
            active.append(key)
            log(out, f"start {key} ({len(active)} active, target {target})")
        if time.time() - last_status > 1800:
            log(out, f"status: {len(active)} active, {len(frozen)} suspended, {n_done} ok, {n_failed} failed, "
                     f"{len(todo)} queued; others use {used_now:.0f} CPUs")
            last_status = time.time()
        time.sleep(POLL_S)
    log(out, f"finished: {n_done} ok, {n_failed} failed" + (" (stopped)" if (out / "stop").exists() else ""))


if __name__ == "__main__":
    main()
