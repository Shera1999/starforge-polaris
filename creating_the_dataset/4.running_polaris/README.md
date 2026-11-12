# Step 4: running POLARIS on all boxes

`run_polaris.py` runs every box of `../3.preparing_polaris_grids/grids_64` (prepare_grids v3.5.1:
ISRF 1e7 photons at radius 10, stars 1e6 photons, 38 wavelengths x 3 views) at nice 19.
Output: `/scratch/sjafaritabar/polaris_runs/<snapshot>/<core>/` (`em/`, `temp/`, `polaris.cmd`,
`summary.json`, `run_summary.log`, and `DONE.json` or `FAILED.json`). ~37 MB per box, ~70 GB in total.
Boxes with stars run first. Started 2026-10-31 22:47 (a radius-20 start at 22:36 was aborted).

The runner shares the machine by itself: it measures the CPU everyone else uses and runs as many
boxes as fit in the rest (8 CPUs kept free, at most `max_jobs`, at least `min_jobs`). When others need
CPUs, the newest boxes are suspended (frozen, no work lost); 10 min after the demand is gone they
resume. Running boxes are never killed.

    OUT=/scratch/sjafaritabar/polaris_runs
    echo 8 > $OUT/max_jobs          # upper limit on boxes at once (default 16)
    echo 0 > $OUT/min_jobs          # boxes kept going when the machine is busy (default 2)
    touch $OUT/pause                # start no new boxes; rm $OUT/pause to continue
    touch $OUT/stop                 # finish running/suspended boxes, then exit
    tail $OUT/runner.log            # starts, suspends, resumes, per-box time, ETA, 30-min status
    ls $OUT/*/*/DONE.json | wc -l   # boxes done
    ls $OUT/*/*/FAILED.json         # failures (reason and log tail inside)

Restarting the same command resumes (finished boxes skipped; `--retry-failed` reruns failures):

    nohup .../envs/starforge/bin/python run_polaris.py --out $OUT > $OUT/runner.out 2>&1 &

If the runner is killed with `kill -9`, suspended POLARIS jobs stay frozen: `pkill -CONT -f bin/polaris`.
