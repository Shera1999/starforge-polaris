#!/usr/bin/env python3
"""Tests for select_boxes.py: synthetic cases plus invariants on the real output.

Run: python test_select_boxes.py   (exits non-zero on any failure)
"""
import itertools
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_boxes import POS, greedy_select, periodic_delta

OUT = HERE / "boxes_2.2pc"
OLD_JOBS = HERE / "legacy_reference" / "polaris_full_batch_jobs.txt"   # copy of the pre-v5.2 pipeline output
STEP1 = HERE.parent / "1.finding_cores" / "dendrogram_cores_v5.2"
S = 2.2
BOX = 300.0 * 3.09e18 / 3.0856775814913673e18


def cat_from(points, masses, peaks=None):
    p = np.asarray(points, float)
    peaks = np.ones(len(p)) if peaks is None else peaks
    return pd.DataFrame(dict(core_id=np.arange(len(p)), peak_x_pc=p[:, 0], peak_y_pc=p[:, 1],
                             peak_z_pc=p[:, 2], mass_selected_gas_Msun=masses, peak_nH2_cm3=peaks))


def brute_greedy(cat, side, box):
    pos = cat[POS].to_numpy(float)
    order = np.lexsort((cat.core_id.to_numpy(), -cat.peak_nH2_cm3.to_numpy(),
                        -cat.mass_selected_gas_Msun.to_numpy()))
    kept = []
    for i in order:
        if all(np.max(np.abs(periodic_delta(pos[i], pos[k], box))) >= side for k in kept):
            kept.append(i)
    return kept


def test_overlap_definition_is_strict_chebyshev():
    # Diagonal offset 2.0 on every axis: Euclidean 3.46 > 2.2, but the cubes overlap.
    kept, _ = greedy_select(cat_from([[10, 10, 10], [12, 12, 12]], [2, 1]), S, BOX)
    assert list(kept) == [0]
    # Just beyond S on one axis: separate cubes -> both kept; just inside -> overlap.
    kept, _ = greedy_select(cat_from([[10, 10, 10], [10 + S * (1 + 1e-9), 10, 10]], [2, 1]), S, BOX)
    assert sorted(kept) == [0, 1]
    kept, _ = greedy_select(cat_from([[10, 10, 10], [10 + S * (1 - 1e-9), 10, 10]], [2, 1]), S, BOX)
    assert list(kept) == [0]


def test_periodic_wrap():
    kept, sup = greedy_select(cat_from([[0.5, 50, 50], [BOX - 0.5, 50, 50]], [1, 2]), S, BOX)
    assert list(kept) == [1] and sup[0] == 1


def test_matches_bruteforce_on_random_clusters():
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = rng.integers(1, 200)
        centres = rng.uniform(0, BOX, (5, 3))
        pts = np.mod(centres[rng.integers(0, 5, n)] + rng.normal(0, 3, (n, 3)), BOX)
        cat = cat_from(pts, rng.lognormal(0, 1, n))
        kept, _ = greedy_select(cat, S, BOX)
        assert list(kept) == brute_greedy(cat, S, BOX), f"trial {trial}"


def test_mass_ties_go_to_higher_peak_then_lower_id():
    cat = cat_from([[10, 10, 10], [11, 10, 10]], [1.0, 1.0], peaks=[1e4, 5e4])
    assert list(greedy_select(cat, S, BOX)[0]) == [1]
    cat = cat_from([[10, 10, 10], [11, 10, 10]], [1.0, 1.0])
    assert list(greedy_select(cat, S, BOX)[0]) == [0]
    assert list(greedy_select(cat.iloc[::-1].reset_index(drop=True), S, BOX)[0]) == [1]  # core_id 0 again


def test_same_selection_as_old_pipeline_except_mass_ties():
    """Compare by physical core (peak particle), since v5.2 renumbered core_ids in
    snapshot_500. Any box that differs must be a mass tie the old unstable sort broke."""
    new = pd.read_csv(OUT / "boxes.csv")
    old_jobs = [l.split() for l in OLD_JOBS.read_text().splitlines() if l.strip()]
    n_diff = 0
    for snap, g in new.groupby("snapshot"):
        old_cat = pd.read_csv(OLD_JOBS.parent / "dendrogram_cores_v5" / snap / "core_catalog.csv").set_index("core_id")
        old_peaks = set(old_cat.loc[[int(c) for s, c in old_jobs if s == snap], "peak_particle_id"])
        only_new = g[~g.peak_particle_id.isin(old_peaks)]
        only_old = old_cat[old_cat.peak_particle_id.isin(old_peaks - set(g.peak_particle_id))]
        assert len(only_new) == len(only_old), snap
        for _, r in only_new.iterrows():
            d = np.max(np.abs(periodic_delta(only_old[POS].to_numpy(float), r[POS].to_numpy(float), BOX)), axis=1)
            tie = (d < S) & np.isclose(only_old.mass_selected_gas_Msun.to_numpy(), r.mass_selected_gas_Msun, rtol=1e-12)
            assert tie.any(), f"{snap} core {r.core_id}: differs from old without a mass tie"
            n_diff += 1
    print(f"    {n_diff} box(es) differ from the old selection, all resolved mass ties")


def test_real_boxes_never_overlap_and_invariants_hold():
    boxes = pd.read_csv(OUT / "boxes.csv")
    sup = pd.read_csv(OUT / "suppressed_cores.csv")
    for snap, g in boxes.groupby("snapshot"):
        p = g[POS].to_numpy(float)
        for a, b in itertools.combinations(range(len(p)), 2):
            assert np.max(np.abs(periodic_delta(p[a], p[b], BOX))) >= S, snap
        cat = pd.read_csv(STEP1 / snap / "core_catalog.csv")
        assert sorted(g.core_id) == sorted(cat.core_id.iloc[brute_greedy(cat, S, BOX)])
        s = sup[sup.snapshot == snap]
        assert len(s) + len(g) == len(cat)                       # every core kept or suppressed once
        assert (s.chebyshev_separation_pc < S).all()             # suppressed only by an overlapping box
        assert (s.kept_mass_Msun >= s.dropped_mass_Msun).all()   # ... of a core at least as massive
        assert s.loc[s.chebyshev_separation_pc < S / 2, "inside_kept_box"].all()


def test_families_link_only_overlapping_consecutive_boxes():
    b = pd.read_csv(OUT / "boxes.csv")
    snaps = sorted(b.snapshot_number.unique())
    for x, y in zip(snaps[:-1], snaps[1:]):
        A, B = b[b.snapshot_number == x], b[b.snapshot_number == y]
        d = np.max(np.abs(periodic_delta(A[POS].to_numpy()[:, None], B[POS].to_numpy()[None], BOX)), axis=2)
        same_family = A.family_id.to_numpy()[:, None] == B.family_id.to_numpy()[None]
        assert not np.any((d < S) & ~same_family)   # every overlapping pair is linked


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"{'ALL PASSED' if not fails else f'{fails} FAILED'}")
    sys.exit(bool(fails))
