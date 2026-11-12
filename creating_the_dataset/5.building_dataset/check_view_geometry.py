#!/usr/bin/env python3
"""Which grid axes do the three POLARIS views look along, and how do image rows/cols map onto the grid?

For each view (detectors 1-38, 39-76, 77-114) the column-density plane N of the 850 um image is compared
with sum(n) * cell along each grid axis, in all 48 orientations (line-of-sight axis x transpose x flips).
The best orientation must match to ~1e-6. Then the polarization angle is checked against a
density-weighted Stokes model of the plane-of-sky B in that orientation (B angle = PA + 90 deg, with
PA = 0.5 atan2(U, Q) measured from the image +col axis), trying both signs of U.

Usage: check_view_geometry.py BOX_DIR [BOX_DIR ...]   (BOX_DIR = finished run dir with grid in grids_64)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
GRIDS = HERE.parent / "3.preparing_polaris_grids" / "grids_64"
sys.path.insert(0, str(GRIDS.parent))
from polaris_octree_reader import read_octree_grid

VIEWS = {0: "(0, 0)", 1: "(-90, 0)", 2: "(0, 90)"}
DET_850 = 33                                    # 850 um is wavelength 33 of 38 in view 0


def orientations(cube_shape_axes=3):
    """Yield (los_axis, transpose, flip_row, flip_col, label) for an image derived from a [x, y, z] array."""
    for los, tr, fr, fc in itertools.product(range(3), (False, True), (False, True), (False, True)):
        yield los, tr, fr, fc


def project(cube, los, tr, fr, fc):
    """Sum along los -> 2D array [a, b] (remaining axes in order); return as image [row, col]."""
    img = cube.sum(axis=los)                    # axes: remaining grid axes in increasing order
    img = img.T if not tr else img              # default: row = second remaining axis, col = first
    if fr:
        img = img[::-1, :]
    if fc:
        img = img[:, ::-1]
    return img


def image_axes(los, tr, fr, fc):
    """Grid axis and sign that the image +col and +row directions point along."""
    rest = [a for a in range(3) if a != los]
    col_ax, row_ax = (rest[0], rest[1]) if not tr else (rest[1], rest[0])
    return (col_ax, -1 if fc else 1), (row_ax, -1 if fr else 1)


def check_box(run):
    key = f"{run.parent.name}/{run.name}"
    summ = json.loads((GRIDS / key / "summary.json").read_text())
    g = read_octree_grid(GRIDS / key / "grid.dat")
    n = g["density_number_m3"].astype(float)
    B = np.stack([g["Bx"], g["By"], g["Bz"]]).astype(float)
    cell_m = summ["cell_pc"] * 3.0856775814913673e16
    out = []
    for v in range(3):
        d = fits.getdata(run / "em" / "data" / f"polaris_detector_nr{DET_850 + 38 * v:04d}.fits.gz")[:, 0].astype(float)
        I, Q, U, N = d[0], d[1], d[2], d[5]
        best = min(orientations(), key=lambda o: np.abs(project(n, *o) * cell_m / N - 1).max())
        err = np.abs(project(n, *best) * cell_m / N - 1).max()
        (ca, cs), (ra, rs) = image_axes(*best)
        # density-weighted Stokes of the plane-of-sky field in image axes
        bc, br = cs * B[ca], rs * B[ra]
        b2 = (B ** 2).sum(0) + 1e-300
        qm = project(n * (bc ** 2 - br ** 2) / b2, *best)
        um = project(n * 2 * bc * br / b2, *best)
        model_b = 0.5 * np.arctan2(um, qm)
        p = np.hypot(Q, U) / I
        sel = p > np.percentile(p, 50)
        res = {}
        for usign in (1, -1):
            pol_b = 0.5 * np.arctan2(usign * U, Q) + np.pi / 2
            dang = np.degrees(np.abs(np.angle(np.exp(2j * (pol_b - model_b))) / 2))
            res[usign] = np.median(dang[sel])
        names = "xyz"
        out.append(f"{key} view {v} {VIEWS[v]}: line of sight = grid {names[best[0]]}, image +col = "
                   f"{'+' if cs > 0 else '-'}{names[ca]}, +row = {'+' if rs > 0 else '-'}{names[ra]}; "
                   f"N match max rel err {err:.1e}; B angle vs model median {res[1]:.2f} deg (U as is), "
                   f"{res[-1]:.2f} deg (U flipped)")
    return out


def main():
    for arg in sys.argv[1:]:
        for line in check_box(Path(arg)):
            print(line)


if __name__ == "__main__":
    main()
