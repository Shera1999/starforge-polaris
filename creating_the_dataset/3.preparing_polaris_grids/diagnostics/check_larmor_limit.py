#!/usr/bin/env python3
"""Where would the Larmor limit stop RAT alignment in our grids, per catalogue?

Experimental POLARIS (commit 6c5e1a8) aligns a grain of size a in a cell only if
a_alig < a < a_larm, with (CDustComponent::get_alarm, SI units)

    a_larm = 1.33e21 * 293 * chi * B * s^2 / (rho_mat * n_g * T_d * sqrt(T_g))

where chi is the 8th header number of the dust catalogue, s the aspect ratio
(0.5), rho_mat the material density, n_g the gas number density [m^-3] and B
in Tesla. This evaluates it on every cell of a sample of grids using the
STARFORGE dust temperature stored in the grid (POLARIS's own T_d from
CMD_TEMP_RAT is within ~10%; a_larm ~ 1/T_d), for:

    astrodust_fe_0.0_sus  chi = 4.47e-4, rho 2898  (our choice)
    astrodust_fe_0.1_sus  chi = 0.1134,  rho 3141  (after fixing its header)
    astrodust_fe_0.1      chi = 1 (old 'align' flag read as chi), rho 3141
    astrodust_fe_0.0      chi = 0 -> a_larm = 0, nothing ever aligns

Usage: check_larmor_limit.py [GRIDS_ROOT] [STRIDE]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polaris_octree_reader import read_octree_grid

CATALOGUES = {"fe_0.0_sus": (4.47e-4, 2898.0), "fe_0.1_sus": (0.1134, 3141.0), "fe_0.1 (chi=1)": (1.0, 3141.0)}
ASPECT = 0.5
SIZES_NM = (100, 250, 500)
MU = 2.33


def a_larm_m(chi, rho_mat, B, n, Td, Tg):
    a = 1.33e21 * 293 * chi * B * ASPECT**2 / (rho_mat * n * Td * np.sqrt(Tg))
    return np.minimum(a, 1.0)


def one(path):
    g = read_octree_grid(path)
    n = g["density_number_m3"].astype(float).ravel()
    B = np.sqrt(sum(g[k].astype(float) ** 2 for k in ("Bx", "By", "Bz"))).ravel()
    Td, Tg = g["dust_temperature_K"].astype(float).ravel(), g["gas_temperature_K"].astype(float).ravel()
    out = {"mass": n.sum(), "n_peak": n.max()}
    for name, (chi, rho) in CATALOGUES.items():
        a = a_larm_m(chi, rho, B, n, Td, Tg)
        out[name] = {s: (n[a < s * 1e-9].sum(), (a < s * 1e-9).sum()) for s in SIZES_NM}
        out[name]["min"] = a.min()
    return out


def main(root, stride):
    paths = sorted(Path(root).glob("snapshot_*/core_*/grid.dat"))[::stride]
    with Pool(32) as p:
        res = p.map(one, paths, chunksize=2)
    m_tot = sum(r["mass"] for r in res)
    n_cells = len(res) * 64**3
    print(f"{len(res)} grids (every {stride}th of {root}), {n_cells:,} cells; "
          "T_d from STARFORGE, s = 0.5, mu = 2.33")
    print(f"{'catalogue':>16} {'chi':>9}  " + "  ".join(f"a_larm<{s}nm: cells / mass / boxes" for s in SIZES_NM)
          + "   min a_larm")
    for name, (chi, _) in CATALOGUES.items():
        cols = []
        for s in SIZES_NM:
            mass = sum(r[name][s][0] for r in res)
            cells = sum(r[name][s][1] for r in res)
            boxes = sum(r[name][s][1] > 0 for r in res)
            cols.append(f"{cells / n_cells:10.2e} / {mass / m_tot:8.2e} / {boxes:4d}")
        print(f"{name:>16} {chi:9.3g}  " + "  ".join(cols)
              + f"   {min(r[name]['min'] for r in res) * 1e9:.3g} nm")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "grids_64"), int(sys.argv[2]) if len(sys.argv) > 2 else 10)
