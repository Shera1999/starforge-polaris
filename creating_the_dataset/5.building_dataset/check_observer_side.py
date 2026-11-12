#!/usr/bin/env python3
"""Which end of each view's line of sight faces the observer?

Where the dust is optically thick, I = sum_k S_k dtau_k exp(-tau_k) depends on the side the absorption
is counted from. Per pixel, dtau_k = tau_total * n_k / sum(n) (one dust mixture), S_k = B_nu(T_k) from
POLARIS's own dust temperature. The model ratio I_model / I_POLARIS is compared for both orderings in
pixels with tau > 1: the right side gives a near-constant ratio (small scatter), the wrong one does not.

Usage: check_observer_side.py BOX_DIR [BOX_DIR ...]
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
GRIDS = HERE.parent / "3.preparing_polaris_grids" / "grids_64"
sys.path.insert(0, str(GRIDS.parent))
from polaris_octree_reader import read_octree_grid

H, C, K = 6.62607015e-34, 2.99792458e8, 1.380649e-23
# view -> (los grid axis, image col axis/sign, image row axis/sign), from check_view_geometry.py
VIEWS = {0: (2, (0, 1), (1, 1)), 1: (1, (0, 1), (2, -1)), 2: (0, (2, -1), (1, 1))}


def to_view(cube, v):
    """Return [los, row, col] with los in increasing grid index."""
    los, (ca, cs), (ra, rs) = VIEWS[v]
    a = np.transpose(cube, (los, ra, ca))
    if rs < 0:
        a = a[:, ::-1, :]
    if cs < 0:
        a = a[:, :, ::-1]
    return a


def main():
    for arg in sys.argv[1:]:
        run = Path(arg)
        key = f"{run.parent.name}/{run.name}"
        n = read_octree_grid(GRIDS / key / "grid.dat")["density_number_m3"].astype(float)
        T = read_octree_grid(run / "temp" / "grid_temp_rat.dat")["dust_temperature_K"].astype(float)
        lam_um = [float(l.split(">", 1)[1].split()[0]) * 1e6 for l in (run / "polaris.cmd").read_text().splitlines()
                  if l.strip().startswith("<detector_dust")]
        for v in range(3):
            nv, Tv = to_view(n, v), to_view(T, v)
            for w in range(38):
                lam = lam_um[38 * v + w] * 1e-6
                d = fits.getdata(run / "em" / "data" / f"polaris_detector_nr{38 * v + w + 1:04d}.fits.gz")[:, 0]
                I, tau = d[0].astype(float), d[4].astype(float)
                sel = (tau > 1) & (I > 0)
                if sel.sum() < 20 or lam < 20e-6:
                    continue
                S = 1.0 / np.expm1(H * C / (lam * K * Tv))
                dtau = tau[None] * nv / nv.sum(0)[None]
                out = {}
                for name, sl in (("low index faces observer", slice(None)), ("high index faces observer", slice(None, None, -1))):
                    dt = dtau[sl]
                    tau_front = np.cumsum(dt, 0) - dt            # optical depth between observer and cell
                    Im = (S[sl] * (-np.expm1(-dt)) * np.exp(-tau_front)).sum(0)
                    r = np.log(Im[sel] / I[sel])
                    out[name] = 1.4826 * np.median(np.abs(r - np.median(r)))     # robust: stars add direct light
                best = min(out, key=out.get)
                print(f"{key} view {v} {lam * 1e6:6.1f} um: {sel.sum():4d} px with tau>1 (max {tau.max():.1f}); "
                      f"robust scatter of log ratio: low-faces {out['low index faces observer']:.3f}, "
                      f"high-faces {out['high index faces observer']:.3f} -> {best}")
                break


if __name__ == "__main__":
    main()
