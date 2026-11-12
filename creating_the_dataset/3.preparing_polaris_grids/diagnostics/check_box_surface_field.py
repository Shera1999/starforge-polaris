#!/usr/bin/env python3
"""STARFORGE's own radiation field on the surface of each box, as a POLARIS G0.

For every box of one snapshot, take the gas in the outer 0.1 pc shell of the
cube, convert its PhotonEnergy into band energy densities, and express them as
the G0 (multiplier of the Mathis 1983 field used by POLARIS) giving the same
dust heating: G0_eff = sum_b w_b * u_b(STARFORGE) / u_b(Mathis), where w_b is
the fraction of Mathis-field heating that astrodust absorbs in band b
(0.522 FUV, 0.199 NUV, 0.269 optical/NIR, computed from POLARIS's own
astrodust cross sections; the ionizing and IR bands contribute ~0).

Usage: check_box_surface_field.py SNAPSHOT_NAME [GRIDS_ROOT]
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from check_isrf_normalisation import mathis_energy_density_erg_cm3

HEATING_WEIGHT = {1: 0.522, 2: 0.199, 3: 0.269}     # band index -> fraction of Mathis heating
SHELL_PC = 0.1


def main(snap, root):
    summaries = [json.loads(p.read_text()) for p in sorted((Path(root) / snap).glob("core_*/summary.json"))]
    path = summaries[0]["snapshot_file"]
    with h5py.File(path, "r") as f:
        h = f["Header"].attrs
        ul, um, uv = (float(h[k]) for k in ("UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS"))
        e_lo, e_hi = h["Radiation_RHD_Min_Bin_Freq_in_eV"], h["Radiation_RHD_Max_Bin_Freq_in_eV"]
        L = ul / 3.0856775814913673e18
        box = float(h["BoxSize"])
        centers = np.array([s["grid_center_pc"] for s in summaries]) / L
        half = summaries[0]["side_pc"] / 2 / L
        inner = half - SHELL_PC / L
        g = f["PartType0"]
        n = len(g["Masses"])
        u_shell = [[] for _ in summaries]
        for a in range(0, n, 4_000_000):
            sl = slice(a, min(a + 4_000_000, n))
            xyz = g["Coordinates"][sl].astype(float)
            hits = []
            for b, c in enumerate(centers):
                d0 = xyz[:, 0] - c[0]; d0 -= box * np.rint(d0 / box)
                cand = np.flatnonzero(np.abs(d0) < half)
                if not len(cand):
                    continue
                rel = xyz[cand] - c; rel -= box * np.rint(rel / box)
                m = np.all(np.abs(rel) < half, axis=1) & (np.max(np.abs(rel), axis=1) > inner)
                if m.any():
                    hits.append((b, cand[m]))
            if not hits:
                continue
            E = g["PhotonEnergy"][sl].astype(float) * um * uv**2
            vol = g["Masses"][sl].astype(float) * um / (g["Density"][sl].astype(float) * um / ul**3)
            for b, idx in hits:
                u_shell[b].append(E[idx] / vol[idx, None])
    u_m = {k: mathis_energy_density_erg_cm3(e_lo[k], e_hi[k]) for k in HEATING_WEIGHT}
    g0 = []
    for s, us in zip(summaries, u_shell):
        u = np.median(np.concatenate(us), axis=0)
        g0.append(sum(w * u[k] / u_m[k] for k, w in HEATING_WEIGHT.items()))
        if s["core_id"] == 395:
            print(f"core 395 (smoke-test box): G0_eff = {g0[-1]:.3f}; band ratios to Mathis "
                  + ", ".join(f"{e_lo[k]:g}-{e_hi[k]:g} eV {u[k] / u_m[k]:.2f}" for k in HEATING_WEIGHT))
    g0 = np.array(g0)
    stars = np.array([s["n_stellar_sources"] for s in summaries])
    print(f"{snap}: {len(g0)} boxes, surface G0_eff median {np.median(g0):.3f}, 5-95% {np.percentile(g0, 5):.3f}-"
          f"{np.percentile(g0, 95):.3f}, min {g0.min():.3f}, max {g0.max():.3f}; boxes with stars inside: {(stars > 0).sum()}")
    np.savetxt(f"box_surface_G0_{snap}.txt", np.column_stack([[s['core_id'] for s in summaries], g0, stars]),
               header="core_id G0_eff n_stars_in_box", fmt=["%d", "%.4f", "%d"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[1] / "grids_64"))
