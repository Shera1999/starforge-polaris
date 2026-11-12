#!/usr/bin/env python3
"""How STARFORGE stores and varies dust: DustToGasRatio_Local over a whole snapshot.

GIZMO writes DustToGasRatio_Local = Z * R, where Z = Metallicity[:, 0] and R is
the local dust-to-metals ratio relative to solar (GIZMO user guide). STARFORGE
uses an absolute dust-to-gas ratio f_d = 0.01 Z / Z_sun (Grudic+2021, sec. 3.2),
so f_d = 0.01 * DustToGasRatio_Local / Z_sun with Z_sun = Solar_Abundances_Adopted[0].

This scans every gas particle and reports Z, R and D, and where R departs from
1: against dust and gas temperature, density and distance to the nearest sink.

Usage: check_dust_to_gas.py SNAPSHOT.hdf5 [OUT.npz]
"""
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

CHUNK = 4_000_000
KEEP_MAX = 3_000_000          # deviating particles stored for the detailed look


def main(path, out):
    with h5py.File(path, "r") as f:
        h = f["Header"].attrs
        ul, um, uv = (float(h[k]) for k in ("UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS"))
        zsun = float(h["Solar_Abundances_Adopted"][0])
        box = float(h["BoxSize"])
        t_myr = float(h["Time"]) * ul / uv / 3.15576e13
        pc = ul / 3.0856775814913673e18
        g = f["PartType0"]
        n = len(g["Masses"])
        sinks = f["PartType5"]["Coordinates"][:] if "PartType5" in f else np.empty((0, 3))
        tree = cKDTree(np.mod(sinks, box), boxsize=box) if len(sinks) else None
        sd = sz = None
        if "PartType5" in f and len(f["PartType5"]["Masses"]):
            s = f["PartType5"]
            sd, sz = s["DustToGasRatio_Local"][:].astype(float), s["Metallicity"][:, 0].astype(float)
        R_edges = np.r_[-np.inf, np.linspace(-6, 1, 141), np.inf]
        hist = np.zeros(len(R_edges) - 1, np.int64)
        z_min, z_max, z_sum, m_sum = np.inf, -np.inf, 0.0, 0.0
        n_exact = n_low = n_high = 0
        mass_low = mass_tot = 0.0
        keep = {k: [] for k in ("R", "Td", "Tg", "nH", "dsink_pc", "mass")}
        kept = 0
        for a in range(0, n, CHUNK):
            sl = slice(a, min(a + CHUNK, n))
            D = g["DustToGasRatio_Local"][sl].astype(float)
            Z = g["Metallicity"][sl, 0].astype(float)
            m = g["Masses"][sl].astype(float) * um / 1.98847e33
            R = np.divide(D, Z, out=np.full_like(D, np.nan), where=Z > 0)
            z_min, z_max = min(z_min, Z.min()), max(z_max, Z.max())
            z_sum += np.sum(m * Z); m_sum += m.sum()
            hist += np.histogram(np.log10(np.clip(R, 1e-30, None)), R_edges)[0]
            exact = np.abs(R - 1) < 1e-5
            low, high = R < 1 - 1e-5, R > 1 + 1e-5
            n_exact += exact.sum(); n_low += low.sum(); n_high += high.sum()
            mass_low += m[low].sum(); mass_tot += m.sum()
            dev = np.flatnonzero(~exact)
            if len(dev) and kept < KEEP_MAX:
                dev = dev[:KEEP_MAX - kept]
                rho = g["Density"][sl].astype(float)[dev] * um / ul**3
                keep["R"].append(R[dev]); keep["mass"].append(m[dev])
                keep["Td"].append(g["Dust_Temperature"][sl].astype(float)[dev])
                keep["Tg"].append(g["Temperature"][sl].astype(float)[dev])
                keep["nH"].append(rho * (1 - Z[dev] - g["Metallicity"][sl, 1].astype(float)[dev]) / 1.6735575e-24)
                if tree is not None:
                    xyz = np.mod(g["Coordinates"][sl][dev], box)
                    keep["dsink_pc"].append(tree.query(xyz)[0] * pc)
                else:
                    keep["dsink_pc"].append(np.full(len(dev), np.inf))
                kept += len(dev)
    print(f"{path}: t = {t_myr:.3f} Myr, {n:,} gas particles, {len(sinks)} sinks, Z_sun(header) = {zsun}")
    print(f"  Z: min {z_min:.6g}, max {z_max:.6g}, mass-weighted mean {z_sum / m_sum:.6g}  (Z/Z_sun = {z_sum / m_sum / zsun:.4f})")
    print(f"  R = D/Z: exactly 1 for {n_exact / n:.4%}; below 1 for {n_low:,} ({n_low / n:.4%}, "
          f"mass {mass_low:.3f} of {mass_tot:.1f} Msun); above 1 for {n_high:,}")
    if sd is not None:
        print(f"  sinks: D/Z min {np.min(sd / sz):.4f} max {np.max(sd / sz):.4f}")
    d = {k: np.concatenate(v) if v else np.empty(0) for k, v in keep.items()}
    if len(d["R"]):
        print("  particles with R < 1, binned by dust temperature:")
        for lo, hi in ((0, 100), (100, 500), (500, 1000), (1000, 1400), (1400, 1600), (1600, 1e9)):
            sel = (d["Td"] >= lo) & (d["Td"] < hi) & (d["R"] < 1)
            if sel.any():
                print(f"    T_dust {lo:>5g}-{hi:<6g} K: N={sel.sum():9,}  median R {np.median(d['R'][sel]):.3g}  "
                      f"median T_gas {np.median(d['Tg'][sel]):9.3g} K  median n_H {np.median(d['nH'][sel]):9.3g}  "
                      f"median dist to sink {np.median(d['dsink_pc'][sel]):.3g} pc")
        print("  particles with R < 1, binned by gas temperature:")
        for lo, hi in ((0, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e12)):
            sel = (d["Tg"] >= lo) & (d["Tg"] < hi) & (d["R"] < 1)
            if sel.any():
                print(f"    T_gas {lo:8.0e}-{hi:<8.0e} K: N={sel.sum():9,}  median R {np.median(d['R'][sel]):.3g}  "
                      f"median T_dust {np.median(d['Td'][sel]):8.3g} K  median dist to sink {np.median(d['dsink_pc'][sel]):.3g} pc")
    np.savez_compressed(out, hist=hist, R_edges=R_edges, **d)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dust_to_gas_scan.npz")
