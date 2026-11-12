#!/usr/bin/env python3
"""Empirical check of STARFORGE snapshot units, as used by
find_starforge_cores_dendrogram_v5.py.

Each check compares a quantity converted with the header units against an
independent physical expectation (initial conditions, ideal-gas relations,
gravitational constant), so a wrong unit assumption shows up as a mismatch
by a large factor rather than going unnoticed.

Usage: check_snapshot_units.py SNAPSHOT_IC.hdf5 SNAPSHOT_LATE.hdf5
"""
import sys

import h5py
import numpy as np

PC_CM = 3.0856775814913673e18
MSUN_G = 1.98847e33
M_H_G = 1.6735575e-24
K_B = 1.380649e-16
G_CGS = 6.67430e-8
MYR_S = 1e6 * 365.25 * 86400
CHUNK = 4_000_000

# Published initial conditions for M2e5_R30 (Grudic+2022, Guszejnov+2022).
M_CLOUD, R_CLOUD, ALPHA_TURB, MU_MAG = 2e5, 30.0, 2.0, 4.2


def units(h):
    ul, um, uv = (float(h[k]) for k in ("UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS"))
    return dict(ul=ul, um=um, uv=uv, ut=ul / uv, rho=um / ul**3, pc=ul / PC_CM, msun=um / MSUN_G)


def header_checks(path):
    with h5py.File(path, "r") as f:
        h = dict(f["Header"].attrs)
    u = units(h)
    g_code = G_CGS * u["um"] * u["ut"]**2 / u["ul"]**3
    print(f"[header] {path}")
    print(f"  1 code length = {u['pc']:.5f} pc, 1 code mass = {u['msun']:.5f} Msun, "
          f"1 code velocity = {u['uv'] / 1e5:g} km/s, 1 code time = {u['ut'] / MYR_S:.2f} Myr")
    print(f"  G from header units = {g_code:.2f}; header Gravitational_Constant_In_Code_Inits = "
          f"{h['Gravitational_Constant_In_Code_Inits']:.2f}  (ratio {g_code / h['Gravitational_Constant_In_Code_Inits']:.5f})")
    print(f"  Time = {h['Time'] * u['ut'] / MYR_S:.4f} Myr; BoxSize = {h['BoxSize'] * u['pc']:.2f} pc; "
          f"Internal_UnitB_In_Gauss = {h.get('Internal_UnitB_In_Gauss')}")
    return h, u


def ic_checks(path):
    h, u = header_checks(path)
    with h5py.File(path, "r") as f:
        g = f["PartType0"]
        n = len(g["Masses"])
        box = float(h["BoxSize"])
        acc = dict(m=0.0, m_cloud=0.0, mv=np.zeros(3), mv2=0.0, n_cloud=0, n_bg=0)
        b_cloud, b_bg, rho_cloud, t_cloud = [], [], [], []
        for a in range(0, n, CHUNK):
            sl = slice(a, min(a + CHUNK, n))
            x = g["Coordinates"][sl].astype(float) - box / 2
            r = np.linalg.norm(x, axis=1) * u["pc"]
            m = g["Masses"][sl].astype(float) * u["msun"]
            v = g["Velocities"][sl].astype(float) * u["uv"] / 1e5
            B = np.linalg.norm(g["MagneticField"][sl].astype(float), axis=1)
            cloud = r < R_CLOUD
            acc["m"] += m.sum()
            acc["m_cloud"] += m[cloud].sum()
            acc["mv"] += (m[cloud, None] * v[cloud]).sum(0)
            acc["mv2"] += (m[cloud] * (v[cloud]**2).sum(1)).sum()
            sub = slice(None, None, 50)
            b_cloud.append(B[cloud][sub]); b_bg.append(B[~cloud][sub])
            rho_cloud.append(g["Density"][sl].astype(float)[cloud][sub] * u["rho"])
            t_cloud.append(g["Temperature"][sl].astype(float)[cloud][sub])
    vmean = acc["mv"] / acc["m_cloud"]
    sigma3d = np.sqrt(acc["mv2"] / acc["m_cloud"] - (vmean**2).sum())
    g_pc = G_CGS * MSUN_G / PC_CM / 1e10                 # pc (km/s)^2 / Msun
    sigma3d_exp = np.sqrt(3 * ALPHA_TURB * g_pc * M_CLOUD / (5 * R_CLOUD))
    sigma_surf = M_CLOUD * MSUN_G / (np.pi * (R_CLOUD * PC_CM)**2)
    b_exp_gauss = 2 * np.pi * np.sqrt(G_CGS) * sigma_surf / MU_MAG
    rho_exp = M_CLOUD * MSUN_G / (4 / 3 * np.pi * (R_CLOUD * PC_CM)**3)
    b_cloud, b_bg = np.concatenate(b_cloud), np.concatenate(b_bg)
    print("[initial conditions]")
    print(f"  mass inside R={R_CLOUD} pc: {acc['m_cloud']:.4g} Msun (expected {M_CLOUD:.3g}); total {acc['m']:.4g}")
    print(f"  median density in cloud: {np.median(np.concatenate(rho_cloud)):.3e} g/cm^3 "
          f"(uniform-sphere mean {rho_exp:.3e})")
    print(f"  3D velocity dispersion in cloud: {sigma3d:.3f} km/s (alpha={ALPHA_TURB} gives ~{sigma3d_exp:.3f})")
    print(f"  median |B| raw: cloud {np.median(b_cloud):.3e}, background {np.median(b_bg):.3e}")
    print(f"  expected |B| for mu={MU_MAG}: {b_exp_gauss:.3e} G = {b_exp_gauss * 1e-4:.3e} T "
          f"-> raw/T_expected = {np.median(b_cloud) / (b_exp_gauss * 1e-4):.3f}, "
          f"raw/G_expected = {np.median(b_cloud) / b_exp_gauss:.2e}")
    print(f"  median T in cloud: {np.median(np.concatenate(t_cloud)):.2f} K")


def late_checks(path, n_dense_max=400_000):
    h, u = header_checks(path)
    t_now = float(h["Time"])
    with h5py.File(path, "r") as f:
        g = f["PartType0"]
        n = len(g["Masses"])
        keys = ("Density", "Temperature", "InternalEnergy", "SoundSpeed", "Dust_Temperature",
                "NeutralHydrogenAbundance", "MolecularMassFraction", "Masses", "MagneticField",
                "ElectronAbundance")
        out = {k: [] for k in keys}
        out["Z"], out["Y"] = [], []
        for a in range(0, n, CHUNK):
            sl = slice(a, min(a + CHUNK, n))
            rho = g["Density"][sl].astype(float) * u["rho"]
            # Every 200th particle everywhere plus every dense particle.
            keep = (rho > 1e-20) | (np.arange(sl.start, sl.stop) % 200 == 0)
            if not keep.any():
                continue
            for k in keys:
                arr = g[k][sl]
                out[k].append(np.asarray(arr[keep] if arr.ndim == 1 else arr[keep, :], dtype=float))
            met = g["Metallicity"][sl, :2][keep]
            out["Z"].append(met[:, 0]); out["Y"].append(met[:, 1])
        d = {k: np.concatenate(v) for k, v in out.items()}
        sinks = {k: f["PartType5"][k][:] for k in ("Masses", "StellarFormationTime", "ProtoStellarAge")
                 if "PartType5" in f and k in f["PartType5"]}
    rho = d["Density"] * u["rho"]
    nH = rho * (1 - d["Z"] - d["Y"]) / M_H_G
    uint = d["InternalEnergy"] * u["uv"]**2          # erg/g
    cs = d["SoundSpeed"] * u["uv"]                   # cm/s
    gamma_gm1 = cs**2 / uint                         # = gamma (gamma - 1) for an ideal gas
    gamma = 0.5 + np.sqrt(0.25 + gamma_gm1)
    mu = K_B * d["Temperature"] / ((gamma - 1) * uint * M_H_G)
    print("[late snapshot: thermodynamics and chemistry, by n_H bin]")
    print("   n_H [cm^-3]       N   gamma   mu(T,u)  T_gas  T_dust  f_neutral  f_mol  x_e     |B| raw  "
          "nH2 neutral-conv / total-conv")
    for lo, hi in [(1e-2, 1), (1, 1e2), (1e2, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e7), (1e7, 1e12)]:
        m = (nH > lo) & (nH <= hi)
        if m.sum() < 10:
            continue
        ratio = np.median(d["NeutralHydrogenAbundance"][m])
        print(f"  {lo:7.0e}-{hi:7.0e} {m.sum():8d}  {np.median(gamma[m]):.3f}  {np.median(mu[m]):6.3f}  "
              f"{np.median(d['Temperature'][m]):6.1f}  {np.median(d['Dust_Temperature'][m]):6.1f}  "
              f"{ratio:8.3f}  {np.median(d['MolecularMassFraction'][m]):6.3f}  "
              f"{np.median(d['ElectronAbundance'][m]):.1e}  {np.median(np.linalg.norm(d['MagneticField'][m], axis=1)):.2e}  "
              f"{ratio:.3f}")
    print(f"  Metallicity col0 (Z): median {np.median(d['Z']):.4f}; col1 (Y): median {np.median(d['Y']):.4f}; "
          f"X_H median {np.median(1 - d['Z'] - d['Y']):.4f}")
    print(f"  particle mass: median {np.median(d['Masses']) * u['msun']:.3e} Msun, "
          f"min {d['Masses'].min() * u['msun']:.3e}, max {d['Masses'].max() * u['msun']:.3e}")
    if sinks:
        sft = sinks["StellarFormationTime"]
        print(f"[sinks] N={len(sft)}; mass {np.min(sinks['Masses']) * u['msun']:.3g}-"
              f"{np.max(sinks['Masses']) * u['msun']:.3g} Msun; formation time "
              f"{sft.min() * u['ut'] / MYR_S:.3f}-{sft.max() * u['ut'] / MYR_S:.3f} Myr "
              f"(snapshot time {t_now * u['ut'] / MYR_S:.3f} Myr; all <= now: {bool(np.all(sft <= t_now))})")
        if "ProtoStellarAge" in sinks:
            age = sinks["ProtoStellarAge"]
            print(f"  ProtoStellarAge raw {age.min():.4g}-{age.max():.4g}; "
                  f"StellarFormationTime raw {sft.min():.4g}-{sft.max():.4g}; Time raw {t_now:.4g}")


if __name__ == "__main__":
    ic_checks(sys.argv[1])
    print()
    late_checks(sys.argv[2])
