#!/usr/bin/env python3
"""What mean molecular weight does the gas inside our boxes actually have?

For every gas particle inside any box of a snapshot this computes three
versions of mu from STARFORGE's own fields:

  mu_particle  mass per free particle (H2, HI, H+, e-, He) from the chemistry:
               MolecularMassFraction (H2 share of H), NeutralHydrogenAbundance,
               ElectronAbundance (per H), X = 1 - Z - Y.
  mu_gizmo     the mu GIZMO itself used to turn internal energy into temperature,
               T = (gamma - 1) u mu m_p / k  ->  mu = k T / ((gamma - 1) u m_p).
  mu_drag      the single mu that makes POLARIS's RAT gas-drag term right.
               POLARIS uses rho * v_th(mu) with v_th = sqrt(2 k T / (mu m_u)), so the
               drag ~ rho / sqrt(mu); the true drag sums the colliders,
               sum_i n_i m_i v_th,i ~ sum_i n_i sqrt(m_i), hence
               1 / sqrt(mu_drag) = sum_i n_i sqrt(m_i) / sum_i n_i m_i  (H2, HI, H+, He).

Usage: check_mu.py SNAPSHOT_NAME [SNAPSHOT_NAME ...]
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

GRIDS = Path(__file__).resolve().parents[1] / "grids_64"
K_B, M_P, M_H = 1.380649e-16, 1.67262192e-24, 1.6735575e-24
GAMMA = 5.0 / 3.0


def box_gas(snap):
    summaries = [json.loads(p.read_text()) for p in sorted((GRIDS / snap).glob("core_*/summary.json"))]
    with h5py.File(summaries[0]["snapshot_file"], "r") as f:
        h = f["Header"].attrs
        ul, um, uv = (float(h[k]) for k in ("UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS"))
        L = ul / 3.0856775814913673e18
        box = float(h["BoxSize"])
        centers = np.array([s["grid_center_pc"] for s in summaries]) / L
        half = summaries[0]["side_pc"] / 2 / L
        g = f["PartType0"]
        n = len(g["Masses"])
        keys = ("Masses", "Density", "Temperature", "InternalEnergy", "MolecularMassFraction",
                "NeutralHydrogenAbundance", "ElectronAbundance")
        out = {k: [] for k in keys + ("Y", "Z")}
        for a in range(0, n, 4_000_000):
            sl = slice(a, min(a + 4_000_000, n))
            xyz = g["Coordinates"][sl].astype(float)
            inside = np.zeros(len(xyz), bool)
            for c in centers:
                d0 = xyz[:, 0] - c[0]; d0 -= box * np.rint(d0 / box)
                cand = np.flatnonzero(np.abs(d0) < half)
                if len(cand):
                    rel = xyz[cand] - c; rel -= box * np.rint(rel / box)
                    inside[cand[np.all(np.abs(rel) < half, axis=1)]] = True
            idx = np.flatnonzero(inside)
            if not len(idx):
                continue
            for k in keys:
                out[k].append(g[k][sl][idx].astype(float))
            met = g["Metallicity"][sl][idx]
            out["Z"].append(met[:, 0].astype(float)); out["Y"].append(met[:, 1].astype(float))
    d = {k: np.concatenate(v) for k, v in out.items()}
    d["rho_cgs"] = d["Density"] * um / ul**3
    d["u_cgs"] = d["InternalEnergy"] * uv**2
    return d, len(summaries)


def mus(d):
    X = 1 - d["Z"] - d["Y"]
    neutral = d["NeutralHydrogenAbundance"]
    f_h2 = np.clip(d["MolecularMassFraction"], 0, 1) * neutral       # H2 share of all H (neutral H x molecular share)
    n_h2, n_hi, n_hii = X * f_h2 / 2, X * (neutral - f_h2), X * (1 - neutral)   # per m_H of gas
    n_e, n_he = X * d["ElectronAbundance"], d["Y"] / 4
    mu_particle = 1.0 / (n_h2 + n_hi + n_hii + n_e + n_he)
    heavy = n_h2 * 2 + n_hi + n_hii + n_he * 4
    mu_drag = (heavy / (n_h2 * np.sqrt(2) + n_hi + n_hii + n_he * 2)) ** 2
    mu_gizmo = K_B * d["Temperature"] / ((GAMMA - 1) * d["u_cgs"] * M_P)
    nH = d["rho_cgs"] * X / M_H
    return dict(mu_particle=mu_particle, mu_gizmo=mu_gizmo, mu_drag=mu_drag, f_h2=f_h2, nH=nH)


def wq(x, w, qs=(5, 50, 95)):
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return [x[o][np.searchsorted(c, q / 100)] for q in qs]


def main(snaps):
    for snap in snaps:
        d, nbox = box_gas(snap)
        r = mus(d)
        m = d["Masses"]
        print(f"== {snap}: {nbox} boxes, {len(m):,} gas particles inside them; Y = {np.median(d['Y']):.4f}, "
              f"Z = {np.median(d['Z']):.4f}")
        print(f"   mass fraction with H2 share > 0.5: {m[r['f_h2'] > 0.5].sum() / m.sum():.3f}, > 0.9: "
              f"{m[r['f_h2'] > 0.9].sum() / m.sum():.3f};  ionized H mass fraction "
              f"{np.sum(m * (1 - d['NeutralHydrogenAbundance'])) / m.sum():.2e}")
        for k in ("mu_particle", "mu_gizmo", "mu_drag"):
            p5, p50, p95 = wq(r[k], m)
            print(f"   {k:12s} mass-weighted 5/50/95%: {p5:.3f} / {p50:.3f} / {p95:.3f}   "
                  f"mean {np.sum(m * r[k]) / m.sum():.3f}")
        print("   by density (mass-weighted medians):")
        for lo, hi in ((0, 1e2), (1e2, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e12)):
            s = (r["nH"] >= lo) & (r["nH"] < hi)
            if s.sum() > 10:
                print(f"     n_H {lo:7.0e}-{hi:<7.0e} cm^-3: mass {m[s].sum() / m.sum():6.3f}  H2 share "
                      f"{wq(r['f_h2'][s], m[s], (50,))[0]:.3f}  mu_particle {wq(r['mu_particle'][s], m[s], (50,))[0]:.3f}  "
                      f"mu_gizmo {wq(r['mu_gizmo'][s], m[s], (50,))[0]:.3f}  mu_drag {wq(r['mu_drag'][s], m[s], (50,))[0]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["snapshot_160", "snapshot_440", "snapshot_500"])
