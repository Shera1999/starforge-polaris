#!/usr/bin/env python3
"""Compare STARFORGE's background radiation field with POLARIS's ISRF, band by band.

STARFORGE (GIZMO RT_ISRF_BACKGROUND) injects the Draine 1978/2011 solar-
neighbourhood field times InterstellarRadiationFieldStrength into its five RT
bands. POLARIS's <source_isrf> G0 multiplies the Mathis, Mezger & Panagia
(1983) field (CMathFunctions::mathis_isrf). This measures the photon energy
density per band in unshielded, low-density gas of a snapshot without stars
and integrates POLARIS's Mathis spectrum over the same bands, giving the G0
that would make POLARIS match STARFORGE in each band.

Usage: check_isrf_normalisation.py SNAPSHOT.hdf5
"""
import sys

import h5py
import numpy as np

C = 2.99792458e8
H_EV = 4.135667696e-15 * C       # h*c in eV m
HABING_ERG_CM3 = 5.29e-14        # Habing u(6-13.6 eV), Draine (2011) Table 12.1


def planck(lam, T):
    h, k = 6.62607015e-34, 1.380649e-23
    return 2 * h * C**2 / lam**5 / np.expm1(h * C / (lam * k * T))


def mathis_isrf(lam):
    """Copy of POLARIS CMathFunctions::mathis_isrf: J_lambda [W m^-2 m^-1 sr^-1]."""
    um = lam * 1e6
    out = np.zeros_like(lam)
    a = (lam >= 0.0912e-6) & (lam < 0.11e-6); out[a] = 3069.0 * um[a] ** 3.4172
    b = (lam >= 0.11e-6) & (lam < 0.134e-6); out[b] = 1.627
    c = (lam >= 0.134e-6) & (lam < 0.25e-6); out[c] = 0.0566 * um[c] ** -1.6678
    d = lam >= 0.25e-6
    out[d] = 1e-14 * planck(lam[d], 7500) + 1e-13 * planck(lam[d], 4000) + 4e-13 * planck(lam[d], 3000)
    return out


def mathis_energy_density_erg_cm3(e_min_ev, e_max_ev):
    lam = np.geomspace(H_EV / e_max_ev, H_EV / e_min_ev, 20001)
    u_si = 4 * np.pi / C * np.trapezoid(mathis_isrf(lam), lam)      # J m^-3
    return u_si * 10.0                                          # erg cm^-3


def main(path):
    with h5py.File(path, "r") as f:
        h = f["Header"].attrs
        e_lo, e_hi = h["Radiation_RHD_Min_Bin_Freq_in_eV"], h["Radiation_RHD_Max_Bin_Freq_in_eV"]
        ul, um, uv = (float(h[k]) for k in ("UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS"))
        print(f"{path}: Time {float(h['Time']) * ul / uv / 3.15576e13:.3f} Myr, "
              f"ISRF strength {h['InterstellarRadiationFieldStrength']}, sinks {h['NumPart_ThisFile'][5]}")
        g = f["PartType0"]
        n = len(g["Masses"])
        pick = slice(0, n, 50)                                  # every 50th particle is plenty
        rho = g["Density"][pick].astype(float) * um / ul**3
        mass = g["Masses"][pick].astype(float) * um
        E = g["PhotonEnergy"][pick].astype(float) * um * uv**2   # erg per particle per band
        nH = rho * 0.7155 / 1.6735575e-24
    u = E / (mass / rho)[:, None]                               # erg cm^-3
    diffuse = nH < 0.1
    print(f"particles sampled {len(rho):,}; unshielded (n_H < 0.1 cm^-3): {diffuse.sum():,}")
    print(f"{'band [eV]':>18} {'STARFORGE u':>13} {'Mathis u':>12} {'G0 to match':>12}")
    for b in range(len(e_lo)):
        u_sf = np.median(u[diffuse, b])
        u_m = mathis_energy_density_erg_cm3(e_lo[b], e_hi[b])
        g0 = u_sf / u_m if u_m > 0 else np.nan
        print(f"{e_lo[b]:8.4g}-{e_hi[b]:<8.4g} {u_sf:13.3e} {u_m:12.3e} {g0:12.3f}")
    u_hab = mathis_energy_density_erg_cm3(6.0, 13.6)
    print(f"Mathis field in Habing units (6-13.6 eV): {u_hab / HABING_ERG_CM3:.3f}  (Draine 1978 = 1.69)")


if __name__ == "__main__":
    main(sys.argv[1])
