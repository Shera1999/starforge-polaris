#!/usr/bin/env python3
"""Step 3: turn each step-2 box into a POLARIS input grid and command file.

For each snapshot, the snapshot is read ONCE and the gas for all of its boxes
is cut out in that pass; the boxes are then deposited in parallel.

Per box (a cube of side --side-pc around the core's density peak). With the
default --peak-position cell-centre the grid is shifted by half a cell so the
peak particle sits at the centre of cell (nc/2, nc/2, nc/2); centring the cube
exactly on the peak puts it on a corner shared by 8 cells, where float rounding
decides which cell receives the (usually sub-cell) peak particle.
  - every gas particle whose kernel overlaps the cube is deposited onto a
    uniform nc^3 grid with GIZMO's compact cubic-spline kernel, sampled at cell
    centres. Each particle's weights are normalised over its FULL kernel and
    only the part inside the cube is kept, so the grid holds exactly the SPH
    mass inside the cube and the faces are not biased (the previous
    polaris_prepare_core.py ignored particles centred outside the cube and
    renormalised clipped kernels inwards, leaving every face ~10% under-dense);
  - grid fields (POLARIS IDs): 0 gas number density n [m^-3], 2 dust
    temperature [K] (STARFORGE value; POLARIS recomputes it), 3 gas temperature
    [K], 4-6 magnetic field [T]. Temperatures and B are mass-weighted means.
    STARFORGE saves MagneticField in Tesla, which is what POLARIS expects, so B
    is not converted (verified in 1.finding_cores/check_snapshot_units.py);
  - sink particles with a radius and luminosity inside the cube become
    blackbody point sources, T_eff from L = 4 pi R^2 sigma T^4;
  - polaris.cmd runs CMD_TEMP_RAT (dust temperature + radiative-torque
    alignment radius, ISRF + stars) then CMD_DUST_EMISSION with RAT alignment,
    39 wavelengths x 3 viewing directions. All paths in it are relative to the
    core directory except the shared dust catalogue, so POLARIS is run with the
    core directory as its working directory.

POLARIS branch: experimental (commit 6c5e1a89c6e14ce806c47bbb87947b98e3f761b0).
  - It reads gas density only as number density (grid ID 28 is disabled) and
    uses rho = n * mu * m_u with mu from <mu>. RAT also uses n and mu directly
    (gas damping, thermal speed sqrt(2kT/(mu m_u))), so mu is the mean mass per
    gas particle: 2.33 for molecular gas with helium, as measured in the
    snapshot for n_H > 1e4 cm^-3 (see check_snapshot_units.py). The grid stores
    n = rho / (mu m_u), so rho, and with it the dust mass, is exact.
  - Dust: astrodust (Hensley & Draine 2023 optical properties, f_Fe = 0, with
    magnetic susceptibility for the Larmor criterion); material density 0 in
    <dust_component> means "use the catalogue value" (2898 kg m^-3). The
    catalogue covers 1 nm-1 um and 0.05 um-5 mm, i.e. all of POLARIS's internal
    wavelength grid (0.0912 um-2 mm) and every detector wavelength.
  - With RAT on, POLARIS launches nr_photons PER WAVELENGTH of its 100-point
    grid, so 1e6 ISRF photons = 1e8 in total, as many as CMD_TEMP used.
  - Without alignment this branch writes only 3 image planes (I, tau, N)
    although the FITS header lists 6; with alignment it writes I, Q, U, V,
    tau, N. Readers must check NAXIS4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from octree import Octree

PC_M = 3.0856775814913673e16
MSUN_KG = 1.98847e30
LSUN_W = 3.828e26
RSUN_M = 6.957e8
SIGMA_SB = 5.670374419e-8
VERSION = "3.5.1"
POLARIS_M_U_KG = 1.66053904e-27   # "m_H" in POLARIS Typedefs.hpp (atomic mass unit)

GAS_FIELDS = ("Masses", "SmoothingLength", "Dust_Temperature", "Temperature", "MagneticField",
              "DustToGasRatio_Local")
GRID_FIELD_IDS = (0, 2, 3, 4, 5, 6)
# Values carried per particle and mass-averaged per cell, in grid-field order 2..6.
AVERAGED = ("Dust_Temperature", "Temperature", "Bx", "By", "Bz")

# Detector wavelengths come from the instrument mapping, the single source of truth
# shared with the dataset builder (detector k = view k // n_wl, wavelength k % n_wl).
WAVELENGTH_CSV = Path(__file__).resolve().parents[2] / "wavelength_telescope_mapping.csv"
WAVELENGTHS_UM = [float(line.split(",")[0]) for line in WAVELENGTH_CSV.read_text().splitlines()[1:] if line.strip()]
# POLARIS detector rotations (rot_angle_1 about x, rot_angle_2 about y).
# (0, 0) looks along the grid z axis; detector pixel (row, col) = grid (iy, ix).
VIEWS = [(0, 0), (-90, 0), (0, 90)]
DUST = dict(name="astrodust_fe_0.0_sus", material_density_kg_m3=0.0,
            a_min_m=5e-9, a_max_m=2.5e-7, size_power=-3.5)


# ----------------------------------------------------------------------------
# Geometry and deposition
# ----------------------------------------------------------------------------
def periodic_delta(x, center, box):
    d = np.asarray(x, float) - center
    return d - box * np.rint(d / box)


def cubic_kernel(q):
    """GIZMO compact cubic-spline shape, support q = r/H < 1 (unnormalised)."""
    w = np.zeros_like(q)
    inner, outer = q < 0.5, (q >= 0.5) & (q < 1)
    w[inner] = 1 - 6 * q[inner] ** 2 + 6 * q[inner] ** 3
    w[outer] = 2 * (1 - q[outer]) ** 3
    return w


def deposit(positions, h, mass, values, nc, side):
    """Deposit particles onto an nc^3 grid spanning [-side/2, side/2)^3.

    Each particle's kernel is sampled at the centres of all cells it touches,
    including cells outside the cube; weights are normalised over that full
    stencil and only in-cube cells are kept. A particle whose kernel contains
    no cell centre (H well below the cell size) goes entirely to the cell that
    contains it, if that cell is inside the cube.

    Returns mass_grid [nc^3], weighted [nc^3, nvalues] (sum of dm * value),
    in_cube_fraction per particle, and the number of nearest-cell fallbacks.
    """
    dx = side / nc
    mass_grid = np.zeros((nc, nc, nc))
    weighted = np.zeros((nc, nc, nc, values.shape[1]))
    fraction = np.zeros(len(mass))
    fallback = 0
    for p in range(len(mass)):
        xyz, hh = positions[p], h[p]
        lo = np.ceil((xyz - hh + side / 2) / dx - 0.5).astype(int)
        hi = np.floor((xyz + hh + side / 2) / dx - 0.5).astype(int)
        w = None
        if np.all(lo <= hi):
            ix, iy, iz = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1),
                                     np.arange(lo[2], hi[2] + 1), indexing="ij")
            ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()
            centres = (np.column_stack((ix, iy, iz)) + 0.5) * dx - side / 2
            w = cubic_kernel(np.linalg.norm(centres - xyz, axis=1) / hh)
            if w.sum() <= 0:
                w = None
        if w is None:
            fallback += 1
            ind = np.floor((xyz + side / 2) / dx).astype(int)
            ix, iy, iz, w = ind[:1], ind[1:2], ind[2:3], np.ones(1)
        w = w / w.sum()
        inside = (ix >= 0) & (ix < nc) & (iy >= 0) & (iy < nc) & (iz >= 0) & (iz < nc)
        if not inside.any():
            continue
        ix, iy, iz, w = ix[inside], iy[inside], iz[inside], w[inside]
        fraction[p] = w.sum()
        dm = mass[p] * w
        np.add.at(mass_grid, (ix, iy, iz), dm)
        for j in range(values.shape[1]):
            np.add.at(weighted[..., j], (ix, iy, iz), dm * values[p, j])
    return mass_grid, weighted, fraction, fallback


def finalize_fields(mass_grid, weighted, dx, floor_factor=1e-6):
    """Cell fields (rho, T_dust, T_gas, Bx, By, Bz) from deposited sums.

    Empty cells (possible only for very sparse boxes) get a density floor of
    floor_factor * min(occupied density) and the averaged fields of the
    nearest occupied cell. The floor mass is reported, not hidden.
    """
    nc = mass_grid.shape[0]
    occupied = mass_grid > 0
    fields = np.empty((nc, nc, nc, 1 + weighted.shape[-1]))
    fields[..., 1:] = np.divide(weighted, mass_grid[..., None], out=np.zeros_like(weighted),
                                where=occupied[..., None])
    rho = mass_grid / dx ** 3
    empty = ~occupied
    floor = float(rho[occupied].min() * floor_factor) if empty.any() else 0.0
    rho[empty] = floor
    fields[..., 0] = rho
    if empty.any():
        nearest = distance_transform_edt(empty, return_distances=False, return_indices=True)
        for j in range(1, fields.shape[-1]):
            fields[..., j][empty] = fields[..., j][tuple(nearest[:, empty])]
    return fields, int(empty.sum()), floor * dx ** 3 * int(empty.sum())


def write_grid(path, fields, side):
    nc = fields.shape[0]
    level = int(round(math.log2(nc)))
    axis = -side / 2 + (np.arange(nc) + 0.5) * side / nc
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    tree = Octree(side)
    for xyz, row in zip(np.column_stack((x.ravel(), y.ravel(), z.ravel())), fields.reshape(-1, fields.shape[-1])):
        tree.insert(xyz, level, row)
    tree.write(path, GRID_FIELD_IDS)


# ----------------------------------------------------------------------------
# Snapshot reading: one pass for all boxes of a snapshot
# ----------------------------------------------------------------------------
def snapshot_units(f):
    h = f["Header"].attrs
    ul, um = float(h["UnitLength_In_CGS"]), float(h["UnitMass_In_CGS"])
    return dict(length_m=ul / 100, length_pc=ul / (100 * PC_M), mass_kg=um / 1000, box_code=float(h["BoxSize"]),
                time=float(h["Time"]))


def cut_boxes(snapshot, centers_pc, side_pc, chunk):
    """Gas whose kernel overlaps each box, from a single pass over the snapshot.

    Returns a list (one per box) of dicts with positions relative to the box
    centre [m], smoothing length [m], mass [kg] and the other GAS_FIELDS, plus
    a flag for whether the particle centre itself is inside the cube.
    """
    out = [{k: [] for k in ("pos", "inside", *GAS_FIELDS)} for _ in centers_pc]
    with h5py.File(snapshot, "r") as f:
        u = snapshot_units(f)
        g = f["PartType0"]
        for name in GAS_FIELDS:
            if name not in g:
                raise KeyError(f"{snapshot} is missing PartType0/{name}")
        centers = np.asarray(centers_pc, float) / u["length_pc"]
        half = side_pc / (2 * u["length_pc"])
        box = u["box_code"]
        n = len(g["Masses"])
        for a in range(0, n, chunk):
            sl = slice(a, min(a + chunk, n))
            xyz = np.asarray(g["Coordinates"][sl], float)
            hsml = np.asarray(g["SmoothingLength"][sl], float)
            hits = []
            for b, c in enumerate(centers):
                d0 = periodic_delta(xyz[:, 0], c[0], box)
                cand = np.flatnonzero(np.abs(d0) < half + hsml)
                if not len(cand):
                    continue
                rel = periodic_delta(xyz[cand], c, box)
                keep = np.all(np.abs(rel) < half + hsml[cand, None], axis=1)
                if keep.any():
                    hits.append((b, cand[keep], rel[keep]))
            if not hits:
                continue
            data = {name: np.asarray(g[name][sl], dtype=np.float64) for name in GAS_FIELDS
                    if name not in ("SmoothingLength",)}
            data["SmoothingLength"] = hsml
            for b, idx, rel in hits:
                out[b]["pos"].append(rel * u["length_m"])
                out[b]["inside"].append(np.all(np.abs(rel) < half, axis=1))
                for name in GAS_FIELDS:
                    out[b][name].append(data[name][idx])
    boxes = []
    for o in out:
        if not o["pos"]:
            raise RuntimeError("A box contains no gas")
        d = {k: np.concatenate(v) for k, v in o.items()}
        d["Masses"] = d["Masses"] * u["mass_kg"]
        d["SmoothingLength"] = d["SmoothingLength"] * u["length_m"]
        boxes.append(d)
    return boxes, u


def load_stars(snapshot, center_pc, side_pc):
    """Luminous sinks inside the cube, as POLARIS blackbody point sources."""
    with h5py.File(snapshot, "r") as f:
        u = snapshot_units(f)
        if "PartType5" not in f:
            return []
        s = f["PartType5"]
        need = ("Coordinates", "ParticleIDs", "ProtoStellarRadius_inSolar", "StarLuminosity_Solar")
        if not all(k in s for k in need):
            raise KeyError(f"{snapshot}: PartType5 lacks one of {need}")
        rel = periodic_delta(np.asarray(s["Coordinates"], float), np.asarray(center_pc) / u["length_pc"], u["box_code"])
        keep = np.all(np.abs(rel) < side_pc / (2 * u["length_pc"]), axis=1)
        stars = []
        for pid, pos, radius, lum, mass in zip(np.asarray(s["ParticleIDs"])[keep], rel[keep] * u["length_m"],
                                               np.asarray(s["ProtoStellarRadius_inSolar"], float)[keep],
                                               np.asarray(s["StarLuminosity_Solar"], float)[keep],
                                               np.asarray(s["Masses"], float)[keep] * u["mass_kg"] / MSUN_KG):
            entry = dict(particle_id=int(pid), position_m=pos.tolist(), radius_rsun=float(radius),
                         luminosity_lsun=float(lum), mass_msun=float(mass))
            if radius > 0 and lum > 0:
                entry["temperature_K"] = (lum * LSUN_W / (4 * math.pi * SIGMA_SB * (radius * RSUN_M) ** 2)) ** 0.25
                entry["used_as_source"] = True
            else:
                entry["used_as_source"] = False
            stars.append(entry)
    return stars


# ----------------------------------------------------------------------------
# POLARIS command file
# ----------------------------------------------------------------------------
def polaris_command(dust_path, dust_to_gas, stars, args):
    common = (f'<common>\n'
              f'  <dust_component> "{dust_path}" "plaw" 1.0 {DUST["material_density_kg_m3"]:g} '
              f'{DUST["a_min_m"]:g} {DUST["a_max_m"]:g} {DUST["size_power"]:g}\n'
              f'  <mu> {args.mu:g}\n'
              f'  <mass_fraction> {dust_to_gas:.9g}\n'
              f'  <nr_threads> {args.nr_threads}\n'
              f'</common>\n')
    sources = "".join(
        f'  <source_star nr_photons = "{args.star_photons}"> {s["position_m"][0]:.9e} {s["position_m"][1]:.9e} '
        f'{s["position_m"][2]:.9e} {s["radius_rsun"]:.9g} {s["temperature_K"]:.9g}\n'
        for s in stars if s["used_as_source"])
    temp = (f'<task> 1\n'
            f'  <cmd> CMD_TEMP_RAT\n'
            f'  <align> ALIG_RAT\n'
            f'{sources}'
            f'  <source_isrf nr_photons = "{args.isrf_photons}"> {args.isrf_strength:g} {args.isrf_radius_factor:g}\n'
            f'  <write_inp_midplanes> {args.nc}\n'
            f'  <write_out_midplanes> {args.nc}\n'
            f'  <path_grid> "grid.dat"\n'
            f'  <path_out> "temp/"\n'
            f'</task>\n')
    detectors = "".join(
        f'  <detector_dust nr_pixel = "{args.nc}"> {wl * 1e-6:.8e} {wl * 1e-6:.8e} 1 1 {r1:.1f} {r2:.1f} '
        f'{args.distance_pc * PC_M:.9e}\n'
        for r1, r2 in VIEWS for wl in WAVELENGTHS_UM)
    emission = (f'<task> 1\n'
                f'  <cmd> CMD_DUST_EMISSION\n'
                f'  <align> ALIG_RAT\n'
                f'  <max_subpixel_lvl> {args.max_subpixel_lvl}\n'
                f'  <path_grid> "temp/grid_temp_rat.dat"\n'
                f'  <path_out> "em/"\n'
                f'{detectors}'
                f'</task>\n')
    return common + "\n" + temp + "\n" + emission


def command_settings(dust_path, args):
    """The polaris.cmd choices, recorded in summary.json next to the command file."""
    return dict(
        dust_to_gas_ratio=args.dust_to_gas, dust_catalog=str(dust_path),
        dust_size_distribution=dict(kind="plaw", a_min_m=DUST["a_min_m"], a_max_m=DUST["a_max_m"],
                                    power=DUST["size_power"], material_density="catalogue"),
        isrf_strength_G0=args.isrf_strength, isrf_photons_per_wavelength=args.isrf_photons,
        star_photons_per_wavelength=args.star_photons, polaris_wavelength_grid="100 points, 0.0912-2000 um",
        isrf_radius_factor=args.isrf_radius_factor, distance_pc=args.distance_pc,
        max_subpixel_lvl=args.max_subpixel_lvl, wavelength_csv=str(WAVELENGTH_CSV),
        wavelengths_um=WAVELENGTHS_UM, views=VIEWS, n_detectors=len(WAVELENGTHS_UM) * len(VIEWS),
        temperature_command="CMD_TEMP_RAT", alignment="ALIG_RAT (radiative torques)",
        temp_output_grid="temp/grid_temp_rat.dat",
        polaris_branch="experimental 6c5e1a89c6e14ce806c47bbb87947b98e3f761b0")


def rewrite_commands(args, dust_path):
    """Regenerate polaris.cmd (and the matching summary fields) of finished boxes; grids are untouched."""
    summaries = sorted(args.output.glob("snapshot_*/core_*/summary.json"))
    if args.snapshots:
        summaries = [p for p in summaries if p.parent.parent.name in args.snapshots]
    for path in summaries:
        summary = json.loads(path.read_text())
        if not math.isclose(summary["mu"], args.mu):
            raise ValueError(f"{path}: grid encoded with mu = {summary['mu']}, not --mu {args.mu}")
        if "starforge_DustToGasRatio_Local_median" not in summary:     # before v3.2 this held the raw STARFORGE value
            summary["starforge_DustToGasRatio_Local_median"] = summary["dust_to_gas_ratio"]
        summary.update(command_settings(dust_path, args), version=VERSION,
                       commands_written_utc=datetime.now(timezone.utc).isoformat())
        for name, text in (("polaris.cmd", polaris_command(dust_path, args.dust_to_gas, summary["stars"], args)),
                           ("summary.json", json.dumps(summary, indent=2) + "\n")):
            tmp = path.parent / f".{name}.tmp"
            tmp.write_text(text)
            os.replace(tmp, path.parent / name)
    print(f"rewrote polaris.cmd for {len(summaries)} boxes", flush=True)


# ----------------------------------------------------------------------------
# One box
# ----------------------------------------------------------------------------
def build_box(job):
    """Deposit, write grid.dat / polaris.cmd / summary.json for one box (atomic)."""
    box, gas, stars, snap_meta, args, dust_path = job
    out_dir = Path(args.output) / box["snapshot"] / f"core_{int(box['core_id']):06d}"
    if (out_dir / "summary.json").exists():
        return out_dir, "skipped"
    t0 = time.time()
    side_m = args.side_pc * PC_M
    values = np.column_stack((gas["Dust_Temperature"], gas["Temperature"], gas["MagneticField"]))
    mass_grid, weighted, fraction, fallback = deposit(gas["pos"], gas["SmoothingLength"], gas["Masses"],
                                                      values, args.nc, side_m)
    dx = side_m / args.nc
    fields, n_empty, floor_mass = finalize_fields(mass_grid, weighted, dx)
    if not np.all(np.isfinite(fields)) or not np.all(fields[..., 0] > 0):
        raise RuntimeError(f"{out_dir}: non-finite or non-positive grid values")
    expected = float(np.sum(gas["Masses"] * fraction))
    if not math.isclose(mass_grid.sum(), expected, rel_tol=1e-10):
        raise RuntimeError(f"{out_dir}: deposited mass {mass_grid.sum()} != expected {expected}")

    inside = gas["inside"]
    crosses_face = np.any(np.abs(gas["pos"]) + gas["SmoothingLength"][:, None] > side_m / 2, axis=1)
    max_face_kernel_cells = float(gas["SmoothingLength"][crosses_face].max() / dx) if crosses_face.any() else 0.0
    # STARFORGE stores Z x (dust-to-metals / solar), not dust/gas; kept for provenance only
    starforge_dtg_local = float(np.median(gas["DustToGasRatio_Local"][inside]))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.building-", dir=out_dir.parent))
    try:
        encoded = fields.copy()
        encoded[..., 0] = fields[..., 0] / (args.mu * POLARIS_M_U_KG)   # POLARIS field 0: n [m^-3]
        write_grid(staging / "grid.dat", encoded, side_m)
        (staging / "polaris.cmd").write_text(polaris_command(dust_path, args.dust_to_gas, stars, args))
        summary = dict(
            version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
            snapshot=box["snapshot"], snapshot_file=snap_meta["file"], time_Myr=snap_meta["time_Myr"],
            core_id=int(box["core_id"]), box_id=int(box["box_id"]), family_id=int(box["family_id"]),
            peak_pc=[float(box[k]) for k in ("peak_x_pc", "peak_y_pc", "peak_z_pc")],
            grid_center_pc=[float(c) for c in box["grid_center_pc"]], peak_position=args.peak_position,
            peak_cell_index=[int(np.floor((float(box[k]) - c) / (args.side_pc / args.nc) + args.nc / 2))
                             for k, c in zip(("peak_x_pc", "peak_y_pc", "peak_z_pc"), box["grid_center_pc"])],
            nc=args.nc, side_pc=args.side_pc, cell_pc=args.side_pc / args.nc,
            grid_field_ids=list(GRID_FIELD_IDS),
            grid_field_units=["m^-3", "K", "K", "T", "T", "T"],
            mu=args.mu, polaris_m_u_kg=POLARIS_M_U_KG,
            n_particles_used=int(len(fraction)), n_particles_centre_inside=int(inside.sum()),
            n_particles_kernel_only=int((~inside).sum()),
            n_particles_nearest_cell_fallback=int(fallback),
            n_particles_kernel_crossing_face=int(crosses_face.sum()),
            max_kernel_crossing_face_cells=max_face_kernel_cells,
            mass_particles_centre_inside_Msun=float(gas["Masses"][inside].sum() / MSUN_KG),
            grid_mass_Msun=float(mass_grid.sum() / MSUN_KG),
            n_empty_cells_floored=n_empty, floor_mass_Msun=floor_mass / MSUN_KG,
            density_kg_m3_min_max=[float(fields[..., 0].min()), float(fields[..., 0].max())],
            dust_temperature_K_min_max=[float(fields[..., 1].min()), float(fields[..., 1].max())],
            B_T_max=float(np.linalg.norm(fields[..., 3:6], axis=-1).max()),
            starforge_DustToGasRatio_Local_median=starforge_dtg_local,
            stars=stars, n_stellar_sources=int(sum(s["used_as_source"] for s in stars)),
            illumination="stars_plus_isrf" if any(s["used_as_source"] for s in stars) else "isrf_only",
            **command_settings(dust_path, args),
            seconds=time.time() - t0,
        )
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if out_dir.exists():
            shutil.rmtree(out_dir)   # an incomplete earlier attempt (no summary.json)
        os.rename(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out_dir, f"ok {time.time() - t0:.0f}s"


def catalogue_susceptibility(path):
    """8th number of the catalogue's parameter line: experimental POLARIS reads it as the
    magnetic susceptibility chi. chi = 0 gives a_larm = 0, i.e. no RAT alignment and Q = U = 0."""
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    values = [float(v) for v in lines[1]]          # lines[0] is the string ID
    if len(values) != 8:
        raise ValueError(f"{path}: parameter line has {len(values)} numbers, POLARIS needs 8")
    return values[7]


def stage_dust(polaris_root, dest):
    src = polaris_root / "input" / "dust_cs" / f"{DUST['name']}.dat"
    if not catalogue_susceptibility(src) > 0:
        raise ValueError(f"{src}: susceptibility <= 0 switches off RAT alignment (use a *_sus catalogue)")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest / src.name)
    side_dir = src.with_suffix("")
    if side_dir.is_dir():   # calorimetry files for stochastic heating
        shutil.copytree(side_dir, dest / side_dir.name, dirs_exist_ok=True)
    return (dest / src.name).resolve()


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxes", type=Path, default=here.parent / "2.selecting_boxes" / "boxes_2.2pc" / "boxes.csv")
    p.add_argument("--snapshot-dir", type=Path, default=Path("/scratch/STARFORGE/M2e5_R30_Res585"))
    p.add_argument("--output", type=Path, default=here / "grids_64")
    p.add_argument("--polaris-root", type=Path, default=here.parent.parent / "M2e5_R30_Res585" / "POLARIS_experimental")
    p.add_argument("--snapshots", nargs="*", help="Only these snapshots (e.g. snapshot_440)")
    p.add_argument("--box-ids", type=int, nargs="*", help="Only these box_ids (for tests)")
    p.add_argument("--nc", type=int, default=64)
    p.add_argument("--side-pc", type=float, default=2.2)
    p.add_argument("--chunk", type=int, default=4_000_000)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--nr-threads", type=int, default=8, help="POLARIS threads written into polaris.cmd")
    p.add_argument("--star-photons", type=int, default=1_000_000,
                   help="Per wavelength (CMD_TEMP_RAT); 1e6 halves the 1e5 temperature noise far from stars (M2e5_R30_Res585/POLARIS/star_photon_test/)")
    p.add_argument("--isrf-photons", type=int, default=10_000_000,
                   help="Per wavelength (CMD_TEMP_RAT); 1e7 at radius 10 gives ~0.008 K noise per cell, 0.014 K in corners (M2e5_R30_Res585/POLARIS/isrf_photon_noise_test/)")
    p.add_argument("--mu", type=float, default=2.33,
                   help="Mean mass per gas particle in m_u, for POLARIS <mu> and the number-density encoding")
    p.add_argument("--dust-to-gas", type=float, default=0.01,
                   help="POLARIS <mass_fraction>; STARFORGE uses 0.01 Z/Z_sun with Z = Z_sun (Grudic+2021 sec. 3.2)")
    p.add_argument("--isrf-strength", type=float, default=1.0, help="G0, matching STARFORGE's ISRF strength")
    p.add_argument("--isrf-radius-factor", type=float, default=10.0,
                   help="Launch sphere radius / (sqrt(3)/2 box side); 10: lowest interior noise, corners 1.8x interior (>= 20 uniform)")
    p.add_argument("--max-subpixel-lvl", type=int, default=0)
    p.add_argument("--distance-pc", type=float, default=140.0)
    p.add_argument("--commands-only", action="store_true",
                   help="Only regenerate polaris.cmd/summary.json of existing boxes (grids untouched)")
    p.add_argument("--peak-position", choices=["cell-centre", "cell-corner"], default="cell-centre",
                   help="cell-corner reproduces the previous pipeline's centring (only for regression tests)")
    args = p.parse_args()
    if args.nc < 2 or 2 ** int(round(math.log2(args.nc))) != args.nc:
        p.error("--nc must be a power of two")

    boxes = pd.read_csv(args.boxes)
    if args.snapshots:
        boxes = boxes[boxes.snapshot.isin(args.snapshots)]
    if args.box_ids:
        boxes = boxes[boxes.box_id.isin(args.box_ids)]
    args.output.mkdir(parents=True, exist_ok=True)
    dust_path = stage_dust(args.polaris_root.resolve(), here / "dust_input")
    config_name = "run_config_commands.json" if args.commands_only else "run_config.json"
    (args.output / config_name).write_text(json.dumps(dict(
        vars(args), boxes=str(args.boxes.resolve()), snapshot_dir=str(args.snapshot_dir), output=str(args.output),
        polaris_root=str(args.polaris_root), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        version=VERSION, created_utc=datetime.now(timezone.utc).isoformat()), indent=2, default=str) + "\n")
    if args.commands_only:
        rewrite_commands(args, dust_path)
        return

    for snap, sb in boxes.groupby("snapshot", sort=True):
        todo = [r for _, r in sb.iterrows()
                if not (args.output / snap / f"core_{int(r.core_id):06d}" / "summary.json").exists()]
        if not todo:
            print(f"{snap}: all {len(sb)} boxes done", flush=True)
            continue
        t0 = time.time()
        snapshot = args.snapshot_dir / f"{snap}.hdf5"
        # Grid centre half a cell below the peak puts the peak at the centre of cell nc/2.
        shift = -args.side_pc / args.nc / 2 if args.peak_position == "cell-centre" else 0.0
        centers = [[r.peak_x_pc + shift, r.peak_y_pc + shift, r.peak_z_pc + shift] for r in todo]
        gas, u = cut_boxes(snapshot, centers, args.side_pc, args.chunk)
        with h5py.File(snapshot, "r") as f:
            h = f["Header"].attrs
            unit_time_s = float(h["UnitLength_In_CGS"]) / float(h["UnitVelocity_In_CGS"])
        meta = dict(file=str(snapshot), time_Myr=u["time"] * unit_time_s / (1e6 * 365.25 * 86400))
        jobs = [(dict(r.to_dict(), grid_center_pc=c), g, load_stars(snapshot, c, args.side_pc), meta, args, dust_path)
                for r, g, c in zip(todo, gas, centers)]
        print(f"{snap}: cut {len(todo)} boxes in {time.time() - t0:.0f}s; depositing with {args.workers} workers",
              flush=True)
        with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
            for out_dir, status in pool.imap_unordered(build_box, jobs):
                print(f"  {out_dir.parent.name}/{out_dir.name}: {status}", flush=True)
        print(f"{snap}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
