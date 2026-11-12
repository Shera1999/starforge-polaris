#!/usr/bin/env python3
"""Build a POLARIS grid and full-Stokes command files for one core, from any
core catalog (v4 or v5) and any core_id.

Generalizes two previously core-729-specific, hand-maintained scripts into
one reusable tool:

  - core_0729_polaris_final/prepare_core.py: the mass-conserving 64^3 grid by
    SPH compact-cubic-kernel deposition (density, dust temperature, gas
    temperature, magnetic field), with per-particle boundary-weight
    renormalization so the selected gas mass is reproduced exactly.
  - core_0729_polaris/prepare_core.py: stellar-source extraction from
    PartType5 (sink radius + luminosity -> blackbody temperature via
    Stefan-Boltzmann), POLARIS command-file generation (CMD_TEMP and
    CMD_DUST_EMISSION, with and without the ISRF), and dust-catalog staging.

The command-file *template* (RAT alignment, Rayleigh reduction factor, ISRF
photon count/radius factor, subpixel refinement) is taken verbatim from
core_0729_polaris_final/core_000729/{temperature,dust_emission}.cmd, which is
the most recently finalized, most complete version of this pipeline. Only the
per-core content (grid path, stellar sources, output paths) is templated.

Command-file paths are written relative to --output-root, matching the
existing convention: POLARIS must be invoked with that directory as the
working directory, e.g. `cd polaris_cores_v5 && polaris core_000471/temperature.cmd`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt

from octree import Octree

PC_M = 3.0856775814913673e16       # meters in a parsec
MSUN_KG = 1.98847e30                # kilograms in a solar mass
LSUN_W = 3.828e26                   # watts in a solar luminosity
RSUN_M = 6.957e8                    # meters in a solar radius
SIGMA_SB = 5.670374419e-8           # Stefan-Boltzmann constant [W m^-2 K^-4]
POLARIS_M_H_KG = 1.66053904e-27     # atomic mass unit as defined in POLARIS's Typedefs.hpp

# Two physics configurations this script can produce, selected with
# --physics-variant. "stable" matches the original core_0729_polaris_final
# convention (basic POLARIS, silicate, CMD_TEMP). "experimental-astrodust-rat"
# matches core_0729_polaris_final2 (pinned experimental commit
# 6c5e1a89c6e14ce806c47bbb87947b98e3f761b0): astrodust optical constants,
# number-density grid encoding (field ID 0, not mass-density field ID 28),
# and CMD_TEMP_RAT. The emission stage's alignment directive is commented
# out ("#<align> ALIG_RAT"), i.e. emission runs with no alignment at all;
# that asymmetry vs. the temperature stage (which keeps <align> ALIG_RAT
# active, matching CMD_TEMP_RAT) is deliberate, not "fixed".
PHYSICS_VARIANTS = {
    "stable": dict(
        grid_density_field_id=28, mu=None,
        dust_filename="silicate_oblate.dat", dust_material_density_kg_m3=3800.0,
        temp_cmd="CMD_TEMP", temp_output_filename="grid_temp.dat",
        phase_function_active=True,
        temp_align_active=True, temp_align_name="ALIG_PA",
        emission_align_active=True, emission_align_name="ALIG_PA",
        r_rayleigh_active_temp=True, r_rayleigh_active_emission=True,
    ),
    "experimental-astrodust-rat": dict(
        grid_density_field_id=0, mu=2.0,
        dust_filename="astrodust_fe_0.0_sus.dat", dust_material_density_kg_m3=0.0,
        temp_cmd="CMD_TEMP_RAT", temp_output_filename="grid_temp_rat.dat",
        phase_function_active=False,
        temp_align_active=True, temp_align_name="ALIG_RAT",
        emission_align_active=False, emission_align_name="ALIG_RAT",
        r_rayleigh_active_temp=True, r_rayleigh_active_emission=False,
    ),
}
# Original 16 core_0729 wavelengths, unioned with every distinct wavelength in
# /Users/sjafaritabar/PhD-Projects/dataset_wavelengths.csv (31 filters, one
# exact duplicate value at 350 um between spire_2 and alma_10_c43_8), with the
# 7 values already present in the original list (12, 22, 100, 160, 250, 350,
# 500) not repeated. 39 unique wavelengths total, ascending.
WAVELENGTHS_UM = [3.4, 3.6, 4.5, 4.6, 5.8, 8.0, 8.28, 12, 12.13, 14.65, 21.34,
                  22, 24.0, 25, 53.0, 60, 63.0, 65, 70.0, 89.0, 90, 100, 140,
                  154.0, 160, 214.0, 250, 349.82, 350, 460.0, 500, 550.08,
                  690.0, 849.27, 870.0, 945.0, 1100.0, 1300.0, 1381.53]
VIEWS = [(0, 0), (-90, 0), (0, 90)]


def periodic_delta(x, center, box):
    d = np.asarray(x, float) - center
    return d - box * np.rint(d / box)


def load_core_row(catalog: Path, core_id: int) -> dict[str, str]:
    with catalog.open(newline="") as handle:
        row = next((row for row in csv.DictReader(handle)
                    if int(row["core_id"]) == core_id), None)
    if row is None:
        raise ValueError(f"Core {core_id} is absent from {catalog}")
    return row


def load_gas(snapshot: Path, center_pc, side_pc, chunk):
    """Read gas properties inside a periodic cube, in SI units."""
    names = ("Masses", "SmoothingLength", "Dust_Temperature", "Temperature",
              "MagneticField", "DustToGasRatio_Local")
    out = {n: [] for n in names}
    positions = []
    with h5py.File(snapshot, "r") as f:
        h = f["Header"].attrs
        ul, um = float(h["UnitLength_In_CGS"]), float(h["UnitMass_In_CGS"])
        length_pc, length_m, mass_kg = ul / (100 * PC_M), ul / 100, um / 1000
        center_code, box_code = center_pc / length_pc, float(h["BoxSize"])
        half_code = side_pc / (2 * length_pc)
        gas = f["PartType0"]
        for required in names:
            if required not in gas:
                raise KeyError(f"Snapshot is missing PartType0/{required}")
        n = len(gas["Masses"])
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            xyz = np.asarray(gas["Coordinates"][a:b], float)
            rel = periodic_delta(xyz, center_code, box_code)
            keep = np.all(np.abs(rel) < half_code, axis=1)
            if keep.any():
                positions.append(rel[keep] * length_m)
                for name in names:
                    out[name].append(np.asarray(gas[name][a:b], dtype=np.float64)[keep])
            if b % 20_000_000 < chunk or b == n:
                print(f"scan {b:,}/{n:,}", flush=True)
    if not positions:
        raise RuntimeError("No gas elements found inside the selected cube")
    data = {k: np.concatenate(v) for k, v in out.items()}
    data["Masses"] *= mass_kg
    data["SmoothingLength"] *= length_m
    return np.concatenate(positions), data


def load_stars(snapshot: Path, center_pc, side_pc):
    """Read luminous sink sources inside the same periodic cube, in SI units."""
    with h5py.File(snapshot, "r") as f:
        h = f["Header"].attrs
        ul = float(h["UnitLength_In_CGS"])
        length_pc, length_m = ul / (100 * PC_M), ul / 100
        center_code, box_code = center_pc / length_pc, float(h["BoxSize"])
        half_code = side_pc / (2 * length_pc)
        stars = []
        if "PartType5" not in f:
            return stars
        sinks = f["PartType5"]
        required = ("Coordinates", "ParticleIDs", "ProtoStellarRadius_inSolar", "StarLuminosity_Solar")
        if not all(name in sinks for name in required):
            return stars
        rel = periodic_delta(np.asarray(sinks["Coordinates"], float), center_code, box_code)
        keep = np.all(np.abs(rel) < half_code, axis=1)
        for particle_id, position, radius, luminosity in zip(
            np.asarray(sinks["ParticleIDs"])[keep], rel[keep] * length_m,
            np.asarray(sinks["ProtoStellarRadius_inSolar"])[keep],
            np.asarray(sinks["StarLuminosity_Solar"])[keep],
        ):
            radius, luminosity = float(radius), float(luminosity)
            if radius > 0 and luminosity > 0:
                temperature = (luminosity * LSUN_W / (4 * math.pi * SIGMA_SB * (radius * RSUN_M) ** 2)) ** 0.25
                stars.append({"particle_id": int(particle_id), "position_m": position.tolist(),
                              "radius_rsun": radius, "luminosity_lsun": luminosity, "temperature_K": temperature})
    return stars


def cubic_kernel(q):
    """GIZMO compact cubic-spline shape with support q=r/H<1."""
    w = np.zeros_like(q)
    inner, outer = q < .5, (q >= .5) & (q < 1)
    w[inner] = 1 - 6 * q[inner] ** 2 + 6 * q[inner] ** 3
    w[outer] = 2 * (1 - q[outer]) ** 3
    return w


def deposit(positions, data, nc, side_m):
    """Mass-conserving SPH-kernel deposition onto a uniform nc^3 grid."""
    dx = side_m / nc
    mass_grid = np.zeros((nc, nc, nc), float)
    quantities = np.column_stack((data["Dust_Temperature"], data["Temperature"], data["MagneticField"]))
    numer = np.zeros((nc, nc, nc, quantities.shape[1]), float)
    fallback = boundary = 0
    for p, (xyz, hh, m, values) in enumerate(zip(positions, data["SmoothingLength"], data["Masses"], quantities)):
        lo = np.ceil((xyz - hh + side_m / 2) / dx - .5).astype(int)
        hi = np.floor((xyz + hh + side_m / 2) / dx - .5).astype(int)
        unclipped_lo, unclipped_hi = lo.copy(), hi.copy()
        lo, hi = np.maximum(lo, 0), np.minimum(hi, nc - 1)
        boundary += int(np.any(unclipped_lo < 0) | np.any(unclipped_hi >= nc))
        if np.any(lo > hi):
            ind = np.clip(np.floor((xyz + side_m / 2) / dx).astype(int), 0, nc - 1)
            ix, iy, iz, weight = np.array([ind[0]]), np.array([ind[1]]), np.array([ind[2]]), np.ones(1)
            fallback += 1
        else:
            ix, iy, iz = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1),
                                      np.arange(lo[2], hi[2] + 1), indexing="ij")
            centers = np.column_stack(((ix.ravel() + .5) * dx - side_m / 2, (iy.ravel() + .5) * dx - side_m / 2,
                                        (iz.ravel() + .5) * dx - side_m / 2))
            weight = cubic_kernel(np.linalg.norm(centers - xyz, axis=1) / max(hh, np.finfo(float).tiny))
            ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()
            positive = weight > 0
            ix, iy, iz, weight = ix[positive], iy[positive], iz[positive], weight[positive]
            if not len(weight):
                ind = np.clip(np.floor((xyz + side_m / 2) / dx).astype(int), 0, nc - 1)
                ix, iy, iz, weight = np.array([ind[0]]), np.array([ind[1]]), np.array([ind[2]]), np.ones(1)
                fallback += 1
        weight = weight / weight.sum()
        dm = m * weight
        np.add.at(mass_grid, (ix, iy, iz), dm)
        for j, value in enumerate(values):
            np.add.at(numer[..., j], (ix, iy, iz), dm * value)
        if (p + 1) % 50_000 == 0:
            print(f"deposit {p + 1:,}/{len(positions):,}", flush=True)
    return mass_grid, numer, dx, fallback, boundary


def stage_dust(source_catalog: Path, output: Path) -> Path:
    if not source_catalog.is_file():
        raise FileNotFoundError(f"Incomplete POLARIS dust catalogue: {source_catalog}")
    destination = output / "dust_input"
    destination.mkdir(exist_ok=True)
    shutil.copy2(source_catalog, destination / source_catalog.name)
    source_data = source_catalog.with_suffix("")
    if source_data.is_dir():
        # Some catalogs (e.g. silicate_oblate) need a sibling calorimetry
        # directory for stochastic heating; astrodust catalogs are
        # self-contained and have no such directory.
        shutil.copytree(source_data, destination / source_data.name, dirs_exist_ok=True)
    return destination / source_catalog.name


def write_commands(output_root: Path, core_dir_name: str, dust_rel: Path, stars, dust_to_gas_ratio, args, variant):
    """Write one polaris.cmd per core: a temperature task followed by a
    dust_emission task, both ISRF-illuminated.

    There is no no-ISRF branch: build_training_dataset.py only ever reads
    temp/grid_temp*.dat and em/data/*.fits.gz (the ISRF branch), so a
    no-ISRF branch is pure unused computation -- and CMD_TEMP_RAT's
    nr_photons-per-wavelength cost (see PHYSICS_VARIANTS) makes that waste
    substantial (~30% of a star-hosting core's runtime).

    POLARIS supports an arbitrary number of <task> blocks per command file,
    run strictly in order by one process invocation (CPipeline::Run()), each
    reloading its own <path_grid> fresh from disk -- so the emission task
    correctly picks up the temperature task's just-written output grid
    (temp/grid_temp*.dat) without needing a separate polaris invocation.
    Only <dust_component>/<mu>/<mass_fraction>/<nr_threads> are truly
    identical across every task and go in <common>; <phase_function>/
    <align>/<R_rayleigh> differ between the temperature and emission stages
    (see PHYSICS_VARIANTS) and are written directly into each task body.

    Caveat: CPipeline::Run() does not abort on a failed task -- it keeps
    running subsequent tasks regardless, and only the *last* task's result
    decides the final "SUCCESSFULLY FINISHED"/"ABORTED" banner. A failed
    temperature task would not necessarily be caught by that banner alone,
    so callers must additionally check that the expected output files exist
    (temp/grid_temp*.dat, the right count of em/data/*.fits.gz).

    Template matches core_0729_polaris_final/core_000729/*.cmd verbatim for
    the "stable" variant, and core_0729_polaris_final2 for
    "experimental-astrodust-rat" (RAT alignment, Rayleigh reduction factor,
    ISRF photon count and radius factor, subpixel refinement level),
    parameterized only where content is genuinely per-core. Paths are
    relative to output_root.
    """
    mu_line = f"\n  <mu> {variant['mu']:g}" if variant["mu"] is not None else ""
    common = f'''<common>
  <dust_component> "{core_dir_name}/dust_input/{dust_rel.name}" "plaw" 1.0 {variant['dust_material_density_kg_m3']:g} 5e-09 2.5e-07 -3.5{mu_line}
  <mass_fraction> {dust_to_gas_ratio:.9g}
  <nr_threads> {args.nr_threads}
</common>
'''

    def alignment_lines(align_active, align_name, r_rayleigh_active):
        phase_line = ("  <phase_function> PH_ISO" if variant["phase_function_active"]
                      else "  #<phase_function> PH_ISO")
        align_line = f"  <align> {align_name}" if align_active else f"  #<align> {align_name}"
        rayleigh_line = "  <R_rayleigh> 1.0" if r_rayleigh_active else "  #<R_rayleigh> 1.0"
        return f"{phase_line}\n{align_line}\n{rayleigh_line}\n"

    temp_alignment = alignment_lines(variant["temp_align_active"], variant["temp_align_name"],
                                      variant["r_rayleigh_active_temp"])
    emission_alignment = alignment_lines(variant["emission_align_active"], variant["emission_align_name"],
                                          variant["r_rayleigh_active_emission"])

    source_lines = []
    for star in stars:
        x, y, z = star["position_m"]
        source_lines.append(
            f'  <source_star nr_photons = "{args.star_photons}"> '
            f'{x:.9e} {y:.9e} {z:.9e} {star["radius_rsun"]:.9g} {star["temperature_K"]:.9g}'
        )
    detectors = []
    for rotation_1, rotation_2 in VIEWS:
        for wavelength_um in WAVELENGTHS_UM:
            wavelength_m = wavelength_um * 1e-6
            detectors.append(
                f'  <detector_dust nr_pixel = "{args.nc}"> {wavelength_m:.8e} {wavelength_m:.8e} 1 1 '
                f'{rotation_1:.1f} {rotation_2:.1f} {args.distance_pc * PC_M:.9e}'
            )
    source_text, detector_text = "\n".join(source_lines), "\n".join(detectors)

    temp_dir, emission_dir = f"{core_dir_name}/temp", f"{core_dir_name}/em"
    isrf = (f'  <source_isrf nr_photons = "{args.isrf_photons}"> '
            f'{args.isrf_strength} {args.isrf_radius_factor:g}\n')
    temp_task = f'''<task> 1
  <cmd> {variant["temp_cmd"]}
{temp_alignment}{source_text}
{isrf}  <write_inp_midplanes> {args.nc}
  <write_out_midplanes> {args.nc}
  <path_grid> "{core_dir_name}/grid.dat"
  <path_out> "{temp_dir}/"
</task>
'''
    emission_task = f'''<task> 1
  <cmd> CMD_DUST_EMISSION
{emission_alignment}  <max_subpixel_lvl> {args.max_subpixel_lvl}
  <path_grid> "{temp_dir}/{variant["temp_output_filename"]}"
  <path_out> "{emission_dir}/"
{detector_text}
</task>
'''
    (output_root / core_dir_name / "polaris.cmd").write_text(common + "\n" + temp_task + "\n" + emission_task)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True,
                        help="Core catalog CSV, e.g. dendrogram_cores_v5/core_catalog.csv")
    parser.add_argument("--core-id", type=int, required=True,
                        help="core_id from --catalog; note core_id is only meaningful within one catalog")
    parser.add_argument("--side-pc", type=float, default=2.2,
                        help="Side length of the box POLARIS actually simulates.")
    parser.add_argument("--crop-side-pc", type=float, default=None,
                        help="If given (and less than --side-pc), POLARIS still simulates the full "
                             "--side-pc box -- e.g. a wider box than the region of interest, so cells "
                             "near the region's edges see a more realistic surrounding radiation field "
                             "and medium instead of an artificial vacuum boundary -- but the grid and "
                             "detector images get cropped to this smaller side length when later read "
                             "by build_training_dataset.py. --nc is unaffected (fixed at a power of 2 "
                             "for the octree writer), so the crop reduces cell/pixel count rather than "
                             "changing resolution; the kept region's actual span is the nearest whole "
                             "number of cells, which may differ slightly from --crop-side-pc.")
    parser.add_argument("--nc", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=2_000_000)
    parser.add_argument("--star-photons", type=int, default=10_000_000)
    parser.add_argument("--isrf-photons", type=int, default=100_000_000)
    parser.add_argument("--isrf-strength", type=float, default=1.0)
    parser.add_argument("--isrf-radius-factor", type=float, default=3.0)
    parser.add_argument("--max-subpixel-lvl", type=int, default=3)
    parser.add_argument("--distance-pc", type=float, default=140.0)
    parser.add_argument("--nr-threads", type=int, default=-1,
                        help="POLARIS <nr_threads>; -1 uses every core on the machine. Set explicitly "
                             "when running several cores' pipelines concurrently to avoid oversubscription.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Parent directory POLARIS will be run from; each core is written to <output-root>/core_NNNNNN/")
    parser.add_argument("--physics-variant", choices=sorted(PHYSICS_VARIANTS), required=True,
                        help="'stable' = core_0729_polaris_final convention (basic POLARIS, silicate, CMD_TEMP, "
                             "mass-density field ID 28). 'experimental-astrodust-rat' = core_0729_polaris_final2 "
                             "convention (pinned experimental commit 6c5e1a89c6e14ce806c47bbb87947b98e3f761b0: "
                             "astrodust, CMD_TEMP_RAT, number-density field ID 0). No default: this materially "
                             "changes the physics, so it must be chosen explicitly every run.")
    parser.add_argument("--dust-catalog", type=Path, default=None,
                        help="Defaults to the matching dust file for --physics-variant under --polaris-root.")
    parser.add_argument("--polaris-root", type=Path, default=None,
                        help="POLARIS install directory (for the default --dust-catalog path only).")
    args = parser.parse_args()
    variant = PHYSICS_VARIANTS[args.physics_variant]
    if args.dust_catalog is None:
        if args.polaris_root is None:
            parser.error("--dust-catalog or --polaris-root is required (to locate "
                         f"input/dust_cs/{variant['dust_filename']})")
        args.dust_catalog = args.polaris_root / "input" / "dust_cs" / variant["dust_filename"]

    level = int(np.log2(args.nc))
    if args.nc < 2 or 2 ** level != args.nc:
        parser.error("--nc must be a power of two and at least 2")
    if min(args.side_pc, args.star_photons, args.isrf_photons) <= 0:
        parser.error("Lengths and photon counts must be positive")
    if args.nr_threads != -1 and args.nr_threads < 1:
        parser.error("--nr-threads must be -1 (all cores) or a positive integer")
    if args.crop_side_pc is not None and not (0 < args.crop_side_pc < args.side_pc):
        parser.error("--crop-side-pc must be positive and less than --side-pc")

    snapshot, catalog = args.snapshot.expanduser().resolve(), args.catalog.expanduser().resolve()
    if not snapshot.is_file():
        parser.error(f"Snapshot not found: {snapshot}")
    if not catalog.is_file():
        parser.error(f"Core catalog not found: {catalog}")

    core_row = load_core_row(catalog, args.core_id)
    center_pc = np.array([float(core_row[name]) for name in ("peak_x_pc", "peak_y_pc", "peak_z_pc")])

    output_root = args.output_root.expanduser().resolve()
    core_dir_name = f"core_{args.core_id:06d}"
    output = output_root / core_dir_name
    output.mkdir(parents=True, exist_ok=True)

    print(f"Core {args.core_id} from {catalog.name}: center={center_pc.tolist()} pc, "
          f"catalog mass={core_row.get('mass_selected_gas_Msun')} Msun")

    positions, data = load_gas(snapshot, center_pc, args.side_pc, args.chunk)
    stars = load_stars(snapshot, center_pc, args.side_pc)
    print(f"Loaded {len(positions):,} gas elements and {len(stars)} luminous sink(s) in the {args.side_pc} pc cube")

    side_m = args.side_pc * PC_M
    mass_grid, numer, dx, fallback, boundary = deposit(positions, data, args.nc, side_m)

    occupied = mass_grid > 0
    fields = np.empty((args.nc, args.nc, args.nc, 6), float)
    fields[..., 0] = mass_grid / dx ** 3
    fields[..., 1:] = np.divide(numer, mass_grid[..., None], out=np.zeros_like(numer), where=occupied[..., None])
    empty = ~occupied
    floor = float(fields[..., 0][occupied].min()) * 1e-6 if empty.any() else 0.0
    atmosphere_mass = floor * dx ** 3 * empty.sum()
    mass_grid[occupied] *= (mass_grid.sum() - atmosphere_mass) / mass_grid.sum()
    mass_grid[empty] = floor * dx ** 3
    fields[..., 0] = mass_grid / dx ** 3
    if empty.any():
        nearest = distance_transform_edt(empty, return_distances=False, return_indices=True)
        for j in range(1, 6):
            fields[..., j][empty] = fields[..., j][tuple(nearest[:, empty])]

    true_mass, grid_mass = float(data["Masses"].sum()), float(mass_grid.sum())
    axis = -side_m / 2 + (np.arange(args.nc) + .5) * dx
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_field_ids = (variant["grid_density_field_id"], 2, 3, 4, 5, 6)
    if variant["mu"] is not None:
        # core_0729_polaris_final2's own convention: the experimental branch's
        # GridBasic.cpp defines rho = n * mu * m_H, so field ID 0 must hold
        # number density, not the mass density `fields` otherwise stores
        # throughout (and that all mass-conservation bookkeeping above is
        # computed from).
        encoded_fields = fields.copy()
        encoded_fields[..., 0] = fields[..., 0] / (variant["mu"] * POLARIS_M_H_KG)
    else:
        encoded_fields = fields

    tree = Octree(side_m)
    for xyz, row in zip(np.column_stack((x.ravel(), y.ravel(), z.ravel())), encoded_fields.reshape(-1, 6)):
        tree.insert(xyz, level, row)
    grid_path = output / "grid.dat"
    tree.write(grid_path, grid_field_ids)

    dust_to_gas_ratio = float(np.median(data["DustToGasRatio_Local"]))
    dust_rel = stage_dust(args.dust_catalog.expanduser().resolve(), output)
    write_commands(output_root, core_dir_name, dust_rel, stars, dust_to_gas_ratio, args, variant)
    if stars:
        illumination = "stars_plus_isrf"
        print(f"Wrote polaris.cmd (ISRF-illuminated): {len(stars)} star(s) + ISRF")
    else:
        illumination = "isrf_only"
        print("No luminous sink sources in this cube; wrote polaris.cmd with ISRF as the sole source")

    report = {
        "method": "cell-centre sampled compact cubic SPH kernel; per-particle boundary normalization",
        "snapshot": str(snapshot), "catalog": str(catalog), "core_id": args.core_id,
        "core_catalog_row": core_row, "center_pc": center_pc.tolist(),
        "nc": args.nc, "side_pc": args.side_pc, "crop_side_pc": args.crop_side_pc,
        "n_gas_elements": len(positions),
        "n_stellar_sources": len(stars), "stellar_sources": stars, "illumination": illumination,
        "particle_mass_msun": true_mass / MSUN_KG, "grid_mass_msun": grid_mass / MSUN_KG,
        "mass_error_percent": 100 * (grid_mass / true_mass - 1),
        "particles_using_nearest_cell_fallback": fallback, "particles_with_kernel_clipped_by_cube": boundary,
        "initial_empty_cells": int(empty.sum()), "density_floor_kg_m3": floor,
        "atmosphere_mass_msun": atmosphere_mass / MSUN_KG, "field_ids": list(grid_field_ids),
        "boundary_policy": "renormalize each selected particle's in-cube weights to one",
        "dust_to_gas_ratio": dust_to_gas_ratio,
        "dust_to_gas_ratio_source": "median PartType0/DustToGasRatio_Local in selected cube",
        "star_photons": args.star_photons, "isrf_photons": args.isrf_photons,
        "isrf_strength": args.isrf_strength, "isrf_radius_factor": args.isrf_radius_factor,
        "max_subpixel_lvl": args.max_subpixel_lvl, "distance_pc": args.distance_pc,
        "physics_variant": args.physics_variant,
        "dust_catalog": str(args.dust_catalog),
        "grid_density_units": ("kg/m^3" if variant["mu"] is None else
                                "m^-3; rho = n * mu * 1.66053904e-27 kg"),
        "grid_mu": variant["mu"],
        "n_detectors_per_task": len(WAVELENGTHS_UM) * len(VIEWS),
        "temp_output_filename": variant["temp_output_filename"],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {grid_path}")


if __name__ == "__main__":
    main()
