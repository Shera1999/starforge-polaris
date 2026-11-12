#!/usr/bin/env python3
"""Build the per-sightline STARFORGE + POLARIS training dataset (Parquet).

One row per (box, view, image pixel): 1908 boxes x 3 views x 64 x 64 = 23.4 M sightlines.

Geometry (check_view_geometry.py, check_observer_side.py): grid arrays are [ix, iy, iz], images [row, col].
  view 0 (0, 0):   line of sight z, image +col = +x, +row = +y
  view 1 (-90, 0): line of sight y, image +col = +x, +row = -z
  view 2 (0, 90):  line of sight x, image +col = -z, +row = +y
The high-index end of the line-of-sight axis faces the observer in every view. Profiles are ordered from
the observer: index 00 is the nearest cell, 63 the farthest. Every box is checked: sum(n) ds along the
rotated profiles must reproduce POLARIS's column-density image.

Columns (float32 unless noted; "_log" = log10, NaN where the value is <= 0):
  metadata   snapshot, time_Myr, core_id, box_id, family_id, n_stars, split, view, pix_row, pix_col
  profiles   <q>_00..<q>_63 for q in
               dens [cm^-3] (gas particles, mu = 2.33), dens_dust [g cm^-3] (dust mass, dust-to-gas 0.01),
               temp_polaris [K], temp_starforge [K],
               B_tot [uG], B_pos [uG]                                           (+ _log)
               cos2psi_B, sin2psi_B  (plane-of-sky field orientation psi, from image +col towards +row)
               sin2gamma_B           (B_pos^2 / B_tot^2; 1 = in the sky plane, 0 = along the line of sight)
               B_col, B_row, B_los [uG] (raw components; B_los > 0 points away from the observer)
  sightline  N_col [cm^-2], Sigma [g cm^-2], Sigma_dust [g cm^-2] (+ _log), from POLARIS's column-density image
             dens-weighted means: temp_polaris_mean, temp_starforge_mean, B_tot_mean (+ _log)
             z = sum(n sin^2(gamma) exp(2i psi)): cos2psi_B_mean, sin2psi_B_mean = direction of z,
             B_coherence = |z| / sum(n)  (field tangling along the line of sight x inclination, 0..1)
  bands      <band>_I, <band>_Q, <band>_U [MJy/sr, float64: at 3-8 um they reach 1e-108], <band>_p (fraction), <band>_polangle [deg, E-vector,
             from image +col towards +row, (-90, 90]; the B-field angle is polangle + 90], <band>_I_log,
             <band>_p_log. Band names from wavelength_telescope_mapping.csv; bands <= 25 um are flagged
             unreliable in manifest.json (no stochastic heating of small grains).

Output: OUT/data/<split>/<snapshot>.parquet (one row group per box and view), OUT/splits.csv, OUT/manifest.json.
Read with pyarrow.dataset.dataset(OUT / "data", partitioning=None) or pandas.read_parquet(OUT / "data" / "train", columns=[...]).
Usage: build_dataset.py [--out DIR] [--workers N] [--limit N] [--assign FAMILY=SPLIT ...]
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

HERE = Path(__file__).resolve().parent
GRIDS = HERE.parent / "3.preparing_polaris_grids" / "grids_64"
RUNS = Path("/scratch/sjafaritabar/polaris_runs")
BANDS_CSV = HERE.parents[1] / "wavelength_telescope_mapping.csv"
sys.path.insert(0, str(GRIDS.parent))
from polaris_octree_reader import read_octree_grid

VERSION = "1.2.0"
PC_M = 3.0856775814913673e16
M_U_KG = 1.66053904e-27                     # POLARIS "m_H"
TESLA_TO_UG = 1e10
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 20261111
MAX_EVAL_FAMILY = 50                        # bigger families always go to train
UNRELIABLE_MAX_UM = 25.0
# view -> (line-of-sight grid axis, (image col axis, sign), (image row axis, sign))
VIEWS = {0: (2, (0, 1), (1, 1)), 1: (1, (0, 1), (2, -1)), 2: (0, (2, -1), (1, 1))}
PROFILE_LOG = ("dens", "dens_dust", "temp_polaris", "temp_starforge", "B_tot", "B_pos")
PROFILE_PLAIN = ("cos2psi_B", "sin2psi_B", "sin2gamma_B", "B_col", "B_row", "B_los")


def read_bands():
    rows = list(csv.DictReader(open(BANDS_CSV)))
    names = [r["column_name"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("duplicate band names in " + str(BANDS_CSV))
    return names, np.array([float(r["wavelength_um"]) for r in rows])


def to_view(cube, v):
    """Grid cube [ix, iy, iz] -> [los, row, col], los index 0 = cell nearest the observer."""
    los, (ca, cs), (ra, rs) = VIEWS[v]
    a = np.transpose(cube, (los, ra, ca))
    a = a[::-1]                                              # high grid index faces the observer
    if rs < 0:
        a = a[:, ::-1, :]
    if cs < 0:
        a = a[:, :, ::-1]
    return a


def field_in_view(B, v):
    """B [3, ix, iy, iz] -> (B_col, B_row, B_los) as [los, row, col]; B_los > 0 points away from the observer."""
    los, (ca, cs), (ra, rs) = VIEWS[v]
    return cs * to_view(B[ca], v), rs * to_view(B[ra], v), -to_view(B[los], v)


def assign_splits(family_sizes, fractions=SPLIT_FRACTIONS, seed=SPLIT_SEED, max_eval_family=MAX_EVAL_FAMILY):
    """Whole families to splits, close to the target box fractions. Families with more than max_eval_family
    boxes always go to train, so no single region dominates val or test; families too big for the room left
    in val/test also go to train. Returns {family_id: split}."""
    rng = np.random.default_rng(seed)
    fams = sorted(family_sizes)
    rng.shuffle(fams)
    total = sum(family_sizes.values())
    target = {s: f * total for s, f in fractions.items()}
    count = dict.fromkeys(fractions, 0)
    out = {}
    for f in fams:
        n = family_sizes[f]
        fits_in = [s for s in fractions if s == "train" or (n <= max_eval_family and count[s] + n <= target[s])]
        s = min(fits_in, key=lambda s: count[s] / target[s])
        out[f] = s
        count[s] += n
    return out


def safe_log(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), np.nan)


def box_tables(task):
    """All three views of one box as pyarrow tables (4096 rows each), plus a QA record."""
    key, split, bands, lam_um = task
    run, grid_dir = RUNS / key, GRIDS / key
    summ = json.loads((grid_dir / "summary.json").read_text())
    nc, mu, dist_m = summ["nc"], summ["mu"], summ["distance_pc"] * PC_M
    d2g = summ["dust_to_gas_ratio"]                           # POLARIS <mass_fraction>
    g_per_particle = mu * M_U_KG * 1e3
    cell_m = summ["cell_pc"] * PC_M
    lam_cmd = np.array([float(l.split(">", 1)[1].split()[0]) * 1e6 for l in (run / "polaris.cmd").read_text().splitlines()
                        if l.strip().startswith("<detector_dust")])
    if len(lam_cmd) != 3 * len(bands) or not np.allclose(lam_cmd, np.tile(lam_um, 3), rtol=1e-6):
        raise ValueError(f"{key}: detector wavelengths do not match the band table")
    g = read_octree_grid(grid_dir / "grid.dat")
    n = g["density_number_m3"].astype(np.float64)
    T_sf = g["dust_temperature_K"].astype(np.float64)
    B = np.stack([g["Bx"], g["By"], g["Bz"]]).astype(np.float64) * TESLA_TO_UG
    T_pol = read_octree_grid(run / "temp" / "grid_temp_rat.dat")["dust_temperature_K"].astype(np.float64)
    pix_sr = (cell_m / dist_m) ** 2
    jy_px_to_mjy_sr = 1e-6 / pix_sr
    snap = int(key.split("/")[0].split("_")[1])
    rows, cols = np.divmod(np.arange(nc * nc), nc)
    tables, qa = [], dict(box=key, split=split)
    for v in range(3):
        nv = to_view(n, v)
        bc, br, bl = field_in_view(B, v)
        b_pos2 = bc ** 2 + br ** 2
        b_tot = np.sqrt(b_pos2 + bl ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos2psi = np.where(b_pos2 > 0, (bc ** 2 - br ** 2) / b_pos2, 1.0)
            sin2psi = np.where(b_pos2 > 0, 2 * bc * br / b_pos2, 0.0)
            sin2gamma = np.where(b_tot > 0, b_pos2 / b_tot ** 2, 0.0)
        prof = dict(dens=nv * 1e-6, dens_dust=nv * 1e-6 * g_per_particle * d2g, temp_polaris=to_view(T_pol, v), temp_starforge=to_view(T_sf, v),
                    B_tot=b_tot, B_pos=np.sqrt(b_pos2), cos2psi_B=cos2psi, sin2psi_B=sin2psi,
                    sin2gamma_B=sin2gamma, B_col=bc, B_row=br, B_los=bl)
        data = {}
        data["snapshot"] = pa.array(np.full(nc * nc, snap, np.int16))
        data["time_Myr"] = pa.array(np.full(nc * nc, summ["time_Myr"], np.float32))
        data["core_id"] = pa.array(np.full(nc * nc, summ["core_id"], np.int32))
        data["box_id"] = pa.array(np.full(nc * nc, summ["box_id"], np.int32))
        data["family_id"] = pa.array(np.full(nc * nc, summ["family_id"], np.int32))
        data["n_stars"] = pa.array(np.full(nc * nc, summ["n_stellar_sources"], np.int16))
        data["split"] = pa.array([split] * (nc * nc)).dictionary_encode()
        data["view"] = pa.array(np.full(nc * nc, v, np.int8))
        data["pix_row"] = pa.array(rows.astype(np.int8))
        data["pix_col"] = pa.array(cols.astype(np.int8))

        # sightline quantities
        w = nv.sum(0)
        N_img = fits.getdata(run / "em" / "data" / f"polaris_detector_nr{38 * v + 1:04d}.fits.gz")[5, 0].astype(np.float64)
        rel = np.abs(w * cell_m / N_img - 1).max()
        if rel > 1e-5:
            raise ValueError(f"{key} view {v}: rotated column density differs from POLARIS by {rel:.1e}")
        z = (nv * sin2gamma * (cos2psi + 1j * sin2psi)).sum(0)
        absz = np.abs(z)
        sl = dict(N_col=N_img * 1e-4, Sigma=N_img * 1e-4 * g_per_particle,
                  Sigma_dust=N_img * 1e-4 * g_per_particle * d2g,
                  temp_polaris_mean=(nv * prof["temp_polaris"]).sum(0) / w,
                  temp_starforge_mean=(nv * prof["temp_starforge"]).sum(0) / w,
                  B_tot_mean=(nv * b_tot).sum(0) / w)
        for q, val in sl.items():
            data[q] = pa.array(val.ravel().astype(np.float32))
            data[q + "_log"] = pa.array(safe_log(val).ravel().astype(np.float32))
        with np.errstate(divide="ignore", invalid="ignore"):
            data["cos2psi_B_mean"] = pa.array(np.where(absz > 0, z.real / absz, 1.0).ravel().astype(np.float32))
            data["sin2psi_B_mean"] = pa.array(np.where(absz > 0, z.imag / absz, 0.0).ravel().astype(np.float32))
        data["B_coherence"] = pa.array((absz / w).ravel().astype(np.float32))

        # bands
        n_I_nonpos = {}
        for b, name in enumerate(bands):
            d = fits.getdata(run / "em" / "data" / f"polaris_detector_nr{38 * v + b + 1:04d}.fits.gz")[:3, 0]
            I, Q, U = (d.astype(np.float64) * jy_px_to_mjy_sr)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(I > 0, np.hypot(Q, U) / I, np.nan)
            ang = np.degrees(0.5 * np.arctan2(U, Q))
            ang = np.where(ang <= -90, ang + 180, ang)
            for q, val in ((f"{name}_I", I), (f"{name}_Q", Q), (f"{name}_U", U)):
                data[q] = pa.array(val.ravel())                   # float64: 3-8 um values reach 1e-108
            for q, val in ((f"{name}_p", p), (f"{name}_polangle", ang), (f"{name}_I_log", safe_log(I)),
                           (f"{name}_p_log", safe_log(p))):
                data[q] = pa.array(val.ravel().astype(np.float32))
            n_I_nonpos[name] = int((I <= 0).sum())

        # profiles, nearest cell first
        for q in PROFILE_LOG + PROFILE_PLAIN:
            arr = prof[q].reshape(nc, nc * nc)
            for k in range(nc):
                data[f"{q}_{k:02d}"] = pa.array(arr[k].astype(np.float32))
                if q in PROFILE_LOG:
                    data[f"{q}_{k:02d}_log"] = pa.array(safe_log(arr[k]).astype(np.float32))
        tables.append(pa.table(data))
        qa[f"view{v}_N_rel_err"] = float(rel)
        qa[f"view{v}_I_nonpositive"] = {k: c for k, c in n_I_nonpos.items() if c}
    qa["snapshot"] = key.split("/")[0]
    return key, split, tables, qa


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/scratch/sjafaritabar/starforge_polaris_dataset_v1"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first N boxes only (test)")
    ap.add_argument("--assign", action="append", default=[], metavar="FAMILY=SPLIT",
                    help="move a whole family to a split after the normal assignment, everything else unchanged "
                         "(repeatable; used for the stars-in-evaluation variant)")
    args = ap.parse_args()
    t0 = time.time()
    bands, lam_um = read_bands()
    boxes = []
    for s in sorted(GRIDS.glob("snapshot_*/core_*/summary.json")):
        key = f"{s.parent.parent.name}/{s.parent.name}"
        if not (RUNS / key / "DONE.json").exists():
            raise SystemExit(f"{key}: POLARIS run not finished")
        boxes.append((key, json.loads(s.read_text())["family_id"]))
    sizes = {}
    for _, f in boxes:
        sizes[f] = sizes.get(f, 0) + 1
    fam_split = assign_splits(sizes)
    overrides = {}
    for a in args.assign:
        fam, sp = a.split("=")
        if int(fam) not in sizes or sp not in SPLIT_FRACTIONS:
            raise SystemExit(f"--assign {a}: unknown family or split")
        overrides[int(fam)] = sp
    fam_split.update(overrides)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["box", "family_id", "split"])
        for key, fam in boxes:
            w.writerow([key, fam, fam_split[fam]])
    tasks = [(key, fam_split[fam], bands, lam_um) for key, fam in boxes][:args.limit]

    writers, qa_all, schema = {}, [], None
    with mp.Pool(args.workers) as pool:
        for i, (key, split, tables, qa) in enumerate(pool.imap(box_tables, tasks, chunksize=1), 1):
            path = args.out / "data" / split / f"{key.split('/')[0]}.parquet"
            if path not in writers:
                path.parent.mkdir(parents=True, exist_ok=True)
                schema = schema or tables[0].schema
                writers[path] = pq.ParquetWriter(path, schema, compression="zstd")
            for t in tables:
                writers[path].write_table(t.cast(schema))
            qa_all.append(qa)
            if i % 50 == 0 or i == len(tasks):
                print(f"{datetime.now():%H:%M:%S} {i}/{len(tasks)} boxes", flush=True)
    for wr in writers.values():
        wr.close()

    split_boxes = {s: sum(1 for t in tasks if t[1] == s) for s in SPLIT_FRACTIONS}
    split_fams = {s: sum(1 for f, x in fam_split.items() if x == s) for s in SPLIT_FRACTIONS}
    manifest = dict(
        version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        builder=str(Path(__file__).resolve()), runs=str(RUNS), grids=str(GRIDS),
        n_boxes=len(tasks), n_rows=len(tasks) * 3 * 64 * 64, n_columns=len(schema),
        geometry=dict(views={0: "(0, 0): los z, +col +x, +row +y", 1: "(-90, 0): los y, +col +x, +row -z",
                             2: "(0, 90): los x, +col -z, +row +y"},
                      profile_order="index 00 = cell nearest the observer (high grid index faces the observer)",
                      B_los_sign="positive = pointing away from the observer",
                      angles="psi and polangle from image +col towards +row; B-field angle = polangle + 90 deg"),
        units=dict(dens="cm^-3 gas particles (n = rho / (2.33 m_u))", temp="K", B="microgauss",
                   dens_dust="g cm^-3 dust mass (dust-to-gas mass ratio 0.01, as given to POLARIS)",
                   N_col="cm^-2 (same particles as dens)", Sigma="g cm^-2", Sigma_dust="g cm^-2", I_Q_U="MJy/sr", p="fraction",
                   polangle="degrees"),
        bands=[dict(name=b, wavelength_um=float(l), reliable=bool(l > UNRELIABLE_MAX_UM)) for b, l in zip(bands, lam_um)],
        unreliable_note="bands <= 25 um: POLARIS run without stochastic heating of small grains",
        splits=dict(seed=SPLIT_SEED, target_fractions=SPLIT_FRACTIONS, max_eval_family=MAX_EVAL_FAMILY,
                    overrides={str(k): v for k, v in overrides.items()}, boxes=split_boxes, families=split_fams),
        max_column_density_rel_err=max(q[f"view{v}_N_rel_err"] for q in qa_all for v in range(3)),
        elapsed_s=round(time.time() - t0, 1), qa=qa_all)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(f"done: {len(tasks)} boxes, {manifest['n_rows']} rows, {len(schema)} columns, splits {split_boxes}, "
          f"{manifest['elapsed_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
