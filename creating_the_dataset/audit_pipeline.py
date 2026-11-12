#!/usr/bin/env python3
"""End-to-end audit of the dataset pipeline (steps 1-5), independent of the code that built it.

Every check recomputes what it tests from the raw inputs of that step with its own code (explicit index
formulas, its own union-find, its own unit conversions) and compares with what the pipeline produced.

  A  step 1 -> 2   boxed cores exist in the step-1 catalogues with the same peak, mass and time; boxes in
                   one snapshot never overlap; families re-derived from scratch match family_id
  B  step 2 -> 3   one grid per box; summary IDs, time, peak and centre match boxes.csv; density peak in the
                   central cell; grid mass and physical ranges; stars inside the box, L = R^2 T^4, and the
                   same stars in polaris.cmd
  C  step 3 -> 4   every run finished with the command file of its grid; settings as decided
  D  step 4        POLARIS output for every box: dust temperatures, alignment radii, 114 images, and the
                   absolute 850 um intensity against an optically thin model I = sum B_nu(T) dtau
  E  step 5        every column of randomly drawn sightlines recomputed from grid.dat, the temperature grid
                   and the FITS images; metadata from summary.json; splits re-derived from boxes.csv
  F  dataset       physical ranges and spectral shape over all 23.4 M sightlines, split representativeness

Usage: audit_pipeline.py DATASET_DIR [--workers 16] [--sample-boxes 36]
Writes DATASET_DIR/../audit/audit_report.txt (and prints it).
"""
import argparse
import csv
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from astropy.io import fits

HERE = Path(__file__).resolve().parent
STEP1 = HERE / "1.finding_cores" / "dendrogram_cores_v5.2"
BOXES = HERE / "2.selecting_boxes" / "boxes_2.2pc" / "boxes.csv"
GRIDS = HERE / "3.preparing_polaris_grids" / "grids_64"
RUNS = Path("/scratch/sjafaritabar/polaris_runs")
BANDS_CSV = HERE.parent / "wavelength_telescope_mapping.csv"
sys.path.insert(0, str(GRIDS.parent))
from polaris_octree_reader import read_octree_grid

PC_M, PC_CM = 3.0856775814913673e16, 3.0856775814913673e18
M_U_G = 1.66053904e-24
H, C, K = 6.62607015e-34, 2.99792458e8, 1.380649e-23
NC = 64
EXPECTED = dict(version="3.5.1", isrf_photons=10_000_000, isrf_radius=10.0, star_photons=1_000_000, mu=2.33,
                dust_to_gas=0.01, nc=64, side_pc=2.2, distance_pc=140.0, n_detectors=114)

REPORT = []


def say(line=""):
    print(line, flush=True)
    REPORT.append(line)


def check(name, ok, detail=""):
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def keyof(snapshot, core_id):
    return f"{snapshot}/core_{int(core_id):06d}"


# ---------------------------------------------------------------- A: step 1 -> 2
def audit_step12(boxes):
    say("A. Step 1 -> 2: cores and boxes")
    bad_core, bad_time, n = [], [], 0
    for snap, grp in boxes.groupby("snapshot"):
        cat = pd.read_csv(STEP1 / snap / "core_catalog.csv").set_index("core_id")
        man = json.loads((STEP1 / snap / "run_manifest.json").read_text())
        t = float(man["header"]["Time"]) * man["units"]["time_myr"]
        for _, b in grp.iterrows():
            n += 1
            c = cat.loc[b.core_id]
            if not (np.allclose([c.peak_x_pc, c.peak_y_pc, c.peak_z_pc], [b.peak_x_pc, b.peak_y_pc, b.peak_z_pc], atol=1e-9)
                    and np.isclose(c.mass_selected_gas_Msun, b.mass_selected_gas_Msun, rtol=1e-9)
                    and c.peak_particle_id == b.peak_particle_id):
                bad_core.append(keyof(snap, b.core_id))
            if not np.isclose(t, b.time_Myr, rtol=1e-9):
                bad_time.append(snap)
    check("every boxed core is in its step-1 catalogue with the same peak, peak particle and mass",
          not bad_core, f"{n} boxes, {len(bad_core)} mismatches {bad_core[:3]}")
    check("box times equal the snapshot header times", not bad_time, f"{len(set(bad_time))} snapshots differ")

    man = json.loads((STEP1 / boxes.snapshot.iloc[0] / "run_manifest.json").read_text())
    L = float(man["header"]["BoxSize"]) * man["units"]["length_pc"]
    S = 2.2
    pos = boxes[["peak_x_pc", "peak_y_pc", "peak_z_pc"]].to_numpy()
    overlaps = 0
    for snap, idx in boxes.groupby("snapshot").indices.items():
        p = pos[idx]
        d = np.abs(p[:, None] - p[None])
        d = np.minimum(d, L - d).max(-1)
        overlaps += int(((d < S) & ~np.eye(len(p), dtype=bool)).sum() // 2)
    check("no two boxes of the same snapshot overlap (periodic Chebyshev distance >= 2.2 pc)", overlaps == 0,
          f"{overlaps} overlapping pairs")

    # families from scratch: link overlapping boxes of consecutive snapshots that have boxes
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    snaps = sorted(boxes.snapshot_number.unique())
    by = {s: np.flatnonzero(boxes.snapshot_number.to_numpy() == s) for s in snaps}
    for a, b in zip(snaps[:-1], snaps[1:]):
        d = np.abs(pos[by[a]][:, None] - pos[by[b]][None])
        d = np.minimum(d, L - d).max(-1)
        for i, j in zip(*np.nonzero(d < S)):
            parent[find(by[a][i])] = find(by[b][j])
    roots = np.array([find(i) for i in range(len(boxes))])
    pairs = pd.crosstab(roots, boxes.family_id.to_numpy())
    same = (pairs.gt(0).sum(axis=1) == 1).all() and (pairs.gt(0).sum(axis=0) == 1).all()
    check("families re-derived from scratch are identical to family_id", same,
          f"{len(set(roots))} re-derived vs {boxes.family_id.nunique()} recorded")
    return L


# ---------------------------------------------------------------- B, C, D: per box (parallel)
def planck_si(lam_m, T):
    nu = C / lam_m
    with np.errstate(over="ignore"):
        return 2 * H * nu ** 3 / C ** 2 / np.expm1(H * nu / (K * T))


def per_box(args):
    key, brow = args
    out = dict(box=key, problems=[])
    P = out["problems"]
    g_dir, r_dir = GRIDS / key, RUNS / key
    s = json.loads((g_dir / "summary.json").read_text())
    # B: identity and geometry
    for k, v in (("core_id", brow["core_id"]), ("box_id", brow["box_id"]), ("family_id", brow["family_id"])):
        if int(s[k]) != int(v):
            P.append(f"summary {k} {s[k]} != boxes.csv {v}")
    if s["snapshot"] != brow["snapshot"] or not np.isclose(s["time_Myr"], brow["time_Myr"], rtol=1e-9):
        P.append("snapshot/time differ from boxes.csv")
    peak = np.array([brow["peak_x_pc"], brow["peak_y_pc"], brow["peak_z_pc"]])
    if not np.allclose(s["peak_pc"], peak, atol=1e-9):
        P.append("peak_pc differs from boxes.csv")
    if not np.allclose(np.array(s["grid_center_pc"]) + 0.5 * s["cell_pc"], peak, atol=1e-9):
        P.append("grid centre is not peak - half a cell")
    for k, v in (("version", EXPECTED["version"]), ("isrf_photons_per_wavelength", EXPECTED["isrf_photons"]),
                 ("isrf_radius_factor", EXPECTED["isrf_radius"]), ("star_photons_per_wavelength", EXPECTED["star_photons"]),
                 ("mu", EXPECTED["mu"]), ("dust_to_gas_ratio", EXPECTED["dust_to_gas"]), ("nc", EXPECTED["nc"]),
                 ("side_pc", EXPECTED["side_pc"]), ("distance_pc", EXPECTED["distance_pc"]),
                 ("n_detectors", EXPECTED["n_detectors"])):
        if s[k] != v:
            P.append(f"summary {k} = {s[k]}, expected {v}")
    g = read_octree_grid(g_dir / "grid.dat")
    n, Tsf = g["density_number_m3"].astype(float), g["dust_temperature_K"].astype(float)
    B = np.stack([g["Bx"], g["By"], g["Bz"]]).astype(float)
    if n.shape != (NC,) * 3:
        P.append(f"grid shape {n.shape}")
    # the box's own core peak must sit in cell 32 (grid centre + half a cell, checked above); the densest
    # cell of the box may be elsewhere (a denser neighbouring core, or kernel deposition): information only
    cell_of_peak = np.floor((peak - (np.array(s["grid_center_pc"]) - 0.5 * s["side_pc"])) / s["cell_pc"]).astype(int)
    if tuple(cell_of_peak) != (32, 32, 32) or tuple(s["peak_cell_index"]) != (32, 32, 32):
        P.append(f"core peak in cell {tuple(cell_of_peak)}")
    pk = np.unravel_index(np.argmax(n), n.shape)
    out["densest_cell_offset"] = int(np.max(np.abs(np.array(pk) - 32)))
    out["centre_over_max"] = float(n[32, 32, 32] / n.max())
    cell_m = s["cell_pc"] * PC_M
    mass = n.sum() * cell_m ** 3 * s["mu"] * M_U_G * 1e-3 / 1.98847e30
    if not np.isclose(mass, s["grid_mass_Msun"], rtol=1e-4):
        P.append(f"grid mass {mass:.3f} != summary {s['grid_mass_Msun']:.3f} Msun")
    if not (np.all(n > 0) and np.all(np.isfinite(B)) and Tsf.min() > 2 and Tsf.max() < 2000):
        P.append("grid has non-positive density, non-finite B or odd STARFORGE temperatures")
    out["B_uG_max"] = float(np.sqrt((B ** 2).sum(0)).max() * 1e10)
    # stars
    cmd = (g_dir / "polaris.cmd").read_text()
    star_lines = [l.split(">", 1)[1].split() for l in cmd.splitlines() if l.strip().startswith("<source_star")]
    used = [st for st in s["stars"] if st["used_as_source"]]
    if len(star_lines) != len(used) or len(used) != s["n_stellar_sources"]:
        P.append(f"{len(star_lines)} stars in polaris.cmd, {len(used)} used in summary, n_stellar_sources {s['n_stellar_sources']}")
    for st, line in zip(used, star_lines):
        x = np.array(st["position_m"])
        if np.abs(x).max() > 0.5 * s["side_pc"] * PC_M * (1 + 1e-9):
            P.append("star outside the box")
        if not np.allclose([float(v) for v in line[:5]], [*x, st["radius_rsun"], st["temperature_K"]], rtol=1e-6, atol=0):
            P.append("polaris.cmd star line differs from summary")
        L_rt = st["radius_rsun"] ** 2 * (st["temperature_K"] / 5772.0) ** 4
        if not np.isclose(L_rt, st["luminosity_lsun"], rtol=2e-3):
            P.append(f"star L {st['luminosity_lsun']:.3g} != R^2 T^4 {L_rt:.3g}")
    out["n_stars"] = len(used)
    # C: run
    done = r_dir / "DONE.json"
    if not done.exists() or (r_dir / "FAILED.json").exists():
        P.append("run not DONE")
        return out
    if (r_dir / "polaris.cmd").read_text() != cmd:
        P.append("run polaris.cmd differs from the grid's")
    for frag in ('<source_isrf nr_photons = "10000000"> 1 10\n', "<mu> 2.33", "<mass_fraction> 0.01\n",
                 '"plaw" 1.0 0 5e-09 2.5e-07 -3.5', "<max_subpixel_lvl> 0", "astrodust_fe_0.0_sus.dat"):
        if frag not in cmd:
            P.append(f"polaris.cmd lacks {frag.strip()}")
    out["run_min"] = json.loads(done.read_text())["seconds"] / 60
    # D: POLARIS output
    t = read_octree_grid(r_dir / "temp" / "grid_temp_rat.dat")
    Tp = t["dust_temperature_K"].astype(float)
    if not np.allclose(t["density_number_m3"], g["density_number_m3"], rtol=1e-6):
        P.append("POLARIS changed the density")
    if not (np.all(np.isfinite(Tp)) and Tp.min() > 2.7 and Tp.max() < 1500):
        P.append(f"POLARIS dust temperature range {Tp.min():.3g}-{Tp.max():.3g} K")
    aal = t["a_alg_m"].astype(float)
    if not (np.all(np.isfinite(aal)) and aal.min() >= 5e-9 * 0.999 and aal.max() <= 2.5e-7 * 1.001):
        P.append(f"a_alig outside the size range: {aal.min():.3g}-{aal.max():.3g} m")
    out["T_pol_minus_sf_median"] = float(np.median(Tp - Tsf))
    out["T_pol_range"] = (float(Tp.min()), float(Tp.max()))
    nimg = len(list((r_dir / "em" / "data").glob("polaris_detector_nr*[0-9].fits.gz")))
    if nimg != 114:
        P.append(f"{nimg} images")
    # absolute 850 um check, view 0 (line of sight z), optically thin
    lam = [float(l.split(">", 1)[1].split()[0]) for l in cmd.splitlines() if l.strip().startswith("<detector_dust")]
    k850 = int(np.argmin(np.abs(np.array(lam[:38]) - 850e-6)))
    d = fits.getdata(r_dir / "em" / "data" / f"polaris_detector_nr{k850 + 1:04d}.fits.gz")[:, 0].astype(float)
    I_jy, tau = d[0], d[4]
    nz = np.transpose(n, (2, 1, 0))                 # [iz, iy(row), ix(col)]
    Tz = np.transpose(Tp, (2, 1, 0))
    dtau = tau[None] * nz / nz.sum(0)[None]
    I_model = (planck_si(lam[k850], Tz) * dtau).sum(0) / 1e-20          # MJy/sr
    I_data = I_jy * 1e-6 / (cell_m / (s["distance_pc"] * PC_M)) ** 2   # Jy/px -> MJy/sr
    ratio = I_data / I_model
    out["I850_ratio_median"] = float(np.median(ratio))
    out["I850_ratio_spread"] = float(np.percentile(ratio, 95) / np.percentile(ratio, 5))
    out["tau850_max"] = float(tau.max())
    return out


def audit_boxes(boxes, workers):
    say("B/C/D. Steps 2 -> 3 -> 4: grids, commands and POLARIS output, every box")
    rows = [(keyof(b.snapshot, b.core_id), b._asdict()) for b in boxes.itertuples(index=False)]
    with mp.Pool(workers) as pool:
        res = pool.map(per_box, rows, chunksize=4)
    grid_dirs = {f"{p.parent.parent.name}/{p.parent.name}" for p in GRIDS.glob("snapshot_*/core_*/summary.json")}
    check("exactly one grid per box in boxes.csv", grid_dirs == {r for r, _ in rows},
          f"{len(grid_dirs)} grids, {len(rows)} boxes")
    probs = defaultdict(list)
    for r in res:
        for p in r["problems"]:
            probs[p.split(":")[0].split(" = ")[0][:60]].append(r["box"])
    check("summaries, grids, stars, commands, runs and POLARIS outputs consistent for every box", not probs,
          "; ".join(f"{k} ({len(v)} boxes, e.g. {v[0]})" for k, v in list(probs.items())[:8]))
    r = pd.DataFrame([x for x in res if "I850_ratio_median" in x])
    off = r.densest_cell_offset
    say(f"       (info) densest cell = the core-peak cell in {(off == 0).sum()} boxes; 1-3 cells away in "
        f"{off.between(1, 3).sum()}; > 3 cells (a denser neighbouring core) in {(off > 3).sum()}; "
        f"central/max density median {r.centre_over_max.median():.2f}")
    say(f"       stars: {int(r.n_stars.sum())} sources in {int((r.n_stars > 0).sum())} boxes; "
        f"run time median {r.run_min.median():.0f} min; |B| max {max(x['B_uG_max'] for x in res):.0f} uG")
    tl = np.array([x["T_pol_range"][0] for x in r.to_dict("records")]); th = np.array([x["T_pol_range"][1] for x in r.to_dict("records")])
    say(f"       POLARIS dust temperature over all boxes {tl.min():.1f}-{th.max():.1f} K; "
        f"POLARIS - STARFORGE per-box median {r.T_pol_minus_sf_median.median():+.2f} K "
        f"(5-95%: {r.T_pol_minus_sf_median.quantile(0.05):+.2f} to {r.T_pol_minus_sf_median.quantile(0.95):+.2f})")
    out_T = r[np.abs(r.T_pol_minus_sf_median - r.T_pol_minus_sf_median.median()) > 5 * r.T_pol_minus_sf_median.std()]
    check("no box with an outlying POLARIS-vs-STARFORGE temperature offset (> 5 sigma)", out_T.empty,
          f"{len(out_T)} boxes {out_T.box.tolist()[:3]}")
    med = r.I850_ratio_median
    check("850 um intensity in MJy/sr matches the optically thin model sum B_nu(T) dtau (units, pixel size, distance)",
          bool(((med > 0.8) & (med < 1.25)).all()),
          f"POLARIS/model per-box median {med.median():.3f} (range {med.min():.3f}-{med.max():.3f}), "
          f"within-box 95/5% spread median {r.I850_ratio_spread.median():.3f}, tau850 max {r.tau850_max.max():.2f}")
    return res


# ---------------------------------------------------------------- E: dataset vs raw files
def expected_sightlines(key, view, pix, s, bands, lam_um):
    """Recompute every column for the given (row, col) pixels from raw files, with explicit index formulas."""
    g = read_octree_grid(GRIDS / key / "grid.dat")
    t = read_octree_grid(RUNS / key / "temp" / "grid_temp_rat.dat")
    n = g["density_number_m3"].astype(float) * 1e-6                     # cm^-3
    Tsf, Tp = g["dust_temperature_K"].astype(float), t["dust_temperature_K"].astype(float)
    Bx, By, Bz = (g[c].astype(float) * 1e10 for c in ("Bx", "By", "Bz"))   # T -> uG
    k = np.arange(NC)
    m = NC - 1
    rows = []
    for (r, c) in pix:
        # cell index along the line of sight, nearest the observer first (high grid index faces the observer)
        if view == 0:
            ix, iy, iz = np.full(NC, c), np.full(NC, r), m - k
            bcol, brow, blos = Bx, By, -Bz
        elif view == 1:
            ix, iy, iz = np.full(NC, c), m - k, np.full(NC, m - r)
            bcol, brow, blos = Bx, -Bz, -By
        else:
            ix, iy, iz = m - k, np.full(NC, r), np.full(NC, m - c)
            bcol, brow, blos = -Bz, By, -Bx
        sel = (ix, iy, iz)
        bc, br, bl = bcol[sel], brow[sel], blos[sel]
        bpos2 = bc ** 2 + br ** 2
        btot = np.sqrt(bpos2 + bl ** 2)
        e = {"snapshot": int(key.split("/")[0][9:]), "core_id": s["core_id"], "box_id": s["box_id"],
             "family_id": s["family_id"], "n_stars": s["n_stellar_sources"], "view": view, "pix_row": r,
             "pix_col": c, "time_Myr": s["time_Myr"]}
        prof = {"dens": n[sel], "dens_dust": n[sel] * s["mu"] * M_U_G * s["dust_to_gas_ratio"], "temp_polaris": Tp[sel],
                "temp_starforge": Tsf[sel], "B_tot": btot, "B_pos": np.sqrt(bpos2), "B_col": bc, "B_row": br, "B_los": bl,
                "cos2psi_B": np.cos(2 * np.arctan2(br, bc)), "sin2psi_B": np.sin(2 * np.arctan2(br, bc)),
                "sin2gamma_B": bpos2 / btot ** 2}
        for q, v in prof.items():
            for kk in range(NC):
                e[f"{q}_{kk:02d}"] = v[kk]
                if q in ("dens", "dens_dust", "temp_polaris", "temp_starforge", "B_tot", "B_pos"):
                    e[f"{q}_{kk:02d}_log"] = np.log10(v[kk])
        w = n[sel]
        psi = np.arctan2(br, bc)
        z = (w * bpos2 / btot ** 2 * np.exp(2j * psi)).sum()
        cell_cm = s["cell_pc"] * PC_CM
        N = w.sum() * cell_cm
        sl = {"N_col": N, "Sigma": N * s["mu"] * M_U_G, "Sigma_dust": N * s["mu"] * M_U_G * s["dust_to_gas_ratio"],
              "temp_polaris_mean": (w * Tp[sel]).sum() / w.sum(), "temp_starforge_mean": (w * Tsf[sel]).sum() / w.sum(),
              "B_tot_mean": (w * btot).sum() / w.sum()}
        for q, v in sl.items():
            e[q], e[q + "_log"] = v, np.log10(v)
        e["cos2psi_B_mean"], e["sin2psi_B_mean"], e["B_coherence"] = np.cos(np.angle(z)), np.sin(np.angle(z)), abs(z) / w.sum()
        omega = (s["cell_pc"] / s["distance_pc"]) ** 2                     # sr per pixel
        for b, name in enumerate(bands):
            d = fits.getdata(RUNS / key / "em" / "data" / f"polaris_detector_nr{38 * view + b + 1:04d}.fits.gz")
            I, Q, U = (float(d[p, 0, r, c]) / omega / 1e6 for p in range(3))
            p = np.hypot(Q, U) / I
            a = np.degrees(0.5 * np.arctan2(U, Q))
            e.update({f"{name}_I": I, f"{name}_Q": Q, f"{name}_U": U, f"{name}_p": p, f"{name}_polangle": a,
                      f"{name}_I_log": np.log10(I), f"{name}_p_log": np.log10(p)})
        rows.append(e)
    return rows


def audit_dataset_columns(root, boxes, n_boxes, seed=7):
    say("E. Step 5: every column of random sightlines recomputed from the raw files")
    bands = [r["column_name"] for r in csv.DictReader(open(BANDS_CSV))]
    lam = np.array([float(r["wavelength_um"]) for r in csv.DictReader(open(BANDS_CSV))])
    splits = pd.read_csv(root / "splits.csv").set_index("box")
    rng = np.random.default_rng(seed)
    names = ds.dataset(root / "data", format="parquet", partitioning=None).schema.names
    pick = []
    for sp in ("train", "val", "test"):
        keys = splits.index[splits.split == sp].tolist()
        pick += [keys[i] for i in rng.choice(len(keys), n_boxes // 3, replace=False)]
    worst = defaultdict(float)
    n_compared = n_values = 0
    missing = set()
    for key in pick:
        s = json.loads((GRIDS / key / "summary.json").read_text())
        f = pq.read_table(root / "data" / splits.loc[key, "split"] / f"{key.split('/')[0]}.parquet",
                          filters=[("core_id", "=", s["core_id"])]).to_pandas()
        if len(f) != 3 * NC * NC:
            say(f"  [FAIL] {key}: {len(f)} rows in its split file")
            continue
        for view in range(3):
            pix = [tuple(p) for p in rng.integers(0, NC, (6, 2))] + [(32, 32), (0, 0), (63, 63), (0, 63)]
            for e in expected_sightlines(key, view, pix, s, bands, lam):
                row = f[(f.view == view) & (f.pix_row == e["pix_row"]) & (f.pix_col == e["pix_col"])]
                if len(row) != 1:
                    worst["row lookup"] = np.inf
                    continue
                row = row.iloc[0]
                n_compared += 1
                for col, want in e.items():
                    if col not in row.index:
                        missing.add(col)
                        continue
                    got = float(row[col])
                    n_values += 1
                    if col.endswith("polangle"):                      # degrees, mod 180
                        err = abs((got - want + 90) % 180 - 90) / 180
                    elif col.endswith("_log") or "2psi" in col or "2gamma" in col or col == "B_coherence":
                        err = abs(got - want)                             # dex / dimensionless
                    elif col.endswith(("_Q", "_U")):                      # signed: relative to I of the band
                        err = abs(got - want) / e[col[:-2] + "_I"]
                    elif col.startswith(("B_col_", "B_row_", "B_los_")):  # signed: relative to |B| of the cell
                        err = abs(got - want) / e["B_tot_" + col[-2:]]
                    else:
                        err = abs(got - want) / abs(want) if want != 0 else abs(got)
                    group = col.rsplit("_", 1)[0] if col[-2:].isdigit() else col
                    for b in bands:
                        if col.startswith(b + "_"):
                            group = "band_" + col[len(b) + 1:]
                    worst[group] = max(worst[group], err)
        # metadata consistency for the whole box
        if not ((f.split.astype(str) == splits.loc[key, "split"]).all() and (f.family_id == s["family_id"]).all()):
            worst["split/family columns"] = np.inf
    extra = set(names) - set(e) - {"split"}
    check("columns: every dataset column was recomputed and every recomputed column exists",
          not missing and not extra, f"missing {sorted(missing)[:5]}, not recomputed {sorted(extra)[:5]}")
    tol = 2e-4
    bad = {k: v for k, v in worst.items() if not v <= tol}
    check(f"{n_compared} sightlines from {len(pick)} boxes ({n_values} values) agree with the raw files",
          not bad, f"worst relative/absolute error {max(worst.values()):.1e} (float32)" + (f"; FAILING {bad}" if bad else ""))
    say("       worst error per column group: " + ", ".join(f"{k} {v:.0e}" for k, v in sorted(worst.items(), key=lambda x: -x[1])[:8]))


def audit_splits(root, boxes):
    say("E2. Splits")
    sp = pd.read_csv(root / "splits.csv")
    sp["key"] = sp.box
    boxes = boxes.assign(key=[keyof(a, b) for a, b in zip(boxes.snapshot, boxes.core_id)])
    m = sp.merge(boxes[["key", "family_id", "time_Myr", "n_sinks_within_2p2pc_box"]], on="key", suffixes=("", "_step2"))
    check("splits.csv covers every box once, with step-2 family IDs", len(m) == len(boxes) == sp.box.nunique()
          and (m.family_id == m.family_id_step2).all())
    check("no family in more than one split", (m.groupby("family_id").split.nunique() == 1).all())
    fs = m.groupby("family_id").agg(n=("key", "size"), split=("split", "first"))
    check("families with more than 50 boxes are all in train", (fs[fs.n > 50].split == "train").all(),
          f"{(fs.n > 50).sum()} large families")
    for s in ("train", "val", "test"):
        x = m[m.split == s]
        f = fs[fs.split == s]
        say(f"       {s:5s}: {len(x)} boxes, {len(f)} families (largest {f.n.max()}, {f.n.max() / len(x):.0%}), "
            f"time {x.time_Myr.min():.2f}-{x.time_Myr.max():.2f} Myr (median {x.time_Myr.median():.2f}), "
            f"boxes with sinks {(x.n_sinks_within_2p2pc_box > 0).sum()}")


# ---------------------------------------------------------------- F: whole-dataset physics
def audit_ranges(root):
    say("F. Whole dataset: physical ranges and spectra (all sightlines)")
    man = json.loads((root / "manifest.json").read_text())
    bands = [b for b in man["bands"]]
    long = [b for b in bands if b["wavelength_um"] >= 50]
    cols = (["N_col", "temp_polaris_mean", "temp_starforge_mean", "B_tot_mean", "B_coherence", "dens_32", "temp_polaris_32",
             "B_tot_32"] + [f"{b['name']}_I" for b in long] + [f"{long[i]['name']}_p" for i in range(len(long))])
    d = ds.dataset(root / "data", format="parquet", partitioning=None)
    acc = defaultdict(list)
    for batch in d.to_batches(columns=cols, batch_size=1 << 20):
        t = batch.to_pandas()
        for c in cols[:8]:
            acc[c].append(np.percentile(t[c], [0, 1, 50, 99, 100]))
        I = t[[f"{b['name']}_I" for b in long]].to_numpy()
        acc["peak_um"].append(np.array([long[i]["wavelength_um"] for i in I.argmax(1)]))
        P = t[[f"{b['name']}_p" for b in long]].to_numpy()
        acc["p_max"].append(np.array([P.max()]))
        acc["p_gt_0.3"].append(np.array([(P > 0.3).sum()]))
    def span(c, unit, fmt="{:.3g}"):
        a = np.array(acc[c])
        return f"{c}: min {fmt.format(a[:, 0].min())}, median ~{fmt.format(np.median(a[:, 2]))}, max {fmt.format(a[:, 4].max())} {unit}"
    for c, u in (("N_col", "cm^-2"), ("dens_32", "cm^-3"), ("temp_polaris_mean", "K"), ("temp_starforge_mean", "K"),
                 ("temp_polaris_32", "K"), ("B_tot_mean", "uG"), ("B_tot_32", "uG"), ("B_coherence", "")):
        say("       " + span(c, u))
    peak = np.concatenate(acc["peak_um"])
    vals, cnt = np.unique(peak, return_counts=True)
    say("       wavelength of peak intensity (>= 50 um bands): " +
        ", ".join(f"{v:g} um {c / len(peak):.1%}" for v, c in zip(vals, cnt) if c / len(peak) > 0.005))
    pmax, p30 = max(a[0] for a in acc["p_max"]), sum(int(a[0]) for a in acc["p_gt_0.3"])
    check("polarization fraction physical (all <= 0.5) at >= 50 um", pmax <= 0.5, f"max {pmax:.3f}; {p30} values > 0.3")
    check("spectra peak in the far-infrared (60-500 um) for > 99% of sightlines",
          ((peak >= 60) & (peak <= 500)).mean() > 0.99, f"{((peak >= 60) & (peak <= 500)).mean():.2%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample-boxes", type=int, default=36)
    args = ap.parse_args()
    boxes = pd.read_csv(BOXES)
    say(f"Pipeline audit, dataset {args.dataset}")
    say(f"{len(boxes)} boxes in step 2, {boxes.family_id.nunique()} families\n")
    audit_step12(boxes)
    say()
    audit_boxes(boxes, args.workers)
    say()
    audit_dataset_columns(args.dataset, boxes, args.sample_boxes)
    audit_splits(args.dataset, boxes)
    say()
    audit_ranges(args.dataset)
    n_fail = sum(1 for l in REPORT if "[FAIL]" in l)
    n_pass = sum(1 for l in REPORT if "[PASS]" in l)
    say(f"\n{n_pass} checks passed, {n_fail} failed")
    out = args.dataset.parent / "audit"
    out.mkdir(exist_ok=True)
    (out / "audit_report.txt").write_text("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
