#!/usr/bin/env python3
"""Tests for prepare_grids.py. Synthetic tests always run; real-data tests run
when the corresponding grids exist.

Run: python test_prepare_grids.py   (exits non-zero on any failure)
"""
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_grids as pg
from octree import Octree
from polaris_octree_reader import read_octree_grid

OLD = HERE / "legacy_reference"   # old deposit code + old snapshot_440 grids (copies of the pre-v5.2 pipeline)


def lattice(side, nc, spacing_cells=0.8, h_cells=2.0, margin_cells=3.0):
    """Equal-mass particles on a cubic lattice extending beyond the cube."""
    dx = side / nc
    s = spacing_cells * dx
    ext = side / 2 + margin_cells * dx
    axis = np.arange(-ext, ext + s / 2, s) + 0.37 * s   # offset so no particle sits on a cell boundary
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    pos = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    n = len(pos)
    return pos, np.full(n, h_cells * dx), np.full(n, 1.0), s


def face_profile(rho):
    """Mean density of layer k from the faces, k = 0..3."""
    nc = rho.shape[0]
    out = []
    for k in range(4):
        m = np.zeros(rho.shape, bool)
        m[k:nc - k, k:nc - k, k:nc - k] = True
        m[k + 1:nc - k - 1, k + 1:nc - k - 1, k + 1:nc - k - 1] = False
        out.append(rho[m].mean())
    return np.array(out)


def test_uniform_medium_is_uniform_to_the_faces_and_old_method_was_not():
    """New deposit is exact at the faces for small and large kernels; the old
    one (centre-inside particles, clipped kernels renormalised inwards) loses
    face mass, badly so for kernels several cells wide."""
    sys.path.insert(0, str(OLD))
    from polaris_prepare_core import deposit as old_deposit
    side, nc = 1.0, 16
    for h_cells, spacing in ((2.0, 0.8), (6.0, 1.5)):
        pos, h, m, s = lattice(side, nc, spacing_cells=spacing, h_cells=h_cells, margin_cells=h_cells + 1)
        expected = 1.0 / s ** 3
        mass = pg.deposit(pos, h, m, np.ones((len(m), 1)), nc, side)[0]
        prof = face_profile(mass / (side / nc) ** 3) / expected
        assert np.all(np.abs(prof - 1) < 0.01), f"h={h_cells}: new face profile {prof}"
        inside = np.all(np.abs(pos) < side / 2, axis=1)
        data = dict(Dust_Temperature=np.ones(inside.sum()), Temperature=np.ones(inside.sum()),
                    MagneticField=np.zeros((inside.sum(), 3)), SmoothingLength=h[inside], Masses=m[inside])
        old_prof = face_profile(old_deposit(pos[inside], data, nc, side)[0] / (side / nc) ** 3) / expected
        print(f"    h={h_cells:g} cells, face-layer density / true: new {prof[0]:.3f}, old {old_prof[0]:.3f}")
    assert old_prof[0] < 0.7, f"expected the old face deficit for large kernels, got {old_prof}"


def test_particle_centred_on_face_puts_half_its_mass_inside():
    side, nc = 1.0, 8
    pos = np.array([[side / 2, 0.01, -0.02]])            # exactly on the +x face
    mass, _, frac, _ = pg.deposit(pos, np.array([0.3]), np.array([2.0]), np.ones((1, 1)), nc, side)
    assert math.isclose(frac[0], 0.5, rel_tol=1e-12) and math.isclose(mass.sum(), 1.0, rel_tol=1e-12)


def test_mass_accounting_and_field_averaging():
    rng = np.random.default_rng(1)
    side, nc, n = 1.0, 8, 500
    pos = rng.uniform(-0.7, 0.7, (n, 3))
    h = rng.uniform(0.005, 0.4, n)
    m = rng.uniform(0.5, 2, n)
    vals = np.column_stack((np.full(n, 12.0), np.full(n, 30.0), np.tile([1e-9, -2e-9, 3e-9], (n, 1))))
    mass, weighted, frac, _ = pg.deposit(pos, h, m, vals, nc, side)
    assert math.isclose(mass.sum(), np.sum(m * frac), rel_tol=1e-12)
    assert np.all((frac >= 0) & (frac <= 1 + 1e-12))
    assert np.all(frac[np.all(np.abs(pos) + h[:, None] < side / 2, axis=1)] > 1 - 1e-12)   # fully inside -> all mass
    fields, n_empty, _ = pg.finalize_fields(mass, weighted, side / nc)
    for j, v in enumerate([12.0, 30.0, 1e-9, -2e-9, 3e-9]):
        np.testing.assert_allclose(fields[..., 1 + j], v, rtol=1e-12)


def test_small_particles_land_in_their_cell_or_nowhere():
    side, nc = 1.0, 8
    pos = np.array([[0.30, -0.40, 0.45], [0.55, 0.0, 0.0]])  # second is outside the cube
    mass, _, frac, fb = pg.deposit(pos, np.array([1e-4, 1e-4]), np.array([1.0, 1.0]), np.ones((2, 1)), nc, side)
    assert fb == 2 and list(frac) == [1.0, 0.0]
    np.testing.assert_array_equal(np.argwhere(mass > 0)[0], np.floor((pos[0] + 0.5) / (side / nc)).astype(int))


def test_empty_cells_get_floor_and_neighbour_fields():
    mass = np.zeros((4, 4, 4)); mass[0, 0, 0] = 2.0
    weighted = np.zeros((4, 4, 4, 1)); weighted[0, 0, 0, 0] = 2.0 * 15.0
    fields, n_empty, floor_mass = pg.finalize_fields(mass, weighted, 1.0)
    assert n_empty == 63 and np.all(fields[..., 1] == 15.0)
    assert math.isclose(fields[1, 1, 1, 0], 2.0e-6) and math.isclose(floor_mass, 63 * 2e-6)


def test_octree_roundtrip_maps_position_to_index():
    nc, side = 8, 2.0
    axis = -side / 2 + (np.arange(nc) + 0.5) * side / nc
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    fields = np.stack([x, y, z, x, y, z], axis=-1)
    with tempfile.TemporaryDirectory() as d:
        pg.write_grid(Path(d) / "g.dat", fields, side)
        g = read_octree_grid(Path(d) / "g.dat")
    np.testing.assert_allclose(g["density_number_m3"], x, atol=1e-6)
    assert "density_kg_m3" not in g                     # no mass density without an explicit mu
    np.testing.assert_allclose(g["dust_temperature_K"], y, atol=1e-6)
    np.testing.assert_allclose(g["gas_temperature_K"], z, atol=1e-6)
    assert g["field_ids"] == pg.GRID_FIELD_IDS


def fake_snapshot(path, n=20000, seed=2):
    """Tiny STARFORGE-like snapshot: code length = 1.0014 pc, box 10 code units."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header").attrs
        h["UnitLength_In_CGS"], h["UnitMass_In_CGS"], h["UnitVelocity_In_CGS"] = 3.09e18, 1.99e33, 100.0
        h["BoxSize"], h["Time"] = 10.0, 0.001
        g = f.create_group("PartType0")
        g["Coordinates"] = rng.uniform(0, 10, (n, 3)).astype(np.float32)
        g["SmoothingLength"] = rng.uniform(0.01, 0.6, n).astype(np.float32)
        g["Masses"] = np.full(n, 1e-3, np.float32)
        g["Dust_Temperature"] = rng.uniform(8, 20, n).astype(np.float32)
        g["Temperature"] = rng.uniform(8, 30, n).astype(np.float32)
        g["MagneticField"] = rng.normal(0, 1e-9, (n, 3)).astype(np.float32)
        g["DustToGasRatio_Local"] = np.full(n, 0.0142, np.float32)
        s = f.create_group("PartType5")
        s["Coordinates"] = np.array([[0.2, 5.0, 5.0], [5.0, 5.0, 5.0]], np.float32)
        s["ParticleIDs"] = np.array([7, 8], np.uint32)
        s["ProtoStellarRadius_inSolar"] = np.array([1.0, 2.0], np.float32)
        s["StarLuminosity_Solar"] = np.array([1.0, 0.0], np.float32)
        s["Masses"] = np.array([0.5, 0.1], np.float32)


def test_cut_boxes_matches_bruteforce_including_periodic_wrap():
    with tempfile.TemporaryDirectory() as d:
        snap = Path(d) / "snap.hdf5"
        fake_snapshot(snap)
        L = 3.09e18 / (100 * pg.PC_M)                       # pc per code length
        centers = [[0.3 * L, 5 * L, 9.8 * L], [5 * L, 5 * L, 5 * L]]   # first box wraps in x and z
        side = 2.2
        boxes, u = pg.cut_boxes(snap, centers, side, chunk=3000)
        with h5py.File(snap) as f:
            xyz = f["PartType0/Coordinates"][:].astype(float)
            h = f["PartType0/SmoothingLength"][:].astype(float)
        for c, b in zip(centers, boxes):
            rel = pg.periodic_delta(xyz, np.array(c) / L, 10.0)
            sel = np.all(np.abs(rel) < side / (2 * L) + h[:, None], axis=1)
            assert len(b["Masses"]) == sel.sum()
            got = np.sort(b["pos"][:, 0])
            np.testing.assert_allclose(got, np.sort(rel[sel, 0] * 3.09e16), rtol=1e-12)
            assert b["inside"].sum() == np.all(np.abs(rel) < side / (2 * L), axis=1).sum()
            np.testing.assert_allclose(b["Masses"], 1e-3 * 1.99e30, rtol=1e-6)
        assert np.any(boxes[0]["pos"][:, 0] < 0) and np.any(boxes[0]["pos"][:, 0] > 0)   # wrapped gas


def test_stars_blackbody_temperature_and_selection():
    with tempfile.TemporaryDirectory() as d:
        snap = Path(d) / "snap.hdf5"
        fake_snapshot(snap)
        L = 3.09e18 / (100 * pg.PC_M)
        stars = pg.load_stars(snap, [9.9 * L, 5 * L, 5 * L], 2.2)    # star at x=0.2 is across the boundary
        assert len(stars) == 1 and stars[0]["particle_id"] == 7
        assert abs(stars[0]["temperature_K"] - 5772) < 5            # 1 Lsun, 1 Rsun -> solar T_eff
        stars = pg.load_stars(snap, [5 * L, 5 * L, 5 * L], 2.2)      # L = 0 sink: recorded, not a source
        assert len(stars) == 1 and not stars[0]["used_as_source"]


def test_catalogue_susceptibility_guard():
    root = Path(__file__).resolve().parent / "dust_input"
    assert pg.catalogue_susceptibility(root / "astrodust_fe_0.0_sus.dat") == 4.47e-4
    with tempfile.TemporaryDirectory() as d:
        src = root / "astrodust_fe_0.0_sus.dat"
        bad = Path(d) / "input" / "dust_cs"
        bad.mkdir(parents=True)
        text = src.read_text().replace("1.5870\t4.47e-4", "1.5870\t0", 1)
        assert text != src.read_text()
        (bad / src.name).write_text(text)
        try:
            pg.stage_dust(Path(d), Path(d) / "out")
            raise AssertionError("chi = 0 catalogue was accepted")
        except ValueError:
            pass


def test_polaris_command_contents():
    class A:
        nr_threads, star_photons, isrf_photons, isrf_strength, isrf_radius_factor = 8, 1000000, 10000000, 1.0, 10.0
        nc, max_subpixel_lvl, distance_pc, mu = 64, 0, 140.0, 2.33
    stars = [dict(position_m=[1.0, 2.0, 3.0], radius_rsun=2.0, temperature_K=4000.0, used_as_source=True),
             dict(position_m=[0.0, 0.0, 0.0], radius_rsun=0.0, used_as_source=False)]
    cmd = pg.polaris_command(Path("/abs/astrodust_fe_0.0_sus.dat"), 0.01, stars, A)
    assert cmd.count("<detector_dust") == 114 and len(pg.WAVELENGTHS_UM) == 38 and cmd.count("<source_star") == 1
    assert cmd.count("<task>") == 2 and '<path_grid> "grid.dat"' in cmd and '"temp/grid_temp_rat.dat"' in cmd
    assert "<cmd> CMD_TEMP_RAT" in cmd and cmd.count("<align> ALIG_RAT") == 2       # temperature AND emission
    assert "phase_function" not in cmd and "R_rayleigh" not in cmd and "ALIG_PA" not in cmd
    assert "<mu> 2.33" in cmd and "<mass_fraction> 0.01\n" in cmd
    assert '"plaw" 1.0 0 5e-09 2.5e-07 -3.5' in cmd and "<max_subpixel_lvl> 0" in cmd                                 # 0 = catalogue density
    assert 'source_isrf nr_photons = "10000000"> 1 10\n' in cmd and 'source_star nr_photons = "1000000"' in cmd
    assert f"{140 * pg.PC_M:.9e}" in cmd


# ---------------------------------------------------------------- real data
def real_grids():
    return sorted((HERE / "grids_64").glob("snapshot_*/core_*/summary.json"))


REGRESSION = HERE / "regression_cell_corner"


def test_real_grids_match_old_grids_in_the_interior():
    """Run with --peak-position cell-corner (the old centring) into REGRESSION.

    Cells deeper than 2*H_max must be identical to the old grids, except the 8
    cells sharing the box-centre corner, where the old placement of the
    sub-cell peak particle was decided by float rounding: those may only
    exchange mass among themselves."""
    pairs = []
    for s in sorted(REGRESSION.glob("snapshot_*/core_*/summary.json")):
        old = OLD / "polaris_cores_v5" / s.parent.parent.name / s.parent.name / "grid.dat"
        if old.exists():
            pairs.append((s.parent / "grid.dat", old))
    if not pairs:
        print("    (no real grids yet; skipped)")
        return
    face_ratio = []
    depths = []
    for new, old in pairs:
        mu = json.loads((new.parent / "summary.json").read_text())["mu"]
        a, b = read_octree_grid(new, mu=mu), read_octree_grid(old)
        # Only particles whose kernel crosses a face are treated differently.
        # Such a particle can sit up to H inside the face and its kernel reaches
        # another H further, so cells deeper than 2*H_max must be identical.
        h_max = json.loads((new.parent / "summary.json").read_text())["max_kernel_crossing_face_cells"]
        depth = int(np.ceil(2 * h_max)) + 1
        depths.append(depth)
        if depth >= 32:
            continue
        sl = slice(depth, 64 - depth)
        centre = (slice(31, 33),) * 3
        da, db = a["density_kg_m3"].astype(float), b["density_kg_m3"].astype(float)
        assert abs(da[centre].sum() / db[centre].sum() - 1) < 1e-6, f"{new}: central mass not conserved"
        for arr in (a, b):
            for f in ("density_kg_m3", "dust_temperature_K", "gas_temperature_K", "Bx", "By", "Bz"):
                arr[f] = arr[f].copy()
                arr[f][centre] = 0
        for f in ("density_kg_m3", "dust_temperature_K", "gas_temperature_K", "Bx", "By", "Bz"):
            x, y = a[f][sl, sl, sl].astype(float), b[f][sl, sl, sl].astype(float)
            scale = np.abs(y).max()
            assert np.max(np.abs(x - y)) <= 1e-5 * scale, f"{new}: interior {f} differs"
        face_ratio.append(face_profile(a["density_kg_m3"].astype(float))[0] /
                          face_profile(b["density_kg_m3"].astype(float))[0])
    print(f"    {len(face_ratio)} boxes: identical to old grids deeper than the widest face-crossing kernel "
          f"(median depth {np.median(depths):.0f} cells; {sum(d >= 32 for d in depths)} boxes too diffuse to test); face-layer density new/old median {np.median(face_ratio):.3f}")


def test_production_grids_put_the_peak_in_the_central_cell():
    files = real_grids()
    if not files:
        print("    (no real grids yet; skipped)")
        return
    for s in files:
        d = json.loads(s.read_text())
        assert d["peak_position"] == "cell-centre" and d["peak_cell_index"] == [d["nc"] // 2] * 3, s
        off = np.array(d["peak_pc"]) - np.array(d["grid_center_pc"])
        np.testing.assert_allclose(off, d["cell_pc"] / 2, rtol=1e-9)
    print(f"    {len(files)} grids have the peak at the centre of cell (32, 32, 32)")


def test_real_summaries_are_consistent():
    files = real_grids()
    if not files:
        print("    (no real grids yet; skipped)")
        return
    for s in files:
        d = json.loads(s.read_text())
        assert d["grid_field_ids"][0] == 0 and d["mu"] == 2.33
        g = read_octree_grid(s.parent / "grid.dat", mu=d["mu"])
        assert g["field_ids"] == (0, 2, 3, 4, 5, 6)
        mass = g["density_kg_m3"].sum() * (d["side_pc"] * pg.PC_M / d["nc"]) ** 3 / pg.MSUN_KG
        assert abs(mass / d["grid_mass_Msun"] - 1) < 1e-5, s          # float32 storage only
        assert 0.8 < d["grid_mass_Msun"] / d["mass_particles_centre_inside_Msun"] < 1.2, s
        assert d["n_detectors"] == 3 * len(d["wavelengths_um"]) and d["dust_to_gas_ratio"] > 0
    print(f"    {len(files)} summaries consistent with their grids")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(bool(fails))
