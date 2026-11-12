#!/usr/bin/env python3
"""Density-leaf finder for single-file, non-cosmological STARFORGE snapshots.

Selects gas with n(H2) above a threshold, links particles whose kernels
overlap, and builds an astrodendro dendrogram on that particle graph. Each
leaf (a local density peak) is one catalog "core". A density leaf is not
assumed to be gravitationally bound or associated with a protostar; with
--gravity direct (default) the gas-only virial ratio 2(K+U)/|W| is reported
so boundedness can be judged.

Snapshot units (verified against the M2e5_R30 initial conditions, see
check_snapshot_units.py): lengths, masses and velocities in the header's
code units (1.0014 pc, 1.0008 Msun, 1 m/s); MagneticField in Tesla, i.e.
1e4 Gauss per saved unit (NOT the header's Internal_UnitB_In_Gauss);
Temperature/Dust_Temperature in K; Metallicity column 0 = Z, column 1 = Y.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

M_H_G = 1.6735575e-24
PC_CM = 3.0856775814913673e18
MSUN_G = 1.98847e33
G_CGS = 6.67430e-8
VERSION = "5.2.0"
# Saved STARFORGE MagneticField unit: Tesla (IC field 7.19e-10 matches mu=4.2).
STARFORGE_B_UNIT_GAUSS = 1e4

REQUIRED = {
    "Coordinates": ("coords", 3), "Density": ("rho", 1),
    "Masses": ("mass", 1), "ParticleIDs": ("ids", 1),
    "SmoothingLength": ("hsml", 1),
    "NeutralHydrogenAbundance": ("neutral", 1),
    "MolecularMassFraction": ("mol", 1),
}
OPTIONAL = {
    "Velocities": ("velocity", 3), "Temperature": ("temperature", 1),
    "Dust_Temperature": ("dust_temperature", 1),
    "MagneticField": ("magnetic", 3), "InternalEnergy": ("internal_energy", 1),
    "SoundSpeed": ("sound_speed", 1),
    "ParticleChildIDsNumber": ("child_ids", 1),
    "ParticleIDGenerationNumber": ("generation", 1),
}
INTEGER_KEYS = {"ids", "child_ids", "generation", "snapshot_index"}
CORE_COLUMNS = [
    "core_id", "dendrogram_leaf_idx", "parent_structure_idx", "trunk_structure_idx",
    "n_cells", "mass_selected_gas_Msun", "mass_H2_Msun", "volume_pc3",
    "parent_subtree_mass_Msun", "parent_direct_mass_Msun", "leaf_min_nH2_cm3",
    "parent_merge_level_nH2_cm3", "peak_nH2_cm3", "peak_particle_id",
    "peak_x_pc", "peak_y_pc", "peak_z_pc", "centroid_x_pc", "centroid_y_pc",
    "centroid_z_pc", "peak_centroid_offset_pc", "radius_volume_pc",
    "radius_centroid_uniform_sphere_pc", "radius_peak_v4_pc",
    "shape_semiaxis_major_pc", "shape_semiaxis_middle_pc", "shape_semiaxis_minor_pc",
    "shape_minor_major_ratio", "max_member_distance_from_peak_pc",
    "member_distance_p95_from_peak_pc", "periodic_extent_ambiguous",
    "sigma_bulk_1d_kms", "mean_radial_velocity_kms", "kinetic_internal_erg",
    "thermal_internal_erg", "rms_sound_speed_kms", "mean_Tgas_K_massweighted",
    "mean_Tdust_K_massweighted", "mean_Bx_snapshot_massweighted",
    "mean_By_snapshot_massweighted", "mean_Bz_snapshot_massweighted",
    "rms_B_snapshot_volumeweighted", "magnetic_energy_erg",
    "self_gravity_status", "gas_self_potential_energy_erg",
    "gas_2K_plus_2U_over_absW", "gas_K_plus_U_plus_EB_plus_W_erg",
    "nearest_sink_id", "nearest_sink_from_peak_pc", "nearest_sink_from_centroid_pc",
    "sink_within_peak_radius_v4", "n_sinks_within_peak_radius",
    "n_sinks_in_member_support", "n_shared_support_sinks", "sink_association_status",
    "sink_within_0p02pc_of_peak", "sink_within_2p2pc_box", "n_sinks_within_2p2pc_box",
]
ASSOCIATION_COLUMNS = [
    "core_id", "sink_id", "sink_mass_Msun", "sink_stage", "sink_formation_time_code",
    "sink_formation_time_Myr",
    "distance_from_peak_pc", "distance_from_centroid_pc", "nearest_member_distance_pc",
    "minimum_member_distance_over_support", "within_peak_radius",
    "in_member_support", "outside_all_member_centers_from_peak",
    "n_core_supports_for_sink", "support_association_ambiguous",
]
STRUCTURE_COLUMNS = [
    "structure_idx", "parent_structure_idx", "trunk_structure_idx", "is_leaf",
    "level", "direct_n_cells", "subtree_n_cells", "direct_mass_Msun",
    "subtree_mass_Msun", "direct_min_nH2_cm3", "direct_max_nH2_cm3",
]


def chunks(n, size):
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def delta(a, b, box=None):
    d = np.asarray(a) - np.asarray(b)
    return d if box is None else d - box * np.rint(d / box)


def require_finite(name, array, positive=False, nonnegative=False):
    a = np.asarray(array)
    bad = ~np.isfinite(a)
    if positive:
        bad |= a <= 0
    if nonnegative:
        bad |= a < 0
    if np.any(bad):
        raise ValueError(f"{name}: {int(bad.sum())} invalid values; refusing silent clipping/dropping")


def uint_values(name, array):
    a = np.asarray(array)
    if a.dtype.kind not in "ui" or (a.dtype.kind == "i" and np.any(a < 0)):
        raise ValueError(f"{name} must contain nonnegative integers")
    return a.astype(np.uint64, copy=False)


def molecular_density(rho, neutral, mol, yz, density_unit, convention):
    """Return n(H2) and the H2 mass fraction, without clipping input errors."""
    rho, neutral, mol, yz = [np.asarray(x, dtype=float) for x in (rho, neutral, mol, yz)]
    require_finite("Density", rho, positive=True)
    if yz.shape != (len(rho), 2):
        raise ValueError("Metallicity must provide Z and Y in its first two columns")
    for name, x in (("NeutralHydrogenAbundance", neutral), ("MolecularMassFraction", mol)):
        require_finite(name, x, nonnegative=True)
        if x.shape != rho.shape or np.any(x > 1):
            raise ValueError(f"{name} must have shape {rho.shape} and values in [0,1]")
    require_finite("Metallicity Z,Y", yz, nonnegative=True)
    xh = 1.0 - yz[:, 0] - yz[:, 1]
    if np.any(xh <= 0):
        raise ValueError("Z + Y must be less than one")
    if convention == "neutral-hydrogen":
        h2_fraction = xh * neutral * mol
    elif convention == "total-hydrogen":
        h2_fraction = xh * mol
    else:
        raise ValueError(f"Unknown molecular fraction convention: {convention}")
    nh2 = rho * density_unit * h2_fraction / (2.0 * M_H_G)
    require_finite("Derived nH2", nh2, nonnegative=True)
    return nh2, h2_fraction


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n")


def snapshot_metadata(path):
    with h5py.File(path, "r") as f:
        header = dict(f["Header"].attrs)
        if int(header.get("ComovingIntegrationOn", -1)) != 0:
            raise ValueError("Only explicitly non-cosmological snapshots are supported")
        if float(header.get("HubbleParam", 1)) != 1:
            raise ValueError("HubbleParam != 1 requires a separately verified unit conversion")
        if int(header.get("NumFilesPerSnapshot", 1)) != 1:
            raise ValueError("Multi-file snapshots must first be combined; partial reads are refused")
        ul, um, uv = [float(header[k]) for k in
                      ("UnitLength_In_CGS", "UnitMass_In_CGS", "UnitVelocity_In_CGS")]
        require_finite("Snapshot units", [ul, um, uv], positive=True)
        box = float(header["BoxSize"])
        require_finite("BoxSize", [box], positive=True)
        require_finite("Time", [header["Time"]])
        g = f["PartType0"]
        n = len(g["Masses"])
        for h5name, (_, width) in {**REQUIRED, **OPTIONAL}.items():
            if h5name not in g:
                if h5name in REQUIRED:
                    raise ValueError(f"Missing PartType0/{h5name}")
                continue
            expected = (n,) if width == 1 else (n, width)
            if g[h5name].shape != expected:
                raise ValueError(f"PartType0/{h5name} has shape {g[h5name].shape}; expected {expected}")
        if "Metallicity" not in g or g["Metallicity"].ndim != 2 or g["Metallicity"].shape[0] != n or g["Metallicity"].shape[1] < 2:
            raise ValueError("Missing or malformed PartType0/Metallicity")
        for key in ("NumPart_ThisFile", "NumPart_Total"):
            if key in header and int(header[key][0]) != n:
                raise ValueError(f"{key} does not match gas dataset length")
        missing = [key for key in OPTIONAL if key not in g]
    u = dict(length_pc=ul / PC_CM, mass_msun=um / MSUN_G, density_gcm3=um / ul**3,
             velocity_kms=uv / 1e5, specific_energy_cgs=uv**2,
             time_myr=(ul / uv) / (1e6 * 365.25 * 86400))
    return u, box, header, missing


def read_gas(path, u, args, scan_only=False):
    """One chunked pass; selection scalars are validated for every gas element."""
    pieces = {}
    report = dict(n_input=0, n_selected=0, n_below_or_equal_density_threshold=0,
                  n_dense_rejected_particle_mass=0, mass_input_Msun=0.0,
                  mass_selected_Msun=0.0, mass_below_or_equal_density_threshold_Msun=0.0,
                  mass_dense_rejected_particle_mass_Msun=0.0)
    with h5py.File(path, "r") as f:
        g = f["PartType0"]
        fields = {name: spec for name, spec in {**REQUIRED, **OPTIONAL}.items() if name in g}
        for _, (key, _) in fields.items():
            if key not in {"neutral", "mol"}:
                pieces[key] = []
        pieces.update(nh2=[], h2_fraction=[], snapshot_index=[])
        n = len(g["Masses"])
        for sl in chunks(n, args.chunk):
            rho = np.asarray(g["Density"][sl], dtype=float)
            mass = np.asarray(g["Masses"][sl], dtype=float) * u["mass_msun"]
            require_finite("Masses", mass, positive=True)
            nh2, h2fraction = molecular_density(
                rho, g["NeutralHydrogenAbundance"][sl], g["MolecularMassFraction"][sl],
                g["Metallicity"][sl, :2], u["density_gcm3"], args.molecular_fraction_convention)
            # Astrodendro also uses a strict > min_value comparison.
            dense = nh2 > args.min_nh2_cm3
            selected = dense & (mass >= args.min_cell_mass_msun)
            rejected = dense & ~selected
            report["n_input"] += len(mass)
            report["mass_input_Msun"] += float(mass.sum())
            for mask, nkey, mkey in (
                (selected, "n_selected", "mass_selected_Msun"),
                (~dense, "n_below_or_equal_density_threshold", "mass_below_or_equal_density_threshold_Msun"),
                (rejected, "n_dense_rejected_particle_mass", "mass_dense_rejected_particle_mass_Msun"),
            ):
                report[nkey] += int(mask.sum())
                report[mkey] += float(mass[mask].sum())
            if not scan_only and selected.any():
                for name, (key, _) in fields.items():
                    if key in {"neutral", "mol"}:
                        continue
                    if key == "mass":
                        arr = mass[selected]
                    elif key == "rho":
                        arr = rho[selected] * u["density_gcm3"]
                    else:
                        arr = np.asarray(g[name][sl])[selected]
                    if key in INTEGER_KEYS:
                        arr = uint_values(name, arr)
                    else:
                        arr = np.asarray(arr, dtype=float)
                        require_finite(name, arr, positive=key in {"mass", "rho", "hsml"},
                                       nonnegative=key in {"temperature", "dust_temperature", "internal_energy", "sound_speed"})
                    pieces[key].append(arr)
                pieces["nh2"].append(nh2[selected])
                pieces["h2_fraction"].append(h2fraction[selected])
                pieces["snapshot_index"].append(np.flatnonzero(selected).astype(np.uint64) + sl.start)
            if sl.stop % 20_000_000 < args.chunk or sl.stop == n:
                print(f"  gas {sl.stop:,}/{n:,}; selected={report['n_selected']:,}", flush=True)
    if scan_only:
        return None, report
    widths = {spec[0]: spec[1] for spec in fields.values()}
    gas = {}
    for key, values in pieces.items():
        shape = (0, 3) if widths.get(key) == 3 else (0,)
        gas[key] = np.concatenate(values) if values else np.empty(shape, dtype=np.uint64 if key in INTEGER_KEYS else float)
    # Base IDs alone need not be unique after splitting. Preserve the available
    # genealogy and refuse ambiguity even in that composite identity.
    identity = [gas[k] for k in ("ids", "child_ids", "generation") if k in gas]
    if len(gas["ids"]):
        keys = np.rec.fromarrays(identity)
        if len(np.unique(keys)) != len(keys):
            raise ValueError("Selected gas has duplicate composite particle identities")
        order = np.lexsort(tuple(identity[::-1]))
        gas = {k: v[order] for k, v in gas.items()}
    report["identity_fields"] = [k for k in ("ids", "child_ids", "generation") if k in gas]
    report["n_duplicate_base_ids_selected"] = len(gas["ids"]) - len(np.unique(gas["ids"]))
    return gas, report


def neighbor_graph(coords, hsml, box=None, eta=1.0, max_link=None, mode="support",
                   neighbors=6, chunk=20000, workers=1, max_directed_edges=100_000_000):
    """Exact union graph: d_ij <= eta*max(H_i,H_j), optionally capped.

    Query each particle's own support, then symmetrize. This includes every
    qualifying pair, including pairs admitted only by the larger support.
    The optional knn mode is explicitly an approximation for convergence runs.
    """
    xyz = np.asarray(coords, dtype=float)
    h = np.asarray(hsml, dtype=float)
    n = len(xyz)
    if xyz.shape != (n, 3) or h.shape != (n,):
        raise ValueError("Expected coordinates (N,3) and smoothing lengths (N,)")
    require_finite("Coordinates", xyz)
    require_finite("SmoothingLength", h, positive=True)
    require_finite("eta", [eta], positive=True)
    if box is not None:
        require_finite("Periodic box", [box], positive=True)
        xyz = np.mod(xyz, box)
    if max_link is not None:
        require_finite("max_link", [max_link], positive=True)
    if mode not in {"support", "knn"} or neighbors < 1 or max_directed_edges < 1:
        raise ValueError("Invalid graph configuration")
    tree = cKDTree(xyz, boxsize=box)
    counts = np.zeros(n, dtype=np.int64)
    radii = eta * h
    if max_link is not None:
        radii = np.minimum(radii, max_link)
    require_finite("Derived support radii", radii, positive=True)

    def query(sl):
        if mode == "support":
            found = tree.query_ball_point(xyz[sl], radii[sl], workers=workers, return_sorted=True)
            return [np.asarray(js, dtype=np.int64)[np.asarray(js) != i]
                    for i, js in zip(range(sl.start, sl.stop), found)]
        result = []
        distances, indices = tree.query(xyz[sl], k=min(n, neighbors + 1), workers=workers)
        distances = np.asarray(distances).reshape(sl.stop - sl.start, -1)
        indices = np.asarray(indices).reshape(sl.stop - sl.start, -1)
        for i, ds, js in zip(range(sl.start, sl.stop), distances, indices):
            not_self = js != i
            js, ds = js[not_self][:neighbors], ds[not_self][:neighbors]
            valid = ds <= eta * np.maximum(h[i], h[js])
            if max_link is not None:
                valid &= ds <= max_link
            result.append(np.sort(js[valid]))
        return result

    for sl in chunks(n, chunk):
        if mode == "support":
            counts[sl] = tree.query_ball_point(xyz[sl], radii[sl], workers=workers, return_length=True) - 1
        else:
            counts[sl] = [len(js) for js in query(sl)]
    n_edges = int(counts.sum())
    if n_edges > max_directed_edges:
        raise MemoryError(f"Graph needs {n_edges:,} directed support/candidate entries, above limit "
                          f"{max_directed_edges:,}. Increase --max-directed-edges deliberately; no links were truncated.")
    print(f"  graph preflight: {n_edges:,} directed entries; symmetric graph has at most twice that", flush=True)
    indptr = np.r_[0, np.cumsum(counts)]
    indices = np.empty(n_edges, dtype=np.int64)
    for sl in chunks(n, chunk):
        for i, js in zip(range(sl.start, sl.stop), query(sl)):
            if len(js) != counts[i]:
                raise RuntimeError("Graph count/fill mismatch")
            indices[indptr[i]:indptr[i + 1]] = js
    graph = csr_matrix((np.ones(n_edges, dtype=np.uint8), indices, indptr), shape=(n, n))
    graph = graph.maximum(graph.T).tocsr()
    graph.sort_indices()
    degree = np.diff(graph.indptr)
    n_components, component = connected_components(graph, directed=False)
    sizes = np.bincount(component, minlength=n_components)
    diagnostics = dict(mode=mode, full_support_graph=mode == "support", neighbors=neighbors if mode == "knn" else None,
                       n_particles=n, n_directed_entries_before_symmetry=n_edges,
                       n_undirected_edges=int(graph.nnz // 2), n_components=int(n_components),
                       n_isolated_particles=int((degree == 0).sum()),
                       largest_component_n=int(sizes.max()) if len(sizes) else 0,
                       degree_min=int(degree.min()) if n else None,
                       degree_median=float(np.median(degree)) if n else None,
                       degree_max=int(degree.max()) if n else None)
    return graph, diagnostics


def compute_dendrogram(nh2, graph, minimum, contrast, min_npix):
    if len(nh2) == 0:
        return None
    from astrodendro import Dendrogram

    def neighbors(d, coordinate):
        i = int(coordinate[0])
        return [(int(j),) for j in graph.indices[graph.indptr[i]:graph.indptr[i + 1]]]

    return Dendrogram.compute(nh2, min_value=minimum, min_delta=contrast,
                              min_npix=min_npix, neighbours=neighbors, verbose=True)


def hierarchy_table(d, gas):
    if d is None:
        return pd.DataFrame(columns=STRUCTURE_COLUMNS)
    structures = list(d.all_structures)
    labels = d.index_map.ravel()
    used = labels >= 0
    n = len(structures)
    direct_n = np.bincount(labels[used], minlength=n)
    direct_m = np.bincount(labels[used], weights=gas["mass"][used], minlength=n)
    subtree_n, subtree_m = direct_n.copy(), direct_m.copy()
    for s in sorted(structures, key=lambda s: s.level, reverse=True):
        if s.parent is not None:
            subtree_n[s.parent.idx] += subtree_n[s.idx]
            subtree_m[s.parent.idx] += subtree_m[s.idx]
    rows = []
    for s in structures:
        rows.append(dict(structure_idx=s.idx, parent_structure_idx=s.parent.idx if s.parent else -1,
                         trunk_structure_idx=s.ancestor.idx, is_leaf=s.is_leaf, level=s.level,
                         direct_n_cells=int(direct_n[s.idx]), subtree_n_cells=int(subtree_n[s.idx]),
                         direct_mass_Msun=float(direct_m[s.idx]), subtree_mass_Msun=float(subtree_m[s.idx]),
                         direct_min_nH2_cm3=float(s.vmin), direct_max_nH2_cm3=float(s.vmax)))
    return pd.DataFrame(rows, columns=STRUCTURE_COLUMNS)


def read_sinks(path, u):
    with h5py.File(path, "r") as f:
        if "PartType5" not in f:
            header_n = f["Header"].attrs.get("NumPart_ThisFile", [0] * 6)[5]
            if header_n:
                raise ValueError("Header lists sinks but PartType5 is missing")
            return dict(coords=np.empty((0, 3)), ids=np.empty(0, dtype=np.uint64),
                        mass=np.empty(0), stage=np.empty(0), formation=np.empty(0),
                        formation_myr=np.empty(0))
        g = f["PartType5"]
        xyz = np.asarray(g["Coordinates"][:], dtype=float)
        ids = uint_values("Sink ParticleIDs", g["ParticleIDs"][:])
        n = len(ids)
        if xyz.shape != (n, 3) or len(np.unique(ids)) != n:
            raise ValueError("Sink coordinates or identities are invalid")
        require_finite("Sink coordinates", xyz)
        result = dict(coords=xyz, ids=ids)
        for name, key in (("Masses", "mass"), ("ProtoStellarStage", "stage"), ("StellarFormationTime", "formation")):
            arr = np.asarray(g[name][:], dtype=float) if name in g else np.full(n, np.nan)
            if arr.shape != (n,):
                raise ValueError(f"Invalid sink field shape: {name}")
            if name in g:
                require_finite(name, arr, nonnegative=True)
            result[key] = arr * u["mass_msun"] if key == "mass" else arr
        result["formation_myr"] = result["formation"] * u["time_myr"]
        return result


def self_potential_energy(xyz_pc, mass_msun, softening_pc=0.0, box_pc=None, chunk=256):
    """Pairwise gas-only energy, with an explicitly chosen Plummer softening.

    Minimum-image pairs are used if periodic. This is NOT the periodic Ewald
    energy, GIZMO's spline-softened energy, or the snapshot's total potential.
    """
    total = 0.0
    n = len(mass_msun)
    for sl in chunks(n, chunk):
        i = np.arange(sl.start, sl.stop)
        for jl in chunks(n, chunk):
            if jl.stop <= sl.start:
                continue
            j = np.arange(jl.start, jl.stop)
            upper = i[:, None] < j[None, :]
            r2 = np.sum(delta(xyz_pc[sl, None, :], xyz_pc[None, jl, :], box_pc)**2, axis=2)
            separation = np.sqrt(r2 + softening_pc**2)
            if np.any(separation[upper] == 0):
                raise ValueError("Coincident gravity particles require nonzero softening")
            mm = mass_msun[sl, None] * mass_msun[None, jl]
            total += float(np.sum(mm[upper] / separation[upper]))
    return -G_CGS * MSUN_G**2 / PC_CM * total


def build_catalog(d, gas, sinks, u, box, args, hierarchy):
    rows, associations, member_indices = [], [], []
    parents = hierarchy.set_index("structure_idx").to_dict("index")
    L = u["length_pc"]
    leaves = [] if d is None else sorted(d.leaves, key=lambda s: s.idx)
    for cid, leaf in enumerate(leaves):
        ix = np.asarray(leaf.indices()[0], dtype=np.int64)
        member_indices.append(ix)
        peak = ix[np.argmax(gas["nh2"][ix])]
        p = gas["coords"][peak]
        xyz, m = gas["coords"][ix], gas["mass"][ix]
        mt = float(m.sum())
        dx = delta(xyz, p, box) * L
        center = np.average(dx, axis=0, weights=m)
        centered = dx - center
        centroid = p * L + center
        if box is not None:
            centroid %= box * L
        rpeak = np.linalg.norm(dx, axis=1)
        rc = np.linalg.norm(centered, axis=1)
        covariance = (centered * m[:, None]).T @ centered / mt
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        axes = np.sqrt(5 * eigenvalues)[::-1]
        volume_cgs = m * MSUN_G / gas["rho"][ix]
        volume = float(volume_cgs.sum() / PC_CM**3)
        ambiguous = box is not None and bool(np.any(np.ptp(dx, axis=0) >= box * L / 2))
        row = {key: np.nan for key in CORE_COLUMNS}
        parent_id = leaf.parent.idx if leaf.parent is not None else -1
        row.update(core_id=cid, dendrogram_leaf_idx=leaf.idx, parent_structure_idx=parent_id,
                   trunk_structure_idx=leaf.ancestor.idx, n_cells=len(ix), mass_selected_gas_Msun=mt,
                   mass_H2_Msun=float(np.sum(m * gas["h2_fraction"][ix])), volume_pc3=volume,
                   parent_subtree_mass_Msun=parents[parent_id]["subtree_mass_Msun"] if parent_id >= 0 else np.nan,
                   parent_direct_mass_Msun=parents[parent_id]["direct_mass_Msun"] if parent_id >= 0 else np.nan,
                   leaf_min_nH2_cm3=float(leaf.vmin),
                   # parent.vmax can include absorbed insignificant peaks; it is
                   # not generally the first saddle connecting its children.
                   parent_merge_level_nH2_cm3=float(leaf.parent.height) if leaf.parent else np.nan,
                   peak_nH2_cm3=float(gas["nh2"][peak]), peak_particle_id=int(gas["ids"][peak]),
                   peak_centroid_offset_pc=float(np.linalg.norm(center)),
                   radius_volume_pc=(3 * volume / (4 * math.pi))**(1 / 3),
                   radius_centroid_uniform_sphere_pc=math.sqrt(5 / 3 * np.average(rc**2, weights=m)),
                   radius_peak_v4_pc=math.sqrt(2.5 * np.average(rpeak**2, weights=m)),
                   shape_semiaxis_major_pc=axes[0], shape_semiaxis_middle_pc=axes[1], shape_semiaxis_minor_pc=axes[2],
                   shape_minor_major_ratio=axes[2] / axes[0] if axes[0] > 0 else np.nan,
                   max_member_distance_from_peak_pc=float(rpeak.max()),
                   member_distance_p95_from_peak_pc=float(np.percentile(rpeak, 95)),
                   periodic_extent_ambiguous=ambiguous, self_gravity_status="not_requested")
        for axis, peak_x, center_x in zip("xyz", p * L, centroid):
            row[f"peak_{axis}_pc"] = peak_x
            row[f"centroid_{axis}_pc"] = center_x
        if "velocity" in gas:
            v = gas["velocity"][ix] * u["velocity_kms"]
            dv = v - np.average(v, axis=0, weights=m)
            v2 = np.sum(dv**2, axis=1)
            row["sigma_bulk_1d_kms"] = math.sqrt(np.average(v2, weights=m) / 3)
            row["kinetic_internal_erg"] = float(0.5 * np.sum(m * v2) * MSUN_G * 1e10)
            radial = np.divide(np.sum(dv * centered, axis=1), rc, out=np.zeros(len(ix)), where=rc > 0)
            row["mean_radial_velocity_kms"] = float(np.average(radial, weights=m))
        for key, column in (("temperature", "mean_Tgas_K_massweighted"), ("dust_temperature", "mean_Tdust_K_massweighted")):
            if key in gas:
                row[column] = float(np.average(gas[key][ix], weights=m))
        if "internal_energy" in gas:
            row["thermal_internal_erg"] = float(np.sum(m * gas["internal_energy"][ix]) * MSUN_G * u["specific_energy_cgs"])
        if "sound_speed" in gas:
            row["rms_sound_speed_kms"] = math.sqrt(np.average(gas["sound_speed"][ix]**2, weights=m)) * u["velocity_kms"]
        if "magnetic" in gas:
            magnetic = gas["magnetic"][ix]
            b2 = np.sum(magnetic**2, axis=1)
            for axis, value in zip("xyz", np.average(magnetic, axis=0, weights=m)):
                row[f"mean_B{axis}_snapshot_massweighted"] = value
            row["rms_B_snapshot_volumeweighted"] = math.sqrt(np.average(b2, weights=volume_cgs))
            if args.magnetic_unit_gauss is not None:
                row["magnetic_energy_erg"] = float(np.sum(b2 * volume_cgs) * args.magnetic_unit_gauss**2 / (8 * math.pi))
        if args.gravity == "direct":
            if ambiguous:
                row["self_gravity_status"] = "skipped_periodic_extent_ambiguous"
            elif len(ix) > args.gravity_max_particles:
                row["self_gravity_status"] = "skipped_particle_limit"
            else:
                try:
                    W = self_potential_energy(xyz * L, m, args.gravity_softening_pc,
                                              None if box is None else box * L)
                    row["gas_self_potential_energy_erg"] = W
                    row["self_gravity_status"] = "computed_gas_only"
                    K, U, EB = [row[k] for k in ("kinetic_internal_erg", "thermal_internal_erg", "magnetic_energy_erg")]
                    if W < 0:
                        row["gas_2K_plus_2U_over_absW"] = 2 * (K + U) / abs(W)
                    row["gas_K_plus_U_plus_EB_plus_W_erg"] = K + U + EB + W
                except ValueError:
                    row["self_gravity_status"] = "skipped_coincident_particles_without_softening"
        nsinks = len(sinks["ids"])
        near_count = support_count = 0
        sink_peak_flag, sink_box_flag, n_sinks_in_box = False, False, 0
        if nsinks:
            # Per-axis delta in pc, used for the axis-aligned box-membership flag
            # (matching the cube used throughout the gas-map figures); its norm
            # gives the radial peak distance used everywhere else.
            sink_delta_pc = delta(sinks["coords"], p, box) * L
            dp = np.linalg.norm(sink_delta_pc, axis=1)
            dc = np.linalg.norm(delta(sinks["coords"] * L, centroid, None if box is None else box * L), axis=1)
            nearest = int(np.argmin(dp))
            row.update(nearest_sink_id=int(sinks["ids"][nearest]), nearest_sink_from_peak_pc=float(dp[nearest]),
                       nearest_sink_from_centroid_pc=float(dc.min()))
            sink_peak_flag = bool(dp.min() <= args.sink_peak_proximity_pc)
            in_box = np.all(np.abs(sink_delta_pc) <= args.sink_box_half_width_pc, axis=1)
            sink_box_flag = bool(in_box.any())
            n_sinks_in_box = int(in_box.sum())
            # A sink is inside the leaf's union of particle supports if at least
            # one member support reaches it. This is a geometric proxy, not binding.
            support_radius = args.sink_support_factor * gas["hsml"][ix] * L
            for sl in chunks(nsinks, 64):
                distances = np.linalg.norm(delta(sinks["coords"][sl, None, :], xyz[None, :, :], box), axis=2) * L
                minimum_distance = distances.min(axis=1)
                ratio = (distances / support_radius[None, :]).min(axis=1)
                for j, dmin, q in zip(range(sl.start, sl.stop), minimum_distance, ratio):
                    near, inside = bool(dp[j] <= args.sink_proximity_radius_pc), bool(q <= 1)
                    near_count += int(near)
                    support_count += int(inside)
                    if near or inside or j == nearest:
                        associations.append(dict(core_id=cid, sink_id=int(sinks["ids"][j]),
                            sink_mass_Msun=sinks["mass"][j], sink_stage=sinks["stage"][j],
                            sink_formation_time_code=sinks["formation"][j],
                            sink_formation_time_Myr=sinks["formation_myr"][j], distance_from_peak_pc=float(dp[j]),
                            distance_from_centroid_pc=float(dc[j]), nearest_member_distance_pc=float(dmin),
                            minimum_member_distance_over_support=float(q), within_peak_radius=near,
                            in_member_support=inside, outside_all_member_centers_from_peak=bool(dp[j] > rpeak.max())))
        row.update(sink_within_peak_radius_v4=near_count > 0, n_sinks_within_peak_radius=near_count,
                   n_sinks_in_member_support=support_count, n_shared_support_sinks=0,
                   sink_association_status="support_overlap" if support_count else "proximity_only" if near_count else "no_associated_sink",
                   sink_within_0p02pc_of_peak=sink_peak_flag,
                   sink_within_2p2pc_box=sink_box_flag,
                   n_sinks_within_2p2pc_box=n_sinks_in_box)
        rows.append(row)
    assoc = pd.DataFrame(associations, columns=ASSOCIATION_COLUMNS)
    if len(assoc):
        counts = assoc.loc[assoc.in_member_support].groupby("sink_id").core_id.nunique()
        assoc["n_core_supports_for_sink"] = assoc.sink_id.map(counts).fillna(0).astype(int)
        assoc["support_association_ambiguous"] = assoc.in_member_support & (assoc.n_core_supports_for_sink > 1)
        shared = assoc.loc[assoc.support_association_ambiguous].groupby("core_id").size()
        for row in rows:
            row["n_shared_support_sinks"] = int(shared.get(row["core_id"], 0))
            if row["n_shared_support_sinks"]:
                row["sink_association_status"] = "ambiguous_shared_support"
    # Nullable uints preserve >2**53 IDs, including rows without any sinks.
    cat = pd.DataFrame(rows, columns=CORE_COLUMNS)
    for key in ("peak_particle_id", "nearest_sink_id"):
        cat[key] = pd.array([None if pd.isna(r[key]) else r[key] for r in rows], dtype="UInt64")
    if len(assoc):
        assoc["sink_id"] = pd.array([r["sink_id"] for r in associations], dtype="UInt64")
    return cat, assoc, member_indices


def save_particles(path, gas, d, members, run_id):
    lengths = np.array([len(ix) for ix in members], dtype=np.int64)
    indices = np.concatenate(members) if members else np.empty(0, dtype=np.int64)
    with h5py.File(path, "w") as f:
        f.attrs["run_id"] = run_id
        f.attrs["identity_note"] = "Genealogy is preserved; matching identities across epochs is not a mass-flow history."
        f.create_dataset("core_ids", data=np.arange(len(members), dtype=np.int64))
        f.create_dataset("offsets", data=np.r_[0, np.cumsum(lengths)])
        for key in ("ids", "child_ids", "generation", "snapshot_index"):
            if key in gas:
                name = "particle_ids" if key == "ids" else key
                f.create_dataset(name, data=gas[key][indices], compression="gzip", shuffle=True)
        # This mapping makes every saved dendrogram index interpretable and
        # accounts for branches and pruned gas as well as leaves.
        selected = f.create_group("selected_gas")
        for key in ("ids", "child_ids", "generation", "snapshot_index"):
            if key in gas:
                selected.create_dataset(key, data=gas[key], compression="gzip", shuffle=True)
        labels = d.index_map.ravel() if d is not None else np.full(len(gas["ids"]), -1, dtype=np.int64)
        selected.create_dataset("structure_idx", data=labels, compression="gzip", shuffle=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    p.add_argument("--min-nh2-cm3", type=float, default=1e4)
    p.add_argument("--min-delta-cm3", type=float, default=1e4)
    p.add_argument("--min-npix", type=int, default=100)
    p.add_argument("--min-cell-mass-msun", type=float, default=0.0,
                   help="Reject dense particles lighter than this. 0 keeps split cells near sinks "
                        "(v5.1 used 9.99e-4, which dropped them)")
    p.add_argument("--molecular-fraction-convention", choices=["neutral-hydrogen", "total-hydrogen"], default="neutral-hydrogen",
                   help="Denominator of MolecularMassFraction; neutral-hydrogen retains v4's assumption")
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--graph-mode", choices=["support", "knn"], default="support")
    p.add_argument("--neighbors", type=int, default=6, help="Only used in knn comparison mode")
    p.add_argument("--neighbor-hsml-factor", type=float, default=1.0)
    p.add_argument("--max-neighbor-link-pc", type=float)
    p.add_argument("--max-directed-edges", type=int, default=100_000_000)
    p.add_argument("--graph-chunk", type=int, default=20000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk", type=int, default=2_000_000)
    p.add_argument("--sink-proximity-radius-pc", type=float, default=0.04)
    p.add_argument("--sink-support-factor", type=float, default=1.0)
    p.add_argument("--sink-peak-proximity-pc", type=float, default=0.02,
                   help="Tight peak-to-sink threshold for the sink_within_0p02pc_of_peak flag")
    p.add_argument("--sink-box-half-width-pc", type=float, default=1.1,
                   help="Half-width of the cube used for sink_within_2p2pc_box (default 1.1 pc = 2.2 pc box, matching the gas-map figures)")
    p.add_argument("--magnetic-unit-gauss", type=float, default=STARFORGE_B_UNIT_GAUSS,
                   help="Gauss per saved MagneticField unit (STARFORGE saves Tesla); NOT Internal_UnitB_In_Gauss")
    p.add_argument("--gravity", choices=["none", "direct"], default="direct")
    p.add_argument("--gravity-max-particles", type=int, default=20000)
    p.add_argument("--gravity-softening-pc", type=float, default=1e-3,
                   help="Plummer softening for the gas-only pair energy. W is converged below ~1e-3 pc "
                        "(snapshot_440: 0.1%% change from 1e-4); the median selected smoothing length "
                        "(~0.02 pc) would weaken W by ~20%%. Pass a negative value to use that median instead")
    p.add_argument("--scan-only", action="store_true")
    p.add_argument("--save-dendrogram", action="store_true")
    return p


def validate_args(args):
    for key in ("min_nh2_cm3", "neighbor_hsml_factor", "sink_proximity_radius_pc", "sink_support_factor",
                "sink_peak_proximity_pc", "sink_box_half_width_pc"):
        require_finite(key, [getattr(args, key)], positive=True)
    for key in ("min_delta_cm3", "min_cell_mass_msun"):
        require_finite(key, [getattr(args, key)], nonnegative=True)
    require_finite("gravity_softening_pc", [args.gravity_softening_pc])
    for key in ("max_neighbor_link_pc", "magnetic_unit_gauss"):
        if getattr(args, key) is not None:
            require_finite(key, [getattr(args, key)], positive=True)
    for key in ("min_npix", "neighbors", "chunk", "graph_chunk", "max_directed_edges", "gravity_max_particles"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.workers != -1 and args.workers < 1:
        raise ValueError("workers must be -1 or a positive integer")


def fingerprint(path):
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def run(args):
    validate_args(args)
    snapshot = args.snapshot.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new run directory.")
    u, box_size, header, missing = snapshot_metadata(snapshot)
    before = fingerprint(snapshot)
    provenance = dict(schema_version=VERSION, args=vars(args), snapshot=before,
                      snapshot_fingerprint_kind="path_size_mtime_not_content_hash",
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      python=platform.python_version(),
                      packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "h5py", "astrodendro")},
                      units=u, header=header, missing_optional_gas_fields=missing,
                      density_definition="nH2=rho*XH*fmol/(2*mH), additionally *fneutral for neutral-hydrogen convention",
                      interpretation="Density leaves; no bound/prestellar classification. Sink association is geometric.",
                      magnetic_conversion=f"{args.magnetic_unit_gauss:g} G per saved unit; internal magnetic header is not used",
                      catalog_units=dict(mean_B_snapshot="saved MagneticField unit (Tesla for STARFORGE)",
                                         energies="erg", velocities="km/s", positions_radii="pc",
                                         masses="Msun", densities="cm^-3 (nH2)", temperatures="K"),
                      created_utc=datetime.now(timezone.utc).isoformat())
    run_id = hashlib.sha256(json.dumps(provenance, default=json_value, sort_keys=True).encode()).hexdigest()
    provenance["run_id"] = run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        print(f"Snapshot: {snapshot}\nTime: {float(header['Time']) * u['time_myr']:.6f} Myr", flush=True)
        print(f"Molecular fraction convention: {args.molecular_fraction_convention}; boundary: {args.boundary}", flush=True)
        write_json(staging / "run_manifest.json", provenance)
        gas, scan = read_gas(snapshot, u, args, args.scan_only)
        write_json(staging / "dense_gas_scan.json", scan)
        if not args.scan_only:
            if args.gravity == "direct" and args.gravity_softening_pc < 0 and len(gas["hsml"]):
                args.gravity_softening_pc = float(np.median(gas["hsml"]) * u["length_pc"])
                print(f"Gravity softening: {args.gravity_softening_pc:.3e} pc (median selected smoothing length)", flush=True)
            box = box_size if args.boundary == "periodic" else None
            max_link = None if args.max_neighbor_link_pc is None else args.max_neighbor_link_pc / u["length_pc"]
            graph, diagnostics = neighbor_graph(gas["coords"], gas["hsml"], box, args.neighbor_hsml_factor,
                max_link, args.graph_mode, args.neighbors, args.graph_chunk, args.workers, args.max_directed_edges)
            write_json(staging / "neighbour_graph_diagnostics.json", diagnostics)
            d = compute_dendrogram(gas["nh2"], graph, args.min_nh2_cm3, args.min_delta_cm3, args.min_npix)
            del graph
            hierarchy = hierarchy_table(d, gas)
            cat, associations, members = build_catalog(d, gas, read_sinks(snapshot, u), u, box, args, hierarchy)
            cat.to_csv(staging / "core_catalog.csv", index=False)
            hierarchy.to_csv(staging / "structure_catalog.csv", index=False)
            associations.to_csv(staging / "sink_associations.csv", index=False)
            save_particles(staging / "core_members_particle_ids.hdf5", gas, d, members, run_id)
            dendrogram_saved = args.save_dendrogram and d is not None
            if dendrogram_saved:
                d.save_to(str(staging / "dendrogram.hdf5"), format="hdf5")
            leaf_mass = float(cat.mass_selected_gas_Msun.sum())
            branch_mass = float(hierarchy.loc[~hierarchy.is_leaf.astype(bool), "direct_mass_Msun"].sum())
            labels = d.index_map.ravel() if d is not None else np.full(len(gas["ids"]), -1)
            pruned_mass = float(gas["mass"][labels < 0].sum())
            if not np.isclose(leaf_mass + branch_mass + pruned_mass, scan["mass_selected_Msun"], rtol=1e-10, atol=1e-10):
                raise RuntimeError("Mass budget failed: leaf + branch + pruned != selected")
            summary = dict(run_id=run_id, n_cores=len(cat), mass_selected_Msun=scan["mass_selected_Msun"],
                           mass_leaves_Msun=leaf_mass, mass_branches_direct_Msun=branch_mass,
                           mass_pruned_Msun=pruned_mass,
                           leaf_fraction_of_selected_mass=leaf_mass / scan["mass_selected_Msun"] if scan["mass_selected_Msun"] else None,
                           n_sink_proximity_v4=int(cat.sink_within_peak_radius_v4.sum()),
                           n_support_associated_cores=int((cat.n_sinks_in_member_support > 0).sum()),
                           n_ambiguous_support_cores=int((cat.n_shared_support_sinks > 0).sum()),
                           n_periodic_extent_ambiguous=int(cat.periodic_extent_ambiguous.sum()),
                           gravity_status_counts=cat.self_gravity_status.value_counts().to_dict(),
                           gravity_softening_pc=args.gravity_softening_pc,
                           n_gas_bound_2KU_over_W_lt_1=int((cat.gas_2K_plus_2U_over_absW < 1).sum()),
                           dendrogram_saved=dendrogram_saved)
            write_json(staging / "core_summary.json", summary)
            print(json.dumps(summary, indent=2), flush=True)
        if fingerprint(snapshot) != before:
            raise RuntimeError("Snapshot changed during processing; results will not be published")
        provenance["completed_utc"] = datetime.now(timezone.utc).isoformat()
        provenance["status"] = "scan_complete" if args.scan_only else "complete"
        write_json(staging / "run_manifest.json", provenance)
        # Same-filesystem directory rename publishes all artifacts together.
        if output.exists():
            raise FileExistsError(f"Output appeared during run: {output}")
        os.rename(staging, output)
        print(f"Saved complete run: {output}", flush=True)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return output


if __name__ == "__main__":
    run(parser().parse_args())
