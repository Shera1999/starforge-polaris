#!/usr/bin/env python3
"""Step 2: choose the POLARIS boxes from the step-1 core catalogs.

A box is an axis-aligned cube of side --box-side-pc centred on one core's
density peak. Within each snapshot, boxes are chosen by greedy non-maximum
suppression: take cores from most to least massive, keep a core unless its
box would overlap an already-kept box. Two cubes of side S overlap iff
|dx|, |dy|, |dz| < S on every axis, i.e. the periodic Chebyshev distance
between centres is < S.

Beyond the selection itself this records:
  - which other cores fall inside each kept box, and how many cores (and how
    much core mass) end up in no box at all;
  - cross-snapshot "families": boxes in consecutive snapshots whose cubes
    overlap are linked, and connected chains share a family_id. Boxes in one
    family image largely the same gas, so a train/test split must keep each
    family on one side;
  - for the central core of each box, the fraction of its member particles
    that were also members of a core in the previous snapshot.

Outputs (in --output): boxes.csv, suppressed_cores.csv, polaris_jobs.txt,
selection_summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

POS = ["peak_x_pc", "peak_y_pc", "peak_z_pc"]
BOX_COLUMNS = [
    "box_id", "snapshot", "snapshot_number", "time_Myr", "core_id", "peak_particle_id",
    "peak_x_pc", "peak_y_pc", "peak_z_pc", "mass_selected_gas_Msun", "peak_nH2_cm3",
    "radius_volume_pc", "gas_2K_plus_2U_over_absW", "n_cores_in_box", "core_mass_in_box_Msun",
    "sink_within_2p2pc_box", "n_sinks_within_2p2pc_box", "sink_association_status",
    "family_id", "family_size", "prev_box_offset_pc", "central_member_overlap_with_prev_cores",
]


def periodic_delta(a, b, box_pc):
    d = np.asarray(a, float) - np.asarray(b, float)
    return d - box_pc * np.rint(d / box_pc)


def greedy_select(cat, side_pc, box_pc):
    """Return (kept positional indices in selection order, suppressor index per core).

    Core masses are integer multiples of the particle mass, so exact ties are
    common (about half of all cores share their mass with another core in the
    same snapshot). Ties go to the higher peak density, then the lower core_id,
    so the result is physical and deterministic.
    """
    n = len(cat)
    suppressor = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return np.empty(0, dtype=np.int64), suppressor
    pos = np.mod(cat[POS].to_numpy(float), box_pc)
    tree = cKDTree(pos, boxsize=box_pc)
    order = np.lexsort((cat.core_id.to_numpy(), -cat.peak_nH2_cm3.to_numpy(),
                        -cat.mass_selected_gas_Msun.to_numpy()))
    suppressed = np.zeros(n, dtype=bool)
    kept = []
    for i in order:
        if suppressed[i]:
            continue
        kept.append(i)
        suppressor[i] = i
        # query_ball_point includes distance == side; such cubes only touch,
        # so exclude them to match the strict-overlap definition.
        for j in tree.query_ball_point(pos[i], r=side_pc, p=np.inf):
            if j != i and not suppressed[j] and np.max(np.abs(periodic_delta(pos[j], pos[i], box_pc))) < side_pc:
                suppressed[j] = True
                suppressor[j] = i
    return np.asarray(kept, dtype=np.int64), suppressor


def snapshot_meta(step1_dir):
    m = json.loads((step1_dir / "run_manifest.json").read_text())
    u, h = m["units"], m["header"]
    return dict(time_Myr=float(h["Time"]) * u["time_myr"], box_pc=float(h["BoxSize"]) * u["length_pc"],
                run_id=m["run_id"])


def member_sets(step1_dir, core_ids):
    """Composite particle identities (id, child_id, generation) for the given cores."""
    path = step1_dir / "core_members_particle_ids.hdf5"
    with h5py.File(path, "r") as f:
        off = f["offsets"][:]
        fields = [f[k][:] for k in ("particle_ids", "child_ids", "generation") if k in f]
    out = {}
    for cid in core_ids:
        sl = slice(off[cid], off[cid + 1])
        out[cid] = set(zip(*[a[sl].tolist() for a in fields]))
    return out


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def run(args):
    step1 = args.step1_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    boxes, suppressed_rows, snapshot_rows = [], [], []
    prev = None   # (box rows of previous snapshot, member sets of all its cores)
    for sdir in sorted(step1.glob("snapshot_*")):
        if not (sdir / "core_catalog.csv").exists():
            continue
        meta = snapshot_meta(sdir)
        box_pc = meta["box_pc"]
        cat = pd.read_csv(sdir / "core_catalog.csv")
        kept, suppressor = greedy_select(cat, args.box_side_pc, box_pc)
        half = args.box_side_pc / 2
        pos = cat[POS].to_numpy(float)
        mass = cat.mass_selected_gas_Msun.to_numpy()
        # Which box (if any) contains each core: |d| < S/2 on every axis.
        in_box = np.full(len(cat), -1, dtype=np.int64)
        for k in kept:
            inside = np.all(np.abs(periodic_delta(pos, pos[k], box_pc)) < half, axis=1)
            in_box[inside & (in_box < 0)] = k
        members = member_sets(sdir, cat.core_id.astype(int).tolist()) if len(cat) else {}
        rows_this = []
        for k in kept:
            r = cat.iloc[k]
            inside = in_box == k
            overlap = np.nan
            offset = np.nan
            if prev is not None and len(prev[0]):
                d = periodic_delta(np.array([b[POS].to_numpy(float) for b in prev[0]]), pos[k], box_pc)
                offset = float(np.min(np.max(np.abs(d), axis=1)))
                me = members[int(r.core_id)]
                if me:
                    prev_all = set().union(*prev[1].values()) if prev[1] else set()
                    overlap = len(me & prev_all) / len(me)
            row = {c: r[c] for c in ("core_id", "peak_particle_id", *POS, "mass_selected_gas_Msun",
                                     "peak_nH2_cm3", "radius_volume_pc", "gas_2K_plus_2U_over_absW",
                                     "sink_within_2p2pc_box", "n_sinks_within_2p2pc_box",
                                     "sink_association_status")}
            row.update(snapshot=sdir.name, snapshot_number=int(sdir.name.split("_")[1]),
                       time_Myr=meta["time_Myr"], n_cores_in_box=int(inside.sum()),
                       core_mass_in_box_Msun=float(mass[inside].sum()),
                       prev_box_offset_pc=offset, central_member_overlap_with_prev_cores=overlap)
            rows_this.append(pd.Series(row))
        for j in np.flatnonzero(suppressor != np.arange(len(cat))):
            k = suppressor[j]
            suppressed_rows.append(dict(
                snapshot=sdir.name, dropped_core_id=int(cat.core_id.iloc[j]), kept_core_id=int(cat.core_id.iloc[k]),
                dropped_mass_Msun=float(mass[j]), kept_mass_Msun=float(mass[k]),
                chebyshev_separation_pc=float(np.max(np.abs(periodic_delta(pos[j], pos[k], box_pc)))),
                inside_kept_box=bool(in_box[j] >= 0)))
        snapshot_rows.append(dict(snapshot=sdir.name, time_Myr=meta["time_Myr"], step1_run_id=meta["run_id"],
                                  n_cores=len(cat), n_boxes=len(kept),
                                  n_cores_in_no_box=int((in_box < 0).sum()),
                                  core_mass_Msun=float(mass.sum()),
                                  core_mass_in_no_box_Msun=float(mass[in_box < 0].sum())))
        boxes.extend(rows_this)
        prev = (rows_this, members)
    df = pd.DataFrame(boxes)
    df.insert(0, "box_id", np.arange(len(df)))

    # Families: link boxes in consecutive non-empty snapshots whose cubes overlap.
    uf = UnionFind(len(df))
    snaps = sorted(df.snapshot_number.unique())
    box_pc = snapshot_meta(next(step1.glob("snapshot_*")))["box_pc"]
    for a, b in zip(snaps[:-1], snaps[1:]):
        A, B = df[df.snapshot_number == a], df[df.snapshot_number == b]
        tree = cKDTree(np.mod(A[POS].to_numpy(float), box_pc), boxsize=box_pc)
        for ib, p in zip(B.box_id, np.mod(B[POS].to_numpy(float), box_pc)):
            for ia in tree.query_ball_point(p, r=args.box_side_pc, p=np.inf):
                if np.max(np.abs(periodic_delta(A[POS].to_numpy(float)[ia], p, box_pc))) < args.box_side_pc:
                    uf.union(int(A.box_id.iloc[ia]), int(ib))
    roots = np.array([uf.find(i) for i in range(len(df))])
    _, family = np.unique(roots, return_inverse=True)
    df["family_id"] = family
    df["family_size"] = df.groupby("family_id").box_id.transform("size")
    df = df[BOX_COLUMNS]

    df.to_csv(out / "boxes.csv", index=False)
    pd.DataFrame(suppressed_rows).to_csv(out / "suppressed_cores.csv", index=False)
    (out / "polaris_jobs.txt").write_text("".join(f"{s} {c}\n" for s, c in zip(df.snapshot, df.core_id.astype(int))))
    per_snap = pd.DataFrame(snapshot_rows)
    per_snap.to_csv(out / "per_snapshot_selection.csv", index=False)
    fam = df.groupby("family_id").agg(size=("box_id", "size"), first=("snapshot_number", "min"),
                                      last=("snapshot_number", "max"))
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        step1_root=str(step1), box_side_pc=args.box_side_pc,
        n_boxes=len(df), n_snapshots_with_boxes=int(df.snapshot.nunique()),
        n_cores_total=int(per_snap.n_cores.sum()),
        n_cores_in_no_box=int(per_snap.n_cores_in_no_box.sum()),
        core_mass_total_Msun=float(per_snap.core_mass_Msun.sum()),
        core_mass_in_no_box_Msun=float(per_snap.core_mass_in_no_box_Msun.sum()),
        n_families=int(len(fam)), largest_family=int(fam["size"].max()),
        family_size_quantiles={q: float(fam["size"].quantile(q)) for q in (0.5, 0.9, 0.99)},
        n_boxes_with_sink=int(df.sink_within_2p2pc_box.astype(bool).sum()),
        n_boxes_gas_bound_central_core=int((df.gas_2K_plus_2U_over_absW < 1).sum()),
    )
    (out / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return df


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    p.add_argument("--step1-root", type=Path, default=here.parent / "1.finding_cores" / "dendrogram_cores_v5.2")
    p.add_argument("--output", type=Path, default=here / "boxes_2.2pc")
    p.add_argument("--box-side-pc", type=float, default=2.2)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
