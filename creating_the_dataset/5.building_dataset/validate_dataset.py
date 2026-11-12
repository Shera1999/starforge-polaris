#!/usr/bin/env python3
"""Validate a built dataset: row counts, split integrity, missing values, and physics consistency on a sample.

Usage: validate_dataset.py [DATASET_DIR]
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/sjafaritabar/starforge_polaris_dataset_v1")
pa_float32, pa_float64 = __import__("pyarrow").float32(), __import__("pyarrow").float64()
CELL_CM = 0.034375 * 3.0856775814913673e18


def nonfinite_counts(path):
    """Non-finite values per float column of one file, one row group at a time."""
    f, out = pq.ParquetFile(path), {}
    for i in range(f.num_row_groups):
        t = f.read_row_group(i)
        for name, col in zip(t.column_names, t.columns):
            if col.type in (pa_float32, pa_float64):
                n = int((~np.isfinite(col.to_numpy())).sum())
                if n:
                    out[name] = out.get(name, 0) + n
    return out


def main():
    m = json.loads((ROOT / "manifest.json").read_text())
    d = ds.dataset(ROOT / "data", format="parquet", partitioning=None)
    names = d.schema.names
    print(f"{len(names)} columns, manifest says {m['n_columns']}; {d.count_rows()} rows, manifest {m['n_rows']}")
    assert len(names) == m["n_columns"] and d.count_rows() == m["n_rows"] and len(set(names)) == len(names)

    meta = d.to_table(columns=["snapshot", "core_id", "family_id", "split", "view", "pix_row", "pix_col"]).to_pandas()
    meta["split"] = meta["split"].astype(str)
    box = meta.snapshot.astype(str) + "/" + meta.core_id.astype(str)
    per_box = meta.groupby(box).size()
    print(f"boxes {per_box.size}; rows per box min {per_box.min()} max {per_box.max()} (expect 12288)")
    assert per_box.size == m["n_boxes"] and (per_box == 3 * 4096).all()
    dup = meta.duplicated(["snapshot", "core_id", "view", "pix_row", "pix_col"]).sum()
    fam_splits = meta.groupby("family_id").split.nunique()
    box_splits = meta.groupby(box).split.nunique()
    print(f"duplicate sightlines {dup}; families in >1 split {(fam_splits > 1).sum()}; boxes in >1 split {(box_splits > 1).sum()}")
    assert dup == 0 and (fam_splits == 1).all() and (box_splits == 1).all()
    nb = meta.groupby("split").size() / 12288
    fams = meta.groupby("split").family_id.nunique()
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {int(nb[s])} boxes ({nb[s] / nb.sum():.1%}), {fams[s]} families, {int(nb[s]) * 12288} rows")

    # missing / non-finite values: every file once, in parallel
    bad = {}
    with mp.Pool(16) as pool:
        for counts in pool.imap_unordered(nonfinite_counts, sorted(map(str, (ROOT / "data").rglob("*.parquet")))):
            for k, n in counts.items():
                bad[k] = bad.get(k, 0) + n
    print(f"columns with NaN/inf: {len(bad)}", dict(sorted(bad.items(), key=lambda x: -x[1])[:10]))

    # physics consistency on a random sample of row groups
    rng = np.random.default_rng(0)
    frags = list(d.get_fragments())
    sample = [frags[i] for i in rng.choice(len(frags), 8, replace=False)]
    b850 = min(m["bands"], key=lambda b: abs(b["wavelength_um"] - 850))["name"]
    cols = ([f"dens_{k:02d}" for k in range(64)] + [f"dens_dust_{k:02d}" for k in range(64)] +
            ["N_col", "Sigma_dust", "view", f"{b850}_polangle", f"{b850}_p", "cos2psi_B_mean", "sin2psi_B_mean", "B_coherence"])
    for f in sample:
        t = f.to_table(columns=cols).to_pandas()
        r1 = t[[f"dens_{k:02d}" for k in range(64)]].to_numpy(np.float64).sum(1) * CELL_CM / t.N_col
        r2 = t[[f"dens_dust_{k:02d}" for k in range(64)]].to_numpy(np.float64).sum(1) * CELL_CM / t.Sigma_dust
        bang = np.degrees(0.5 * np.arctan2(t.sin2psi_B_mean, t.cos2psi_B_mean))
        da = np.degrees(np.abs(np.angle(np.exp(2j * np.radians(t[f"{b850}_polangle"] + 90 - bang))) / 2))
        print(f"  {Path(f.path).parent.name}/{Path(f.path).name}: N check {np.abs(r1 - 1).max():.1e}, dust check "
              f"{np.abs(r2 - 1).max():.1e}, B angle median {np.median(da):.2f} deg, corr(p850, B_coherence) "
              f"{np.corrcoef(t[f'{b850}_p'], t.B_coherence)[0, 1]:.2f}")
        assert np.abs(r1 - 1).max() < 1e-4 and np.abs(r2 - 1).max() < 1e-4
    print("all checks passed")


if __name__ == "__main__":
    main()
