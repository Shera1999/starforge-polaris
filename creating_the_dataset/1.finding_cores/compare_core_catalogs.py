#!/usr/bin/env python3
"""Compare two core-finder runs (e.g. v5.1 vs v5.2) snapshot by snapshot.

Cores are matched by peak_particle_id. Also reruns the 2.2 pc box thinning
(same greedy Chebyshev non-maximum suppression as thin_all_snapshots.py) on
both catalogs, because the thinned boxes are what goes to POLARIS.

Usage: compare_core_catalogs.py OLD_ROOT NEW_ROOT [OUT_CSV]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BOX_PC = 300.0 * 3.09e18 / 3.0856775814913673e18
MIN_SEP_PC = 2.2


def thinned_ids(cat):
    if len(cat) == 0:
        return set()
    pos = np.mod(cat[["peak_x_pc", "peak_y_pc", "peak_z_pc"]].to_numpy(float), BOX_PC)
    tree = cKDTree(pos, boxsize=BOX_PC)
    suppressed = np.zeros(len(cat), bool)
    kept = []
    for i in np.argsort(-cat.mass_selected_gas_Msun.to_numpy()):
        if suppressed[i]:
            continue
        kept.append(i)
        suppressed[tree.query_ball_point(pos[i], r=MIN_SEP_PC, p=np.inf)] = True
    return set(cat.peak_particle_id.to_numpy()[kept])


def main(old_root, new_root, out_csv):
    rows = []
    for new_dir in sorted(Path(new_root).glob("snapshot_*")):
        old_dir = Path(old_root) / new_dir.name
        if not (new_dir / "core_catalog.csv").exists() or not (old_dir / "core_catalog.csv").exists():
            continue
        old = pd.read_csv(old_dir / "core_catalog.csv")
        new = pd.read_csv(new_dir / "core_catalog.csv")
        so = json.loads((old_dir / "dense_gas_scan.json").read_text())
        sn = json.loads((new_dir / "dense_gas_scan.json").read_text())
        m = old.merge(new, on="peak_particle_id", suffixes=("_old", "_new"))
        dm = (m.mass_selected_gas_Msun_new - m.mass_selected_gas_Msun_old)
        bo, bn = thinned_ids(old), thinned_ids(new)
        v = new.gas_2K_plus_2U_over_absW
        rows.append(dict(
            snapshot=new_dir.name,
            dense_particles_old=so["n_selected"], dense_particles_new=sn["n_selected"],
            dense_mass_old=so["mass_selected_Msun"], dense_mass_new=sn["mass_selected_Msun"],
            cores_old=len(old), cores_new=len(new), cores_matched=len(m),
            cores_only_old=len(old) - len(m), cores_only_new=len(new) - len(m),
            matched_cores_mass_changed=int((dm.abs() > 1e-9).sum()),
            matched_mass_change_max_Msun=float(dm.abs().max()) if len(m) else 0.0,
            leaf_mass_old=old.mass_selected_gas_Msun.sum(), leaf_mass_new=new.mass_selected_gas_Msun.sum(),
            boxes_old=len(bo), boxes_new=len(bn), boxes_same_peak=len(bo & bn),
            virial_median=float(v.median()) if len(v) else np.nan,
            n_bound_gas_only=int((v < 1).sum()),
            n_virial_lt_2=int((v < 2).sum()),
            median_EB_over_absW=float((new.magnetic_energy_erg / new.gas_self_potential_energy_erg.abs()).median())
            if len(new) else np.nan,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    cols = ["snapshot", "dense_particles_old", "dense_particles_new", "cores_old", "cores_new",
            "cores_only_old", "cores_only_new", "matched_cores_mass_changed", "matched_mass_change_max_Msun",
            "boxes_old", "boxes_new", "boxes_same_peak", "virial_median", "n_virial_lt_2"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nTotals:")
    for a, b in (("dense_particles_old", "dense_particles_new"), ("dense_mass_old", "dense_mass_new"),
                 ("cores_old", "cores_new"), ("leaf_mass_old", "leaf_mass_new"), ("boxes_old", "boxes_new")):
        print(f"  {a[:-4]}: {df[a].sum():.6g} -> {df[b].sum():.6g}")
    print(f"  boxes whose centre peak is unchanged: {df.boxes_same_peak.sum()} of {df.boxes_new.sum()}")
    print(f"  cores with gas-only 2(K+U)/|W| < 1: {df.n_bound_gas_only.sum()}, < 2: {df.n_virial_lt_2.sum()} "
          f"of {df.cores_new.sum()}; median E_B/|W| = {df.median_EB_over_absW.median():.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "comparison_v5.1_vs_v5.2.csv")
