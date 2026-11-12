#!/usr/bin/env python3
"""Read back POLARIS uniform-octree grid files (format ID 20) written by
octree.py / polaris_prepare_core.py.

The write side (octree.py) always inserts every leaf at the same fixed
level = log2(nc), i.e. a complete, uniform-depth octree: every cell splits
into exactly 8 children, in a fixed order (child index bit0=+x half,
bit1=+y half, bit2=+z half), until every leaf sits at grid resolution nc.
insert() assigns each leaf's position purely from spatial comparison against
each cell's midpoint, so the on-disk traversal order (pre-order: node header,
then children 0..7 in order) is enough to reconstruct the exact (ix, iy, iz)
grid index of every leaf without needing the original insertion order.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

FIELD_NAMES = {28: "density_kg_m3", 0: "density_number_m3", 2: "dust_temperature_K",
               3: "gas_temperature_K", 4: "Bx", 5: "By", 6: "Bz",
               13: "a_alg_m", 40: "a_larm_m", 34: "avg_th", 35: "avg_dir",
               36: "avg_ux", 37: "avg_uy", 38: "avg_uz"}


def crop_index_range(nc: int, side_pc: float, crop_side_pc: float) -> tuple[int, int]:
    """Half-open [lo, hi) cell-index range, along one axis of a uniform nc-cell
    grid spanning side_pc, whose cell centers fall within +/- crop_side_pc/2 of
    the grid's own center.

    Used to crop a core that was simulated over a wider box than the region of
    interest (see polaris_prepare_core.py's --crop-side-pc): since --nc is
    fixed at a power of two, the kept span is the nearest whole number of
    cells, not exactly crop_side_pc.
    """
    dx = side_pc / nc
    axis = -side_pc / 2 + (np.arange(nc) + 0.5) * dx
    idx = np.nonzero(np.abs(axis) <= crop_side_pc / 2)[0]
    if idx.size == 0:
        raise ValueError(f"No cells fall within crop_side_pc={crop_side_pc} of side_pc={side_pc}, nc={nc}")
    return int(idx[0]), int(idx[-1] + 1)

# The experimental POLARIS branch stores gas density as number density
# (field 0) and uses rho = n * mu * m_u, with mu from the command file. The
# reader cannot know mu, so callers pass it (prepare_grids.py records it in
# each summary.json); without it no mass density is derived.
POLARIS_M_U_KG = 1.66053904e-27


def read_octree_grid(path, mu=None) -> dict:
    """Return {field_name: (nc, nc, nc) float32 array, ...} plus 'nc' and 'side_m'.

    If the grid stores number density (field 0) and mu is given, also return
    density_kg_m3 = n * mu * m_u (float64).
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic, nfields = struct.unpack("<HH", f.read(4))
        if magic != 20:
            raise ValueError(f"{path}: unsupported grid format id {magic} (expected 20)")
        field_ids = struct.unpack(f"<{nfields}H", f.read(2 * nfields))
        side_m = struct.unpack("<d", f.read(8))[0]

        # Peek the first node's level-doubling to find nc without assuming it
        # up front: read the whole tree once into a flat leaf list, tracking
        # (ix, iy, iz) at the deduced maximum level, then infer nc = 2**level.
        leaves = []

        def _read(base_ix, base_iy, base_iz, node_level, step):
            is_leaf, level = struct.unpack("<HH", f.read(4))
            if is_leaf:
                values = struct.unpack(f"<{nfields}f", f.read(4 * nfields))
                leaves.append((base_ix, base_iy, base_iz, level, values))
                return
            child_step = step // 2
            for child_index in range(8):
                dx = (child_index & 1) * child_step
                dy = ((child_index >> 1) & 1) * child_step
                dz = ((child_index >> 2) & 1) * child_step
                _read(base_ix + dx, base_iy + dy, base_iz + dz, level + 1, child_step)

        # `step` starts as an arbitrarily large power of two (root spans the
        # whole grid); nc is recovered afterwards from the deepest leaf level.
        _read(0, 0, 0, 0, 1 << 20)

    max_level = max(leaf[3] for leaf in leaves)
    nc = 1 << max_level
    if len(leaves) != nc ** 3:
        raise ValueError(f"{path}: got {len(leaves)} leaves, expected {nc}**3={nc**3} "
                          f"for a complete uniform octree at level {max_level}")
    # Leaf (ix, iy, iz) values were accumulated using child_step starting from
    # 2**20; rescale down to the true nc-resolution integer grid index.
    arrays = {FIELD_NAMES.get(fid, f"field_{fid}"): np.empty((nc, nc, nc), dtype=np.float32)
              for fid in field_ids}
    names = [FIELD_NAMES.get(fid, f"field_{fid}") for fid in field_ids]
    denom = (1 << 20) // nc
    for ix, iy, iz, level, values in leaves:
        gx, gy, gz = ix // denom, iy // denom, iz // denom
        for name, value in zip(names, values):
            arrays[name][gx, gy, gz] = value
    if "density_number_m3" in arrays and "density_kg_m3" not in arrays and mu is not None:
        arrays["density_kg_m3"] = arrays["density_number_m3"].astype(np.float64) * (mu * POLARIS_M_U_KG)
    arrays["nc"] = nc
    arrays["side_m"] = side_m
    arrays["field_ids"] = field_ids
    return arrays


if __name__ == "__main__":
    import sys
    grid = read_octree_grid(sys.argv[1] if len(sys.argv) > 1 else "grid.dat",
                            mu=float(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(f"nc={grid['nc']}  side_m={grid['side_m']:.4e}  fields={grid['field_ids']}")
    dens = grid.get("density_kg_m3", grid.get("density_number_m3"))
    print(f"density: min={dens.min():.4e} max={dens.max():.4e} mean={dens.mean():.4e}")
    if "dust_temperature_K" in grid:
        t = grid["dust_temperature_K"]
        print(f"dust_T:  min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}")
