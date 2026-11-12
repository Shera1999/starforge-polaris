#!/usr/bin/env python3
"""Tests for build_dataset.py (run: python test_build_dataset.py)."""
import numpy as np

import build_dataset as bd

N = 8
IX, IY, IZ = np.indices((N, N, N)).astype(float)


def test_view_axes_and_observer_side():
    # view 0: los z (observer at high iz), +col = +x, +row = +y
    assert np.all(bd.to_view(IZ, 0)[0] == N - 1) and np.all(bd.to_view(IZ, 0)[-1] == 0)
    v = bd.to_view(IX, 0); assert all(np.all(v[:, :, c] == c) for c in range(N))
    v = bd.to_view(IY, 0); assert all(np.all(v[:, r, :] == r) for r in range(N))
    # view 1: los y (observer at high iy), +col = +x, +row = -z
    assert np.all(bd.to_view(IY, 1)[0] == N - 1)
    v = bd.to_view(IX, 1); assert all(np.all(v[:, :, c] == c) for c in range(N))
    v = bd.to_view(IZ, 1); assert all(np.all(v[:, r, :] == N - 1 - r) for r in range(N))
    # view 2: los x (observer at high ix), +col = -z, +row = +y
    assert np.all(bd.to_view(IX, 2)[0] == N - 1)
    v = bd.to_view(IZ, 2); assert all(np.all(v[:, :, c] == N - 1 - c) for c in range(N))
    v = bd.to_view(IY, 2); assert all(np.all(v[:, r, :] == r) for r in range(N))


def test_field_components():
    one, zero = np.ones((N, N, N)), np.zeros((N, N, N))
    Bx = np.stack([one, zero, zero])
    bc, br, bl = bd.field_in_view(Bx, 0); assert np.all(bc == 1) and np.all(br == 0) and np.all(bl == 0)
    bc, br, bl = bd.field_in_view(Bx, 2); assert np.all(bl == -1) and np.all(bc == 0)   # +x points at the observer
    Bz = np.stack([zero, zero, one])
    bc, br, bl = bd.field_in_view(Bz, 0); assert np.all(bl == -1)
    bc, br, bl = bd.field_in_view(Bz, 1); assert np.all(br == -1) and np.all(bl == 0)    # +row = -z
    bc, br, bl = bd.field_in_view(Bz, 2); assert np.all(bc == -1)                         # +col = -z
    By = np.stack([zero, one, zero])
    bc, br, bl = bd.field_in_view(By, 1); assert np.all(bl == -1)


def test_splits():
    rng = np.random.default_rng(1)
    sizes = {f: int(s) for f, s in enumerate(rng.integers(1, 30, 300))}
    sizes[999] = 400                                   # bigger than val/test can take
    a, b = bd.assign_splits(sizes), bd.assign_splits(sizes)
    assert a == b and a[999] == "train" and set(a) == set(sizes)
    assert all(a[f] == "train" for f, n in sizes.items() if n > bd.MAX_EVAL_FAMILY)
    tot = sum(sizes.values())
    for s, f in bd.SPLIT_FRACTIONS.items():
        got = sum(n for fam, n in sizes.items() if a[fam] == s) / tot
        assert abs(got - f) < 0.02, (s, got)


def test_safe_log():
    out = bd.safe_log(np.array([100.0, 0.0, -1.0]))
    assert out[0] == 2 and np.isnan(out[1]) and np.isnan(out[2])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
