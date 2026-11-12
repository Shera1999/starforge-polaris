# Step 5: STARFORGE + POLARIS sightline dataset

`build_dataset.py` turns the finished POLARIS runs (`/scratch/sjafaritabar/polaris_runs`, step 4) and the
grids (`../3.preparing_polaris_grids/grids_64`) into one row per sightline:
1908 boxes x 3 views x 64 x 64 pixels = 23,445,504 rows, 1443 columns, Parquet (zstd).

Output: `/scratch/sjafaritabar/starforge_polaris_dataset_v1/`
- `data/<split>/<snapshot>.parquet`: one row group per box and view (4096 rows)
- `splits.csv`: box -> family_id -> split
- `manifest.json`: units, geometry conventions, band table (with `reliable` flag), split sizes, per-box QA

```python
import pandas as pd, pyarrow.dataset as ds
root = "/scratch/sjafaritabar/starforge_polaris_dataset_v1/data"
cols = ["planck_hfi_353ghz_I_log", "planck_hfi_353ghz_p"] + [f"dens_{k:02d}_log" for k in range(64)]
train = pd.read_parquet(f"{root}/train", columns=cols)               # only the columns you need
full = ds.dataset(root, format="parquet", partitioning=None)         # lazy, all splits
```

## Conventions
- **Views**: 0 = (0, 0), line of sight along grid z; 1 = (-90, 0), along y; 2 = (0, 90), along x.
  Image +col / +row = +x / +y, +x / -z, -z / +y. Verified by `check_view_geometry.py`.
- **Profiles** `<q>_00 .. <q>_63` run along the line of sight, **00 = cell nearest the observer**
  (the high-index grid end faces the observer, `check_observer_side.py`). Cell = 0.034375 pc.
- **Angles** (`psi`, `polangle`) are measured from image +col towards +row. Polarization angle
  `polangle` is the E-vector angle in (-90, 90]; the inferred field angle is `polangle + 90`.
- **B_los** > 0 points away from the observer.
- `_log` columns are log10 (NaN where the value is <= 0). Signed quantities have no `_log` column.

## Columns
| Group | Columns | Units |
|---|---|---|
| metadata | snapshot, time_Myr, core_id, box_id, family_id, n_stars, split, view, pix_row, pix_col | |
| density | dens_k (+_log) | cm^-3, gas particles, n = rho / (2.33 m_u) |
| dust density | dens_dust_k (+_log) | g cm^-3, = 0.01 x gas mass density (the dust-to-gas ratio POLARIS used) |
| dust temperature | temp_polaris_k, temp_starforge_k (+_log) | K |
| field strength | B_tot_k, B_pos_k (+_log) | uG |
| field geometry | cos2psi_B_k, sin2psi_B_k (sky-plane orientation), sin2gamma_B_k (B_pos^2 / B_tot^2) | |
| raw field | B_col_k, B_row_k, B_los_k | uG |
| sightline | N_col, Sigma, Sigma_dust, temp_polaris_mean, temp_starforge_mean, B_tot_mean (+_log) | cm^-2, g cm^-2, g cm^-2, K, uG |
| sightline field | cos2psi_B_mean, sin2psi_B_mean, B_coherence | |
| bands (38) | <band>_I, _Q, _U (float64: at 3-8 um they reach 1e-108), _p, _polangle, _I_log, _p_log | MJy/sr, fraction, deg |

Means are density-weighted. `cos2psi_B_mean`, `sin2psi_B_mean` give the direction of
z = sum(n sin^2(gamma) exp(2i psi)), the field orientation a polarization map measures, and
`B_coherence` = |z| / sum(n) (0..1) combines tangling along the line of sight and inclination; it
correlates with p(850 um) at r = 0.87.

Bands <= 25 um (14 of 38) are flagged `reliable: false` in `manifest.json`: POLARIS was run without
stochastic heating of small grains. Images have no observational noise or beam; add those when training.
Monte Carlo noise: ~0.008 K in dust temperature (ISRF 1e7 photons at radius 10, stars 1e6).

## Splits
Whole cross-snapshot families go to one split (consecutive snapshots are 0.041 Myr apart up to snapshot 420 and 0.41 Myr after it, so splitting
boxes or pixels would leak near-copies into the test set); all 3 views of a box share its split.
Target 80/10/10 by box count, seed 20261111; families with more than 50 boxes always go to train, so
val and test are built from many independent regions (v1.1: val 58 families, test 52, none above 21%
of its split). Actual numbers in `manifest.json`.

### Two variants (same data, different splits)
Star-containing boxes are rare (52 of 1908, in 12 families), and the default split puts all of them in train.
A second dataset moves four small star families out of train; every other box keeps its split:

| Dataset | train | val | test | star boxes train / val / test |
|---|---|---|---|---|
| `starforge_polaris_dataset_v1` (default) | 1538 | 187 | 183 | 52 / 0 / 0 |
| `starforge_polaris_dataset_v1_stars_eval` | 1458 | 216 (+ families 41, 226) | 234 (+ families 133, 139) | 36 / 8 / 8 |

Built with `build_dataset.py --out .../starforge_polaris_dataset_v1_stars_eval --assign 41=val --assign 226=val
--assign 133=test --assign 139=test`. Families 140 and 280 stay in train: they are only 3.3 and 2.8 pc from the large
training family 2. The non-star val and test families are identical in both datasets, so a model trained on either
one can be compared on that shared subset. All star families lie in the same central star-forming region
(their nearest boxes are 2.6-4.8 pc apart, at snapshots 430-500), so their scores measure performance on new
cores in that region, not in a different cloud.

## Files
- `build_dataset.py`, `test_build_dataset.py` (run directly: `python test_build_dataset.py`)
- `validate_dataset.py`: row counts, split integrity, NaN/inf scan of every column (~1 min), physics checks on sampled files
- `check_view_geometry.py`, `check_observer_side.py`: the geometry checks the conventions rest on
