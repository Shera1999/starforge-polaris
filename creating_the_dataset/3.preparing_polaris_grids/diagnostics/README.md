# Step 3 diagnostics

One-off checks behind the polaris.cmd choices (not part of the pipeline run).
Run from anywhere with the starforge env; outputs sit next to each script.

| Script | Question | Output |
|---|---|---|
| check_dust_to_gas.py | What does STARFORGE's DustToGasRatio_Local hold? (Z x DTM/DTM_sun; absolute f_d = 0.01) | dust_to_gas_scans/ |
| check_isrf_normalisation.py | STARFORGE ISRF = 1 vs POLARIS Mathis G0, band by band (heating-equivalent G0 = 1.40) | isrf_check_snapshot_100.log |
| check_box_surface_field.py | Radiation field reaching each box surface, as G0_eff (median ~0.2) | box_surface_G0_snapshot_440.* |
| check_larmor_limit.py | Where POLARIS's Larmor limit stops RAT alignment, per catalogue chi | larmor_limit_all_grids.log |
| check_mu.py | Mean molecular weight of box gas (chemistry vs GIZMO's own mu; drag-weighted mu) | mu_check.log |
