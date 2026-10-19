# Session010 reduced data

`vista_20261018/` contains byte-verified copies of the latest top-level CSV,
NPY and NPZ scientific arrays. NumPy files use `allow_pickle=False`. The full
[data dictionary](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/session010_summary_20261019/reports/REDUCED_DATA_GUIDE.md)
is authoritative; inspect CSV headers before using column positions.

| Tables | Meaning |
|---|---|
| `tracker_*.csv`, `track_*.npy`, `orbit.csv`, `orbit.npz` | Coordinate tracks, separation and observed arc; exclude gaps and truncate binary curves at strict common enclosure |
| `accepted_horizon_*.csv`, `horizon_*.npy` | Accepted individual/common candidates; strict common enclosure is a separate test |
| `matter_components.csv`, `matter.npy`, `removals.npy` | Component census/rest mass and matter-only Jz; removals use alpha<0.05 |
| `raw_quadrupole.csv`, `raw_wave.npy` | Raw R40 modes, no strain; NPY retains source-case boundaries |
| `history.npy`, `mesh.npy`, `amr_history.csv` | Native constraint-history reduction and mesh activity |
| `matter_near_r40.csv`, `cohort_tags.npy` | Extraction-shell matter and fixed movie cohort identities |

Companion [JSON receipts](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/vista_analysis/REDUCTIONS.json)
include `common_enclosure.json`, `horizon_joins.json`, source inventories,
AMR/native audit and scientific reduction metadata. Frames and full reduced
field/particle caches remain only in the intact summary at
`../session010_summary_20261019/workspace/runtime_s10/analysis/through_segment002_20261018/`.
Use its corrected `cache/fields/`, not superseded `cache/fields_pre_level_correction/`.
No native checkpoint or full field/VTK dataset is included in this folder.

The new [recheck](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/SCIENTIFIC_RECHECK.json)
independently reproduces enclosure, observed orbit arc, raw mode peak and
particle-ledger closure. Plotting/reductions belong in Slurm allocations.
