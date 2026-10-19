# Reduced data and replotting on Perseus

Use `workspace/runtime_s10/analysis/through_segment002_20261018/` relative to the archive root. The historical `first_production_20261015/` has a smaller time range and a known AMR display offset. Both releases, caches, frames, QA images and original scripts are copied unchanged. This guide describes the **October 18** schema unless noted otherwise.

## Data dictionary

| Files | Meaning and caveats |
|---|---|
| `tracker_0.csv`, `tracker_1.csv`, `track_0.npy`, `track_1.npy` | Native 17-column tracker record plus zero-based `case` index: cycle, time, xyz, velocity xyz, level, dx/dy/dz, source, core count/rest mass, minimum lapse, AH tracking flag, case. CSV headers are authoritative. Coordinate, not invariant, positions. |
| `orbit.csv`, `orbit.npz` | Time, coordinate separation, phase within valid intervals, observed cumulative arc excluding gaps, coordinate angular speed, and valid-pre-common mask. NPZ keys are `time`, `distance`, `arc`, `valid`. Phase and arc are radians; multiply by 180/pi for degrees or divide by 2pi for turns. End binary curves at first strict common enclosure, not at coincident post-merger tracker positions. |
| `accepted_horizon_0.csv` … `_2.csv`, `horizon_*.npy` | IDs 0/1 individual, 2 common; 21 native fields plus `rmin_cells` and case. CSV includes cycle,time,M_BH,Sx/Sy/Sz/S,area,hrms,hmean,meanradius,rmin,M_irr,spin,Px/Py/Pz/P,center_xyz,rmin_cells,case. Accepted consumer candidates still require strict common enclosure for the first common time. Post-common individual candidates are retained but individual curves are truncated. No accepted common candidate after 98.1984375 Mref. |
| `common_enclosure.json`, `horizon_joins.json` | Strict same-time/cycle/branch association and enclosure evidence. Do not substitute the first raw finder convergence for these tests. |
| `matter_components.csv`, `matter.npy` | Eleven columns: time, component index, alive count, alive rest mass, COM xyz, alive covariant Jz, removed count, removed rest mass, removed covariant Jz. Component labels are envelope/left/right (0/1/2). These are matter-only ledgers, not total gravitational angular momentum or independent horizon-accretion fluxes. |
| `removals.npy` | October 18 rows: time, component, rest mass, covariant Jz, immutable tag, cycle. The historical first-production array has five columns and no cycle column. Low-lapse events, not AH-based removal. Original component-removal CSVs are also copied among native diagnostics. |
| `raw_wave.npy` | N×156: time; 77 real modes; 77 imaginary modes; case. Mode order l=2…8 with m=-l…l. Raw finite-radius rPsi4 at R40, not strain. Preserve restart/gap breaks and avoid unsupported ringdown or extrapolation claims. |
| `raw_quadrupole.csv` | Headered time and real/imaginary (2,-2), (2,0), (2,2), in that order. No case column; use `raw_wave.npy` for source-case boundaries. Prefer header names over positional assumptions. |
| `history.npy` | Native Z4c user-history table, with its original column header preserved in the copied `*.hst`. Proper-volume RMS H/M uses sqrt(column 3 or 4 divided by column 10), with zero-based indexing; chi>=0.0625 mask. |
| `mesh.npy`, `amr_history.csv` | Time, evolution cycle, leaf-block count. Activity is not a convergence demonstration. See `AMR_HEALTH.json` for native geometry audit. |
| `matter_near_r40.csv` | Saved particle time, alive count and rest mass in the R38–R42 shell, and total alive rest mass. Finite-radius waveform interpretation needs the surrounding matter caveat. |
| `cohort_tags.npy`, `cache/particles/*.npz` | Fixed 90,000-tag t0 cohort (40k envelope, 25k per clump). NPZ keys include `time`, `xyz`, `tag`, `group`, and `alive` (the scalar full-population alive count, not a per-cohort mask). Removed tags are never replaced. This is not the full particle population. |
| `cache/fields/*.npz`, `FIELDS.json` | 445 synchronized pixel-center native-leaf rasters, 512², central ±6 and wide ±35. Keys include `time`, `cycle`, `energy_c`, `energy_w`, `energy_xz`, `alpha`, `H`, `level`. Correct relative levels are 0–10. These are reduced slices, not the full 3D mesh/state. |
| `frames/`, `*.ffconcat`, `*.mp4`, `MOVIES.json` | Actual source-render frames, ffmpeg concatenation schedules, encoded movies and frame/time metadata. Movies run at 12 fps, 2 Mref per playback second, holding actual saved states without interpolated motion. Each current MP4 fully decoded to 1,333 frames. |

Inspect CSV headers and `.npz.files` directly before introducing new calculations. Preserve original units, masks, source-case boundaries, gaps, source-component colors and tag identity. Missing values in angular speed are intentional at invalid/gap boundaries, not data corruption. The archived scientific code is the authoritative implementation of transformations.

## Portable use

CSV files open without the native simulation. NumPy `.npy`/`.npz` files use `allow_pickle=False`. A fresh CPU Python environment with NumPy and Matplotlib is sufficient for new plots from the top-level reduced data. The original full rendering pipeline additionally uses Pillow, imageio-ffmpeg, h5py and the packaged native reader; exact dependency evidence is in [the current analysis scripts](../workspace/analysis_s10_update_20261018/). Use an allocated compute job for full plotting/movie work on Perseus, with `MPLBACKEND=Agg`.

Example data access (from the summary root; this only reads arrays):

```python
from pathlib import Path
import numpy as np

d = Path('workspace/runtime_s10/analysis/through_segment002_20261018')
orbit = np.genfromtxt(d / 'orbit.csv', delimiter=',', names=True)
keep = orbit['valid_pre_common_binary'].astype(bool)
time = orbit['time'][keep]
separation = orbit['coordinate_separation'][keep]
with np.load(d / 'cache/fields/0000.npz', allow_pickle=False) as fields:
    print(fields.files)
```

Do not invoke the archived full pipeline unchanged on Perseus: it enforces Vista allocation/native-source gates and refers to absolute source/raw paths. `FIELDS.json`, ffconcat schedules and old provenance documents deliberately preserve those original paths. `inventory/RELOCATION_MAP.json` provides archive-relative field-cache paths without rewriting evidence. Paths beginning with the recorded original workspace root can be mapped to `<your summary>/workspace/<relative suffix>` **only when the file is actually copied**; bulk raw files are not present. Cache filenames can also be enumerated directly with `Path.glob` and their own embedded time/cycle read.

The copied ARM executable is provenance only; new native runs on Perseus require a machine-native rebuild and new gates. The two Git bundles preserve native and Kokkos source history. Production scripts are historical evidence and must not be submitted just to make plots. No raw-data transfer has been performed; for new 3D variables or every particle, use the exact Vista inventories and plan an independent verified raw copy.

## Historical error handling

Keep `cache/fields_pre_level_correction/` and the first-production AMR labels as superseded evidence. The current `cache/fields/` uses the correct 0–10 relative-level convention. Historical file bytes were not patched. Checksum verification proves this copy matches those historical bytes; it does not endorse incorrect historical labels or establish numerical/physical convergence.
