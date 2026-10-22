# Hourly monitoring: first review and constraint-mask investigation

Hourly AMD analysis474359 completed260s/1node with plots, report and two1080p movies;
138 exact cohort states per movie, both fully decoded.29compact files18,931,442bytes
checksum-transferred to Perseus by18818. Next AMD refresh474369 is scheduled on the
hour; next Perseus review transfer18819 at quarter past. Actual starts depend on
Slurm resources. New cache tools0c367de8 passed6focused checks (QA474361,9s/1node).
The next report job uses this source; the first immutable release uses811a3672.

Latest audited checkpoint is11.470312500001118/cycle3609:3,434,414alive plus1,565,586
lapse removals=5M. Exact IDs/rest weights/native and online Jz pass; unknown native
energy count0 in this state. Prior unknown energies remain in their historical
records. Both individual AHs accepted; no strict common horizon yet. Production
segment5 job474260 completed7458s/6nodes. Continuation474357 is PENDING/Priority,
requesting24h with23h20m app limit and1Mref periodic checkpoint output; actual
submitted Slurm batch verified. Saved-state regression474293 has Nice10000 so
production takes priority;474294 remains dependent. Do not submit duplicate runs.

## Scientific observations and limits

At11.4703125 the individual AH masses are0.10676595 and0.10643040Mref; their
coordinate-rotation spin estimates are0.00414333 and0.00339020. Both masses grow
smoothly through accepted states. Only0.0149568 post-formation turns are observed
over valid intervals so far. This short interval is insufficient for a final
orbital or merger interpretation. The inherited quantized tracker positions still
make instantaneous angular/radial derivatives noisy. Final75 science and saved-state
angular checks remain incomplete.

## Constraint-domain change investigated immediately

Visual review of exterior_constraints.png found a sharp norm drop near first AH
formation. Non-evolving allocated check474376 compared existing whole-grid mask0
and exterior mask1 data, without restarting or modifying production:

| State time | Accepted bits | Whole-grid H RMS | Exterior H RMS |
|---|---:|---:|---:|
|8.428125|0|0.003821|0.003821|
|8.478125|1|0.003547|0.002501|
|8.503125|3|0.007735|0.000096|
|11.470313|3|0.024462|0.000056|

The whole-grid norm does not fall with the exterior norm. The sharp drop coincides
with both individual horizons being accepted and excluding their high-constraint
interiors from the exterior integration region. It must not be described as sudden
whole-grid numerical improvement. Unknown AH coverage at8.453125 correctly leaves
exterior quantities NaN. The exterior proper line also changes its excluded segments
as surfaces qualify; its formation-time jump is not an isolated inspiral measurement.
This evidence explains the plotted discontinuity through the changed measurement
domain; it does not prove convergence or good constraint behavior in the interiors.
No data corruption requiring a restart was found in this check. Continue monitoring
both domains and the physical/checkpoint guards.

Figures four_states.png, horizon_mass_spin.png, orbit_diagnostics.png and
exterior_constraints.png were visually inspected. Both AH glyphs are visible at
unchanged spectral sizes; missing close/common events use explicitly labelled latest
saved frames. Particle/surface depth remains unresolved as disclosed in the report.

Exact diagnostic rows and job receipt:
/data/jiaxiwu/NRPIC/GI_in_cluster/session_011_amd_ah_orbit_20261021/evidence/hourly_release_20261022/evidence/hourly_mask_check_474376/.
First hourly review:/data/jiaxiwu/NRPIC/GI_in_cluster/session_011_amd_ah_orbit_20261021/products/hourly_20261022T045010Z_474359/REPORT.md.
All raw/checkpoint/cohort/reduced data remain on AMD. No Mac transfer or raw cleanup.

Authoritative continuation and hourly handles:474357production,474369hourlyAMD,
18819hourlyPerseus; read live scheduler state. CompletedAMD74.955000nodeh before
474369. This record is timestamped evidence, not final t75 completion.
