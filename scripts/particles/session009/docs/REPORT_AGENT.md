# Session 009 implementation and durable handoff

Updated October 5, 2026, Pacific time. Initialization and production remain queued. Authoritative ledgers are AMD `control/state.json` and Slurm accounting, plus Anta `evidence/active_archive_jobs.json`. This is an implementation/status report, not a completed scientific calculation.

## Frozen source and provenance

Independent `code/` checkout, branch `project/GI-in-cluster`, based on pushed Session 008 `c3873b4e0f46ab9397c2f270cb7139515013a0f3`; origin `git@github.com:JiaxiWu1018/athenak.git`. Session 008 and unrelated working trees/jobs remain unchanged.

- Compiled AMD source: **6892be3e3f04ec573f91cb2034bc9d3009a3bdff**. AMD `athenak/` remains frozen at this commit despite later analysis/documentation commits.
- Kokkos: **6739bc623081648af9e752b616d9671527922cbf**.
- Restart-header repair **263dcf21a7f6ce6ab1ccd8f1dd595ae0ca79c8c1** is present by ancestry; real >16 KiB header verification remains in the startup gate.
- C++ diagnostic change `fe68d684`: print GW time at 17 digits before the timestamp; mode values/ordering unchanged. No pusher force change.
- Scientific input commit `1519986f` is separate from operations/analysis changes.
- Post-build operations: `fccae16a` analysis semantics; `3551e1bd`/`28575d86`/`b2a8c66e` archive/calibration fixes; `27cb8d42` initial science report refresh; `37700af142794d4d488d2899537ecbc3baee4fc4` unknown-checkpoint/report semantics. Final stop/deadline guard ced70045 is also pushed. All pushed successfully. Later docs commits are separate from the compiled source/runtime scripts.

| Asset | SHA256 |
|---|---|
| Canonical input | `164002e107ae3023c37b206a4e83923410c38c5de7c18c9f704afd48f947ffff` |
| GI executable | `9717c98bed18a8d4307286115a402920aa2ca2fbf254049841aa3719b806c515` |
| Built-in wave executable | `0e496446ba7791365285c2c53e4bb516d9a4e3fd3c401d6ec2cc0c06916b1764` |

Exact runtime script hashes/limits: `evidence/frozen_config.json`, copied from AMD `control/config.json`. Previous configurations are preserved in AMD `evidence/frozen_config_before_*.json`. Only operational/analysis scripts were updated after compilation; canonical input, compiled source and executables stayed fixed. Pending jobs were temporarily held, then released after binding checks passed. These were preparation holds, not scientific retries or user stops.

`src/pgen/particles/gi_cluster.cpp` reads the explicit left/right bulk-vy input, boosts thermal four-velocities and converts using the initialized geometry. New combined initialization/weights are generated in the new run; no Session 008 checkpoint is used. Inherited `s7` generator filenames are historical output-contract names, not reused boosted/checkpoint data.

## AMD build/runtime

Root: `/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005`.

Modules: GNU12.2/OpenMPI4.1.8/CMake3.25.2/prun2.3/ROCm6.4.1; recorded UCX1.18.1. HIP/GFX90A, MPI enabled, cc/hipcc, -O3. Two build directories for GI and the existing built-in linear wave. Exact CMake commands: `scripts/amd_build.sbatch`; compiler/module/source evidence: `evidence/build_provenance.txt`. Build452576 completed0:0 on k006-004-v2 in644seconds; default ROCm7.2 was explicitly replaced by6.4.1.

Launch: prun, four ranks/node, one rank/GPU, `--kokkos-map-device-id-by=mpi_rank`; `OMPI_MCA_btl_vader_single_copy_mechanism=none`, OMP/OpenBLAS threads1. Automatic PML selection retained. Every production allocation runs the actual all-rank communication probe before loading data. Outputs use unique `runs/segment_NNN` working directories with input/provenance, never an old campaign directory.

## Tests and current jobs

Perseus allocated QA11050/11055/11059/11065/11066/11067,11080,11089,11092,11093,11094,11095 passed nine operations/horizon-join checks and syntax checks at their respective revisions. Checks cover archive-before-delete, raw node-hour reservation, phase gaps, synthetic fixed-frequency integration, strict horizon joining and complete wave floors. No full Session007 validation campaign was repeated.

AMD wavegate452578 completed0:0 in834seconds on three nodes k005-002/k005-003/k005-009, twelve ranks. All12 Alltoall probe reports had zero errors. Existing generator tlim is in periods:32/16 periods at λ2.5/5 give physicalt80. Static dx.25/.125/.0625, RK4/CFL.4 and inherited gauge/dissipation. Area-weighted sine/cosine/constant fits use all leaf plane cells; endpoint phase is compared modulo2π.

| Wavelength | Fractional amplitude error | Phase error(rad) |
|---|---:|---:|
|2.5|.0024702188121441537|.007031620893847169|
|5|.0006263057437972952|.00008929511526326572|

Both pass≤.1/≤.2. This is axis-aligned linear evidence, not nonlinear spherical convergence or a check of every coarser production transition.

The original wave inputs disabled extraction, so Weyl arrays were uncalculated zeros. Completed452666 enabled extraction for eight cycles at both wavelengths; it compares evolved saved rΨ4 with the TT limit at actual sliced coordinates. The original propagation receipt is preserved. `weyl_convention.json` records input/snapshot hashes and both passed cases: scales .9999647247660314/.9999660143710315, residuals .0000109177/.0000137038. Job completed0:0 in7seconds on3nodes (.005833333nodeh). Strain now awaits sufficient uninterrupted production coverage rather than convention verification. Plus-polarization scale/sign is checked numerically; the imaginary sign is audited from source tetrad/contraction, not a second polarized numerical test. See `evidence/TT_CONVENTION.md`.

| Host/job | Role | Nodes/max wall | Recorded state |
|---|---|---|---|
|AMD452576|Build|1/.5h|Completed0:0,644s|
|AMD452578|Propagation|3/2h|Completed0:0,834s|
|AMD452580|Full5M memory/output/MPI/restart gate|12/4h|Pending resources|
|AMD452582|Gate inspection/first production submission|1/.5h|Pending dependency|
|AMD452666|Convention calibration|3/.5h|Completed0:0,7s;passed|
|Anta2333|Wave archive/verify/status|1,oneA100/4h|Completed0:0,115s|

Full startup measurements remain pending: actual leaf spacings/counts, five-million-particle ledger, finite positive weights, centers/thermal/boost signs, clumpJz>0 and momentum residuals, all48GPU memory<85%, finite required fields/outputs, batch paths, uninterrupted-versus-restart comparison and large-header checkpoint. Accepted reference0..2 and matched output2..5 cycles join production history; duplicate validation branches are excluded. Future collapse mesh differs from startup; runtime limits halt for review without reducing particles/resolution.

The lightweight Session007 comparison uses its retained boosted-startup receipt with matched time/rectangular regions/coordinate-volume norms and no field mask. Enlarged unmatched global empty volume is not a better-constraint claim. Initial t0 constraint arrays precede calculation and are not valid zero-error evidence; initialized bounded-startup fields are used where times/regions actually match.

## Accounting, scheduler fixes and limits

AMD actual **.8797222222 raw node-hours** = (644+3×834+3×7)/3600. Registered maximum possible total49.3797222222; pending jobs currently zero actual usage. Anta actual **.0641666667 node-hours/GPU-hours**, two of96 jobs;2333/2334 completed in115/116seconds. Initial archive snapshot83,226,624bytes. Current AMD campaign/whole-user allocated storage sample is pending startup. Anta2334 completed0:0 and verified both convention runs; all four test archives are checksum-verified. No monetary tariff is available.

Initial AMD1.5h devel request was rejected before allocation; corrected to.5h, no scientific retry. Evidence: initial_submission_rejection.md and aborted configuration.

Anta rejected eight requested CPU cores. `/etc/slurm/job_submit.lua` allows at most four; its GPU error came from parsing the unsupported CPU token. Corrected to four cores, one explicitly typed A100,64GiB. Scheduler test-only2332 was not an allocation; actual2333 succeeded. Rejection evidence/flags were preserved before clearing reviewed scheduler-error stops; no USER_STOP was cleared. Future rejected submissions halt cron for review.

Configuration creation UTC2026-10-05T22:01:43.732344Z; deadline UTC2026-11-19T22:01:43.732344Z. Hard t400/10000rawAMDnodeh/45days includes preparation/failures/inspectors/AMDanalysis. At most90 production segments,≤12h each, application11h20m. Active full exposure reserved before submission. Locked state, unique names and recovery of the accepted-sbatch/state-write window prevent duplicate launches. One successor only; firstt12 then50M milestones. Numerical failures have no automatic retries.

Sticky USER_STOP/REQUEST_STOP, source/input/executable/script checks, progress/heartbeat checks and clean checkpointing apply. Scientific stop requires the accepted-common/enclosure, usable in-band ringdown and checked gaps receipt plus100savedunits after outgoing peak; otherwise hard caps. Exact controls in README; never cancel unrelated ST `s9_prod` by name.

AMD allocated five-minute watchdog checks entire user root, tolerates transient du errors, warns1.5TiB/stops1.7TiB/projected1.9TiB; session1.25TiB with256GiB finalization reserve. GPU limit85%; actual allocated samples under runs/*/storage_status.json. Latest3verified checkpoints stay; unarchived old checkpoints remain until destination verified.

Anta root `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005`, about27TiB filesystem free, /data2full. Cap16TiB plus512GiB free reserve,≤96four-hour allocated jobs. Sealed immutable manifests hash science AND checkpoints; Anta pulls, verifies sizes/SHA256, records ARCHIVE_VERIFIED, then permits scoped AMD cleanup. These are hash-verification receipts, not cryptographically signed documents. Both propagation runs verified by2333.

Metadata-only cron JEANS9_20261005_METADATA contacts AMD every5min and submits allocated work when sealed data exist. No GPU polling allocation. Initial status report update_0_2333 has no binary science data; initial science report refreshes when verified full-particle reference arrives, followed by12/50/...400/final. This workflow needs no persistent agent/shell.

## Analysis and next actions

On an Anta allocated node:

```sh
/home/jiaxiwu/miniconda3/bin/python /data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/scripts/analyze_s9.py --root /data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005 --milestone 50
```

Restart assembly checks overlaps/gaps and preserves complex raw modes. Strict accepted AH summaries join uniquely by same-run time/area/rmin/center, not finder-iteration-as-cycle. Coordinate phase unwraps only within live intervals; binary phase stops at accepted common enclosure/unresolved coincidence. Covariant matter Jz=Σµ(xuy−yux), separately from horizon/removed J. Never infer BH momentum as mass times coordinate speed.

Conditional strain: uniform.025 interpolation, linear detrend,5% cosine edge taper, fixed-frequency double integration at.003/.006/.012; sufficient contiguous causal coverage and verified convention required. Ringdown review records band, approximate propagation time, gaps and extraction-shell matter. Single radius offers no radius dependence/infinity extrapolation. Session008 comparison uses retained sharedt≤12 only, with no distant-merger-waveform claim.

Movie colors gray/blue/orange/green for envelope/left/right/accepted common. Deterministic rendering subsample leaves evolved count unchanged. Fixed-grid density uses all surviving particles, |z|<.7,256² bins/fixed color limits: coordinate rest-weight surface density, not proper energy density. Actual frame times include restart/final extras. Representative frames and plot values remain pending review when available.

Next: allow queued gates; inspect ledger/mesh/restart/memory/MPI receipts.452582 submits t12 production only on successful gate. Check convention receipt before strain. Failures preserve evidence and halt; no speed/separation tuning or surveys. Exact pending jobs and push status in evidence/CURRENT_STATUS.md.

Perseus holds about57MiB source plus small inputs/scripts/docs/QA evidence, retained without automatic deletion. **No Session009 raw particle/volume/checkpoint data onPerseus.** Large data, logs, reductions and movies stay AMD/Anta under caps.
