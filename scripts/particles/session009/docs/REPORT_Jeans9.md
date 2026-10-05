# Jeans-in-cluster Session 009 — setup and startup status

Updated October 5, 2026, afternoon Pacific time. **The workflow is submitted; the full-particle startup check is queued. No Session 009 binary evolution result exists yet.** Current machine ledgers take precedence over this dated report.

## Scientific setup

Start from t=0 with five million particles and the approved local orthonormal Lorentz speed **0.133215**: left clump −y, right clump +y. Retain the established envelope/clump masses, centers, thermal sampling, widths, deterministic seeds and immutable tags. The canonical input explicitly gives both boosts; the existing generator reads them and boosts the thermal four-velocities before converting to stored covariant momenta. Session 008 remains preserved; none of its checkpoints is used as initial data.

Companion support is an approximate prescription, not an exact general relativistic circular orbit. Initial K_ij≈0 leaves the local momentum constraint unsolved. No new force, recoil, imposed internal spin or radial boost was introduced. All time units use **M_ref=1**, the inherited reference unit. Source/model masses .76/.12/.12, sampled rest masses, measured horizon masses and any ADM estimate are distinct. Sampling residuals will be measured without forcing cancellation.

| Setting | Configuration |
|---|---|
| AMD allocation | 12 exclusive MI210 nodes; 48 GPUs/MPI ranks |
| Domain/root mesh | [−1024,1024]³; 256³ cells; root dx=8 |
| MeshBlocks/refinement | 32³ cells; physical levels 0–11; configured finest dx=1/256 |
| Initial compact clump grids | Inherited towers initially reach dx=1/64; actual balanced mesh awaits measurement |
| Continuous wave region | dx≤1/4 throughout r≤56 |
| Outer transitions | dx≤1/2 to r72, 1 to r104, 2 to r168, 4 to r296 |
| Adaptive refinement | Löhner αψ⁷ threshold .2; tracker_floor=false |
| Evolution/removal | RK4/CFL .4, inherited protections; remove α<.05; horizon-driven removal OFF |
| GW extraction | One coordinate sphere r50; raw complex rΨ4, ell=2…8, every .025M_ref |
| Particle/plane output | Full particles every .25; three diagnostic planes every .1 |
| Volumes/checkpoints | Every 10, plus clean final checkpoints |

Outflow boundaries are retained. The weak-field gauge estimate from the boundary to r50 is about 689 reference time units, buffering t400; evolving metric/shift speeds and boundary behavior still require inspection. Coarser interfaces outside r56 can reflect part of the signal.

## Completed checks

Both AMD HIP/MPI executables built with the pinned ROCm 6.4.1 stack. The existing linear-wave generator passed the approved 80-unit propagation tests across static refinement interfaces:

| Wavelength | Amplitude error | Phase error | Approved thresholds |
|---|---:|---:|---|
| 2.5 | 0.2470% | .007032 rad | ≤10%, ≤.2 rad |
| 5 | 0.06263% | .00008930 rad | ≤10%, ≤.2 rad |

All twelve ranks in the three-node test allocation exchanged data without detected errors. These are axis-aligned linear tests, not nonlinear/spherical convergence evidence or a check of every coarser production interface. The intended interpretation band remains f≤.4/M_ref.

The propagation inputs disabled extraction, so their Weyl fields were uncalculated zero arrays. A separate eight-cycle test with extraction enabled also passed. Its measured scale was .9999647/.9999660 at the two wavelengths, within .004% of the analytic convention; residuals were below .002%. This confirms plus-polarization normalization/sign; the imaginary sign is audited from the source tetrad. Strain still requires enough contiguous production coverage; raw complex rΨ4 remains primary.

## Submitted work and resources

| Job | Work | Recorded status |
|---|---|---|
| AMD 452576 | Build both executables | Completed, exit 0 |
| AMD 452578 | Propagation gate | Completed, exit 0; passed |
| AMD 452580 | Full initialization/MPI/memory/output/restart check | Queued for 12 nodes |
| AMD 452582 | Inspect gate; submit first production segment if valid | Queued by dependency |
| AMD 452666 | Enabled-extraction convention check | Completed, exit 0; passed |
| Anta 2333 | Archive and verify wave tests; initial status report | Completed, exit 0 |

Measured AMD usage: **.879722 raw node-hours** (.178889 build + .695 propagation + .005833 convention check). Pending jobs have zero actual cost so far; current registered maximum possible total is 49.379722 node-hours. Anta’s first job used **.031944 node-hours/GPU-hours**; second archive job 2334 was submitted automatically for the convention files. Each counts toward the 96-job cap; latest accounting is in its ledger. Monetary cost is unknown without an actual account tariff.

Anta measured **83,226,624 bytes** in the archive at its initial report snapshot; `/data3` had about 27 TiB free. Both propagation runs were copied and verified by checksum. Current AMD campaign/whole-user storage measurements await the full startup watchdog; filesystem-wide free space does not replace the effective user budget.

Still pending: actual mesh/spacings, complete particle ledger and momentum residuals, all 48 GPU memory peaks below 85%, finite evolved fields and required outputs, batch output paths, and uninterrupted/restart comparison with a real large-header checkpoint. Long production is conditional on these checks. Accepted startup cycles and the t12 continuation form the beginning of production rather than a discarded pilot.

## Limits, analysis and limitations

Stop at **t400**, **10,000 raw AMD node-hours**, or **45 calendar days**, whichever applies first. Earlier scientific termination requires a strictly accepted common horizon enclosing both tracked objects, usable in-band outgoing ringdown with gaps checked, and **100 saved time units after the observed post-merger waveform peak**. A common horizon alone does not stop the run; merger is not guaranteed.

Production jobs last at most 12 hours, with evolution limited to 11h20m to reserve finalization. A locked controller permits one successor segment, retains sticky user-stop intent, and halts numerical failures for review. Continuation, archival and reporting require no persistent agent or interactive shell.

AMD session storage is capped at 1.25 TiB; whole-user warning/stop/projected limits are 1.5/1.7/1.9 TiB. Keep the latest three verified AMD checkpoints; older checkpoints/science files require verified Anta copies before approved removal. Anta archive cap: 16 TiB; at most 96 four-hour allocated jobs, each one A100/four CPU cores under the current site rules.

Initial-data, t12, t50/100/…/400 and final reports will include separation from tagged matter/live trackers/simultaneous accepted horizon centers, trajectories and envelope-relative motion, phase/revolutions after both individual horizons form, radial/tangential motion, accepted masses/spins and failures, component/removal/angular-momentum histories, constraints, numerical health and waveforms. Conditional fixed-frequency strain includes cutoff sensitivity. Missing diagnostics remain gaps; binary revolutions stop at a proven accepted common horizon or unresolved coincidence.

Central-orbit, context and fixed-grid density movies reuse the corrected pipeline. Rendering selection never changes the simulated particle count. Representative frames and plot values must be reviewed when available. No scientific plots or movies are claimed yet.

One extraction radius cannot establish radius dependence or extrapolate to infinity. Coordinate separation/phase are not gauge-invariant orbital elements. Shrinking separation alone does not establish radiation-driven inspiral. Removed-particle and horizon angular momentum are separate diagnostics, not an exact conserved sum.

## Locations

- Setup: `notes/APPROVED_PLAN.md`; canonical input: `inputs/gi_cluster_s9.athinput`.
- AMD: `/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005`.
- Anta: `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005`.
- Anta products: `analysis/update_<milestone>_<job>/`; `analysis/latest.json` identifies the latest. Current `update_0_2333` is a preparation/status report with no binary science data.
- Technical provenance: `REPORT_AGENT.md`; live commands: `README.md`; dated handoff: `evidence/CURRENT_STATUS.md`.

Perseus retains source, inputs, scripts, Markdown and small verification evidence only. No Session 009 particle, volume or checkpoint files have been copied here.
