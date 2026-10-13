# Session009 — progress report and continuation handoff

Updated **October 12, 2026 05:01 PM PDT**. This report supersedes the earlier preparation/status text, preserved under `history/handoff_20261012/`. Read [REPORT_AGENT.md](REPORT_AGENT.md) for exact checkpoint, source, job history and continuation steps.

## Current result and why the run stopped

The fresh Session009 run reached **t = 9.33125 M_ref** and is stopped. No registered AMD or Anta job is running or queued at this handoff. The production job ended cleanly and saved a verified checkpoint after one GPU reached **85.075% memory use**, crossing the approved **85% limit** on October 8 at 4:01:27 PM PDT. This was a preventive resource stop; it was not an out-of-memory crash, a numerical failure, or completion of the scientific goal.

Both clumps have accepted individual apparent horizons by **t = 8.48594**. Only **0.0042207 revolutions, about 1.52 degrees**, were measured afterward. The available run is too short to establish a nearly circular orbit, sustained inspiral or secular orbital shrinkage. No accepted common horizon, merger or ringdown waveform exists. Neither the early t12 review nor any 50M milestone was reached.

The checkpoint and science outputs are checksum-archived on **Anta /data3**, and plots and movies have been generated and reviewed. The word `final` in the stopped-run analysis directory describes the closeout of this partial attempt; it does not mean a completed inspiral–merger–ringdown calculation.

## Setup and changes from earlier sessions

Session007 used envelope-only orbital boosts and showed rapid inward motion. Session008 introduced the approved approximate companion-supported boost; its short orbital arc did not establish a full revolution. Session009 starts independently at t0 with the same approved boost and a much larger, continuously refined wave region. The R50 proposal was changed at Jiaxi's request to **one extraction surface at coordinate R40**, with fine propagation spacing through R46.

| Parameter | Approved and actually used |
|---|---|
| Envelope | Source/model mass 0.76, center (0,0,0), 3,000,000 particles; independently solved spherical Einstein–Vlasov envelope, areal radius 30 |
| Left/right clumps | Source/model mass 0.12 each, centers x = −3/+3, 1,000,000 particles each |
| Sampling | Gaussian sigma = 0.70; internal orthonormal spread s = 0.02; established deterministic seeds including 4001 and immutable component tags |
| Initial orbital boost | Local orthonormal Lorentz speed 0.133215; left −y, right +y; positive orbital J_z |
| Total initial particles | Exactly 5,000,000; no imposed envelope recoil, internal spin or radial boost |
| AMD allocation | 12 exclusive MI210 nodes, 48 GPUs/MPI ranks |
| Domain/root grid | [−1024,1024]^3; 256^3 cells; root dx = 8 |
| MeshBlocks and refinement | 32^3 interior cells, 4 ghost cells; physical levels 0–11; central allowed minimum dx = 1/256; initial compact clump towers reach 1/64 |
| Wave propagation region | dx ≤ 1/4 throughout r ≤ 46; dx ≤ 1/2, 1, 2, 4 to r72, 104, 168, 296 |
| Adaptive refinement | Löhner estimate of alpha*psi^7, threshold 0.2; tracker_floor=false |
| Evolution | RK4, CFL 0.4; inherited gauge, damping, conservative deposition and pusher protections |
| Particle removal | alpha < 0.05; AH-driven removal OFF |
| Gravitational-wave data | Raw complex rPsi4, ell = 2–8, coordinate R40, cadence 0.025 M_ref |
| Other output cadences | Full particles 0.25M; three diagnostic planes 0.1M; full 3D snapshots/checkpoints 10M, plus checkpoints at clean stops |

The initial R40 mesh has **5,384 blocks**, compared with 7,456 for R50: 2,072 fewer blocks (27.79%). The final checkpoint has 5,020 blocks; tracker locations reached dx=1/256. Neither this count nor an allowed refinement ceiling describes every block's actual spacing.

The boost is an approximate Newtonian companion correction applied through the existing local Lorentz transformation, not an exact general-relativistic circular-orbit construction. The initial K_ij is approximately zero and the local momentum constraint is unsolved. No speed tuning, added force, eccentricity survey or constraint solver was introduced.

**Time and the 50M updates use M_ref = 1**, the inherited reference unit. Source masses sum to one, but sampled rest mass is approximately **1.04465**, and measured horizon masses are different again. No equality to ADM mass is assumed.

## Checks completed before and after production

The full initialization census, component tags, finite positive weights, centers, thermal samples, boost signs and positive orbital J_z passed. The net initial covariant momentum was approximately **(0, −3.7e−7, 0)**, a residual fraction 8.59e−6 relative to the sum of clump momentum magnitudes; no new symmetry or recoil forced cancellation.

The actual 48-rank allocation passed communication and startup checks. Startup memory peaked at about **70.34%**. Brief uninterrupted/restart and clean-stop tests passed, required output paths were checked from batch working directories, and a real checkpoint with a 35,007-byte parameter header exercised the retained restart-header repair. The final production checkpoint has a 52,193-byte header.

The retained linear propagation tests at wavelengths 2.5M and 5M over 80M passed the approved amplitude/phase bounds. This is limited test-wave evidence; it does not establish convergence or accuracy of the nonlinear binary waveform.

Allocated stopped-run analysis checked particle accounting, accepted horizon associations and finite waveforms/history. Anta job2423 checked all fields in the three latest saved metric planes: all finite. This is not a full 3D or convergence audit. Local constraint comparisons are retained, but the global Session007/009 comparison is unmatched and no fully matched physical momentum-constraint ratio is claimed.

## Available scientific measurements

| Measurement | Available finding |
|---|---|
| Surviving-particle center separation | 5.999221 at initialization to 6.028254 at t9.33125, about +0.484% |
| Paired accepted horizon-center separation | 6.060942 to 6.067694 over t8.64531–9.32969; range/mean about 0.142% |
| Individual horizon masses | Approximately 0.074953 and 0.074854 at t9.32969, in inherited reference units |
| Inherited horizon spin estimates | chi ≈ 0.002973 and 0.004864; qualitative diagnostics with no precision/convergence claim |
| Accepted measurements | Left 423 rows, right 463; uniquely associated with their summaries; no accepted common horizon |
| Post-formation phase | About 1.52 degrees; insufficient to establish circularity or sustained inspiral |
| Raw waveform coverage | 318 finite, strictly increasing samples through t9.3265625 at R40 |

Particle centers, trackers and horizon centers have different definitions. Removal changes surviving-particle centroids; all reported orbital diagnostics are coordinate quantities. Gaps and rejected/stale horizon surfaces are excluded rather than converted to physical zeros. A small separation change cannot identify gravitational radiation as its cause.

Exactly **3,996,300 particles survive**. The remaining **1,003,700** were removed by the approved lapse criterion:

| Component | Initially simulated | Surviving | Removed |
|---|---:|---:|---:|
| Envelope | 3,000,000 | 2,998,001 | 1,999 |
| Left clump | 1,000,000 | 498,891 | 501,109 |
| Right clump | 1,000,000 | 499,408 | 500,592 |
| Total | 5,000,000 | 3,996,300 | 1,003,700 |

The exact event ledger agrees with full-particle snapshots and the checkpoint. Matter angular momentum uses covariant particle momenta; removed-matter J and horizon spin are separate diagnostics, not an exact conserved sum.

There is not enough time for radiation from the central collapse/possible later merger to reach R40 and provide a complete waveform. Strain integration and radiated energy/angular momentum claims are deferred. With one extraction radius, radius dependence and extrapolation to infinity cannot be measured. The intended frequency band f ≤ 0.4/M_ref and ten cells per shortest intended wavelength are planning limits, not convergence evidence.

## Plots and movies to inspect

- [Separation versus time](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/separation_vs_time.png): compares surviving-particle centers, trackers and paired accepted horizon centers; watch the brief coverage and horizon gaps.
- [Orbital motion](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/orbit.png): shows the short trajectories; the saved arc does not establish a full orbit.
- [Accepted horizon measurements](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/accepted_horizons.png): individual masses and small spin estimates, with gaps and no common-horizon curve.
- [Central movie representative final frame](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/central_frames_00036.png): separated blue/orange clumps at t9.331.

Full products are on Anta in **`/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/analysis/update_final_2408/`**: `central.mp4`, `context.mp4`, `density.mp4`, and separation/orbit/horizon/constraint/removal/raw-waveform figures. The central movie shows tagged clumps, the context movie shows the envelope, and the density movie uses deposition onto fixed Cartesian bins to avoid native-AMR seams. Rendering subsampling never reduces the simulated particle count. Plot values and representative movie frames were visually checked; provenance is in `evidence/visual_review_2408.json`.

## Costs, storage and original limits

| Resource | Recorded usage |
|---|---|
| AMD total | **72.697778 raw node-hours**, including preparation, failed allocations, production and inspectors |
| Production job455477 | 20,119 allocated seconds × 12 nodes = **67.063333 node-hours** |
| Anta | **3.851944 allocated node/GPU-hours**, 11 of 96 registered jobs; all completed |
| AMD storage | Last allocated sample before archival cleanup: Session009 0.217TiB, whole user 0.912TiB; not a current post-cleanup measurement |
| Anta archive | Last allocated measurement 0.644TiB; checkpoint and science copies checksum-verified |
| Monetary cost | No account tariff supplied; no dollar estimate |

The corrected historical production average is 7.211111 node-hours/M over t0.03125–9.33125. It is not a forecast for later collapse, orbits or merger. An earlier generated report used stale controller time; originals are preserved and the false extrapolation was withdrawn.

Original hard limits remain **t400 or 10,000 raw AMD node-hours**, at most 90 finite evolution segments of no more than 12 hours with 40 minutes reserved for finalization, and the original deadline **November 19, 2026 02:01 PM PST**. AMD session cap is 1.25TiB, with whole-user warning/stop/projected limits 1.5/1.7/1.9TiB. Anta archive cap is 16TiB and analysis/archive allowance 96 jobs of at most four hours. Resetting the detector or restarting does not renew these budgets.

An earlier scientific stop requires a strictly accepted common horizon enclosing both objects, usable in-band outgoing ringdown at R40, checked gaps and at least 100 saved M after the observed outgoing waveform peak. None of these completion conditions has been reached.

## Earlier memory remedy and proposed continuation

The read-only review found a directly relevant earlier repair in **Singularity Session002**, commit **`7093f488d926338ed67eef9af51db28395cc8873`**. It retains particle communication/storage buffers and enlarges them only when necessary, with 12.5% growth margin. Its matched test reduced host-memory growth by 88.4% and GPU-memory growth by 54%; residual growth was not eliminated. [Historical report](/data/jiaxiwu/NRPIC/Singularity/session_02_validation_20260907/REPORT.md:181).

The equivalent buffer-reuse implementation is **absent from Session009's frozen compiled source**. Four relevant source files were compared read-only against that frozen revision. This is a strong candidate for the observed growth; the Session009 stop alone does not prove that all memory increase has the same cause. Mesh changes, rank populations and reserved capacity still need measurement.

Recommended after approval: adapt the existing repair while preserving Session009's newer diagnostics, validate a short continuation from **t9.33125**, and select shorter checkpointed segments using the measured restart memory and growth rate. Earlier ST-migration Session007 completed five bounded segments with memory resets and no lost progress. Keep the 85% guard; any capacity change must retain measured mesh headroom and approved physical resolution. Shorter segments must also respect the original 90-segment/calendar/resource caps.

**The repair has not been implemented, built or tested in Session009.** Jiaxi explicitly requested the solution without implementation; this handoff request authorizes documentation. The next chat should read the technical report and discuss authorization for the concrete repair/continuation before execution. No job was submitted or restarted for this handoff.

## Monitoring and where to start the next chat

Jiaxi cancelled the two-hour checks on October8 at 7:33 PM PDT. The campaign cron entry is absent and `control/WAKE_STOP` persists on Perseus. Anta's five-minute trigger removed itself after stopped-run archival/analysis completed. AMD `REQUEST_STOP` and `RESOURCE_STOP.json` remain set. These states were checked again for this handoff; no monitor was re-enabled.

Start with **[REPORT_AGENT.md](REPORT_AGENT.md)**, then `evidence/CURRENT_STATUS.md`, `notes/APPROVED_PLAN.md` and `manifest.json`. The final verified checkpoint defines current progress; controller `state.json` still displays stale t0.03125. Exact status/stop commands and required repairs are in the technical report. Documentation push status is recorded in `evidence/PUSH_STATUS.txt`.
