# Current update — October 08, 2026 07:35 PM PDT

## Periodic checks cancelled by the user

No registered AMD or Anta job is running or queued; the verified checkpoint is still t9.33125. The two-hour checks have been cancelled at your request and persistent wake-disable intent is recorded. Usage remains 72.697778 AMD raw node-hours and 3.851944 Anta allocated node/GPU-hours. Saved plots, movies and verified archive receipts are retained. Cancellation receipt: evidence/two_hour_wake_cancelled.json.

**Simulation stopped cleanly at t9.33125M_ref after a GPU reached85.075%, crossing the approved85% memory limit.** No simulation job is running or queued. The final checkpoint and segment science outputs are verified on AMD and checksum-archived on Anta. REQUEST_STOP/RESOURCE_STOP remain set; no automatic restart, tuning or physics change. Controller time0.03125 is stale because its continuation inspector refused the stop; it is not current science time.

Strict individual horizons are accepted for both clumps by t8.48594. At t9.32969 their measured horizon masses are about0.074953 and0.074854M_ref (distinct from0.12 source parameters), with small inherited spin estimates0.002973/0.004864. M_ref=1 is the fixed inherited reference unit, not a measured horizon or ADM mass. The0.133215 local Lorentz boost remains approximate companion support, not exact circular GR initial data; K_ij≈0 leaves the local momentum constraint unsolved. Only about1.52degrees of motion are saved after both horizons have accepted measurements; this cannot establish nearly circular motion or sustained inspiral. No accepted common horizon, merger or ringdown waveform exists. Raw R40 complex multipoles are retained through t9.32656; strain is deferred for insufficient time/causal coverage.

The final count is3,996,300 surviving particles. The exact removal ledger conserves all five million initially simulated particles: envelope1,999 removed; left501,109; right500,592. No imposed recoil, new symmetry or reduction of the initial particle count. Saved metric fields in all three latest diagnostic planes are finite, checked by allocated Anta2423; this is not convergence evidence.

Usage: **72.697778 AMD raw node-hours**, **3.851944 Anta node/GPU-hours**,11/96Antajobs completed. Last allocated AMD storage was0.217TiB session/0.912TiB whole user before archival cleanup; no fresh post-cleanup size is claimed. Anta archive measured0.644TiB. Generated cost estimates using stale controller time were corrected with originals preserved; no remaining-cost forecast is issued while memory feasibility requires review.

## Recommended review products

- [Separation versus time](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/separation_vs_time.png): surviving-particle centers, trackers and simultaneous accepted horizon centers are distinct; the short post-formation interval and diagnostic gaps remain visible. Full centroid separation5.99922→6.02825; AH separation6.06094→6.06769.
- [Accepted horizon masses and spins](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/accepted_horizons.png): only strict accepted measurements, with gaps preserved; spin estimates are qualitative.
- [Central movie final frame](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/central_frames_00036.png): separated blue/orange clumps and their envelope at t9.331. Full movie on Anta:`/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/analysis/update_final_2408/central.mp4`.
- Envelope/context and fixed-grid-density movies on Anta in the same directory:context.mp4 and density.mp4. Representative final frames and plot values were inspected. Rendering subsampling never changes simulated particle counts.

This is a partial production closeout after a resource stop; no t12 or50M milestone was reached. The two-hour checks were cancelled by the user; Anta removed its terminal archival trigger after completion. Stop flags remain intact. Exact status, original limits, source/configuration hashes, outstanding memory/operations review and scoped controls are in evidence/CURRENT_STATUS.md. Receipt:evidence/scheduled_check_20261009T000844Z.json.

## Earlier dated setup and status records



# Session009 — fresh orbit and gravitational-wave production at R40

Updated 2026-10-06T22:47:06.919839+00:00. **Full startup submitted and waiting for AMD resources. No binary evolution/results yet.** See evidence/CURRENT_STATUS.md for exact jobs and controls. Historical R50 preparation/failure reports are preserved under history/r50_before_reset_20261006/.

## Aim and approved change

Session008 showed both clumps forming horizons with a short orbital arc; its t12 assessment did not establish a full orbit, merger or a distant merger waveform. Session009 starts again at t0 with the same approximate companion-supported local Lorentz speed0.133215, left−y/right+y. The companion estimate v_new²≈v_old²+Gm/(4R) motivates that approved value; it is not an exact general-relativistic circular orbit. No speed/separation optimization or new pusher force is introduced.

October6 amendment moves the one detector from coordinateR50 to **R40**, retaining a six-unit fine-grid buffer through **R46**. Exact initial seeded mesh drops7456→5384blocks,2072 fewer(27.79%). Later collapse refinement and particle work mean this does not imply an identical reduction in runtime. Domain and outer wave layers are unchanged.

| Setting | Current approved configuration |
|---|---|
| Particles | Envelope3M; left1M; right1M; total5M |
| Source mass parameters/centers | .76 atorigin; .12 atx−3; .12 atx+3 |
| Envelope/clump sampling | Independent spherical EV envelope, arealR30; Gaussian sigma.70; local internalspread.02; seed4001/tags retained |
| Bulk velocity | Local orthonormal Lorentz boost±.133215 alongy; no recoil/radial boost/internal spin |
| Allocation |12exclusiveMI210nodes,48GPUs/ranks |
| Domain/root |[−1024,1024]³;256³cells;dx8 |
| Blocks/central AMR |32³cells,ghost4;physicallevels0–11;configuredceilingdx1/256;compactinitialtowersdx1/64 |
| Refinement |Löhneralpha*psi⁷threshold.2;tracker_floor=false |
| Wave region |dx≤.25 throughout r≤46;dx≤.5/1/2/4 to72/104/168/296 |
| Evolution |RK4/CFL.4; inherited gauge/deposition/pusher protection; removealpha<.05; AH-removalOFF |
| GW |One coordinateR40 sphere; rawcomplexrPsi4 ell2–8;dt.025M_ref |
| Saved data |Fullparticlesdt.25;3diagnosticplanesdt.1;full3D/checkpointsdt10;checkpointsatcleanstops |
| Hard endpoint |t400M_ref or10000rawAMDnodeh,original45-daydeadline;mergernotguaranteed |

M_ref=1 is the inherited reference unit. Source normalization, sampled rest mass, measured horizon masses and any ADM estimate are distinct. Kij≈0 and the local momentum constraint remains unsolved; opposite global boosts do not fix it. Initialization must regenerate combined geometry/weights and boosted samples, preserving unaffected envelope assets where valid.

## Wave interpretation and stopping

Intended band f≤.4/M_ref, wavelengths≥2.5M_ref, gives at least10cells atdx.25. This is a planning heuristic. The retained80M linear/interface test passes approved amplitude≤10% and phase≤.2rad limits; it is not nonlinear spherical convergence. R40 shortens propagation but increases finite-radius/near-source/gauge concerns relative toR50. Check matter near coordinateR40 and do not confuse envelope arealR30 with isotropic coordinateR30. With one detector, no multi-radius accuracy test or extrapolation to infinity is possible.

Weak-field boundary gauge travel estimate(1024−40)/sqrt2≈696M_ref exceeds hardt400; actual metric/shift propagation and boundary diagnostics still require inspection. Coarser transitions can reflect waves.

An early scientific stop needs a strictly accepted common horizon enclosing both objects, usable inband post-merger/ringdown signal atR40, diagnostic gaps checked, and at least100savedM_ref after the observed outgoing wavepeak. A common horizon alone is insufficient. If merger/ringdown is absent or outside supported coverage at a cap, report the incomplete outcome.

## Current findings

The old R50 attempt did not evolve: a strict input override was missing. The input now explicitly defines output7/last_time. The repaired R40 actual-executable input preflight453630 passed; full5M startup453631 is pendingresources, followed by inspector453632. GPU memory, initialization residuals, finite evolved fields, real large-header restart and production timing remain pending. Reused propagation errors are .2470%/.06263% amplitude and .007032/.00008930rad phase at wavelength2.5/5; these are test-wave findings, not binary radiation measurements.

No new separation/orbital/horizon/waveform result or movie is available. Earlier Anta analysis/update_final_2353 is the preserved failed R50 closeout. Do not infer a merger waveform from a central merger before radiation arrives at the detector.

## Monitoring, analysis and budgets

Every2hours, the Perseus monitor checks registered jobs and wakes this same Codex thread when idle. Five-minute Anta metadata checks submit allocated archive/analysis work as sealed outputs arrive. Successful AMD inspectors chain finite checkpointed jobs independently of the agent. Initial/t12/every50M/final reports include separation, absolute/envelope-relative motion, post-horizon revolutions and radial/tangential ratios, strictly accepted horizons, particles/removals/covariantmatterJ, numerical health, rawGW and supported strain. Coordinate shrinkage alone does not prove radiation-driven inspiral. Figures/movies require representative visual/value review once data exist.

AMD used0.9225rawnodeh so far; Anta0.096944444node/GPUh(3/96jobs). Queued AMD startup/inspector reserve48.5nodeh; maximum49.4225 so far. All earlier jobs count; reset did not renew budgets/deadline. No monetary tariff is known. Session009 AMDcap1.25TiB plus whole-user guards; Anta/data3cap16TiB and96four-hourjobs; latest3verifiedAMDcheckpoints retained. No rawsimulation data onPerseus.

Future most useful products will be separation/orbital_motion/horizon/waveform figures and central-orbit/context/fixed-grid-density movies under Anta/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/analysis/update_*/. These are planned paths, not existing Session009 science products. README and manifest map exact inputs, status, stop commands, verified data, reports and provenance.
