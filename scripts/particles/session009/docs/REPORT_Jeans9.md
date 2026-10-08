# Current update — October 08, 2026 07:02 AM PDT

Session009 remains at **t = 0**, with no checkpoint or binary science results. Startup454538 failed in MPI initialization after6seconds; no particles or geometry were initialized. Its evidence is preserved on AMD and Anta under history/mpi_failure_454538_20261007/. After review, startup and production now use the exact twelve-node group that previously passed the48-rank communication test. The simulation input, physics and binaries are unchanged. Operationsff4b5cb4 was pushed and verified.

Allocated Perseus11238 passed19 targeted checks. AMD454731 passed16 checks plus three compiled input commands in4seconds. Full startup **454732 is queued for resources**, and inspector454733 will continue production only if all mesh, particle, memory, output and restart gates pass. A repeated communication failure on this group requires further review before another full allocation.

Total use is **0.965 AMD raw node-hours** and **0.141944444 Anta node/GPU-hours** across5 archive jobs. Anta2365 archived the failed-startup closeout; it did not produce a binary waveform. The two-hour wake reached this existing thread; both it and the five-minute archive trigger are active. Science reports, plots and movies remain pending data. All original physical, storage, cost and deadline caps apply. See evidence/CURRENT_STATUS.md for evidence, exact controls and next actions.

Latest scheduled check: startup454732 remains queued for resources and inspector454733 waits for it. Time0; no checkpoint, evolved state or new science output. No new failure or stop flag. Both monitors are active, usage unchanged; no additional jobs submitted. Scheduler provisional start isOctober 08 at 07:49 AM PDT, subject to change. Receipt:evidence/scheduled_check_20261008T140205Z.json.

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
