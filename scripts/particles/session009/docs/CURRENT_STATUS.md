# Session009 — stopped by GPU memory guard; data archived and analyzed

Checked **October 08, 2026 07:35 PM PDT**. **The simulation is stopped and no registered AMD or Anta job is live.** The memory watchdog requested a clean stop at **October 08, 2026 04:01:27 PM PDT** after a GPU reached **85.075%**, crossing the approved85% threshold. Production455477 exited0:0 and wrote a verified final checkpoint at **t=9.33125M_ref**. REQUEST_STOP and RESOURCE_STOP.json remain intact; no USER_STOP is present. No automatic retry, physics change, stop override, production restart or agent was performed.

## User cancellation of periodic checks

The two-hour Session009 checks were cancelled at your request. The campaign cron entry is absent, unrelated entries were preserved, and control/WAKE_STOP records persistent wake-disable intent. No registered AMD or Anta job is running or queued. The existing GPU-memory stop remains in effect; no simulation was restarted. Cancellation receipt: evidence/two_hour_wake_cancelled.json.

## Checkpoint, controller and health

Latest verified checkpoint: /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/runs/segment_001/rst/s9_orbit_gw.00005.rst; 64,575,856,941 bytes; header52,193 bytes; cycle2256; t9.33125; 5020 blocks; 3,996,300 surviving particles. SHA2567b4a82e8ec74a027a97ab51b6b27d9a36eb0ad81f6cad97f2cfc0c82b708d77b. The native checkpoint census conserves five million initial particles:3,996,300 surviving plus1,003,700 removed. The stored three removal counters are categories, not the component ledger.

This final checkpoint was verified and hashed in the production allocation, sealed, and independently checksum-archived on Anta. Inspector455478 exited1:0 because bindings() rejected REQUEST_STOP before advancing controller time. Thus control/state.json still shows t0.03125/segments_completed0 and its old accepted startup checkpoint; those fields are stale science progress. The verified final checkpoint in control/latest_checkpoint.json is authoritative here. The binding rejection is expected stop enforcement, not evidence of changed source/configuration. No guard was bypassed and the controller was not mutated by this check.

No fatal/MPI/HIP/migration-search failure was found in the inspected final log tail; evolution exited0. Raw real/imaginary rPsi4 data are finite with318 strictly increasing timestamps through t9.3265625. Final history is finite; constraints remain limitations of the approximate initial data, with no convergence claim. Allocated Anta2423 checked all fields in the three actual latest xy/xz/yz metric planes at t9.33125: all finite. The largest sampled far-field gauge-speed estimate is1.412744, corresponding to a coordinate boundary-to-R40 estimate696.52M; this plane estimate is not a full3D causal/convergence proof.

## Scientific findings from allocated analysis2408

- Strict accepted individual horizon rows: left423, right463; every accepted surface has a unique matching summary, with zero unmatched/ambiguous joins. Both have accepted rows by t8.4859375. First simultaneous paired accepted-center separation is at t8.6453125; gaps remain visible.
- At t9.3296875, accepted horizon masses are approximately0.074953 and0.074854, with inherited coordinate-spin estimates chi0.002973 and0.004864. These are horizon diagnostics in M_ref units, distinct from each0.12 source parameter and sampled rest mass; no precision spin/convergence claim.
- Observed phase after both individual horizons have accepted measurements:0.0042207 revolutions, about1.52degrees. This small arc cannot establish a nearly circular orbit, sustained inspiral, secular shrinkage or merger. There is no accepted common horizon and no captured merger/ringdown waveform. Strain integration is deferred for insufficient causal/time coverage; raw complex multipoles remain primary.
- Full surviving-particle centroid separation runs from5.999221 to6.028254 (approximately+0.484%) through t9.33125. Simultaneous accepted AH-center separation runs from6.060942 to6.067694 through t9.3296875, with range/mean0.142%. These are different coordinate center definitions; removal changes surviving-particle centroids. No arbitrary rescaling, alignment, interpolation over one-sided AH acceptance or circularity claim.
- Exact removal ledger: envelope1,999 removed/2,998,001 alive; left501,109 removed/498,891 alive; right500,592 removed/499,408 alive. All available full snapshots match component initial counts, and the1,003,700-event ledger matches the final checkpoint. Approved alpha<0.05 removal remains ON and AH removal OFF. Removed-matter angular momentum is kept separate from horizon spin.

## Archival, plots, movies and review

All six startup runs and segment_001 are checksum-verified on Anta. Its stopped-run analysis directory is **/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/analysis/update_final_2408**. The word final denotes this resource-stop closeout, not a completed production orbit/merger/ringdown calculation. It contains separation/orbit/constraint/removal/horizon/waveform plots and central/context/fixed-grid-density movies. No t12 or50M milestone was reached.

The separation and horizon plots plus representative final central/context/density movie frames were visually inspected; plot values were checked against the saved summaries. Review copies on Perseus: figures/stopped_review_2408/. Full data, reduced ledgers and movie frame cache stay on Anta; no large raw particle/volume/checkpoint data were pulled to Perseus.

The generated report initially divided production allocation by stale controller t0.03125 and returned a false cost estimate. The original reports/resources are preserved at Anta analysis/stopped_review_20261009/original_reports/. Allocated2423 corrected coverage to t0.03125..9.33125:67.063333 production raw node-hours over9.3M, historical mean7.211111 node-hours/M. This average includes inexpensive early evolution and is not a forecast for collapse or future orbits. The remaining-to-t400 forecast is withdrawn while the memory stop needs review. The empty metric-health receipt was corrected by presenting the actual run/bin/ planes to frozen health9.py through symlinks. The44frozen AMD runtime files and compiled source/binaries were unchanged.

## Resources, monitoring and limits

AMD cumulative **72.697778 raw node-hours**, including preparation, unsuccessful allocations, production and inspectors; no active allocation. Anta cumulative **3.851944 node/GPU-hours**,11/96 registered jobs, all completed0:0, including the prior stopped-run review2423. No monetary tariff is available.

Last allocated AMD storage sample (epoch1791500414.5438387, before completed archival cleanup): Session009 **0.217TiB**, whole user **0.912TiB**, warningfalse. This is not a fresh post-cleanup du measurement. Anta last allocated archive measurement is **0.644TiB**, below16TiB; filesystem free metadata is26.31TiB. Raw science/checkpoints are retained under Anta runs/. Latest three verified AMD checkpoints remain subject to checksum-verified retention rules.

The Perseus two-hour checks were cancelled at the user’s request. The campaign cron entry was removed and control/WAKE_STOP records persistent wake-disable intent; unrelated cron entries were preserved. Receipt: evidence/two_hour_wake_cancelled.json. Anta completed terminal archival/analysis and removed its five-minute trigger as designed; its old heartbeat is no longer a live production heartbeat. No automatic continuation is enabled while REQUEST_STOP remains. A future resume requires review of GPU memory feasibility and a concrete approved operational configuration. Before any such resume, repair the inspector stop-finalization ordering and durable metric-path/cost reporting, then freeze and test that operations revision. No new physics or numerical survey is proposed by this check.

Unchanged R40/R46, local boost0.133215, five million initial particles, central ceiling1/256, domain±1024, root256^3/dx8,32^3 cells/block, physical levels0..11; Löhner0.2/tracker_floor=false; RK4/CFL0.4 and inherited protections. M_ref=1 is the inherited reference unit, not a measured horizon/ADM mass. Sampled rest-mass sum≈1.04465 differs from source/model parameters summing to1. The opposite local Lorentz boosts provide deliberately approximate companion support, not exact GR circular data; K_ij≈0 leaves the local momentum constraint unsolved. Raw complex rPsi4 ell2..8 every0.025M, full particles0.25M, planes0.1M,3D/checkpoints10M plus clean stop. Hardt400 or10,000rawAMDnode-hours;90finite12hsegments with40min finalization; original deadline2026-11-19T22:01:43.732344Z. AMD1.25TiBsession/whole-user warn1.5,stop1.7,projected1.9TiB+256GiBreserve. Anta16TiB/96×4h/original deadline. Earlier scientific termination requires accepted enclosing common horizon, usable in-band ringdown and100savedM after the observed outgoing peak. None is achieved here.

~~~sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py status'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py cancel'
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/code/scripts/particles/session009/wake_check.py remove
~~~

Do not clear stop flags, launch a duplicate/retry or cancel unrelated ST jobs. Source6892be3e3f04ec573f91cb2034bc9d3009a3bdff; Kokkos6739bc623081648af9e752b616d9671527922cbf; frozen operationsa60f312e7a5145325f543bc48b108ca8f5f47cb6; input SHA256deb6f631f7b06f8df6927662e01d6a3fa88bda4b0322ade3a1a07ff10f05dffd. Standalone reviewed analysis scripts are outside the frozen flat runtime directory. Receipt:evidence/scheduled_check_20261009T020312Z.json.

## Reviewed repair and durable continuation

Full startup454732 passed48-rank communication and initialized5384blocks. Simulation exit0 and exact particle conservation; the later validator requested gi_profile_M076_two_clump_s7.txt instead of the canonical Session009 filename. Original scripts, inputs, controls, logs, reports and accounting are preserved with hashes on AMD and Anta in history/validation_failure_454732_20261008/, with small metadata preservation on Perseus. Sealed reference data and checksum-verified Anta copies are retained.

Operations repairs ef5363c6 and **a60f312e7a5145325f543bc48b108ca8f5f47cb6** were pushed and the remote branch verified. Allocated Perseus11356/11357 each passed19 targeted tests plus syntax/input-contract checks. Anta2397 checked all initial particles, local constraints, complete physical mesh and finite saved fields, returning receipts and only291,767,869bytes of reference comparison data. AMD455389 passed three acceptance regression tests and rehashed the restored files and checkpoint. Gate455390 completed0:0 in1066seconds, passing split/restart, clean-stop, all required outputs and48GPU memory checks without repeating reference evolution. Inspector455391 completed0:0 in126seconds and automatically submitted production455477 plus inspector455478. Production continued the accepted checkpoint; no duplicate launch or new submission by this check. Numerical failures still stop for review.

## Preserved earlier dated checks

## Regular two-hour check — October 07, 2026 03:02 PM PDT

The scheduled message reached this same thread and the check completed. Registered AMD startup454732 remains PENDING(Resources); inspector454733 is PENDING(Dependency). The actual scheduler currently estimates **October 08 at 11:44 AM PDT** for startup; this can change and is not a guarantee. No new failure, stop flag, duplicate launch or configuration change occurred. Frozen bindings pass.

Physical time remains0 and no verified binary checkpoint, initialization ledger or evolved state exists. Only historical linear-wave preparation runs are present; their t80 output is not binary evolution. Numerical health, actual full-particle GPU memory and current AMD storage remain unmeasured until startup is allocated.

AMD cumulative usage0.965rawnodeh and reserved maximum49.465 are unchanged. Anta cumulative0.141944444node/GPUh,5/96 completed jobs; no active archive/analysis job and no new science milestone. Anta/data3free26.98TiB. Both cron entries are installed; the AMD archive heartbeat was49.9seconds old at inspection. Reports, plots and movies await sealed binary science output. Original R40/R46/boost and all physical/resource/calendar caps remain intact. Receipt:evidence/scheduled_check_20261007T220232Z.json.

## Regular two-hour check — October 07, 2026 05:02 PM PDT

The scheduled prompt reached this same thread. Startup454732 remains PENDING(Resources); inspector454733 remains PENDING(Dependency). Physical time0; no verified checkpoint, binary initialization or evolved state. No new failure, stop flag, duplicate submission or configuration change. Frozen bindings pass. Actual scheduler estimate remains **October 08 at 11:44 AM PDT**, subject to change.

AMD use0.965rawnodeh; reserved maximum49.465. Anta0.141944444node/GPUh across5/96completed jobs, no active archival/analysis job or new science products. Both monitors installed and active; archive heartbeat was47.8s old at inspection. Anta/data3free26.98TiB. Current AMD storage/GPU memory and numerical health remain unmeasured until allocated startup; no old measurements are substituted. Reports, plots and movies await science data. Original R40/R46, boost0.133215, particle/model configuration, hard limits and sticky stop remain intact. No failure retry or additional allocation. Receipt:evidence/scheduled_check_20261008T000201Z.json.

## Regular two-hour check — October 07, 2026 07:01 PM PDT

Scheduled message delivered to the existing thread; check completed. Startup454732 remains PENDING(Resources), inspector454733 PENDING(Dependency). Physical time0; no verified checkpoint, initialized particles, evolved state or binary science data. No new failure or stop flag; frozen bindings pass. Scheduler estimate **October 08 at 11:44 AM PDT** remains provisional.

Usage unchanged: AMD0.965rawnodeh (maximum reserved49.465); Anta0.141944444node/GPUh,5/96completed archive jobs. Both monitors installed; archive heartbeat30.9s old, no active Anta job or new science milestone. Anta/data3free26.98TiB. Actual AMD storage/memory and numerical health await allocated startup; reports/plots/movies await data. R40/R46, boost0.133215 and all original limits/deadline/sticky stop remain intact. No duplicate job, extra allocation, agent or numerical retry. Receipt:evidence/scheduled_check_20261008T020141Z.json.

## Regular two-hour check — October 07, 2026 09:01 PM PDT

Actual scheduled message reached this existing thread and the check completed. Startup454732 remains PENDING(Resources); inspector454733 is PENDING(Dependency). Physical time0, no verified checkpoint or initialized/evolved binary state. No new failure or stop flag; frozen bindings pass. Scheduler estimate **October 08 at 11:44 AM PDT** is provisional.

AMD use0.965000000rawnodeh, reserved maximum49.465000000; Anta0.141944444node/GPUh, 5/96completed archive jobs. Both monitors installed and active; archive heartbeat29.6s old at inspection, /data3free26.98TiB. No active Anta job, new science report, plot or movie; outputs await sealed binary data. Actual AMD storage/memory and numerical health remain unmeasured until startup. Original R40/R46, boost0.133215 and all caps/deadline/sticky stop retained. No duplicate job, extra allocation, agent or numerical retry. Receipt:evidence/scheduled_check_20261008T040156Z.json.

## Regular two-hour check — October 07, 2026 11:01 PM PDT

Actual scheduled message reached this existing thread and the check completed. Startup454732 remains PENDING(Resources); inspector454733 is PENDING(Dependency). Physical time0, no verified checkpoint or initialized/evolved binary state. No new failure or stop flag; frozen bindings pass. Scheduler estimate **October 08 at 08:25 AM PDT** is provisional.

AMD use0.965000000rawnodeh, reserved maximum49.465000000; Anta0.141944444node/GPUh, 5/96completed archive jobs. Both monitors installed and active; archive heartbeat29.6s old at inspection, /data3free26.98TiB. No active Anta job, new science report, plot or movie; outputs await sealed binary data. Actual AMD storage/memory and numerical health remain unmeasured until startup. Original R40/R46, boost0.133215 and all caps/deadline/sticky stop retained. No duplicate job, extra allocation, agent or numerical retry. Receipt:evidence/scheduled_check_20261008T060115Z.json.

## Regular two-hour check — October 08, 2026 01:01 AM PDT

Actual scheduled message reached this existing thread and the check completed. Startup454732 remains PENDING(Resources); inspector454733 is PENDING(Dependency). Physical time0, no verified checkpoint or initialized/evolved binary state. No new failure or stop flag; frozen bindings pass. Scheduler estimate **October 08 at 07:42 AM PDT** is provisional.

AMD use0.965000000rawnodeh, reserved maximum49.465000000; Anta0.141944444node/GPUh, 5/96completed archive jobs. Both monitors installed and active; archive heartbeat32.3s old at inspection, /data3free26.98TiB. No active Anta job, new science report, plot or movie; outputs await sealed binary data. Actual AMD storage/memory and numerical health remain unmeasured until startup. Original R40/R46, boost0.133215 and all caps/deadline/sticky stop retained. No duplicate job, extra allocation, agent or numerical retry. Receipt:evidence/scheduled_check_20261008T080121Z.json.

## Regular two-hour check — October 08, 2026 03:01 AM PDT

Actual scheduled message reached this existing thread and the check completed. Startup454732 remains PENDING(Resources); inspector454733 is PENDING(Dependency). Physical time0, no verified checkpoint or initialized/evolved binary state. No new failure or stop flag; frozen bindings pass. Scheduler estimate **October 08 at 07:49 AM PDT** is provisional.

AMD use0.965000000rawnodeh, reserved maximum49.465000000; Anta0.141944444node/GPUh, 5/96completed archive jobs. Both monitors installed and active; archive heartbeat27.1s old at inspection, /data3free26.98TiB. No active Anta job, new science report, plot or movie; outputs await sealed binary data. Actual AMD storage/memory and numerical health remain unmeasured until startup. Original R40/R46, boost0.133215 and all caps/deadline/sticky stop retained. No duplicate job, extra allocation, agent or numerical retry. Receipt:evidence/scheduled_check_20261008T100141Z.json.

## Regular two-hour check — October 08, 2026 05:01 AM PDT

Actual scheduled message reached this existing thread and the check completed. Startup454732 remains PENDING(Resources); inspector454733 is PENDING(Dependency). Physical time0, no verified checkpoint or initialized/evolved binary state. No new failure or stop flag; frozen bindings pass. Scheduler estimate **October 08 at 07:49 AM PDT** is provisional.

AMD use0.965000000rawnodeh, reserved maximum49.465000000; Anta0.141944444node/GPUh, 5/96completed archive jobs. Both monitors installed and active; archive heartbeat33.2s old at inspection, /data3free26.98TiB. No active Anta job, new science report, plot or movie; outputs await sealed binary data. Actual AMD storage/memory and numerical health remain unmeasured until startup. Original R40/R46, boost0.133215 and all caps/deadline/sticky stop retained. No duplicate job, extra allocation, agent or numerical retry. Receipt:evidence/scheduled_check_20261008T120135Z.json.

## Regular two-hour check — October 08, 2026 07:02 AM PDT

Actual scheduled message reached this existing thread and the check completed. Startup454732 remains PENDING(Resources); inspector454733 is PENDING(Dependency). Physical time0, no verified checkpoint or initialized/evolved binary state. No new failure or stop flag; frozen bindings pass. Scheduler estimate **October 08 at 07:49 AM PDT** is provisional.

AMD use0.965000000rawnodeh, reserved maximum49.465000000; Anta0.141944444node/GPUh, 5/96completed archive jobs. Both monitors installed and active; archive heartbeat34.9s old at inspection, /data3free26.98TiB. No active Anta job, new science report, plot or movie; outputs await sealed binary data. Actual AMD storage/memory and numerical health remain unmeasured until startup. Original R40/R46, boost0.133215 and all caps/deadline/sticky stop retained. No duplicate job, extra allocation, agent or numerical retry. Receipt:evidence/scheduled_check_20261008T140205Z.json.


## Regular two-hour check — October 08, 2026 11:05 AM PDT

Production455477 RUNNING on the approved12nodes/48GPUs; inspector455478 waits. Saved diagnostics and complex rawR40 multipoles reach t2.15625; latest verified checkpoint t0.03125. All startup/restart/output gates passed, including real35kBheader restart and explicitly flagged tracker reacquisition gaps. No new failure/stop flag; frozen source/input/executable/script bindings pass. All5Mparticles still present in live conservation logs,48GPU peak70.98% below85%, finite increasing waveform timestamps and finite recent history. No accepted horizons yet; no established orbit or merger/ringdown result.

AMD12.165556 rawnodeh, maximum reserved150.075556; Anta2.023889node/GPUh,8/96jobs with2400 actively archiving sealed startup output. AMDsession0.350TiB, wholeuser1.045TiB; /data3free26.66TiB. Both monitors installed, fresh archive heartbeat. Initial startup plot/movies retained; t12/every50M analysis awaits sealed coverage. No new job, duplicate, numerical retry, physics change or agent. Original R40/R46/boost/limits/deadline/sticky stop unchanged. Receipt:evidence/scheduled_check_20261008T180553Z.json.

## Regular two-hour check — October 08, 2026 01:01 PM PDT

The scheduled message reached the existing thread. Production455477 is RUNNING on12nodes/48GPUs; inspector455478 waits. Raw R40 multipoles reach t6.90938; latest verified checkpoint remains t0.03125. No new failure/stop flag; frozen source/input/executable/script bindings pass. All5Mparticles remain; production peak memory75.92% below85%. Waveform timestamps and recent history are finite. Constraint diagnostics are rising during contraction and need continued review. No accepted horizons; failed searches are preserved and excluded from measurements. Both tracker locations now use dx1/256. No post-BH orbit/merger/ringdown finding yet.

AMD36.312222rawnodeh, maximum reserved150.075556; Anta3.203333node/GPUh,9/96jobs allcompleted. All sealed startup runs are checksum-verified on Anta; no active Anta job or new t12/50M products. AMD session0.208TiB, wholeuser0.903TiB; /data3free26.45TiB. Both monitors remain installed, heartbeat84.0s old. No new submission, duplicate, numerical retry, physics change or agent. Original R40/R46, boost0.133215, caps/deadline/sticky stop remain intact. Receipt:evidence/scheduled_check_20261008T200739Z.json.

## Regular two-hour check — October 08, 2026 03:00 PM PDT

Existing-thread scheduled delivery confirmed. AMD455477 RUNNING on12nodes/48GPUs;455478 PENDING(Dependency). Raw waveform time8.35156, latest verified checkpoint0.03125. Live count4,381,564, removal rule unchanged. GPU peak81.05% below85%; finite waveform/history, no evolution failure found in inspected tail. Individual finite horizon candidates fail strict geometry/persistence/publication; zero accepted individual/common rows. No accepted BH masses/spins or post-BH orbit/merger/ringdown finding. Frozen bindings pass, no stop flags.

AMD60.192222rawnodeh (reserved maximum150.075556); Anta3.203333node/GPUh,9/96completedjobs, none active. AMD session0.214TiB/wholeuser0.909TiB; Anta/data3free26.45TiB. All startup archives verified; production unsealed, no new t12/50M plots. Both monitors installed; heartbeat45.0s old. No new submission, duplicate, agent, numerical retry or physics change. Original R40/R46, boost0.133215 and all caps/deadline/sticky stop retained. Receipt:evidence/scheduled_check_20261008T220335Z.json.

## Regular two-hour check — October 08, 2026 05:00 PM PDT

Existing-thread delivery confirmed. No live AMD/Anta jobs. GPU watchdog crossed85.075% and cleanly stopped455477 at verified t9.33125;0:0 evolution, complete64,575,856,941-byte checkpoint independently archived on Anta. REQUEST_STOP/RESOURCE_STOP retained. Inspector455478 rejected that stop; controller t0.03125 is stale. No evolution retry, stop override or physics change.

Anta2408 archived segment and generated stopped-run figures/movies/reports;423/463accepted individual horizons, no common horizon, only1.52degrees of post-formation phase. Exact component removal ledger conserves5M initial particles. New bounded Anta2423 completed0:0: actual three metric planes finite; stale-time cost estimate corrected with originals preserved and future forecast withdrawn. Representative final figures/frames inspected. AMD72.697778rawnodeh; Anta3.851944node/GPUh,11/96completedjobs. Last allocatedAMDsession0.217TiB/wholeuser0.912TiB; allocatedAntaarchive0.644TiB. Two-hourwake remains; Anta terminal trigger stopped as designed. Original R40/R46/boost/caps/deadline/stop intent remain intact. Memory feasibility needs review before any resume. Receipt:evidence/scheduled_check_20261009T000844Z.json.

## Regular two-hour check — October 08, 2026 07:01 PM PDT

Scheduled delivery to the existing thread confirmed. No live registered AMD or Anta jobs. Verified checkpoint remains t9.33125, cycle2256,3,996,300 surviving particles; retained AMD file size and Anta SHA256 receipt agree. GPU memory REQUEST_STOP/RESOURCE_STOP remains set; no user-stop or new failure. All44frozen scripts, source HEAD and canonical input match configuration; no large executable/checkpoint hashes were repeated on login nodes. Final conservation log, completed allocated three-plane health review and generated plots/movies remain retained. No new science data or claim; both individual accepted horizons exist, but no common horizon/merger/ringdown and only1.52degrees of measured post-formation motion.

AMD usage unchanged at72.697778rawnodeh; Anta3.851944node/GPUh across11/96completedjobs. Last allocatedAMDstorage sample remains historical pre-cleanup; last measured Anta archive0.644TiB, filesystem free26.30TiB. Two-hour wake installed; Anta terminal archive trigger has stopped as designed. No submission, rerender, numerical retry, physics change, stop override or agent. Original R40/R46/boost/caps/deadline preserved. Memory feasibility/operations review is required before a future resume. Receipt:evidence/scheduled_check_20261009T020312Z.json.

## Session009 two-hour checks cancelled — October 08, 2026 07:35 PM PDT

The two-hour Session009 checks were cancelled at your request. The campaign cron entry is absent, unrelated entries were preserved, and control/WAKE_STOP records persistent wake-disable intent. No registered AMD or Anta job is running or queued. The existing GPU-memory stop remains in effect; no simulation was restarted. Cancellation receipt: evidence/two_hour_wake_cancelled.json.

The verified final checkpoint remains t=9.33125M_ref, with 3,996,300 surviving particles from five million initially simulated. Production stopped cleanly because GPU memory reached 85.075%, exceeding the approved 85% guard; this was not an out-of-memory crash or a scientific endpoint. Both individual horizons are accepted by t8.48594; only about 1.52 degrees of subsequent motion were recorded, with no accepted common horizon, merger or ringdown. Verified raw data, checkpoints, figures and movies remain on Anta. AMD cumulative usage is 72.697778 raw node-hours; Anta is 3.851944 allocated node/GPU-hours. Physics, source and stop flags are unchanged. No new jobs, rendering, agents or raw-data transfers were performed.
