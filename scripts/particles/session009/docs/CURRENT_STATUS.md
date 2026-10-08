# Session009 — automatic check during production

Checked **October 08, 2026 03:00 PM PDT**. AMD production job **455477 is running** on 12 nodes / 48 GPUs; inspector **455478** waits for it. Saved raw R40 waveform data reach **t=8.35156M_ref**; tracker records reach t8.36875. The latest verified checkpoint remains **t=0.03125**, separate from live diagnostic progress. The first finite segment targets t12, with checkpoints every10M and at a clean stop. No new job submission, physics change, numerical retry, or agent was started by this check.

## Live jobs and checkpoint

~~~
455478|jn9_20261008_resume_inspect1|PENDING|(Dependency)
455477|jn9_20261008_resume_segment1|RUNNING|k002-005,k003-[003-007],k005-[002-006,009]
~~~

Latest verified checkpoint: /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/runs/gate_output/rst/s9_orbit_gw.00004.rst; 69,315,343,208 bytes; header 35,180 bytes; cycle5; t0.03125; five million particles and zero removals at that checkpoint. SHA256 7b8ac7e6875377f429426ab95756c630bd029fc9caefb63255c34ec52da78ca2. The actual large-header restart and uninterrupted-versus-restart checks passed; particle positions/momenta/weights matched exactly at t0.0125. Tracker reacquisition gaps remain flagged as unavailable diagnostics. No10M checkpoint exists yet.

## Current numerical and scientific evidence

- The latest logged live count is **4,381,564 particles** at cycle1643, down 618,436 from the initial five million. The approved alpha<0.05 removal remains enabled and AH-driven removal remains OFF. Exact component/removal conservation will be checked at the sealed segment; the old checkpoint particle count is not the current live count.
- No fatal/MPI/HIP/migration-search failure was found in the inspected log tail. The latest deposition identity residual was6.66e-16 versus3.15e-8 tolerance. Recent history and all raw complex waveform samples are finite. Constraint norms remain elevated during contraction, with the most recent saved C-norm2 decreasing from12.4871 to12.2559; this is not accuracy or convergence evidence. Full saved metric-plane assessment belongs in allocated analysis.
- All48GPU telemetry histories are present and fresh (oldest sample3.4s). Production peak memory is **81.05%**, latest busiest GPU **80.60%**, below the85% stop gate. The existing allocated watchdog continues checking memory/storage/progress.
- Each real/imaginary raw rPsi4 file has **279 samples**,78columns (time plus77 multipoles), finite values and strictly increasing timestamps, through t8.35156. No central merger/ringdown waveform is established at this early time.
- At tracker time8.36875, left/right centers are approximately(-2.90820,-0.84961,-0.00195)/(2.90430,0.84961,0.00195), minimum lapse0.1050/0.1017. Both trackers retain their component identities and report localdx1/256, logicallevel14 = physicallevel11. Transverse motion has the expected signs; sustained post-BH orbital motion and circularity remain unestablished.
- Individual horizon candidates began near t7.546875 (left) and7.453125 (right). The complete small consumer-table audit found **zero accepted/published rows** in either table; the common table remains empty. Latest individual candidates fail quality_geometry_ok, quality_persist_ok and published_this_candidate despite finite positive summary values. These surfaces do not supply accepted BH masses/spins or formation times. Failed/rejected/stale measurements are preserved and excluded from scientific plots. No merger claim.

## Setup and initial validation retained

R40 extraction and a continuous dx≤1/4 wave floor throughR46 remain unchanged. Domain[-1024,1024]^3, root256^3/dx8,32^3 interior cells per block, physical levels0..11 and central ceiling1/256. The complete initial mesh had5384blocks/176,422,912cells and minimumdx1/64; current tracker spacing1/256 is not a new complete evolved-mesh census. Löhner alpha*psi^7 threshold0.2, tracker_floor=false; RK4/CFL0.4 and inherited gauge/deposition/pusher protections remain fixed.

Initial3M/1M/1M census, immutable unique tags, positive finite weights, centers, widths, thermal sampling and boost signs passed. Local Lorentz speed0.133215 (left-y/right+y) is approved approximate companion support, not an exact GR circular-orbit construction. Positive initial orbitalJz and the small covariant linear-momentum residual were measured without recoil or new symmetry. Sampled rest-mass sum≈1.04465 differs from source/model mass parameters summing to1. K_ij≈0 leaves the local momentum constraint unsolved. M_ref=1 is the inherited reference unit. The lightweight Session007 local comparison used matched times/regions/masks/volume normalization; no global empty-volume constraint-improvement claim.

## Resources and Anta progress

AMD cumulative **60.192222 raw node-hours**, including elapsed active allocation, preparation, failures and inspectors; maximum reserved **150.075556**. Anta cumulative **3.203333 allocated node/GPU-hours**,9/96 jobs, all completed0:0; none is currently running. AMD cost keeps growing during production. No account tariff has been supplied, so monetary cost is unquoted.

Allocated AMD storage sample at epoch1791496799.5895905: session **0.214TiB**, whole user **0.909TiB**, warningfalse. Anta/data3 free **26.45TiB**. All six sealed startup runs remain checksum-verified on Anta (gate_reference, gate_mesh, gate_split, gate_restart, gate_output, gate_clean_stop). Their verified manifest sizes are in the check receipt; no current archive du measurement is inferred from those sizes. The running production segment remains unsealed. Startup reports/plots/short movies are preserved at Anta analysis/update_final_2394/; its old final name is a failed-attempt closeout, not completed science. New t12/every50M reports and plots await verified sealed production coverage. Anta analysis/latest.json has no active pointer; the old terminal pointer was preserved during reviewed recovery.

## Limits, monitoring and scoped controls

The scheduled message reached this existing thread. The two-hour native wake cron and five-minute Anta trigger are installed. Archive heartbeat was **45.0s old** at inspection. No USER_STOP, REQUEST_STOP, RESOURCE_STOP.json, ARCHIVE_ERROR or ARCHIVE_STORAGE_STOP was present. Frozen source/input/executable/runtime-script bindings passed. Continuation uses finite jobs and duplicate protection; numerical failures stop for review. Monitoring cannot guarantee queue access or successful evolution.

Raw complex rPsi4 ell2..8 every0.025M; full particles0.25M; three planes0.1M; full3D/checkpoints10M plus clean-stop checkpoints. Hard limits **t400 or10,000 AMD raw node-hours**,90 finite12-hour segments with40minutes finalization, original deadline2026-11-19T22:01:43.732344Z. AMD session1.25TiB; whole-user warn1.5/stop1.7/projected1.9TiB plus256GiB reserve. Anta16TiB,96×4h allocated jobs and original deadline unchanged. Retain latest three verified AMD checkpoints after checksum-verified archival. Earlier scientific stop still requires a strictly accepted common horizon enclosing both objects, usable outgoing in-band ringdown with gaps checked, and at least100 savedM_ref after the observed post-merger waveform peak. Merger/ringdown is not guaranteed.

~~~sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py status'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py cancel'
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/code/scripts/particles/session009/wake_check.py remove
~~~

Never cancel unrelated ST jobs. No raw Session009 particles/3Dfields/checkpoints are on Perseus; only source/input/scripts/docs, small evidence and startup review figures are retained. Next: monitor455477/455478, verified checkpoints, health and Anta archival; accepted inspection continues bounded segments automatically. No duplicate launch.

Compiled source6892be3e3f04ec573f91cb2034bc9d3009a3bdff; Kokkos6739bc623081648af9e752b616d9671527922cbf; canonical input SHA256deb6f631f7b06f8df6927662e01d6a3fa88bda4b0322ade3a1a07ff10f05dffd. Operations revisiona60f312e7a5145325f543bc48b108ca8f5f47cb6; all44runtimefiles remain frozen in evidence/frozen_config.json. Receipt:evidence/scheduled_check_20261008T220335Z.json.

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
