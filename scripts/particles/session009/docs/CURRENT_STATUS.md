# Session009 — scheduled check and saved-startup continuation

Checked **October 08, 2026 09:29 AM PDT**. The production run is **not running yet**. The full five-million-particle reference run completed two evolution cycles at **t=0.0125M_ref**, then its inherited checking script failed on a Session007 profile filename. That configuration failure was reviewed and repaired. Anta2397 passed the actual saved-data validation, and AMD455389 reverified the69.3GB checkpoint, restored comparison data,48GPU memory record and storage. Remaining gate455390 is queued for resources and inspector455391 waits for it. The controller keeps production time0 until all startup gates are accepted.

## Live jobs and verified checkpoint

Registered queue at the metadata snapshot:

~~~
455391|jn9_20261008_resume_inspect0|PENDING|(Dependency)
455390|jn9_20261008_resume_gate|PENDING|(Resources)
~~~

Checkpoint: /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/runs/gate_reference/rst/s9_orbit_gw.00001.rst; 69,315,343,035bytes, 35,007byte header, cycle2, t0.0125, five million particles, zero removals. SHA256 899c920a37f6bcd1a20ebe081256ce0ac36baf5e6bbb87d4e445348c32bf67b7. Native header/layout and hashes passed; the actual uninterrupted-versus-restart evolution comparison remains a required unfinished gate.

## Reviewed repair and durable continuation

Full startup454732 passed48-rank communication and initialized5384blocks. Simulation exit0 and exact particle conservation; the later validator requested gi_profile_M076_two_clump_s7.txt instead of the canonical Session009 filename. Original scripts, inputs, controls, logs, reports and accounting are preserved with hashes on AMD and Anta in history/validation_failure_454732_20261008/, with small metadata preservation on Perseus. Sealed reference data and checksum-verified Anta copies are retained.

Operations repairs ef5363c6 and **a60f312e7a5145325f543bc48b108ca8f5f47cb6** were pushed and the remote branch verified. Allocated Perseus11356/11357 each passed19 targeted tests plus syntax/input-contract checks. Anta2397 checked all initial particles, local constraints, complete physical mesh and finite saved fields, returning receipts and only291,767,869bytes of reference comparison data. AMD455389 passed three acceptance regression tests and rehashed the restored files and checkpoint. The remaining gate skips duplicate reference evolution, runs the split/restart and clean-stop/output checks once, and repeats the actual48-rank communication probe. Success allows the standard durable inspector to start the bounded t12 continuation; a failure stops for review.

## Initial evidence and scientific limits

- Exact counts3,000,000/1,000,000/1,000,000; unique immutable tags0..4,999,999; finite positive weights; centers, widths, thermal spread, boost signs and positive orbitalJz passed.
- Sampled rest-mass sum approximately1.04465, distinct from source/model parameters summing to1. Approved local speed0.133215 remains approximate companion support. K_ij approximately0 leaves the local momentum constraint unsolved.
- Rounded total covariant momentum(0,-3.7e-7,0), about8.6e-6 of the summed clump momentum magnitudes. Sampling residual retained without recoil or new symmetrization.
- Actual mesh5384blocks/176,422,912 interior cells; current minimum dx1/64, maximum8, available finest collapse spacing1/256. Physical levels0..11, root logical level3. Domain±1024. All radial floors pass, including dx≤.25 throughout r≤46 with extractionR40.
- All48GPU histories present; peak70.3403%, below85%. This establishes startup memory, not the later collapse peak.
- The t0.0125 Session007 comparison matches left/right/common-central regions, masks and volume normalization. Global volumes differ and its ratio is suppressed. No constraint improvement or convergence claim from changed empty volume; raw con_M is already squared.
- Startup plots and short central/context/density movies at Anta analysis/update_final_2394/. Two tracker samples cover only0..0.00625 with cached centers and unavailable derivatives. Separation plot/CSV inspected. These do not establish circularity, a revolution, collapse, merger or ringdown. The final directory name describes the closed failed attempt, not completed production; its active pointer/terminal marker are preserved in history.

## Resources and approved limits

AMD actual **1.987222222 raw node-hours**, including preparation/failures/inspectors; reserved maximum **50.487222222**. Anta actual **1.267222222 node/GPU-hours**,7/96 registered jobs. No supplied monetary tariff; no dollar cost invented. No reliable production-rate estimate from this short startup.

Fresh allocated AMD sample: whole user **0.822TiB**, Session009 **0.127TiB**, epoch1791476619.7253911. Last allocated Anta archive sample203,205,316,608bytes (~.185TiB), plus subsequent small metadata; /data3 free **26.80TiB**.

Unchanged R40/R46, local boost0.133215,5Mparticles, RK4/CFL.4, Löhner.2/tracker_floorfalse, alpha<.05 removal/AH removalOFF. Complex raw rPsi4 ell2..8 every.025; particles.25, planes.1, full3D/checkpoints10 plus clean stops. Hard t400 or10,000 AMD raw node-hours,90 finite12-hour segments with40minutes finalization, original deadline2026-11-19T22:01:43.732344Z. AMD session1.25TiB; whole-user warn1.5/stop1.7/projected1.9TiB and256GiB reserve. Anta16TiB,96 four-hour allocated jobs and original deadline unchanged. Latest three verified AMD checkpoints retained after checksum-verified archival. No numerical retries.

Earlier scientific stopping still requires a strictly accepted common horizon enclosing both objects, usable outgoing ringdown in the supported band with gaps checked, and at least100 saved M_ref after its observed waveform peak. Merger/ringdown not guaranteed. M_ref=1 is the inherited reference unit.

## Durable monitors and scoped controls

Scheduled prompt reached this same thread; completion receipt:evidence/scheduled_check_20261008T162940Z.json. The native existing-thread two-hour wake remains installed. The five-minute Anta archival trigger was restored after preserving terminal failure records and generated reports; heartbeat is active. Continuation and allocated archive/analysis use bounded jobs, duplicate protection and sticky user stop. Monitoring cannot guarantee queue access or successful evolution. Initial/t12/every50M/final products run as verified data arrive.

~~~sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py status'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py cancel'
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/code/scripts/particles/session009/wake_check.py remove
~~~

Never cancel unrelated ST jobs. No Session009 raw particles,3Dfields or checkpoints on Perseus; retained source/inputs/scripts/docs, small QA/acceptance receipts and startup review figure. Next: inspect455390/455391 and accepted gate receipt; do not submit duplicate jobs. Compiled source6892be3e3f04ec573f91cb2034bc9d3009a3bdff, Kokkos6739bc623081648af9e752b616d9671527922cbf, input/executable hashes unchanged. Canonical input SHA256deb6f631f7b06f8df6927662e01d6a3fa88bda4b0322ade3a1a07ff10f05dffd. All44 flat runtime files frozen in evidence/frozen_config.json.

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
