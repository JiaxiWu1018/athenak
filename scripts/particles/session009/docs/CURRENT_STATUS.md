# Session009 — scheduled check and reviewed startup repair

Checked **October 07, 2026 03:02 PM PDT**. The simulation is **not running**: startup **454732** is queued for resources and inspector **454733** waits for it. Physical time remains **t = 0**; no particle initialization, verified checkpoint, orbit or binary waveform exists. Live AMD control/state.json and Slurm are authoritative.

## Failure and reviewed repair

Startup454538 failed after6seconds, before geometry or particles, because k003-009 selected MPI PMLob1 while peer k002-006 selecteducx. Inspector454539 halted further jobs. Failure logs, configuration, scripts and accounting are preserved on AMD and Anta under history/mpi_failure_454538_20261007/. Earlier failures remain preserved separately. Anta2365 completed the failed-startup status archive/report; that is not completed science.

Both full startup and all future production segments now request exactly **k002-005,k003-[003-007],k005-[002-006,009]**. These twelve nodes previously passed all48 ranks of the Session009 all-to-all communication test452580. A checksum-bound receipt requires this exact node group at runtime. That historical communication success does not establish memory feasibility or successful binary evolution. Nodesk003-009 andk003-010 are excluded. No transport, physics, source or executable change was made.

Operations commit **ff4b5cb42bc67375f0b9f852de78dfe0dc073a49** was pushed and the remote branch verified. Allocated Perseus11238 passed19 targeted tests, syntax and input contract checks. AMD **454731 completed0:0 in4seconds**, passing16 checks, the original48-rank witness validation and all three actual compiled input commands. The new full-particle startup remains pending; actual48-rank communication, complete mesh, initial particle ledger, allGPUmemory, finite fields, outputs and restart checks must still pass. If this previously successful group also fails communication, preserve the failure and review bounded allocated MPI diagnostics before another full allocation.

## Configuration and limits

Approved R40 extraction and dx≤0.25 through R46, local boost0.133215, five million particles, domain[-1024,1024]³, central dx ceiling1/256, sampling, tags, RK4/CFL0.4, removal and output cadences are unchanged. Initial seeded mesh estimate5384blocks awaits the actual AMD inventory. Compiled source remains6892be3e3f04ec573f91cb2034bc9d3009a3bdff; Kokkos6739bc623081648af9e752b616d9671527922cbf. Canonical input SHA256deb6f631f7b06f8df6927662e01d6a3fa88bda4b0322ade3a1a07ff10f05dffd. All37 flat runtime files are bound in evidence/frozen_config.json.

Original hard limits remain t400 or10000 raw AMD node-hours,90 finite segments,12-hour production jobs with40minutes for finalization, and deadline2026-11-19T22:01:43.732344Z. The clock and cumulative budgets were not reset. AMD1.25TiB session cap and whole-user storage guards apply; Anta16TiB and96 four-hour archive/analysis jobs. Keep the latest three verified AMD checkpoints after checksum-verified Anta archival. Sticky user stop remains intact. Numerical failures stop for review.

An earlier scientific stop requires a strictly accepted common horizon enclosing both objects, usable outgoing ringdown within the validated frequency band with gaps checked, and at least100 saved M_ref after its observed waveform peak. No merger or ringdown is guaranteed. M_ref=1 is the inherited reference unit, not a measured horizon mass.

## Resources and archive/analysis

- AMD cumulative actual **0.965 raw node-hours**, including preparation and failed allocations. Current maximum reserved exposure **49.465 raw node-hours**. Queue wait is not charged.
- Anta **0.141944444 node/GPU-hours**,511seconds across5/96 completed jobs. The original ledger and deadline remain intact; no active Anta analysis job is needed until new sealed output exists.
- Anta/data3 has **26.98TiB free**. Last allocated archive measurement90308608bytes (~86.13MiB) was from failed-startup2365 before this small metadata repair. Current AMD storage/GPU memory has not yet been sampled by allocated startup; no old value is substituted.
- Monetary costs remain unavailable without an account tariff. There is no measured production rate or reliable updated completion estimate yet.
- No binary science plots or movies exist. Initial/t12/every50M/final science reports, plots and central/context movies are implemented and await data. The old failure-only products in Anta analysis/update_final_2365/ are retained but are not the active science pointer.

## Durable monitors and controls

This actual regular two-hour cron prompt reached the same Codex thread and this check completed. Receipt: evidence/scheduled_check_20261007T201254Z.json. The Perseus nativequeue wake and Anta five-minute archive trigger are installed. Archive tick succeeded and the AMD heartbeat is fresh. Successful inspector454733 will continue the accepted startup checkpoint to t12 and finite50M segments; continuation and allocated archival/analysis do not depend on an interactive shell. Monitoring cannot guarantee queue access or successful evolution; failures halt for review.

```sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py status'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py cancel'
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/code/scripts/particles/session009/wake_check.py remove
```

Never cancel unrelated ST jobs named s9_prod or s9_controls_build. No Session009 raw particle, volume or checkpoint data are on Perseus: only source, inputs, scripts, documentation and small retained diagnostic evidence. Next action: wait for454732, inspect the actual communication/full startup receipts; do not submit a duplicate job or automatically retry a numerical failure.

## Regular two-hour check — October 07, 2026 03:02 PM PDT

The scheduled message reached this same thread and the check completed. Registered AMD startup454732 remains PENDING(Resources); inspector454733 is PENDING(Dependency). The actual scheduler currently estimates **October 08 at 11:44 AM PDT** for startup; this can change and is not a guarantee. No new failure, stop flag, duplicate launch or configuration change occurred. Frozen bindings pass.

Physical time remains0 and no verified binary checkpoint, initialization ledger or evolved state exists. Only historical linear-wave preparation runs are present; their t80 output is not binary evolution. Numerical health, actual full-particle GPU memory and current AMD storage remain unmeasured until startup is allocated.

AMD cumulative usage0.965rawnodeh and reserved maximum49.465 are unchanged. Anta cumulative0.141944444node/GPUh,5/96 completed jobs; no active archive/analysis job and no new science milestone. Anta/data3free26.98TiB. Both cron entries are installed; the AMD archive heartbeat was49.9seconds old at inspection. Reports, plots and movies await sealed binary science output. Original R40/R46/boost and all physical/resource/calendar caps remain intact. Receipt:evidence/scheduled_check_20261007T220232Z.json.
