# Session009 fresh production

Authoritative Perseus record: /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005.
AMD: /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005.
Anta archive/analysis: /data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005.

Approved plan in notes/APPROVED_PLAN.md; canonical input inputs/gi_cluster_s9.athinput. Isolated source code/ branchproject/GI-in-cluster, versioned workflow code/scripts/particles/session009/. Build/gates precede conditional production. No Session008 checkpoint is used.

## Campaign-scoped operations

```sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py status'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py cancel'
```

Stop is sticky and requests a clean checkpointed stop. Cancel is emergency campaign-only cancellation; no guaranteed new checkpoint. Never cancel a generic `s9_prod`: that job name belongs to unrelated ST migration. Status contains exact registered Jeans009 jobIDs. Scheduler: `squeue -u jiaxiwu -o "%i %j %T %R"`; inspect only the registered Jeans009 job IDs (current prefix jn9_20261007_known12_).

On Anta, `evidence/active_archive_jobs.json` records jobs; `evidence/metadata_trigger.log` logs the finite cron. `scripts/archive_trigger.py remove` removes only the JEANS9 metadata entry. Raw science, verified checkpoints and manifests live in runs/. Milestone plots/movies/reports under analysis/update_*; analysis/latest.json identifies most recent products. Human/technical top reports are updated after verified data become available. Source/config/evidence are preserved on Perseus; no large raw data there.

Hard t400/10000rawAMDnodeh/45days; single-r40waveform and approximate initialdata limitations remain. Allheavywork usesSlurm. Failure haltsforreview, noautomaticretries. Anta≤96four-hourjobs/16TiB; AMD1.25TiBsession andwholeuserguards; oldcheckpoints archivedandverified before removal. No runtime depends on the interactive agent.

## October 6 reset and two-hour checks

Approved extraction is now R40, with dx<=0.25 through R46 (5,384 initial blocks), the same five million particles and local boost0.133215. The failed R50 startup and its reports remain under history/r50_before_reset_20261006/. Original deadline, node-hour cap and archive job ledger remain cumulative. The AMD binary/source revision is reused unchanged.

The corrected input explicitly supplies output7/last_time. A one-node actual-executable input preflight precedes the full12-node gate. Production continues from its accepted t0 history only after mesh, particles, allGPUmemory, restart and required outputs pass.

Five-minute Anta archival and allocated50M report/plot/movie updates are re-enabled for this attempt. The separate Perseus two-hour wake checks registered AMD jobs and wakes this same Codex thread when idle, avoiding overlapping agent turns. It is bounded by the existing45-day deadline.

```sh
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/code/scripts/particles/session009/wake_check.py probe
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/code/scripts/particles/session009/wake_check.py remove
```

These commands inspect/remove the two-hour wake only. The AMD `workflow.py stop` command above requests the sticky scientific stop. Evidence/two_hour_check_latest.json and two_hour_checks.jsonl record checks; actual wake delivery is recorded separately from schedule installation.

For current jobs, measured usage and tested-versus-pending automation, read evidence/CURRENT_STATUS.md.

## October7 scheduled check and reviewed recovery

Startup454538 failed before initialization on an MPI backend mismatch at k003-009. Failure evidence remains in history/mpi_failure_454538_20261007/. Startup and every production segment now request the exact twelve-node group whose prior48-rank test passed. AMD454731 input checks passed; full startup454732 is pending resources and inspector454733 waits for it. Operationsff4b5cb4 was pushed. Physical time remains0, with no checkpoint or binary science results.

The two-hour nativequeue wake delivered this check to the existing thread. Anta's five-minute archive trigger was reactivated after preserving the failed-startup closeout from2365. Original cumulative caps/deadline and sticky stop remain unchanged. Authoritative evidence and next action: evidence/CURRENT_STATUS.md.
