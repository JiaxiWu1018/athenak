# Session 008 durable handoff

## Current state

Prepared and submitted; full GPU validation is queued. No evolved science data exist yet. AMD build447620 completed; gate447655 precedes inspectors447656/447658/447660/447662 and production447657/447659/447661. All continuation is finite and gated, with maximum44.5 raw AMD node-hours against the approved48. No numerical retries or t50 extension.

Perseus Slurm10327 passed fifteen targeted checks (twelve workflow/checkpoint and three analysis), plus syntax checks. The final AMD gate repeats twelve before actual numerical validation. Compiled source is 0c0a5a9b; exact operations revision and verified push are in evidence/OPERATIONS_REVISION.txt. The actual GPU memory/initialization/restart outcome remains pending.

## Commands from Perseus

```sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py status'
ssh hpcfund.amd.com 'squeue -j 447655,447656,447657,447658,447659,447660,447661,447662'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py cancel'
```

Graceful stop is sticky and requests a checkpoint. Emergency cancel targets only recorded AMD campaign jobs; it can leave the latest verified checkpoint older than the last evolved state. Anta archival continues to preserve completed data. Its own jobs are recorded separately in `evidence/active_archive_jobs.json` on Anta.

## Independent archival trigger

Anta has a supported, active user cron service. One campaign-tagged entry runs lightweight metadata checks every five minutes; it expires 48 hours after installation or removes itself after final analysis. Maximum four active archival/analysis Slurm jobs, six hours each, one required GPU/two CPUs/16 GiB per job. It never allocates GPUs simply to wait. No archival job has been submitted yet because no completed science exists. A stale heartbeat stops production safely.

```sh
ssh jiaxiwu@anta.caltech.edu 'cat /data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/evidence/trigger_config.json; crontab -l'
ssh jiaxiwu@anta.caltech.edu 'python3 /data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/scripts/archive_trigger.py remove'
```

Removing the trigger alone does not request a clean stop; use the AMD graceful-stop command first if stopping the workflow. Do not recreate or extend the trigger after its cap/deadline without a new bounded decision.

## Evidence and review

AMD: `/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/`; state/config/checkpoint pointers under control/, exact logs/evidence under logs/ and evidence/, data under runs/. At most three verified checkpoints total across validation and production remain on AMD. Failed/in-progress writes cannot displace verified checkpoints.

Anta: `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/`; per-run checksums under runs/*/ARCHIVE_VERIFIED.json, final plots/movies under analysis/assessment_JOBID/, final reports at root. Source/checkpoint ABI, input/executable/script hashes, GPU memory, initialization and restart results are captured by the gates.

Anta cannot SSH back to Perseus with its current key, so final reports are generated on Anta and can be pulled from Perseus with `scripts/collect_review.sh`. No production continuation depends on that pull. Representative plot/frame visual QA remains pending. Any numerical, scheduler, archive, resource, or configuration failure halts for review; inspect the recorded reason rather than automatically resubmitting.
