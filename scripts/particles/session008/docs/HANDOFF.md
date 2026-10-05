# Session 008 durable handoff

## Current state

Updated2026-10-05: full GPU gate447655 and production447657/447659 passed, reaching verifiedt9.325. Startup447661 failed before loading the checkpoint because k003-010 selected a communication method incompatible with peers. The user requested continuation after review. No automatic failure retry or t50 extension is enabled.

New chain: preflight451757; continuation451758 -> inspector451759 -> conditional continuation451760 -> inspector451761. All GPU jobs use three nodes/twelve ranks drawn only from six previously successful nodes. It retains t12 and48 raw AMD node-hours. Actual usage before continuation21.645833; additional maximum25.5; maximum total47.145833. Anta archive2329 completed and checksum-verified both production segments; approved AMD science copies were removed. See RECOVERY_PLAN_20261005.md and evidence/recovery_20261005/ for tested scripts, prior state, caps and receipts.

Perseus10327 passed15 original checks. Recovery10957 reran those and passed6 recovery checks;10958 passed the extended7 recovery checks including delayed accounting for newly submitted jobs. Syntax passed. Compiled source remains0c0a5a9b, unchanged input/executable; peak gate VRAM70.4%. Original operations remain frozen. Separate recovery hashes/configuration are in AMD control/recovery_20261005.json; actual restart validation receipt is retained.

## Commands from Perseus

```sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py status'
ssh hpcfund.amd.com 'squeue -j 451757,451758,451759,451760,451761'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py cancel'
```

Graceful stop is sticky and requests a checkpoint. Emergency cancel targets only recorded AMD campaign jobs; it can leave the latest verified checkpoint older than the last evolved state. Anta archival continues to preserve completed data. Its own jobs are recorded separately in `evidence/active_archive_jobs.json` on Anta.

## Independent archival trigger

Anta's corrected metadata trigger runs every five minutes and expires48 hours after the reviewed October5 renewal, or removes itself after final analysis. The four-job limit is cumulative:2134 and2329 completed, leaving at most two further six-hour allocations. Each uses one mandatory GPU/two CPUs/16GiB for real transfer/analysis work. A stale heartbeat stops production safely.

```sh
ssh jiaxiwu@anta.caltech.edu 'cat /data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/evidence/trigger_config.json; crontab -l'
ssh jiaxiwu@anta.caltech.edu 'python3 /data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/scripts/recovery_20261005/archive_trigger.py remove'
```

Removing the trigger alone does not request a clean stop; use the AMD graceful-stop command first if stopping the workflow. Do not recreate or extend the trigger after its cap/deadline without a new bounded decision.

## Evidence and review

AMD: `/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/`; state/config/checkpoint pointers under control/, exact logs/evidence under logs/ and evidence/, data under runs/. At most three verified checkpoints total across validation and production remain on AMD. Failed/in-progress writes cannot displace verified checkpoints.

Anta: `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/`; per-run checksums under runs/*/ARCHIVE_VERIFIED.json, final plots/movies under analysis/assessment_JOBID/, final reports at root. Source/checkpoint ABI, input/executable/script hashes, GPU memory, initialization and restart results are captured by the gates.

Anta cannot SSH back to Perseus with its current key, so final reports are generated on Anta and can be pulled from Perseus with `scripts/collect_review.sh`. No production continuation depends on that pull. Representative plot/frame visual QA remains pending. Any numerical, scheduler, archive, resource, or configuration failure halts for review; inspect the recorded reason rather than automatically resubmitting.
