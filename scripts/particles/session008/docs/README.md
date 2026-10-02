# Session 008 companion-supported AMD assessment

**Status, 2026-10-02 UTC:** build and script checks passed; full GPU validation is queued. Production segments are submitted behind acceptance gates. No evolved scientific results or final plots exist yet.

## Files and locations

| Item | Location |
|---|---|
| Approved scope | `APPROVED_PLAN.md` |
| Canonical input | `inputs/gi_cluster_s8.athinput` |
| Human setup/status report | `REPORT_Jeans8.md` |
| Implementation and tests | `REPORT_AGENT.md` |
| Durable continuation and review | `HANDOFF.md` |
| File mapping | `MANIFEST.tsv` |
| Independent source and scripts | `code/`, branch `project/GI-in-cluster` |
| Compiled/operations revisions | `evidence/OPERATIONS_REVISION.txt` |
| AMD root | `/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/` |
| Anta science root | `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/` |

AMD data are under `runs/`; state and checkpoint pointers are under `control/`. Anta checksums are under `runs/*/ARCHIVE_VERIFIED.json`; terminal analysis is written to `analysis/assessment_JOBID/`. Final reports are generated on Anta. `scripts/collect_review.sh` pulls a thin review copy to `review_from_anta/` on Perseus without raw data or checkpoints.

## Status and stop commands

```sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py status'
ssh hpcfund.amd.com 'squeue -j 447655,447656,447657,447658,447659,447660,447661,447662'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py cancel'
```

`stop` sets sticky intent and requests a clean checkpoint. `cancel` immediately cancels only recorded AMD campaign jobs; the latest verified checkpoint can be older than the last evolved state. Archive and collection commands are in `HANDOFF.md`.

## Bounds and continuation

The approved target is **t=12**, with **48 total raw AMD node-hours**, including preparation. The finite chain reserves at most 44.5: build 0.5, three-node gate 6, four inspectors 2, and three three-node evolution jobs 36. Each four-hour evolution job reserves forty minutes for finalization. There are no automatic retries or longer extensions.

Slurm dependencies, locked persistent state, unique job names and frozen hashes protect continuation. Production requires successful numerical gates, a verified checkpoint, remaining bounds and a fresh Anta archive heartbeat. A failure stops for review. Continuation does not depend on an agent or interactive shell.

Anta's metadata trigger checks every five minutes and expires 48 hours after installation. It submits at most four six-hour archival/analysis Slurm jobs, with the site's mandatory one-GPU request, only when work is ready. Transfers, checksums, reductions and rendering run in those jobs. A missing heartbeat stops production safely. A long queue can exhaust the archive window; extending it requires a new bounded decision.

AMD retains the **latest three verified Session 008 checkpoints total**, including gate checkpoints, without a Perseus backup. Failed or unfinished writes cannot displace them. Completed science binaries can be removed from AMD only after Anta verifies checksums. Logs, inputs, manifests, raw complex waveforms and failure evidence remain. Limits are 1.25 TiB for this AMD campaign; whole-user warning/stop/projected limits are 1.5/1.7/1.9 TiB. Anta's stage cap is 1 TiB with 512 GiB free reserve.

## Scientific interpretation

Fresh five-million-particle data use the approved local boost 0.133215 and inherited approximate initial data. `M_ref=1` is the inherited source unit, not a measured ADM, rest or horizon mass. The estimated initial coordinate period is about 227 units, so t=12 is an early assessment. It does not cover central-collapse radiation at r=40–70. The inherited propagation mesh does not establish high-frequency merger-wave accuracy. Orbital and waveform outcomes remain pending.
