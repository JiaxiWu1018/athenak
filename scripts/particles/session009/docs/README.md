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

Stop is sticky and requests a clean checkpointed stop. Cancel is emergency campaign-only cancellation; no guaranteed new checkpoint. Never cancel a generic `s9_prod`: that job name belongs to unrelated ST migration. Status contains exact registered Jeans009 jobIDs. Scheduler: `squeue -u jiaxiwu -o "%i %j %T %R"`; inspect only jn9_20261005*.

On Anta, `evidence/active_archive_jobs.json` records jobs; `evidence/metadata_trigger.log` logs the finite cron. `scripts/archive_trigger.py remove` removes only the JEANS9 metadata entry. Raw science, verified checkpoints and manifests live in runs/. Milestone plots/movies/reports under analysis/update_*; analysis/latest.json identifies most recent products. Human/technical top reports are updated after verified data become available. Source/config/evidence are preserved on Perseus; no large raw data there.

Hard t400/10000rawAMDnodeh/45days; single-r50waveform and approximate initialdata limitations remain. Allheavywork usesSlurm. Failure haltsforreview, noautomaticretries. Anta≤96four-hourjobs/16TiB; AMD1.25TiBsession andwholeuserguards; oldcheckpoints archivedandverified before removal. No runtime depends on the interactive agent.
