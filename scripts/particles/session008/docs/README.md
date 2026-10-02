# Session 008 companion-supported AMD assessment

**Status:** implementation in progress; AMD build447620 completed; GPU validation and evolution not yet submitted. No scientific outcome is claimed.

- Canonical input: `inputs/gi_cluster_s8.athinput`; approved decisions: `APPROVED_PLAN.md`; request record: `PROMPT.md`.
- Isolated source: `code/`, branch `project/GI-in-cluster`; compiledsource0c0a5a9b includes repair263dcf21; Kokkos6739bc62.
- AMD root: `/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/`.
- Anta root: `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/`.
- Reproducible scripts: `code/scripts/particles/session008/`, copied to session `scripts/` and remote sites. Source bundles/provenance/tests in `evidence/`.
- Science: per-segment `runs/`; verifiedAMDcheckpoint pointer `control/latest_checkpoint.json`, latestthreeonly. Antaverifiedscience `runs/*/ARCHIVE_VERIFIED.json`; analyses in `analysis/assessment_JOBID/`.

## Status and stop commands

```sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py status'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/scripts/workflow.py cancel'
```

`stop` requests a clean checkpoint with sticky intent; `cancel` immediately cancels only recorded campaign jobs and may leave the latest checkpoint older than the final evolved time. Never use a user-wide cancellation command.

## Bounds and continuation

Targett12, approvedtotal48rawAMDnodehours. Normalchainmaximum44.5raw: build0.5, exactgate6, fourinspectors2, threeevolutionjobs36. Noautomaticretry or t50extension. Each4hour evolutionallocation reserves40minutes forfinish/checkpoint, withstop polling16cycles. Checksbind source/input/executable/scripts; persistentstate/lock plus uniquejobnames protect submission/recovery. Only successfulgates, verifiedprogress/checkpoint andremainingcaps release eachsegment.

Anta performs checksum-verifiedpulls because AMD-to-Anta SSH is blocked while Anta-to-AMD works. A finitefour-job Anta chain provides atmost24hours ofpolling/analysis allocations, twoCPUcores perjob, noGPU request. Production requires a fresh archiveheartbeat andhalts safely if it is absent/stale; a longqueue can exhaust this archivewindow without completingproduction. Then inspectrecords andrequest a new bounded recovery decision. No agent or interactive shell is needed fornormalcontinuation orsafehalt.

Heavyreductions/rendering runthroughSlurm. Latestthreecheckpointdeletion is explicitlyapproved withoutbackup. CompletedAMDsciencebinaries may be deleted onlyafterAntachecksumverification; manifests/logs/inputs/rawwaveforms/failureevidence remain. AMDcampaign1.25TiB/wholeuser1.9TiB thresholds; Antastage1TiB/reserve512GiB.

## Scientific interpretation

Fresh5Mparticles, localboost0.133215, inheritedapproximateKij0/unsolvedlocalmomentumconstraints, strictmeasurement-onlyAHs, lapse0.05removal. M_ref1 is inheritedsourceunit, not measuredADM/rest/horizonmass. The ~227unitinitial coordinateperiod makes t12 anearlyassessment. r40–70 cannot yet contain centralcollapse radiation atthisendpoint; rootdx2 doesnot justifyhighfrequencymergerwaveclaims. Numericalacceptance andorbitaloutcome areseparatequestions.
