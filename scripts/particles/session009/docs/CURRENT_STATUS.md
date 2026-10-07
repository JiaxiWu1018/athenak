# Session009 — reviewed MPI failure, recovery queued

Checked 2026-10-07T17:40:53.015102+00:00. **Physical time remains0: no particles initialized, checkpoint, orbit or binary waveform yet.** Live AMD control/state.json and Slurm remain authoritative.

## Failure found and action taken

Full startup453631 began October6 at21:02Pacific and failed after6seconds. k003-010 selected MPIbackendob1 while k002-006 selecteducx; MPI_Init aborted before geometry/particles. This repeats the observed Session008 node fault. Failure logs/provenance/config/scripts are preserved in AMD andAnta history/mpi_failure_453631_20261007/. No numerical evolution was retried; no physics/source/executable change.

Both full-startup and everyfutureproduction Slurm script now exclude **k003-010**, retaining12exclusiveMI210nodes/48GPUs/ranks, pinnedROCm6.4.1 and siteautomaticPML selection. Operationscommit4c348a505304878bf4a976228964f8151fa6fb07 pushed. CanonicalR40/R46input, approvedboost.133215,5Mparticles,centralceiling1/256,domain/cadences/caps allunchanged.

Perseus allocated11233 passed16targeted tests plus syntax/inputcontract. AMD newpreflight **454537 completed0:0 in4s**,13regression checks and allthreeactualcompiledinput commands passed. Full startup **454538 PENDING(Resources)**, inspector **454539 PENDING(Dependency)**. Onsuccessfulfullmesh/ledger/48GPUmemory/restart/outputchecks, inspectorcontinuesacceptedstartupcheckpointtot12then50Msegments. Fullgatesremainpending; no scientificresult or productionsegmentclaim.

## Monitors

Two-hour metadata checks actually ran, but oldcodexexecresume failed because thischat already had anactivewriter evenwhenidle. Corrected to **codexqueue --thread existingUUID --message** ontheexistingdaemon. Actualdeliveryprobe accepted queuedmessage01a1176a-f493-70b1-92e3-227042130ebe for thisthread; receipt evidence/wake_queue_delivery_probe.json and wake_20261007T173047Z.log. Acceptance is not a completedsciencecheck. Current-turn overlapguard and45-daydeadline remain; alloldfailurelogs retained. Cron still0 */2 atUTC evenhours; nextregularidlecheck uses thecorrectedcommand. No newthread/agent ordaemonrestart.

Anta2356 completed0:0 in79s andwrote the failedstartup statuscloseout, not completedscience. Oldterminalmarker/latestpointer preserved; five-minute archive/report cron reinstalledandtickpassed forrecovery. Originalledger4/96jobs anddeadline retained. Initial/t12/every50M/finalplots/reports/movies remain datadependent; representativeframe/valueQAawaitsdata.

## Accounting and limits

AMD actual **0.943611111 rawnodehours**, pendingstartup+inspector reserve48.5, maximum49.443611111. Allpriorjobs retainedinaccounting. Anta4jobs/428seconds = **0.118888889 node/GPUhours**. Monetarytariff unavailable; currentAMDstorage/48GPUmemory measurementawaits allocatedstartup. No currentstorage estimate substitutedfromoldruns.

Original hardt400 or10000rawAMDnodehours,90segments,12hjob/application11h20,deadline2026-11-19T22:01:43.732344Z unchanged. AMD1.25TiBsession pluswholeuserguards; Anta16TiB/96four-hourjobs,latest3verifiedAMDcheckpoints retained afterarchiveverification. Userstop remainssticky. Numericalfailuresstopforreview; no endlessretries. Earlystop requiresstrictacceptedcommonenclosure, usableinband/gapcheckedringdown and100savedMafterobservedoutgoingwavepeak.

## Controls and next action

```sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py status'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py cancel'
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/code/scripts/particles/session009/wake_check.py remove
```

Waitfor454538allocation, inspect actual48rankprobe andfullgate receipts;454539 schedulesonefiniteproduction successoronlyonacceptance. Donot manuallyduplicatejobs orchangephysics to bypass checks. Nevercancel unrelatedSTs9_prod. README/manifest maproots. NoSession009rawparticle/volume/checkpointdata onPerseus. Sourcecompiled6892be3e unchanged; canonicalinputSHAdeb6f631f7b06f8df6927662e01d6a3fa88bda4b0322ade3a1a07ff10f05dffd; frozen_script_bindingsevidence/frozen_config.json. Previousstatus retained in localhistory/mpi_failure_453631_20261007/.

## Scheduled-message delivery confirmed

At 2026-10-07T17:46:14.925615+00:00, the queued delivery-test message reached this same thread and triggered this check. Receipt: evidence/wake_queue_delivery_received.json. Live AMD job454538 still waits for resources; inspector454539 waits for it. Time remains0 and there are no stop flags. Both cron entries were verified. The test submitted no duplicate job and started no separate agent. The next regular two-hour check uses the repaired queue command.
