# Session009 submission and durable handoff

Frozenconfiguration createdUTC 2026-10-05T22:01:43.732344+00:00; deadlineUTC 2026-11-19T22:01:43.732344+00:00. User-facingdates useAmerica/Los_Angeles. No Session009 evolution result exists atthis submissionrecord. Build started onAMD k006-004-v2; scientificchecks queued withdependencies. No expensivepilot willbe repeated.

| Job | Role | Allocation/maximumtime | Status atsubmission |
|---|---|---|---|
|452576|GI plusbuilt-inlinearGW build|1node,devel30min|running|
|452578|linearGW wavelengths2.5/5,80M|3MI210nodes/12ranks,2h|queued afterbuild|
|452580|full5M initialization,48rankMPI,output/memory/restart gate|12MI210nodes/48ranks,4h|queued afterwavegate|
|452582|inspection andconditionalfirstproduction submission|1node,devel30min|queued aftergate|

Maximum possible rawAMD exposure ofthese fourregistered allocations: **55node-hours**. A measured snapshot atbuildelapsed348seconds was0.0966667rawnodeh; queuedjobs costzero. This is a timestampedsnapshot, not a laterfinalcost. Each successfulinspector submitsone12-node segment≤12h andits30mininspector, reservingfullpossibleusage beforeeitherjob iscreated. No productionsegment iscreated untilmandatorygates pass. Firsttargett12, thencheckpointed50Mmilestones through400; startup acceptedreferencecycles0..2 andmatchedoutputcycles2..5 arekept aspartofsciencehistory, othervalidationbranchesexcluded. No force/boost/separationtuning.

Sourcecompiledsnapshot: **6892be3e3f04ec573f91cb2034bc9d3009a3bdff**, branchproject/GI-in-cluster, successfullypushed andremoteHEADverified beforebuild. Kokkos6739bc623081648af9e752b616d9671527922cbf. Diagnostic-onlytimestampfix fe68d684; scientificinput1519986f; lateroperations/analysis corrections separatelycommitted. Restart-header263dcf21 ancestorcheckpassed. InputSHA256 **164002e107ae3023c37b206a4e83923410c38c5de7c18c9f704afd48f947ffff**. Exactfrozenflat-scripthashes andlimits in frozen_config.json. Executablehashes/buildprovenance aregenerated onAMD andremainpending untilbuild completes; do notinfer orinvent them. Any subsequentdocumentation-onlycommit isdistinctfromthisbuilt/frozenrevision.

PerseusSlurm11050/11055/11059/11065/11066/11067 eachpassed9targeted retention/resource/gap/FFI/strictAH-join tests andPython/shellsyntaxchecks attheirrespectivepreparationrevision. Finalpre-freeze suite11067 passed. NoPerseusbuild/simulation/bulkrawanalysis ormovie rendering. Perseusaccountingstorage disabled; joblogsarepreserved. No rawscience/checkpoint/volumeoutputs fromthissession onPerseus; source/docs/inputs/QAevidence retained. `/data` was99%used with~1.2TiBfree atstartup, soheavydata goesdirectlyAMD→Antapull.

Theinitial01:30devel buildsubmission wasrejected bytheAMDsitefilter (max.5h), withzeroallocatednode-hours andnojobID. Evidence initial_submission_rejection.md andAMD aborted_initial_freeze.json preserved. Fixedsource9df771b5 validatesacceptedbuildID beforefreezing andusesdevel30min; subsequentfourjobswereaccepted. No scientificparameter orsitechange. SourceROCm7.2default isexplicitlyreplacedby pinned6.4.1 inthebuildlog.

Anta /data3 filesystemrechecked~27TiBfree; /data2full. Destination /data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005. Boundmetadata-onlycroninstalled, markerJEANS9_20261005_METADATA; initialheartbeat succeeded. Existingcrontabpreserved; currently noAntaarchivejobneeded untilsealedoutputs exist. Deadline45days,≤96jobs×4h (mandatory1GPU,8CPU,64GiB). Archive/analysis cap16TiB,512GiBfree reserve. AMDsession1.25TiB,256GiBfinalizationreserve pluswholeuserwarn1.5/stop1.7/projected1.9TiB. Raw nodeh tracked separately fromAntanode/GPUhours; no dollarcostwithouttariff. AMDwebsite `df` free capacity does notoverride per-userbudget.

AMD /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/control/state.json isauthoritativejob/statusledger. Source/data manifests frozen; Anta receipts required beforeapprovedAMDsciencecleanup oroldercheckpointrotation. Latestthreeverifiedcheckpoints retainedAMD. Numerical/MPI/memory/gate/archivefailure haltsforreview; noautomaticretry. SourceSession008 andunrelatedSTjob452524(s9_prod) untouched; nevercancel generic s9_prod.

## Exactcampaign commands

```sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py status'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py cancel'
ssh hpcfund.amd.com 'squeue -h -j 452576,452578,452580,452582 -o "%i %j %T %M %R"'
ssh jiaxiwu@anta.caltech.edu 'cat /data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/evidence/active_archive_jobs.json'
```

`stop` issticky andrequests a cleancheckpoint. `cancel` cancelsregisteredJeans009jobs onlyanddoesnotguarantee a newcheckpoint. Workflow/archiving/analysis require nointeractiveagent orshell. Node/calendar/storagecapsremainhard. A commonAH alone neverstops; approvedearlysuccess requiresprovenenclosure, usablein-bandringdown withgaps/matterchecked and100Msavedafterwavepeak. t400withno merger orcompletewaveform ishonest possibleoutcome.

## Outstandingchecks/products

AMDactualcompletebalancedmesh,initialledger,finitefieldoutputs,all48VRAMpeaks/MPImessages,large-headerrestartcomparison andlinearpropagationerrors arepending. No scientificrunhasyetstarted, noS9plot/movie/result isclaimed. Onsuccessfulgates, firstt12collapsecheckisproductionhistory. Antaauto-updates initial/12/50/100/...400/final withseparation/motion/horizons/removals/covariantJ/health/rawGW/conditionalFFI pluscentral/context/fixed-gridmovies; exactvalues dependondata. Checkrepresentativeframes andplots onceavailable; encoder/decodechecks aloneareinsufficient. ComparisonwithretainedS8usesonlysharedt≤12 andcannotimplya distantmergerwaveformthere. Single-r50 cannotmeasure radiusdependence/extrapolateinfinity. No precisionconvergence/equilibrium claim.
