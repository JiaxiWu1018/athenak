# Session009 R40 technical status and reproducible handoff

Updated 2026-10-06T22:47:06.919839+00:00. Full startup453631 pendingresources; inspector453632 pendingdependency; fresh physicalt0, no checkpoint. Authoritative state/accounting remains onAMD. Original technical preparation report and configurations are preserved in history/r50_before_reset_20261006/.

## Frozen revisions and hashes

Compiled source unchanged:6892be3e3f04ec573f91cb2034bc9d3009a3bdff; Kokkos6739bc623081648af9e752b616d9671527922cbf. Restart repair263dcf21 is included by ancestry; its real>16KiB header gate remains pending. GW timestamp precision diagnostic commitfe68d684 is retained; source generator reads explicit approved local boosts and Lorentz-transforms thermal samples. Source and executable are not rebuilt for the input amendment.

Scientific R40/R46 input commitae89f066 pushed separately from operations6b9806eb and receipt-pathrepairddf31ebd47f515756637b643656d47cb465c3fa4. Later report/docs commits are distinct from compiledsource and frozen runtime scripts.

| Asset | SHA256 |
|---|---|
|R40 canonical input|deb6f631f7b06f8df6927662e01d6a3fa88bda4b0322ade3a1a07ff10f05dffd|
|GI executable|9717c98bed18a8d4307286115a402920aa2ca2fbf254049841aa3719b806c515|
|Wave executable|0e496446ba7791365285c2c53e4bb516d9a4e3fd3c401d6ec2cc0c06916b1764|

All36flat scripts and exact resource/calendar bindings are frozen in evidence/frozen_config.json =AMDcontrol/config.json. Compiled AMDathenak checkout remains frozen even as Perseus repo documentation advances. Provenance build flags/module/compiler/submodule/executable evidence under evidence/build_provenance.txt and *.sha256. No large rawdata/credentials committed.

## AMD configuration and checks

Root/workdir:/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005. Pinned GNU12.2/OpenMPI4.1.8/CMake3.25.2/prun2.3/ROCm6.4.1,HIPGFX90A,MPI,cc/hipcc,-O3. Automatic PMLselection, vader single-copy none, OMP/OpenBLASthreads1.12nodes×4ranks=48GPUs; prun wrapper --kokkos-map-device-id-by=mpi_rank. Every new allocation probes communication before loading particles/checkpoint. Unique runs/segment_NNN directories. Production application11h20 inside12h allocation reserves40minfinalization.

Perseus allocated meshcomparison11135 reproduces oldAMD7456blocks and revised5384, with mesh-tree source equivalence to retained CUDA executable documented. QA11150:14tests plus pycompile/bash-n/inputcontract passed. Actual AMD453630:11tests plus all three -n input contracts (full/slim/segment overrides) passed,4s1node. Prior453626 accepted commands but receipt from runtime import failed from actualbatchcwd; sys.path now explicitly includesROOT/scripts. Reviewed5s1nodefailure and cancelledgate preserved, no numerical rerun.

Retained AMD propagation452578:80physicalM atlambda2.5/5, interface mesh .25/.125/.0625, amplitude .0024702188/.0006263057fraction, phase .0070316209/.0000892951rad. Eightcycleenabled-extraction452666 pluspolarization scale .999964725/.999966014, residual<.002%; imaginary sign source audited. Original disabledWeyl zeros remain flagged as uncalculated fields. These do not replace fullproductionmesh propagation/convergence evidence.

New full gate453631 pending: exactcomplete leafmesh/spacings, full5Mcomponent ledger and positiveweights/boost signs/center/Jz/linear momentum residuals,48GPUmemory<85%, finitefields and actualbatchoutputpaths, two-cycle uninterrupted vs1+restart comparison, real large-header checkpoint, clean-stop behavior and allR40raw modes/planes/particles/3Doutputs. Each gate has receipt verification. Diagnostic reacquisition gaps remainmissing. Suitable Session007 constraint comparison only at matched time/regions/masks/volume-normalization; initial uncalculated zeros and global empty-volume norms are not improvement evidence.

## Audited reset and accounting

Before mutation, registered old jobs were allclosed, USER_STOP/REQUEST_STOP absent; oldfailedruns already checksumarchived byAnta2353. AMD/Anta history/r50_before_reset_20261006/ preserves metadata, oldscripts/inputs/reports and runmoves with RESET_INVENTORY.json. No data deletion or source/binary replacement. One-off scripts reset_r40_20261006.py and repair_receipt_453626.py are retained under scripts/ and versioned docs. Oldrunanalysis directories remain; latestR50report pointer/terminalcompletion flags moved tohistory so currentanalysis is not falselycomplete.

AMDhistorical ledger retained alljobs452576,452578,452580,452582,452666,453626,453627,453628,453630 plus newpending453631/453632. Actual0.9225rawnodeh=(644+3*834+12*12+1+3*7+5+4)/3600. Pending48.5 ->maximum49.4225. No charge forqueue orcancelledbeforeallocation. Anta115+116+118seconds=0.096944444node/GPUh,3/96. No accounttariff; noinvented monetarycost. CurrentAMDstorage sample awaits fullgate watchdog; priorvalues are historical.

Originalcreated2026-10-05T22:01:43.732344Z/deadline2026-11-19T22:01:43.732344Z unchanged; hardt400,10000rawAMDnodeh includingprep/failures/analysis,90segments,45calendar days. AMDsession1.25TiB, whole-userwarn1.5/stop1.7/projected1.9TiB,256GiBwrite reserve. Anta16TiB/512GiBfree reserve,96jobs×4h oneA100/fourCPU/64GiB. Approvedscientificstop needs strictacceptedcommonenclosure +usableinband/gapcheckedringdown +100savedMafterwavepeak. Numericalfailuresstopforreview,noautomaticretry.

## Durable monitors and continuation

AMDworkflow lockedstate and submissionwindow recovery prevent duplicates; separate R40 jobprefixjn9_20261006_r40b_. Inspector schedules at most one successor plusoneinspector, targett12then50Mmilestones. Stickyuserstop, resource/calendar checks, exact source/input/executable/script bindings, watchdog/archiveheartbeat and finiteprogress guards apply. No persistent interactive session required.

Antametadata-only cron */5minutes markerJEANS9_20261005_METADATA reinstalled; successfultick refreshesAMDheartbeat. Heavytransfer/hash/reduce/plot/render runs in allocatedAntajobs. Sealedscience ANDcheckpoints SHAverified before AMDcleanup; latest3verifiedcheckpoints retained. Original active_archive_jobs ledger3 and trigger deadline retained. Snapshot updates rawaccounting/config and onlyverifiedphysics evolution; duplicatevalidation branches excluded. Initial/t12/every50M/final updates refreshplots/reports/movies. Nextdataarrival ispending.

Perseusmetadata-only cron 0 */2 * * * markerJEANS9_R40_TWO_HOUR_WAKE installed. It resumes existingCodexthread01a0f9d8-f1e1-7250-b3b3-0bb0c6dfa014 with officialcodexexecresume whenidle,1800sboundpercheck; useslock and boundedread-only persistedthreadactivity. It nevereditsCodexdatabase orspawnsa newagent. Minimalcronenvauth/metadata/activeguard verified; actualidlemodelresumedeliverypendingaftercurrentturn. Receiptfiles evidence/two_hour_*.json(l), wake_*.jsonl/md, cron_environment_probe.json. No rawcredentials read/copied. Originaldeadline orstickyuserstop removesowncronentry. ContinuousSlurmworkflow operatesindependently ofwakes.

## Reproduction and products

On an allocatedAntacomputenode:

```sh
/home/jiaxiwu/miniconda3/bin/python /data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/scripts/analyze_s9.py --root /data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005 --milestone 50
```

CanonicalR40input/rawrpsi4_real_0040.txt+imag_0040.txt,77complexmodesell2–8plus17digitstime; savedquantityrPsi4. Restartsegments remainseparate and assembled withoverlap/gapchecks. AtoneR40 approximate retardedt−40 only; noarbitraryrescale/phasealignment orradiusconvergenceclaim. FFIuniform.025sampling,linear detrend,5%edge taper,cutoffs .003/.006/.012 ifsufficientcontiguouscausalcoverage; rawPsi4primary. StrictAHunique time/area/rmin/center join; iterationcolumnnotcycle. CovariantmatterJz=sum mu(xuy−yux), independentAH/removedJ, noexactsumconservationorBHmomentum=Mcoordvclaim. Acceptedcommonenclosure neededmergerclaim.

Central/context taggedmovies plusfullparticlefixedCartesian density|z|<.7slab,256²fixedbins/colors: coordinate restweight density, not properenergy density. Renderingcohort subsampling doesnotchange5Msimulation. Futurefigures/frames requirevalue/visualreview; noneavailable fromnewR40productionyet. Manifest/README/CURRENT_STATUS giveexactpaths andscopedstatus/stop/cancelcommands. NoSession009rawparticle/volume/checkpointdataonPerseus; smallsource/QA/docs retained atthisroot. Nevercancel unrelatedSTs9_prodbyname.
