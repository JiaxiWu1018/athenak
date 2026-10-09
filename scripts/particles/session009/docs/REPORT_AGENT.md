# Current update — October 08, 2026 07:35 PM PDT

## Periodic checks cancelled by the user

No registered AMD or Anta job is running or queued; the verified checkpoint is still t9.33125. The two-hour checks have been cancelled at your request and persistent wake-disable intent is recorded. Usage remains 72.697778 AMD raw node-hours and 3.851944 Anta allocated node/GPU-hours. Saved plots, movies and verified archive receipts are retained. Cancellation receipt: evidence/two_hour_wake_cancelled.json.

**Simulation stopped cleanly at t9.33125M_ref after a GPU reached85.075%, crossing the approved85% memory limit.** No simulation job is running or queued. The final checkpoint and segment science outputs are verified on AMD and checksum-archived on Anta. REQUEST_STOP/RESOURCE_STOP remain set; no automatic restart, tuning or physics change. Controller time0.03125 is stale because its continuation inspector refused the stop; it is not current science time.

Strict individual horizons are accepted for both clumps by t8.48594. At t9.32969 their measured horizon masses are about0.074953 and0.074854M_ref (distinct from0.12 source parameters), with small inherited spin estimates0.002973/0.004864. M_ref=1 is the fixed inherited reference unit, not a measured horizon or ADM mass. The0.133215 local Lorentz boost remains approximate companion support, not exact circular GR initial data; K_ij≈0 leaves the local momentum constraint unsolved. Only about1.52degrees of motion are saved after both horizons have accepted measurements; this cannot establish nearly circular motion or sustained inspiral. No accepted common horizon, merger or ringdown waveform exists. Raw R40 complex multipoles are retained through t9.32656; strain is deferred for insufficient time/causal coverage.

The final count is3,996,300 surviving particles. The exact removal ledger conserves all five million initially simulated particles: envelope1,999 removed; left501,109; right500,592. No imposed recoil, new symmetry or reduction of the initial particle count. Saved metric fields in all three latest diagnostic planes are finite, checked by allocated Anta2423; this is not convergence evidence.

Usage: **72.697778 AMD raw node-hours**, **3.851944 Anta node/GPU-hours**,11/96Antajobs completed. Last allocated AMD storage was0.217TiB session/0.912TiB whole user before archival cleanup; no fresh post-cleanup size is claimed. Anta archive measured0.644TiB. Generated cost estimates using stale controller time were corrected with originals preserved; no remaining-cost forecast is issued while memory feasibility requires review.

## Recommended review products

- [Separation versus time](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/separation_vs_time.png): surviving-particle centers, trackers and simultaneous accepted horizon centers are distinct; the short post-formation interval and diagnostic gaps remain visible. Full centroid separation5.99922→6.02825; AH separation6.06094→6.06769.
- [Accepted horizon masses and spins](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/accepted_horizons.png): only strict accepted measurements, with gaps preserved; spin estimates are qualitative.
- [Central movie final frame](/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/figures/stopped_review_2408/central_frames_00036.png): separated blue/orange clumps and their envelope at t9.331. Full movie on Anta:`/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/analysis/update_final_2408/central.mp4`.
- Envelope/context and fixed-grid-density movies on Anta in the same directory:context.mp4 and density.mp4. Representative final frames and plot values were inspected. Rendering subsampling never changes simulated particle counts.

This is a partial production closeout after a resource stop; no t12 or50M milestone was reached. The two-hour checks were cancelled by the user; Anta removed its terminal archival trigger after completion. Stop flags remain intact. Exact status, original limits, source/configuration hashes, outstanding memory/operations review and scoped controls are in evidence/CURRENT_STATUS.md. Receipt:evidence/scheduled_check_20261009T000844Z.json.

## New operational review

Anta2423 used the established long partition, one A100 GPU, four CPUs,64GiB, bounded4hours and no requeue. It completed0:0 in1allocated second. New standalone scripts:code/scripts/particles/session009/reviews/stopped_run_20261009/review.py and review.sbatch; exact deployed SHA256s and registration are in evidence/stopped_review_submission_20261009.json. The script requires SLURM_JOB_ID, preserves original reports/resources, checks three latest sealed planes via symlinks and corrects time coverage using the archive/checkpoint receipts. AMD frozen44runtimefiles/source/input/binaries remain unchanged. No production submission. New analysis receipt:evidence/stopped_review_2423.json; visual review:evidence/visual_review_2408.json. Reproduce only in an allocated Anta job with the preserved review.sbatch; an existing REVIEW_COMPLETE.json makes reruns exit without replacing evidence.

## Earlier dated setup and status records



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
