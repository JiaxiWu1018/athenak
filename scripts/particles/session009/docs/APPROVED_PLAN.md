# Session 009 — approved execution plan

Authorization: the user explicitly requested implementation of this plan on2026-10-05. Fresh t0, same approved local Lorentz speed0.133215 left−y/right+y; no tuning, survey, new particle force, constraint solver, recoil or initial spin. Preserve Session008 and unrelated jobs.

## Physical and numerical configuration

Envelope source mass.76 atorigin,3M particles, spherical independently solved Einstein–Vlasov model, arealradius30. Clumps source mass.12each atx±3,1Meach, sigma.70 and orthonormal thermalspread.02, deterministicseed4001 and immutablecomponenttags. Exactly5M initial particles. ApproxKij0, unsolved local momentum constraint. M_ref=1 referenceunit, not a measured horizon/rest/ADMmass. Quantify residual momentum; do not impose symmetry or envelope recoil.

12exclusive MI2104x nodes,48GPUs/ranks, accounteliasmost. GNU12.2/OpenMPI4.1.8/CMake3.25.2/prun2.3/ROCm6.4.1, HIPGFX90A; vader single-copy disabled. RK4/CFL.4, validated gauge/damping/deposition/pusher protections, removealpha<.05, AH removal OFF. Domain[-1024,1024]^3, root256^3 dx8,32^3 blocks,ghost4, physicallevels0..11 finest1/256. Old compactclump towers remappedby+2levels; Löhneralpha*psi^7 threshold.2, tracker_floor=false. Continuous radialwavefloor dx≤.25 to46; .5to72,1to104,2to168,4to296. Explicit initial seed cubes conservatively cover these spheres; real balanced mesh and all48GPUmemory<85% must pass. No automatic coarsening or particle reduction.

Single coordinate extractionr40, complex raw rPsi4 ell2..8 every.025M. Full particles.25M, three diagnosticplanes.1M, full3Dsnapshots10M, checkpoints10M and clean segment stops. Retain raw multipoles and restart segments separately. Coordinate tetrad/radius/sign/normalization documented. Intendedfrequency≤.4 cycles/M_ref, wavelength≥2.5 gives≥10cells at.25, a heuristic requiring the actual linear propagation gate, not convergence proof. With one radius no radius-dependence test or infinity extrapolation. Boundaryweak-field gaugeestimate(1024−40)/sqrt2≈696M; inspect actual metric/shift and boundary speeds.

## Finite checks and execution

Existing linear-GW generator atwavelength2.5 and5, static refinementinterfaces, physical80M; amplitudeerror≤10%, phase≤.2rad. Full initialization ledger/census/tags/positiveweights/finitefields/centers/Lorentz signs/Jz/totalmomentumresidual. Actual48-rankMPIalltoall and Athena mesh preflight on each allocation. AllGPUmemory<85%. Uninterrupted2cycle vs1+restart2 comparison, real large-header checkpoint repair verification, clean stop, actual outputpaths and requireddiagnostics. Suitable matched Session007 initial constraints where possible, with time/region/volume/normalization matches; unmatched global-empty-volume norms never become claimed improvement ratios.

The first finite collapse assessment tot12 is retained as the beginning of this new production, then continued from its verified checkpoint. Segments≤12h with40minfinalizationreserve. Each inspector schedules at most one successor, with exact source/input/executable/script binding, submission crash-window recovery, uniqueJeans9jobnames and sticky human-stop intent. Numerical or MPI/gate failures stop for review; no retries or change ofscientific parameters.

## Termination and resources

Hard t400 OR10000rawAMDnodeh including builds/gates/failures/AMDanalysis. Maximum45calendar days,90finite evolution segments. Earlier scientific success only strict accepted commonAH proven to enclose both trackedobjects, usable outgoing ringdown at r40 within validatedband, gapschecked and≥100M saved after observed post-merger waveformpeak. A commonAH alone neverstopsproduction. Merger is not guaranteed. If the binary keepsorbiting or ringdown fails atcap, report incompleteoutcome honestly.

Planning4000–8000rawnodeh,2–4computingweeks on12nodes excludingqueuewait; revise from measured newmesh timing. No monetarycost withoutactualaccounttariff. Antaallocation reported separately.

AMDsessioncap1.25TiB plus wholeuser five-minute watchdog warn1.5/stop1.7/projected1.9TiB; transientdufailureslogged without killingwatchdog. Reserve256GiB forfinal output/checkpoint. Latestthree verified checkpoints onAMD; oldercopies onlyremoved after Anta checksumreceipt. No unapproved raw transfer toPerseus/Vista.

Anta /data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005, approx27TiB free atplanning; archivecap16TiB, standing512GiBfree reserve. Antapulls becauseAMD cannotconnect toAnta. Copy sealed science AND checkpoints, verify everySHA256 before AMDcleanup. Metadata-only finitecrontrigger; heavytransfer/hash/analysis/render uses mandatoryGPU Slurmjobs,≤96jobs×4h. Preserve existingcrontab.

## Reports and analysis

Initial/earlycollapse report, t50/100/...400 andfinal. Restart assembly neverduplicatesvalidation branches; diagnosticgaps remainmissing. Separateleft/righttrackidentities acrossx0/removal/restart. Strictindividual/commonAH acceptance, unique time+geometry summaryjoin; summaryfirstcolumn is finderiteration, notcycle. Separationfromsurviving tagged full-particle centers, live trackers and simultaneousacceptedAH centers; absolute/envelope-relative trajectories, unwrappedphase/Omega/revolutions afterbothindividualAHs, radial/tangentialratio with small-denominator masks. Eccentric closepassages/coordinate shrinkage do notaloneprove sustainedorbit or radiation-driven inspiral. Acceptedmass/spin/quality/searchfailures, component/removal histories, covariantmatterJ, constraints/health; noexactremovedJ+AHJtotal or BHmomentum=Mcoordv.

Raw (2,±2) first, selectedadditional modes, approximate t−r only; supported finite-radiusstrain with FFI convention verification andcutoffsensitivity, rawPsi4primary. Radiated E/Joptionalonly ifnormalization/coverage/resolutionsupport. Central/context tagged movies and fullparticle fixedCartesian densityslab pipeline; rendering subsampling neverreducessimulatedparticles. Inspect representativeframes/plotvalues when dataarrive. HumanREPORT_Jeans9.md, technicalREPORT_AGENT.md, README/manifest andappendcampaignlog. Record exactsource/submodule/toolchain/input/exehashes/jobs/checkpoints/resource/storage/reproductioncommands andrealpushstatus. Preservehistoricalresults. Freezeandpushfocusedscientificinput andoperationschanges separately. Durable bounded handoff ifrunoutlivesagent; neverlabelqueued/partialworkcompletedresults.

Ringdown assessment reference: https://pages.jh.edu/eberti2/ringdown/ ; fitcoefficients: https://pages.jh.edu/eberti2/ringdown/fitcoeffsWEB.dat (checked2026-10-05).

## User amendment — October 6, 2026

The user authorized extraction R40 and a fresh reset from t0 after the reviewed startup input-contract failure, plus automatic wake/check every two hours. Retain the recommended six-unit fine-grid buffer through R46. Exact initial seeded count5384blocks (2072 fewer than R50,27.79%). Source particle model, boosts, seeds, count, central ceiling, other outer transitions, output cadences, all resource/calendar/storage caps and accepted-common/ringdown-plus100-tail rule remain as approved. All prior allocation and Anta jobs count toward the same campaign caps. Preserve failed history and numerical evidence. No Vista transfer is authorized by this amendment.
