# Session 008 implementation and operations record

## Source and approval

Independentcheckout underthissession; branchproject/GI-in-cluster. Base36b64e23 wasclean. Recovered repair263dcf21a7f6ce6ab1ccd8f1dd595ae0ca79c8c1 fromAnta preservedbundleSHA2563f0e220d56e4504062da6a396b318836aa20493b645d1c2bffac684d2e14fe27. Exactdiff raises boundedparameterheaderguard40KiBto1MiB only; fast-forwarded. Physics/inputcommit0c0a5a9bd47fa10ed8732ea12f1a53cd591b3f69 pushedtoorigin successfully. Kokkos6739bc623081648af9e752b616d9671527922cbf (4.7.02).

Pgenreads gi_clump1_bulk_vy/gi_clump2_bulk_vy anduseslocalorthonormalLorentzboost, storing covariantu_i=psi^2*u_hat_i. Clumprestweightnormalization usesboost-dependentmeanLorentzfactor. Fixedsourcemodelconformalgeometry itselfdoesnotdependonboost. Canonicalinputexplicitlysetsspeeds+/-0.133215; no pgen/pusherphysicschange required.

Pre-editMarkdownstreaminventory:1094paths/20,336,235bytes, per-fileSHA256/headings in evidence/MARKDOWN_INVENTORY.json. Governinglogisticsread; campaignSessions004–007/readactualinputs/laterOctober1record andcorrectedanalysis inspected. ExistingSTmigration447610wasunrelated andnotmodified.

## Build and test status

AMDbuild447620: COMPLETED0:0,313seconds,one16CPUdevelnode; 0.08694rawnodehours. Initial32CPUrequestwasrejectedbeforejobsubmission becausecurrentdevelnodeshave16CPUs. PinnedGNU12.2/OpenMPI4.1.8/CMake3.25.2/prun2.3/ROCm6.4.1; HIPGFX90A/MPI/O3/particles/gi_cluster. ExecutableSHA256cc7faaaadc8eb0eb74afdd3de81a748fb6adf9b069361496a8011ec86a7e5c2f. Actualtoolchain/source/submodule logs onAMD,evidence/build_provenance.txt andbuild/CMakeCache.txt.

Perseus10302: sixcriticaltests passed (real>40KiBcheckpointheaderlayout/truncation,latest-threeinventoryretention,finitegraph/duplicates,stickystop/scopedcancel,changed-inputgate,submissioncrashrecovery); Pythonandshellsyntaxpassed. CPU-onlysubmissionwasrejectedbysitepolicy; actualjobrequestedoneGPU/oneCPU/10minute maximum. Testsareoperationschecks, notGPU/numericalvalidation. Full5MAMDstartup/restart gates remainpending.

## Durable workflow

Build -> exact3nodegate -> afteranyinspector -> atmostthree finite4hour evolutionjobs, eachfollowedbyafteranyinspector. Totalmaximum44.5rawnodehours, leaving3.5oftheapproved48unreserved. Noauto numerical/infrastructureretries. Stateupdatesatomicandflocked; pre-submitteddependencies, campaignuniquejobnames andpending-submissionrecoverypreventduplicates. Source/input/executable/scripthashesbindgates. Fortyminutes are reservedwithin eachfourhour jobforcheckpoint/finalization; stopfilepoll16cycles. Lastthreeverifiedproduction/testcheckpoints globallyonAMD; metadata/header/census/exactfilesize/SHA256verification beforepruning.

AMP/MPIusesvalidatedmulti-nodeauto plus OMPI_MCA_btl_vader_single_copy_mechanism=none; noVistaMPIsettings. EveryrankwrapperproducesVRAMsamples, one sampler/node forallfourcards; gate requiresall12GPUs below85%. WholeAMDuser-root five-minutestoragewatch toleratesdufailures, warns1.5TiB, stops1.7/projected1.9; campaign1.25TiB with64GiBcheckpointreserve. Numericalfaults/absentprogress/invalidcheckpoint stopforreview.

Anta-to-AMDSSHworks; AMD-to-Anta isnetworkblocked. FiniteAntaCPUjobspullsealedscience, verifyagainstAMDcompute-generatedSHA256manifests, thenremoveonlyapprovedAMD .bin/.vtk/.cbin copies; keepcheckpoints underseparaterotation pluslogs/inputs/manifests/rawwaveforms/failureevidence. Archiveheartbeat gatesproduction; absent/staleheartbeat halts safely. Antastagecap1TiB/free-reserve512GiB. Four6hourAntaallocations maximum; noagent/livesshdaemon dependency. Analysis/rendering onAntaSlurm; localcheckpointbackupnotrequested.

## Interpretation and outstanding work

Reused strictaccepted-horizon/identityfixtures fromSession0076531/6533; no newAMDaccepted-horizonfixture claimed. ExactS8gatesverifycensus/ledger/boost/finiteoutputs/restartanddiag gaps; matchedconstraintscompareS7boosted0.0125norms onlyifphysicaltimes/regions/volumes agree. Horizons remainmeasurement-onlyandstrict. RawrPsi4 isprimary; t12/oldpropagationgrid do notsupport mergerstrain/ringdownclaims.

AppendactualGPUgate/job/checkpoint/accounting/resource/analysis outcomes whenavailable. Current scientificstatus is prepared/buildcomplete, notcompletedproduction. See README.md forexactstatus,gracefulstop,andemergencycommands; APPPROVED_PLAN.md (actualfilename APPROVED_PLAN.md) forlimits. No large rawdatawereproducedonPerseus by thissession.
