# Jeans-in-cluster Session 008 — setup and assessment status

## Approved setup

This is a fresh t=0 calculation with approximate companion support in the **initial** clump boost. The evolution already includes both clumps' gravity. The approved local orthonormal speed0.133215 is applied with the existing Lorentz transformation to the thermal samples: left-y/right+y, counterclockwise from+z. Its approximate Newtonian companion motivation is not an exact GR circular-orbit construction. No speed tuning or new force is used.

Envelope: source0.76, arealR30,3Mparticles. Clumps: source0.12each, isotropicCartesianx+/-3,1Mparticleseach,sigma0.70,s0.02. Five million initialparticles and deterministicseed4001/componenttags are retained. Restweights and covariantmomenta are regenerated. For fixedsourceprofile, the analyticcombined conformalgeometry is boostindependent; it is freshlyinitialized rather than takenfromanoldcheckpoint.

The source/model sum1 defines the inherited referenceunitM_ref=1. Sampledrestmass, anyADMestimate and measuredhorizonmasses are distinct. ApproximateKij0 leaves nonzero localmomentumconstraints unsolved; cancellingglobalboosts does not solve them.

## Numerical setup and limits

Session007 mesh/evolution baseline: domain+/-256/rootdx2/32^3MeshBlocks; logicalroot3, physicallevels0..9, finestdx1/256; compactinitialtowersdx1/64; Lohneralpha*psi7 threshold0.2 andtrackerfloorfalse; RK4/CFL0.4 andvalidated gauge/deposition/feedback/pusherguards. Lapsealpha<0.05 removesparticles; AHremovalOFF. No explicitwave-zonefloor is addedforthisassessment. Sommerfeld/outflowboundary gaugeestimate reachesR30 onlyaroundt160, beyondt12.

Three AMD MI210nodes/twelveMPIranks. Targett12, hard48rawAMDnodehours includingpreparation; threefourhour evolutionsegments maximum. Initial coordinateperiod~227, so acompletepostcollapse revolution isoutside thisstage. t50continuation needsmeasuredcostreview. No merger/removalplateau/commonAHstop is authorized.

ComplexrawrPsi4 ell2..8 at40/50/60/70 is retained atdt0.025. At t12 it cannotcontaincentralcollapse radiation arrivingfromt~8. No resolvedmerger/ringdown, strain, radiation-driveninspiral, or precisionwaveform claim is made. Companion-supportedmotion, rapidplunge, and partialearlyorbit remain possibleoutcomes.

## Current evidence

Source/inputcommit0c0a5a9b and repair263dcf21 were pushed successfully. AMD HIP build447620 completedin313seconds usingpinnedROCm6.4.1/GFX90A. FullGPU initialization/restart/physicsoutcome are **not yet tested**. PerseusSlurm10302 passed sixcriticalworkflow tests plusPython/shellsyntax checks. No finalresultplots ormovies existyet.

## Planned review products

Anta science/analysis destination: `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/`; availablefilesystemspace27.10TiB atinspection. Central/contextfixed-tagmovies, coordinateorbitplots, rawearlywaveplots and diagnosticrecords will be generatedafterverifieddataexist. Rejected/stalehorizons and restartreacquisition gaps are excludedfromphysicalmeasurements. Acommonpublication alone willnot be calledmergerwithoutusableenclosureverification.

The finalassessment update mustreportactualendpoint/jobstate, initializationledger/samplingresiduals, peakmemory/mesh/checkpointtiming/storage, collapse/tracking findings, availablemovies/plots and estimatedcosttowardt50. This documentcurrentlyrecordssetup andcompletedbuild evidence, not finalscientificresults.
