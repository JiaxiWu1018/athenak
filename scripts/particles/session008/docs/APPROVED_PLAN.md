# Approved execution plan: Session 008 AMD assessment

Approved by “Implement the plan” after switching to execution mode, 2026-10-02.

- Fresh t=0, 5,000,000 particles; envelope source0.76/R30/3M; Gaussian clumps source0.12 each, centers(-3,0,0)/(3,0,0), sigma0.70/s0.02/1M each; seed4001, immutable component tags.
- Approved opposite local orthonormal Lorentz boosts0.133215; no velocity optimization, force, recoil, imposed spin, radial boost, constraint solver, or parameter survey.
- Retain Session007 evolution/mesh: domain+/-256, rootdx2, 32^3 blocks, compact initial towers, minimumdx1/256, Lohner(alpha*psi^7)0.2, tracker_floor=false; RK4/CFL0.4 and established gauges/protections; alpha<0.05 removal, AH removalOFF; approximate K_ij=0.
- AMD eliasmost/mi2104x, 3exclusive nodes/12ranks, pinned GNU12.2/OpenMPI4.1.8/CMake3.25.2/prun2.3/ROCm6.4.1, HIP GFX90A, validated auto MPI/Vader single-copy disabled; blockcapacity240 subject measured peakGPU<85%.
- Target t12; hard48 total raw AMD node-hours including preparation. At most3 four-hour evolution jobs (36raw), at most12raw preparation/operations. No automatic continuation towardt50, no automatic numerical retry. Reuse validation trajectory rather than duplicate a pilot.
- Verify actual mesh, full initialization ledger/particles, finite GPU evolution/output, actual checkpoint/restart handling, strict identity/acceptance, and honest diagnostic gaps. Matched Session007 initial constraints where feasible.
- History/coarsemesh/rawcomplex rPsi4 ell2..8 r40/50/60/70 dt0.025; three-plane metric/constraint/matter/Weyl dt0.1; fullparticles0.25; E3d1; 3D metric/constraint/matter/Weyl2; checkpoint0.5 andcleanstop.
- Latest3 verified checkpoints total onAMD only; transient fourth permitted duringwrite/verification. Inventory anddeleteoldest afterverified replacement; nocheckpointcopy toPerseus.
- Anta destination: /data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/. /data3 had27.10TiB available anddirectorywritable atinspection. Copy completedscience aftersegments or256GiB accumulated; checksum destination beforeauthorizedAMDscience cleanup.
- AMDcampaign1.25TiB; wholeuser warn1.5/stop1.7/projected1.9TiB. Antastage1TiB with512GiB free reserve. Bounded durable jobs, locks, stickyuserstop, exactcommands; failurestopsforreview.
- Slurm analysis/rendering, setup/collapse/orbit/acceptedAH/particles/health/rawwaveplots, central/contextmovies; human andtechnicalreports, manifests, campaignlog append, focusedcommits andverifiedpushes.
- Thisstage does not promise revolution/inspiral/merger/ringdown or resolvedhigh-frequency waves. Approximately227-unit initial coordinateperiod, sourceM_ref1 unit convention, unsolvedlocal momentumconstraint; longerproduction/finetuningrequiresnewdecision.
