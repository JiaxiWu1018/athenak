# Session011 implementation and launch record

As of 2026-10-21 09:29UTC, the approved implementation is deployed. The single
AMD production job473069 has passed its release gates and is queued for six nodes,
24MPI ranks, capacity256blocks/rank. No production evolution or scientific endpoint
result exists yet. The hard endpoint is75Mref; extension150 requires a user decision.

## Implemented

- Separate persistent AH search state from published accepted surfaces; preserve
  provisional/common seeds, residual and geometry gates, and per-search evidence.
- Full-population component/shell/moving-AH/rest/energy/J diagnostics, exact-ID
  deletion records and restart population audits. Invalid measurements remain
  unknown; zero surviving components remain zero.
- Fixed90k component-stratified cohort, exact accepted spectral AH geometry,
  aligned mass/spin plots, four event snapshots, two1080p movies, finite-window
  wave/FFI products, five-hypothesis evidence and resource/storage reporting.
- Bounded checkpointed job continuation, GPU/host/block/storage/AH watchdogs,
  same-trajectory7/8-node fallback, protected milestones and verified recovery
  rotation. First-formation search failures do not trigger an AH-dropout pause.
- Endpoint-only non-evolving angular checks and product jobs; one-way compact
  artifact pull with per-file sizes/SHA256 verification on Perseus.

## Validation and exact versions

AMD engine df6ab41d5cdd95c7e33a8b70726891d52691c0cf passed473059 on four separate
MI210 devices/fourMPI ranks. Perseus CUDA18291 passed the same engine's analytic
frozen checks, serialization, invalid seeds, nonmutation, exact cohort/native
state readback, and32-to32 particle census, with zero evolution advances.

AMD473106 passes13 controller and7 product tests. These include a complete
synthetic report/figure/two-movie run, full movie decoding, and manifest checks.
No evolved pilot, replay, alternate seed or control trajectory was run.

Sealed AMD tooling3272c7bc4f6bd7e7d378ab6893e11033f0229ac9; branch
experiment/GI-in-cluster-session011. Every commit has been pushed. Later
Perseus-only release/documentation commits do not change the sealed AMD runtime.

Executable SHA256:
`e4fce096299d9d0ef852e7b713e3636afccb08a1d12a9c44cca9a657b9aa35ba`.
Input SHA256:
`11fb21b93abacf02e5410d77c7922967773535e329aabca44939a59a4026ef31`.
The input manifest records preserved Session010 physics/grid/RK4/CFL/seed.
Exact build/module/compiler/linker records are under AMD evidence/final_473059/;
compact copies are local evidence/amd_release_473069/.

## Locations and retention

AMD root: `/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021`.
Production log: `evidence/production_473069.slurm.log` there.
First run directory when launched: `runs/segment_0000/`.
Controller: `control/`; final products: `products/endpoint75_JOB/`.

Latest preflight whole-user AMD usage:481117669476bytes. Actual production raw
size is zero before launch; the watchdog will record growth every5min. All raw
production remains onAMD under the approved protected-state/recovery retention.

Perseus non-evolving raw evidence: `/data/jiaxiwu/NRPIC/GI_in_cluster/session_011_amd_ah_orbit_20261021/evidence/frozen_18291/fresh`,
354068308bytes, retained. Exact file inventory and checkpoint hashes:
`evidence/frozen_18291/RAW_INVENTORY.json` and `readback.json`. No raw cleanup,
raw transfer toMac, or unrestricted mirror occurred. Archive watcher18288 will
pull only completed manifest-listed compact products and verify every checksum.

## Remaining work

The scheduler must start473069. The first production checkpoint and full-population
audit remain to be observed. Automated successors run only after clean verified
progress; safety/science pauses retain evidence for review. At75 the saved-state
checks and products run automatically, followed by verified compact archival.
Scientific interpretation and the decision about150 follow the actual75 results;
no convergence or causal-dominance claim is made by this implementation record.
