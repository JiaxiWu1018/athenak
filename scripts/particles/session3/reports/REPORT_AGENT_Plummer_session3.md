# Plummer session 03 technical record

Status at 2026-10-05 22:54 UTC: initialization, CPU diagnostics and matched
four-GPU AMD preflights validated. Production is stopped because its measured
403.08-node-hour projection exceeds the authorized 200 hours. Awaiting the
user's revised scope or ceiling. No production simulation is running.

## Source and reproducibility

Session root:
`/data/jiaxiwu/NRPIC/Plummer-cluster/session_03_isotropic_benchmark_20261005`.
Independent checkout: `src/athenak`, branch `project/Plummer-cluster`, origin
`git@github.com:JiaxiWu1018/athenak.git`. Starting commit
`aadffae2b15e6ff59c73d7a5c5327a97572d7c4c`; Kokkos
`6739bc623081648af9e752b616d9671527922cbf` (4.7.02). Historical shared `../code`
was left untouched. The new clone is registered in the standing source inventory.

Focused commits 16656468, 4210d505, b63917f9, 4ebd7565, 17739a86 and 5c0f38b0
are pushed to the matching origin branch. Session scripts, exact decks, analysis,
reference evidence and standing plan are versioned under
`scripts/particles/session3/`. The prompt and derivation are copied unchanged
into the session root, with the originals retained and matching SHA-256 in
[source_documents.sha256](../evidence/source_documents.sha256).

## Allocated validation and build record

| Jobs | Purpose and outcome |
|---|---|
| Perseus 11041, 11048, 11084 | C++/independent Python continuum comparison, 354 checks pass; latest applies relative CDF tolerance |
| Perseus 11042, 11046 and later focused rebuilds | Serial/MPI build; successful |
| Perseus 11047 | Mesh inspection: main 624, coarse 568 leaves; pass |
| Perseus 11058, 11064 | Full-N frozen/live t0; sampling checks pass |
| Perseus 11072, 11073, 11077 | New COM ledger build/t0/verification; pass after correcting a test's absolute sum tolerance |
| Perseus 11079 | Sticky health failures, endpoint/checksum safeguards, syntax checks; pass |
| Perseus 11078 | Static initialization PDF/PNG generation; pass and visually inspected |
| Perseus 11081 | Complete retained raw-data size/checksum inventory; pass |
| AMD 452552 | ROCm build at b63917f9; pass, superseded by corrected cleanup build |
| AMD 452585 | ROCm build at 4ebd7565; pass, 338 seconds |
| AMD 452594 | Four-rank frozen 100-step preflight; completed, healthy checkpoint |
| AMD 452595 | Four-rank live 100-step preflight; completed, healthy checkpoint |
| AMD 452612 | Build at 5c0f38b0 with common COM bands; completed, 316 seconds |
| AMD 452625, 452651 | Frozen/live preflight reductions; completed, 16 seconds each |
| AMD 452637 | Budget analysis; intentionally returns 2 for failed budget gate, evidence retained |
| Perseus 11085, 11086 | Scope alternatives and preflight figure generation |
| Perseus 11087 | Preflight figure revision with histogram bounds; visually verified |
| AMD 452661 | Complete retained raw checksum inventory and reduced-transfer checksums; completed |

AMD modules are gnu12/12.2.0, openmpi4/4.1.8, cmake/3.25.2, prun/2.3,
rocm/6.4.1. HIP builds use GFX90A, MPI, O3 and
`PROBLEM=particles/nr_pic_plummer`. Four-GPU runs use four ranks and
`--kokkos-map-device-id-by=mpi_rank`; vader single-copy remains disabled.
Builds run in the devel partition with 30-minute bounds; production uses
exclusive mi2104x nodes. Each segment records module versions, binary/deck
hashes, the exact launch and host/job identifiers.

The 4ebd7565 AMD binary SHA-256 is
`32cc99d530d3defaeada5c4a34489b08ec9c1f11065a7614d580cae8c25ca74d`.
The 5c0f38b0 AMD binary SHA-256 is
`792a1de94b231c6289fdc993c19015b42ec64cce26c3abf1c745a84e388da613`.
It compiles successfully; its additional COM-band kernel has passed CPU t0,
but has not been exercised in a four-rank GPU evolution. The completed AMD
preflights use the preceding 4ebd7565 diagnostic binary.
The latest CPU ledger-test binary hash is
`b8c26852ea57f67b16550180127a3a28b02071e69dc2999ec44e8d94f588b519`.
The CPU compilation preceded the 5c0f38b0 commit but uses its C++ diagnostic
content; do not substitute a Git identifier for the recorded executable hash.

## Corrections and preserved failures

Mesh inspection 11045 exposed an existing uninitialized MeshBlockPack coordinate
pointer in the pre-physics cleanup path; its constructor is now initialized.
CPU t0 11052 exposed a new global Kokkos cache surviving Kokkos finalization;
the problem final callback now releases it. The first callback registration
attempt in 11056 used Mesh.pgen before its constructor returned; registration
now occurs through the problem-generator member. All failed artifacts remain.
The independent validator's NumPy boolean serialization was corrected and rerun.

The live ADM array omits gauge components. Diagnostic lapse/shift access was
corrected to the owning Z4c array. The characteristic bound now includes the
inverse conformal metric for the legacy longitudinal shift mode. Corrected
analytic t0 bound is 1.4141532911; this is not a future-run bound.

Supplementary AMD single-rank startup 452606 failed before initialization because
the full live mesh exceeded the development VM's device memory. It is preserved
under `runs/sanity1_live`, excluded from four-GPU throughput inference, and not
retried. Original pending/held session jobs 452534/548/549/553/554 were cancelled
when replacing the queued build and using shorter preflights. No unrelated job
was cancelled; ST-migration and the other concurrent sessions remain separate.

## Health and continuation

The in-code field/particle checks reject non-finite states, invalid spatial
metrics, particle loss and relative rest-mass error above 1e-10. Finite lapse
below 0.2 or three successive constraint samples above ten times the pilot's
post-transient reference request checkpoint-and-stop. The shared Python health
module bounds runners, continuation and downstream products at the last healthy
time. A repeated valid row cannot erase an earlier failure.

The continuation service `scripts/controller.py` polls registered jobs only.
All numerical analysis runs through `amd_analysis.sbatch`. Slurm accounting
reserves job limits against 200 node-hours; production retains a postprocessing
reserve. A resource failure permits at most one retry from a verified checkpoint.
Explicit extension to a later endpoint is supported, while a completed endpoint
cannot be redundantly restarted. No checkpoint or raw output is pruned.

Production stays closed until measured budget, validated build and pilot gates
exist and pass. The quarter-period pilot is extended to 0.30 P_ref to collect a
post-transient constraint reference. A matched interrupted arm terminates on
cycle 740 and restarts to that endpoint; short baseline and half-dt arms cover
0.025 P_ref. These are gates, not the production convergence controls.

Finite-radius extraction points at R=9000M/10000M share the dx=160M refinement
level. Eight-point derivative stencils can meet AMR seams. Initial analytic
positive controls pass; later causal placement must be checked against measured
characteristic bounds. Do not describe these spheres as stencil-seam-free or
describe the infinite metric as vacuum outside the particle sampling cut.

Physical stresses use a metric-orthonormal frame. Older coordinate-velocity
dispersions remain auxiliary. The legacy particle energy history omits the live
shift contribution and is auxiliary; frozen E/L invariant diagnostics and the
new physical stress ledger are the intended interpretation sources. Boris
fallback counters restart locally; report segment increments rather than
mistaking them for a serialized global cumulative counter.

## Raw-data paths and retention

AMD root: `/work1/eliasmost/jiaxiwu/plummer_s03_20261005`.
Completed preflights: `runs/preflight_frozen` and `runs/preflight_live`.
Their out/ payload sizes are 11,802,547,806 and 16,852,874,003 bytes respectively,
as measured in allocated job 452637. Both are retained on AMD. Run manifests,
logs and compact diagnostic ledgers are copied to `evidence/amd_runs/` on
Perseus; reductions were computed on AMD and copied into `reduced/` for plotting.
The `reductions` directory is an alias to `reduced`. Four-hour
production segments reserve 15 minutes for final output; short preflights
reserve 20%. Histories P_ref/200, particles P_ref/100, movie cell slices
P_ref/50, coarsened ADM P_ref/10; milestone and segment checkpoints retained.
Live lapse slices explicitly use `z4c_alpha`, since `adm` excludes live gauge.

Whole run-directory sizes from AMD inventory452661 are11,802,560,951 bytes
frozen,16,852,889,782 bytes live,10,496 bytes failed single-GPU startup.
[amd_raw_manifest.tsv](../evidence/amd_raw_manifest.tsv) records every retained
file and checksum. Anta destination planned for
`/data2/jiaxiwu/Plummer-cluster/session_03_isotropic_benchmark_20261005/AMD_raw/`;
not created or populated while /data2 is full. Current AMD accounted use,
including the inventories, is0.6072222222node-hours; no session job remains active.

Perseus retained raw initialization trees, bytes from job 11081:

| Relative run path | Bytes |
|---|---:|
| cpu_t0_frozen (finalization failure evidence) | 11,717,767,921 |
| cpu_t0_frozen_11056 (callback registration failure) | 2,273 |
| cpu_t0_frozen_11058 (healthy) | 11,717,767,248 |
| cpu_t0_live_11064 (healthy) | 16,781,406,277 |
| cpu_ledger_t0_11073 (compact diagnostic evidence) | 197,142 |

Full inventory: [perseus_raw_manifest.tsv](../evidence/perseus_raw_manifest.tsv).
All are retained; no cleanup is authorized. /data has approximately 1.2TiB free
at the current check. Anta /data2 reports zero available space, so no large
archive transfer was attempted. The whole AMD user root retains the five-minute
watchdog: warn 1.5TiB, stop our growth 1.7TiB, projected cap 1.9TiB. A transient
du failure retries instead of terminating the watchdog.
