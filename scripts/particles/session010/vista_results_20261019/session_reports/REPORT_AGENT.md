# Session010 — technical report and Vista reconciliation

Updated October 19, 2026 UTC / October 18 PDT. Start with [REPORT.md](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/REPORT.md)
for the scientific interpretation. This report reconciles the supplied Vista
archive; it does not authorize a new run, extend the old resource limits or
claim live scheduler status.

## Evidence hierarchy and current endpoint

The supplied [summary](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/session010_summary_20261019/README.md) is retained intact.
Its [session history](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/session010_summary_20261019/reports/SESSION_HISTORY.md),
[raw-location report](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/session010_summary_20261019/reports/RAW_DATA_PATHS.md),
[data guide](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/session010_summary_20261019/reports/REDUCED_DATA_GUIDE.md), native
receipts and exact scripts are the source evidence. All 4,044 manifest-listed
files passed the strong SHA256 verifier in Perseus Slurm job 17024; the verifier
also rejects unexpected files, symbolic links and hardlinks. No contents were
added to the supplied summary.

Use the [October 19 scheduler snapshot](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/source_inventory/SCHEDULER_SNAPSHOT.json)
over the older copied `HEALTH_AND_CONTINUATION.md`: 1075832 completed; 1075833
was cancelled by UID 896508 before allocation at October 18 19:28:02 CDT
(17:28:02 PDT). No third-segment output is present. The snapshot was captured
2026-10-19T05:27:04Z; this task did not contact Vista to establish later status.

The final native checkpoint is t=221.9531250005331, cycle 138334, dt=0.0015625,
3,984 leaf blocks, 17,073 active particles and 4,982,927 low-lapse removals.
Its Vista path is:

```text
/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_012/out/rst/s10_orbit_gw.00020.rst
```

Bytes: 50,996,709,000. SHA256 from the October 18 Vista native audit:
`fd11638112e5778b0255b4d6bab9c3652b281151436de224d943e6e8a1c1009c`.
The [native audit](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/vista_analysis/RESTART_HEALTH.json)
is preserved evidence, not a new Perseus read of that omitted checkpoint.

## As-run source, executable and input

| Binding | Value |
|---|---|
| Native science commit | `386870f88f60e25e1a8fac76fe04a071b87a01cf` |
| Kokkos | `6739bc623081648af9e752b616d9671527922cbf` |
| Native build | Vista GG job 1069850, i614-112, aarch64 |
| Modules | nvidia/24.7, openmpi/5.0.5, cmake/4.1.1 |
| Native executable SHA256 | `a6221fb3ac3ae9f3e17208bd2fb1d7811ab1c4040adcf5272cab40a7c68218e7` |
| Production input SHA256 | `aa712954cd7d0594395082670893378530bf8f7efc047ee3f51224a19c389ffe` |
| Verified notification email | `jiaxiwu@caltech.edu`, BEGIN/END/FAIL |

[BUILD_VERIFIED.json](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/BUILD_VERIFIED.json)
binds the binary and toolchain evidence. Original build logs, CMake cache,
ARM/Hopper executable and source bundles remain under
`session010_summary_20261019/workspace/`. Do not execute the aarch64 binary on
Perseus. The native recipe uses CUDA, HOPPER90, ARMV9_GRACE, MPI, Release,
`PROBLEM=particles/gi_cluster`, gcc and Kokkos nvcc_wrapper, plus the supplied
Kokkos 4.7.2 ARM/SVE CMake compatibility include. See the exact
[build script](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/scripts/vista_original_20261019/launch_s10/build.sh) and
[toolchain provenance](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/session010_summary_20261019/workspace/runtime_s10/build/evidence/native_build_provenance.txt).
Session007's executable was not reused.

The source includes the historical-buffer-reuse adaptation, capacity-safe
particle VTK repair and full-precision persisted Reals. Earlier matched memory,
AMR/MPI, particle, restart, controller and accounting gates are preserved in the
[preparation agent report](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/history/perseus_preparation_20261013/REPORT_AGENT.md).
This task repeated no build or historical validation campaign and changed no
native physics or numerical code. Later Git commits add analysis and reports;
the native compiled source stays pinned above.

## Runtime decisions and execution history

The owner-approved chain used 16 GH nodes with one MPI rank/GPU per node and
finite 8h+24h+24h allocations. The final recovery requested the two 24h legs;
only its first leg ran. Native checkpoint chunks were approximately two hours,
with a forty-minute finalization reserve. The actual native input contains
`nlim=-1`, `tlim=1e30`; these are parser compatibility settings, not open-ended
resource authorization. The input's introductory tlim=400 template comment is
historical and does not supersede the numeric setting or finite allocation.

The frozen configuration warns at 80% memory and stops at 85%, monitors host
and GPU telemetry, and uses 400 blocks/rank with a 90% capacity proxy of 5,760
global leaf blocks. It has 10s sampling, 120s maximum stale telemetry, 900s
startup/diagnostic allowances, 2% filesystem reserve and zero automatic retries.
These are the recorded Vista decisions, not inherited AMD storage caps. Reusing
this setup for new work requires an explicit finite scope and current site
storage/purge checks; no new limits were agreed or applied in this task.

| Job / branch | Outcome |
|---|---|
| 1069961, gh-dev | Four-node compact gate, completed in 444s. |
| 1069970 / 20261013T200009Z | Wrapper exit 42 after 42s before native production initialization; explicit nlim override was missing. Dependent 1069971/72 cancelled unallocated. |
| 1071751 / 20261014_gh_retry_01 | Completed 16-node segment, 26,526s; t=0 to 36.55. Full-size native reference/restart gates passed. |
| 1071752 / same | Wrapper exit 42, 14,683s; telemetry_failure. Native chunks exited cleanly to a verified checkpoint at t=55.00625. No native crash/OOM found. Dependent 1071753 cancelled unallocated. |
| 1075832 / 20261016_gh_resume_01 | Completed 16-node segment, 84,075s; twelve clean native chunks, t=55.00625 to 221.953125. Six transient stale-file-handle errors recovered. |
| 1075833 / same | Cancelled before allocation, no third-segment evolution. |

The original telemetry failure did not retain its exact read exception. It is
not proven to be the stale-file-handle error seen later. The recovery reader
uses a fixed sixteen-path set, three read attempts 0.25s apart, bounded stale
samples and diagnostic logging while retaining memory guards. Native source,
input and executable stayed unchanged. Historical failures and superseded
configuration receipts remain preserved; this task performed no cancellation,
submission or retry on Vista.

Recorded GH usage for this known chain is 557.004444 node-hours including the
42-second failed job, plus 0.493333 gh-dev node-hours. These totals exclude GG
preparation/build/analysis and are not an SU invoice. The recovered segment's
peak GPU fraction was 0.6758692565 and host fraction 0.4778130941. This supports
the tested finite chunk policy; no unlimited memory-safety conclusion follows.

## Independent science checks and limits

The new [recheck](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/SCIENTIFIC_RECHECK.json)
reproduces strict common enclosure at 64.36875 using exact case/cycle/time joins
and live trackers. It finds 1,081 enclosure samples, 241.352585 degrees of
observed pre-common arc across ten valid intervals, and 0.625921 observed turns
after both individual horizons appear. Gaps are not extrapolated. The raw
accepted common row at 64.309375 and the endpoint principal angle are not
substitutes for these quantities.

It independently checks all 4,982,927 removal tags for finite values, integer
identity, uniqueness and the original five-million range, plus exact component
counts and rest-mass closure to the recorded VTK precision. All final 17,073
active tags belong to the envelope; no original clump tags remain active.
This checks reduced ledgers; the preserved Vista audit checks the native
survivor state. Low-lapse removal is not an independent horizon flux.

All 8,734 paired raw wave rows are finite with increasing time. The raw (2,2)
peak is at t=114.939062500144 and the last sample at 221.940625000533. No strain,
spectrum or ringdown analysis is added. The constraint and native-mesh numbers
in the report come from the preserved Vista audit: final H/M RMS
3.22846e-6/1.72857e-6 under the chi>=0.0625 proper-volume mask; finest dx=1/256;
R46 floor retained; relative-level counts
`448,448,376,904,1464,56,56,56,56,56,64` sum to 3,984. This is not convergence.

Accepted late common surfaces end at t=98.198437500083, mass 0.6369645064 and
coordinate-prescription spin 0.1746263436. Later candidates fail hrel<=0.2;
final hrel=0.5146317031. The timing near a restart motivates investigation,
but this evidence does not establish a cause. No final mass/spin or validated
ringdown can be reported. Binary curves end at strict common enclosure.

## Traditional session layout and reproducibility

Current PNG/PDF pairs are under [plots/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/plots/README.md); MP4s under
[movies/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/README.md); current tables under [reduced_data/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/reduced_data/README.md);
exact input/branch decisions under [inputs/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/inputs/vista_production/README.md)
and [config/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/config/README.md). The initial analysis is retained under
[archive/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/archive/vista_first_production_20261015/README.md), with its known AMR
label offset explicitly marked superseded. The original summary, including
frames, raster/particle caches, native text logs and source bundles, is intact.
Only final products and current top-level numerical arrays are duplicated.

[Promotion manifest](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/PROMOTION_MANIFEST.json)
records each source, destination, byte count and SHA256: 163 files,
451,070,796 bytes (430.1746 MiB), excluding newly generated overview/documents.
The preparation reports were copied into history before replacing root reports;
those manifest source paths consequently refer to the pre-consolidation bytes.
All copied destination bytes remain sealed. Historical report copies retain
original relative links; use the intact summary versions for working links.

Perseus Slurm 17024 performed full-summary hashing, copying, independent
reduction, overview rendering and complete decoding of all six promoted MP4s.
The [completion receipt](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/COMPLETE.json)
records software, source-manifest hash, job and times; provenance and logs sit
beside it. Slurm accounting storage is disabled on Perseus, so the saved
`scontrol` allocation and completion receipt provide the job record.
Final Slurm audit 17032 rehashed all 163 promoted destinations, checked 125
links in fourteen canonical documents, parsed the new Python/shell scripts,
confirmed unchanged native source, verified eight PNG/PDF pairs and reproduced
the independent measurements and overview PNG identically. See
[FINAL_AUDIT.json](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/FINAL_AUDIT.json).
The new overview and original all-figure/movie contact sheets were visually
inspected; [review receipt](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/VISUAL_REVIEW_PERSEUS.json)
records scope. No native raw simulation data were generated here. The original preparation
raw inventory of 180,575,528,723 bytes remains a historical retained snapshot;
this reconciliation did not clean, recalculate or overwrite those runs.

The new overview code is
[reconcile.py](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/code/scripts/particles/session010/vista_results_20261019/reconcile.py).
The original one-time promotion refuses to overwrite a completed release.
To reproduce only the independent check/overview, run inside a Slurm allocation
with NumPy/Matplotlib and a new unused output prefix:

```sh
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 code/scripts/particles/session010/vista_results_20261019/reconcile.py \
  --session "$PWD" --replot-to "$PWD/evidence/new_review/00_scientific_summary"
```

The [historical Vista scripts](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/scripts/vista_original_20261019/README.md)
retain allocation/architecture checks and absolute workspace assumptions. They
are evidence and a basis for reviewed reuse, not portable Perseus launchers.
The current NumPy/CSV tables can be used without running AthenaK. Full native
rerendering requires data still on Vista. New reports and reproduction code
are committed/pushed on project/GI-in-cluster; see the final Git receipt for
actual analysis/documentation HEAD. No merge, force push or native rebuild.

## Storage, retention and remaining work

The supplied summary is 8.681 GiB and is not a raw backup. The October 19
inventory records 15,164 raw files, 2,114,843,497,546 bytes total, of which
2,110,949,351,161 bytes are omitted bulk payloads. The analyzed twenty closed
folders contain 14,883 files / 1,712.737 GiB including twenty seal receipts;
the science-only SEALED count is 14,863 files / 1,712.735 GiB. These are
metadata plus historical seals, not a fresh full raw rehash.

Original raw root on Vista:

```text
/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/
```

Exact paths, sizes and inherited hashes are in the
[raw inventory](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/source_inventory/RAW_DATA_INVENTORY.jsonl).
There is no supplied proof of independent Anta raw archival. Current Vista
space/purge policy was not freshly queried from Perseus. A separate preservation
operation should check destination capacity/retention, archive by checksum and
retain originals until verified; no transfer or deletion is authorized by this
reconciliation. The original AMD caps do not apply to Vista SCRATCH.

Scientific follow-up should first investigate the late common-finder/restart
quality failure and establish any ringdown interpretation from appropriate
validated diagnostics. Longer production solely to wait for the main R40 burst
is not indicated by the saved coverage. A wider-start, low-eccentricity inspiral
would be a separate initial-data experiment with momenta/constraints reviewed;
simply separating these collapsing clumps is not a validated prescription.
Session009 remains closed, with its stops and archive disposition unchanged.
