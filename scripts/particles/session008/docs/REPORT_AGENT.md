# Session 008 implementation and operations record

**2026-10-05 UTC:** numerical gates passed and t=9.325 is saved; reviewed continuation to the unchanged t=12 endpoint is queued. See `HANDOFF.md` for exact controls.

## October 5: reviewed failure and authorized continuation

The completed GPU gate 447655 and production jobs 447657/447659 reached verified **t=9.325**, cycle 2252. There are **3,998,466 surviving particles** and **1,001,534 lapse removals**, with the particle ledger reporting conservation OK. Both separate horizon consumers publish accepted, associated surfaces at the last inspected time; no accepted common surface was present in the inspected output. Orbital and waveform analysis is still pending.

Job 447661 failed during MPI initialization, before checkpoint loading or simulation steps. Its log reports `ob1` on k003-010 and `ucx` on k003-001, followed by an unreachable MPI peer. This identifies incompatible messaging choices at startup; it does not demonstrate a broken physical network or a numerical failure. The user explicitly requested continuation on October 5. Earlier failure logs/state and automatic archive-stop flags were preserved before resuming; human/resource stop intent is never cleared automatically.

The original full-particle gate passed initialization and restart checks. Peak measured VRAM was **70.37%**, below the 85% gate. Total initial covariant momentum residual was `[0, -3.700000000030068e-7, 0]`. The lightweight Session 007 comparison matched t=0.0125 and the documented regions/normalization. The retained real checkpoint has a **51,865-byte parameter header**, greater than the former 40 KiB limit, SHA-256 `981af0bf7deb074b9a0ab2f9fc83cd547baa2b7cdad0cd265d0d9337b1a4be6a`, and size 26,227,169,973 bytes. Recovery preflight rechecks its entire digest on allocated compute.

All C++ physics, compiled source, input, executable and originally frozen scripts remain unchanged. New operations are isolated under `scripts/recovery_20261005/`, with independent frozen hashes in AMD `control/recovery_20261005.json`. The six eligible GPU nodes all ran this configuration successfully before: k005-002/003/004/005/007 and k003-007. This is a bounded recovery recommendation, not a guarantee that future node combinations cannot fail.

| Recovery role | Submitted AMD job | Maximum allocation |
|---|---:|---|
| MPI/mesh and checkpoint preflight | 451757 | 3 nodes, 10 minutes |
| Continuation 1 | 451758 | 3 nodes, 4 hours |
| Inspector 1 | 451759 | 1 node, 30 minutes |
| Conditional continuation 2 | 451760 | 3 nodes, 4 hours |
| Inspector 2 | 451761 | 1 node, 30 minutes |

The five jobs were verified queued with the intended dependencies. Continuation 2 is canceled when the endpoint is reached in continuation 1; any failure stops for review. The physical endpoint remains **t=12**, the total budget **48 raw AMD node-hours**. Historical actual usage is **21.645833 node-hours**; additional maximum exposure is **25.5**, making the worst permitted total **47.145833**. Queue waiting consumes no node-hours. Node-hours count elapsed allocation seconds times allocated nodes / 3600, including build, gate, inspectors and failed startup; this is not GPU-hours or a monetary charge.

Recovery verification used Perseus Slurm **10957** (15 original checks plus 6 recovery checks) and **10958** (7 extended recovery checks), including accounting lag, duplicate/finite submission, preserved stop intent, endpoint cancellation and failure handling. Syntax passed. Logs/recipes are in `evidence/recovery_checks*`. No bulk work was run on login nodes and no new raw science or checkpoint was copied to Perseus.

Anta's previous trigger failed when `squeue -j` no longer recognized a completed job. The separate corrected trigger queries all active user jobs, then accounting for retired jobs. It has a new authorized 48-hour window, retaining the original cumulative four-job limit. Original job 2134 and restored archive job **2329** have completed. Job 2329 finished 0:0 in 888 seconds and verified both production segments before removing only approved AMD science binaries. Independent metadata heartbeat and checkpoint retention remain in place. Final analysis and visual review follow completion or another terminal stop; no full orbit or distant merger waveform is claimed at t<=12.

At **2026-10-05 07:52 UTC**, separate allocated-byte inventories measured:

| Storage scope | Bytes | GiB |
|---|---:|---:|
| AMD Session 008 | 82,172,391,424 | 76.53 |
| Entire AMD user root | 169,274,462,208 | 157.65 |
| Anta Session 008 on /data3 | 99,611,504,640 | 92.77 |

AMD usage fell after verified archival; the latest three checkpoint files remain on AMD. Whole-user usage includes unrelated campaigns and is measured separately to avoid nested `du` deduplication. Anta /data3 still has about **27.01 TiB available**. These are timestamped measurements, not promised final sizes. Campaign cap 1.25 TiB and whole-user 1.5/1.7/1.9 TiB watchdog thresholds are unchanged.

Exact controls, paths and remaining work are in `HANDOFF.md`; the reviewed execution bounds are in `RECOVERY_PLAN_20261005.md`. The October 2 sections below record the original preparation/submission state and are retained as dated history.

## October 2 preparation: approval and source

The approved bounded assessment is recorded in `APPROVED_PLAN.md`. Governing logistics and Sessions 004–007 were read, including Session 007's October 1 execution/plotting follow-up and corrected analysis. The pre-edit stream inventory read 1,094 Markdown files, 20,336,235 bytes; per-file hashes/headings are in `evidence/MARKDOWN_INVENTORY.json`.

An independent clone was created here on `project/GI-in-cluster`. The historical preparation checkout at `36b64e23` was preserved. Restart repair `263dcf21a7f6ce6ab1ccd8f1dd595ae0ca79c8c1` was recovered from the preserved Anta bundle, checked and fast-forwarded. Its source change raises the bounded restart parameter-header limit from 40 KiB to 1 MiB. Bundle SHA-256: `3f0e220d56e4504062da6a396b318836aa20493b645d1c2bffac684d2e14fe27`.

| Frozen item | Revision or SHA-256 |
|---|---|
| Actual compiled source | `0c0a5a9bd47fa10ed8732ea12f1a53cd591b3f69` |
| Kokkos, version 4.7.02 | `6739bc623081648af9e752b616d9671527922cbf` |
| Canonical input | `43ff1b21049bf59bfa6f565ebdb39d1333323627e496b0b97fa6b0e093bdf33a` |
| AMD executable | `cc7faaaadc8eb0eb74afdd3de81a748fb6adf9b069361496a8011ec86a7e5c2f` |
| Operations revision/push result | `evidence/OPERATIONS_REVISION.txt` |

The compiled source remains at the physics/input commit; later commits contain operations, analysis and documentation. Bindings check source, input, executable and every staged script. Pre-start script updates preserved earlier configurations before replacing hashes while gate 447655 was held. It was released after those updates. No C++ rebuild was needed.

The existing pgen reads `gi_clump1_bulk_vy`/`gi_clump2_bulk_vy`, applies local orthonormal Lorentz boosts, stores covariant `u_i=psi^2*u_hat_i`, and uses boost-dependent mean Lorentz factors for rest weights. Analytic conformal geometry depends on the fixed source profile. Geometry is initialized afresh and weights/momenta regenerated. No new particle-pusher force was added.

## AMD build and runtime

Build **447620** completed `0:0` in 313 seconds on one 16-CPU devel node: **0.08694 raw node-hours**. An initial 32-CPU request was rejected before submission because current devel nodes have 16 CPUs. Unrelated ST-migration job 447610 was left alone.

Modules: `gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1`. CMake enables HIP/GFX90A/MPI and `-O3`, with `PROBLEM=particles/gi_cluster`, `cc` and `hipcc`; build parallelism is 16. Exact commands are in `scripts/amd_build.sbatch`; actual versions, source/submodule state and flags are in AMD `evidence/build_provenance.txt`, `build/CMakeCache.txt` and `logs/configure.log`/`make.log`. Metadata copies are under this session's `evidence/amd_snapshot/`.

GPU jobs use `eliasmost`/`mi2104x`, three exclusive nodes, twelve ranks and one GPU per rank. Launch uses `prun` and `--kokkos-map-device-id-by=mpi_rank`, automatic MPI transport, and `OMPI_MCA_btl_vader_single_copy_mechanism=none`. Actual working directories are per-run, including waveform side files. Rank wrappers record all twelve GPUs' memory.

## Tests and pending numerical gates

Local checks ran through Perseus Slurm with the mandatory GPU request. Jobs 10302/10303/10308 passed progressively extended tests. Final **10327 completed `0:0` in six seconds**, passing **15 tests** plus Python/shell syntax:

- Twelve controller/checkpoint checks cover native header layout/truncation, latest-three rotation and failed-write protection, finite dependencies, duplicate submission/crash recovery, hash binding, archive heartbeat, endpoint cancellation, sticky stop, scoped cancel and numerical-failure classification.
- Three analysis checks cover circular-motion derivatives/small denominators, no derivative across diagnostic gaps, and a synthetic fixture verifying accepted/rejected horizon rows, both (2,+/-2) columns, plot generation and movie encoder/decode execution.

Logs are `evidence/script_checks.JOBID.log`; the final recipe is `evidence/final_script_checks.sbatch`. Synthetic fixtures were temporary and removed. These checks do not establish GPU numerical correctness or science-quality rendering.

The queued AMD gate repeats the twelve operations tests, then runs:

1. Actual twelve-rank mesh inventory from the canonical deck.
2. Full five-million-particle fresh initialization and two cycles: counts/tags, finite positive weights, centers/widths, inverse-boosted thermal spread, boost signs, positive J_z, compiled/written ledger agreement and total momentum residuals.
3. One-cycle checkpoint plus restart to two cycles, compared with uninterrupted evolution; identity and live-diagnostic reacquisition gaps are checked.
4. A real restart/clean-stop checkpoint and continuation to five cycles; required outputs and four complex ell=2–8 waveform streams are checked.
5. All twelve GPUs' memory in every full case, requiring peak VRAM below 85%, plus checkpoint census/exact size/header/hash verification.

Each numerical case has a twenty-minute application bound inside the two-hour gate allocation; gate storage is bounded at 256 GiB. Actual mesh inventory, peak memory and initialization/restart results are **not measured yet**. Strict accepted-horizon/identity tests from Session 007 jobs 6531/6533 are reused evidence; no new accepted-AH AMD fixture is claimed. AHs remain measurement-only with strict association.

The lightweight Session 007 constraint comparison requires matching time, rectangular regions, coordinate volumes and identical lack of masks before quoting ratios. `con_M` is already squared; its saved-field RMS differs from momentum-vector RMS derived from Mx/My/Mz. Evolution history norms use proper volume and `chi>=0.0625`, not an accepted-horizon exterior mask.

## Submitted finite chain

| Role | Job IDs | Allocation/bound | Current status |
|---|---|---|---|
| Build | 447620 | one node, 0.5 h | completed |
| Full gate | 447655 | three nodes, 2 h | pending resources |
| Inspectors | 447656 / 447658 / 447660 / 447662 | one node, 0.5 h each | pending dependencies |
| Evolution | 447657 / 447659 / 447661 | three nodes, 4 h each | pending dependencies |

Maximum reserved exposure is **44.5 raw AMD node-hours**, within **48** including preparation. Target **t=12**, at most three evolution segments. Each application stops at 3 h 20 min, reserving forty minutes for finalization; stop polling is every sixteen cycles. No automatic retries, velocity changes or t=50 extension.

Atomic locked state, campaign-unique names and accounting lookup protect submission/continuation. Inspectors run after each predecessor and release the next segment only after successful gates, verified increasing physical time and a valid checkpoint. Numerical/configuration/scheduler failures, absent progress, stop intent or exhausted caps halt for review.

## Storage and durable archive

AMD: `/work1/eliasmost/jiaxiwu/gi_s008_amd_20261002/`. Anta: `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/`.

At inspection Anta `/data2` had about 1.69 TiB ordinarily available. `/data3` had **27.10 TiB available** on an 87.31 TiB filesystem, a writable empty `/data3/jiaxiwu`, and no reported user quota. The 1.7 TiB figure was available `/data2` space, not Anta's total capacity. User direction changed the science destination to `/data3`.

AMD retains the latest **three verified Session 008 checkpoints total**, including gate checkpoints, without a Perseus backup. A transient fourth write is allowed. Native ABI, particle census, exact size, stable state and SHA-256 are checked before rotation; inventories are retained. Failed writes cannot replace verified checkpoints.

Five-minute whole-user watchdog: warn at 1.5 TiB, stop at 1.7 TiB/projected 1.9 TiB; tolerate transient `du` failures. Campaign cap 1.25 TiB with 64 GiB checkpoint reserve. Anta cap 1 TiB with 512 GiB free reserve. Completed segments, or a 256 GiB accumulation stop, bound archival opportunities.

Anta-to-AMD SSH works; AMD-to-Anta networking is blocked. Anta pulls sealed outputs, verifies against compute-generated SHA-256 manifests, then deletes only approved AMD `.bin`/`.vtk`/`.cbin` copies. Logs, inputs, manifests, complex waveforms and failure evidence stay; checkpoints use separate rotation. No production raw data or checkpoint is placed on Perseus.

Anta rejected a CPU-only request before creating a job. Its supported cron service now has one tagged **metadata-only** trigger every five minutes, expiring after 48 hours. It submits at most four six-hour archival/analysis jobs with one mandatory GPU/two CPUs/16 GiB only when work exists. Independent heartbeat updates were observed after the submitting shell exited. The rejected polling launcher is preserved as superseded evidence. Stale archive availability stops production safely.

## Reproducible analysis and outstanding work

Automatic archival runs these commands **inside Anta Slurm**, with `MPLBACKEND=Agg`, one OMP/OpenBLAS thread:

```sh
python3 scripts/anta_archive.py --once
/home/jiaxiwu/miniconda3/bin/python scripts/analyze_s8.py --root /data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002
```

The wrapper uses existing NumPy/Matplotlib and installs binary `imageio-ffmpeg==0.6.0` into session `python_deps` on allocated compute. Verified segments are assembled without duplicated validation trajectories. Phase/derivatives are confined to contiguous live intervals; unknown gap turns are excluded. Products cover absolute/envelope-relative trajectories, radial/tangential motion, accepted horizons, particle/matter J, constraints, raw (2,+/-2) at multiple radii, and central/context/density movies.

Particle movies use a fixed tagged cohort. Density uses all particle rest weights in fixed 256² Cartesian bins within `|z|<0.7`, divided by coordinate area; this is coordinate surface density, not proper energy density. AH circles illustrate accepted `rmin`. Encoder/decode checks are automatic; representative real plot/frame visual QA remains pending.

The t=12 stage cannot cover central-collapse waves at r=40–70 or a full initial orbit. Raw r*Psi4 remains primary; strain, merger/ringdown and precision radiation interpretation are deferred. No outcome is fabricated. Anta cannot authenticate back to Perseus; reports are generated there and pulled using `scripts/collect_review.sh`.

Remaining work: actual GPU gates, any accepted evolution, measured resources/accounting/checkpoint history, terminal analysis and visual review. The bounded workflow runs independently of the interactive agent. No Session 008 raw science has been created on Perseus.
