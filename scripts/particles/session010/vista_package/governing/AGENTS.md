# NRPIC campaign archive: agent instructions

This directory is the authoritative Perseus-side archive and working record for the
NRPIC validation campaigns. Read this file before starting work, then read the named
entry document for the campaign you will touch. Do not infer current scientific status
from directory names or from an old top-level summary.

## Required session startup reading

Before starting or resuming any Claude or Codex session in this tree, read and follow
[`AGENT_LOGISTICS.md`](/data/jiaxiwu/NRPIC/AGENT_LOGISTICS.md) in full, then read the
campaign entry document. The logistics file is shared standing guidance for this
Perseus folder, `/data/jiaxiwu/NRPIC`. Apply it together with this file; the more
specific archive operating rules below take precedence over general logistics defaults
(in particular, heavy CPU work as well as GPU work belongs in Slurm jobs here).

## Directory map

- `reconstruction_ST/` — 2026-08-19 reconstruction of the NRPIC/ST branch, numerical
  revalidation against the historical campaign, and diagnosis/fix of the multi-MeshBlock
  particle-migration failure, then (session 03) the production port of the NaN guard and
  the fatal-motion check. Start with `SESSION_LOG.md`, then the three `REPORT.md` files.
- `gpu/` — focused two-A100 particle-cluster tests (25 and 1,000,000 particles), including
  exact inputs, raw VTK, rendering scripts, figures, and movies. Start with `README.md`.
- `homogeneous_cluster/` — the long homogeneous tangential-orbit/Einstein-cluster
  validation history across Anta, Perseus, and AMD. Start with `evidence/session_index.md`;
  `homogeneous_cluster_validation_session_summary.md` is historical and is not the final
  scientific status.
- `os/` — Oppenheimer-Snyder collapse, convergence/scaling, apparent-horizon, and joint
  AMR/no-AMR campaigns. Start with `OS_collapse_validation_session_summary.md`.
- `relativistic_cluster/` — historical Shapiro-Teukolsky relativistic-cluster validation,
  stability/collapse scans, reductions, figures, and AMD provenance. Start with `README.md`
  and `figures/FIGURES.md`.
- `Plummer-cluster/` — 2026-09-06 campaign asking whether replacing the homogeneous
  density profile with a **Plummer** profile changes the internal `l=1` (dipole) growth
  seen in the homogeneous tangential-orbit Einstein cluster. All-circular-geodesic cluster
  with the static-observer energy density Plummer in areal radius, `b=20M`, hard cutoff
  `r_t=400M`, `N=2,113,536` co-located `+/-u_i` pairs, seven fixed refinement levels to
  `dx=1M`, BSSN `use_z4c=false`. Branch `project/Plummer-cluster` from `origin/NRPIC`
  @ `ca0f5298`. Start with `README.md`, then `SESSION_LOG.md`, then
  `session_01_initial_data_and_production_20260906/reports/REPORT_Plummer_session1.md` and
  `session_01_initial_data_and_production_20260906/reports/REPORT_AGENT_Plummer_session1.md`.
  All session artifacts now live in that session directory; `code/` remains shared source.
  The originating prompt and the theoretical derivation PDF are in
  `session_01_initial_data_and_production_20260906/` (moved there from the former
  top-level `Plummer/` staging folder on 2026-09-07 and checksum-verified; that folder no
  longer exists).
  New code: `src/pgen/particles/nr_pic_plummer.cpp` and `plummer_profile.hpp`. This
  campaign also fixed an upstream bug that faults every GPU backend — the `trk` output
  atomically incremented a host-stack `int` from a device kernel
  (`src/outputs/track_prtcl.cpp`); any campaign that requests `file_type = trk` needs
  that fix.
- `ST_migration/` — 2026-08-23 campaign asking whether an unstable monoenergetic
  Shapiro-Teukolsky cluster can be pushed outward, at fixed per-particle energy, onto the
  stable branch instead of collapsing. Branch `project/ST-migration` from
  `project/ST-cluster` @ `b51efb27`. Start with `SESSION_LOG.md`, then
  `session_01_perturbation_design_and_scouts/REPORT.md`.

## Non-negotiable operating rules

1. Do not run simulations, builds, bulk reductions, full-data plotting, or movie rendering
   on any login node. Use Slurm jobs. Login nodes are for editing, git, `sbatch`, short
   inspections, and lightweight metadata checks only.
2. Preserve provenance: record host, job ID, git commit, submodule state, exact input,
   module/toolchain versions, build flags, executable hash, launch command, and output
   manifest. Never silently replace a historical run or its evidence.
3. Keep raw and reduced data, long logs, Python analysis, plotting, and movie generation on
   the cluster. Produce Markdown reports, exact inputs, final figures/PDFs, and final movies
   on the cluster too; those review artifacts are then pulled to the Mac as a one-way thin
   mirror. Do not use an unrestricted bidirectional sync.
4. Before deleting data anywhere, make and retain an inventory, confirm the relevant report
   and reproduction inputs exist, and verify the destination copy by checksum. Cluster
   storage is not automatically a backup; respect the filesystem's retention policy.
5. Many preserved scripts are historical evidence, not portable launchers. At least 16
   scripts contain hard-coded `/Users/jiaxiwu/...` paths, and many Slurm scripts hard-code
   old `/data`, `/data2`, or AMD `/work1` roots. Review and parameterize paths before reuse.
   The archived `os/.venv` is a macOS environment and must not be used on Linux; recreate
   dependencies and use headless plotting (for example `MPLBACKEND=Agg`).
6. If an AMD run faces an excessive queue delay, consider running it on Perseus instead or
   using both clusters, including a checksum-verified restart transfer between them when
   appropriate. Preserve the machine-specific build, executable, input and restart
   provenance for each segment. Whenever a run generates raw data on Perseus, report the
   exact path, size and retention or cleanup status so its storage can be handled explicitly.
7. If a Claude or Codex session reaches its token limit, ensure that the session will
   automatically resume after the token allowance resets. Before the interruption, persist
   enough state to continue safely, including running job IDs, completed work, the next
   action and any unresolved decisions; do not leave continuation dependent on a manual
   re-prompt.

## AMD HPC Fund essentials

- Host/account: `hpcfund.amd.com`, allocation/account `eliasmost`.
- Main production partition: `mi2104x`, one exclusive node with 4 x AMD MI210 GPUs;
  typically four MPI ranks, one rank per GPU.
- Known-good toolchain: `gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3` plus
  `rocm/6.4.1`. Do not substitute the system/default ROCm 7.x for these preserved runs:
  ROCm 7.2 built successfully but faulted at runtime in particle stress-energy deposition.
- Reconstructed ST build (`reconstruction_ST`) uses:

  ```sh
  cmake -S "$repo" -B "$build" \
    -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_AMD_GFX90A=On \
    -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc \
    -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 \
    -DPROBLEM=particles/relativistic_cluster
  cmake --build "$build" -j 32
  ```

  Historical `relativistic_cluster/` builds used `-DPROBLEM=relativistic_cluster` before
  the pgen moved under `src/pgen/particles/`; choose the value that matches the checked-out
  commit. The exact authoritative scripts are stored in each campaign.
- AMD work root is `/work1/eliasmost/jiaxiwu`. The effective user budget recorded by the
  campaigns is 1.9 TiB even when filesystem-wide `df` shows much more. Preserve the
  five-minute `du` watchdog convention: warn at 1.5 TiB, cancel/stop growth at 1.7 TiB or
  when projected usage would cross 1.9 TiB. A transient `du` failure must not kill the
  watchdog subshell. Check the whole user root, not only the current campaign.
- The stable launch recipes disable OpenMPI vader single-copy where recorded:
  `OMPI_MCA_btl_vader_single_copy_mechanism=none`. Do not remove this without a new gate.

## Perseus and Anta notes

- This archive lives at `/data/jiaxiwu/NRPIC` on Perseus. At migration time (2026-08-19),
  `/data` was 93% used with about 8.2 TiB free; recheck before every large campaign.
- Perseus homogeneous-cluster work used NVIDIA A100 GPUs and a CUDA-aware OpenMPI/UCX
  stack; exact build/launch recipes are preserved under `homogeneous_cluster/evidence/`.
  Anta and Perseus paths and module stacks are not interchangeable.
- Prefer a new dated run root for new work. Never point a job at this archive root in a way
  that can overwrite evidence. Reduce raw output promptly, but retain exact inputs,
  manifests, source/commit identifiers, logs needed for diagnosis, and the data needed to
  reproduce every promoted figure.

## Git repository layout (2026-08-29)

`/data/jiaxiwu` is a storage root, **not** an AthenaK checkout. Do not rely on parent-directory
Git discovery from the archive. The old `/data/jiaxiwu/.git` common repository was retired
after every active source tree was replaced by an independent clone from
`git@github.com:JiaxiWu1018/athenak.git` at the same path and exact commit. Each clone has its
own `.git/` object store and its own Kokkos submodule metadata.

The authoritative inventory is
`data_handling/migration/git_root_relocation_20260829/SOURCE_REPOSITORIES.tsv`. Consult it
before classifying source snapshots, refreshing the Mac mirror, or changing a campaign's
code. It distinguishes the seven active independent clones from two historical repositories
under `data_handling/migration/perseus_home_cleanup_20260819/`; the historical repositories
are preservation evidence and must not be normalized, cleaned, or replaced.

All listed source repositories remain source-snapshot class under rule 3 and stay on
Perseus. The retired root repository, including divergent legacy branches, its stash, and all
formerly dangling objects, is preserved in the checksum-verified bundle recorded by
`data_handling/migration/git_root_relocation_20260829/RECORD.md`.

## Migration record (2026-08-19)

The five complete local test trees under
`/Users/jiaxiwu/Desktop/Research/NRPIC` were archived on Perseus with platform-neutral
names:

| Local source | Perseus name |
|---|---|
| `AMD_reconstruction_ST_test` | `reconstruction_ST` |
| `anta_gpu_test` | `gpu` |
| `anta_homogeneous_cluster_test` | `homogeneous_cluster` |
| `anta_os_test` | `os` |
| `perseus_relativistic_cluster_test` | `relativistic_cluster` |

The migration manifest and verification records are under `migration/`. The local
`homogeneous_cluster` source was still active during that migration, so its final
synchronization and local thinning were deferred; both were completed on 2026-08-20 and are
recorded in the last section of `migration/MIGRATION_20260819.md`. All five local test trees
are now thin review mirrors, verified by checksum against this archive, and every local tree
may be kept only in that form. The Mac holds Markdown/HTML reports, exact inputs, final
figures/PDFs, and final movies; raw and reduced data, logs, scripts, source snapshots, and
movie frames exist only here.

Also on 2026-08-19, obsolete NRPIC-related Perseus home paths (`worktrees/`, `opt/`,
`code/`, `waveforms/`, and loose homogeneous-cluster logs/sbatch files) were removed
after exact checksum-verified preservation under
`migration/perseus_home_cleanup_20260819/`. One preserved worktree contains extensive
uncommitted historical changes. Do not assume the old home paths exist; restore or
rebuild from the archived copy into a new dated working location.
