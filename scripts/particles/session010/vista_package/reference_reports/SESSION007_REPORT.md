# Session 007 Part 1: Vista production handoff

> **Status reconciliation (2026-10-01):** The original sections through Paths
> describe the September 8 Part 1 handoff as it stood then. Vista later ran the
> two-clump evolution through t/M = 64. Execution, plotting, storage, and
> evidence limits are documented in the follow-up section at the end.

## Result

The two-clump counterclockwise configuration is frozen in one canonical five-million-
particle input and a pushed source commit. The package is relocatable, hash-verifiable,
and contains a finite Slurm workflow that can outlive the later 90-minute Vista agent.
No Vista transfer or job was performed, and no detached submitter was installed.

The 2026-09-08 user-directed revision moves the complete Vista workspace to `$SCRATCH`,
adds high-cadence three-plane and full-3-D field/matter/Weyl output plus multipolar GW
extraction, and replaces the old fixed 32-node example with a provisional 16-node value
that cannot be finalized until the packaged H200 capacity ladder runs on Vista.

The simulation places a 0.76 source-parameter Einstein--Vlasov envelope of areal radius
30 around two 0.12 Gaussian clumps at x=+-3. Each clump contains one million particles;
the envelope contains three million. The left/right orthonormal boosts are respectively
-/+0.0880124692901454 in y, giving counterclockwise orbital angular momentum. This is the
envelope-only circular-geodesic prescription (`Omega=0.0244412311298025`, period
257.073192173129), not exact binary equilibrium.

The recommended run uses RK4/CFL 0.4, domain `[-256,256]^3`, root spacing 2, `32^3`
MeshBlocks, natural Lohner refinement without a tracker floor, and finest spacing 1/256.
Root-inclusive `num_levels=10` maps physical levels 0--9 to logical tree levels 3--12.
The finite endpoint is the Part-1 recommendation `t=64`, about 0.249 orbit. Lapse removal
at alpha<0.05 remains on; apparent-horizon removal is off.

Root dx=2 remains deliberate for the wave zone: the leading quadrupolar wavelength is
about 128.5M, so it has about 64 root zones per wavelength and about 20 radial zones to
the nearest r=40 extraction sphere; dx=4 would halve both and requires a separate
convergence design while preserving the 1/256 central ceiling.

## Implementation

Production commit `36b64e23867358e52f92ba96f9bcb4eb5e6839e6` on
`origin/project/GI-in-cluster` adds backward-compatible, opt-in operational/diagnostic
support: three-component removal labels; restart persistence of two tracker identities and
non-authoritative full-shape horizon warm starts; rejection of an individual surface that
encloses the other tracker; and a rank-broadcast clean stop-file path. No problem-generator
or evolution equation was changed.

The final follow-up commit also null-initializes the two `MeshBlockPack` owning pointers
that are not allocated during `-m`, making normal mesh-inventory cleanup safe. This changes
no initialized simulation path. Inventory output is isolated in each job's evidence root.

The two individual strict finders are independent; finder 2 is a distinct later common-
horizon search. A restored surface cannot publish a horizon until a live find passes all
current gates. Optional common-horizon early stopping is disabled by default and cannot be
inferred from center separation.

## Verification

The final-source CUDA build passed from the clean commit on Perseus A100 job 6511;
executable SHA-256 is
`fcb293fffdf94812a947dbcba61866c5ab0a755d178ee0e110b93bb36e51c8bd`.
Seventeen controller tests passed locally and in independent Slurm job 6544. The final
130-payload-file relocation audit passed bundle reconstruction, all script syntax, pinned
hashes, the 3,072-node-hour workflow dry run, orbit recomputation, portable tests,
forbidden runtime-path scanning, dependency closure, symlink checks, and Git-LFS checks
in a fresh temporary directory.

Exact final-source job 6533 passed every simulation leg: four-rank uninterrupted and
split runs, a four-to-three-rank restart at capacity 460/rank, a three-rank clean-stop
restart, and the original bounded unboosted case. Time-matched two-step control job 6541
and reduction job 6542 then passed. All 12 initialization assertions hold for exactly
5,000,000 particles and the three immutable tag ranges. At `t=0.0125`, boosted/control
global constraint RMS ratios are 1.00169 for H, 1.01734 for M, and 1.02477 for My; this
quantifies the known local-momentum-constraint penalty rather than claiming it is solved.
Restart particles agree exactly, while single-precision field output agrees within
`rtol=1e-5, atol=1e-8`. Peak A100 memory was 65,477 MiB with four ranks and
75,787 MiB with three, which is why the Vista gate keeps the eight-to-four-rank layout.

Static accepted-horizon fixture job 6531 also passed: 200 uninterrupted accepted rows,
100 pre-restart rows, serialized surface keys present, and 100 live reaccepted rows after
restart. Its scope is surface/warm-start persistence; it is not a moving-puncture or
production-collapse equivalence test. All raw validation trees are retained.

## What Vista still must prove

Vista must fill the real allocation/modules/absolute scratch paths, obey current placement
policy, build for Grace/H200, run the four-rank capacity ladder, select a tested-safe
MeshBlocks/rank and node count, smoke-test `ibrun` and Slurm dependency/accounting, and run
the packaged exact eight-node quick test. It must measure all-node H200 memory, startup and
step rate, checkpoint/output duration/size, and purge/transfer headroom before production
is released. Any deferred gate remains a gate, not a caveat relabeled as passed.

The inherited physics estimate is 1,280--3,072 H200 node-hours; expanded GW/full-volume
I/O widens planning to roughly 1,500--4,600 node-hours.  The provisional example cap is
3,072 node-hours (16 x 48 x four slots), with zero numerical retries and at most one
infrastructure retry; measured timing may select 24 or 32 nodes while the original 4,608
node-hour authorization ceiling remains binding.  Measured startup file sizes and
the expanded cadence project about 2.8--3.2 TiB before AMR growth; broad runaway bounds
warn/stop at 8/12 TiB with 1 TiB reserved for a final checkpoint.  These are deliberately
labeled estimates; Vista must replace them with measured H200 rate, AMR growth, output
time/size, and all-node memory before release.

## Handoff

Read `vista_handoff/README_FIRST.md`, `VISTA_RUNBOOK.md`, `RESOURCE_SELECTION.md`, and
`VISTA_AGENT_PROMPT.md`. The later agent has exactly
5,400 seconds, begins closeout by minute 75, and exits by minute 90. It submits or prepares
one finite dependency chain after the separate capacity probe; verify/build/quick-test receipts gate production, inspectors
release only valid checkpoint continuations, and terminal states suppress successors.

Scientific limitations remain: unsolved local momentum constraints with `K_ij=0`,
approximate combined initial data, unverified pre-horizon lapse removal, coordinate-based
horizon momentum/spin, boundary/convergence uncertainty, and no guarantee of merger.

## Paths

- Session: `/data/jiaxiwu/NRPIC/GI_in_cluster/session_007_part1_vista_handoff_20260908`
- Handoff: `/data/jiaxiwu/NRPIC/GI_in_cluster/session_007_part1_vista_handoff_20260908/vista_handoff`
- Archive: `/data/jiaxiwu/NRPIC/GI_in_cluster/session_007_part1_vista_handoff_20260908/vista_handoff.tar.zst`
  (46,776,845 bytes)
- External checksum:
  `3c680954e01cbd1e6684d717bd0f88393ed4d0aaa7fc8322d505b7df8893f1bd`

## Vista execution and plotting follow-up (reconciled 2026-10-01)

This section uses later records retained in the Anta Session 7 plotting bundle
and the Perseus paper-figure record. The Part 1 statements above that no Vista
job had run and that Vista gates were pending are historical September 8
statements. This reconciliation did not rerun the simulation or inspect Vista
scratch directly.

### Probe, production, and recovery

- Vista Part 2 reached an allocated gg compute node on September 8, but its
  first closeout recorded no accepted verify/build/capacity job. The first
  login-node submission failed because the Slurm-spooled script resolved
  helpers under /var/spool/slurmd. The next build exposed a Kokkos 4.7.2
  ARM-SVE CMake include error. Both were repaired with preserved evidence.
  A further capacity attempt hit UCX CUDA DMA-BUF registration in MPI_Waitall;
  UCX_IB_GPU_DIRECT_RDMA=no provided the documented host-staged transport.
  The completed four-rank capacity ladder (jobs 984789 and 984790) tested
  360, 400, 440, 480, and 520 blocks/rank and selected 440 blocks/rank with
  a 16-node memory candidate.
- The first finite production chain was jobs 984833-984843. Verify 984833,
  build 984834, and exact quick test 984835 succeeded. Its 16-node segment
  984836 advanced to cycle 5888 and t/M = 14.98125, then cleanly stopped:
  the monitor looked in the output directory for tracker/horizon side files
  that AthenaK had written in its handoff working directory. Its
  27,945,207,725-byte checkpoint has SHA-256
  773d6dbbc52cc6a50365ffcae38913ac5bf3a864be0f2d5129df801f9bcc0984.
  Inspector 984837 blocked the rest of that chain. The working-directory
  and result-path handling were repaired without discarding the checkpoint.
- The first restart chain (986036-986046) passed verify/build/quick-test, but
  production 986039 advanced no cycle: the checkpoint parameter end marker
  was at byte 53,207, beyond AthenaK's historical 40-KiB reader guard.
  The documented restart-header repair is source commit
  263dcf21a7f6ce6ab1ccd8f1dd595ae0ca79c8c1, raising the guard to 1 MiB
  with a mandatory real-checkpoint restart gate. The retained continuation
  history on Anta subsequently ends at t/M = 64.00000. The plotting bundle
  has no final Vista scheduler closeout or continuation result.json, so this
  endpoint is supported by retained output, not a final Slurm receipt.

Accepted individual horizons first appear near t/M = 8.11-8.18; the first
accepted common horizon is at t/M = 49.35. The four paper particle views use
t/M = 0, 8.25, 49.50, and 64.00. These are horizon/output milestones, not a
claim that the approximate binary initial data are equilibrium or that the
Part 1 numerical limitations were removed.

### Data, plots, and movies

- The original Vista segment is documented under
  /scratch/10222/jiaxiwu1018/nrpic_s7_runs/s7_two_clump_ccw_20260909T0617Z/.
  The later restart-fix campaign is s7_ccw_fix_20260910T1600Z. Vista scratch
  is purge-eligible; its current contents were not checked because TACC login
  required interactive authentication.
- The accessible plotting-data copy is
  anta:/data2/jiaxiwu/NRPIC_test/s7_plotting_bundle_20260911/.
  Its data/production_segment_001 and data/continuation_segment_001 held
  11,125 ordinary files totalling 940,199,318,395 bytes on 2026-10-01.
  They include particle VTK and field output, history, horizon, removal, and
  waveform products through t/M = 64. The bundle deliberately omits rst/
  checkpoints, estimated in its README to save about 2.7 TB. It is a
  plotting-data copy, not a restart archive.
- The Anta plotting entry point is
  analysis/vista_handoff/analysis_support/render_s7_interim_visuals.py.
  The artifacts/s7_interim_visuals_20260911/ directory contains
  s7_interim_diagnostics.{png,pdf}, s7_matter_evolution_t0_t64.mp4, and
  s7_constraint_amr_t0_t64.mp4, plus render manifests and caches. A separate
  t/M = 0-15 progress movie and its checksum are documented in
  analysis/vista_handoff/evidence/vista_progress_movie_20260909.md; its
  historical Vista home path was not verified as present now.
- The Anta cca_talk_20260912/ directory retains the final talk media at
  revision_20260912k/slide27_jeans_3d_plus_tracker.mp4,
  revision_20260912g/slide27_jeans_particles_3d.mp4,
  revision_20260912c/slide27_two_clump_collapse.mp4, and
  static/slide27_ah_mass_spin.png. The first three movies match the
  SHA-256 hashes of the convenient top-level copies in the Mac-only
  legacy/cca_talk_20260912/movies/ review package, outside the
  NRPIC thin mirror.
- The September 29 corrected paper figures are at
  /data/jiaxiwu/NRPIC/paper/figures/jeans/:
  jeans_particles_3d_sequence.{pdf,png} and
  jeans_black_hole_mass_spin.png. The adjacent paper/README.md gives the
  render provenance. These paper-facing figures are also in the local mirror.

Sources: the Anta bundle README, its evidence/vista_*.md records, the
continuation z4c.user.hst endpoint, the Mac-only CCA README and SHA256SUMS,
and /data/jiaxiwu/NRPIC/paper/README.md. The current Vista queue, scratch
retention, and any final on-Vista closeout remain unverified.
