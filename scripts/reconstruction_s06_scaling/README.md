# Session6 paired-sampling and fixed-grid particle scaling

Approved three fresh0..1000M evolutions at y_c=.819: mirror-paired3M,
ordinary randomized-stratified3M and6M, seed1. Reuse the completed ordinary1.5M
baseline. Pair each exact original x/u with -x/-u and halve both masses. The
native STSAMP05 header remains compatible with the protected loader; the sample
receipt identifies antipodal construction and the exact parent/partner mapping.
No re-symmetrization during evolution, solver rebuild or changed mesh/physics.

Perseus follow-up root:
/data/jiaxiwu/NRPIC/reconstruction_ST/06_yc0819_long_stability_20261016/followups/paired_scaling_20261017
AMD runtime root:
/work1/eliasmost/jiaxiwu/st_reconstruction_s06_paired_scaling_20261017

## OS-first execution

All ST allocations, including staging and preflight, wait. The metadata-only
controller reads the OS canonical45-row ledger, verified80-epoch/40-case/5-init
release and complete AMD queue. Running or queued os_* jobs, incomplete science,
unverified delivery, transport errors and changed hashes keep the gate closed.
It follows release versions and live jobs, including future OS continuations;
it never changes OS files, jobs, priorities or controllers.

Even after numeric completion becomes a candidate, the held stage additionally
requires evidence/OS_PRIORITY_RELEASE.json. The resumed same-chat agent must
verify the OS owner's actual completion and record verified_complete=true,
matching os_delivery_sha256 and the closeout evidence used. No new user approval
is needed. The controller rechecks the gate before releasing only its own stage.
The allocated stage checks it again. Never fabricate a release from a queue gap
or unsuccessful terminal jobs.

The OS owner subsequently changed its own final-release criteria (documented in
its2026-10-17 handoff). Do not demand that explicitly waived OS records become
strict passes. The additional owner-closeout route requires an actual finished
OS delivery/closeout, independently inspected by the resumed agent, plus zero
active/queued OS jobs. Record verified_complete=true, owner_completion_verified=true,
completion_basis and owner_closeout_evidence:[{path,sha256},...] in the ST release
receipt, binding at least two distinct actual files under the OS authority root.
A waiver announcement, partial report, empty queue or scientific-control fixture
alone does not suffice. ST physics, sampling and preflight criteria are unchanged.

Stage: source/sample/executable checksum-preserving copy, unchanged template with
only case basename/N/sample path/tlim varied; submit one AMD prepare job then
an afterok prepare preflight array0-2%1. Prepare constructs all samples in an
allocation and measures full-sample native initial transport velocities. Each
preflight runs0..20M, checks complete native checkpoints and all particle states;
the6M case also actually restarts20..22M. Separate preflight outputs are retained.

After all preflight receipts pass, the same-chat agent reviews timing/memory
and submits the three production runs with at most two active at once. Set
ST_WALL_HOURS to the job walltime; run.py leaves ten minutes for closure. Initial
forecasts are8h walltime for3M and16h for6M, updated from actual preflight rates.
Production uses fresh initial conditions, not the20M smoke checkpoints.

After all native preflight/restart receipts pass, submit allocated
`analysis.sbatch preflight_health` (10min wall is ample). It copies only the
source-hash-verified short-run history files and summarizes the same lapse,
constraint and core-energy diagnostics versus the preserved baseline. Inspect
its actual measurements; no new scientific stability threshold is imposed.
Then after actual native receipts, health, timing and memory review, write
`evidence/PREFLIGHT_REVIEW.json`: all_accepted=true,
six_million_restart_accepted=true, memory_and_timing_reviewed=true,
physics_and_mesh_unchanged=true, short_run_health_reviewed=true,
health_QA_sha256:SHA256(of evidence/PREFLIGHT_HEALTH_QA.json),
preflight_sha256:{CASE:SHA256,...} and
wall_hours:{CASE:INTEGER,...}. The hash map binds the mirrored actual AMD
evidence/preflight receipts; choose walltime at least their recommendations.
Then the lightweight metadata-only `submit_production.py` records unique
submission intents/receipts and submits the two3M cases, with6Mafterok of the
paired3M case to guarantee max2 concurrent productions. Existing recorded jobs
are verified rather than resubmitted; unresolved intents require scheduler
inspection. This is agent scientific review evidence, not a new user approval.

Forecast23 AMD production node-hours;25..30 with preparation/preflights/overhead,
plus3..5 allocated Perseus job-hours for analysis/movies. New AMD raw footprint
~350..400GB plus conservative420GB admission reservation. Entire-user watchdog
every5min: warn1.5TiB, stop own growth1.7TiB; no unrelated cancellation or deletion.

## Required continuation after production

Complete case-aware reductions preserving native timestamps, r10..r90 about COM
and origin, initial-ID cohorts, COM/pair-defect histories, lapse/density,
constraints/equilibrium error and fallbacks. Use identical100M/P/2P windows,
P=197.625957866M. Compare ordinary1.5M->3M and3M->6M time-weighted shell differences
normalized by common analytic radii; compare amplitudes, means and early/late
growth. One seed at fixed grid supports descriptive N dependence, not a
statistical or continuum-convergence claim. Pairing also suppresses all initial
odd-under-inversion structure and is not a pure COM-only correction.

Deliver migration-style absolute/initial-fractional r10/r50/r90 comparison plots,
health/COM context, three individual x-y movies plus paired and N-sequence
synchronized comparisons. Fixed20000 displayIDs, complete mirrored displaypairs,
initial3D coordinate-radius colours, common fixed axes,501 native states,15fps.
Use existing baseline movie cache for0..300M, whose full raw snapshots were
deliberately archived away. Full native baseline r10..r90 histories are retained;
do not invent exact pre300M r25/r75/r99 from the sparse display cache.

Analysis/rendering must run under Slurm. Fully decode MP4s, inspect actual
encoded frames and PNG/PDFs, independently verify representative shell/windows,
then amend Session6 reports while preserving earlier versions/manifests. Retain
all new raw/checkpoints and record actual paths/sizes/job/source/hashes/accounting.
No five-case paper rerun, cleanup or Mac mirror is authorized. The follow-up's
own evidence/COMPLETE closes its controller; old campaign markers remain intact.

## Allocated postproduction commands

Production closes with an allocated SHA256 manifest of every native output.
Perseus preempt has a two-hour maximum (verified17Oct); the launcher uses2h,
32GB/twoCPUs and the required unusedGPUslot. Existing first reductions15801/2
were corrected in place from3h to2h before any allocation, not resubmitted.
Inspect actual throughput; if a later case cannot fit, separate allocated
staging and reduction without changing diagnostic definitions or discarding
partial evidence.

For case index0/1/2, submit `analysis.sbatch reduce INDEX` on Perseus. It pulls
only particle/field/history diagnostics and the final full checkpoint, verifies
all selected files against that source manifest, then reduces every full-N
saved state and creates the20000-ID trajectory cache in the same pass. All
source outputs and intermediate checkpoints remain on AMD. A failed partial
reduction directory is evidence: use a new attempt destination after inspection,
never delete it to get past the refusal to overwrite.

Also stage the two2M/4M interior preflight particle frames with source hashes.
Check full per-ID native fields against production using the original Session6
rtol=atol=1e-7 tolerance. The shortened final20M alignment step is excluded from
this comparison. Record actual maxima in the case summary.

Once all three reductions pass, `analysis.sbatch compare` writes exact-window
Before6M finishes, `analysis.sbatch interim INDEX EXECUTION_COMMIT` can review
one completed case in a separate interim/CASE directory. It reuses the existing
independent audit functions for153 baseline and153 new-case windows plus three
full-N native frames, writes measured early/late amplitudes and health context,
and renders a provisional COM-shell PNG/PDF with an actual PDF preview. Pass
the submitted execution commit explicitly while an older reduction retains its
own active source pin. This interim artifact needs actual visual review and
does not replace the final four-case comparison/audit/movies.

Once all three reductions pass, `analysis.sbatch compare` writes exact-window
N differences, descriptive difference ratios, combined amplitude tables and
COM/origin migration-style shell/health/amplitude PNG/PDFs. The health density
proxy is the COM-centred0..0.5M coordinate-rest-mass bin divided by its actual
initial value. Baseline central-E slices begin at302M and are never treated as
an initial-density measurement. New central-E histories remain separately saved.

Then `analysis.sbatch movies GROUP` accepts0/1/2 (new individual cases),paired
(ordinary1.5M and antipodal3M) orscaling(ordinary1.5M/3M/6M). All groups require
all caches to choose the same unclipped fixed axes (atleast±16M), the same
0..8.93623112M initial-radius colour range,501frames and15fps. Comparison panels
match nominal2M output slots and show each case's actual native timestamp;
trajectories are not interpolated. Each movie is fully decoded and six actual
encoded frames saved for visual review. Computational receipts deliberately
leave actual visual review pending for the resumed agent.

`analysis.sbatch audit`, after reductions and comparison, independently checks
all612 baseline/new-case amplitude windows and all351 shell differences/117
N-difference ratios with scalar interpolation and quadrature. It also checks
all nine COM/origin deciles by partial order-statistic selection in nine full-N
native frames (0, approximately500,1000M across the three new cases), COM/core
bin masses, displayed IDs/positions/initial-radius colours and paired defects.
The audit writes RESULTS_ARITHMETIC_QA.json only after actual checks succeed;
it does not assert scientific stability or substitute for visual review.

These drivers are prepared code until actual allocated execution succeeds;
syntax checks are not claimed as scientific QA. Review preflight times and
memory, actual staged storage, independent quantile/window arithmetic, decoded
movies and PNG/PDFs before amending the main Session6 report and completing.
