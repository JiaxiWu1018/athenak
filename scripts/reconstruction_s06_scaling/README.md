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
For case index0/1/2, submit `analysis.sbatch reduce INDEX` on Perseus. It pulls
only particle/field/history diagnostics and the final full checkpoint, verifies
all selected files against that source manifest, then reduces every full-N
saved state and creates the20000-ID trajectory cache in the same pass. All
source outputs and intermediate checkpoints remain on AMD. A failed partial
reduction directory is evidence: use a new attempt destination after inspection,
never delete it to get past the refusal to overwrite.

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

These drivers are prepared code until actual allocated execution succeeds;
syntax checks are not claimed as scientific QA. Review preflight times and
memory, actual staged storage, independent quantile/window arithmetic, decoded
movies and PNG/PDFs before amending the main Session6 report and completing.
