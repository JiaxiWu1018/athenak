# Approved Session 8 contract

Implementation authorized 2026-10-21 in thread 01a15d4b-acc1-7623-b3e7-209e4daa341c.

Five fresh proper-profile stratified seed-1 equal-mass 2,000,000-particle runs:
0.283, 0.500, 0.605, 0.819, 0.900 in that execution order. No imposed perturbation,
no additional convergence trajectories. Preserve Z4c, gauge, RK4/CFL .4,
GR-Boris safeguards and conservative deposition of Session 7.

Stable candidates: domain +/-1024, root128 cubed/block32 cubed, initial finest .5
inside +/-32, AMR ceiling .25 (num_levels7 including root).
Collapsing candidates: domain +/-512, initial finest .25 inside +/-16,
AMR ceiling .0625 (num_levels8). Both initial nested trees have344 blocks.
Loehner alpha*psi^7, threshold .2, derefine .02, ncycle_check16,
refinement_interval16, protected initial hierarchy floors. Log AMR/dt history.
Stable endpoint300M; unstable min(300, first accepted AH+20M).
Unstable lapse excision .1, AH diagnostic only; stable removal disabled.

AH search .5M, ntheta10/lmax6, initial radius .5. Require finite positive
geometry, every point on grid, R_A sqrt(<Theta^2>_A)<=.05, minimum coordinate
radius >=4 local cells, three consecutive acceptable searches. Mass stagnation
alone insufficient. Persist safe search state, reacquire validity after restart.
Report M_irr and Christodoulou mass separately, final-five-M summary, no assumed
settling. Missing/failing AH intervals remain explicit.

Histories .5 stable/.25 unstable; full particles and 2D z4c/tmunu/con slices
2M stable/1M unstable; fixed20k-ID movie cohort .5M. Initial/final3D fields,
AH-onset checkpoint, recovery every50M/segment boundary. Preserve exact time,
cycle, IDs and provenance. No legacy trk writer.

Products: five-column2-row COM rest-mass-decile/core-lapse figure (stop shell
curves at first removal), mass closure figure alive+removed+escaped versus
initial rest mass, three-panel AH mass figure. Five individual movies and one
synchronized comparison, initial-radius colors,15fps, contemporaneous AHs,
explicit ended panels. PDFs/PNGs/MP4s/tables/captions/science and technical
reports. Session7 supports particle sensitivity, not convergence of this AMR setup.

One ST trajectory at a time on AMD, initially1node/4ranks/max_nmb_per_rank128;
measured fallback2nodes/same capacity, no automatic third node. Warn GPU80%,
clean-stop before85%; predicted AMR capacity checked before growth. Fixed
ROCm6.4.1/gnu12/OpenMPI4/CMake/prun, HIPgfx90a, vader single-copy none.
Provisional20-60allocated nodeh; review60, hard cap100 including all work.
Track billing separately. Reserve350GiB ST and concurrent Jeans growth up to
700GiB; five-minute whole-user warning1.5TiB/stop1.7TiB/projected1.9TiB.
Transient du errors tolerated, stale telemetry triggers clean-stop.
User explicitly allows rotation of superseded Session8 recovery checkpoints:
inventory and verify replacements, retain latest2 plus AH-onset and finals.
Do not delete prior-session or Jeans data. Raw/computation staysAMD, thin final
products checksum-mirrored toPerseus; exact raw paths/size/retention recorded.

Acceptance: input/sample/mesh/source/memory checks, focused AMR/ledger/cohort/
stop/restart tests, non-evolving AH checks. Five bounded trajectories and actual
scientific/numeric/visual verification, full movie decode, final accounting and
manifests required. Resource stops explicit, no fabricated completion. Preserve
continuation state and automatic same-thread events until done or user stops.
