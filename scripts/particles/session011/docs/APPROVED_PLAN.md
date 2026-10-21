# Approved Session 011 implementation contract

The user approved implementation after the plan discussion on 2026-10-21.

## Scope and source

One fresh t0->75 Mref production trajectory on AMD. Same Session 010 physical
input, particle realization (5M, seed4001), separation6, boost0.133215, domain±512,
root256^3, blocks32^3, ghost4, initial dx1/64, ceiling1/256, wave floor dx<=.25
through R46, RK4/CFL.4, unchanged gauge/pusher/deposition/removal. Lapse removal
alpha<.05; AH removal off. Preserve memory/VTK/full-precision repairs.
No extra evolved tests, controls, pilots, seeds or replay trajectories. Code tests
and non-evolving AH checks on saved production states are explicitly allowed.
Stop at75 with a verified checkpoint and resource/science report; extension to150
requires a new user decision. No overall node-hour ceiling; monitor and report usage.

## Finder and evidence

Separate search state from publication; retain safe provisional full shapes and
serialize centers/coefficients; restore common-horizon warm starts; residual quality
must accompany stagnation. Preserve existing geometry/resolution/association/
persistence/residual gates. Record every search including failure reasons; accepted
shape and physical quantities use exact cycle/time provenance. Tests cover pending
persistence, individual/common serialization, invalid guesses and non-mutating
saved-state searches. Normal production restarts verify reacquisition. No claim of
uninterrupted-vs-restarted evolution equivalence or convergence.

## Diagnostics and five hypotheses

Use exact component/tag identities, rest weights, covariant momenta, Eulerian mW
and coordinate energy -m*u_t with explicit conventions. Full-population moving-AH
crossings distinguish first enclosure, later crossings/surface sweep, common-AH
reassignment and lapse deletions. Unknown AH intervals stay unknown. Store
component/shell rest/energy/angular momentum and transfer ledgers. Coordinate
Jz(O)=sum m[(x-Ox)u_y-(y-Oy)u_x]; fixed origin for global accounting.

A: compare horizon growth, entering matter angular momentum, M/dotM, L/|dotL|
and d/|dotd|, without identifying AH growth with rest-mass accretion.
B: envelope wakes/multipoles, shell angular momentum and flux-corrected changes;
effective torques remain coordinate dependent and must not double count A.
C: exterior-envelope profiles, shell motion, proper volume and metric averages;
compare evolving and frozen approximate circular support, label model dependence.
D: separation, unwrapped phase, Omega, radial/tangential ratio and slice/path-
dependent exterior proper separation. Count turns only over valid intervals;
binary interpretation ends at accepted common surface enclosing both live trackers.
E: exterior H/M proper-volume norms, AMR/dt, restart continuity, sampling
asymmetries and non-evolving AH resolution checks. No grid/dt/seed evolution controls.

## Runtime and resources

Start6 nodes/24 ranks, max_nmb_per_rank256, one MI210 per rank. Known-good gnu12
12.2.0/OpenMPI4.1.8/CMake3.25.2/prun2.3/ROCm6.4.1; HIP GFX90A, MPI, O3,
PROBLEM=particles/gi_cluster. Preserve vader single-copy=none.
Warn memory80%, checkpointed stop before85% allowing observed growth. Track
per-rank block occupancy. Fallback same trajectory:7nodes/cap240, then8/cap224.
12h allocations, initially ~2h application chunks, >=40min finalization reserve
adjusted to measured I/O and memory growth. No separate benchmark trajectories.
Provisional cost500-1500 AMD nodeh, not a cap or benchmark. Every invocation
enforces tlim75. Record actual nodeh for all builds/checks/failures/production.

## Outputs/storage

AH events/geometry every eligible search. Crossing/removal accounting each step;
compact ledgers/track/profile/constraint output .025 through common enclosure,
.05 afterward. Waves ell2..8 at .025, R40 plus R36/R44 diagnostic companions.
Fixed90k component-stratified cohort .25, denser .05 near formation/merger; exact
IDs/weights/momenta, no replacement. Selected slices .5. Retain full states at
initialization, first simultaneous individual AH acceptance, first inward d<1
while individual AHs are distinct, strict common enclosure, and final75.
Keep two verified recovery checkpoints plus in-progress write; rotate only
session-created recovery/temporary files under inventory/checksum verification.
Retain irreplaceable science states. Anticipated peak450-700GiB, recalibrate
from actual bytes. Five-minute whole-user watchdog warns1.5TiB, stops growth1.7TiB
or projected1.9TiB, tolerates transient du failures but stops on stale telemetry.
Raw staysAMD; durable compact results/scripts/provenance/figures/movies are
checksum-transferred to Perseus. No unrestricted mirror or unrelated cleanup.

## Products and endpoint decision

Four-panel 3D particle/accepted-AH snapshots, two aligned mass/spin panels,
wide/close1080p movies through75. Session007 navy/purple/orange/cyan, camera
elev22/azim-58; accurate geometry, times, identities, held frames and gaps.
Spin labelled coordinate-rotation estimate. Event states chosen from actual data.
Strain method FFI for h=hplus-i*hcross, saved rPsi4; document cutoff/window/gaps,
endpoint trust and differentiation check, finite-radius/retarded-time limitations.
R40 burst historically near115, so final burst strain figure awaits possible
extension. Report at75: endpoint/AH coverage/orbits/common time, all five causes
and limits, nodeh/memory/throughput/diagnostic/I/O costs, storage/retention, cost
forecast75->150 and reproducible products. Do not submit beyond75.
