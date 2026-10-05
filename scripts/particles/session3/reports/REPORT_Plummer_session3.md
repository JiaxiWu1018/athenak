# Plummer session 03: isotropic equilibrium benchmark

Status at 2026-10-05 22:54 UTC: initialization and matched AMD 100-step preflights
validated. Production is stopped at the approved budget gate: the measured
projection is 403.08 AMD node-hours, above the 200-hour ceiling. Scientific
classification is **inconclusive because evolution/control coverage is insufficient**;
the completed preflights are numerically healthy.

## Model and reference clock

Part I of the preserved derivation defines a metric-first, isotropic
Einstein–Vlasov equilibrium. We use G=c=M=1, a=10M, eta=0.10, zero initial shift
and extrinsic curvature, psi=1+M/(2 sqrt(r²+a²)) and
alpha=(1-u)/(1+u). Here r is isotropic radius. The circular model remains the
default initializer; session 03 explicitly selects the isotropic model.

The full energy distribution is reconstructed with the regularized Abel integral
and factored kernel. The independent Python reference uses the PDF's elementary
closed form at 70-digit precision and adaptive quadrature. Proper rest mass sets
the radial sampling measure and the equal particle rest mass. Independently drawn
momentum directions accompany deterministic stratified radii and co-located
opposite-momentum pairs, seed 1985.

| Continuum quantity | Value |
|---|---:|
| Infinite-model rest mass | 1.0134873008561143 M |
| Infinite-model rest-mass median, isotropic r | 13.000000002572222 M |
| Common circular-geodesic reference clock P_ref | 461.99570043659014 M |
| Five reference periods | 2309.9785021829507 M |
| Rest mass inside 100a | 1.0133372946305388 M |
| Omitted rest mass at 100a | 0.0148009970573% of M0_inf |
| Rest mass inside 200a | 1.0134497989044768 M |
| Omitted rest mass at 200a | 0.00370028826269% of M0_inf |

P_ref labels every arm and is a reference clock, not a common period of the
isotropic particle orbits. The sampling cut leaves the analytic infinite metric
and its asymptotic mass unchanged. There is no reflecting wall, removal or
excision at either sampling radius.

## Initialization evidence

All 354 independent Python/C++ comparisons passed the 1e-8 relative numerical
agreement target; the largest relative error was 3.7120e-9. Job 11084 strengthens
the earlier CDF checks to relative error, including small quantiles. The earlier
absolute-CDF record is retained separately.
Evidence: [profile agreement](../evidence/profile_agreement.json).

Full-N frozen and live initializations each contain exactly 2,113,536 contiguous
tags in 1,056,768 co-located pairs. Stored opposite momenta cancel exactly.
The sampled proper density and all three orthonormal stresses agree with
continuum shell integrals within measured pair-aware sampling uncertainty and
the radial-stratum edge bound. The validator applies a five-sigma familywise
screen; passing this screen is not a claim of two-percent precision in sparse
shells. Particle-dump and in-code integrated energy/stress moments differ by at
most 1.25e-10 in these initial tests.

[Frozen sampling](../evidence/sampling_cpu_frozen.json),
[live sampling](../evidence/sampling_cpu_live.json), and
[initialization fidelity figure](../figures/initialization_fidelity.pdf).

The initial minimum lapse is about 0.904795. Initial live core Hamiltonian L2 is
5.6854e-5; the core is the fixed r<=100M region. This initial value is not the
post-transient runaway reference. Positive ADM-mass controls at R=9000M and
10000M reproduce the analytic finite-radius surface integrals to better than
1e-7. Their values differ from the asymptotic M=1 as expected. The positive
Bowen–York momentum quadrature check passes to floating-point precision.

## Evolution and interpretation gates

The approved matrix contains main and tail frozen/live arms for five P_ref and
coarser-core and N/4 frozen/live arms for two P_ref. Main/tail/low-N meshes have
624 leaves; the coarser mesh has 568. Mesh inspection verifies both counts.
Coarse CFL=0.125 preserves the main nominal dt=0.078125M.

Before production, the matched quarter-period pilot, its post-transient
reference interval, restart comparison, half-timestep comparison and causal/
diffusive geometry checks must pass. No such evolved pilot result is available
at this report's current timestamp. The 100-step preflight reaches only
7.8125M = 0.01691033 P_ref in each arm. It is a performance and startup test;
it cannot establish equilibrium or secular angular behavior.

Primary angular ledgers use common infinite-model rest-mass quartiles in COM
coordinate radius, retaining l=1..4 signed coefficients and the dipole vector.
Origin-referenced modes remain auxiliary. Eulerian radial cohorts may mix, and
individual radii may change in an isotropic distribution. Frozen orbit accuracy
is judged from energy and angular momentum.

Final classification will distinguish equilibrium within targets, near
equilibrium with measured drift/growth, substantial departure, and compromised
or inconclusive data. Targets are 1% for bulk rest-mass radii/integrated stresses
and 2% for resolved, adequately populated profiles. Full startup will remain
visible; secular assessment begins after 0.25 P_ref. Sparse or uncertain bins
are inconclusive. An exponential rate requires a resolved excess over matched
frozen fluctuations, two e-folds, eight independent correlation blocks, and
consistent windows. Neither numerical health nor particle-number agreement
proves continuum stability.

The frozen preflight's final relative energy error is 1.0493e-9 RMS
(maximum 1.5040e-8); its ensemble-normalized angular-vector error is 3.7612e-11
RMS. Both preflights retain exact particle counts and relative rest-mass
accounting error 1.4287e-13. Live core Hamiltonian L2 declines from 5.6854e-5
to 1.9260e-5 during this startup interval. One live Boris step uses the documented
forward-Euler fallback at cycle 91; that event is recorded, not erased. The
half-timestep pilot is still needed to assess its effect.

## Measured budget and pending decision

The conservative preflight rates are 5.85415 seconds/step frozen and 7.01408
seconds/step live, dt=0.078125M. Applying baseline cost to the unmeasured controls
gives 295.94 hours of matrix stepping. Output/checkpoint margin, measured pilot
cost and build/reduction reserves bring the projection to 403.08 node-hours.
Registered AMD work actually consumed 0.5925 node-hours through preflight
reduction; the 200-hour allocation has not been exhausted.

[Budget evidence](../evidence/budget_preflight.json) and
[reviewable alternatives](../evidence/budget_options.json) retain the arithmetic.
No alternative is approved and no production deck has been changed:

| Proposed scope | Estimated node-hours | Coverage limitation |
|---|---:|---|
| Retain original matrix; proposed ceiling 425 | 403.08 | Original five/two-period coverage |
| Keep 200; main/tail 2P and mesh/N controls 1P | 191.69 | No five-period conclusion; controls cover only 1P |
| Keep 200; main pair alone to 5P | 165.26 | Tail, mesh and N controls absent |

## Products and retention

The initialization figure is complete. Production milestone figures, period
notes and the two full-run main movies await successful production gates and
actual evolution coverage. Raw initial evidence remains on Perseus; AMD
preflight raw output remains at the run paths in the technical report. Anta's
/data2 reports zero available space, so archive transfer is pending. No campaign
data has been deleted.

[Matched startup health figure](../figures/preflight_health.pdf) includes both
M and P_ref time axes, the full available startup, and radius histogram bounds.
It contains no extrapolation to the unexecuted production periods.
