# Session010: does a half-turn merger at separation 6 Mref make sense?

Analysis-only update, October 19, 2026. No input, boost, refinement, physics,
finder or simulation was changed. [Numerical estimates](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/plots/orbit_duration_20261019_v1/ESTIMATES.json)
and [figure](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/plots/orbit_duration_20261019_v1/orbit_duration_assessment.png)
were produced in Perseus Slurm17091 from existing data and analytic reference models.

**A short collapse/plunge is plausible for this evolving cluster model, but the
measured 0.67 turns are not explained merely by saying that a circular binary
starts at6M.** For the actual system,6 is coordinate separation in Mref units,
not separation in units of the black-hole pair's mass. This distinction changes
the usual vacuum-binary estimate substantially. The present run does not yet
validate a clean radiation-driven inspiral or identify the dominant cause of
the rapid inward motion.

## What was measured

The run accumulates 0.670424 observed turns from t0 to first strict common
enclosure at 64.36875; after both individual horizons appear,0.625921 turns.
Restart/sampling gaps are excluded. This is an accumulated angle, not half of
a constant orbital period. Initial separation is5.96877; around t=20 it is6.07585,
at t=40 it is4.90541, and near t=60 it is1.14784 before plunging to common enclosure.
At t=20 the two measured AH masses total 0.26256Mref; by t=60 they total 0.34953.
The masses and surrounding matter are evolving, so a fixed-mass two-body
approximation cannot describe the entire sequence.

The local Lorentz boost0.133215 is not the coordinate speed. With the inherited
initial central lapse/conformal factor, it corresponds to approximately0.08312
in coordinate speed and an initial orbital period about 226.77Mref. Applying
`2*pi*(3/0.133215)` would mix local speed and coordinate radius. The angular
frequency subsequently rises strongly, so neither that initial period nor a
constant-rate extrapolation predicts the merger time.

## Vacuum reference calculation

For a circular, isolated, equal-mass binary with constant total pair mass Mb,
eta=1/4, the leading quadrupole approximation gives

```text
T = 2*pi*sqrt(a^3/Mb)
da/dt = -(64/5)*eta*Mb^3/a^3
N(a0 -> af) = [(a0/Mb)^(5/2) - (af/Mb)^(5/2)] / (64*pi*eta).
```

These relations assume slow, nearly circular vacuum evolution. The orbit count
above follows by integrating Omega/(2*pi) over the decay, with G=c=1.
[Peters1964](https://journals.aps.org/pr/abstract/10.1103/PhysRev.136.B1224)
derives the leading point-mass radiation-reaction framework.

The two clumps' source masses total 0.24Mref; the envelope contributes0.76.
Using Mb=0.24 only as an illustrative isolated-pair normalization, a0=6Mref
is 25 Mb. The reference period is 188.50Mref, formal quadrupole lifetime to zero
separation 7324Mref, and integrated turns down to af=6Mb about 60.4. Holding
Mb=0.28 or0.36 instead still gives about 40.5 or 20.8 turns, respectively.
The cutoff 6 Mb is a convenient reference, not an exact equal-mass-binary ISCO.
These are **counterfactual scales**, not predictions for the matter-filled run.
They show why a half-turn here cannot be justified as an ordinary circular
vacuum binary simply starting at 6 times its own total mass. Replacing Mb by
the entire Mref also incorrectly treats the extended envelope as pair mass.

Collapse, changing hole masses, matter torques/drag, eccentric motion, and
initial constraint/gauge transients can change the result. Their relative
contributions are not measured by the existing particle-removal or raw rPsi4
ledgers; no radiated-energy/angular-momentum budget was established. The inherited
local momentum constraint is unsolved and convergence has not been demonstrated.
The later t≈98.2 finder restart defect is separate and does not establish the
cause of the pre-merger inward motion.

## How far apart for 2–3 turns?

A blind calibration of the vacuum scaling `N proportional to a0^(5/2)` through
0.6704 turns at 6 yields separations 9.29 and 10.93Mref for2 and3 turns. Starting
from a literal half-turn instead gives 10.45 and 12.29. Those arithmetic values
are **not a calibrated model**: the observed baseline does not satisfy the
constant-mass, vacuum, circular assumptions. They cannot guarantee the requested
number of turns, especially with unchanged boost.

**My first exploratory candidate would be separation 12 Mref, centers at ±6,
with a reviewed orbital-momentum prescription.** Bracketing 10 and 14 would help
establish the response if a new campaign is approved. This is a trial choice,
not a demonstrated minimum or a promised2–3-orbit setup. No such input has been
written. A separation-only change while retaining 0.133215 has no defensible
orbit-count guarantee.

The background becomes more important farther out. I computed instantaneous
radial force-balance speeds in the inherited analytic t0 metric. At one clump
center, hold the source centers fixed when differentiating; the self Gaussian
has zero gradient there, the companion is at distance d, and the background
profile is the archived M0.76/R30 solution. With beta=0 and spatial metric
psi^4 delta_ij, the static test-particle balance is

```text
v_local^2 = R * d(ln alpha)/dR / [1 + 2*R*d(ln psi)/dR],  R=d/2.
v_coord = alpha*v_local/psi^2;  T_coord=2*pi*R/v_coord.
```

| Trial separation d/Mref | Envelope conformal source inside R=d/2 | Static-metric local speed | Static coordinate period/Mref |
|---:|---:|---:|---:|
|6|0.02283|0.1278|236|
|8|0.05034|0.1376|289|
|10|0.08994|0.1512|324|
|12|0.14012|0.1651|353|
|14|0.19817|0.1776|379|
|16|0.26089|0.1879|405|

The enclosed source integral is 4*pi*integral(R^2*q0 dR); it is not an evolved
AH mass or a proper-volume accretion measurement. These speeds are starting
estimates in a frozen inherited metric, not approved boosts or a consistent
self-gravitating binary equilibrium. The current boost is close to the d=6
estimate, but at d=12 the estimate increases rather than following vacuum
Kepler's decreasing-speed rule. Keeping 0.133215 there may make inward motion
more eccentric. The d=6 estimate is only about 4% below the actual boost; gross initial
under-support is not established. A local boost around 0.165 at d=12 is only an initial estimate to review,
not a value implemented here.

A credible2–3-orbit design therefore needs momenta tuned with the envelope,
consistent constraints, and a short validation of separation/phase/eccentricity
after clump collapse. Controlled orbital parameters are standard in vacuum
low-eccentricity initial-data work; those fitted vacuum prescriptions do not
transfer directly to this evolving cluster.
[Mroue and Pfeiffer2012](https://arxiv.org/abs/1210.2958).
At d12, two or three *initial-period* lengths would be roughly 706–1059 Mref;
actual inspiral periods evolve, so this is only a scale for planning, not a
runtime or merger-time forecast. Such a longer run would exceed the earlier
conditional boundary/gauge-arrival estimates for the ±512 domain; contamination
amplitude has not been shown small. Resource limits, mesh placement, boundary
tradeoffs and the known finder repair would need separate agreement before a
new production campaign. No production or repair is part of this request.
