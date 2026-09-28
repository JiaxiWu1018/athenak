# Plummer Session 02 — scientific report

**Compactness scan of the relativistic Plummer Einstein cluster at
`(R/M)_eff = 10` and `6.5`.**

## Answer

> **As the Plummer Einstein-cluster family is made more relativistic, the weak core-local
> `l = 1` signal of Session 1 does not merely strengthen — it is replaced by a strong,
> broadband angular disruption of the whole inner cluster, and at `(R/M)_eff = 6.5` that
> disruption destroys the cluster's tangential support and drives its core into
> gravitational collapse within `2.6` half-mass periods.**

Neither case remains stable, and neither is well described as a dipole instability.

| | Session 1 | **R10** | **R6p5** |
|---|---|---|---|
| `(R/M)_eff` | 51.77 | 10 | 6.5 |
| interval established | `3 P_1/2` | **`5 P_1/2`** (complete) | **`2.550 P_1/2`** (run fails at `2.555`) |
| classification (§13) | weak/local mode, globally stable | **clear growing instability** | **clear growing instability terminating in core collapse; numerically compromised beyond `2.555 P_1/2`** |
| core `q0` `A_1^CoM` last/start | — | `86.3x` | `85.7x` **in half the time** |
| global `A_1 / A_2 / A_3 / A_4` | — | `71x / 124x / 76x / 83x` | `68x / 199x / 137x / 243x` |
| `sigma_r/sigma_t` | ~0 | `0.42` | `0.66` |
| `\|dL\|` rms per particle | — | `1.30` | **`19.3`** |
| inner tenth of the mass | — | **expands `+21.8 %`** | **contracts `-35.6 %`** |
| minimum lapse | — | `0.741 -> 0.629` (`-15 %`) | **`0.603 -> 0.115`** (`-81 %`) |
| matter-region `‖H‖_2` | — | `7.4e-04 -> 5.9e-05` (falls) | `2.2e-03 -> 1.8e-03`, then diverges |
| particles lost | — | **none** | none until failure, then all |
| `P^ADM` | — | `0` while a clean radius existed | `0` while a clean radius existed |

## 1. What was run

Two live NRPIC runs of the Session-1 construction — the static-observer energy density of
Plummer form in areal radius, hard-cut at `r_t = 20 b`, renormalised to `M_ADM = 1`,
supported entirely by circular geodesics with random orbital-plane orientations, zero net
rotation and zero initial radial dispersion — with `b` solved so that
`(R/M)_eff = 1/max_r[m(r)/r]` equals `10` and `6.5` exactly (`3.6e-15` and `8.9e-16`
absolute error). `b = 3.86344456867 M` and `2.51123896964 M`.

Everything else was held at Session 1's values: `N = 2,113,536` co-located `+/-u_i` pairs,
seed `1985`, BSSN with `use_z4c = false`, the same gauge, damping and dissipation, RK4 at
`CFL = 0.25`, the GR Boris pusher, live feedback, conservative cross-level deposition, no
filtering, no excision. Resolution *improved* on Session 1 (`24.7` and `26.8` cells per
`b` against `20.0`) and a seventh refinement level pushed the earliest inward
gauge-signal arrival to `7.9` and `8.6 P_1/2`, so — unlike Session 1, whose boundary
signal arrived at `2.19 P_1/2` before its own endpoint — **the outer boundary is causally
irrelevant over both records**.

Full parameters: [`../README.md`](../README.md). Reproduction:
[`REPORT_AGENT_Plummer_session2.md`](REPORT_AGENT_Plummer_session2.md).

## 2. `(R/M)_eff = 10`: growing, broadband, survives

The run completes `5 P_1/2` with all 2,113,536 particles alive, no non-finite states, and
a matter-region Hamiltonian constraint norm that *falls* by an order of magnitude
(`7.4e-04 -> 5.9e-05`) as the parabolic damping absorbs the initial finite-`N` violation.
The timestep never leaves its hyperbolic bound. This is a healthy calculation.

It is also unambiguously unstable. The centre-of-mass-referenced `l = 1` amplitude grows
`71x` globally and `86x` in the innermost equal-rest-mass quartile. The growth is not
confined to the core — by the endpoint `q1` has grown `51x` and `q2` `104x`, the largest
of any band — so it spreads outward through the cluster over the five periods, reaching
even the outer quartile (`13.8x`).

**It is not an `l = 1` mode.** Growth by multipole is `A_1 71x`, `A_2 124x`, `A_3 76x`,
`A_4 83x`: the higher multipoles grow *as fast or faster*. A displacement of the
distribution from its reference point is a dipole effect; this is the angular distribution
being scrambled at every scale. `l = 1` is merely the largest in absolute amplitude, as it
was at `t = 0`. Session 1's framing — an internal dipole mode — does not carry over.

The mechanism is visible in the velocity structure. This cluster is supported *only* by
tangential motion: `sigma_r = 0` exactly at `t = 0` by construction. By `5 P_1/2` the rms
change in individual specific angular momentum is `130 %` and `sigma_r/sigma_t` has risen
to `0.42`. Orbits are being scattered out of circularity, converting tangential support
into radial motion. The cluster responds by **expanding**: the inner tenth of the mass
moves outward `+21.8 %` and the `75 %` radius `+16.8 %`, while the half-mass radius holds
to `-2.2 %`. The minimum lapse deepens only `15 %`, from `0.741` to `0.629`.

Dipole direction behaviour is not that of a coherent translation: `cos(D_0, D_1)` changes
sign four times over the record, so neighbouring shells are alternately aligned and
anti-aligned, while the core's direction stays within `13.2 deg` of its own time-mean
(isotropic wander would be `90 deg`). The coordinate centre of mass moves to
`0.042 R_1/2`.

## 3. `(R/M)_eff = 6.5`: the same disruption, then collapse

The same instability appears, faster and with a different ending.

By `t/P_1/2 = 2.550` the core quartile has grown `85.7x` — the amplitude R10 needs five
periods to reach, attained in **half the time**. The disruption is again broadband and
more strongly so: `A_2 199x`, `A_3 137x`, `A_4 243x` against `A_1 68x`. The rms angular
momentum change reaches `1930 %`, fifteen times R10's, and `sigma_r/sigma_t` reaches
`0.66`.

Here the cluster does **not** expand. The inner tenth of the mass contracts `-35.6 %` and
the inner quarter `-36.1 %`, while the half-mass radius moves *outward* `+13.8 %` and the
`90 %` radius is unchanged at `-0.2 %`. Mass is concentrating in the core while the middle
shells expand — a core-collapse signature, not a global contraction. The minimum lapse,
flat at its continuum value `0.603` until `t/P_1/2 ~ 1.5`, then falls monotonically and
with increasing steepness to `0.115` — a factor `5.3` — while the volume-averaged lapse
over the inner `R < R_1/2/2` falls only `0.618 -> 0.581`. So the lapse collapse is sharply
**localised**, far smaller than the innermost diagnostic bin.

At `t = 151.878908 M` (`t/P_1/2 = 2.55535`), one history interval after the last healthy
row, `2,088,590` of `2,113,536` particles go non-finite simultaneously and the calculation
is over.

**This is consistent with the onset of gravitational collapse, and the failure is the
expected consequence of not being equipped for it.** The deck runs BSSN with no excision,
no puncture gauge and no apparent-horizon finder, because Session 1 was a weak-field
baseline where no particle should ever be removed. A configuration forming a horizon
cannot be carried by that setup. What the record *establishes* is the localised lapse
collapse, the core mass concentration and the destruction of tangential support; it does
**not** establish horizon formation, because no horizon diagnostic was enabled. That
distinction is kept throughout.

The collapse is reached through the instability, not independently of it: the tangential
support that holds an Einstein cluster up is precisely what the broadband angular
scattering destroys.

## 3a. Morphology, and one tension in the record

The uniform-grid equatorial slices (Movie B, one pixel per finest cell) show what the
scalars cannot. At R6p5's last healthy frame the core is a sharp bright concentration
ringed by a distinct shell near `R ~ 2.5-3 M`, and the Hamiltonian-constraint panel
carries a strong, tightly localised violation exactly at the centre — the same place the
lapse collapses. R10's final frame instead shows a broadly flattened core.

The azimuthally averaged slice profiles agree on flattening but not on concentration:

| | central `E` at `t = 0` | at the end |
|---|---|---|
| R10 (`5 P_1/2`) | `4e-03` | `8e-04` |
| R6p5 (`2.54 P_1/2`) | `1e-02` | `5e-03` |

**For R10 this is consistent** with everything else: the slice flattens and the inner
Lagrangian shells expand `+21.8 %`.

**For R6p5 it is in apparent tension** with the enclosed-rest-mass radii, which contract
`-35.6 %` (`r_q10`) and `-36.1 %` (`r_q25`). A 3D Lagrangian shell moving inward while the
equatorial-slice density at small radius falls by a factor two is not a contradiction —
the enclosed-mass radii are 3D and measured over all particles, the slice is one plane —
but it does require the collapsing region to be markedly **non-spherical**, which the
broadband `l`-content independently implies. The coordinate centre of mass is far too
small (`0.023 R_1/2 = 0.056 M`) to explain it as a displacement.

This session does not resolve which geometry the core takes. It is recorded as an
observation with its tension intact rather than smoothed into the collapse narrative; a
follow-up wanting to characterise the collapse should dump the 3D density rather than a
plane, and enable an apparent-horizon finder.

## 3b. What the particle movie adds: the inner shells collapse *coherently*

Movie A colours every particle by its **initial** radial group and keeps that colour for
all time, so blending on screen is physical shell mixing and nothing else. R6p5's last
healthy frame shows the innermost cohorts as a **compact, still-distinct dark core**
inside `R_1/2`, with the outer cohorts spread beyond it — the particles that began
innermost have contracted together into a dense central concentration rather than being
scattered through the cluster.

That bears on the continuum-versus-finite-`N` question, and in the opposite direction to
the raw `|dL|` numbers. Two-body relaxation is a *diffusive* process: it would mix the
Lagrangian shells, blending the colours as inner and outer particles exchange places. The
frame instead shows the inner shells retaining their identity while moving inward
together, which is what a collective mode does.

It is an argument, not a measurement — the `N/4` run is the measurement — and it is
recorded as such. It is also consistent with the two other arguments already on the
record: the core-band log slope accelerates rather than growing linearly in time, and the
growth is faster at higher compactness at fixed `N`.

## 4. The momentum question, answered

Session 2's required new diagnostic asked whether the dipole and centre-of-mass behaviour
is accompanied by growth of total linear momentum. **It is not.**

While an uncontaminated extraction sphere existed, `P_i^ADM` stayed at the `1e-12`
quadrature floor and was radius-independent across four spheres spanning
`R/R_t = 3.45` to `8.18`. The matter-side momenta, meanwhile, drifted to `1e-4`—`1e-3` and
tracked each other to four digits. Coordinate momenta move; the spacetime's momentum does
not. So the disruption is **internal, with no recoil** — the first of the three
possibilities the specification asked to separate.

This came with a caveat that had to be measured rather than assumed. The finite-`N`
constraint violation launches a disturbance at the density truncation that propagates
outward at the 1+log gauge speed — measured two independent ways, `1.363` from the
constraint field and `1.484` from the sphere arrival times, against `sqrt(2) = 1.4142` —
and as it crosses a sphere it lifts that surface integral by up to ten orders of
magnitude. It is a *weak* disturbance (`|H|` in the lit vacuum bins is `5e-7`—`5e-4` of the
matter value) that a *hypersensitive* diagnostic sees, because `P_i` is a near-perfect
cancellation. Every quoted value therefore comes from the outermost sphere the front had
not yet reached. By `5 P_1/2` in R10 the front has passed all six spheres and **no clean
radius remains**; the R10 endpoint momentum is reported with that caveat, not as a
measurement. See
[`../evidence/pmom_radii_seam_free_20260911/RECORD.md`](../evidence/pmom_radii_seam_free_20260911/RECORD.md).

## 5. What is not established

**Whether this instability is a property of the continuum model or of the finite-`N`
realisation is not settled by these two runs, and it is the single most important open
question.** The growth is accompanied by rms per-particle angular-momentum changes of
`130 %` (R10) and `1930 %` (R6p5). Graininess in a cluster supported entirely by
tangential orbits is a plausible mechanism for exactly this, and `N = 2,113,536` is the
same in both runs, so nothing here separates the two.

The homogeneous campaign settled the analogous question with an `N`-scaling study across
an `8x` range, which found the rate unchanged. A supplementary `N/4` run
(`N = 528,384`, same mesh, seed and deck, to `2 P_1/2`) is in flight for R6p5 to give one
point of that comparison; its result is recorded in
[`../SESSION_LOG.md`](../SESSION_LOG.md) and this section will be updated. Two arguments
bear on it now, neither decisive: the core-band log slope *accelerates* (`+0.44` then
`+2.66` per period for R6p5), whereas diffusive relaxation would be closer to linear in
time; and the growth is faster at higher compactness at fixed `N`, which a purely
numerical relaxation rate should not care about. Both are arguments, not measurements.

No growth rate is fitted. The sliding one-period log slopes are `+0.13, +2.27, +1.43,
+0.34, +0.10` (R10) and `+0.44, +2.66` (R6p5) — accelerating then decelerating, not a
constant exponential — and Session 1 withdrew three published rates for less scatter than
this. For scale only: R6p5's second period corresponds to `2.6` e-folds, the same order as
the homogeneous cluster's `3.24 +/- 0.22` e-folds per surface period at the same
`R/M = 6.5`.

Also unestablished: horizon formation in R6p5 (no horizon finder was enabled); the
behaviour of R6p5 beyond `2.555 P_1/2` (the specification's `5 P_1/2` endpoint is
unreachable for this configuration with this setup); and whether R10 would eventually
collapse too, given that it is still disrupting at its endpoint.

## 6. Reading the products

* **Use** `notes/INTERIM_R10_P1..P5.md` and `figures/R10/after_1P..after_5P/` for R10 —
  the full record is healthy.
* **Use** `notes/INTERIM_R6p5_P1.md`, `P2` and `notes/INTERIM_R6p5_FINAL.md`, with
  `figures/R6p5/after_1P`, `after_2P` and `figures/R6p5/final_bounded/` for R6p5.
* **Do not use** `notes/superseded_unbounded/` or
  `figures/R6p5/superseded_unbounded/`. Those milestone-3, -4 and -5 products were
  generated before the analysis layer bounded itself at the failure, and they report
  `min alpha = 1.8e308`, `N_alive = 0`, `nan` constraint norms and amplitudes computed
  from zeros as though they were measurements. The
  [README there](../notes/superseded_unbounded/README.md) lists the specific false
  statements.

All amplitudes are centre-of-mass referenced, with `A_l` on the established campaign
definition and the finite-`N` null set by the number of independent angular positions
`N_pair = N/2` (the sampler co-locates each pair). Session 1's caution that the core
band's nominal Poisson floor is not a valid baseline is honoured: the quoted amplification
is always against a band's own initial value, never against a floor it did not start on.
