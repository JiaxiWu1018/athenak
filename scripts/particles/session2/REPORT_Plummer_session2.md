# Plummer Session 02 — scientific report

**Compactness scan of the relativistic Plummer Einstein cluster at
`(R/M)_eff = 10` and `6.5`.**

## Answer

> **As the Plummer Einstein-cluster family is made more relativistic, Session 1's weak
> core-local `l = 1` signal is replaced by a strong, *broadband* angular disruption of the
> inner cluster, and at `(R/M)_eff = 6.5` that disruption destroys the cluster's tangential
> support and drives its core into gravitational collapse within `2.6` half-mass periods.**
>
> A supplementary run at `N/4` shows the growth to be **`N`-independent to within 18 %**
> (reference-free ratio `1.180 +/- 0.030` over four bands, where a relaxation rate `~1/N`
> would require a factor of order `2`–`4`), so the effect is a property of the model and
> not of the particle number — at the one compactness and the one `N`-pair tested.

| | Session 1 | **R10** | **R6p5** |
|---|---|---|---|
| `(R/M)_eff` | 51.77 | 10 | 6.5 |
| interval established | `3 P_1/2` | `5 P_1/2` (complete) | `2.550 P_1/2` (run fails at `2.555`) |
| core `A_1` last/start | — | `86.3x` | `85.7x` **in half the time** |
| global `A_1/A_2/A_3/A_4` | — | `71/124/76/83x` | `68/199/137/243x` |
| `sigma_r/sigma_t`; `\|dL\|` rms | ~0 | 0.42; 1.30 | 0.66; 19.3 |
| inner tenth of the mass | — | **expands `+21.8 %`** | **contracts `-35.6 %`** |
| minimum lapse | — | `0.741 -> 0.629` | **`0.603 -> 0.115`** |
| `N`-dependence of the growth | not tested | not tested | **`1.18x` at `N/4`, i.e. `N`-independent** |
| **classification (§13)** | weak/local mode, globally stable | **clear growing instability** | **clear growing instability terminating in core collapse; numerically compromised beyond `2.555 P_1/2`** |

**A correction is recorded in §5 and should be read with this table.** The first analysis
of the `N/4` run reported the opposite conclusion — that the growth was
relaxation-dominated — because it normalised each band by its own `t = 0` amplitude. For
the centre-of-mass-referenced bands that amplitude is the geometric artifact Session 1
identified, not a shot-noise seed, and dividing by it inverted the result. The error, how
it was caught, and the reference-free measure that replaces it are set out in full.

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

## 3b. The Lagrangian shells keep their order, and that is measurable

Movie A colours every particle by its **initial** radial group and keeps that colour for
all time, so blending on screen is physical shell mixing and nothing else. Both cases'
final frames show the cohorts still radially ordered — a dark inner core, then the
intermediate groups, then the outer ones — rather than the blended field that scattering
would produce.

That is testable rather than impressionistic. The reduction tracks 32 equal-rest-mass
Lagrangian cohorts, so the question "have shells interchanged?" is the rank correlation
between a cohort's **initial** radial index and its **current** mean radius, plus the
count of adjacent cohort pairs that have swapped:

| | `t/P = 0` | 1 | 2 | 2.55 | 5 |
|---|---|---|---|---|---|
| **R10** Spearman `rho` | 1.000000 | 1.000000 | 1.000000 | 0.999633 | **0.997434** |
| **R10** adjacent inversions (of 31) | 0 | 0 | 0 | 1 | **3** |
| **R6p5** Spearman `rho` | 1.000000 | 1.000000 | 0.999633 | **0.993035** | — |
| **R6p5** adjacent inversions (of 31) | 0 | 0 | 1 | **3** | — |

After five periods and an `86x` mode amplification, **28 of R10's 31 shell boundaries are
still intact**, and R6p5's ordering is likewise preserved to `rho = 0.993` at its last
healthy time.

**This is an argument about the bulk motion, and it has a precise limit.** Preserved
ordering of cohort *mean* radii does not mean individual particles have not mixed — they
demonstrably have, with rms per-particle angular-momentum changes of `130 %` and
`1930 %`. What the measurement shows is that the mass distribution is being reorganised
**coherently**: the shells move, strongly, without passing through one another. A
diffusive process driven by graininess would progressively scramble that ordering, and
it does not.

So the bulk is reorganised coherently, with strong individual scattering on top.

**This was offered as evidence for a continuum mode, and it is not evidence — see §5.**
Secular relaxation-driven core collapse is also a process in which shells contract *in
order*, so preserved ordering excludes nothing either way. The `N`-scaling measurement
independently supports the continuum reading, but not because of this. The ordering
result stands as an accurate description of the motion; the inference drawn from it was
never sound.

One number in this table resists the simple collapse narrative and is left standing:
R6p5's innermost cohort mean radius goes `0.392 -> 0.730 M` by `2 P_1/2` and is back to
`0.563 M` at `2.55`, i.e. it does not simply contract, even while the `10 %` and `25 %`
enclosed-rest-mass radii fall by `36 %`. Those measure different things — a fixed
particle set's mean radius versus the radius enclosing a fixed mass fraction — and
reconciling them needs the 3D density this session did not dump.

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

## 4a. Direct comparison (specification section 14)

Every row is measured at each run's own endpoint — `5.000 P_1/2` for R10, the last
numerically healthy time `2.550 P_1/2` for R6p5 — with Session 1 shown where the same
quantity exists.

| | Session 1 | **R10** | **R6p5** |
|---|---|---|---|
| achieved `(R/M)_eff` | 51.767276 | **10** (err `3.6e-15`) | **6.5** (err `8.9e-16`) |
| `b` | 20 M | 3.86344456867 M | 2.51123896964 M |
| `r_t = 20 b` | 400 M | 77.2688913734 M | 50.2247793927 M |
| `P_1/2` | 1192.496781 M | 107.500505 M | 59.435698 M |
| `dx_fine`; cells per `b` / `r_1/2` / `R_1/2` | 1 M; 20.0 / 26.0 / 25.2 | 0.15625 M; 24.7 / 32.2 / 27.0 | 0.09375 M; 26.8 / 34.8 / 26.1 |
| leaf MeshBlocks; `dt` | 400; 0.25 | 456; 0.0390625 | 456; 0.0234375 |
| run length reached | `3 P_1/2` | **`5.000 P_1/2`** | **`2.550 P_1/2`** (fails at 2.555) |
| boundary signal arrival | 2.19 `P_1/2`, before its own endpoint | 7.92 `P_1/2` | 8.55 `P_1/2` |
| global `A_1^CoM`, last/start | — | 71.1x | 68.2x |
| **core `q0` `A_1`**, last/start | — | **86.3x** | **85.7x, in half the time** |
| `A_2 / A_3 / A_4`, last/start | — | 124 / 76 / 83x | 199 / 137 / 243x |
| radial localisation | inner quartile only | q0 86x, q1 51x, **q2 104x**, q3 14x — spreads outward | q0 86x, q1 20x, q2 8.9x, q3 3.8x — stays in the inner half |
| dipole direction | — | sign of `cos(D_0,D_1)` changes 4x; core within 13.2 deg of its own time-mean | changes 3x; core within 10.1 deg |
| CoM motion, `R_CoM/R_1/2` | — | 0.042 | 0.023 |
| ADM momentum | — | at or below `3.1e-12` while a clean radius existed; none left by `5 P_1/2` | at or below `5.5e-12`; two clean radii remain |
| radial equilibrium, `r_q10`/`r_q25`/`r_q50` | held | **+21.8 / -2.5 / -2.2 per cent** (expands) | **-35.6 / -36.1 / +13.8 per cent** (core collapses) |
| `sigma_r/sigma_t`; per-particle `dL` rms | ~0 | 0.42; 1.30 | 0.66; 19.3 |
| minimum lapse (continuum value) | — | 0.629 (0.741) | **0.115** (0.603) |
| matter-region `H` norm, start to end | — | `7.4e-04` to `5.9e-05`, falls | `2.2e-03` to `1.8e-03`, then diverges |
| particles lost; non-finite states | none | **none; none** | none, then all at once |
| `N`-dependence of the growth | not tested | not tested | **`1.18x` at `N/4` — `N`-independent** (§5) |
| **classification (section 13)** | weak/local mode, globally stable | **clear growing instability** | **clear growing instability terminating in core collapse; numerically compromised beyond `2.555 P_1/2`** |

The scan is monotonic in compactness in every measure, and the two cases differ in
**outcome**, not only in degree: R10 disrupts and **expands**, R6p5 disrupts and its core
**collapses**. §5 shows the growth to be `N`-independent to 18 % at `(R/M)_eff = 6.5`, so these are
properties of the model and not of the particle number — at the one compactness and the
one `N`-pair tested.

## 5. The `N`-scaling test, and an error it took an adversarial review to catch

Both production runs used `N = 2,113,536`, so neither could separate a continuum
instability from graininess-driven relaxation of a cluster whose only support is
tangential — with rms per-particle angular-momentum changes of `130 %` and `1930 %`, the
question was not academic. A supplementary run repeated R6p5 at `N = 528,384`, mesh, seed,
gauge and every other setting byte-identical, to `2 P_1/2`. It completed cleanly with all
528,384 particles alive.

### The measurement

Both runs' modes are seeded by shot noise of amplitude exactly `A^shot = (N/2)^{-1/2}`, so
the quantity that isolates the `N`-dependence **of the growth** is the mode amplitude
divided by that run's **own** shot floor:

| band | full-`N` `A_1/A^shot` | `N/4` `A_1/A^shot` | ratio |
|---|---|---|---|
| `all` (origin, global) | 58.56 | 68.72 | **1.174** |
| `com` (CoM, global) | 55.02 | 65.98 | **1.199** |
| `m0` (core) | 133.61 | 162.04 | **1.213** |
| `m1` | 24.76 | 28.09 | **1.135** |
| | | **mean** | **1.180 +/- 0.030** |

A continuum instability predicts `1.00`. A relaxation rate falling like `1/N` predicts a
factor of order `2`–`4` depending on how rate maps onto amplitude. **The measurement is
`1.18`**: `N`-independent to 18 %, four bands agreeing to 3 %.

The raw amplitudes corroborate it. The `N/4` mode ends `2.357 +/- 0.056` times higher in
absolute terms, against the `2.000` that an `N`-independent growth of a `sqrt(N)`-smaller
seed predicts — the same 18 % excess, and stable to 2.4 % across all six bands including
the origin-referenced ones.

### The error: dividing by the artifact this campaign exists to avoid

The first analysis instead compared each band's amplification against its own `t = 0`
value and reported `2.94x faster at N/4`, concluding relaxation. That was wrong.

For a CoM-referenced band, `A_1` at `t = 0` is **not** shot noise: it is dominated by the
geometric `(2/3)<1/r>|s|` term a displaced reference manufactures on a centrally peaked
profile. Measured here, the core band's `t = 0` value is `4.98x` its own shot floor at
full `N` and `2.05x` at `N/4` — and it is **larger at full `N`** (`9.69e-03`) than at
`N/4` (`8.00e-03`), which a genuine shot seed cannot be. The denominator was an artifact,
and the two runs' artifacts differ by a factor unrelated to `sqrt(N)`.

The decomposition is exact:

    2.941  =  [A_low(2P)/A_full(2P)]  x  [A_full(0)/A_low(0)]
           =        2.426              x        1.212

The second factor should be `0.500` if the `t = 0` references scaled as `n_uniq^{-1/2}`.
It is `1.212`. **That single factor is the entire reported result.**

Two diagnostics make the failure unambiguous, and both were available before the
conclusion was written:

* **The verdict flipped with the reference point.** Same particles, same physics: the core
  band's amplification ratio reads `2.94` ("faster at low `N`") CoM-referenced and `0.84`
  ("slower at low `N`") origin-referenced. No physical result may depend on that choice.
* **The scatter gave it away.** Across bands the amplification ratio spreads `1.530 +/-
  0.709` — **46 %** — while the absolute-amplitude ratio spreads `2.357 +/- 0.056`, **2.4 %**.
  A measurement whose band-to-band scatter is fifteen times its physical counterpart's is
  measuring its own denominator. The reported `2.94` was the largest of four noisy draws.
* The one band with a clean `t = 0` reference — `all`, whose `t = 0` is `0.97x` and `0.99x`
  shot in the two runs — gives `1.150` even on the old measure, and it was not in the
  table, because the tool looped only over `m0..m3` and `com`.

Session 1's correction was that an origin-referenced band dipole is contaminated by
exactly this geometric term. This session quoted that warning in four places and then used
the contaminated quantity as a denominator. The measure is now reference-free,
`analysis/compare_nscaling.py` documents the trap at the top of the file, and the old
number is printed beside the new one and labelled unreliable.

### Consequences for the three arguments

Three arguments for a continuum origin had been assembled from the production data, then
retracted when the flawed measurement disagreed, and the retraction is now itself
withdrawn. The arguments' *conclusion* is supported by the corrected measurement, but two
of them were weak arguments regardless and remain so:

1. **"The core log slope accelerates."** Still not discriminating — relaxation-driven core
   collapse is also a runaway.
2. **"Growth is faster at higher compactness at fixed `N`."** Still not worked out here.
3. **"The Lagrangian shells keep their order."** Still not discriminating — secular
   relaxation-driven collapse contracts shells in order too.

They pointed the right way for inadequate reasons. The measurement, not the arguments,
carries the conclusion.

### What is still not established

* **The 18 % excess is unexplained.** It is small, consistent across bands, and in the
  direction relaxation would push. It may be a genuine subdominant relaxation
  contribution, or a single-realisation fluctuation. One seed per `N` cannot tell.
* **One `N`-pair, one compactness, one time.** This tests the hypotheses at
  `(R/M)_eff = 6.5` and `2 P_1/2`; it does not measure a scaling law and says nothing
  directly about R10. The homogeneous campaign used an `8x` range in `N` and four seeds.
* **The collapse's origin is separately untested.** The `N/4` run was stopped at
  `2 P_1/2` by design. At that time the full-`N` minimum lapse has fallen `12.5 %`
  (`0.603 -> 0.528`) while the `N/4` run's is unchanged at `0.603`, even though the `N/4`
  inner mass has contracted *more* (`-18.5 %` against `-4.8 %`). The lapse and the
  enclosed-mass radii disagree about which run is nearer collapse, and neither run was
  carried far enough in the other's frame to date the event. **No claim is made about
  whether the collapse itself is `N`-dependent.**
* The initial matter-region constraint violation scales as pure shot noise (`N/4` is
  exactly `2.00x` full `N` at every time sampled). Since the growth is `N`-independent
  while its seed is not, a constraint-violation-driven mechanism is disfavoured, but not
  excluded by these data alone.

### What a follow-up needs

A third and fourth `N` to fit the scaling rather than test it; several seeds per `N`, since
one realisation cannot separate an 18 % excess from a fluctuation; the same at
`(R/M)_eff = 10`; and for the collapse, an `N/4` run carried past `2.6 P_1/2` together with
excision or a puncture gauge and an apparent-horizon finder.

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
