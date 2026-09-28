# Plummer Session 02 — technical/agent report

Reproduction record for the compactness scan at `(R/M)_eff = 10` and `6.5`.
Companion to [`REPORT_Plummer_session2.md`](REPORT_Plummer_session2.md) (scientific) and
[`../SESSION_LOG.md`](../SESSION_LOG.md) (chronology).

**Status: complete.** R10 completed all five milestones healthy; R6p5 collapsed and
failed at `t/P_1/2 = 2.55535`; the supplementary `N/4` run finished and showed the growth
to be **finite-`N` relaxation-dominated**, which changed the session's conclusion.

---

## 1. Provenance

| item | value |
|---|---|
| repository | `git@github.com:JiaxiWu1018/athenak.git` |
| branch | `project/Plummer-cluster` |
| base for Session 2 | `82817777` (Session-1 close-out) |
| Session-2 commits | `72ad8481`, `8cc534eb`, `e8c07841`, `7c7ce43f`, `373d0622` |
| Perseus checkout | `/data/jiaxiwu/NRPIC/Plummer-cluster/code` |
| AMD checkout | `/work1/eliasmost/jiaxiwu/plummer_s02_20260910/src/athenak` (independent clone) |
| Kokkos submodule | `6739bc62` (4.7.02) |
| production executable | `build_rocm641/src/athena` built from `8cc534eb` |

`72ad8481` adds the ADM linear-momentum diagnostic; `8cc534eb` fixes its weight-closure
test; `e8c07841` and `7c7ce43f` add the Session-2 inputs and analysis layer; `373d0622`
adds the movie renderers. Only `72ad8481` and `8cc534eb` touch `src/`.

## 2. Build

Partition `mi2101x`, script `scripts/amd_build.sbatch`.

```
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
cmake -S $repo -B $build \
  -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_AMD_GFX90A=On \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc \
  -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_FLAGS=-O3 -DCMAKE_C_FLAGS=-O3 \
  -DPROBLEM=particles/nr_pic_plummer
cmake --build $build -j 16
```

**Do not substitute the default ROCm 7.x**: it builds but faults at runtime in particle
stress-energy deposition (`AGENTS.md`). Clean build job `413112` (5 m 12 s);
incremental rebuilds via `scripts/amd_rebuild.sbatch` (`413147`, `413229`, ~1 m each),
which `git fetch`+`reset --hard origin/project/Plummer-cluster` first so the built
executable always corresponds to a pushed commit.

Executable SHA-256 at `8cc534eb`: recorded in `build_rocm641/exe.sha256`.

## 3. The continuum models

`analysis/solve_compactness.py` root-finds `b` against the numerically measured
`max_r[m(r)/r]` of `analysis/plummer_1d.py` (unchanged from Session 1) and cross-checks the
closed form. Because `r_t/b = 20` is held fixed, `f_t = 8000/401^{3/2}` is independent of
`b`, `max(m/r) = 2 M_P/(3 sqrt(3) b)` at `r = b sqrt(2)`, and

    (R/M)_eff = 3 sqrt(3) b f_t / (2 M) = 2.588362 b     (M = 1),

so the scan is exactly linear in `b`. The script's regression gate is that it reproduce
every published Session-1 number at `b = 20 M`; it does, to all printed digits.

Run as `python3 analysis/solve_compactness.py --npanel 40000 --ngl 20 --out
initial_data/compactness_solution.json`; quadrature converged to `1e-10` relative in
`P_1/2` (`npanel` 40 000 -> 160 000). Per-case values are in
`initial_data/reference_values_{R10,R6p5}.json` and tabulated in
[`../README.md`](../README.md).

Pre-production checks (specification §2), both cases PASS:

| check | criterion | R10 | R6p5 |
|---|---|---|---|
| circular geodesic exists | `max m/r < 1/3` | 0.1 | 0.153846 |
| individual orbit radially stable | `r^2 m' + r m - 6 m^2 > 0`, min dimensionless margin | 0.646012 | 0.455403 |
| no horizon / regular geometry | `max 2m/r < 1`, `min alpha > 0` | 0.2, 0.740835 | 0.307692, 0.602881 |
| continuum Hamiltonian identity | relative | 8.86e-13 | 6.57e-13 |
| continuum lapse identity | relative | 7.92e-13 | 8.53e-13 |

## 4. Mesh

Session 1's family unchanged (root `128^3` over `[-L,L]^3`, MeshBlock `32^3`, nghost 4,
level-`l` region `[-L/2^l, L/2^l]`), with one extra level. Every region is exactly four
parent blocks across, so it is MeshBlock-aligned for any `L`, and the leaf count is
`56 N + 64`. Designed and evaluated with `analysis/mesh_design.py`; the full table,
including both timestep bounds, boundary arrival times, cost and the two properties that
are tighter than Session 1, is in [`../SESSION_LOG.md`](../SESSION_LOG.md).

`athena -i <deck> -m` confirms `456` MeshBlocks for both cases (`56` leaves on each of
levels 0–6, `64` on level 7), and the startup banner confirms `dt_par/dt_hyp = 2.07737`
(R10) and `1.13564` (R6p5) against `2.077` and `1.135` predicted — so the evaluator's
model of `z4c_newdt.cpp` is correct.

## 5. Input decks

Generated, never hand-edited, by `scripts/make_session2_inputs.py`. Session 1's
`make_plummer_inputs.py` cannot do this: its `set_key()` exits when a block or key is
absent and cannot *add* either, and the Session-1 deck has six `<refined_region*>` blocks
where Session 2 needs seven. Every derived number — `b`, `r_t`, `P_1/2`, `r_1/2`, `R_1/2`,
`R_t`, `M_0`, `mu`, the ledger ranges, `tlim`, the extraction radii and all six output
cadences — is computed from `plummer_1d.py` at generation time, so a deck cannot drift
from the reference values. `--check` re-derives and diffs, so that claim is mechanically
auditable; it currently reports all eight artefacts matching.

Preserved verbatim from the Session-1 production deck, because the scan changes
compactness and not the numerics: the whole `<z4c>` block (BSSN `use_z4c=false`, 1+log
lapse with `lapse_oplog=2`, legacy Gamma-driver with `shift_eta=2`, `diss=0.1`,
`hdamp_cH=0.02`, `hdamp_par_safety=0.5`), the whole `<particles>` block (`gr_boris`,
`feedback=true`, `cross_level_deposit=conservative`, `tmunu_filter_passes=0`, excision
off), `cfl_number=0.25`, `integrator=rk4`, `nghost=4`, `N` and the seed.

Changed, with reasons: `plummer_b` and `plummer_rt` (the point of the scan — **both** must
be set, since each is `GetOrAdd` with a Session-1 default, so omitting `rt` would silently
build a different-`f_t` cluster); the mesh; `tlim = 5 P_1/2`; the output cadences (exact
fractions of each case's own `P_1/2`); `plummer_shell_rmin/rmax` rescaled to
`[0.025 b, 40 b]` and `plummer_field_rmin/rmax` to `[dx_fine, 2L]`, which preserves
Session 1's log-bin *geometry* (same dynamic range, same bin count) rather than its
absolute radii; `plummer_metric_R0 = R_t/800`, the same ratio Session 1 used; and the new
`plummer_pmom_*` keys. `trk` output stays disabled.

## 6. The ADM linear-momentum diagnostic (specification §9)

### 6.1 Derivation

Required: `P_i = (1/8 pi) oint dS_m (K^m_i - delta^m_i K)`. On a coordinate sphere
`R = const` of the Cartesian grid coordinates, with `nu_i = x_i/R` the **flat** unit radial
one-form, the oriented surface element is

    dS_m = sqrt(gamma) nu_m R^2 dOmega ,

with `sqrt(gamma)` the determinant of the spatial metric in **Cartesian** components.
Proof: the unit normal one-form is `s_m = nu_m / sqrt(gamma^{kl} nu_k nu_l)`, and the
proper area element of the coordinate sphere is

    sqrt(det sigma) dtheta dphi = sqrt(det gamma) sqrt(gamma^{kl} nu_k nu_l) R^2 dOmega ,

which follows from `det gamma_(R,theta,phi) = det sigma / gamma^{RR}` together with
`det gamma_(R,theta,phi) = det gamma_cart R^4 sin^2(theta)` and
`gamma^{RR} = gamma^{kl} nu_k nu_l`. The normalisation cancels between `s_m` and
`sqrt(det sigma)`, leaving

    P_i = (R^2 / 8 pi) oint sqrt(gamma) [ gamma^{mk} nu_m K_{ki} - nu_i K ] dOmega ,
    K = gamma^{ij} K_{ij} .

Under conformal flatness `gamma_ij = psi^4 delta_ij` this reduces to the familiar
`dS_m = psi^6 n_m R^2 dOmega`, with `s^i dA = psi^2 n^i R^2 dOmega`.

`K_ij` is read from `padm->u_adm[I_ADM_KXX..I_ADM_KZZ]`, which holds the **physical**
extrinsic curvature: `src/z4c/z4c_adm.cpp:234` sets
`K_ij = psi4 A~_ij + (Khat + 2 Theta) g_ij / 3`, and `Z4c::ConvertZ4cToADM` is queued every
cycle (`src/z4c/z4c_tasks.cpp:82`), so no BSSN conversion factor is needed and `u_adm` is
current whenever the history hook runs. This was established by reading the code, not
assumed.

### 6.2 Implementation

All in `src/pgen/particles/nr_pic_plummer.cpp` (purely additive, 436 insertions, inert
when `plummer_pmom_nrad = 0`):

| symbol | role |
|---|---|
| `BowenYorkFields` | analytic `gamma_ij = delta_ij`, trace-free `K_ij` for a given `P` |
| `PlummerMomentumQuadrature` | the angular sum above, from interpolated `g_dd` and `K_dd` |
| `PlummerSetupADMMomentum` | builds the spheres, fixes angle ownership, runs the unit test |
| `PlummerADMMomentum` | evaluates every sphere and appends the ledger |
| `PlummerDepositedMomentum` | `int S_i sqrt(gamma) d^3x` in one mesh sweep |

Quadrature is `GaussLegendreGrid` (`src/geodesic-grid/gauss_legendre.cpp`): `ntheta`
Gauss-Legendre nodes in `cos(theta)` times `2 ntheta` uniform nodes in `phi`,
`nangles = 2 ntheta^2 = 2048` at `ntheta = 32`, weights `w_k^GL * pi/ntheta` summing to
`4 pi` exactly. This is spectrally accurate for this integrand, and materially better than
the equal-area geodesic `SphericalGrid` used by the Weyl extraction.

**Rank double counting is made impossible, not unlikely.** The ownership test in
`SetInterpolationIndices` is inclusive at both ends, and this grid always places whole
`phi` columns on `y = 0` and `x = 0`, which are MeshBlock faces on a mesh centred at the
origin; 32 of 2048 angles per sphere are so shared in production. Every weight is divided
by the number of ranks claiming that angle; an angle claimed by no rank is fatal; and the
corrected weights are Allreduced and asserted to sum to `4 pi`. The first version of that
assertion summed `w/own` once per angle instead of once per owning rank and therefore
could never close — it fired at `12.468196` on the first full-`N` preflight, which is how
the 0.78 % of shared solid angle was measured. Fixed in `8cc534eb`.

### 6.3 Extraction radii

Each sphere lies **wholly within one refinement level**. The refined regions are cubes,
and on a sphere `max_i|n_i|` runs over `[1/sqrt(3), 1]`, so the sphere crosses
`|x_i| <= H` whenever `H < R <= sqrt(3) H`; with `H_{l+1} = H_l/2` the seam-free band is
`R/H_l in (sqrt(3)/2, 1]`. `seam_free_radii()` takes two radii per level from the
vacuum-side bands. The first radii, placed midway between half-widths, had five of six
spheres per case straddling a seam at up to a 42.8 % split — see
[`../evidence/pmom_radii_seam_free_20260911/RECORD.md`](../evidence/pmom_radii_seam_free_20260911/RECORD.md),
including why neither `t = 0` test could have caught it.

| | R10 (M) | level, dx | R6p5 (M) | level, dx |
|---|---|---|---|---|
| r1 | 141.335346 | 3, 2.5 | 84.801208 | 3, 1.5 |
| r2 | 156.0 | 3, 2.5 | 93.6 | 3, 1.5 |
| r3 | 282.670692 | 2, 5 | 169.602415 | 2, 3 |
| r4 | 312.0 | 2, 5 | 187.2 | 2, 3 |
| r5 | 565.341384 | 1, 10 | 339.204830 | 1, 6 |
| r6 | 624.0 | 1, 10 | 374.400000 | 1, 6 |

All in vacuum (`R/R_t >= 1.72`), all resolved at `R/dx >= 56.5`, spanning a factor 4.4
across three levels with two radii per level, so a difference between levels can be
separated from a difference between radii. Verified independently by
`analysis/check_pmom_radii.py`, which reproduces the diagnostic's own quadrature and
reports per-level solid-angle fractions: all twelve spheres 100 % on one level.

### 6.4 Validation

| test | expectation | measured |
|---|---|---|
| Bowen-York unit test, fatal on failure | returns `P` exactly at every radius | `1.11e-15` in production |
| quadrature closure after the rank reduction | `4 pi` | `12.5663706144` |
| `t = 0` null | `P_i = 0` (`K_ij = 0` in static data) | exactly `0` at all radii |
| proper-area control `A/4 pi R^2` | `(1 + M/2R)^4` in the `t=0` vacuum | to `5.0e-12` |
| particle momentum `sum_p m_p u_i` | `0` by pair cancellation | exactly `0` |
| deposited `int S_i sqrt(gamma) d^3x` | roundoff | `~1e-21` |

The Bowen-York test is the load-bearing one: it checks the tensor algebra and the
quadrature against a **closed form** rather than against zero. Its expected value was
derived independently in Python before implementation (`n^k K_{ki} = (3/2R^2)[P_i +
(P.n) n_i]`, `oint [P_i + (P.n)n_i] dOmega = (16 pi/3) P_i`, times `(1/8pi)(3/2)` gives
`P_i`), together with the `adm.hpp` device helpers `SpatialDet`/`SpatialInv`/`Trace`, all
of which agreed to `<= 1e-15`.

The `t = 0` null and the area control are recorded as *insufficient*: `K_ij` vanishes
identically in static initial data so the null exercises nothing downstream, and the
analytic ADM fields are written into ghost zones so there is no seam error for the area
control to expose. Both pass identically at a clean radius and at a straddling one.

### 6.5 Ledger

`<basename>.plummer_admmom.csv`, one row per sphere per history time:

```
time,cycle,R,Px_adm,Py_adm,Pz_adm,absP_adm,area_ratio,mean_trK,sum_w,
Px_matter,Py_matter,Pz_matter,absP_matter,Px_dep,Py_dep,Pz_dep,mom_l2
```

`P_matter_i = sum_p m_p u_i(p)` over all particles with `u_i` the covariant spatial
4-velocity the pusher stores, accumulated in three previously unused slots of the existing
global reducer. `P_dep_i = int S_i sqrt(gamma) d^3x` from the deposited source.
`mom_l2` is the volume-weighted momentum-constraint norm, which the pgen already computed
but never reported. A separate ledger was necessary because
`NHISTORY_VARIABLES = 20` is saturated by Session 1's 20 columns.

## 7. Production

Driver `scripts/drive_milestones.sh CASE`, one detached supervisor per case.

**Milestones are exact, not detected.** Each segment is submitted with
`time/tlim = k P_1/2` on the command line: `main.cpp:288` applies `ModifyFromCmdline`
*after* loading the restart's parameter dump so the override wins, `mesh.cpp:668` clamps
the final step with `dt = tlim - time` so the run lands on `tlim` exactly, and
`Driver::Finalize` (`driver.cpp:683`) writes every output type including the restart. So
"milestone `k` reached" is a discrete, deterministic event with a complete output set and
a checkpoint at exactly that time. Session 1 instead polled for "`t` has crossed
`k P_1/2`" with an in-memory flag, which double-fired once and truncated its CSVs.

A milestone may take more than one Slurm job: each runs until the earlier of `tlim` and
its wall-clock budget, reporting `CASE DONE` or `CASE INCOMPLETE`; the driver resubmits
the identical command on the latter and is idempotent across interruptions, with
per-milestone markers under `state/<case>/`.

| | R10 | R6p5 |
|---|---|---|
| AMD run dir | `runs/prod_R10` | `runs/prod_R6p5` |
| `tlim` at milestone `k` | `k x 107.50050549470241` | `k x 59.43569795258119` |
| milestone-1 job | `414891` | `414892` |
| segment wall time | 4 h (MI210 VRAM-safe) | 4 h |
| **measured throughput** | **4.469 s/cycle** | **4.264 s/cycle** |
| cycles per `P_1/2` | 2752 | 2536 |
| wall time per `P_1/2` | 3.42 h | 3.00 h |
| wall time to `5 P_1/2` | **17.1 h** | **15.0 h** |

Measured from the first two `ndiag` reports of jobs `414891`/`414892` (cycle 0 at
`elapsed = 15.23`/`15.68 s`, cycle 100 at `462.16`/`442.08 s`). Both are *faster* than the
`5.03 s/cycle` predicted by scaling Session 1's `4.41 s/cycle` by the leaf-block ratio
`456/400`, because a fixed share of the cost is per-particle and `N` is unchanged. Total
for the pair is **32.1 node-hours**, against an allocation balance of 2260 remaining.

Each milestone therefore fits in a single 4 h Slurm job: the runner hands Athena
`lim - 900 s = 13 500 s`, and a milestone needs `12 299 s` (R10, 91 % of budget) and
`10 813 s` (R6p5, 80 %). R10 is close enough that output I/O may push a milestone into a
second job; that is the designed `CASE INCOMPLETE` path and costs only a resubmission.

`dt` is holding at the hyperbolic bound in both runs — at cycle 100,
`t = 3.90625 = 100 x 0.0390625` (R10) and `2.34375 = 100 x 0.0234375` (R6p5) exactly, so
the parabolic branch has not taken over.

Segments are 4 h because MI210 VRAM grows over a process lifetime: Session-1 job `407606`
died after 10 h 34 m with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at 70 % per-card VRAM.
The runner also keeps the `OMPI_MCA_btl_vader_single_copy_mechanism=none` gate, without
which the stack floods `errno=14` and loses ~4.8x throughput, and carries the five-minute
`du` watchdog on the whole AMD user root (warn 1.5 TiB, cancel 1.7 TiB).

Checkpoint retention: the driver keeps every integer-`P_1/2` checkpoint (the `rst` cadence
is `P/4`, so index `% 4 == 0`) plus the two newest, and prunes the rest; unpruned the 21
checkpoints per case would be 126 GB at 6.006 GB each.

## 8. Milestone analyses, and the run outcomes

`scripts/analyze_milestone.sh CASE K` ran automatically from the driver after every
completed `P_1/2`: reduce cumulatively on `mi2101x` into `reduced/<case>_P<k>/` with
`--tmax = k P_1/2`, pull the reduction, the four in-code ledgers, both history files and
the movie frames to Perseus, regenerate the nine figure groups into
`figures/<case>/after_<k>P/`, write `notes/INTERIM_<case>_P<k>.md`. Per-milestone output
directories rather than appending, because `reduce_run.py` opens its CSVs with mode `w`;
this also guarantees no milestone's products can be overwritten by a later one.
Reduction ran at ~3 s/frame on `mi2101x` (~75 min per case for all five milestones), on a
different partition from production so it never delayed a run.

| milestone | R10 | R6p5 |
|---|---|---|
| 1 `P_1/2` | `CASE DONE` (jobs `414891` incomplete, `415234`) | `CASE DONE` (`414892`) |
| 2 `P_1/2` | `CASE DONE` (`415319`) | `CASE DONE` (`415917` after `415237` incomplete) |
| 3 `P_1/2` | `CASE DONE` (`415962`) | `CASE DONE` (`416029`) — **already past the failure** |
| 4 `P_1/2` | `CASE DONE` (`416123`) | `CASE DONE` (`416166`) — **garbage** |
| 5 `P_1/2` | `CASE DONE` (`416416`) — healthy | `CASE DONE` (`416166`) — **garbage** |
| outcome | complete, healthy at the endpoint | **failed at `t = 151.878908 M`** |

### 8.1 The R6p5 failure, and what the harness got wrong

At `t = 151.878908 M` (`t/P_1/2 = 2.55535`), one history interval after the last healthy
row at `151.568355 M` (`2.55012`), `2,088,590` of `2,113,536` particles went non-finite
simultaneously. The preceding rows show the cause: the minimum lapse falling
`0.149 -> 0.137 -> 0.125 -> 0.115` while the matter-region constraint norm rose
`9.5e-04 -> 1.1e-03 -> 1.3e-03 -> 1.8e-03`. A collapsing core in a BSSN run with no
excision, no puncture gauge and no horizon finder.

**AthenaK kept running to the `5 P_1/2` endpoint, and the harness let it.** Two design
gaps, both mine:

1. **The driver had no physics-blocker gate.** `drive_milestones.sh` advances on the
   Slurm-level verdict (`CASE DONE` / `CASE INCOMPLETE` / `CASE FAILED`) and on transient
   GPU OOM, none of which a non-finite particle state triggers — the run terminated
   normally at each `tlim`. The specification's "continue automatically unless there is a
   genuine physics/numerics/safety blocker" was implemented as *continue automatically*,
   with the blocker check left to a human reading the notes. About **2.45 `P_1/2` of
   allocation, roughly 7 node-hours**, was spent integrating a dead configuration.
2. **The analysis layer reported on it.** The milestone-5 note stated
   `min alpha = 1.79769313486232e+308` against a continuum `0.602881`,
   `N_alive = 0/2113536` with a verdict still printed, `|P^ADM| = 0.000e+00` at every
   radius "consistent with zero", `|dE| rms = nan`, and a core amplification of `86.31x`
   computed from zeros. All of it from well-formed output that reduces without error.

The one honest symptom was `f9_health` failing outright — a matplotlib `TypeError` from an
axis height of `1.6e14`, derived from the `1.8e308` lapse. That crash, visible in the
milestone notifications, is what led to the discovery.

**Fixed** (`ebebd05b`): `analysis/figs_session2.py:valid_window()` judges health on the
history columns that cannot be wrong in a good state — full particle count alive, no
non-finite particles, lapse finite and in `(0, 1]`, finite matter-region constraint norm —
and returns the last healthy time. The figure suite bounds every dataframe inside
`dedup()` (which every panel reads through) and every axis inside `periods_axis()`,
marking the failure on each panel and stamping "record truncated there"; the note bounds
its milestone and prints the failure above the table. Contaminated products are
quarantined under `notes/superseded_unbounded/` and
`figures/R6p5/superseded_unbounded/` with a README listing the false statements, not
deleted. **Still not fixed:** the driver itself has no health gate, so a future session
must either add one or watch the notes.

### 8.2 Measured performance

| | R10 | R6p5 |
|---|---|---|
| throughput, compute only | 4.469 s/cycle | 4.264 s/cycle |
| cycles to the endpoint | 13,760 | 12,680 (7,010 useful) |
| segments | 6 (4 h then 6 h) | 6 (4 h then 6 h) |
| wall clock | ~19 h | ~17 h |

Segments were raised from 4 h to 6 h after the first milestone: the measured s/cycle is
compute-only and output I/O adds about 25 %, so R6p5's first milestone consumed 96 % of a
3 h 45 m Athena budget and R10 overran its own. MI210 VRAM was measured flat at
`44-45 %` across a full segment with no growth, against the `70 %` at which Session-1 job
`407606` died after 10 h 34 m, so 6 h is safe. **37.3 node-hours** across 14 jobs for the
whole session, against an allocation balance of 2260 at launch.

### 8.3 Restart continuity (specification §5) — PASS

R6p5's `t = 0.75 P_1/2` checkpoint restarted in a separate run directory and evolved to
`t = 1 P_1/2` (job `415229`), then compared against the production history row at the same
time. `N_alive`, `M0_alive`, `r_q50` and `N_nonfinite` agree **exactly**; `sigma_t` to
`3.5e-16`; `E_part` and `alpha_min` to `~1e-15`; `R_com`, `com_x`, `sigma_r` to `~2e-14`;
every `A_l` to `~1e-13`; both Hamiltonian norms to `8e-13`; worst of all columns `com_y`
at **`1.227e-12`**.

`boris_nfail_cum` is excluded and must be: it is a **process-lifetime** tally of GR-Boris
fallbacks, not checkpointed state, so a segment starting at `0.75 P_1/2` has necessarily
counted fewer (`16`) than one running since `t = 0` (`53`). Including it made the first
automated verdict read FAIL at `6.98e-01` on a test that passed by twelve orders of
magnitude. The same fact means the fallback count in each interim note is **per segment**,
not a run total.

### 8.4 The supplementary `N/4` run — result

Launched at Jiaxi's direction to address the one question the two requested runs cannot:
whether the instability is a continuum property or finite-`N` relaxation.

| | value |
|---|---|
| job | `442036`, `-t 08:00:00`, `prod_R6p5_N4`, `CASE DONE` in 2 h 51 m |
| deck | `R6p5_prod` with command-line overrides |
| overrides | `job/basename=pl_R6p5_N4_s1985`, `problem/plummer_npair=264192`, `time/tlim=118.87139590516237` |
| `N` | **528,384**, confirmed from the first history row and the final particle accounting (`initial=528384 final=528384 ... conservation OK`) |
| `M_0` | `1.04976569178031731` — identical to full `N`, as it must be |
| endpoint | `2 P_1/2` exactly, inside R6p5's healthy window |
| reduction | job `442123`; `reduce_run.py` measured `n_uniq = 264,192` itself (`N/n_uniq = 2.0000`) and derived `A_shot = 1.945540e-03`, exactly twice the full-`N` floor |

**Result** (`analysis/compare_nscaling.py`, `initial_data/nscaling_R6p5.json`):

| band | full-`N` | `N/4` | ratio |
|---|---|---|---|
| core `q0` | 26.82x | **78.86x** | **2.94** |
| `q1` | 18.08x | 27.18x | 1.50 |
| global CoM | 19.07x | 34.17x | 1.79 |
| `sigma_r` | 0.0966 | 0.3112 | 3.22 |
| `\|dL\|` rms | 1.091 | 1.531 | 1.40 |

A continuum instability predicts `1.00`. The measurement is `2.94`, between pure `1/N`
(`4.00`) and `1/N` with a Coulomb-log correction (`3.55`). Robustness checks: the `N/4`
run ends `2.4x` higher in **absolute** amplitude, not only in amplification, and its
growth is `159x` its own larger noise floor, so it is signal.

**Consequence for the session's conclusion.** The growth measured in both production runs
— and the R6p5 core collapse it produced — is relaxation-dominated at
`N = 2,113,536`. Both cases are reclassified from "clear growing instability" to
**inconclusive for the continuum**. Three arguments for a continuum origin that had been
assembled from the production data alone are retracted in §5 of the scientific report,
with the reason each failed.

### 8.5 Two methodological notes from this run

* The comparison must be of **amplification against each run's own `t = 0`**, never of
  raw amplitudes: the nulls differ by exactly `2.000x` by construction. Note also that
  the `t = 0` band amplitudes do **not** differ by 2 (`m0` differs by `0.82`), because
  `A_1` at `t = 0` is dominated by the CoM-subtraction term on the cusped profile rather
  than by shot noise — Session 1's caution resurfacing.
* `compare_nscaling.py` was first validated on the partial `N/4` data at `t/P = 0.8`,
  where it issued five confident per-band verdicts from amplifications of `0.27`–`2.02`,
  i.e. from noise over noise. It now refuses a verdict unless the full-`N` band has
  amplified at least `3x`. At the real comparison point the core band is at `26.8x`, well
  clear of that gate.

## 9. Diagnostic definitions


`A_l = sqrt( 4 pi/(2l+1) sum_m <Y_lm(nhat)>^2 )` with real orthonormal `Y_lm` and an
unweighted mean over particles (equal rest masses, so unweighted equals mass-weighted) —
the established campaign definition, so Session 1, Session 2 and the homogeneous campaign
are directly comparable. `A_l = 1` for an angular delta function.

The finite-`N` null is `A_l^shot = n_uniq^{-1/2}` with `n_uniq` the number of
**independent angular positions**. The sampler co-locates every `+/-u_i` pair, so
`n_uniq = N/2 = N_pair`, **not** `N`; `reduce_run.py` measures it from frame 0 and aborts
on mismatch. Globally `9.7277000994e-04`; per equal-rest-mass quartile
`(N_pair/4)^{-1/2} = 1.9455400199e-03`.

All primary amplitudes are **centre-of-mass referenced**. Session 1 established that an
origin-referenced band dipole carries a purely geometric `(2/3)<1/r>|s|` term that is
largest exactly where the signal was claimed — the core — and falls off like `<1/r>`, so a
rigid offset of the cluster from the grid origin manufactures both a large core amplitude
and an apparent radial confinement. The origin-referenced series is retained only as a
coordinate-drift diagnostic.

Bands are the four **Lagrangian** equal-count (hence equal-rest-mass) quartiles of the
initial isotropic radius, fixed from frame 0 — not the Eulerian current-radius quartiles
the in-code shell ledger uses. A coherent cohort displacement registers fully in the
Lagrangian bands.

`dipoles.csv` adds the `l = 1` dipole **vector** `D_q` of each band about the CoM, which
the specification requires: the sign of `cos(D_q, D_q')` separates a coherent
whole-cluster translation (neighbouring shells aligned) from an internal sloshing mode
(anti-aligned), and the vector is what shows whether the dominant direction locks or
wanders. `|D_q| = A_{1,q}` exactly for this definition and is emitted as a consistency
check; the two agree bit-for-bit.

**On growth rates.** Session 1 withdrew three fitted rates because sliding one-period
windows on the same data spanned `-0.58` to `+1.51` e-folds per period and its
frozen-metric control produced a `-0.85` slope from pure phase mixing. The interim notes
therefore report amplification ratios and window-to-window scatter and explicitly refuse
to fit an exponential; a rate is a claim the final report may make only if the record
supports a sustained law over enough independent temporal samples.

## 10. Storage

| product | per frame | frames per case | per case |
|---|---|---|---|
| `pvtk prtcl_all` (P/100) | 81 MB | 501 | 40.6 GB |
| `rst` (P/4), after pruning | 6.006 GB | ~7 retained | ~42 GB |
| `cart` x4 panels (P/50) | 0.52 MB total | 251 | 0.13 GB |
| `bin` con+tmunu slices (P/25) | 3.5 MB each | 126 | 0.9 GB |
| `cbin adm` coarsen 2 (P/5) | 93 MB | 11 | 1.0 GB |
| `hst` + four CSV ledgers | — | 1001 rows | ~0.06 GB |
| **total** | | | **~85 GB** |

Both cases ~170 GB. AMD user root was 827 GB of the 1.9 TiB effective budget at launch,
shared with four other live campaigns. Raw `pvtk` stays on AMD, where the reduction runs;
Perseus receives the reductions, ledgers, `cart`/`bin` frames, figures, movies and reports
(`/data` has 2.1 TiB free of 110 T at 99 %).
