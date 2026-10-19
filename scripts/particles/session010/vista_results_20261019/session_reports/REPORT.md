# Jeans / GI-in-cluster Session010 — Vista results

Reconciled on Perseus on October 19, 2026 UTC (October 18 PDT), from the supplied
`session010_summary_20261019`. This report supersedes the preparation-only status;
the original preparation reports and Vista releases are preserved.

**The run reached t=221.953125 Mref and produced two individual apparent horizons,
followed by a common horizon after a short inward orbit and plunge.** The measured
pre-common motion is about 0.67 turns from t=0. The main raw R40 wave burst has
arrived. The late common-horizon finder is failing its quality gate, so final
remnant mass, spin and a validated ringdown remain unresolved.

![Scientific overview](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/plots/through_segment002_20261018/00_scientific_summary.png)

[Overview PDF](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/plots/through_segment002_20261018/00_scientific_summary.pdf) ·
[All figures](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/plots/README.md) · [Movies](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/README.md) ·
[Technical report](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/REPORT_AGENT.md) · [Exact configurations](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/config/README.md)

## What was simulated

Session010 starts fresh at t=0 in [-512,512]^3 with a 256^3 root grid, retaining
five million particles, boost magnitude 0.133215, inherited physics and the
central refinement ceiling dx=1/256. The initial hierarchy has 4,936 leaf blocks;
its finest initial spacing is 1/64. The wave mesh retains dx<=0.25 through R46,
with coarser floors 0.5/1/2 through radii 72/104/168. R40 is the sole extraction
radius. This is the approved exploratory domain choice, with no demonstrated
small boundary-contamination amplitude.

The original envelope contains 3,000,000 particles and each clump 1,000,000.
Source masses are 0.76/0.12/0.12 in inherited Mref=1 units, distinct from particle
rest masses and measured horizon masses. The inherited initial local momentum
constraints are unsolved. Equilibrium and quasi-circular initial motion have not
been demonstrated. See the [actual production input](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/inputs/vista_production/production.athinput).

## Scientific measurements

| Event or measurement | Result in Mref units |
|---|---:|
| First accepted individual horizon, original left | t=8.434375 |
| First accepted individual horizon, original right | t=8.4859375 |
| First accepted common surface strictly enclosing both live trackers | t=64.36875 |
| Last pre-common tracker sample | t=64.3375; separation 0.2358353 |
| Observed rotation before common enclosure, excluding gaps | 241.3526 degrees = 0.670424 turns |
| Observed rotation after both individual horizons appear | 0.625921 turns |
| Last accepted common-horizon measurement | t=98.1984375; mass 0.636965; coordinate-prescription spin 0.174626 |
| Largest raw abs(rPsi4_22) at R40 | t=114.9390625; amplitude 0.0413465 |
| Last raw R40 sample | t=221.940625 |
| Final native checkpoint | t=221.953125; cycle 138,334 |

The common-horizon time uses the same output case, cycle and time for both live
trackers and requires their enclosure by the accepted common surface. The first
raw accepted common-finder row at t=64.309375 does not yet meet that enclosure
test. There are 1,081 strict simultaneous enclosure samples in the retained
data. Post-common coincident trackers are not treated as a continuing binary.

The coordinate separation falls from 5.96877 initially to 0.235835 before common
enclosure. Rotation is accumulated only within valid intervals; restart and
sampling gaps are excluded. The endpoint principal-angle change of -116.715
degrees is not the number of orbits. Together with accepted surfaces, the tracks
support collapse, two black holes, short prograde inward motion/plunge and a
common-horizon merger interpretation within this numerical model. They do not
establish a long, quasi-circular or radiation-driven inspiral.

After the restart near t=98.2, common candidates stop being published under the
unchanged hrel<=0.2 quality/persistence gates. The final candidate has
hrel=0.514632. The last accepted mass/spin above are intermediate measurements,
not final remnant values. Rejection does not show that the black hole disappeared.

### AH restart defect: fix and validate before the next restarted production run

The October 19 read-only investigation identified a concrete finder-state defect
in the compiled Session010 AthenaK/NRPIC branch. The first post-restart candidate
at t=98.200M is good (hrel=0.002212, radius about 0.544M), but its three-hit
publication streak has reset to one. The strict gate then clears the search
seed while waiting for persistence. The common-horizon branch skips saved-shape
reuse and returns to its 0.400M input guess. The next candidate has hrel=0.280992,
fails quality, and repeats the same reset path. A mass-stagnation stopping test
can declare a finder candidate without satisfying the publication residual.

This explains the observed restart failure sequence and is distinct from the
earlier broken-segment telemetry/recovery episode. The [detailed diagnosis and
required repair/validation](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/notes/AH_RESTART_DIAGNOSIS_20261019.md) bind raw
consumer rows to the unchanged compiled finder source. **The defect is identified,
but no fix or controlled replay has been performed.** Preserve the quality gates,
separate search warm-start state from publication state, support common-horizon
shape reuse, and validate an uninterrupted/restarted comparison across this
boundary before another restarted production campaign. Late remnant properties
remain unvalidated.

There are 8,734 paired finite raw wave samples, containing 77 complex modes
l=2..8 at R40. The largest time interval is 0.05, twice the nominal 0.025 cadence.
The main raw (2,2) burst has more than 107 Mref of coverage afterward; another
full 24-hour allocation is unnecessary solely to wait for its arrival. The plot
is finite-radius raw rPsi4, with matter/gauge and resolution caveats. No strain,
extrapolated waveform, radiated-energy budget, final spectrum or validated
ringdown is inferred from it.

## Matter, resolution and numerical health

The final count is exactly **17,073 active + 4,982,927 low-lapse removals =
5,000,000 initial particles**. All surviving particles belong to the envelope;
both original clump populations have been removed by the alpha<0.05 rule.
Removal accounts for 99.65854% of initial tags. Apparent-horizon removal is off.
This accounting is not an independent measurement of accretion across horizons.
Matter angular-momentum curves are matter-only ledgers, not a total matter plus
gravitational-field angular-momentum balance or a GW loss measurement.

The preserved October 18 native checkpoint audit found finite full-grid fields,
unique active tags, positive particle weights, valid mesh IDs and exact census.
The final mesh has 3,984 leaf blocks, dx_min=1/256, and the R46 wave floor is
preserved. Final proper-volume constraint RMS values are H=3.22846e-6 and
M=1.72857e-6; their recorded maxima are 4.08127e-4 and 1.91797e-4. These norms
use chi>=0.0625, **not an apparent-horizon exterior mask**. Neither finite fields
nor AMR activity establishes convergence or late physical accuracy. The bulk
checkpoint is absent from this Perseus summary, so its native audit is preserved
Vista evidence rather than a new local checkpoint inspection.

During the recovered segment, recorded peak GPU and host fractions were
67.5869% and 47.7813%, below the 85% stop guards. Twelve native chunks stopped
cleanly. The repaired source and finite checkpoint segmentation enabled useful
evolution far beyond Session009's t=9.33125 memory stop. This is evidence for
the tested Vista configuration and finite segments; it does not establish zero
remaining growth, an unlimited safe run, or a controlled cross-machine causal
measurement of the buffer repair.

The earlier flat-far-field travel estimate from the nearest box face to R40 is
approximately 334 Mref for a sqrt(2) gauge speed and 472 Mref for unit light
speed; to the center the corresponding estimates are 362 and 512 Mref. These
assume an initial boundary disturbance, fixed coordinate speeds, shortest axial
travel and approximately flat far fields. The present endpoint precedes those
estimates, but they are not a rigorous causal isolation result or a bound on
contamination amplitude.

## Outputs and present status

The [current plots](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/plots/README.md) contain seven original October 18 PNG/PDF
figure pairs and the new independently computed overview. Three [current movies](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/README.md)
show matter slices, collapse/AMR health and a fixed 90,000-particle cohort. Field
movies use 445 saved slices and the particle movie 112 saved states. Movies hold
actual saved states, preserve fixed scales and identities, and never replace
removed tags. A vanishing clump sample is not a vanishing black hole. White energy
pixels represent nonpositive sampled E, not absent geometry.

A separately requested standalone 3D particle/AH movie must stop at **t=98Mref**.
The [copyable rendering request](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/notes/MOVIE_3D_TO98_REQUEST_20261019.md) matches
the earlier Session007 dark-background movie and excludes t=98.2 and later data.
This documentation update has not generated that movie.

The first analysis through t=36.55 is [historical](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/archive/vista_first_production_20261015/README.md).
Its AMR display had an extra root-level subtraction. The October 18 renderer
corrects displayed relative levels to 0..10; the native mesh and evolution were
unaffected. Historical files remain intact and are not the promoted release.

According to the supplied scheduler snapshot at **2026-10-19 05:27:04 UTC
(October 18, 22:27 PDT)**, job 1075832 completed and successor 1075833 was
cancelled before allocation. There is no third-segment evolution in this
archive. The older copied health page saying the successor was queued is stale.
This reconciliation did not query live Vista, restart production or modify jobs.

The summary was fully checksum-verified on Perseus: 4,044 listed files, 8.681 GiB.
All six current/historical copied movies fully decoded, and the new reduced-data
recheck independently reproduced common enclosure, rotation, wave peak and
particle accounting. See [verification](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/COMPLETE.json)
and [numerical results](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/SCIENTIFIC_RECHECK.json).

**Raw-data preservation is still separate.** The October 19 inventory records
2,114,843,497,546 bytes of raw Vista outputs, including validation and historical
branches. Bulk checkpoints, field binaries and particle VTK were omitted from
the summary; there is no proof of an independent Anta archive in the supplied
evidence. The [exact raw paths and checkpoint receipt](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/session010_summary_20261019/reports/RAW_DATA_PATHS.md)
must be used for a separately authorized archive operation with current storage
and purge checks. No raw transfer or deletion occurred here. Preserving those
data and repairing/replay-validating the documented AH restart defect are the
main remaining tasks.
