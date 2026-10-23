# Session 11 preparation report

Prepared 22 October 2026 CDT, allocated Vista GG job **1091077**, host
`i614-071.vista.tacc.utexas.edu`. This report, not Session 10's task log, owns all
new preparation. Latest instruction: **finish code and compilation only; launch
all physics/science in a separate Codex chat opened in this folder.**

## Submission launcher implementation (preparation only)

Implemented the discussed `submit_s11.sh` launcher, backed by versioned tools in
`code/scripts/particles/session011/`. Commands are `check` (also the default),
`validation`, and `production`. Nothing has been submitted or simulated by this
implementation chat; no GitHub push has been attempted. Current preparation uses
allocated GG job **1091475** on `i614-064.vista.tacc.utexas.edu`.

The OS/spin input and generator typo `x2` is corrected to `nx2`. Production keeps
all frozen physical/mesh/removal settings; its checkpoint `last_time` changes
from 0 to -1 to retain t0 and restore the first periodic checkpoint at t10. The
native driver writes initial output unconditionally, so the old value advanced
the next checkpoint target to t20. No native C++ source changed or rebuild was
needed. All three executables retain their verified build hashes and native tree
`d8d40c41cd60852058a9b3f39021ffd9c2dea8f1e90e618da09ecfda267a1fce`.
Current production input SHA256: `b402a20c6ed7306286bd458418d79ffe212c5ed499c40d68a83d77f724e5ae8c`. The initial PREPARATION.json,
PREPARATION_FINAL.json and SOURCE_TESTS.json remain historical evidence. Consult
`evidence/SUBMISSION_PREPARATION.json` for this implementation's current manifest,
check results and commit; neither record is a GPU validation receipt.

Validation remains one four-node/two-hour gh-dev allocation. Every case must
reach its native checkpoint endpoint. Its checks now require actual diagnostic
contents/cohort identities, strict accepted-horizon quality, three fresh finds,
bad-fallback restoration and exact-cycle uninterrupted/restart comparisons.
Real future validation receipts bind the exact source, binaries, all input decks,
workflow tools, command overrides and retained output/log/checkpoint checksums.
The conditional fast-forward push and production readiness reject stale/legacy
receipts, changed executables and a changed/unpushed/dirty checkout. Existing
upstream commits are preserved.

The launcher checks TACC's refreshed local project/membership/balance maps,
current Slurm partition limits, existing Session 11 jobs, modules and storage.
A controller-side test-only request precedes the single requested submission.
It uses inherited S11_ROOT and plain sbatch. Submission is restricted to Vista
login nodes. An intent is recorded before actual sbatch; ambiguous responses,
previous attempts or existing runtime folders block another submission pending
explicit review. Attempts/job IDs/Slurm log and output paths append here. There
is no automatic retry, production chaining, deletion, transfer or push.

Production remains one sixteen-node/36-hour fresh run to strict t150, with one
rank/GPU, unchanged 85% host/GPU memory and 5,760-leaf guards, 150 GiB write reserve
and forty-minute allocation reserve. Startup now checks the actual native t0
particle states/weights/tags/GID ranges and component census, 4,936 blocks,
retained model profile and all-node telemetry. Failure reasons remain latched.
Completion requires runtime telemetry, unchanged hashes, finite native final
fields/particles and an output-coverage audit. The audit checks native layouts,
all prescribed output families/cadence, rank cohorts, removal ledger, wave
state-time mapping and unique angular ownership of published AH quadrature.
AH accepted gaps, conditional common-finder absence, sparse cohort rows and the
measured waveform endpoint gap remain explicit. This is not an independent
scientific validity or numerical convergence claim. Runtime output sizes/free
space/rate are recorded; inside-allocation ACCOUNTING.json is provisional until
final Slurm accounting is inspected after the job exits.

The refreshed TACC maps inspected here show AST21006 membership and **2960.289 SU**
available; submission rechecks the maps, dates and sufficient balance. The shared
scratch filesystem still reports about 2.9 PiB available, not archival protection.
Validation requests at most 8 and production at most 576 GH node-hours. Production
preflight reserves 3 TiB output headroom plus the 150 GiB write reserve. Live
scheduler access is blocked by this chat's network sandbox; existing jobs and live
queue limits cannot be certified here. Recheck on the login node. No new job IDs
or actual science accounting exist from this chat. Native GPU validation, remote
push, full-size memory behavior, actual output coverage and scientific analysis
remain pending; all failures and raw outputs must be retained when executed.

Non-science verification: **37 tests passed**, covering the original eight checks,
mocked scheduler submissions/ambiguity, malformed inputs, retained-artifact
changes, native endpoint/persistence gates, finite particle/tag/weight/GID checks,
partial diagnostics, truncated binary/VTK layouts, angular ownership, removal
ledger and completion prerequisites. Python/shell syntax and git diff checks pass.
Mocks and synthetic temporary files are not GPU validation evidence; no passing
GPU/push receipt has been created. Detailed logs are retained under
`evidence/preparation_checks/` and referenced by SUBMISSION_PREPARATION.json.

## Original handoff status and authority

Final local preparation commit: **356ea67c** on `project/GI-in-cluster`.
All three binaries match the final native-tree hash and all eight non-science
checks pass. The checkout is clean. This commit is deliberately **not pushed**:
real native GPU/science validation belongs to the next chat and remains pending.
No simulation/build process is intentionally left running after handoff.

The independent checkout is `code/`, branch `project/GI-in-cluster`, based on
GitHub tip `cb394a814e7a403a47f8c14cb7b35bf90a85ce0f`. Four upstream commits add
Session 10 reports/movies without changing the native source from
`386870f88f60e25e1a8fac76fe04a071b87a01cf`. Kokkos is pinned to
`6739bc623081648af9e752b616d9671527922cbf`. Existing remote work is preserved.

No OS collapse, spin calibration, compact GI evolution, production, archival
transfer or deletion is authorized/performed in this preparation chat. No Slurm
job is submitted by these preparation tools. No GitHub push is allowed yet:
the user requires native simulation validation to pass first. Source/static
tests and successful compilation are explicitly not those simulation tests.

Compilation receipts/logs are in `evidence/build/`: GI, OS and Kerr/spin use
separate native ARM/Grace/Hopper executables and the same repaired common source.
Check actual receipt existence, executable SHA256 and native source-tree hash;
do not infer compilation success from the existence of a build directory.
The immutable initial input comparison is recorded in `evidence/PREPARATION.json`.
Final preparation checks are in `evidence/PREPARATION_FINAL.json`; eight ordinary
unit/source tests, including synthetic complex-wave integration and exact Jz
algebra, are recorded in `evidence/SOURCE_TESTS.json`. None evolves a spacetime.
The source and compilation preparation is distinct from the intentionally pending
physics validation. A final incremental compile pass binds all three binaries to
the same final native-tree hash after code editing has stopped.

## Agreed experiment and finite resources

Repeat Session 10 fresh from t=0: exactly 5,000,000 particles, source masses
0.76/0.12/0.12, initial centers ±3, Lorentz boost ±0.133215, widths 0.70, thermal
dispersion 0.02, seed 4001. Preserve the ±512 domain, root256³/dx4, block32³,
eleven root-inclusive levels, initial 4,936 blocks, initial central dx1/64,
ceiling dx1/256 and the inherited fine wave-zone floors. Preserve gauge,
RK4/CFL, conservative deposition, GR Boris guards and lapse removal alpha<0.05;
AH removal remains OFF. Initial local momentum constraints remain unsolved.
Source/reference/rest/AH masses are different quantities.

Production: **one sixteen-node, thirty-six-hour GH allocation, one rank/GPU per
node, hard tlim=150 Mref**, nlim=-1. No planned application restarts. A periodic
checkpoint write does not restart the program. Forty minutes of allocation
reserve are kept for clean stop, checkpoint checks and final reporting. If a
resource/wall stop occurs before 150, report incomplete and do not auto-resume.
Keep 85% GPU/host memory and 5,760-leaf global capacity-proxy stop guards; monitor
the long continuous run, which is not certified by the shorter Session 10 chunks.

Validation in the next chat: one four-node, two-hour gh-dev allocation for compact
OS collapse/reference/restart, a bad-fallback-guess regression, compact GI output
smoke and spin/angular-resolution checks. At most 8 validation plus 576 production
GH node-hours requested; GG preparation/analysis is accounted separately.
The full 3,984-block late Session 10 checkpoint does not fit the unchanged
eight-node gh-dev capacity (8×400=3,200). Do not load it there or raise capacity
without a separately approved memory gate.

The inherited account is AST21006; mail BEGIN/END/FAIL goes to the previously
user-verified `jiaxiwu@caltech.edu`. Verify membership, balance, modules and live
queue rules again before submission. Scratch free space was about 2.9 PB at
inspection; that is not independent archival protection. No AMD storage caps
are imported. Rich raw output plus retained native checkpoints is expected to
be order 1–3 TiB; measure actual sizes/output rate and destination headroom.

## AH defect and implementation

Session 10's last accepted common surface is near 98.1984375 M. Immediately after
the 98.2 restart, a candidate with radius ~0.544 and hrel~0.00221 is good but only
the first of three required successful finds. The old code discards its seed
because it is not yet published; the next search falls back to radius0.4 with
hrel~0.281 and keeps failing. Restored full shapes were ignored by the common
branch (`use_puncture=-1`). Mass stagnation could also terminate an invalid flow.

The patch separates provisional search-coefficient buffers from authoritative
consumer buffers, retains an on-grid instantaneous-quality-passing guess during
the persistence streak, and reuses full shapes for individual/common/fixed-center
searches. Publication remains false after restoration until three new successful
finds pass the existing gates. The mass-stagnation exit additionally requires
the unchanged normalized expansion limit. No rejected late Session 10 candidate
is retrospectively promoted, and no physics/removal gate is relaxed.

New AH tables distinguish current `found`/`published` from sticky consumer state,
record full-precision physical state time/cycle, and put NaNs instead of stale
properties on failed attempts. Full-precision coefficients and rank-owned surface
quadrature (angles, radii/derivatives, metric, K and metric derivatives) accompany
them. Merge quadrature by state/angle with ownership checks, not by summing copies.
Native legacy files are retained; their timestamps/first-column conventions
must not be confused with the new state-keyed diagnostics.

## Frozen output contract

Inputs and scripts live in `code/scripts/particles/session011/`.

| Product | Interval |
|---|---:|
| New AH, component/cohort/profile/tracker diagnostics, history and mesh metadata | 0.025 M |
| Complex raw rPsi4, l2…8, at R40/R50/R60 | target 0.025 M |
| Matter, metric, constraints and Weyl: xy/xz/yz planes | 0.25 M |
| Full-particle VTK snapshots | 0.5 M |
| Full 3D fields and non-stopping safety checkpoints | 10 M |

All initial/final states are retained. Native periodic outputs occur at completed
timesteps, not synthetic interpolated times. GW remains legacy rPsi4; the new
`wave_state_times.csv` maps its saved legacy time to the completed RK state's
actual time/cycle. Raw GW may end up to one nominal interval before the endpoint;
record measured coverage rather than claiming an exact t150 wave sample.
Full-precision native checkpoints, not float32 particle VTK, are the restart truth.

New particle files include component counts/rest mass/Jz, unnormalized COM and
momentum sums, and radial profiles in coordinate shells dr0.1 from R0 to50.
Profiles contain coordinate/covariant-momentum diagnostics, not invariant enclosed
gravitating mass or velocities. Particle mass/Jz outside R50 remains in the global
component budget. A fixed immutable tag cohort has 4,000 tags per component,
never replenished after removal. Cohort update records sum actual accepted
per-step delta x and covariant delta u, including a last update before removal.
They are coordinate force/motion proxies, not a unique split into BH, envelope,
gauge or radiation forces.

For EVERY live particle before excision, a separate diagnostic kernel accumulates
the exact discrete product-rule decomposition of delta[mu(x*u_y-y*u_x)] into
midpoint-position×momentum-change and midpoint-momentum×position-change terms.
It does not change the pusher. The native event ledger independently records
removal mass/Jz. Use alive Jz changes + removed Jz changes versus the push ledger;
do not interpret removal alone as a physical torque or BH accretion.

## Scientific investigation and paper requirements

1. Noncircular motion: separation, accumulated phase (gaps excluded), Omega,
   radial/tangential motion and orbital/inward-motion timescales; compare tracker,
   accepted-AH and particle-center definitions. Stop binary curves at a common
   surface strictly enclosing both live tracked objects.
2. Changing BH masses: accepted AH mass/spin and M/|dMdt| versus orbital and inward
   timescales. AH growth is not particle rest-mass accretion.
3. Matter interaction: surviving/removed component mass/Jz, matched surviving tags,
   envelope structure, wakes and redistribution; separate momentum evolution,
   changing membership and position terms. Matter-only Jz is not total GR J.
4. Orbital support: evolving profiles versus t0, explicitly labelled spherical,
   coordinate, rest-mass-based and/or Newtonian approximations. The isolated circular
   vacuum estimate can give tens of turns when separation6 Mref is ~25 pair-mass
   units; it does not predict this collapsing, matter-filled, constraint-imperfect
   run. Do not dismiss the discrepancy merely by calling separation6 “small”.
5. Numerical effects: proper-volume masked constraints, AMR geometry/local dx,
   timestep changes, fallback counters, tracker quantization, accepted-AH gaps and
   any actual restart boundaries (none planned for production).

Paper products: 3D fixed-identity particle views/movies, AH mass/spin, orbital and
matter-budget panels and finite-radius strain. Current spin is a coordinate
prescription, not a newly implemented approximate-Killing-vector measurement.
Preserve quadrature for independent postprocessing. Strain requires analytic
plus/cross sign tests, finite-length/window/cutoff sensitivity and radius/matter
checks. R60 has a shorter retarded-time tail than R40 at the fixed150 endpoint.
Do not claim complete ringdown, DC/memory recovery, waveform at infinity, numerical
convergence or a unique causal explanation from this single production run.

## Next chat: required order

1. Read this report and AGENTS.md; inspect compilation receipts/source hash and
   run the non-science unit/source checks. Build/rebuild stale binaries on GG.
2. On explicit user instruction, submit `validate.sbatch` from a login node with
   `S11_ROOT` exported. Retain the real job ID and all native output/failure files.
   Passing source tests is NOT a replacement. Fix any failure and recompile;
   do not silently bypass the two-hour test limit or falsify receipts.
3. Inspect OS collapse, reference/restart continuity, the deliberately poor fallback,
   three-fresh-hit persistence, spinning mass/spin and angular-resolution checks.
   A compact fixture is not a replay of Session10's actual late background.
4. Only after required tests pass, commit/push to GitHub's existing campaign branch,
   rechecking its tip and refusing non-fast-forward updates. Record the verified
   remote commit and native source hash as `evidence/PUSH_VERIFIED.json`.
   `push_after_validation.py` implements this explicit hash-checked conditional
   push and receipt; it refuses dirty work or a non-ancestor remote tip.
5. Check production input/source/executable hashes, account/balance/output headroom
   and all guards. Give the user the single login-node submission command.
   Production does not submit from this preparation chat or automatically after tests.
6. Require actual full-size t0 particle census/model and all-node memory/output
   gates in the production allocation, record true field/tracker/wave coverage,
   and inspect AH and resource health without loosening quality thresholds.
7. After completion, seal raw outputs, independently audit the final native grid,
   tags/weights/GIDs and t150 checkpoint, perform the five investigations and
   render/review the paper artifacts. Archive with destination checksums only on
   new user authorization; never delete Session10 or Session11 originals here.

## Session 10 evidence locations

Original workspace:
`/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff`.
The immutable `session010_summary_20261019/` beneath it has history, both analysis
releases, movies/figures/reductions and exact raw-file inventories/checksums.
The analyzed branch ends at221.953125; its common-horizon quality ends98.1984375.
Its main R40 burst peaks~114.94, supporting the agreed150 endpoint for capturing
the main burst but not an assertion of a complete weak late-time tail.

The actual defect-boundary checkpoint is under
`runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_004/out/rst/s10_orbit_gw.00012.rst`
in that ORIGINAL workspace (98.2M; not copied here). The final221.95 checkpoint
is not a substitute for reproducing the 98.2 boundary. Session10 raw data and
its summary are not modified, deleted or copied in bulk by this preparation.
