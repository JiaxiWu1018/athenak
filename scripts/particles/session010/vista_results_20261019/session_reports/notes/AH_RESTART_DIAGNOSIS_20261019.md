# Session010 common-AH restart defect — fix before the next native restart campaign

Recorded October 19, 2026. **Status: defect identified from the as-run source and
native diagnostic sequence; no repair, rebuild, production restart or controlled
replay has been performed.** This is a restart/initial-guess defect in the
Session010 AthenaK/NRPIC branch, not evidence that the black hole disappeared.

## Observed failure at the 98.2M chunk boundary

The affected boundary is the normal application restart from chunk_004 to
chunk_005 of recovered `runs/20261016_gh_resume_01/segments/segment_002/`.
It is distinct from the earlier telemetry-failure/recovery episode. The first
post-restart candidate remains good, so the records do not implicate corruption
of the earlier failed segment or of the checkpoint payload.

Raw `out/s10_orbit_gw.horizon_consumer_2.csv` gives:

| Native cycle | Time/Mref | Minimum radius/Mref | hrel | Quality streak | Published |
|---:|---:|---:|---:|---:|---:|
| 59131 | 98.198437500082889 | 0.5440166248 | 0.0021997429 | 8337 | yes |
| 59132 | 98.200000000082895 | 0.5440179472 | 0.0022120913 | 1 | no |
| 59133 | 98.201562500082900 | 0.4002129922 | 0.2809919004 | 0 | no |
| 59134 | 98.203125000082906 | 0.4002130109 | 0.2810113116 | 0 | no |

The first post-restart candidate passes geometry/association and the unchanged
`hrel<=0.2` limit. It is withheld solely because `consumer_persist_2=3` requires
three successive acceptable candidates and the new process starts its streak
at zero. The next candidate starts near the configured `initial_radius_2=0.40`,
instead of continuing the good radius near 0.544. The later final common
candidate has hrel=0.5146317031 and remains unpublished.

## Cause in the compiled finder

The inspected `src/z4c/fastflow.cpp` and `.hpp` are byte-identical between native
compiled source `386870f88f60e25e1a8fac76fe04a071b87a01cf` and inspected
documentation HEAD `790386988625240f2a081f5b79d38ff6e2bcaa91`.

1. `FastFlow::Find`, lines 621–635 and 683–688: strict publication waits for the
   persistence streak, but **every unpublished candidate clears `last_a0`**, even
   when that candidate passes all instantaneous quality checks and is merely
   waiting for the three-hit streak. Search warm-start validity is conflated with
   permission to publish a measurement.
2. `FastFlow::InitialGuess`, lines 744–810: full saved/restart surface reuse is
   implemented only inside `use_puncture>=0`. The common horizon uses
   `use_puncture_2=-1` and a mass-weighted center, so it skips that branch despite
   `reuse_last_surface_shape_2=true` and `persist_surface_shape_2=true`.
   Once `last_a0` is cleared, it returns to the input 0.40 spherical guess.
3. `FastFlow::FastFlowLoop`, lines 1072–1104: the flow may declare `ah_found`
   when irreducible-mass changes fall below `mass_tol_2=1e-4`, without requiring
   the publication residual to pass. `hmean_tol_2=100` is only a loose failure
   guard. The smaller seed can therefore stop at a poor candidate; rejection
   clears its seed again and repeats the same failure path.

This source path explains the observed good-candidate → streak-reset → discarded
seed → 0.40-radius/rejected-candidate sequence. A controlled replay is still
required to demonstrate that a proposed patch restores accepted common surfaces
and preserves the native evolution and valid measurements. The existing late
mass/spin values must not be promoted by loosening the quality threshold.

## Required repair and validation before another restarted production run

- Separate a finite, on-grid, quality-passing **search guess** from a published
  consumer surface. Retain the good provisional guess while the publication
  persistence streak rebuilds; do not bypass the three-hit publication gate.
- Apply serialized/full-shape warm starts to the mass-weighted common-horizon
  branch as well as individual puncture branches, with correct center handling.
- Review the flow stopping criterion so mass stagnation alone is not treated as
  a validated AH. Preserve geometry, resolution, association, persistence and
  `hrel<=0.2` publication gates.
- Add targeted restart regressions for individual/common surfaces and for a
  valid candidate waiting on persistence. Compare uninterrupted and restarted
  searches, including shape/center, radius, residual, publication timing and
  mass/spin. Check that bad guesses remain unpublished.
- Replay a short interval across the actual t≈98.2 checkpoint with frozen
  original and proposed finder versions in allocated compute jobs. Bind source,
  executable, exact input and checkpoint hashes, and inspect native residuals
  and consumer flags. Bulk checkpoints are absent from the Perseus summary;
  locate the retained Vista checkpoint through its raw inventory before planning
  this test. The final t≈221.95 checkpoint alone cannot reproduce this boundary.

No fix is implemented by this documentation update. Do not restart production,
relax gates, or reinterpret rejected late candidates as validated horizons.

## Evidence and scope

Native raw text copied in the immutable supplied summary is the evidence source:
`session010_summary_20261019/workspace/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/`.
The preserved excerpt, exact paths, source/input hashes and source comparison are
in `evidence/ah_restart_diagnosis_20261019/RECEIPT.json` and accompanying files.
Only short source/text inspections were needed; no simulation or bulk reduction
was run. Original summary, native outputs and prior reports remain preserved.

For the separately requested matching 3D movie, use **0<=t/Mref<=98 only**.
The movie ends before the failure boundary and does not require a finder repair.
