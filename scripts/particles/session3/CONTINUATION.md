# Session 03 continuation state — 2026-10-05 22:58 UTC

Status: stopped at the user-approved budget discussion gate, awaiting a revised
scope or ceiling. **Do not launch the original production matrix under 200 hours.**
The estimate is 403.07579043557627 AMD node-hours; actual registered use through
preflight reductions is 0.5925 hours. No production gate has been created.

Read campaign/session README, SESSION_LOG and scientific/technical reports.
The prompt and complete Part I derivation are checksum-verified in this root.
All failures and successful initialization data are retained; no deletion is
authorized. Historical shared ../code remains untouched. Anta /data2 reports
zero available bytes: large archive transfer is pending, not accomplished.

## Required user decision

An asynchronous question is pending. Review evidence/budget_options.json:
original matrix with proposed ceiling425 -> 403.08 estimated hours;
main/tail2P and mesh/N controls1P under200 -> 191.69 hours;
main frozen/live pair alone5P under200 -> 165.26 hours.
No option is approved; do not treat silence or elapsed time as approval.
Preserve original inputs/APPROVED_PLAN and record any accepted change explicitly.

## Completed evidence

- Independent C++ vs Python PDF Eq9.7 at70digits: jobs11041/11048; latest
  relative CDF comparison11084 passes354checks, max3.712e-9. Earlier absolute
  CDF report retained in evidence/profile_agreement_11048_absolute_cdf.json.
- P_ref=461.99570043659014M; infinite median r=13.000000002572222M;
  M0_inf=1.0134873008561143. Tail fractions0.014800997%/0.003700288%.
- Mesh inspection11047 passes:624baseline/568coarse leaves.
- Full-N frozen/live CPU initialization11058/11064 passes exact tag/pair and
  continuum sampling checks. New COM quartile ledger build11072/t0 11073/check
  11077 passes. Analytic characteristic bound1.4141532911. An absolute harmonic
  sum test tolerance was too strict; normalized agreement passes1e-10.
- Health/restart adversarial checks11079 pass.
- AMD frozen/live preflights452594/452595 finish healthy at7.8125M=.01691033P,
  cycle100. Exact N counts, mass error1.4287e-13. Verified checkpoint hashes.
  One live Boris fallback atcycle91 is recorded. AMD reductions452625/452651 pass.
  Frozen E RMS1.0493e-9, max1.5040e-8; angular-vector RMS3.7612e-11.
- Budget job452637 intentionally exits2 for403.08>200.
- Initialization figure11078; preflight figure11086 then11087 corrected to
  show histogram bounds and readable axes; budget alternatives11085. PDFs/PNGs viewed.
- Perseus inventory11081 records40.217GB retained raw evidence under runs/.

## Source/build state

Independent src/athenak clone, project/Plummer-cluster, GitHub origin.
Baseline aadffae2b15e6ff59c73d7a5c5327a97572d7c4c; Kokkos
6739bc623081648af9e752b616d9671527922cbf. Focused commits through5c0f38b0
are pushed; subsequent report/budget reproducibility commit is recorded in SESSION_LOG.
Source inventory retains starting identifiers and points to current session records.

AMD root /work1/eliasmost/jiaxiwu/plummer_s03_20261005.
Build452585 at4ebd7565 succeeded, SHA32cc99d530d3defaeada5c4a34489b08ec9c1f11065a7614d580cae8c25ca74d;
both preflights use it. Build452612 at5c0f38b0 succeeds, SHA
792a1de94b231c6289fdc993c19015b42ec64cce26c3abf1c745a84e388da613.
The new COM-band kernel has CPU t0 coverage but no four-rank GPU runtime coverage.
Do not create validated_build.json until that remaining check actually passes.
Use ROCm6.4.1, 4ranks/4MI210, disabled vader single-copy. Devel builds work in
~5minutes. Single-rank full live mesh exceeds devel VM device memory: startup
452606 failed before initialization, preserved/excluded from throughput; do not retry.
No unrelated ST/wave jobs were touched. Failure details are in the technical report.

## Continuation and next steps

AMD scripts/controller.py --watch performs login-safe metadata orchestration only;
numerical work uses Slurm. Its state/controller.json is **stopped** with the
403-hour reason; the service process exited. Do not revive before the budget
decision. Approval must update the decision record, matrix/decks, estimate and
budget_guard's explicit ceiling consistently. The current200-hour default remains.

Then verify new GPU diagnostics and binary; quarter-period matched pilot extended
to.30P for post-transient reference; interrupted/restart continuity; short baseline/
half-dt comparison; causal/diffusive geometry and extraction controls. Scripts/decks
exist but those gates have not run. Review recorded tolerances/outcomes. Never
fabricate production_gate.json or bypass a failed numerical/scientific gate.

Remaining: chosen run matrix; cumulative milestone figures/period notes; physical
profiles/control uncertainty and angular detection tests; two full-run main movies
with fixed scales and conservative cell-to-pixel averaging; final scientific/
technical closeout. Current reports cover initialization/preflights only. Quantitative
judging and rendering beyond the current preflight products still need implementation
and validation before promotion of those products.

Retain whole-root AMD watchdog (warn1.5TiB, stop1.7TiB, forecast1.9TiB), all
milestone/segment checkpoints, verified healthy restart-only resource retries and
no-op endpoint guard. Retry at most once, then diagnose. No checkpoint pruning
or cleanup is authorized. Anta archive requires available space and checksum
verification. Exact paths/sizes/retention remain in reports.
