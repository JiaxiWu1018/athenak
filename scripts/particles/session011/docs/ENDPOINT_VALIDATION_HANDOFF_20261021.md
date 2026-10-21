## Endpoint validation prepared — 2026-10-21 16:50 PDT

Goal ACTIVE. This turn made progress: implemented checksum-bound analysis-binary
selection and tamper/receipt tests, prepared a real saved-state zero-evolution AH
regression, and queued guarded deployment. Last authoritative queue poll: production
473967 PENDING/Priority, analysis build473989 PENDING/Priority, saved check473992
PENDING/Dependency, deployment474002 PENDING/Dependency. No new simulation time
or additional allocated node-hours yet. Keep polling these exact handles; do not
resubmit because of queue waiting. Unrelated ST jobs are not ours to alter.

New code committed/pushed through7d4e9950. AMD checkout remains pinned to
5fa3100ae08a855ab8d97e584f9de9b4cc1c7a35 for build473989; do NOT update remote HEAD
until this build has started/completed its source guard. Live production scripts
and immutable production executable remain the released42823d82/df6ab41d versions.
No endpoint-tooling deployment has occurred yet.

473989 (devel,1node,30min bound) runs18 controller tests and10 product tests,
including full synthetic report/movies/decode, then incrementally builds the
analysis-only binary from existing cache. It asserts the only C++ diff from the
production engine is src/z4c/frozen_ah_checks.hpp. Output:
AMD evidence/analysis_build_473989/{RESULT.json,athena_saved_ah,provenance.txt,
analysis_only_source.diff,CMakeCache.txt,runtime_tests.log,product_tests.log,build.log}.
The production executable evidence/final_473059/athena_production is not replaced.

473992 (6nodes/24ranks,30min bound) has verified Slurm AND dependencies
`after:473967,afterok:473989`: production must start first, and build/tests must pass.
It reads retained state00005 at t8.803125, verifies its full hash, runs6 frozen
searches of horizon0 with nlim0 and ordinary outputs/GI accounting disabled, and
requires accepted AH, bitwise particle/grid nonmutation and unknown-IPEN count>0.
No evolution/replay. Check evidence/analysis_saved_473992.slurm.log and
analysis_saved_473992/{RESULT.json,command.json,run.log}. Regression checkpoint00005
was marked protected in control/checkpoint_inventory.json (60,291,241,129bytes),
with pre-change inventory preserved in evidence/analysis_preparation_20261021/.
It must remain available even if ordinary recovery rotation advances.

474002 (devel,1node,5min bound) depends afterok473992. Installer is
AMD evidence/deploy_analysis_tooling_20261021.py, versioned locally under
diagnostics/deploy_analysis_tooling.py. It reads exact tested Python files from
Git at the build receipt's source, verifies all hashes and unchanged production
binary, locks control/submit.lock, and requires current production RUNNING with
initialization complete and no END_EPOCH. It then snapshots old files/gate, installs
only endpoint.py/products.py/two tests and control/SAVED_AH_TOOLING.json, and updates
the gate's hashes. Rollback restores all prior files on write failure. Exit75 with
RESULT.status=deferred means no changes: resubmit only after checking a suitable
active production interval. Other failure/unsatisfied dependencies need diagnosis,
not blind retries. Live runtime.py and production executable are never replaced.

Perseus accounting query proved Slurm accounting storage is disabled. Local and
AMD evidence/perseus_resource_accounting.json record status accounting_unavailable,
null node-hours and exact local job list/error. Never resolve those IDs with AMD
sacct. The pending products fix excludes foreign-cluster receipt IDs; AMD totals
remain measurable. The final report must disclose this Perseus limitation and
use available local provenance for timing bounds, rather than inventing a total.

Next: inspect473989 build/tests; confirm473967 restart8.803125/AH reacquisition;
inspect473992 genuine NaN regression/accepted AH; inspect474002 or safely defer it;
archive compact receipts and mirror deployed Python files only after its pass.
Then continue audits/resources through75 and complete all requested endpoint
science/report/figures/two1080p movies/manifest archive. No extension beyond75.
