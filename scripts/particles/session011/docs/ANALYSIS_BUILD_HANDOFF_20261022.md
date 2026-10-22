# Session 011 continuation state

Updated 2026-10-22 00:09 UTC. Goal ACTIVE: finish the one trajectory to t75 with validated AH,
full population accounting, reports/figures/two1080p movies, resource monitoring
and checksum-verified compact archive. No replay, extra seed/control evolution,
physics/grid/dt/pusher/deposition/removal change, or automatic extension past75.
Current turn made progress:4h reservation policy released, analysis build and
18controller/10product tests passed, compact receipts checksum-archived.

## Live handles and immediate next actions

AMD root /work1/eliasmost/jiaxiwu/gi_s011_amd_20261021.
-473967: production segment0004,6nodes/24ranks/cap256, PENDING/Priority at last poll.
  Same continuation from audited8.803125. Do not duplicate or describe as running.
-473989: analysis build COMPLETED/0:0,69seconds/1node.18controller and10product
  tests pass (synthetic full report/two1080p movies/full decode included).
-473992:6node/24rank saved-state AH regression PENDING/Dependency. Verified AND
  dependency after473967 starts and afterok473989. It must exercise unknown IPEN,
  require accepted horizon0 and bitwise particle/grid nonmutation with nlim0.
-474002: endpoint tooling deployment PENDING/Dependency afterok473992. It safely
  defers with exit75 if no initialized production RUNNING interval is available.
  Check RESULT before retrying. No changes occur on deferred guards.
Unrelated ST jobs are not ours to alter. Queue waiting is a verified wait, not a
failed/stopped job. Revalidate these handles; do not restart on observation timeout.

Next: confirm473967 startup from8.803125, AH reacquisition and later audits; inspect
473992 actual zero-step/NaN/accepted-AH receipt; inspect474002 deployment or safely
reschedule deferred deployment; mirror deployed files and compact receipts after
pass. Then keep monitoring to75 and audit actual endpoint science/products/AH.

## Current verified science state

Latest native checkpoint: runs/segment_0003/rst/s11_amd.00005.rst,
t8.803125000000511,cycle1902,60291241129bytes,
SHA2561fd79b708bbb8ba5547d3c0390b914e63e90d802a377f4989d62ea4a9842d3f2.
4198470alive+801530removed=5M; exact IDs/partition/per-tag rest weights and
native/online component Jz pass.330unknown native diagnostic IPEN,165/clump,
explicit unknown rest mass2.1654597172130847e-5 per clump; physical fields finite.
All330 were traced to non-positive-definite high-order metric interpolation;
all330 trilinear interpolants valid, all25 saved grid fields finite,330finite
energy samples match reconstruction within2.84e-15. IPEN is not a pusher/deposition
input. Existing engine remains unchanged. Both individual AHs accepted from
8.4859375/cycle1699; no common AH yet; separation6.06151958 at final sample.
Pair state00004 protected,60301921689bytes.00005 additionally protected as the
irreplaceable NaN regression state in checkpoint_inventory.json; pre-protection
inventory retained in evidence/analysis_preparation_20261021. Raw remainsAMD.
Use current inventory for authorized recovery rotations, not old progress notes.

## Production provenance and reservation

Immutable engine evidence/final_473059/athena_production, source df6ab41d5cdd95c7e33a8b70726891d52691c0cf,
SHA256e4fce096299d9d0ef852e7b713e3636afccb08a1d12a9c44cca9a657b9aa35ba.
Canonical input SHA25611fb21b93abacf02e5410d77c7922967773535e329aabca44939a59a4026ef31.
Live Python correction42823d82 records unknown diagnostic energy without weakening
physical field/ID/rest/Jz checks; product totals propagate unknowns; scheduler
lookup handles aged-out jobs; finalization exceptions publish PAUSED.json.
Release473947 passed full audit/hashes and submitted473967. No PAUSED marker.
Previous gates/control/scripts preserved in evidence/nonfinite_20261021/release_473947.

Pending473967 and future amd_segment.sbatch now reserve4h around unchanged2h
application chunks, based on measured6node totals7382,7356,7413,7421seconds.
Metadata-only change released while own pending job briefly held; executable
script body identical. Receipt in evidence/reservation_4h_20261021/ (also local).
New script SHA25620b05ec10b4f78753d389fc2970712d7a652e8c150944049c36b2890c5daa59f.
Gate allocation_policy preserves timing evidence. Local scripts/ mirrors this.
No source/binary or physical input change. Scheduler estimate remains tentative;
last was23:29CDT Oct21 =04:29UTC Oct22 =21:29PDT Oct21. Do not promise it.

## Analysis-only executable and deployment (NOT yet deployed)

Remote checkout stays5fa3100ae08a855ab8d97e584f9de9b4cc1c7a35. Local branch
experiment/GI-in-cluster-session011 commits through c2ddbbfc pushed; newer changes
are reservation/docs/installer, not the pinned analysis build. Avoid blind syncs.
Analysis binary evidence/analysis_build_473989/athena_saved_ah,
SHA2562f5c21ce1fce83794015884825bcd57719745fcebcab4fd13a366521647c1337.
Only C++ difference from production is frozen_ah_checks.hpp: bitwise comparison
of every active Real handles unchanged NaN and logs unknown-energy coverage.
Do not replace production executable with this binary.
Build receipt/source diff/cache/provenance/test logs copied to local
 evidence/analysis_build_473989 and all7 COMPACT_SHA256SUMS verified.

473992 reads retained00005, full checksum verifies, disables ordinary output/GI
accounting, searches horizon0 six times with require_accept=true/nlim0. Requires
accepted AH, cycle1902 unchanged and unknown_native_energy>0. Paths:
evidence/analysis_saved_473992.slurm.log and analysis_saved_473992/RESULT.json,
command.json,run.log. Read real verdicts; build success is not this regression pass.
474002 runs evidence/deploy_analysis_tooling_20261021.py (versioned under code/
scripts/particles/session011/diagnostics). It reads exact four tested Python files
from Git at receipt source, verifies hashes and production binary, holds submit.lock,
requires an initialized RUNNING production interval without END_EPOCH, preserves
old files/gate, installs endpoint.py/products.py/two tests and SAVED_AH_TOOLING.json,
and updates hashes transactionally with rollback. It preserves4h reservation and
runtime.py/production binary. Endpoint probes then checksum-select the analysis
binary; production continues using old immutable executable. Defer exit75 means
no mutation; retry only after evidence supports a suitable active interval.

## Resources, limitations and remaining completion proof

Production through473355:49.2866667nodeh. Debug/recovery through473947:0.175nodeh.
Analysis build473989 adds69/3600=0.0191667nodeh (total debugging0.1941667).
473967/473992/474002 currently unallocated, no cost yet. Account later allocations
including failures/retries. Pendingjobs charge no node-hours. Original AMD build
costs are additional; final products' corrected AMD discovery excludes foreign IDs.
Perseus sacct is disabled: evidence/perseus_resource_accounting.json (local/AMD)
records unavailable/null actual total and exact error/job list. Never query these
IDs in AMD sacct (18291 collided with unrelated AMD embedding). Preserve limitation.

Perseus fallback considered: node metadata reports8GPUs, preserved CUDA provenance
A10080GiB; /data has13,011,515,977,728bytes available. AMD final-segment telemetry
shows24devices/6nodes used1,120,092,606,464GPUbytes (~1.02TiB), total1,648,864,788,480.
Perseus is not an immediately validated capacity replacement; do not transfer/replay
or assume the same grid fits8GPUs. Direct NVML query outside allocation failed;
that is not evidence of hardware failure. Keep validated AMD continuation.

Original approved contract notes/APPROVED_PLAN.md remains governing for all75
endpoint science hypotheses, exact full-population ledgers, accepted surfaces,
ntheta10/14 saved checks, figures, two1080p movies/full decode and one-way manifest
archive. Completion requires inspecting actual outputs, not merely green jobs.
No Mac raw mirror; no unrelated cleanup. Current whole-user last QA about489.6GB;
watch1.5TiB warning/1.7TiB stop/projected1.9TiB. Archive watcher last receipt18632.
Historical failures/receipts are in SESSION_LOG.md; don't rerun one-off energy
release helper (it pins old segment3/cycle1230). Goal remains ACTIVE/incomplete.
