# Continuous production policy, user instruction 2026-10-21 PDT

The user explicitly replaced two-hour application chunks: use the longest AMD
wall time, write checkpoints during evolution, and keep checkpoint analysis from
interrupting production. Partition mi2104x advertises MaxTime=4-00:00:00 and the active account
QOS has no additional MaxWall. However, actual sbatch --test-only rejected96h:
AMD submission filter enforces max24h. New launches reserve24h and permit23h20m
application time, leaving40m for checkpoint/finalization. This is the longest
request accepted by the submission policy. Every launch still enforces tlim75.
The existing process474260 has a fixed2h limit already passed to its Driver; it
is left running and allowed its normal checkpointed exit. No debugger/process
mutation, cancellation, replay or alternate trajectory is used.

New operational output overrides set output16/dt=1Mref and reset its last_time
to the exact restart time. The executable, canonical input, physical parameters,
grid/dt and finder settings remain unchanged. Milestone states remain protected.
A spare CPU within the production allocation audits stable completed checkpoints
only after a later simulation cycle is published. Exact alive/dead IDs, per-tag
weights, finite physical fields, component Jz and native census remain mandatory.
Death-log entries after each checkpoint time are excluded. Online profile matching
is required when that exact cycle was sampled; unsampled periodic cycles explicitly
record online_profile_compared=false. Final boundary audits retain strict online
matching. Live audits never modify the boundary cumulative dead-tag ledger.

Retention inventories before deletion and checksum-verifies both surviving recovery
copies, preserving initial/science/regression/final states. Failed live audits retain
raw evidence, publish a review receipt, and never issue REQUEST_STOP. Production
still has its memory/storage watchdog and hard t75 endpoint. The background auditor
finishes any current audit before finalization; no evolution is paused for analysis.

Source checks and compute-node regressions precede transactional runtime deployment.
The running application's executable and launched arguments are never modified.

Initial QA474345:21 existing tests passed; one new test relied on filesystem
mtime changing for same-size rapid writes. The fixture now changes size explicitly.
No runtime deployment or production interruption occurred during this failed QA.
