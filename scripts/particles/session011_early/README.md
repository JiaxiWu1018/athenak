# User-authorized Session 11 production and repaired builds

The user approved starting production before compact GPU validation and a verified
GitHub push, accepting later manual repair/requeue. Physics, input and resource
limits remain fixed. No validation or push PASS receipt is created by this policy.

## Repair of validation job 1091645

Job 1091645 failed in 19 seconds, before its first timestep. Its incremental binary
mixed particle class layouts: the energy calculation read the mesh-pack pointer at
byte1544, while the caller used byte1568. All three executables have been rebuilt
in fresh directories; old builds, receipts, logs and failures are retained.
`session011/build.sh` now always creates a fresh build tree, checks the native
source hash before/after compilation and inspects the compiled energy routines.
This inspection is not a GPU test. Source physics and all inputs are unchanged.
The launcher starts native processes in their own output directory so model and
constructor diagnostics also stay with the run.

## Keep the pending production job

From a Vista login node in the session root, after preparation is complete:

```sh
bash update_s11_production.sh 1091797
bash submit_s11.sh validation --retry-of 1091645
```

The first command keeps job1091797. It requires that the job is still PENDING,
owned by this user, with the approved16-node/36h GH request. It checks the clean
committed checkout, new executable/input/tool hashes, clean-build receipts,
allocation, queue, duplicates, modules and storage. It creates an immutable
`evidence/production_updates/1091797.json` binding the original submission to the
new manifest and commit, then updates **only** the job's Slurm Comment to the
SHA256 of that receipt. It verifies the controller response and retains both
snapshots. It does not submit, cancel, hold, release or change job priority or
resources. The original submitted environment and receipt remain unchanged.
Slurm supports user updates of pending job metadata, including Comment:
https://slurm.schedmd.com/scontrol.html#SECTION_JOBS---SPECIFICATIONS-FOR-UPDATE-COMMAND

At startup the launcher verifies the exact original environment/receipt, the
controller's amendment digest, the new manifest, unchanged native physics/inputs,
and the new clean commit. Missing or changed evidence fails closed. RUN_MODE.json
records the original and amended bindings, with real validation/push status.
The ordinary run and its resource/output/final guards then use the rebuilt binary.
If the job has already started, the update command refuses; inspect its log and
accounting rather than replacing a running program.

The second command submits one new four-node/two-hour gh-dev validation attempt.
`--retry-of` explicitly reviews the latest retained attempt against live squeue
and final sacct accounting. Active, ambiguous, completed or mismatched attempts
block it. All previous attempts remain intact; no automatic retry or chained job.
Validation may run alongside the production job under the existing user policy.

## Normal submission and limits

`bash submit_s11.sh check` performs preflight only. A new production submission
uses `bash submit_s11.sh production`; another same-stage job or retained attempt
blocks duplicates. If a reviewed failed/cancelled production attempt actually
needs replacing, use `bash submit_s11.sh production --retry-of JOBID` after its
terminal state is visible in sacct. This command never cancels another job.

Production is fresh t0, one rank/GPU on sixteen GH nodes, at most36h, strict t150.
All memory/telemetry/capacity/disk guards, forty-minute reserve, native t0 census,
final native audits, output coverage and failure retention remain active.
The current manifest and clean commit are bound at submission, or explicitly
amended in Slurm for the same pending job as described above. A successful FINAL
is an execution/output result, not a scientific validity or convergence claim.
Native GPU validation is still required before any GitHub push. Final figures,
movies, reviewed strain and scientific analysis remain post-run work.
