# Session 11 job preparation and manual submission

Read the session-root REPORT.md and AGENTS.md first. This chat prepares scripts
only. Native science execution requires explicit direction in a new chat. Unit
checks and matching compilations do not validate the AH repair or full-size run.

From the Session 11 folder, on a **Vista login node**:

```sh
bash submit_s11.sh check
bash submit_s11.sh validation
```

No argument also means `check`. Checking reads readiness and submits nothing.
`validation` submits exactly one four-node gh-dev job, at most two hours (8 GH
node-hours). It runs the compact GI diagnostics, OS collapse/uninterrupted/restart
comparison with a deliberately poor fallback guess, three-fresh-hit persistence,
and coarse/fine angular spin checks. Each case must reach its endpoint; an exit
code of zero after a wall stop is insufficient. This is a compact fixture, not a
replay of Session 10's actual late background or full-size memory/convergence proof.

The root script calls the versioned launcher in this directory. Preflight checks
native/executable/input/tool hashes, clean campaign checkout, fresh TACC account
membership/balance maps, live Slurm partition rules, existing Session 11 jobs,
modules and output headroom. It also asks Slurm to test the exact resource request
before submission. It uses inherited `S11_ROOT` and plain sbatch, without --export,
--gres, or --gpus-per-task, following the Vista guide:
https://docs.tacc.utexas.edu/hpc/vista/#submitting-batch-jobs-with-sbatch

Submission is refused on compute nodes. If queue access or any other prerequisite
cannot be verified, nothing is submitted. If the actual sbatch response is lost,
ambiguous, or times out, the attempt remains blocked: inspect squeue, sacct and
`evidence/submissions/` before any further action. A previous attempt or runtime
folder also blocks repeat submission. Keep all records, including failures;
request a separate review and resource decision rather than deleting a record to
retry. No job retries or chains itself, and no raw data are deleted or transferred.

The launcher prints the returned job ID, `slurm-JOBID.out`, the receipt path, and
`runtime/validation_JOBID/`. Native COMMAND/RESULT/log files and diagnostics are
retained there. A genuine PASS.json and evidence/VALIDATION_PASS.json are written
only after every real GPU case and scientific gate passes. They bind the exact
executables, source, input decks, command overrides, workflow tools, restart
checkpoint and retained diagnostic/log checksums. Legacy or changed evidence is
rejected. Do not create or edit a passing receipt by hand.

After inspecting those results, perform the conditional fast-forward-only push
from the clean campaign checkout, **only when real validation has passed**:

```sh
python3 -B code/scripts/particles/session011/push_after_validation.py
```

That tool verifies the remote campaign branch and writes PUSH_VERIFIED.json.
It preserves upstream work and refuses a non-ancestor remote tip. A changed
native executable, validation input, workflow tool or production input requires
fresh real GPU validation; a matching source hash alone is insufficient.

Only after this review and push, on the login node:

```sh
bash submit_s11.sh check
bash submit_s11.sh production
```

Production is one fresh t0 application invocation on sixteen GH nodes, one
rank/GPU per node, at most 36 hours (576 GH node-hours), strict tlim=150 and
nlim=-1. The agreed particles/physics/mesh/removal gates remain frozen. Native
periodic checkpoints do not restart the application. The t0 checkpoint is kept,
with the first periodic checkpoint at t10. Forty minutes remain reserved for
clean stopping, final checkpoint/output checks and reporting. Missing/stale
telemetry, 85% host/GPU memory, the 5,760-leaf capacity proxy and the existing
150 GiB write reserve stop the run without automatic continuation. Preflight
requires 3 TiB plus that write reserve; measure actual usage during the job.

FULL_SIZE_STARTUP.json requires the actual t0 component census, 4,936 blocks,
all-node telemetry and native particle state/weight/tag/GID checks, plus the
retained native GI model profile. Runtime output bytes/free space and measured
output rate are logged every five minutes. Production completion additionally
requires a finite final native grid/particle audit, unchanged hashes, recorded
runtime telemetry and OUTPUT_COVERAGE.json. Coverage checks include diagnostics,
rank cohorts, field planes/3D files, particle VTK, native checkpoints, all raw
complex l2..8 wave modes at R40/R50/R60 and their state-time mapping. Published
surface quadrature must have exactly one owner per angle. Conditional AH gaps
and sparse cohort data are reported explicitly. A permitted sub-interval wave
tail is recorded; an exact t150 waveform sample is not assumed.

Inspect FINAL.json/FAILURE.json and all retained output. ACCOUNTING.json captured
inside an allocation may still show RUNNING and is explicitly provisional. After
job exit, obtain final accounting from the login node, for example:

```sh
sacct -X -j JOBID -o JobID,State,Elapsed,AllocNodes,TotalCPU
```

Record the actual job/accounting, native hashes, output sizes/completeness,
coverage gaps and failures in REPORT.md. An early/resource stop is incomplete.
Scientific analysis, independent final-checkpoint auditing, figures, strain
validation, convergence claims and archival transfer remain separate future work.

Compilation, only if native hashes are stale and only on allocated GG:

```sh
bash code/scripts/particles/session011/build.sh
python3 -B -m unittest discover -s code/scripts/particles/session011/tests -v
```

Historical PREPARATION.json/PREPARATION_FINAL.json/SOURCE_TESTS.json are retained.
The new submission preparation and non-science checks are recorded separately in
`evidence/SUBMISSION_PREPARATION.json`; they cannot substitute for GPU validation.

For later analysis, analysis_utils.py supplies utilities, not publication-ready
results. Profile momenta are covariant u, not coordinate velocities. Cohort
updates include a last accepted update before removal; cohort rank tables hold
surviving states. Component deltaJ includes every live update before excision.
Native death-ledger cycle n belongs to completed state n+1. Merge by exact state
cycle/tag and use wave_state_times.csv rather than rounded legacy timestamps.
