# Session010 restart after the monitoring stop

From a Vista login shell, outside idev, in the handoff directory:

```bash
bash launch_s10_recovery/submit.sh
```

This submits exactly **two 24-hour GH jobs on 16 nodes**. The second has an
`afterok` dependency on the first. No debug/build job is queued. Both jobs use
AST21006 and BEGIN/END/FAIL email to the verified jiaxiwu@caltech.edu. Maximum new
exposure is 768 GH node-hours; there is no automatic requeue or numerical retry.

The restart begins from the verified **55.00625 Mref** checkpoint, not t0 or the
end of the first segment. The original failed run, logs, status files, inputs and
launcher are unchanged. New evolution lives in
`runtime_s10/runs/20261016_gh_resume_01/`; the active pointer switches only when you
invoke the real login submission. A preview submits nothing:

```bash
bash launch_s10_recovery/submit.sh --dry-run
```

Job 1071752 lasted 4h 04m 43s. Its native simulation exited cleanly with a verified
checkpoint and balanced five-million-particle census. The Python wrapper returned
42 because one status snapshot read failed. All 16 sampler histories have no
reported errors and no interval longer than 11.65 seconds; peak GPU memory was
67.57%, peak host memory 45.64%, below the unchanged 85% stop. The exact original
file-read exception was not logged, so the precise filesystem error is unknown.
This supports a transient monitoring-read problem, not a numerical crash, OOM,
or exhausted wall time. The dependent old job 1071753 was canceled unallocated.

A status file is a small JSON file reporting a node's GPU/host memory and sample
time. The repair discovers all node files at sampler startup, then reads those
known paths rather than repeatedly listing a directory while files are atomically
replaced. A transient read has at most three attempts, 0.25 seconds apart; errors
and recovery are logged in each chunk's `telemetry_read_events.jsonl`. Persistent
read failure, stale data and explicit sampler errors still stop safely. Real
memory-limit readings still trigger the original immediate stop.

The native CUDA executable, physics, domain, resolution, output cadence, particle
removal, memory/capacity/storage limits, two-hour internal checkpoints and
40-minute finalization reserve are unchanged. Existing native/full-size gates are
reused only with matching hashes. Preparation independently hashes the current
checkpoint and audits the entire native grid and particle state for finite values.
The user-authorized monitoring recovery has its own immutable seed receipt and
provenance; it does not clear a failure in the old run or permit arbitrary retries.

If sbatch output is ambiguous or either submission fails, inspect the new
`submission.json` and squeue; do not run the script repeatedly. A recorded attempt
blocks duplicate submission. Earlier and new allocated node-hours remain together
in the restart run's accounting. No production job has been submitted by the agent.
