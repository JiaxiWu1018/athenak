# GH production repair, 2026-10-14

Job 1069970 failed before creating the mesh or initial particles. The input omitted
`time/nlim`, while the launcher attempted to override it. The pinned native parser
only replaces parameters already present in the input; its exact error was:

```text
Parameter 'nlim' in block 'time' on command line not found
```

This was a launcher/input mistake. The secondary CUDA finalization messages follow
that early exit; they do not establish a GPU-memory failure. No production
checkpoint or scientific evolution was produced. The two dependent 24-hour jobs
were canceled without allocations. The old run, inputs, receipts and logs remain
under `runtime_s10/runs/20261013T200009Z/`.

The corrected input declares `nlim=-1`, meaning unlimited cycles. Physics,
resolution, particle count, boost, safeguards and the previously verified native
CUDA executable are unchanged. Fresh and restart command-line keys are now checked
before launch. A CPU-only diagnostic build of the exact pinned source reproduced
the old error and passed normal, one-step, two-step, and restart parser checks with
`-n`; it performed no evolution and submitted no job. Its evidence is under
`runtime_s10/build_parser_check/evidence/PARSER_VERIFIED.json`.

## Submit the repaired GH chain

From Vista login in the handoff directory:

```bash
bash launch_s10/submit_gh.sh
```

This submits **exactly three GH jobs on 16 nodes: 8h → 24h → 24h**, with success
dependencies. No gh-dev job or build job is submitted. The existing successful
debug receipt from 1069961 is copied and tied to the same CUDA executable and
compact fixture. Full-size memory/MPI/restart/output/timing and actual AMR gates
remain inside the first 8-hour GH job. Production still starts fresh at t0, because
the failed attempt created no restart state.

The new prepared root is `runtime_s10/runs/20261014_gh_retry_01/`; its metadata is
in `runtime_s10/GH_READY_RUN.json`. The previous active-run record is preserved
before the user submission switches it. Duplicate submission is refused, previous
live jobs block a new chain, and every job retains BEGIN/END/FAIL notifications to
`jiaxiwu@caltech.edu`. Existing two-wall-hour checkpoint/restart and memory guards
remain. Nothing is automatically submitted or numerically retried.

Preview without submitting:

```bash
bash launch_s10/submit_gh.sh --dry-run
```

## Usage and charges

The debug job used 0.493333 actual GH node-hours (4 nodes × 444 seconds). The failed
production job used 0.186667 actual GH node-hours (16 nodes × 42 seconds). Their
combined actual usage is **0.68 GH node-hours**.

Vista's published charging rule applies a 15-minute minimum to allocated jobs:
debug is **1 billed GH node-hour**, failed production is **4**, and both canceled
continuations are **0**. Thus the simulation chain's charge is **5 GH node-hours /
5 SUs**. GG preparation and inspection cost 0.33 SU per billed GG node-hour.
These are tariff estimates from Slurm allocation rows, not a queried portal invoice.
Source: https://docs.tacc.utexas.edu/hpc/vista/#job-accounting .

`runtime_s10/repair_20261014/USAGE.json` records the timestamp, allocation rows,
confirmed-session total, and broader total for all your Vista allocations in this
period. The current GG allocation is provisional and continues accruing. Nearby
GG job 1070169 is included only in the broader total because its connection to
Session010 is unproven. Refresh with `python3 launch_s10/usage_snapshot.py` on
allocated compute.
