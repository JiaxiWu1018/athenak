# Session010: user submission from Vista login

For the repaired production attempt after job 1069970, use
`bash launch_s10/submit_gh.sh` from login. This submits only three GH jobs
(16 nodes, 8h → 24h → 24h), reusing the completed compact check without a new
debug/build job. See `GH_REPAIR.md` for the failure and accounting explanation.

The native ARM/CUDA build is complete on allocated GG job 1069850. The executable is
`../runtime_s10/build/build_cuda_gi/src/athena`; its SHA256 is
`a6221fb3ac3ae9f3e17208bd2fb1d7811ab1c4040adcf5272cab40a7c68218e7`.
`../runtime_s10/build/BUILD_VERIFIED.json` records source/Kokkos pins, compiler,
architecture flags, logs and hashes. CUDA execution remains unverified until the
GH jobs run. GG lacks the GPU driver; no GPU simulation has been run here.

From a **Vista login shell outside idev**, in the handoff directory:

```bash
bash launch_s10/submit.sh
```

This one command submits exactly:

| Stage | Partition | Nodes | Maximum wall time | Dependency |
|---|---|---:|---:|---|
| Compact native check | gh-dev | 4 | 2h | verified native build |
| Full-size gates and fresh production | gh | 16 | 8h | compact check succeeds |
| Continuation | gh | 16 | 24h | full-size first job succeeds |
| Continuation | gh | 16 | 24h | preceding continuation succeeds |

If the native build receipt is absent, the launcher first queues a one-node GG
build, capped at two hours. All jobs use `AST21006`, with BEGIN/END/FAIL notifications
to the verified `jiaxiwu@caltech.edu`. Requested production exposure is 896 GH
node-hours; the compact check adds at most 8 GH node-hours. These describe the finite
chain, not an additional campaign resource budget. The user's later instructions
supersede the original handoff's request for separate time/node-hour/calendar caps.
No t300/t400, nlim, orbit, merger, or ringdown completion target is imposed.

The launcher refuses duplicate submission and refuses submission from compute.
Ambiguous or failed sbatch responses are retained in `submission.json`; inspect
the ledger and queue before taking any recovery action. It never retries or
submits jobs from inside a batch allocation. Afterok dependencies and invalid
dependency cancellation hold later jobs after failure or a safety/user stop.
Vista's stdout welcome banner is accepted before the parsable job ID. For a
preparation-stage partial submission, run `bash launch_s10/submit.sh --resume`
from login: it independently confirms an accepted job's ID/name/user/account,
adopts it, and submits only the missing dependent jobs. Unresolved ambiguity or
a failed predecessor still prevents resubmission.

## Runtime and progress

The launcher creates a dated run beneath `../runtime_s10/runs/` and records its
absolute path in `../runtime_s10/ACTIVE_RUN.json`. Jobs freeze input/configuration
and launcher hashes, use the approved five-million-particle ±512 model, and start
fresh at t0. Reference and compact branches are never production restarts.
The concrete reviewed inputs are `production.athinput` and `compact.athinput` in
this directory; submission copies them only after checking their approved derivation.

The pinned driver requires `tlim`, so the production input uses the agreed
compatibility value `1e30`; the input explicitly declares `nlim=-1` (unlimited).
The native parser requires this declaration before command-line overrides can
replace it. It is explicitly reset to unlimited on restart. Horizon diagnostic
stop times are extended to `1e30`.
Short validation branches use cycle limits solely for comparisons.

Application launches cleanly stop and restart within the same allocation about
every two wall hours. Each checkpoint is checked for native layout, time/cycle,
particle-plus-removal conservation, exact size and SHA256. It must match the native
application exit progress. Resume uses `-r` directly, preserving saved tracker
parameters and output counters. At least 40 minutes is reserved near allocation
end; measured slow finalization increases that reserve. Insufficient wall time
prevents another application launch.

Internal restarts reset the inherited AMR cooldown counters and force all enabled
final outputs. Thus full-volume fields also appear at checkpoint boundaries,
in addition to their scheduled 25M cadence. All checkpoints and failures are
retained. No deletion, node increase, physics change, or numerical retry is automatic.

Memory is sampled on every allocated node every ten seconds, with 80% warnings
and 85% preventive stops for GPU and host memory. Production stops at 90% of total
reserved block capacity (5,760 blocks on 16 ranks), leaving room for review and a
larger-node continuation. This global proxy cannot guarantee prevention of a
large instantaneous or uneven AMR capacity jump; periodic checkpoints limit lost
work if native capacity checks or hardware fail. Storage guards use current free
space plus room for native/bulk final outputs and a 2% filesystem cushion, with
no inherited AMD byte cap. Missing memory or essential tracker telemetry holds
continuation.

The first full-size job checks 4,936 initial blocks, initialization census/model,
same-rank native restart equivalence, output formats, memory, timing, finite native
state, and an actual eligible adaptive change after restart. Its short comparison
precedes AMR; it does not claim identical long adaptive histories. Both 24-hour
jobs require `FULL_SIZE_VERIFIED.json`, a successful predecessor, and verified
increasing native progress.

Use these lightweight commands for status or a graceful stop:

```bash
python3 launch_s10/s10.py status
python3 launch_s10/s10.py stop
```

Status queries actual Slurm allocation rows; in-job accounting remains provisional
until Slurm records final exit. The third job leaves a checkpointed pause and never
claims scientific completion. `PRODUCTION_OUTPUTS.json` lists ordered science
directories separately from validation branches, including verified progress on
stopped runs. Tracker and waveform gaps remain separate from checkpoint time.

## Outputs and verification

Scheduled outputs: raw complex R40 modes and history every 0.025M, compact mesh
every 0.025M, three-plane diagnostics every 0.5M, particles every 2M, full-volume
fields every 25M, plus all native finalization outputs. Existing tracker, horizon
acceptance/rejection and particle-removal diagnostics are preserved.

Local controller tests exercise dependency construction, duplicate/ambiguous
submission, corruption/truncation rejection, capacity/model isolation, memory and
telemetry stops, progress promotion, and wall-time reserves. No GPU gate is claimed
from these tests. A preview that submits nothing is available:

```bash
python3 launch_s10/s10.py dry-run
```

Analysis, plotting, movies and archival jobs are excluded. Your later allocated
Vista analysis should select only `PRODUCTION_OUTPUTS.json` directories. Retain
the handoff's `ANALYSIS_AND_ARCHIVAL.md` for sealing and independent Anta checksum
verification. Anta connectivity/space is still unverified; arrange archival early
enough for scratch retention. The original handoff and its checksum manifest are
unchanged.

Each application directory also contains `SEAL_RESULT.json`, compatible with the
existing `scripts/seal_run.py --run CASE/out --result CASE/SEAL_RESULT.json`.
Run sealing in your later compute allocation; subsequent Anta verification remains
independent. No integrity receipt substitutes for scientific acceptance.
