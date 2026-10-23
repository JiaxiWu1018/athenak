# Session 11 preparation tools

Read the session-root REPORT.md and AGENTS.md first. Compilation/source tests do
not certify the AH repair. The preparation chat must not execute native simulations.

Compilation on allocated GG:

```sh
bash code/scripts/particles/session011/build.sh
python3 -B -m unittest discover -s code/scripts/particles/session011/tests -v
```

In the **next chat**, after explicit user direction, from Vista login:

```sh
export S11_ROOT=/scratch/10222/jiaxiwu1018/jeans_s11_20261022
sbatch --export=ALL,S11_ROOT "$S11_ROOT/code/scripts/particles/session011/validate.sbatch"
```

Keep the returned job ID; inspect squeue/submission status before any retry if the
response is ambiguous. The validation job never submits production. Its genuine
PASS.json binds source, executable builds, input and checkpoint to real GPU runs.
Inspect native residuals, persistence flags and all retained output before push.

After validation and a fast-forward-only push to origin/project/GI-in-cluster,
record the confirmed remote HEAD and matching native hash in
`evidence/PUSH_VERIFIED.json` with `passed:true`. Do not fabricate this receipt.
Only then run `launch.py readiness` and hand the user the single production command:

```sh
sbatch --export=ALL,S11_ROOT "$S11_ROOT/code/scripts/particles/session011/production.sbatch"
```

No production is submitted now. It is one application invocation, fresh t0, at
most36h on16 GH nodes, hard150M. Native checkpoint writes every10M do not restart
the application. Runtime memory/telemetry/capacity and final full-grid/tag checks
remain active; no automatic retries, relaxed AH acceptance or raw-data cleanup.

For post-run paper work, `analysis_utils.py` provides gap-aware orbital quantities,
an exact discrete Jz decomposition and tested finite-radius FFI. These are utilities,
not finished publication figures or proof of a valid strain band. Use the preserved
Session10 analysis/rendering code in this repository as reference, adapt to the new
state-keyed schema, and finish scientific/visual validation in the science chat.

Important: radial profile momenta are covariant u, NOT coordinate velocities.
`cohort_updates.csv` includes summed delta u/delta x for selected tags even in their
last interval before removal; rank cohort files contain surviving states only.
Component deltaJ accumulates EVERY live particle update before removal. Native
death-ledger cycle n belongs to completed state n+1. Merge by exact cycle/tag,
not rounded legacy timestamps. GW state-time mapping is explicit in its sidecar.
