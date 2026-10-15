# Paper Session 02 — code efficiency

Perseus performance-only campaign on branch `experiment/code-efficiency`, from `origin/NRPIC` at `ca0f5298e3752363e7e0b105e4004d0631aeca91`. Kokkos: `6739bc623081648af9e752b616d9671527922cbf`.

Start with the [measurement design](reports/DESIGN.md), [live report](reports/REPORT_code_efficiency.md), [PDF report](reports/REPORT_code_efficiency.pdf), and [figures](figures/). [SESSION_HANDOFF.md](SESSION_HANDOFF.md) records the implementation checkpoint; SESSION_STATE.md is regenerated after each case.

The full matrix is active: 60 throughput cases, 27 phase profiles and six sampled-clock diagnostics. Main runs target 5M, exclude a 0.5M warm-up, use one pass and write no scientific snapshots/restarts. Allocation-limited cases state their actual endpoint. Diagnostic runs target 1.5M. One GPU per MPI rank, four ghosts, native layouts, ST yc=0.819 and flat controls. Dynamical source replicas preserve the initial cell stress-energy while particle count/distribution changes.

The larger cases and queue waits can extend the campaign beyond one day; no total GPU-hour cap was requested. Jobs automatically preserve failed/interrupted attempts, retry unsuccessful measurements at closure, and regenerate reports/plots. CLOSURE.md says COMPLETE only after all coverage, numerical/source checks and seven PDF/PNG figure sets validate. Do not resubmit the whole matrix or replace the primary executable while it is active.

Source and exact build/run/input/executable provenance remain here. Scientific archives and unrelated jobs are preserved. No Mac sync or external publication was requested.

Remaining one-rank/four-rank groups may use separate dependency chains to backfill resource waits. Each group is serial; four-rank jobs consume all four enforced per-user debug GPU licenses, so the two groups cannot overlap. Closure depends on both tails. Exact scheduler/license proof is in `provenance/license_serialized_chains.json`; the watcher flags any unexpected overlap.
