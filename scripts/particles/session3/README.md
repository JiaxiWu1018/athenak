# Plummer session 03 — isotropic equilibrium benchmark

Started 2026-10-05 (America/Los_Angeles). Status: implementation and preflight.
No scientific equilibrium claim has been made.

The authoritative specification is `Plummer_Isotropic_Session_Prompt.md` and Part I
of `Relativistic_Plummer_Step_by_Step.pdf`. Both are checksum-verified copies;
the originating campaign-root files remain intact.

Use M=1, a=10, eta=0.10, seed 1985, and 2,113,536 equal-rest-mass particles in
opposite-momentum pairs. The run matrix comprises matched live/frozen pairs:
100a and 200a spatial sampling cuts to five common reference periods, and
coarser central mesh and N/4 controls to two periods. AMD budget: 200 node-hours.
The sampling cut is neither a reflecting wall nor a particle-removal boundary.
The analytic infinite-model metric is retained; omitted-tail errors are measured.

All builds, numerical validation, reductions, plotting and rendering run in Slurm.
The independent source clone is `src/athenak`, branch `project/Plummer-cluster`;
the historical shared checkout `../code` is untouched. See `CONTINUATION.md`
before resuming, and `SESSION_LOG.md` for progress and running job IDs.

Targets: 1% bulk-radius/integrated-stress drift and 2% well-resolved profile
changes, with uncertainty and frozen-control comparison. Interpret only healthy
data. Small angular growth, bulk departure and numerical failure have separate
classifications. No compactness scan is authorized in this session.
