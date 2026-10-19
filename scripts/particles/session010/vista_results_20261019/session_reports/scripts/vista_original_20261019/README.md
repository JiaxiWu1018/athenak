# Preserved Vista scripts

These files are copied unchanged from the supplied summary. They preserve the
original machine checks, absolute Vista workspace assumptions and submission
entry points. Review paths and immutable bindings before reuse. Do not execute
a submission entry point on Perseus or interpret this archive as approval for
new production. Historical tests are retained; the historical campaign was not
repeated in this reconciliation.

| Folder | Role |
|---|---|
| [launch_s10/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/scripts/vista_original_20261019/launch_s10/README.md) | Native build, debug/full-size gates, original launcher, output/chunk controls, nlim repair |
| [launch_s10_recovery/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/scripts/vista_original_20261019/launch_s10_recovery/README.md) | Bounded telemetry reader and recovery job preparation |
| [analysis_s10/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/scripts/vista_original_20261019/analysis_s10/README.md) | Historical first-production reductions, plotting, movies and QA |
| [analysis_s10_update_20261018/](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/scripts/vista_original_20261019/analysis_s10_update_20261018/README.md) | Current science, corrected AMR renderer, native audit, plotting/movie/QA pipeline |

The current full analysis normally runs inside a Vista GG allocation with its
native source/raw paths. Original execution and provenance are documented in
[REPORT_AGENT.md](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/REPORT_AGENT.md). The independent portable reduced-data
plot/recheck is [reconcile.py](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/code/scripts/particles/session010/vista_results_20261019/reconcile.py).
The original Perseus promotion batch script is
[perseus_reconcile_vista_20261019.sbatch](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/scripts/perseus_reconcile_vista_20261019.sbatch);
its completed release is protected against overwriting. Use `--replot-to` in an
allocation with an unused prefix to reproduce only the new overview.
