# Session010 exact Vista configurations

Three original branches are preserved byte-for-byte under `vista_original_20261019/`:

| Branch | Role |
|---|---|
| [20261013T200009Z](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/config/vista_original_20261019/20261013T200009Z/config.json) | Initial compact debug gate and failed original production wrapper |
| [20261014_gh_retry_01](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/config/vista_original_20261019/20261014_gh_retry_01/config.json) | Corrected fresh t=0 production and telemetry-stopped continuation |
| [20261016_gh_resume_01](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/config/vista_original_20261019/20261016_gh_resume_01/config.json) | Final recovery from verified t=55.00625 checkpoint |

Each contains `config.json`, `production.athinput` and `compact.athinput`.
The final [native production input](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/inputs/vista_production/production.athinput)
is also placed under inputs/. JSON settings describe original allocation,
checkpoint, telemetry, capacity, memory and filesystem guards; these are not
newly approved limits. Keep historical attempts, run states and source bindings
associated with the appropriate branch. Original output/movie settings are in
[the current analysis metadata](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/vista_analysis/MOVIES.json)
and [analysis scripts](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/scripts/vista_original_20261019/analysis_s10_update_20261018/README.md).
See [technical report](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/REPORT_AGENT.md) before any future reuse.
