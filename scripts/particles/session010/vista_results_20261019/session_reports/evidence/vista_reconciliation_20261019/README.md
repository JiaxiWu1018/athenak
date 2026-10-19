# Session010 Vista reconciliation evidence

Perseus Slurm 17024 verified all 4,044 summary files (8.681 GiB), promoted 163
files by checksum, rechecked reduced scientific measurements and fully decoded
six current/historical movies. See [COMPLETE.json](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/COMPLETE.json),
[source verification](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/source_verification.log), [promotion manifest](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/PROMOTION_MANIFEST.json)
and [independent measurements](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/SCIENTIFIC_RECHECK.json). Saved Slurm allocation:
[slurm_17024.txt](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/slurm_17024.txt). All heavy reads, plotting and decode ran in Slurm.

`vista_analysis/` holds verbatim current Vista JSON/Markdown receipts, including
RESTART_HEALTH, SIMULATION_HEALTH, AMR_HEALTH, INPUTS, SEALED, PROVENANCE,
common_enclosure and horizon_joins. These are preserved historical evidence;
the bulk native checkpoint is not local. Copied Markdown keeps original relative
links and the older queued-successor statement; use [canonical report](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/REPORT.md)
and the intact [original index](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/session010_summary_20261019/workspace/runtime_s10/analysis/through_segment002_20261018/INDEX.md)
for current status/working original links.

`source_inventory/` holds the raw-file/directory inventory and the **newer**
[SCHEDULER_SNAPSHOT.json](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/source_inventory/SCHEDULER_SNAPSHOT.json), which records
1075832 completed and 1075833 cancelled unallocated. No live Vista query was
performed here. `BUILD_VERIFIED.json` and `VISTA_SESSION_LOG.md` preserve native
build/runtime provenance. `movie_qa/` contains the source decoded contact sheet.
Original QA contact sheets remain in the supplied archive. Current figure/movie
copies and the newly generated overview are outside this evidence directory.

The supplied summary is immutable. Preparation root reports were preserved in
history before consolidation, so manifest entries sourcing those old root
reports identify their pre-consolidation bytes. Final audit checks promoted
destination hashes rather than treating a replaced root report as original.
No production, raw transfer, data deletion or earlier-session change occurred.

Final Slurm audit 17032 passed: [FINAL_AUDIT.json](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/FINAL_AUDIT.json). All 163
promoted files were rehashed, all eight current PNG/PDF pairs checked, and the
new overview was independently reproduced with identical metrics and PNG.
The [visual review](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/VISUAL_REVIEW_PERSEUS.json) records inspected images.
The final [Git receipt](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/evidence/vista_reconciliation_20261019/GIT_RECEIPT.json) records the pushed analysis commit;
the native as-run source remains 386870f8.
