# Hourly monitoring and partial products

User requested hourly monitoring, analysis, required plots, two movies and reports
from existing production data, with immediate investigation/documentation/correct
relaunch for bad checkpoints or output. This supplements the continuous24h policy.
The hard production endpoint remains75. No evolved controls or replay is added.

hourly.py runs on an independent AMD devel allocation, initially immediately and
then at the next wall-clock hour. Each invocation schedules its successor before
analysis; Slurm resource availability can delay actual start. A refresh lock stops
overlapping renderers. Immutable timestamped source windows, reports and manifests
are retained. No signal, stop file or simulation mutation is issued by this code.

A snapshot reads complete lines within observed source-size bounds, filters exact
state time/cycle, and aligns all six wave arrays to the common state-time prefix.
Published cohort files are read in place with exact native header/size validation;
no full native state or particle data is moved. Diagnostic surviving counts plus
all recorded removals must close5M. Reports explicitly distinguish the live sample
cutoff from the last full native checkpoint audit. Angular AH tests and final t75
interpretation are not claimed complete. Existing scientific figures and both
1920x1080 movies are refreshed. Exact raw cohort, AH geometry, timing and code hash
key a shared verified frame cache, limiting repeated rendering and storage.

Health receipts include actual Slurm state/accounting, GPU telemetry and controller/
checkpoint review markers. Terminal failures and bad-data evidence are preserved
for immediate agent investigation; no blind replay occurs on queue/observer errors.
Memory/storage watchdogs continue. Report construction errors are separate from
simulation failures and never trigger production interruption.

Initial QA474354 exposed missing PRODUCT_QA in the launcher; corrected474356
passed10product+4hourly regressions in32s/1node. First scheduler request was
explicitly rejected: AMD devel max30minutes, so hourly jobs now request30minutes.
No production interruption or failed-data inference was made. Definite scheduler
filter rejections are recorded separately from ambiguous submission receipts.
