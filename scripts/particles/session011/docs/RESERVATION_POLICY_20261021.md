## Reservation update — 2026-10-21 17:00 PDT

Production473967 remains PENDING/Priority. It is the same single continuation,
with no new evolution or replay submitted. Measured prior6-node allocation times
were7382,7356,7413,7421seconds (maximum2h3m41s). Existing application chunks stay
2hours. Shortened the pending Slurm reservation and future amd_segment.sbatch
header from12h to4h, leaving nearly2h for startup/finalization. This routine
operational adjustment supports the user's priority to finish production and
watch costs; no input, executable, physics, grid/dt, rank count or AH gate changed.
Scheduler estimate has not improved yet; never promise that shortening forces an
earlier start. AMD estimates are CDT(-0500), user-facing times PDT(-0700).

Metadata-only updater confirmed pending ownership and measured durations,
preserved old script/gate/job records, briefly held only473967, changed exactly
the SBATCH time line, updated the matching gate hash, and released the same job.
Successful receipt: AMD/local evidence/reservation_4h_20261021/RESULT.json.
New live scripts/amd_segment.sbatch SHA256:
20b05ec10b4f78753d389fc2970712d7a652e8c150944049c36b2890c5daa59f.
Gate allocation_policy records4h reservation/2h app/max7421s observed duration.
Local scripts/ mirrors it. Exact source change/updater committed and pushed in
d38cb14a. Remote analysis checkout stays5fa3100a so queued473989's pin remains valid.
The later endpoint deployment installs only its four Python files and preserves
this reservation policy/hash.

Latest queue poll18:58:20CDT/16:58:20PDT:473989 PENDING/Resources;
473967 PENDING/Priority;473992 and474002 PENDING/Dependency.
No extra allocation cost while these wait. Prior debugging total0.175nodeh;
production through47335549.2866667nodeh. Current state remains audited8.803125.
Next: poll these handles, inspect build/test failures or pass, and continue the
production-first saved-AH/deployment chain. Do not duplicate/restart queued jobs.
