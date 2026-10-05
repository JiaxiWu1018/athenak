# Session 008 continuation authorized on 2026-10-05

The user requested continuation after reviewing the stopped run, and asked for current AMD storage and node-hour usage. The scientific endpoint remains t=12 and the total AMD cap remains 48 raw node-hours. The boost, particle model, mesh, outputs, executable and original production scripts remain unchanged.

## Diagnosis and measured baseline

Job 447661 failed in 26 seconds during MPI initialization: node k003-010 chose `ob1`, while its peer k003-001 chose `ucx`. No simulation step was executed. The verified checkpoint is t=9.325, cycle2252, with3,998,466 alive particles and1,001,534 lapse removals; its SHA-256 is `981af0bf7deb074b9a0ab2f9fc83cd547baa2b7cdad0cd265d0d9337b1a4be6a`. The saved header is51,865 bytes, exercising the repaired >40KiB restart path.

AMD allocation usage through the failed job and all inspectors is21.645833 raw node-hours, leaving26.354167 under48. Session storage is123,379,183,616 bytes (114.906GiB); the whole AMD user root, measured separately, is196,519,669,760 bytes (183.023GiB). Anta holds58,136,141,824 bytes (54.144GiB) of verified validation data. The production segments are sealed onAMD and awaiting archival.

Archival stopped because `squeue -j` returns an error when a completed job disappears from the active queue. A separately versioned trigger now lists active user jobs and confirms completed jobs with accounting. Its earlier48-hour window expired; this continuation request authorizes one new48-hour metadata window, with the original cumulative four archival-job limit unchanged (one job already completed).

## Finite continuation

- Use only the six nodes already used successfully by this exact configuration: k005-002/003/004/005/007 and k003-007. Three nodes/twelve ranks per GPU job; automatic transport and the existing module stack remain unchanged.
- One ten-minute mesh/communication preflight, including a full allocated-node SHA-256 check of the real t=9.325 checkpoint.
- At most two four-hour continuation jobs, with3h20 applications and forty-minute finish reserves. A second job runs only if the first verified continuation stops belowt=12. Each has a thirty-minute allocated inspector.
- Added maximum exposure:25.5 raw node-hours; maximum total, including all previous actual allocation time:47.145833, below48. Numerical, infrastructure, archive or resource failure stops for review; no automatic failure retry.
- New operations live in `scripts/recovery_20261005/`, with separate frozen hashes/configuration; original tested scripts and their frozen configuration are preserved. Old failed state, flags and logs are retained before the reviewed continuation is prepared.
- Restore archival before submission. Keep sticky human stop intent and latest-three checkpoint policy. Copy/hash/delete completed science only under the existing approved cleanup scope.
- Produce analysis, plots, central/context/density movies, and updated reports from the complete available verified run. t<=12 remains an early assessment with no complete-orbit or merger-wave claim.

## Submission and archive update, October 5 07:52 UTC

Recovery checks10957/10958 passed; AMD jobs451757–451761 are queued with verified dependencies. Anta archive2329 completed0:0, verified both production segments, and removed only approved science copies fromAMD. Current allocated storage: AMD campaign76.53GiB, entire AMD user root157.65GiB, Anta campaign92.77GiB. Actual AMD node-hours remain21.645833; queue waiting adds none. This update supersedes the earlier pre-archive storage figures above.
