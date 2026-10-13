# Agent Logistics Guidelines

Read this file before starting or resuming any session. These are standing defaults; Jiaxi's explicit session instructions may change them, within site policies and access permissions.

- **Use the correct node.** On AMD and Vista, heavy computation and analysis must run on allocated compute nodes, never login nodes. On Perseus, Python and other CPU tasks may run directly; GPU jobs require Slurm. CPU work may run locally on a laptop. Check site rules on unfamiliar clusters.

- **Respect resource limits.** Use the requested machine, account, and queue. Estimate wall time, node-hours, and output size before large jobs; stay within the session's limits.

- **Isolate each session.** Use a separate session folder and checkout/worktree on the requested branch. Preserve unrelated edits and jobs. Do not force-push, delete branches, discard existing work, or merge into `main`/`NRPIC` without explicit authorization.

- **Push committed work.** Check changes, make focused commits, and push every commit to the matching branch on `origin` for Jiaxi's review. Verify push success; report any failure.

- **Track running jobs.** Record job IDs and log paths, monitor progress and failures, and check existing jobs before resubmitting. At handoff, state exactly what is still running.

- **Control storage.** Monitor free space and quotas, especially AMD's `$WORK`. Avoid accumulating huge 3D dumps; analyze data where it lives when practical.

- **Archive large outputs.** For AMD/Perseus work, use a project subfolder at `jiaxiwu@anta.caltech.edu:/data2/jiaxiwu/` unless directed otherwise. Transfer completed files and verify sizes and checksums before removing originals within the approved cleanup scope.

- **Follow the approved cleanup scope.** Obtain authorization before deleting campaign data; do not ask again when that deletion is already authorized. Preserve agreed reports, plots, movies, and reproducibility files. Keep checkpoints needed for planned continuation; permanent retention of every checkpoint is not required.

- **Make handoffs reproducible.** Save the code commit, build instructions, input files, run scripts, and analysis scripts. Record machine names, job IDs, and output paths.

- **Close each session clearly.** Write a concise report, update `SESSION_LOG.md`, and produce the requested plots and movies. List completed work, failures, remaining tasks, and exact deliverable paths.
