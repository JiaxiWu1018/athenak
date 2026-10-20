# ST session7: nested MC particle-count comparison

Source isolated from completed session6 follow-up20267f44. Compiled solver5a5db71c,
Kokkos6739bc62 and executable5fb4fb9f remain unchanged. Four fresh0..1000M runs,
N=1/2/4/8 million decimal, publicseed1/internal PCG64(3001), proper radialmeasure,
random position/velocity directions, nested original positions and momenta,
equal fixed massesM0/N. No recentering, boost or pairing.

All raw particle/field/checkpoint data, generation intermediates and movie caches/frames
stay on AMD. All substantial generation/reduction/plotting/rendering runs in Slurm.
Only compact tables/reductions, reports, PNG/PDF, final movies, PDF figure previews and
reproduction records may be mirrored to Perseus. Movie caches and all rendered/decoded
movie frame files stay AMD-only; inspect them by streaming without saving frame files
on Perseus. No deletion is authorized.

`stage.sbatch` stages exact source/inputs/executable and submits allocated AMD preparation.
`advance.py preflight` uniquely submits sequential full-count0..20M runs,8M20..22M restart
and native health readout. Actual agent preflight review binds hashes in
PRODUCTION_REVIEW.json before `advance.py production` submits at most two productions.
`advance.py analysis` queues AMD-only serialized reductions, comparison/independentaudit,
five movies and report. All submission intents are durable; an ambiguous sbatch result
must be reconciled against the scheduler before retrying. Preserve failed attempts.

`controller.py` only polls metadata and queues continuation into the existing Codex chat.
It never copies raw states or automatically approves scientific health. All scripts
are hash-bound to SOURCE_PIN.json and checked on every AMD allocation. Freeze runtime
source while jobs are active; preserve any new attempt/source version explicitly.

Quantiles use all particles. Corelapse is coordinate-volume mean r<0.75M about the
fixed origin. Differences are finer minus coarser on common actual times, normalized
by common analytic shell radii (absolute units for lapse). Both absolute and
change-from-initial histories have RMS/mean/max differences and two successive ratios.
Window means, raw/detrended amplitudes and slopes use100M/P/2P windows. No phase alignment,
error-to-exact-solution claim, statistical independence, or fitted convergenceorder.

Actual PNG/PDF and encoded-movie review, scientific interpretation and final integrity
remain required after automated report generation. COMPLETE is written only afterward.

`final_integrity.py` and `final_integrity.sbatch` are staged in a separate immutable
AMD postprocessing directory, with their own source/hash pin. They do not replace
files in the frozen native runtime. One success-dependent job after report generation
rehashes retained files, checks output/checkpoint/reduction/movie receipt bindings,
and inventories sidecars outside out/. It preserves per-job manifests/accounting and
reports exact retained storage. Its NATIVE_INTEGRITY_QA receipt is a closed-file
snapshot; actual scientific/visual review and the audit job's own terminal accounting
remain necessary before COMPLETE. All raw and movie caches/frames remain on AMD.

`interim_shells.py`/`interim_shells.sbatch` make the requested single COM-shell
preview through400M for the available4M/8M productions. All201native outputs and
all particles are used; equal-mass cumulative-CDF ranks plus partition reproduce
the original reducer exactly (checked by full reference sorting at0/200/400M).
Stage these tools and INTERIM_SOURCE_PIN.json in a separate versioned AMD source
directory. Results live under interim/com_shells_to400M_SOURCE/, outside canonical
production reductions/plots. Only final PNG/PDF/figure preview, compact CSVs and
source/QA metadata may be copied to Perseus. No simulation/runtime changes.

The user clarified that the interim figure must contain all four production runs
through400M and chose to wait for real1M/2M production data. `watch_all4_400.py`
performs only lightweight remote file checks on Perseus, every60s. Once all four
402M files exist (so the400M files are closed), it uniquely submits the already
staged `interim_all4_400.py` AMD Slurm job and resumes this chat. Submission intents
and receipts prevent duplicate jobs. The plot reuses validated4M/8M compact CSVs,
reduces all201native1M/2M production frames with the preserved exact reducer, and
checks12full-sort reference frames across the four cases. No preflight traces,
extrapolation, simulation restart or concurrency change. Actual PNG/PDF review and
thin delivery remain required after rendering. Frozen native runtime9e8a0614 and
the independent integrity-audit version are not overwritten.
# CPU postprocessing queue admission

`postprocess_cpu_gate.sbatch` checks AMD `mi2101x` as a CPU postprocessing host.
It imports the frozen analysis/movie/report modules and exact dependency stack,
checks the native runtime pin, ffmpeg availability, at least32GiB effective RAM
and the whole-user storage reserve. It reads no native data and runs no GPU kernel.
Stage it separately with `GATE_SOURCE_PIN.json`; never replace frozen native files.
After its actual receipt passes, existing pending CPU-only reduction/comparison/
movie/report/integrity jobs may move to this partition in place, preserving job
IDs, code, paths, limits and success prerequisites. Preserve before/after scheduler
receipts and the gate's actual environment. This does not admit production runs on
a different GPU layout and does not authorize duplicate analysis submissions.
