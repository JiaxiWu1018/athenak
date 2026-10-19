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
