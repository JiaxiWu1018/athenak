# Session010 first-production analysis

Run `bash analysis_s10/run.sh` only inside an allocated Vista **GG** job. No Slurm
submission, simulation change, raw-file deletion, transfer, or historical source
edit is performed. The default analysis wall cap is two hours, additionally bounded
by the actual current allocation end with a two-minute reserve.

The input branch is fixed to the completed first segment of `20261014_gh_retry_01`,
through 36.55 Mref. Outputs are under
`runtime_s10/analysis/first_production_20261015/`. The environment is the isolated
ARM/Linux `runtime_s10/analysis_env` virtual environment (uses installed numpy/h5py).
The ARM imageio-ffmpeg bundled executable provides encoding and full decode/frame
count validation. The system ffmpeg/ffprobe are not usable here (missing libunwind).

Stages `seal`, `reduce`, `fields`, `render`, `verify` may be selected with
`--stage NAME`; the default `all` runs them in order. Completed field and movie
source frames are reusable. A frozen manifest, SHA256 input receipts, reduced
arrays/tables, output hashes and verification receipts accompany the deliverables.
If a stage fails or allocation time is low, inspect LAST_FAILURE.json and resume
from that stage in a new GG allocation. The tool does not request that allocation.

Seven figures have PNG and PDF companions; three MP4 movies use fixed cameras,
fixed scales and original particle component colors. Playback holds recorded
states at 2 Mref per second and 12 fps rather than inventing intermediate motion.
The displayed timestamp is the actual saved source time. Near-zero finalization
snapshots may not each appear on the uniform playback timeline.

Horizon acceptance uses the packaged unique time-and-geometry join, not raw finder
convergence. Gaps stay distinct, initial units remain Mref=1, raw GW remains rPsi4,
and constraint norms retain the inherited chi>=0.0625 proper-volume mask. Neither
checksums nor successful rendering establish physical convergence.

After machine verification, run `runtime_s10/analysis_env/bin/python analysis_s10/qa.py`
on GG to render the actual PDFs and decoded movie review sheets. Inspect those
images before invoking `analysis_s10/finalize.py --reviewed-by NAME` with the same
environment. This explicit sign-off binds the review to current PDF/movie hashes;
the pipeline does not silently claim a visual review. The final INDEX.md links all
deliverables. `analysis_s10/check.py` runs and records the unit suites; all scheduler
calls in those fixtures are mocked, and no real jobs are submitted or cancelled.
