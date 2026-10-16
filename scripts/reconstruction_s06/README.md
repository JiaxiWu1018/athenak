# ST reconstruction session 6

Continue only `proper_stratified_s1` at 300M to absolute tlim=1000M with the
protected session-5 HIP executable. Compiled source 5a5db71c; Kokkos 6739bc62.
No physics, mesh, integrator, gauge, particle or output-cadence changes.

The archived areal radius R/M=9.96420711980792 gives the reference surface
circular period at infinity P/M=2*pi*(R/M)^1.5=197.62595786609717.
Thus 300M=1.5180192P; 5P=988.1297893M; 1000M=5.0600640P.
This is a reference timescale, not an assumed shell-pulsation period.

Pipeline: allocated Perseus staging, AMD 300->306M smoke, human review of
receipt, unchanged 300->1000M production, allocated reduction/QA/rendering.
The smoke and production start independently at the original saved state.
Common interior 302/304M outputs verify reproducibility without terminal-step
clipping. Native restart dt ramps from the tiny saved terminal dt without editing
the checkpoint or solver. Keep completed-dt constraint masking (>1e-6M).

Use AMD one exclusive mi2104x node, four MPI ranks/four MI210 GPUs, ROCm6.4.1;
the inherited OpenMPI setting and node exclusions remain. Forecast ~2 solver
hours for 700M, 45-50GB raw growth, four-hour production allocation.
Monitor whole-user storage every five minutes; warn1.5TiB, stop1.7TiB against
the1.9TiB budget. Preserve data; no cleanup is authorized in this session.

Retain session-5 0..300M compact prefix and original t=0 normalizations. Join new
2M particle/history outputs by actual time, preserve all9 COM/origin shells,
initial-ID cohorts, constraint-volume conventions and matched-CIC error reference.
Central lapse is the existing r<0.75M mean. New slice central E is a separately
labelled proxy and has no invented pre300M history.

Deliver histories and amplitude/drift PNG/PDF figures. Existing100M windows plus
complete P and2P windows distinguish excursions from drift and expose window
sensitivity. Report descriptive slopes and ratios, without treating samples or
shells as independent trials. User selected descriptive results only: no automatic
scientific pass/fail threshold and no launch of the later five-case paper rerun.
Discuss single-seed, fixed-resolution/domain and boundary limitations.

All bulk transfers, hashing, validation, reductions and rendering run in Slurm.
Record exact source/environment/input/command/hash/job/output manifests. Reports,
figures, compact reductions and final checkpoint are mirrored to Perseus and
verified there; retain AMD raw and report its exact size/path/status.
