# Native energy audit diagnosis and correction

Production473355 reached cycle1902,t8.803125, then exited0 on its normal two-hour
application limit. The Python audit blocked continuation because it required all
particle reals, including diagnostic IPEN, to be finite. Last promoted state1230
remains t7.753125 until all correction gates pass. Both individual AHs accepted
at cycle1699,t8.4859375. No common AH has been observed.

Non-evolving job473899 (2 seconds, one node) found330 nonfinite IPEN values,
165 per clump, but finite mass/position/covariant momentum/GID/tag, positive masses
and exact unique in-range IDs. Pair milestone00004 has all particle fields finite.
Job473909 (94 seconds, one node) reconstructed every affected particle from the
unchanged final checkpoint: all330 high-order interpolated spatial metrics are
not positive definite and all330 energy quadratics have negative discriminants.
All330 trilinear interpolants are positive definite with positive lapse; high-order
stencil nodes are finite. A330-particle finite-energy sample has valid high-order
and trilinear interpolants. All25 saved grid fields are finite across all4684
blocks including ghost cells (full saved-grid scan, no evolution).

The code's IPEN consumers are diagnostics/output only. Neither the particle
pusher nor stress-energy deposition consumes it. The existing pusher independently
checks geometry and has its own validated trilinear fallback. This repair changes
no C++ code, executable, physical input, timestep, pusher, deposition or AH gate.
It does not re-evolve from an earlier checkpoint or replace any stored energy.

The corrected Python audit rejects nonfinite physical fields and retains exact
all-tag partition, per-tag rest-weight and online/native Jz checks. IPEN nonfinite
values remain unknown with component counts/rest mass in every checkpoint audit.
Cohort rendering requires finite geometry/momentum/mass and exact valid IDs but
retains unknown energy and records its count. Product aggregation explicitly
propagates unknown native-energy shell totals, including mixed known/unknown
shells, rather than pandas silently omitting NaNs. Independently labelled
trilinear energy retains its separate metric-validity mask. Any future finalization
exception now writes PAUSED.json and cannot silently schedule a successor.

Validation473917 runs15 controller and9 product tests, including negative physical
field corruption tests, mixed valid/invalid energy totals, complete synthetic
report/two1080p movies/decode, and an isolated full real-population audit plus real
cohort parsing. No authoritative control or raw state is modified by QA.

Release is conditional on successful473909 and473917 evidence, exact staged
script hashes, unchanged executable/input and all previous script seals, no live
production, previous cycle1230 and stopped application exit0. It preserves the
previous seal/control/scripts, installs only four tested Python files, reruns full
checkpoint hashing and population audit, and submits continuation only from
cycle1902. See diagnostics/release_energy_audit.py. Earlier raw failures remain
preserved. Diagnostic root on AMD:
/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021/evidence/nonfinite_20261021/.

Production cost through473355:49.2867node-hours. Debugging jobs are separately
registered as evidence/diagnostic_JOB.slurm.log so final accounting includes them.
Target remains75 with validated AH/evidence, endpoint analysis and both movies;
no extension or separate evolved trajectory is authorized by this correction.
