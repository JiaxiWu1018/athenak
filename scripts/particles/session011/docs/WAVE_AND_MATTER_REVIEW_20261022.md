# Finite-window strain and matter-J review — October 22, 2026

The midnight release contains an arbitrary constant offset in its provisional
strain comparison. This affects the interpretation of the plotted cutoff
sensitivity. It does not change saved rPsi4, particle data, AH measurements or
production. No trajectory restart is indicated by these checks.

AMD job 474499 completed in 9 seconds on one node, reading the immutable midnight
source snapshot through t=11.4703125. The 223-sample R40 interval spans
t=5.88125–11.43125 with first nonzero FFT frequency 0.1793721973/Mref. All three
cutoffs, 0.01, 0.02 and 0.04/Mref, are below that frequency and regularize no
nonzero Fourier bins. Tapering after mean subtraction reintroduces a mean; the
original FFI turns it into a constant strain proportional to 1/f0². The measured
r h+ means are −3.332709771e−7, −8.331774428e−8 and −2.082943607e−8, matching
that prediction. After subtracting each interval mean, the three complex curves
differ by at most 3.14e−22. Their apparent separation therefore does not measure
resolved waveform cutoff sensitivity.

Source f5258a68 explicitly sets the strain FFT zero mode to zero and records each
interval's first nonzero frequency and affected bin count. This chooses the
unconstrained integration constant; it does not recover memory, improve frequency
resolution or validate a burst/ringdown. AMD QA 474503 completed in 29 seconds on
one node: all 12 product tests passed, including tapered nonperiodic input,
resolved low-frequency suppression and complex periodic reconstruction. The
corrected real strain agrees with the old strain minus its mean within 3.32e−22.
Raw wave tables at all three radii are byte-identical. The second-derivative RMS
checks are unchanged: approximately 2.51% against the filtered source and 11.36%
against the original tapered source, including its projected-out mean.

The matter-Jz spike also received a full discrete bookkeeping check. For each
component and compact-output interval,

`ΔJz(survivors) + Jz(deleted particles) − recorded push ΔJz`

has absolute residual at most 7.55e−16. The corresponding rest-mass residual is at
most 2.71e−14. Shell-advection terms cancel globally within 2.27e−16. The exact
removed counts are 8,664 envelope, 778,695 clump 1 and 778,227 clump 2 particles,
matching the audited census. The clump peaks are Jz=0.07791076 at t=7.49375 and
Jz=0.07050555 at t=7.61875. These features exist in recorded particle updates;
they are not explained by a missing removal or shell-transport term. This check
does not decide how much is physical, gauge dependent or numerical. Coordinate
matter Jz alone is not a conserved spacetime angular momentum.

At the clump-1 peak interval, the +0.01824285 survivor-Jz change equals a
+0.00064757 recorded push change minus −0.01759528 carried by removed particles.
At clump 2's peak, +0.00666506 equals +0.00205369 minus −0.00461137. Selective
removal of negative-Jz particles therefore raises the surviving population's Jz
in these intervals. This is a discrete selection effect in the displayed
survivor ledger; it does not imply an equal gain in total matter-plus-gravity Jz.

Diagnostic source a1536505 and correction/QA source f5258a68 are committed and
pushed on experiment/GI-in-cluster-session011. Jobs advance zero timesteps and
leave production and saved inputs unchanged. Their combined cost is 38 node
seconds (0.010556 node-hours), bringing completed AMD usage to 75.147500
node-hours before future allocations. Production continuation 474357 remains
queued with its 24-hour reservation. Saved-AH angular validation and the final
t75 science report remain pending.

Compact figures, results, exact submitted batches and provenance are archived
under evidence/wave_ffi_review_20261022/. Raw/reduced data remain on AMD under
/work1/eliasmost/jiaxiwu/gi_s011_amd_20261021/. Earlier hourly releases and their
manifests must remain unchanged; dated correction notes accompany them. Deployment
and the next hourly product verification are recorded in SESSION_STATE.md.

Validated hourly export hourly_tools/f5258a68 now serves job474506 at08:00UTC.
Only the unallocated report474480 was superseded; production474357 and the
Perseus mirror18865 were unchanged. The corrected figure was visually inspected,
and21compact files (809,644bytes) were checksum-verified on Perseus. Dated
WAVE_CORRECTION notes accompany allfour original releases on both clusters;
their original affected-file hashes were verified unchanged. Endpoint tooling
still awaits saved-AH validation/guarded deployment and must also receive this
FFI correction before any final t75 product is promoted.
