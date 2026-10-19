# Session010 orbital-duration analysis and accepted standalone t0–98 movie

Native simulation source remains386870f88f60e25e1a8fac76fe04a071b87a01cf;
this package changes no native code, input, boost, refinement or AH finder.
The known t98.2 restart defect remains unfixed. No simulation was restarted.

The orbital analysis distinguishes6Mref from6 times the pair mass: the clump
sources total0.24Mref, with an extended0.76 envelope. Vacuum circular reference
scales give tens of turns rather than0.67. A12Mref separation (centres±6) with
reviewed momenta is a first exploratory candidate only, without a guaranteed
2–3turn count. Initial static-metric speed there is about0.1651; retaining the
current0.133215 may increase eccentricity. See [analysis](docs/ORBIT_DURATION_AND_SEPARATION_20261019.md),
[measured/reference numbers](receipts/ESTIMATES.json) and
[assessment figure](orbit_duration_assessment.png).

The accepted movie matches the exact Session007 standalone renderer's navy,
purple/orange/cyan colors, marker styles, fixed camera and equal±24 cube.
Measured accepted coordinate AH centers/mean radii render as opaque black
spheres. Geometry requires exact native case/cycle and printed-time matching.
Strict common enclosure gates the switch; missing accepted surfaces remain
absent. Surviving members of the existing fixed90k cohort are never replaced
or resampled. [Caption](docs/CAPTION.md) and [acceptance](receipts/ACCEPTED.json).

Output:1920×1080 H264/yuv420p,12fps,393frames,32.75seconds,3Mref/playback second.
Fifty actual saved states through98 were rendered; a0.25Mref CFR clock holds
the latest actual saved state. The tiny0.00625 state receives zero encoded
frames. The final frame is saved t98; no t98.2/later particle cache was loaded.
[Timing table](receipts/TIMING.csv) · [Frame/AH provenance](receipts/FRAMES.json) ·
[Source hashes](receipts/SOURCE_MANIFEST.json) · [Final checks](receipts/FINAL_CHECKS.json).

Canonical release on Perseus:
`/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/`.
MP4: `jeans_session010_particles3d_t0_to98.mp4`,5,005,907bytes,
SHA256 `724d73b0508efa8c81fdd025203770d150fd1ed2ff91ad888da181b305f8ae22`.
Movies, frames, caches and the11MB old reference movie stay outside Git.
[Verified one-way Mac pull](docs/PULL_TO_MAC.md) copies only final review artifacts.

Reference:
`jiaxiwu@anta.caltech.edu:/data2/jiaxiwu/NRPIC_test/s7_plotting_bundle_20260911/cca_talk_20260912/revision_20260912g/slide27_jeans_particles_3d.mp4`;
its renderer is preserved byte-for-byte at [reference renderer](reference/cca_render_gi_3d_movie.py).
The accepted canonical release has the reference movie/JSON and comparison QA.

Slurm jobs:17079 schema inspection,17088 render/encode/full decode,17091 orbital
estimates,17094 independent source/frame/timing/AH/movie-hash/figure checks.
Representative source and decoded early/pre/post-common/final frames were
visually reviewed. All completed; no production, raw deletion or bulk transfer.

Reproduction uses NumPy, Matplotlib, Pillow and native ffmpeg/ffprobe in Slurm.
`render.py --session SESSION --out NEW_VERSION` requires an unused versioned
output with `reference/` populated from the canonical reference copy. It refuses
existing images/movie/completion receipts. `orbit_estimate.py --session SESSION`
is likewise a one-time dated release. Change the output version deliberately
before a rerender; preserve prior evidence. The three batch scripts show exact
allocation, thread limits, paths and BEGIN/END/FAIL notifications. Full native
reconstruction or AH repair is not part of these tools.
