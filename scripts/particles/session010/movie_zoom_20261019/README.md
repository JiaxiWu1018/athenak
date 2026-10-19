# Session010 sixfold 3D close-up, t0–98

Accepted canonical release:
`/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/`.
[Caption](docs/CAPTION.md) · [Pull commands](docs/PULL_TO_MAC.md) ·
[Full record](docs/MOVIE_ZOOM_20261019.md) · [Final checks](receipts/FINAL_CHECKS.json).

The fixed ±4Mref cube is six times closer than the original ±24 view. Camera,
styles, physical AH centers/radii, tagged particles, quality gaps and timing
are preserved. Schematic AH glyphs draw above particle collections so cloud
sorting cannot hide them; this is an annotation convention. All56 accepted
markers fit in the image and all20 pre-common pairs have separate bounds.
No fabricated pre-formation/post-merger individual holes or missing surfaces.

The accepted MP4 is1920×1080 H264/yuv420p,12fps,393frames,32.75s, t0–98 inclusive.
No interpolation, new sampling, later particle state, finder repair or native
restart. Source code remains native386870f8. The original wide movie and the
unpromoted zoom preview are preserved. Their hashes/geometry/timing are bound
by receipts; source and decoded contact sheets were visually inspected.

Run `render_zoom.py --session SESSION --out NEW_VERSION --view-limit 4` in
Slurm with NumPy/Matplotlib/Pillow/ffmpeg/ffprobe. It reads the existing accepted
wide release and immutable particle caches, refuses existing frames, and fully
decodes output. `check_zoom.py` checks marker visibility in actual raster images,
unchanged geometry/gaps/timing, separation and original native source. Batch
recipes include BEGIN/END/FAIL email and allocation/thread settings.

Jobs17387(preview),17392(final render),17394(initial pixel-check assumption),
17396/17397(pixel diagnostics),17398(final passing check). Tiny antialiased
facets blend with the background; checking exact RGB0 was inappropriate.
The final raster check tests dark contrast instead. The preview source is
retained in draft/. Movies/frames/caches stay outside Git. Commit receipt lives
under the session's evidence/movie_zoom_20261019/ after verified push.
