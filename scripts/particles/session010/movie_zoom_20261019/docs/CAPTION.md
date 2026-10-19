# Jeans Session010 — close-up 3D particle/AH movie

[Accepted MP4](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/jeans_session010_particles3d_t0_to98_zoom.mp4): fixed **±4 Mref** cube, a **sixfold linear zoom**
relative to the original ±24 view. The camera, colors, particle marker sizes,
physical horizon centers/radii and time schedule are preserved. Both accepted
individual AH spheres remain inside the view before common enclosure.

The black spheres are schematic AH annotations rendered **above particle
clouds** so the plotting collection order cannot hide one. This layer order
is a visibility convention, not physical ray-traced occlusion. Radii are the
measured mean coordinate radii; they have not been enlarged. Surviving tagged
particles are unchanged; the close-up clips the displayed particle volume to
the central cube. No new resampling, replacement or trajectory interpolation.

The run starts as clumps, with accepted individual horizons appearing near
8.43/8.49 Mref. The first saved particle state showing both is t=10. Existing
missing/rejected-surface gaps remain absent; a missing glyph does not imply a
missing black hole. Individual markers switch to the accepted common sphere
only after strict enclosure at64.36875, first rendered at saved t=66. After
merger the display contains one common sphere. The final frame is saved t=98;
no t=98.2 or later particle state is used. No simulation or finder repair.

1920×1080 H.264/yuv420p,12fps,393frames,32.75s at3Mref/playback second. Actual
saved states are held with exactly the original [TIMING.csv](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/TIMING.csv).
[FRAMES.json](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/FRAMES.json) records source cache hashes, measurements, image
hashes and projected AH bounds. Every accepted sphere fits inside both the
3D cube and1920×1080 image. The largest coordinate extent is2.92631Mref.

Slurm17387 produced a preserved preview where particle collections still
obscured some AH glyphs; that version is not promoted. Slurm17392 rendered
this accepted version, fully decoded it and extracted representative decoded
frames. See [review](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/evidence/ACCEPTED.json), [machine checks](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/evidence/COMPLETE.json),
[format](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/evidence/FFPROBE.json), [visibility checks](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/evidence/FINAL_CHECKS.json)
and [representative frames](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/qa/representative_frames.jpg).
The old wide release's80 sealed files were verified unchanged before and after.

Movie bytes:6300268. SHA256:`70220846f958d2067143f313eab637486278460bfaadb9e925b231876fc91556`.
[Verified one-way local pull](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/PULL_TO_MAC.md). Keep frames/caches/sequence links
on the cluster; pull only the final MP4, checksum and caption to the thin mirror.
