# Session010 close-up movie — October 19, 2026

User requested a zoom to keep the two initial holes visible. The accepted
[close-up MP4](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/jeans_session010_particles3d_t0_to98_zoom.mp4)
uses a fixed **±4 Mref cube**, six times closer in field of view than the
original ±24 view. Camera angles, perspective focal length, colors and particle
marker sizes/opacity are unchanged. The maximum accepted AH coordinate extent
is2.92631 Mref, so all accepted spheres fit in the fixed cube. Every projected
sphere also fits inside the image. Twenty saved states have both accepted
individual markers; their projected bounds remain separated in all twenty.

The initial zoom preview retained Matplotlib's collection sorting and showed
particle clouds painting over one AH. The final version draws schematic opaque
black AH spheres above particle collections. Their measured centers and mean
coordinate radii are unchanged; this is an annotation visibility convention,
not a ray-traced opacity calculation. No sphere radius is enlarged. The same
fixed cohort, original xyz, case/cycle-matched accepted geometry and quality
gaps are used. No replacement, resampling, interpolation or rejected-surface
promotion. Particles outside the central cube are clipped from this close-up.

There are no accepted horizons at t0: the first appear near8.43/8.49, and the
first saved particle state showing both is t10. Missing accepted measurements
remain gaps. After strict common enclosure64.36875 the movie uses the common
sphere only, first at saved t66; two individual spheres are not continued after
merger. The final image is saved t98, with no t98.2/later state. Timing is
byte-identical to the original:393 frames,12fps,32.75s,3Mref/playback second.
H.264/yuv420p at1920×1080; no axes/text or side panel.

Render Slurm17387 is a preserved, unpromoted preview; its exact script is retained
in that version's evidence directory. Slurm17392 rendered the final close-up,
fully decoded it, verified format/frame count and extracted QA. Visual review
covered early formation, pre-merger pairs including t62, common and final t98.
Check17394 initially assumed pure RGB0 for the tiny antialiased sphere facets;
probes17396/17397 identified that invalid raster-check assumption. The final
visibility check17398 allows the expected dark antialiased pixels, checks all
56 markers, all20 separated pre-common pairs, all50 saved states and identical
timing/geometry/gaps. Actual pixel thresholds and diagnostic logs are retained.
The MP4 itself was not changed by that checker correction. Native source,
inputs and finder remain unchanged; no simulation restart or repair.

[Final checks](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/evidence/FINAL_CHECKS.json) ·
[Caption](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/CAPTION.md) ·
[Verified local pull](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_zoom_20261019_v2/PULL_TO_MAC.md).

Movie bytes:6300268; SHA256:`70220846f958d2067143f313eab637486278460bfaadb9e925b231876fc91556`.
The original wide movie's80 sealed files passed before/after hashes. All earlier
files and previews remain on Perseus. Final-version regular-file snapshot before
sealing:24580453 bytes, with393 internal sequence
links. No native raw generated, data deleted, raw transfer or Mac pull performed.
Code/reports are committed and pushed; the exact Git receipt is under
`evidence/movie_zoom_20261019/GIT_RECEIPT.json` at the session root.
