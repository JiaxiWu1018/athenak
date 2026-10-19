# Jeans Session010: standalone 3D particles, t=0–98 Mref

Accepted movie: [jeans_session010_particles3d_t0_to98.mp4](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/jeans_session010_particles3d_t0_to98.mp4).
1920×1080, H.264/yuv420p, 12 fps, 393 frames, 32.75 seconds, approximately
3 Mref per playback second. The final encoded frame is the saved t=98 state.
No t=98.2 or later particle cache was used.

The purple envelope and orange/cyan original clumps show surviving members of
the existing fixed t=0 cohort: 40,000 envelope tags and 25,000 per clump. Removed
particles are never replenished. Positions are true xyz; trajectories are never
interpolated. The view is a fixed equal ±24 Mref cube with elevation22°, azimuth−58°,
roll0°, perspective focal length1.25, dark navy background and hidden axes/text.
Colors, marker sizes and opacity come directly from the preserved Session007
standalone renderer on Anta. The [reference comparison](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/qa/reference_comparison.jpg)
shows the matched appearance.

Opaque black spheres use accepted, measured coordinate centers and mean radii,
not mass-based radii or reconstructed horizon shapes. Geometry is joined by
exact native output cycle and case, with tolerance only for printed particle
output times. The native case is recovered from persistent VTK output numbering
and checked against checkpoint cycle ranges. Consumer-matched accepted CSVs
and strict enclosure evidence are checksum-bound. Individual surfaces are used
before strict common enclosure at64.36875; a common sphere is first visible in
the saved t=66 particle state. Missing accepted surfaces are omitted without
extrapolation; this includes several chunk-boundary frames and isolated
individual-surface gaps. A missing marker is not a disappearing black hole.
Particles inside displayed spheres (with the original0.2% rendering buffer)
are suppressed.

Fifty actual saved states through98 were rendered. The video uses a0.25Mref
CFR sampling clock and holds the most recent saved state. The very short t0.00625
state falls between clock samples and has zero encoded duration. t98 is included
for one final frame; no fictitious moving clock or interpolated states are shown.
[TIMING.csv](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/TIMING.csv) gives every encoded frame's saved-state time, cycle,
case and hold clock; [FRAMES.json](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/FRAMES.json) gives counts, AH measurements,
missing-surface reasons and hashes for every source image.

Slurm17088 rendered/encoded, fully decoded the MP4, counted all393 frames and
verified format/rate/resolution/duration. Early collapse, individual horizons,
pre/post-common states and the final frame were inspected. [Acceptance](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/evidence/ACCEPTED.json),
[machine checks](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/evidence/COMPLETE.json), [probe](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/evidence/FFPROBE.json),
[provenance hashes](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/SOURCE_MANIFEST.json) and [representative frames](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/qa/representative_frames.jpg)
are retained. The known t98.2 finder defect is deliberately outside this movie;
no repair, new simulation, production submission or raw transfer occurred.

For a thin local mirror, pull only the MP4, its adjacent `.sha256`, this caption
and optional reports/final figures. Do not pull `frames/`, `sequence/`, caches,
reference movies or native data. Check with:

```sh
shasum -a 256 -c jeans_session010_particles3d_t0_to98.mp4.sha256
```

SHA256: `724d73b0508efa8c81fdd025203770d150fd1ed2ff91ad888da181b305f8ae22`.
Movie bytes:5,005,907. See [pull commands](/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/movies/particles3d_to98_20261019_v1/PULL_TO_MAC.md).
