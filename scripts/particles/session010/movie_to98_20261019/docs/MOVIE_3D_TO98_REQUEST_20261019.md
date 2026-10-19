# Copyable request: Session010 3D particle/AH movie, t=0–98Mref

Generate a new movie on Perseus for Jeans/GI-in-cluster Session010, using only
the existing archived data and ending at **t=98Mref inclusive**. This request is
for visualization; do not restart or modify the simulation or repair the finder.

Session root:
`/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/`

Data directory beneath that root:
`session010_summary_20261019/workspace/runtime_s10/analysis/through_segment002_20261018/`

Match the old standalone 3D movie on Anta:
`/data2/jiaxiwu/NRPIC_test/s7_plotting_bundle_20260911/cca_talk_20260912/revision_20260912g/slide27_jeans_particles_3d.mp4`
Its renderer is `scripts/cca_render_gi_3d_movie.py` in the same revision directory.
Read that renderer and adapt its appearance to Session010's cache schema.

- True 3D particle view; dark navy `#071A2B`, purple envelope `#8E5AB5`, orange
  clump A `#FF9F1C`, cyan clump B `#31B7E5`. Match the old marker sizes/opacity,
  fixed camera (elevation 22°, azimuth −58°, roll 0°, perspective focal length
  1.25), equal fixed ±24Mref cube, hidden axes and clean presentation. Use the
  standalone view, without a tracker side panel.
- Read `cache/particles/*.npz`: time, xyz, tag and group. Preserve the fixed
  t0 cohort (40k envelope, 25k per clump); removed tags stay absent. Use only
  actual cached states with **time<=98** and include the saved t=98 state as the
  final image. Exclude t=98.2 and every later state. No new random resampling,
  particle replacement or interpolation of trajectories.
- Show accepted apparent horizons as opaque black spheres using their measured
  coordinate centers and mean radii, as in the old movie; suppress particles
  inside each displayed sphere. Use accepted_horizon_0/1/2.csv, the consumer
  gates and strict common_enclosure.json evidence. Show individuals before the
  common horizon, then the accepted common sphere after strict enclosure near
  64.36875Mref. Match overlays to each displayed saved state and native case/time;
  preserve gaps, never extrapolate a rejected/missing surface, and do not use
  mass as a coordinate radius. Spheres are schematic AH markers, not full shape
  reconstructions.
- Export 1920×1080 H.264/yuv420p MP4 at 12fps. Match the old pacing of about
  3Mref per playback second by holding actual snapshots for their time intervals;
  retain a time mapping/sidecar that makes holds and actual source times clear.
  Keep the last frame at t=98. If a visible clock is added, label the actual saved
  state time, not a fictitious interpolated state.
- Render and encode in Slurm under a new versioned Session010 folder. Preserve
  the old releases and all source data. Inspect early collapse, both individual
  horizons, pre/post-common states and the final t=98 frame; verify no frame
  uses later data. Fully decode the final MP4, verify frame count/rate/resolution,
  and record source/data hashes, time schedule, selection, camera and AH rules.
  Add a brief caption/report and make the accepted MP4 available for a checksum-
  verified thin-mirror pull. Do not copy caches or bulk frames to the Mac.

Read `notes/AH_RESTART_DIAGNOSIS_20261019.md` first. The known t≈98.2 finder
restart defect remains unfixed; this movie's cutoff is deliberately t=98.
