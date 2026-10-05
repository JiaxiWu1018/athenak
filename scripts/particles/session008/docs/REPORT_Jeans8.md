# Jeans-in-cluster Session 008 assessment

Fresh companion-supported initial local boosts: left -y, right +y, magnitude 0.133215. This approximate prescription is not exact GR binary equilibrium. Five million initial particles, source masses 0.76/0.12/0.12; M_ref=1 is the inherited source unit. Approximate K_ij=0 leaves the local momentum constraint unsolved.

Workflow status: **complete_t12**. Last verified checkpoint time: **12.0**. The approved early assessment used four successful evolution segments after one failed startup, within 48 raw AMD node-hours including preparation. No campaign jobs remain active.

## Compiled initialization ledger

| Component | Count | Sampled rest mass | Sum(mW) | P_cov,y | J_origin,z |
|---|---:|---:|---:|---:|---:|
| envelope | 3000000 | 0.7821717958 | 0.7961148546 | 0 | 0 |
| left | 1000000 | 0.1312399829 | 0.1324996133 | -0.02152634 | 0.064518169 |
| right | 1000000 | 0.1312399829 | 0.1324996382 | 0.02152597 | 0.064513473 |

Total covariant momentum residual: [0.0, -3.700000000030068e-07, 0.0]. Independent written-particle checks: {'exact_counts': True, 'tags': True, 'finite': True, 'positive_weights': True, 'inside_domain': True, 'ledger': True, 'signs': True, 'Jz': True, 'centers': True, 'widths': True, 'thermal': True}. These sampled quantities are distinct from source masses, horizon masses, and an ADM estimate.

Both individual horizons detected: True; later first individual acceptance time: 8.485937500000432; measured coordinate revolutions after both form: 0.017573613981682643. These are coordinate diagnostics; gaps are flagged. No merger or radiation-driven inspiral claim is made.

The assessment precedes central collapse signals at r=40–70. Raw complex r*Psi4 is retained; strain and merger/ringdown interpretation are deferred. The inherited propagation mesh does not establish high-frequency waveform accuracy.

## Review products

- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331/orbit.png` — Coordinate trajectories, separation and phase within live intervals; inspect gaps and orbital arc.
- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331/radial_tangential.png` — Coordinate angular frequency and radial/tangential relative-motion ratio; no derivatives bridge gaps.
- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331/envelope_relative.png` — Trajectories relative to the rest-weighted sampled envelope center.
- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/horizon_review_20261005/accepted_horizons.png` — Strictly published same-candidate masses and coordinate spin; absent measurements are gaps.
- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331/particle_components.png` — Alive component counts and covariant matter angular momentum; no conserved total is implied.
- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331/constraints.png` — Proper-volume history norms with the code chi mask, not an AH exterior.
- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331/raw_waveform.png` — Both (2,+/-2) early outer-field/initialization response, not a merger waveform.
- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331/central.mp4` — Central fixed-tag particle projection and accepted rmin illustrations.
- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331/context.mp4` — Envelope/context fixed-tag view.
- `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/assessment_2331/density.mp4` — Full-particle coordinate rest-mass slab projection on a fixed Cartesian grid; no native-AMR seams.

Movies passed encoder/decode checks. Representative final frames of all three movies, the orbit/motion/constraint plots and the corrected horizon plot were inspected on October 5. Component colors, axes and times are consistent; the density view uses a fixed grid. The small dashed horizon circles illustrate rmin, not measured projected horizon shapes. Source, logs, complex multipoles, masks, normalization, cadence and restart validity are preserved with checksum manifests.

## October 5 reviewed results

The continuation reached verified **t=12**, cycle3964. The left and right black holes remain separate: no common candidate was published. They move counterclockwise through a small arc; the coordinate diagnostic records **0.01757 revolutions (about6.33degrees)** after both individual horizons are first accepted near t=8.486. This is insufficient to establish sustained orbiting. Separation near the last tracker sample is **6.088** reference units; the available evolution does not show sustained shrinking separation. No radiation-driven inspiral is claimed.

The corrected last accepted horizon measurements near t=11.99844 are:

| Object | Horizon mass / M_ref | Coordinate spin chi |
|---|---:|---:|
| Left | 0.111209 | 0.003144 |
| Right | 0.110847 | 0.005329 |

These horizon masses differ from the fixed source/model parameters0.12 and the sampled rest masses. Spin is small and fluctuates; its coordinate prescription and numerical limitations remain relevant. Masses continue growing during this early collapse/accretion interval. The surviving particle count is **3,348,166**, with **1,651,834 lapse removals**; removals are not measured BH masses.

The original empty horizon plot was a post-processing error: the summary's first column is a finder iteration, while the acceptance table's first column is the simulation cycle. The corrected plot uses unique same-run/time/area/rmin/center matches to strictly accepted candidates. All **2,127 left** and **2,167 right** accepted records matched, with no missing or ambiguous matches. The original plot/summary are retained as superseded evidence; use the corrected horizon directory above.

The assessment is too short for collapse radiation to reach radii40–70. Raw complex r*Psi4 is archived; a merger waveform, strain, radiated-energy estimate or a completed inspiral is not claimed. A longer orbital assessment still needs a finite approved endpoint, boundary/wave-zone design and resource budget.

Final actual AMD allocation usage is **32.308889 raw node-hours** out of48. At October5 17:18UTC, AMD session storage was **78.853GiB**, predominantly the latest three verified checkpoints and retained evidence; Anta /data3 science/archive storage was **123.030GiB**. All four bounded archive jobs completed, and the metadata cron entry removed itself. No new continuation or archive job is enabled.

## Requested separation plot, October 5

A standalone plot is at `/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/analysis/separation_20261005/separation_vs_time.png` (PDF and measured-point CSVs in the same directory). It shows full-clump surviving-particle mass centroids, the existing local-lapse trackers, and strictly accepted individual AH centers; a second panel enlarges the AH curve. These are distinct center definitions.

The clump mass-centroid separation changes5.99922→6.03051 over t0–12 (0.52147% endpoint increase). The2114 simultaneous accepted AH pairs cover t8.64531–11.99844 and range6.06036–6.08993 (0.48650% range/mean,0.44396% endpoint increase). One-sided/restart/failed-acceptance gaps remain missing data. Both individual horizons had each been detected by t8.48594, but their first simultaneously accepted pair is later,t8.64531. No stale center was substituted during this interval.

This small variation is consistent with predominantly tangential early motion, but does not establish a nearly circular orbit across the observed short arc (~6degrees after individual formation). Distance is coordinate distance, not proper distance. Remaining-particle centers can shift as particles are removed; grid-based AH centers show small stair steps. The original orbit.png middle panel already plotted tracker separation and is retained.

## Mesh and half-orbit planning estimate, October 5

The final mesh has2304 leaf blocks, physical levels0–9, dx_min1/256, domain[-256,256]^3 and5million initial particles. It uses the approved production particle/central-resolution baseline, but only the bounded t12 assessment has run. Final extraction spheres cross mixed spacings: r40 dx0.25–1, r50/60 dx0.5–1, r70 dx0.5–2. The frozen input's blanket dx2 propagation comment was an incomplete shorthand; no additional uniform wave-zone refinement or high-frequency accuracy is established.

Extrapolating the measured6.3265degree arc over3.5078 time units gives about100 units for half an orbit after both individual horizons were detected, endpointt108.3. From the savedt12 state, recent measured costs imply about382 additional raw AMD node-hours,5.3 computing days on3nodes,415 cumulative session node-hours. Literal rounded6degrees gives about404 additional node-hours and5.6days. These assume unchanged angular speed and cost; continued accretion, AMR growth and any revised wave-zone hierarchy can change them. No longer run is enabled or approved by this estimate. See MESH_AND_HALF_ORBIT_20261005.md for source metadata, arithmetic and physical-unit conventions.
