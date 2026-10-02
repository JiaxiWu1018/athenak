# Jeans-in-cluster Session 008 — setup and queued assessment

## Scientific setup

Session 008 starts from fresh initial data. The only intended physical change from Session 007 is the approved initial local orthonormal boost: **0.133215**, directed along −y for the left clump and +y for the right clump. Both move counterclockwise viewed from +z. The existing problem generator applies a Lorentz transformation to the thermal samples and regenerates their weights and covariant momenta. No force was added to the evolution.

| Component | Source mass | Center | Initial particles | Local boost |
|---|---:|---|---:|---|
| Envelope | 0.76 | (0,0,0) | 3,000,000 | unchanged |
| Left clump | 0.12 | (−3,0,0) | 1,000,000 | (0,−0.133215,0) |
| Right clump | 0.12 | (+3,0,0) | 1,000,000 | (0,+0.133215,0) |

The independently solved envelope has areal radius 30. Clump centers use isotropic Cartesian coordinates, with Gaussian width 0.70 and internal orthonormal momentum spread 0.02. Sampling, seed 4001, and immutable component tags are retained. No new spin, radial boost, symmetry, or envelope recoil was imposed.

The companion correction motivates the speed approximately; it is not an exact GR circular orbit. The prescribed conformal geometry depends on the source profile, so changing only the boost does not change that analytic geometry. It is nevertheless freshly initialized, and boost-dependent weights and momenta are regenerated. Approximate `K_ij=0` leaves the local momentum constraint unsolved. Opposite global boosts do not solve it.

All plotted times use the inherited reference unit `M_ref=1`. The source normalization 0.76+0.12+0.12=1 does not equate sampled rest mass, horizon mass, and a measured ADM estimate.

## Numerical setup and assessment limits

The Session 007 baseline is retained: domain ±256, root spacing 2, 32³ cells per MeshBlock, compact initial refinement around both clumps, minimum spacing 1/256, Löhner refinement of `alpha*psi^7` at threshold 0.2, and `tracker_floor=false`. The logical root level is 3; physical levels 0–9 give logical levels 3–12. Initial towers reach spacing 1/64. RK4/CFL 0.4, gauge/damping, conservative deposition/feedback, and pusher protections are unchanged. Particles are removed at `alpha<0.05`; AH-driven removal is OFF.

The approved assessment uses three AMD MI210 nodes and twelve MPI ranks. It targets **t=12**, bounded by **48 total raw AMD node-hours**, including preparation. The submitted chain reserves at most 44.5 raw node-hours. There are at most three four-hour evolution jobs and no automatic retries or extension toward t=50.

The initial coordinate period in the combined metric is approximately 227 reference time units, using initial coordinate tangential speed about 0.08312 from `alpha*v_local/psi^2` with zero initial shift. This estimate does not change the approved local boost. This stage can assess collapse and an early orbital arc, not a full post-collapse revolution. The inherited Sommerfeld/outflow boundary and approximately sqrt(2) outer gauge speed put the boundary-to-envelope estimate near t=160, beyond this stage.

Raw complex `r*Psi4`, ell=2–8 at coordinate radii 40/50/60/70, is retained every 0.025 time units. A central collapse signal from around t=8 cannot reach those radii by t=12. The inherited propagation mesh also does not establish high-frequency merger-wave accuracy. No inspiral, merger, strain, ringdown, or precision waveform result is claimed.

The outer propagation spacing is 2. A twenty-cells-per-wavelength heuristic corresponds to frequency at most about 0.025 in the inherited units; ten cells correspond to 0.05. These are planning estimates, not convergence evidence. The initial quadrupolar frequency estimate is about 0.0088, but the shortened assessment cannot yet test its radiation at the extraction spheres. A longer waveform production stage requires a separate duration, boundary and wave-zone decision.

## Completed work and current status

- The independent checkout includes restart-header repair `263dcf21`, recovered from a checksum-verified source bundle. The approved input and operations changes were pushed to `project/GI-in-cluster`.
- AMD build **447620** completed successfully in 313 seconds with ROCm 6.4.1, HIP GFX90A, and the established GNU/OpenMPI toolchain.
- Fifteen checks passed in Perseus Slurm job 10327: twelve controller/checkpoint checks and three analysis checks. They cover unfinished-write protection, orbital diagnostic gaps, both (2,+/-2) columns and a synthetic rendering fixture. The AMD gate repeats the twelve operations checks before numerical validation.
- Full five-million-particle GPU validation **447655** is **queued for resources**. Production jobs **447657, 447659, 447661** are queued behind its acceptance gates. No Session 008 simulation has started and no scientific findings or final plots exist yet.

## Storage and later review

Science data will be retained at:

`anta:/data3/jiaxiwu/NRPIC/GI_in_cluster/session_008_companion_supported_orbit_gw_20261002/`

At inspection, `/data3` had 27.10 TiB available. AMD retains the latest three verified Session 008 checkpoints total, without a Perseus backup. Large completed science binaries are removed from AMD only after Anta independently verifies their checksums. Logs, inputs, manifests, waveform streams, and failure evidence remain.

Anta requires a GPU request for every Slurm job. A finite metadata trigger submits allocations only when transfer or analysis work is ready; it does not reserve a GPU while waiting. The trigger expires after 48 hours and can launch at most four six-hour jobs. Heavy processing remains in Slurm.

After verified data arrive, the pipeline produces absolute/envelope-relative orbit plots, radial/tangential motion, accepted horizon diagnostics, particle counts and covariant angular momentum, constraint histories, both (2,+/-2) waveform plots, and central/context/density movies. Phase and derivatives do not bridge diagnostic gaps. Density uses full-particle rest weights on fixed Cartesian bins and is labelled as a coordinate slab surface density. Horizon illustrations use accepted coordinate `rmin` circles. Final reports are generated on Anta; representative real frames and plots still require visual review. See README.md and HANDOFF.md for exact commands and paths.
