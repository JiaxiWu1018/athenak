# Efficiency measurement design

This is a numerical workload campaign on Perseus A100 GPUs. It uses ST initial geometry with `y_c=0.819` and unit gravitational mass, plus a flat-metric control. One MPI rank uses one GPU and two allocated CPUs; host sorting uses one thread. Rank scaling therefore measures the combined GPU/CPU allocation. Every main case targets 5M and uses a fixed 0.05M timestep; a wall-time guard can stop a large case earlier, with its actual endpoint reported. No scientific snapshots or restarts are requested.

| Comparison | Mesh | Particles | MPI ranks | Controlled variation |
|---|---|---:|---|---|
| Particle count | One 32³ block, four ghosts, periodic ±4M | 10⁴, 10⁵, 10⁶, 10⁷, 10⁸ | 1 | ST/flat, deposition on/off |
| Block layout | Global 64³, periodic ±4M | 1,000,000 | 1 | 1×64³, 8×32³, 64×16³; ST/flat and deposition on/off |
| Rank scaling | Global 64³, 8×32³ blocks, periodic ±4M | 1,000,000 | 1, 2, 4 | Matching initial particle realization, ST/flat, deposition on/off |
| Particle/mesh cost | Global 64³, 8×32³ blocks, outflow ±12M | 6, 24, 96, 384 per cell | 1 | Dynamical ST; mesh-only and matched frozen controls |
| Number distribution | Same dynamical mesh and box | 6,291,456 | 1, 2, 4 | Uniform, Gaussian σ=4M/1M, extra particles in one block/site |

Repeated baselines are reused rather than rerun, giving 60 throughput cases. The native Kokkos layouts and production push/interpolation/deposition/migration algorithms are retained. Flat and frozen ST cases use the same counter-based random draws for positions, positive masses and local momentum magnitudes/directions. Their total random mass is normalized in expectation; these small-box workloads are deliberately not self-consistent cluster realizations.

The dynamical box encloses the ST isotropic cluster radius 8.93623112M with a modest boundary buffer. Each physical cell has a six-direction, equal-mass monoenergetic template. Higher particle counts split each template into identical weighted replicas. The cell mass follows the ST rest density times proper volume, normalized to the common rest mass 1.030106482M. Vacuum cells keep zero-mass particles in the workload. Concentrated cases retain the six-particle floor everywhere and redistribute the additional replicas, adjusting their masses so the initial cell source remains fixed. At 24 particles/cell, up to 75% of all particles can therefore be placed at one source site without removing the source elsewhere. The centered radial sampling controls retain approximately balanced octant populations; the one-block and one-site controls deliberately stress block/rank imbalance. Comparing these two hotspots at the same rank count also tests spatial concentration within the overloaded block.

Timing excludes initialization, audits and the first 0.5M. GPU fences bound the measurement window; the reported elapsed time is the maximum across MPI ranks. Rates count completed full timesteps, not RK substages:

- Particle updates/s = total live particles × measured cycles / maximum elapsed seconds. Mesh-only reports zero particle updates.
- Cell cycles/s = physical cells × measured cycles / maximum elapsed seconds. Ghosts are excluded; for fixed spacetime this is a mesh-size normalization, not spacetime evolution.
- Seconds/M = maximum elapsed seconds / measured physical-time interval; M/s is its inverse.

Main throughput uses the verified uninstrumented binary. The 27 phase profiles use separate 1.5M runs and synchronized phase timers. Six additional runs use a separately compiled sampled device-clock pusher. Nsight traces identify kernel execution time and MPI activity; phase timers also capture CPU sorting, host/device copies and waits. Nested deposition timers are subtracted before phase ranking. Interpolation fractions describe sampled geodesic-evaluation device cycles and are not percentages of the production wall time. Single passes have no replicate-based confidence intervals; node sharing, clocks and resource samples are retained.

Every promoted result must pass count/tag checks, finite particle/geometry checks, ownership, zero destruction and the appropriate compiled-mode check. Frozen cases additionally require unchanged geometry hashes, including ghosts and both pusher time levels. Replica cases check deposited source against the common cell reference. The short build gate compares aligned full frozen states, exact tags/masses, bitwise hashes, zero/nonzero device-clock counters and a captured Nsight timeline. Failed and interrupted attempts remain preserved. Reports distinguish incomplete coverage from completed results.
