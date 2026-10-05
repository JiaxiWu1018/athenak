# Authorized benchmark specification

User approved 2026-10-05. Part I isotropic Einstein–Vlasov construction at eta=0.10,
M=1, a=10; benchmark only, no compactness scan. Total AMD allocation ceiling:
200 node-hours, including preflight and reductions. Stop for discussion if the
measured required matrix cannot fit.

| Arm (each matched frozen/live) | Spatial sampling cut | N | dx finest | CFL | Coverage |
|---|---:|---:|---:|---:|---:|
| Main | 100a | 2113536 | a/32 | 0.25 | 5 P_ref |
| Tail | 200a | 2113536 | a/32 | 0.25 | 5 P_ref |
| Coarse central mesh | 100a | 2113536 | a/16 | 0.125 | 2 P_ref |
| Lower particle count | 100a | 528384 | a/32 | 0.25 | 2 P_ref |

All use seed 1985, deterministic stratified radial sampling and co-located
opposite-momentum pairs. Draw independent uniform position and momentum directions.
Use the full reconstructed F(e), not a velocity-dispersion proxy. Spatial radial
measure is rho0 4 pi r^2 psi^6 dr. Conditional momentum measure is q^2 F(alpha W)dq.
Rest mass is finite-cut M0/N. Retain the untruncated analytic metric and asymptotic
M; report omitted-tail mismatch. No sampling-radius wall, removal or excision.
Circular initialization remains the default option.

P_ref is the coordinate circular-geodesic period at the untruncated rest-mass
median; it is a common clock, not a period common to isotropic orbits. Show t/M
and t/P_ref. Root 128^3, block 32^3, domain [-2048a,2048a]^3, ten nested fixed levels
to half-width 2a, expected 624 leaf blocks. Coarse omits the innermost level,
preserving the baseline timestep. Verify through Athena's mesh inspection.

Keep established BSSN (use_z4c=false), gauge, Hamiltonian damping, conservative
deposition and GR-Boris. AMD ROCm 6.4.1, four ranks/four MI210, four-hour restart
segments with finalization allowance. Preserve period and segment checkpoints.

Preproduction gates: independent Python/C++ metric, full F, moments, M0 and CDF
agreement <=1e-8; sampled density and all three physical stresses consistent with
continuum shell averages and measured sampling uncertainty; exact particle count
and initial pair cancellation; matched 0.25-P_ref pilot; restart continuity;
short half-timestep comparison; measured throughput, VRAM, timestep bounds and
output sizes; causal boundary/gauge arrival, diffusion and extraction placement.
Frozen validation tests E and vector L conservation. Changing radii and cohort
mixing are expected.

One health policy is shared by runner, continuation, reducers, figures and movies.
Abort on non-finite particle/field states, particle loss, invalid positive-definite
spatial metrics or relative rest-mass error >1e-10. Checkpoint and stop for minimum
lapse <0.2 or constraints >10 times the pilot post-transient reference at three
consecutive histories. Do not restart completed endpoints. Resource retries require
a verified healthy checkpoint. Bound products at last healthy time.

Retain five-minute whole-user-root AMD storage monitoring: warn at 1.5 TiB, stop
growth at 1.7 TiB, prevent projected crossing of 1.9 TiB. Preserve unrelated jobs.
Record actual growth. Do not automatically delete campaign data.

Record density, rest mass, three metric-orthonormal stresses, anisotropy, enclosed
mass radii, constraints, particle count, fallback counts and finite-radius ADM mass
and momentum. Coordinate velocity dispersions remain auxiliary. COM-referenced
l=1..4 amplitudes and vectors, global and radial bands, are primary; compare origin
results as auxiliary. Compare matched frozen time series; never normalize a band
by its initial amplitude. Report sampling uncertainty and control coverage.

Targets: <=1% bulk-radius and integrated-stress drift; <=2% density/stress profile
changes in resolved/populated bins; comparable 100a/200a agreement; no unresolved
health failure or sustained constraint runaway. Show full startup, assess secular
behavior after 0.25 P_ref. Uncertainty preventing a tolerance decision is inconclusive.
The short controls establish coverage only through 2 P_ref.

Classification: equilibrium within targets; near equilibrium with measured drift
or growth; substantial departure; numerically compromised or inconclusive. Fit an
exponential only above frozen fluctuations, spanning >=2 e-folds and >=8 independent
correlation blocks with consistent windows. Otherwise report amplitudes, temporal
behavior and detection limits. Healthy runs or N agreement do not prove continuum
stability or that the circular-cluster instability is physical.

Cadences: histories/compact ledgers P_ref/200; full particles P_ref/100; native cell
movie slices P_ref/50; coarsened ADM P_ref/10. Cumulative milestone figures cover
initial fidelity, profiles, radii, stresses, angular modes, frozen invariants,
conservation and health. Main movies: fixed-subset matched frozen/live particles;
live density/lapse/constraint slices. Fixed scales; conservative pixel averaging.
Scientific/technical reports and a note after every period include failures,
uncertainties, exact provenance, jobs, manifests and coverage.

Builds, simulations, numerical validation and reductions run through Slurm.
AMD hosts production/reductions; Perseus hosts final figures/movies and authoritative
records. Retain raw output on AMD during execution. Archive completed large outputs
to Anta under /data2/jiaxiwu/ with checksum verification. Record exact paths, sizes
and retention status. Later cleanup needs an explicit reviewable scope.
