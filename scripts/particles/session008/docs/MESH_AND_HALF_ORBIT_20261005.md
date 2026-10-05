# Session 008 mesh and half-orbit estimate, October 5

This is a read-only inventory and planning estimate. The approved t=12 assessment
is complete. No longer continuation, changed input, build or new job was enabled.

## Actual configuration at t=12

| Quantity | Value |
|---|---|
| Domain | [-256,256]^3 in inherited coordinate units |
| Root cells / spacing | 256^3 / dx=2 |
| MeshBlock interior / ghosts | 32^3 / 4 cells each side |
| AMR levels | Physical 0–9 (10 total); logical 3–12 |
| Finest actual spacing | 2/2^9 = 1/256 = 0.00390625 |
| Final leaf MeshBlocks / interior cells | 2304 / 75,497,472 |
| Nominal block capacity | 240 per rank; 2880 across 12 ranks |
| Initial particles | 5,000,000: envelope 3M, left 1M, right 1M |
| Surviving particles | 3,348,166: envelope 2,989,262; left 179,362; right 179,542 |
| Removed particles | 1,651,834 at alpha<0.05; AH removal OFF |
| AMD layout | 3 exclusive mi2104x nodes, 12 MPI ranks, 12 MI210 GPUs |
| Evolution | RK4, CFL0.4; final dt=0.0015625, cycle3964 |
| Refinement | Lohner(alpha*psi^7) threshold0.2; tracker_floor=false |
| GW extraction | Complex r*Psi4, ell2..8, r40/50/60/70; cadence0.025 |

The full initial particle count and finest central resolution are the approved
production baseline. The run was a bounded early assessment; long orbital
feasibility and reliable higher-frequency wave propagation are not established.

### Final propagation mesh

The frozen input comment describing an inherited “dx=2 propagation grid” is a
worst-spacing shorthand, not a complete description of the actual mesh.
Inspection of every final leaf block gives these spacings among blocks touching
each origin-centered extraction sphere:

| Radius | Final spacings | Number of intersecting blocks at each spacing |
|---|---|---|
| 40 | 0.25, 0.5, 1 | 24, 92, 8 |
| 50 | 0.5, 1 | 96, 28 |
| 60 | 0.5, 1 | 104, 36 |
| 70 | 0.5, 1, 2 | 76, 52, 8 |

Outer propagation includes dx=2. No additional uniform wave-zone refinement was
introduced. These are block intersections, not solid-angle weighting or evidence
of waveform convergence. Future mesh growth may also reach the block capacity.

Metadata source: final coarsened mesh output
`evidence/half_orbit_estimate_20261005/s8_companion_supported.mesh.00480.cbin`,
264,068 bytes, t12/cycle3964. Only its header and block locations were inspected;
no fine-volume field reduction was performed. Format was checked against
`src/outputs/coarsened_binary.cpp`: 51,865-byte parameter header, 92-byte records
(10 int32 indices/locations/physical level, 6 float64 bounds, one float32 average).
For each block, dx=(xmax-xmin)/32 was checked against 2/2^physical_level in all axes.
A block touches a sphere when its nearest and farthest distances from the origin
bracket the sphere radius. Nearest squared distance is the sum of squared
per-axis distances from zero to the closed block intervals; farthest uses the
farthest interval endpoint on each axis. Physical-level block census:
484,196,196,196,196,188,256,224,240,128 for levels0..9.

This tiny coarsened catalog is retained on Perseus as an explicitly recorded
metadata copy; full particles, fine-volume data and checkpoints remain on their
recorded clusters. Canonical input and executable hashes are unchanged.

## Half-orbit estimate from the measured post-formation arc

Source `evidence/final_analysis_summary_20261005.json` gives:
later first individual-horizon detection t=8.4859375, last live tracker sample
t=11.9937500, post-formation phase=0.01757361398 revolutions=6.326501 degrees.
The phase is a coordinate tracker diagnostic, not a gauge-invariant orbital element.

With constant measured angular rate:

    measured interval = 11.9937500 - 8.4859375 = 3.5078125
    half-orbit duration = 3.5078125 * 180 / 6.326501 = 99.80339
    half-turn endpoint counted from individual formation = 108.28933
    remaining from the saved t=12 state = 96.28933

Using literally rounded six degrees instead gives half-orbit duration105.23438,
endpoint113.72031 and remaining101.72031. A fresh 180-degree arc from t=12 instead
requires approximately99.8 additional time units under the same assumption.

### Recent measured cost and forecast

Use recent three-node continuations, not the cheaper whole-run startup average:

| AMD job | Simulated interval | Elapsed allocation seconds | Nodes |
|---|---|---:|---:|
| 451758 | 9.325–11.925 | 12130 | 3 |
| 451760 | 11.925–12.000 | 619 | 3 |

    rate = 3*(12130+619)/3600 / (12-9.325)
         = 3.97165109 raw node-hours per simulated unit

The allocations include restart/finalization overhead. A node-hour is one
allocated node for one hour; this estimate does not multiply by GPU count.

| Estimate | Using measured 6.3265 degrees | Using rounded 6 degrees |
|---|---:|---:|
| Additional AMD node-hours from t12 | 382.43 | 404.00 |
| Additional elapsed computing hours on 3 nodes | 127.48 | 134.67 |
| Additional elapsed computing days | 5.31 | 5.61 |
| Cumulative session node-hours, including existing32.30889 | 414.74 | 436.31 |

Queue waits, additional analysis and a revised wave-zone/boundary hierarchy are
excluded. Continued accretion, changing orbital motion and AMR growth can change
both angular speed and cost. A six-degree arc does not establish circularity,
an eventual half-turn or merger. This estimate exceeds the approved48-node-hour
assessment cap and is not permission to launch a longer calculation.

Times are G=c=1 with the inherited source reference M_ref=1, not a measured ADM,
rest or horizon mass. Physical seconds require an assigned mass scale:
t_seconds = (t/M_ref)*G*M_ref_physical/c^3. No physical mass scale is assigned here.
Wall-clock days above describe computer time, not the modeled system's lifetime.

