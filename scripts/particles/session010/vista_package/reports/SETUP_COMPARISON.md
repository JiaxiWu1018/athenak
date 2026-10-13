# Domain, grid and cost comparison

Session007 actually used ±256, root 256³/dx2, 32³ cells per block with four ghost zones, ten root-inclusive levels, central ceiling 1/256 and 1,352 initial leaves. Its boost 0.08801247 differs from the preserved Session010 0.133215; its R40 spacing dx2 does not retain Session009's fine wave band.

Session009's actual R40/R46 setup used ±1024, root 256³/dx8, twelve levels, the same central ceiling, and 5,384 initial leaves. The obsolete R50 setup had 7,456. Verified final progress is t=9.33125, cycle 2256, 5,020 blocks and 3,996,300 survivors; controller t=0.03125 is stale. Session010 omits that checkpoint and starts fresh.

Exact dyadic crops of the archived tree retain all physical wave/central spacings, without cutting leaves. Native mesh-only job 12809 independently confirmed all five counts below using preserved exact inputs and the repaired executable; evidence/mesh_options/ contains the inputs and logs. This adds no evolved physics trial.

| Half-box / root dx (root 256³) | Initial blocks | Interior cells | Saving vs S9 | Active-grid HBM proxy | Modeled t100 wall hours on 16 GH | Modeled node-hours |
|---|---:|---:|---:|---:|---:|---:|
| 1024 / 8, actual S9 | 5,384 | 176,422,912 | 0% | 845 GiB | 48–80 | 766–1,276 |
| 512 / 4, approved S10 | 4,936 | 161,742,848 | 8.32% | 775 GiB | 44–73 | 702–1,170 |
| 256 / 2, retain fine waves | 4,488 | 147,062,784 | 16.64% | 705 GiB | 40–67 | 638–1,064 |
| 128 / 1, retain fine waves | 4,040 | 132,382,720 | 24.96% | 634 GiB | 36–60 | 575–958 |
| 256 / 2, actual S7 hierarchy | 1,352 | 44,302,336 | 74.89% | 212 GiB initially | 20–33 | 313–521 |

The approved 4,936 blocks separate as follows:

| Region / physical dx | S9 blocks | S10 blocks | S7 blocks |
|---|---:|---:|---:|
| Coarse exterior: dx8/4/2/1 | 448 / 448 / 448 / 296 | 0 / 448 / 448 / 296 | 0 / 0 / 496 / 112 |
| Wave band: dx0.5/0.25 | 1,512 / 1,712 | 1,512 / 1,712 | 112 / 112 |
| Finer central seeds: dx0.125/0.0625/0.03125/0.015625 | 112 / 104 / 176 / 128 | 112 / 104 / 176 / 128 | 112 / 104 / 176 / 128 |

S10 therefore has 1,192 coarse, 3,224 wave-band and 520 finer central blocks. Cutting S9's 1,640-block exterior saves 448 blocks, independently of box-volume ratios. The wave band dominates. S7's much lower cost comes mainly from changing the wave mesh, which was not selected.

Every option retains initial minimum dx=1/64 and dt=0.00625, then a ceiling dx=1/256 and dt=0.0015625 at CFL=0.4. The solver uses global timesteps without level subcycling. Smaller boxes do not increase these timesteps; evolution changes when the ceiling is populated.

HBM is an incremental proxy of 0.157 GiB/block, inferred from the actual Vista capacity ladder: 360/400/440/480/520 reserved blocks/rank produced 60.87/67.36/73.83/80.17/86.77% peaks on 96 GiB-class GPUs. Reserved capacity, particles, communication, caches and residual growth determine actual memory. At 400/rank the historical startup peak was about 64.7 GiB/GPU; 16×400 reserves 6,400 blocks regardless of active count. These proxies neither certify final Session010 memory nor select its rank count/capacity.

The timing proxy uses actual Vista job 984836: sixteen GH nodes, 2:05:01, 5,888 cycles and t=14.98125. Sparse coarse-mesh headers at cycles 0/307/1420/2700/3980/5260/5888 give 1,352/1,254/2,080/2,080/2,360/2,360/2,164 blocks. Trapezoidal weighting gives about 12.20 million block-cycles, or 1,627 block-cycles/s. Planning uses 1,200–2,000 block-cycles/s and 64,000 steps per t100 at the finest ceiling. The S7-style estimate allows 2,200 evolved blocks. These are conditional planning ranges, not confidence intervals or measured Session010 performance, and t100 is not an approved limit. Changed boost, AMR occupancy, MPI, outputs, buffer repair, checkpoint segmentation and I/O can substantially alter cost. Vista must replace them with measured rates. Under ideal doubling of throughput, 32 GH nodes would halve the approved-box wall estimate to 22–37 hours for t100 while leaving node-hours similar; that is an unverified scaling scenario, not a measured alternative. Short full-mesh timing gates should decide whether extra nodes meet the priority of meaningful evolution quickly.

## Boundary and gauge arrival

Assume a boundary-origin disturbance at t=0, the closest face, a nearly flat outer metric with negligible shift, and coordinate speed 1 (light) or sqrt(2) (1+log gauge). Coordinate R30 is an envelope-edge proxy, distinct from the model's areal R30.

| Half-box | R40 light / gauge | Envelope R30 light / gauge | Clump R3 light / gauge |
|---|---:|---:|---:|
| 1024 | 984 / 696 | 994 / 703 | 1021 / 722 |
| 512 | 472 / 334 | 482 / 341 | 509 / 360 |
| 256 | 216 / 153 | 226 / 160 | 253 / 179 |
| 128 | 88 / 62 | 98 / 69 | 125 / 88 |

An origin-emitted light pulse reflected back to R40 has path length 2L−40, giving 2008/984/472/216 plus emission time. Initialization and boundary errors can arrive before reflected central radiation. Metric, shift and gauge evolution change coordinate speeds; these are estimates, not rigorous isolation or amplitude bounds. Boundary contamination is an accepted exploratory risk, and its amplitude is not proved small. Saved far-field metric/shift and validity coverage must inform interpretation on Vista.
