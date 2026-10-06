# October6 radius and Vista assessment; Session009 status

## Live progress

Checked October6,2026 around13:35Pacific (20:35UTC). Full startup452580 ran for12seconds on12AMDnodes and failed1:0 before particle initialization. Its first evolved-case command requested output7/last_time=0, but the canonical input lacks this optional key; the strict command-line parser rejects it. This is a startup-script/input-contract error, not a failed physical evolution or MPI communication failure. All48MPI ranks passed the all-to-all check; the actual mesh-only inventory is7456blocks, physical levels0..9 initially populated, initial finestdx1/64, configured ceiling1/256. Full GPU memory/evolution/restart checks remain pending.

Inspector452582 completed0:0 after1second, set configuration_failure and prevented successors. No Session009 binary checkpoint or scientific result exists; physical time remains0. Anta2353 archived the sealed failed startup evidence and generated analysis/update_final_2353; the finite trigger removed its own cron after the terminal report. This final report closes a failed startup, not a completed science run. No retry or migration was performed in this assessment. AMD cumulative actual allocation is0.920rawnodehours including build/wave/convention/fullgate/inspector.

The startup fix must respect immutable canonical input bindings and preserve the failed directories/receipts. A suitable operations repair would prepare explicitly recorded per-case startup inputs containing any optional command-line output keys, or stop passing missing optional keys. Validate the full override list before another expensive allocation. Do not change boost/physics or silently resurrect terminal state.

## Exact initial block comparison

Perseus allocated mesh-only job11135 completed allfive cases with no particle initialization or evolution. Retained CUDA executable is Session007 build6511, SHA recorded in radius_assessment_20261006/provenance.txt. Git comparison establishes src/mesh/build_tree.cpp and meshblock_tree.cpp unchanged between that executable's source36b64e23 and Session009. The baseline reproduces actualAMD7456blocks. Full counts/logs/hypothetical inputs/script are in evidence/radius_assessment_20261006/. Canonical Session009 input remains untouched.

Keep domain±1024, root256cubed,32cubedblocks, central towers and all coarser outer wave layers fixed. Change only detector radius, finest wave-region radius and matching level5initial seed cube in the hypothetical inputs.

| Detector radius | Requested dx<=.25 extent | Initial blocks | Saved vs7456 | Fraction |
|---|---:|---:|---:|---:|
|50 current |56 |7456 |0 |0 |
|40 |40 |5384 |2072 |27.7897% |
|30 |30 |4320 |3136 |42.0601% |
|40 with6unitbuffer |46 |5384 |2072 |27.7897% |
|30 with6unitbuffer |36 |5384 |2072 |27.7897% |

These are exact initial seeded-tree counts for this comparison, not the later collapse-refined mesh. The current cube extent56 rounds outward to64 at the level5parent block alignment;40/46/36 round to48;30 rounds to32. This explains why a6unitbuffer removes R30's extra saving. Moving the detector alone without changing refinement saves zero blocks. Runtime savings will be less directly predictable because central refinement, particles, outputs and communication remain. Coarser outer layers could also be redesigned, but are not part of these numbers or an approved change.

Same dx and GW time cadence retain the current planning frequency bandf<=.4 (ten cells per wavelengthatlambda2.5). Shorter propagation may reduce accumulated grid error, but smaller extraction radius increases near-source/gauge/finite-radius bias. These compete; no accuracy percentage can be assigned from this count. R30 is near the envelope edge, requiring explicit areal-versus-isotropic coordinate mapping and evolved matter checks, not the assumption that envelope arealR30 equals coordinate30. Even R50 alone cannot establish radius dependence or an infinity waveform. R40 with a buffer is the more useful cost compromise if the user approves a change.

Scientific reference: https://arxiv.org/abs/1309.3605 (finite-distance Psi4 near-zone contamination) and https://arxiv.org/abs/0910.3656 (finite-radius/gauge errors).

## Vista feasibility and payoff

Migration is feasible in principle: the code has an existing CUDA backend and preserved portable Vista preparation scripts. The current exact source/Kokkos must be rebuilt for ARM/Grace + CUDA/Hopper with VistaMPI/ibrun, and actual startup/memory/output/restart/communication gates must pass. The old Session007 package is reference evidence, not the current source or a proved Vista production benchmark. Current t0 means migration would begin fresh without wasting completed binary evolution; future AMD checkpoints would need a real cross-platform restart comparison.

TACC's current guide specifies one96GB/GiB-class Hopper GPU perGHnode, compared with four64GBMI210 GPUs perAMDnode. Approximately32VistaGHnodes match the aggregate GPU memory of current12AMDnodes/48GPUs;48VistaGHnodes match GPU count. The per-GPU FP64 peak comparison is34/22.6~1.5, but32VistaGPUs have similar aggregate peakFP64 to48MI210s. This is not a measured NRPIC speedup. Faster perGPU hardware, communication and queue differences may reduce walltime; output/checkpoint I/O can limit benefit. A matched short benchmark with required outputs is necessary.

For long jobs, raw node-hour ratio is (Vista_nodes/12)/measured_walltime_speedup. At48nodes and a hypothetical2x speedup, walltime halves but rawnodehours double. Vista's gh queue charges1SU/nodehour (15minute minimum per job); AMD monetary tariff remains unknown, so allocation costs are not financially comparable here. Vista scratch is purgeable after10days; maintain verifiedAnta/data3archive and adapt durable continuation/reporting and site job submission to Vista. Nothing was transferred to or submitted onVista.

Primary machine references checked October6: https://docs.tacc.utexas.edu/hpc/vista/ ; https://www.amd.com/en/products/accelerators/instinct/mi200/mi210.html .
