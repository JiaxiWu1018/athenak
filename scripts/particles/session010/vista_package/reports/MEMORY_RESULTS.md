# Matched memory results

Same native baseline checkpoint; 792 fixed MeshBlocks, five million particles, two MPI ranks, cap 400/rank on two allocated A100 80 GiB GPUs. Each arm advances 128 steps (dt=0.00625); fit advances 32–126, excluding startup/final output. Sampling every two seconds. Fits use pairwise median slopes and endpoints, rather than a sawtooth-sensitive ordinary least-squares fit.

| Arm / device | Median growth MiB/min | Endpoint MiB/min | Fit minutes | Begin / end / peak MiB |
|---|---:|---:|---:|---:|
| memory_baseline / gpu/0 | 0.000 | 0.000 | 20.17 | 64959 / 64959 / 64959 |
| memory_baseline / gpu/1 | 0.000 | 0.000 | 20.17 | 64951 / 64951 / 64951 |
| memory_baseline / host/total | 0.015 | 0.038 | 20.17 | 1366 / 1367 / 1367 |
| memory_patched / gpu/0 | 0.000 | 0.000 | 19.69 | 65025 / 65025 / 65025 |
| memory_patched / gpu/1 | 0.000 | 0.000 | 19.69 | 64969 / 64969 / 64969 |
| memory_patched / host/total | 0.012 | 0.032 | 19.69 | 1421 / 1422 / 1422 |

## Observed outcome

Both baseline and patched GPU traces are flat at NVML MiB granularity over the matched ~20-minute fit windows. This CUDA test did not reproduce Session009’s HIP/MI210 GPU growth, so it cannot quantify a GPU-growth reduction or prove that all causes of that growth are removed. The patched host RSS retains a small positive drift: 0.0123 MiB/min median, 0.0321 MiB/min endpoint (~0.63 MiB over the window). The baseline host drift is similarly small; a percentage reduction of such a small drift is not strong causal evidence. The repair retains about 66/18 MiB more steady GPU allocation and ~55 MiB more host RSS here, consistent with high-water capacity. It is an implemented and numerically validated repair with unproved full-size/site memory feasibility.

Device UUIDs and every sample are in the CSVs. A positive slope is residual growth, not a bounded-memory proof; a flat or negative short-window slope does not prove zero longer-term growth. High-water buffers can retain more steady-state memory while reducing allocation churn. This CUDA/UCX fixed-mesh result does not measure MI210/HIP or Grace/H200 behavior, nor future AMR/collapse allocations. Historical Singularity growth reductions were 54% GPU and 88.4% host; do not substitute them for these measurements.

Both final native checkpoints retain all five million unique tags, exact GIDs/rest weights and the same mesh; tag-sorted real-state comparison uses rtol=1e-10/atol=1e-12. The exact outcome is in matched_native_particle_comparison.json.

## Safe segmentation

No numerical segment length is certified for full-size Vista by this compact test. Measure each GPU restart baseline, host RSS, active/reserved blocks, AMR maxima and residual growth with the final native executable. Select a finite short first segment after discussion, and reserve the measured checkpoint/manifest wall time and allocation headroom. For a conservative positive growth estimate g in GiB/min and remaining headroom H to the agreed memory guard, use T <= min(agreed wall budget minus finalization reserve, 0.5 H/g); also enforce node-hour/segment/calendar/scientific-time ceilings. Use the worst device and add AMR and checkpoint reserves. If g is unresolved or changes during collapse, use a short measured bound rather than an unlimited extrapolation. Stop cleanly before the guard, verify actual native checkpoint progress/hash/accounting and reassess on restart.

The preparation uses an 85% guard only as a bounded test safeguard; Vista must explicitly agree its own limits. The 512-step draft was shortened to 128 matched advances after measuring compact timing to keep each arm within one hour. No full historical validation campaign was repeated.
