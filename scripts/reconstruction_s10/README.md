# Reconstruction Session 10: RK3 predictor coupling

`particles/coupling=rk3_predictor` opts into the supplied partitioned scheme:
Euler particle prediction at U_n; SSPRK3 metric RHS at (U_n,Z_n),
(U_A,Z_pred), and (U_B,(Z_n+Z_np1)/2); a strict particle midpoint solve
holds U_B fixed. There is no outer metric iteration. The coupled method targets
second temporal order, not third order. The default remains `legacy`.

The opt-in mode requires fixed-mesh uncharged dust/gr_boris with particle feedback,
RK3, no fluid, and no particle excision. Physical/periodic domain exits are rejected;
the target cluster remains far from the outer boundary. MPI ownership and static
refinement are supported. Stage deposition copies are routed using existing
transport, then committed arrays and rank counts are restored. Physical particles
are committed once after metric stage3. Invalid geometry, more than one-cell motion,
or a midpoint solve failing the scaled1e-11 residual (100iterations) aborts, without
Euler or trilinear fallback. `midpoint_tolerance` and `midpoint_max_iterations` are
explicit experiment controls. A100CUDA and adaptive/excision modes are not qualified.

Stage1 reuses the preceding endpoint source (initialization seeds it). Stage2,
stage3 and final endpoint each deposit once: three deposits/global step, three
metric RHS evaluations. Endpoint deposition seeds the next step and its constraints.
`synchronized_constraints=true` also provides observation-only endpoint constraints
for legacy short controls. Legacy defaults preserve historical diagnostic timing.

Campaign: exact preserved Session7 proper/random nested seed1 samples,N=1/2/4/8M,
Z4c,dx=.5M,CFL=.4,new scheme to500M; short1M old/new timestep controls to100M at
CFL=.4/.2/.1, plus frozen-source RK3/CFL.4 control. Compare archived500M history
prefixes directly with explicit lagged-source diagnostic caveat; no four long old
reruns. Native order uses double-precision checkpoints, not float32 PVTK.
Small native regression fixtures use the supported internal legacy sampler only;
scientific runs use exact archived external proper-profile samples.

CPU and HIP compilation and manufactured second-order solver checks pass in AMD495009.
Native CPU/HIP regression, archived-executable comparison, restart, tolerance, guards, and five MPI seam-crossing fixtures pass in AMD495053. Scientific pilots and production are pending. This is experimental code,
not a claim of validated ST accuracy. Authoritative campaign handoff, job records,
limits and reports live in reconstruction_ST/10_rk3_predictor_20261102.
All scientific computation runs inside Slurm. Max2 simulation nodes+1support;
40allocatednodeh/300GiB review thresholds. Preserve all historical evidence.
