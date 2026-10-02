# Jeans-in-cluster Session 008 — companion-supported orbit and GW production on AMD

You are Codex running on Perseus, launched in `/data/jiaxiwu/NRPIC`. Prepare and, after my approval, execute Jeans-in-cluster Session 008 on the established AMD machine. This is a fresh production run for the NRPIC method paper, followed by analysis, plots, movies, and human-readable and technical reports. It is not a Vista handoff or a restart of Session 007.

I am starting in Plan Mode. First inspect the existing instructions, implementation, inputs, and reports, then discuss the simulation setup with me. Do not implement changes, build/run tests, or submit jobs before I approve the plan and we move to execution. Read-only inspection is appropriate during planning.

## 1. Scientific goal and decisions already made

Session 007 used clump orbital boosts calculated from the background envelope alone. I observed a rapid, approximately head-on merger rather than the intended inspiral. In Session 008, include approximate companion support in the INITIAL orbital boost, aiming to obtain a clear post-collapse orbiting/inspiral phase and extract its gravitational radiation. The evolution already includes both clumps' gravity: do not add a new force to the particle pusher.

The boost magnitude is ALREADY APPROVED:

    v_local = 0.133215, with c = 1.

Use this value directly as the local orthonormal Lorentz-boost speed. The left clump receives a -y boost and the right clump a +y boost, giving counterclockwise motion viewed from +z.

Its motivation is the approximate Newtonian companion correction for equal clumps at +/-R:

    v_new^2 ≈ v_old^2 + G*m_companion/(4*R),
    G = 1, v_old ≈ 0.0880124692901454, m_companion = 0.12, R = 3.

This motivates approximately 0.133215; it is NOT an exact GR circular-orbit construction. The local-speed, coordinate-radius, and source-mass conventions make the prescription deliberately approximate. Document this limitation and proceed. Do not recalculate or optimize the approved speed, solve an initial-acceleration root problem, tune ddot(d) to zero, introduce a constraint solver, perform an eccentricity-reduction campaign, or launch a velocity/separation survey.

This method-paper test needs qualitative results, not precision waveform templates or exact binary equilibrium. Moderate eccentricity is acceptable. Clear orbital motion before a possible merger is the goal; neither inspiral nor merger is guaranteed. Do not label every shrinking separation radiation-driven inspiral without examining the orbital evolution and other possible angular-momentum exchanges/losses.

## 2. Read the campaign and recover the correct implementation

Read and obey governing `AGENTS.md`, `CLAUDE.md`, `Logistic.md`/other logistics files, relevant ancestor instructions, and current Perseus/AMD operating rules. Follow any applicable instruction to inventory/read additional documentation. Read the Jeans campaign log, relevant Sessions 004–007 records, actual production inputs, and corrected analysis scripts. Preserve historical records.

Relevant reference documents include:

- `Jeans_Session7_Part1_Codex_Prompt_v2(1).md` or its campaign copy;
- the Session 007 `REPORT_AGENT.md`;
- `REPORT_Jeans7.md`, especially its October 1 execution/plotting follow-up, not just the original Part 1 preparation status.

The known Session 007 preparation directory is:

    /data/jiaxiwu/NRPIC/GI_in_cluster/session_007_part1_vista_handoff_20260908

Locate the real source checkout using Git. Work safely on `project/GI-in-cluster`, using an independent checkout/worktree if needed; do not disrupt other active sessions or unrelated modifications. Inspect later fixes instead of blindly using the old preparation commit `36b64e23867358e52f92ba96f9bcb4eb5e6839e6`. In particular, verify inclusion or an equivalent implementation of the production restart-header repair documented at `263dcf21a7f6ce6ab1ccd8f1dd595ae0ca79c8c1`.

Use the established AMD build/MPI/launcher configuration, not Vista settings. This prompt supersedes Session 007's envelope-only boost and Vista-only scope/budgets; retain compatible validated physics and diagnostic fixes. Do not inherit the old Vista 90-minute agent restriction or Vista node-hour estimates as AMD requirements.

After approval, create a clearly named Session 008 directory under `/data/jiaxiwu/NRPIC/GI_in_cluster/`. Keep the prompt, approved plan, decisions, canonical inputs, scripts, evidence, analysis, and reports there. Commit and push relevant source/scripts/documentation under the existing repository rules; never commit large raw data or credentials.

## 3. Preserve the physical model; start from t = 0

Unless a genuine blocker is discussed and I approve a change, retain:

| Component | Source/model mass parameter | Center | Particle count | Local boost |
|---|---:|---|---:|---|
| Envelope | 0.76 | (0,0,0) | 3,000,000 | Unchanged; no added recoil |
| Left clump | 0.12 | (-3,0,0) | 1,000,000 | (0,-0.133215,0) |
| Right clump | 0.12 | (+3,0,0) | 1,000,000 | (0,+0.133215,0) |

Keep the independently solved spherical Einstein–Vlasov envelope, envelope areal radius 30, isotropic Cartesian clump positions, Gaussian width `sigma=0.70`, and internal orthonormal velocity spread `s=0.02`. Preserve the established sampling and deterministic seeds, including seed 4001 where applicable, immutable component tags, and exactly 5,000,000 initial particles. Neither clump receives imposed internal spin or a new radial boost.

Apply opposite local Lorentz boosts to the thermal samples using the existing implementation, not naive addition to stored covariant momenta. Regenerate the two-clump initialization and all geometry/weights affected by the changed boost. Reuse unaffected envelope assets where valid, but do not reuse incompatible combined geometry or boosted particle data. Never modify velocities in a Session 007 checkpoint and call that Session 008 initial data.

Make the effective boost explicit in the canonical input and verify that the compiled two-clump problem generator actually reads it. Check counts/tags, finite positive weights, centers, momentum conventions, boost signs, positive orbital J_z, and approximately cancelling total linear momentum. Quantify sampling residuals; do not impose a new symmetry or envelope recoil to force cancellation.

Distinguish source/model normalization, sampled rest mass, measured horizon masses, and any ADM estimate. Do not assume `0.76+0.12+0.12=1` makes all of these equal to one. Preserve inherited units and define any plotted `t/M` normalization explicitly.

## 4. Discuss the remaining setup before execution

After inspection, present one recommended configuration with its rationale, a compact parameter table, and a bounded execution plan. Clearly separate already approved choices from choices needing my approval. Do not ask me again whether to use 0.133215.

Discuss together:

1. AMD allocation/GPU layout and memory feasibility for the full configuration, including planned wave-zone refinement.
2. Domain, central and wave-zone resolutions, extraction radii, and output cadences.
3. An approximate orbital timescale, finite physical endpoint, post-merger wave-propagation/ringdown tail, checkpoint segments, and total compute/storage caps.
4. A bounded startup/collapse/orbital check and how it transitions into production without duplicating an expensive run.

Give recommendations rather than an unbounded menu or a long questionnaire. Existing machine configuration and logistics should be discovered from the records/site, not asked of me unnecessarily. Identify any genuine unresolved budget or scientific tradeoff, then wait for approval.

## 5. Numerical baseline, duration, and resource feasibility

Use the validated Session 007 evolution baseline: RK4, CFL 0.4, established gauge/damping, conservative deposition/feedback and pusher protections, particle removal at `alpha<0.05`, and AH-driven particle removal OFF. Keep approximate `K_ij=0` initial data and acknowledge the unsolved local momentum constraint; opposite global boosts do not solve it locally.

Retain initial compact refinement around both clumps and Löhner refinement of `alpha*psi^7` at threshold 0.2, with `tracker_floor=false`. Start the assessment from central `dx_min=1/256`, root `dx=2`, `32^3` cells per MeshBlock, and domain `[-256,256]^3`. These domain/wave-zone settings are a starting point, not automatic approval for a much longer run. Do not silently restore `1/512`, add a moving tracker refinement floor, or coarsen the physics to fit memory. Any explicit wave-zone refinement is a separately documented choice.

Assess boundary contamination for the proposed duration using the actual boundary conditions and relevant coordinate/gauge propagation speeds. If needed, propose an outer buffer or revised hierarchy while preserving the intended central/wave resolution. Verify the complete mesh and physical spacings; do not copy logical refinement-level numbers across changed root layouts.

Do not inherit Session 007's `t=64` limit. Estimate the new orbital timescale using the actual local-to-coordinate convention, without changing the approved boost. Aim for at least one clear post-BH-formation revolution, preferably more if feasible. Select a finite target and hard resource cap; an orbiting binary at the cap is an honest possible outcome.

Do not stop merely because particles were removed, mass plateaued, or a common horizon first appeared. For a complete merger waveform, continue long enough for the signal to reach the outermost extraction sphere plus an adequate ringdown tail. Any common-horizon-plus-tail stopping rule must be explicitly approved and bounded by the hard endpoint. No open-ended continuation until merger.

The full five-million-particle configuration may be demanding on AMD. Use the actual memory inventory and bounded measurements, not scaled Vista estimates. If it does not fit or is impractical, explain the bottleneck and propose a concrete alternative for approval. Do not silently reduce particle count, resolution, outputs, or change the execution site.

## 6. Gravitational-wave extraction and resolution

Session 007 already enabled `r*Psi4` multipoles for ell=2..8 at radii 40, 50, 60, and 70 in the inherited code units. Locate and reuse the actual implementation and output contract; do not assume a new extraction module is needed.

Start with those radii/modes. Discuss a farther radius only if useful and affordable. Check sphere placement, matter contamination, radius/tetrad/sign conventions, and whether the saved quantity is Psi4 or r*Psi4. Preserve complex raw multipoles throughout the run with restart-safe timestamps and no overwritten/duplicated segments.

Design spatial resolution around the shortest gravitational wavelength we intend to interpret, not merely the initial orbital wavelength. Inspect the ENTIRE propagation region: fine black-hole grids or a thin fine extraction shell do not establish adequate wave propagation. Propose the least expensive defensible wave-zone mesh and state its resolved frequency range and limitations. A cells-per-wavelength estimate is a planning heuristic, not convergence evidence. Do not turn this into a precision-convergence campaign, but do not claim a resolved signal from obviously inadequate sampling.

Set the GW time cadence independently of particle/movie/volume-dump cadence. Retain sufficient samples for the highest intended frequency, without writing huge 3-D dumps at every waveform sample. Budget output and checkpoint storage before production. Discuss duration, boundary placement, wave-zone cost, and extraction together.

Afterward, prioritize the (2,+/-2) modes and selected useful additional modes. Compare radii at documented approximate retarded times; do not hide discrepancies through arbitrary rescaling or alignment. Produce strain with a documented method such as fixed-frequency integration after verifying conventions, and show reasonable integration-cutoff sensitivity where the data allow. Preserve raw Psi4 as the primary evidence. Separate early initialization transients, collapse, orbital evolution, and merger/ringdown only where supported. Radiated energy/angular momentum are optional diagnostics when normalization, time coverage, and resolution support them, not mandatory precision claims.

## 7. Bounded checks, tracking, and production execution

After approval, perform targeted checks on the actual final configuration: input/mesh inventory, full-particle initialization ledger, a brief AMD GPU startup, finite evolved state and required outputs, and a short uninterrupted-versus-restart comparison. Verify restart-header handling with a real relevant checkpoint and confirm that diagnostic reacquisition gaps are flagged rather than treated as physical zeros. Reuse trustworthy existing tests rather than rebuilding the entire Session 007 validation campaign.

Do a lightweight initial constraint comparison with a suitable Session 007 baseline where feasible. Match times, regions, masks, and normalization before quoting ratios; global empty-volume changes must not masquerade as better constraints. Acknowledge the approximate initial data without demanding a new solver or an expensive unboosted control evolution.

Maintain separate left/right tracker identities even after they cross x=0, survive substantial particle removal, and restart. Keep strict individual and common-horizon acceptance, object association checks, and recorded search failures. A merger claim requires a usable accepted common horizon enclosing both objects, not merely small separation. Do not use rejected/stale surfaces as measurements.

Use a bounded early continuation to check collapse and subsequent orbital behavior. It can be the first checkpointed part of the eventual production run; do not rerun an identical expensive pilot unnecessarily. This is validation, not velocity tuning. If the corrected system still plunges rapidly, document the outcome and inspect diagnostics; do not automatically change speed/separation or launch a survey.

Run heavy computation on authorized AMD compute resources; obey Perseus rules for any local tests/analysis. Use checkpointed, finite job segments and established site-supported continuation, with duplicate-launch protection and sticky user-stop intent. Test output paths from the actual batch working directory. No dependence on a persistent Codex session or an interactive shell to continue or stop production.

Record physical-time, wall-time, segment, compute, and storage caps. Numerical failures stop for review, not endless retries. Preserve verified checkpoints, raw science outputs, and failure evidence; no unapproved deletions or large transfers to Vista/Anta. Provide exact status, graceful-stop, and emergency-cancel commands scoped to this campaign.

## 8. Scientific analysis, plots, and movies

Prepare reproducible post-processing and, once data are available, analyze the complete available run with restart segments assembled correctly. Use approved compute resources for heavy analysis. Where possible compare Session 008 with the retained Session 007 data over their shared time range; do not imply Session 007 contains a merger waveform at distant radii simply because it contains a central merger.

Track separation, absolute and envelope-relative trajectories, unwrapped orbital phase, orbital frequency, and accumulated revolutions after BOTH individual horizons form. Inspect radial versus tangential motion, including `abs(d_dot)/(d*abs(Omega))` where meaningful; handle small denominators and diagnostic gaps honestly. Distinguish eccentric close passages from sustained orbital motion and secular shrinkage. These are coordinate diagnostics, not gauge-invariant orbital elements.

Include accepted individual/common horizon masses, spins and quality flags; component particle/removal histories; matter angular-momentum diagnostics; constraint and numerical-health histories; and waveform amplitude, phase/frequency and multi-radius comparisons. Preserve the established covariant-particle angular momentum convention and horizon-diagnostic limitations. Do not add removed-particle J to horizon J as an exact conserved total or replace BH momentum with mass times coordinate velocity.

Produce useful plots and at least a clear central-orbit movie plus an envelope/context view. Reuse the corrected Session 007 rendering pipeline where valid: tagged particle views, trajectories, accepted horizon overlays when available, and fixed-grid particle-deposited density views rather than misleading native-AMR seams. Make axes, units, time, component identity, frame cadence, and restart/diagnostic gaps clear. Rendering may use documented subsampling, but this must never reduce the simulated particle count.

Retain the particle, metric and diagnostic information needed for these products without enormous unnecessary full-domain fine dumps. Check representative rendered frames and plot values, not just script exit codes. State the few most informative plots/movies in the human report with absolute paths and captions explaining what to look for.

## 9. Reports, provenance, and honest closeout

Write:

- `REPORT_Jeans8.md`: a human-readable setup/results report explaining the approved approximate boost, differences from Session 007, collapse/orbital/merger outcomes, waveform findings, important limitations, and recommended plots/movies.
- `REPORT_AGENT.md`: implementation, tests, exact source/submodule revisions, input/executable hashes, AMD build/runtime details, job/segment/checkpoint history, measured resources/storage, reproducible analysis commands, and outstanding issues.
- A short README and manifest mapping canonical inputs, raw outputs, verified checkpoints, scripts, figures, movies, and status/stop commands; update the campaign log without rewriting prior sessions.

Freeze and record the tested source/configuration, commit and push relevant changes, and report the actual push status. Keep scientific changes separate from operations/diagnostic changes. Do not overwrite Session 007 products.

If the production outlives the interactive agent session, leave a bounded durable workflow and runnable post-processing, with a precise handoff/status record. Clearly distinguish planned, tested, submitted, queued, running, and completed work. Do not fabricate final plots or findings before the data exist, or call a partially completed run a completed inspiral–merger–ringdown calculation.

Your FIRST response should follow read-only inspection and discuss the recommended setup and remaining approval decisions. Do not execute yet.
