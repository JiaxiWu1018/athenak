## Proposed new Plummer session: isotropic-pressure equilibrium

**The primary goal is to establish a reliable isotropic relativistic cluster benchmark for the NRPIC method paper.** A compactness scan to investigate stability is a secondary goal—not a requirement to complete a separate study of relativistic cluster stability.

### 1. Motivation: move beyond repeating the circular-cluster debugging

The homogeneous circular-orbit campaign already tested frozen versus live spacetime, particle sampling, deposition, interpolation, filtering, and small injected radial dispersion. The later seed-scaling investigation found that **sampling primarily sets the initial perturbation amplitude, while the growing internal dipole’s rate was unchanged within realization scatter**. This is substantial evidence to reuse, rather than reproduce in another broad debugging campaign.[^homogeneous]

Plummer Session 2 likewise showed disruption even though its initial circular orbits passed the individual-orbit stability criterion. However, its disruption was broadband rather than the homogeneous campaign’s specifically dipolar mode, so their mechanisms should not be assumed identical. **Neither campaign conclusively distinguishes a physical collective instability from an instability of the discretized coupled evolution.**[^session2]

### 2. Use the existing Part I construction

Use the **metric-first isotropic Einstein–Vlasov model already derived in the PDF**:
\[
\boxed{
\text{prescribed metric}
\;\longrightarrow\;
\epsilon,\ p_r=p_\theta=p_\phi
\;\longrightarrow\;
\text{compatible }F(e).
}
\]

There is no need, for this session, to develop the alternative simple-power-law, distribution-first model. The existing construction already provides a mutually consistent metric, stress tensor, and particle distribution—the ingredients needed for an equilibrium benchmark.[^construction]

Choose configurations within its particle-admissibility range,
\[
\boxed{\eta=\frac{GM}{ac^2}\le0.619472.}
\]
This bound ensures that the reconstructed distribution is nonnegative over accessible energies; **it is not a stability threshold**.[^admissibility]

### 3. Implement and validate the isotropic initialization

This requires a **new isotropic sampler**, not merely randomizing the directions of the existing circular velocities.

For equal-rest-mass particles, sample positions using particle number and proper spatial volume:
\[
dN=n_s(r)\,dV_{\rm proper}.
\]
At each radius, sample the local momentum magnitude from
\[
\mathcal P(q\mid r)\propto
q^2F\!\left(A(r)\sqrt{1+q^2}\right),
\]
then choose its direction uniformly over the sphere. Here \(q=|\mathbf p_{\rm local}|/(m_\star c)\); the physical speed is \(v=cq/\sqrt{1+q^2}\). **Sampling only the desired velocity dispersion would not specify the full equilibrium distribution.**[^sampling]

The initialization checks should establish that the sampled particle moments reproduce the intended density and isotropic stresses, with the correct rest-mass normalization. **Finite-domain treatment needs explicit attention:** the circular model’s hard cutoff cannot simply be transferred without checking its effect on an isotropic population whose orbits cross radii.[^cutoff]

### 4. Proposed simulation sequence

**First: a matched frozen/live validation.** Start from the same isotropic particle realization and initial metric. Evolve one copy in the fixed background and another with self-consistent spacetime evolution. This validates the **new metric implementation and sampler**; it is not a repetition of the entire circular-cluster investigation.

**Second: establish the method-paper benchmark.** Determine whether the live isotropic model remains close to equilibrium over the selected duration. Track the quantities already emphasized in our discussion: radial profiles and enclosed-mass radii, velocity stresses, angular-mode amplitudes, conservation, constraints, and numerical health. Any observed growth should be assessed using growth rates and numerical controls, not merely the time at which disruption becomes visible.

**Third, optionally: a modest compactness scan.** After validating the implementation, investigate whether the admissible isotropic family exhibits a stability transition. The objective is to **test for and numerically bracket a transition**, not assume that one—and only one—must exist.

### 5. What the results would establish

A successful isotropic run would provide a useful equilibrium benchmark and show that the difficulties encountered with circular clusters do not automatically affect every collisionless equilibrium.

It would **not**, by itself, prove that the circular-cluster instability is physical or that the code is bug-free. Parts I and II differ in their density profiles and metrics as well as velocity anisotropy, so their comparison does not isolate the velocity distribution alone. Likewise, no detected growth over a finite runtime supports a bounded-duration stability statement, not unconditional stability.[^interpretation]

**Still to decide before launching:** the compactness cases, particle number, mesh, timestep, runtime, cutoff treatment, and quantitative acceptance criteria. The working priority is **validate the existing isotropic model first; investigate its stability boundary only as far as useful for the method paper.**

[^homogeneous]: `session_index(3).md`, entries “2026-07-28 — live monopole and multipole diagnostic” and “2026-09-01 — particle-sampling (N) scaling of the internal ℓ=1 mode, R/M=6.5,” particularly lines 74–85 and 797–819.

[^session2]: `REPORT_AGENT_Plummer_session2.md`, Section 3, “The continuum models” (pre-production checks); `REPORT_Plummer_session2.md`, Section 2, “(R/M)_eff = 10: growing, broadband, survives.”

[^construction]: `Relativistic_Plummer_Step_by_Step.pdf`, Section 19, “What was assumed, what was proved, and what remains,” page 20.

[^admissibility]: `Relativistic_Plummer_Step_by_Step.pdf`, Section 10, “Nonnegative particles impose a compactness bound,” page 11.

[^sampling]: `Relativistic_Plummer_Step_by_Step.pdf`, Appendix B, “From a continuum equilibrium to particle sampling,” page 22, especially subsections 1–2.

[^cutoff]: `Relativistic_Plummer_Step_by_Step.pdf`, Appendix B, page 22; `Newtonian_Plummer_Derivation.pdf`, Section 9, “Practical appendix: sampling and finite-radius truncation,” page 10, final boxed note.

[^interpretation]: `Relativistic_Plummer_Step_by_Step.pdf`, Section 19, page 20, especially subsections 1–2.
