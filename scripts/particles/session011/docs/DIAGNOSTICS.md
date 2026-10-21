# Session 011 diagnostic definitions and validation inventory

Implementation remains under validation. Production has not started. All times are
in inherited Mref units; particle weights are rest masses, not the source masses.
Source hooks are passive: independent scratch arrays, no writes to evolved Z4c or
particle state. Derived ADM constraints may be refreshed for measurement.

| Product | Population and definition | Sampling | Main use |
|---|---|---|---|
| `ah_events_H.csv` | Every eligible search; current acceptance, reason, persistence, center, area, MH, coordinate-rotation spin/J/P, residual, full accepted coefficients | Each step | Acceptance coverage, restarts, geometry, masses/spins |
| `gi_crossings.rankR.csv` | Every particle changing membership of each currently measured spectral surface; exact tag, component, old/new position and covariant momenta, weight, energy, origin Jz | Each step before removal/migration | Accretion, surface sweep, initial enclosure, common reassignment |
| Existing `prtcl_destroy.csv` | Every lapse/escape/removal event; exact identity, rest weight, position and covariant momentum | Each removal | Count/rest/J closure with the surviving population |
| `gi_particles.csv` | Full population, component, radial shell, mask; count, rest mass, En, coordinate energy, native coordinate energy, P/J vectors, first/second position moments, radial covariant momentum | .025 before strict common, .05 after, forced final state | A/B/C; envelope multipoles and coordinate transport |
| `gi_shell_transfer.csv` | Full population shell-entry/exit rest/J, J change during push, union-AH advection and surface sweep; missing-AH mass-step coverage | Accumulate each step; flush at profile cadence/final | Coordinate angular-momentum accounting without summing overlapping horizons |
| `gi_grid.csv` | Proper-volume shell integrals of H², M², lapse, radial spatial metric, radial shift, deposited Eulerian energy | Profile cadence/final | C/E; exterior profiles and constraint norms |
| `gi_status.csv` | dt, blocks/max per rank/capacity, current/ever horizon validity, live tracker positions/separation, exterior proper line length | Profile cadence/final | Orbit intervals, resource/AMR record, gauge-sensitive distance |
| `.trk2` | Fixed 30k IDs per component; no replenishment; int64 tag and float64 mass, -ut, xyz, covariant ui | .25; .05 near collapse/close binary; forced milestones/final | Movies and display snapshots only |
| Native waves | Saved rPsi4, ell2..8, coordinate R40 primary, R36/R44 companions | .025 | Wave/strain assessment with finite-radius limitations |
| Native slices | All existing compact Session 010 slice outputs | .5 | Wakes, non-spherical structure, metric/constraint context |
| Native checkpoints | Full Z4c and all surviving particles; exact continuation metadata | Initialization, individual pair, inward d<1, strict common, final, recovery | Reproduction and non-evolving searches |

Shells are coordinate-origin radii: 64 bins of width .125 on [0,8), then 64
unit-width bins [8,72), then an explicit overflow bin. Particle masks: 0 all,
1 known exterior of the union of accepted horizons, 2/3 individual interiors,
4 common interior. Individual/common masks overlap and MUST NOT be summed.
Grid masks: 0 all, 1 known AH exterior, 2 the inherited chi cutoff for comparison.
Rows of zeros may be omitted for empty particle bins, but mask-1 coverage failure
is explicitly NaN and current/ever-valid bits are recorded independently.

An AH is usable at a sample only when accepted at that step. The sticky last
accepted consumer snapshot is not treated as a current measurement. A previously
accepted horizon missing from the current search invalidates the union exterior.
On restart, established horizons must requalify; first capture following a gap is
labelled requalification, not new physical accretion. Before a horizon has ever
been accepted, the exterior is relative to the set of known accepted surfaces,
not evidence that no unobserved trapped surface exists.

Containment evaluates the finder's own spherical harmonics. A conservative
coefficient-norm outer bound rejects distant points; collocation extrema are not
assumed to bound the continuous spectral surface. At each step, membership is
tested at (old position, old surface), (new position, old surface), and (new
position, new surface). Their differences separate advection and surface motion.
They are endpoint classifications, not substep-resolved root crossing times.
The union ledger avoids double counting individual/common overlaps; individual
event rows retain their horizon identity for separate analyses.

Jz about fixed origin O is sum m[(x-Ox)uy-(y-Oy)ux]. The shell `Jz_force_delta`
field currently means the entire change during the discrete particle push,
including coordinate-position changes; it is a coordinate effective torque
diagnostic, not a covariant force or separately measured gravitational torque.
Fluxes use the pre-push J, with the remaining change assigned to the destination
shell. AH transfer uses the same pre-push J; changes in momentum and surface
membership are reported separately. No exact matter-only angular-momentum
conservation is asserted in the evolving spacetime. Common first enclosure and
requalification are not ordinary flux measurements.

En=mW and coordinate E=-m ut=m(alpha W-beta^i ui) use trilinear interpolation
of the spatial metric and gauge for diagnostics. This is explicitly different
from the high-order native pusher/energy interpolation. Both diagnostic and
native coordinate energy are saved in profiles. Invalid metric mass is counted
separately and must not be interpreted as zero physical energy. H is squared in
the reduction; native M² is already squared. Proper volume is sqrt(det gamma)
dx dy dz, rejecting nonpositive/nonfinite determinant rather than taking abs.

The proper separation is a 2048-midpoint quadrature of ds along the straight
coordinate segment joining the two live trackers, excluding accepted AH
interiors. This quantity depends on both slice and chosen path. It is NaN when
coverage/ownership fails or strict common enclosure has ended binary tracking.
Orbit counts must be segmented at tracker/AH gaps, not unwrapped through them.

The fixed cohort is the modular permutation
`((tag-component_start)*104729+4001) % component_size < 30000`.
104729 is coprime to 1M and 3M, giving exactly 30k initial tags per component.
The binary record is 512-byte text header followed by 72-byte records:
int64 tag; float64 mass, minus_ut, x,y,z,u1_cov,u2_cov,u3_cov. Native endianness
is declared in the header and the AMD/Perseus hosts are little endian.

Expected compact output is several to tens of GiB, with crossings and selected
slices dominating uncertainty. A 4936-block full checkpoint is about 59 GiB;
five protected science states plus two recoveries and an in-progress write
account for roughly 470 GiB before diagnostics. Actual bytes and I/O times,
not these estimates, govern the whole-user storage watchdog and continuation.

Validation must include both GPU backends, analytic AH persistence/common reuse,
parameter serialization and native restart readback, invalid seeds, nonmutation,
analytic metric/containment math, exact-ID cohort readback, full-population count/
rest closure, zero transport in frozen fields, and controller refusal of a second
fresh trajectory or submission at/beyond75. Production is gated until these pass.
