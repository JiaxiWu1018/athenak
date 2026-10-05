# Saved Weyl quantity and TT strain convention

Compiled source6892be3e, src/z4c/z4c_calculate_weyl_scalars.cpp.

Near +x, the implemented orthonormal tetrad has theta approximately -z and phi approximately +y. Define hplus=(h_theta_theta-h_phi_phi)/2 and hcross=h_theta_phi. An outgoing TT plane has hzz=hplus, hyy=-hplus and hyz=-hcross. In the weak-field contraction, spatial Ricci, mixed extrinsic-curvature derivative and spatial Riemann terms contribute respectively one quarter, one half and one quarter of d²hplus/dt² to the real scalar; the same terms give d²hyz/dt² to its imaginary component. Thus Psi4=d²(hplus-i*hcross)/dt² in this implemented tetrad convention. The final source multiplies both scalar fields by coordinate radius; spherical extraction preserves that r*Psi4 convention.

The numerical eight-cycle calibration tests plus-polarization sign/normalization and small imaginary leakage. It does not independently test cross-polarized radiation. Original 80M propagation inputs disabled extraction, so zero Weyl arrays are uncalculated fields, not evidence for a zero signal. Strain must stay deferred until the enabled-extraction receipt passes and production coverage/gaps permit integration.

Fixed-frequency integration uses H=r(hplus-i*hcross), Hddot=rPsi4; -FFT(rPsi4)/(2π max(|f|,f0))², with detrending/taper and documented cutoff sensitivity. Single finite radius and approximate retarded time do not provide waveform extrapolation to infinity.
