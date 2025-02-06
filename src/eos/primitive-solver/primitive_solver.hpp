#ifndef EOS_PRIMITIVE_SOLVER_PRIMITIVE_SOLVER_HPP_
#define EOS_PRIMITIVE_SOLVER_PRIMITIVE_SOLVER_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file primitive_solver.hpp
//  \brief Declares PrimitiveSolver class.
//
//  PrimitiveSolver contains all the infrastructure for the inversion
//  procedure from conserved to primitive variables in GRMHD. This
//  particular implementation is based on the solver described in
//  Kastaun et al., Phys. Rev. D 103, 023018 (2021).

#include <math.h>
// FIXME: Debug only!
#include <iostream>
#include <algorithm>

#include "numtools_root.hpp"

#include "eos.hpp"
#include "geom_math.hpp"
#include "ps_error.hpp"
#include "ps_types.hpp"

namespace Primitive {

template<typename EOSPolicy, typename ErrorPolicy>
class PrimitiveSolver {
 private:
  // Inner classes defining functors
  // UpperRootFunctor {{{
  class UpperRootFunctor {
   public:
    //! \brief function for the upper bound of the root
    //
    //  The upper bound is the solution to the function
    //  \f$\mu\sqrt{h_0^2 + \bar{r}^2(\mu)} - 1 = 0\f$
    //
    //  \param[out] f    The value of the root function at mu
    //  \param[out] df   The derivative of the root function
    //  \param[in   mu   The guess for the root
    //  \param[in]  bsq  The square magnitude of the magnetic field
    //  \param[in]  rsq  The square magnitude of the specific momentum S/D
    //  \param[in]  rbsq The square of the product \f$r\cdot b\f$
    //  \param[in]  h_min The minimum enthalpy
    KOKKOS_INLINE_FUNCTION
    void operator()(Real &f, Real &df,
                    Real mu, Real bsq, Real rsq, Real rbsq, Real min_h) const {
      const Real x = 1.0/(1.0 + mu*bsq);
      const Real xsq = x*x;
      const Real rbarsq = rsq*xsq + mu*x*(1.0 + x)*rbsq;
      const Real dis = sqrt(min_h*min_h + rbarsq);
      const Real dx = -bsq*xsq;
      //const Real drbarsq = rbsq*x*(1.0 + x) + (mu*rbsq + 2.0*(mu*rbsq + rsq)*x)*dx;
      const Real drbarsq = rbsq*xsq + mu*rbsq*dx + x*(rbsq + 2.0*(mu*rbsq + rsq)*dx);
      f = mu*dis - 1.0;
      df = dis + mu*drbarsq/(2.0*dis);
    }
  };
  // }}}

  // MuFromWFunctor {{{
  class MuFromWFunctor {
   public:
    KOKKOS_INLINE_FUNCTION
    void operator()(Real &f, Real &df,
                    Real mu, Real bsq, Real rsq, Real rbsq, Real W) const {
      const Real musq = mu*mu;
      const Real x = 1.0/(1.0 + mu*bsq);
      const Real xsq = x*x;
      const Real rbarsq = rsq*xsq + mu*x*(1.0 + x)*rbsq;
      const Real vsq = musq*rbarsq;
      const Real dx = -bsq*xsq;
      //const Real drbarsq = rbsq*x*(1.0 + x) + (mu*rbsq + 2.0*(mu*rbsq + rsq)*x)*dx;
      const Real drbarsq = rbsq*xsq + mu*rbsq*dx + x*(rbsq + 2.0*(mu*rbsq + rsq)*dx);
      //const Real drbarsq = 2.0*rsq*dx + x*(1.0 + x)*rbsq + mu*dx*(1.0 + 2.0*x)*rbsq;
      const Real dvsq = 2.0*mu*rbarsq + musq*drbarsq;
      f = vsq + 1.0/(W*W) - 1.0;
      df = dvsq;
    }
  };
  // }}}

  // RootFunctor {{{
  class RootFunctor {
   public:
    KOKKOS_INLINE_FUNCTION
    Real operator()(Real mu, Real D, Real q, Real bsq, Real rsq, Real rbsq, Real *Y,
        const EOS<EOSPolicy, ErrorPolicy> * peos, Real* n, Real* T, Real* P) const {
      // We need to get some utility quantities first.
      const Real x = 1.0/(1.0 + mu*bsq);
      const Real xsq = x*x;
      const Real musq = mu*mu;
      //const Real den = 1.0 + mu*bsq;
      //const Real mux = mu*x;
      //const Real muxsq = mux/den;
      //const Real rbarsq = rsq*xsq + mu*x*(1.0 + x)*rbsq;
      //const Real rbarsq = xsq*(rsq + mu*(2.0 + mu*bsq)*rbsq);
      // An alternative calculation of rbarsq that may be more accurate.
      //const Real rbarsq = rsq*xsq + (mux + muxsq)*rbsq;
      const Real rbarsq = x*(rsq*x + mu*(x + 1.0)*rbsq);
      //const Real qbar = q - 0.5*bsq - 0.5*musq*xsq*(bsq*rsq - rbsq);
      const Real qbar = q - 0.5*bsq - 0.5*musq*xsq*fma(bsq, rsq, -rbsq);
      const Real mb = peos->GetBaryonMass();

      // Now we can estimate the velocity.
      //const Real v_max = peos->GetMaxVelocity();
      const Real h_min = peos->GetMinimumEnthalpy();
      const Real vsq_max = fmin(rsq/(h_min*h_min + rsq),
                                    peos->GetMaxVelocity()*peos->GetMaxVelocity());
      const Real vhatsq = fmin(musq*rbarsq, vsq_max);

      // Using the velocity estimate, predict the Lorentz factor.
      // NOTE: for extreme velocities, this alternative form of W may be more accurate:
      // Wsq = 1/(eps*(2 - eps)) = 1/(eps*(1 + v)), where eps = 1 - v.
      //const Real What = 1.0/std::sqrt(1.0 - vhatsq);
      const Real iWhat = sqrt(1.0 - vhatsq);

      // Now estimate the number density.
      Real rhohat = D*iWhat;
      Real nhat = rhohat/mb;
      peos->ApplyDensityLimits(nhat);

      // Estimate the energy density.
      Real eoverD = qbar - mu*rbarsq + 1.0;
      Real ehat = D*eoverD;
      // peos->ApplyEnergyLimits(ehat, nhat, Y);
      //eoverD = ehat/D;

      // Now we can get an estimate of the temperature, and from that, the pressure and
      // enthalpy.
      Real That = peos->GetTemperatureFromE(nhat, ehat, Y);
      peos->ApplyTemperatureLimits(That);
      //ehat = peos->GetEnergy(nhat, That, Y);
      Real Phat = peos->GetPressure(nhat, That, Y);
      Real hhat = peos->GetEnthalpy(nhat, That, Y);

      // Now we can get two different estimates for nu = h/W.
      Real nu_a = hhat*iWhat;
      //Real ahat = Phat / ehat;
      Real nu_b = eoverD + Phat/D;
      //Real nu_b = (1.0 + ahat)*eoverD;
      //Real nu_b = (1.0 + ahat)*eoverD;
      Real nuhat = fmax(nu_a, nu_b);

      // Finally, we can get an estimate for muhat.
      Real muhat = 1.0/(nuhat + mu*rbarsq);

      *n = nhat;
      *T = That;
      *P = Phat;

      // FIXME: Debug only!
      /*std::cout << "    D   = " << D << "\n";
      std::cout << "    q   = " << q << "\n";
      std::cout << "    bsq = " << bsq << "\n";
      std::cout << "    rsq = " << rsq << "\n";
      std::cout << "    rbsq = " << rbsq << "\n"*/

      return mu - muhat;
    }
  };
  // }}}
 private:
  /// A constant pointer to the EOS.
  /// We make this constant because the
  /// possibility of changing the EOS
  /// during implementation seems both
  /// unlikely and dangerous.
  //EOS<EOSPolicy, ErrorPolicy> *const peos;
  EOS<EOSPolicy, ErrorPolicy> eos;

  /// The root solver.
  NumTools::Root root;
  UpperRootFunctor UpperRoot;
  MuFromWFunctor MuFromW;
  RootFunctor RootFunction;

  //! \brief Check and handle the corner case for rho being too small or large.
  //
  //  Using the minimum and maximum values of rho along with some physical
  //  limitations on the velocity using S, we can predict if rho is going
  //  to violate constraints set by the EOS on how big or small it can get.
  //  We can also use these constraints to tighten the bounds on mu.
  //
  //  \param[in,out] mul   The lower bound for mu
  //  \param[in,out] muh   The upper bound for mu
  //  \param[in]     D     The relativistic density
  //  \param[in]     bsq   The square magnitude of the magnetic field
  //  \param[in]     rsq   The square magnitude of the specific momentum S/D
  //  \param[in]     rbsq  The square of the product \f$r\cdot b\f$
  //  \param[in]     h_min The minimum enthalpy
  //
  //  \return an Error code, usually RHO_TOO_BIG, RHO_TOO_SMALL, or SUCCESS
  KOKKOS_INLINE_FUNCTION
  Error CheckDensityValid(Real& mul, Real& muh, Real D, Real bsq,
                          Real rsq, Real rbsq, Real h_min) const;

 public:
  Real tol;

  /// Constructor
  //PrimitiveSolver(EOS<EOSPolicy, ErrorPolicy> *eos) : peos(eos) {
  PrimitiveSolver() {
    //root = NumTools::Root();
    tol = 1e-15;
    root.iterations = 30;
  }

  /// Destructor
  ~PrimitiveSolver() = default;

  //! \brief Get the primitive variables from the conserved variables.
  //
  //  \param[out]    prim  The array of primitive variables
  //  \param[in,out] cons  The array of conserved variables
  //  \param[in,out] bu    The magnetic field
  //  \param[in]     g3d   The 3x3 spatial metric
  //  \param[in]     g3u   The 3x3 inverse spatial metric
  //
  //  \return information about the solve
  KOKKOS_INLINE_FUNCTION
  SolverResult ConToPrim(Real prim[NPRIM], Real cons[NCONS], Real b[NMAG],
                         Real g3d[NSPMETRIC], Real g3u[NSPMETRIC], Real metric[NTPO]) const;

  //! \brief Get the conserved variables from the primitive variables.
  //
  //  \param[in]    prim  The array of primitive variables
  //  \param[out]   cons  The array of conserved variables
  //  \param[in]    bu    The magnetic field
  //  \param[in]    g3d   The 3x3 spatial metric
  //
  //  \return an error code
  KOKKOS_INLINE_FUNCTION
  Error PrimToCon(Real prim[NPRIM], Real cons[NCONS], Real b[NMAG],
                 Real g3d[NSPMETRIC]) const;

  /// Get the EOS used by this PrimitiveSolver.
  /*KOKKOS_INLINE_FUNCTION EOS<EOSPolicy, ErrorPolicy> *const GetEOS() const {
    return peos;
  }*/
  KOKKOS_INLINE_FUNCTION EOS<EOSPolicy, ErrorPolicy>& GetEOSMutable() {
    return eos;
  }
  KOKKOS_INLINE_FUNCTION const EOS<EOSPolicy, ErrorPolicy>& GetEOS() const {
    return eos;
  }

  /// Get the root solver used by this PrimitiveSolver.
  KOKKOS_INLINE_FUNCTION NumTools::Root& GetRootSolverMutable() {
    return root;
  }
  KOKKOS_INLINE_FUNCTION const NumTools::Root& GetRootSolver() const {
    return root;
  }

  //! \brief Do failure response and adjust conserved variables if necessary.
  //
  //  Note that in ConToPrim, the error policy dictates whether or not we
  //  should adjust the conserved variables if the primitive variables are
  //  floored. That may appear to be the case here, too, because of the bool
  //  returned by DoFailureResponse. However, DoFailureResponse simply tells
  //  us whether or not the primitives were adjusted in the first place, not
  //  whether or not we should adjust the conserved variables. Thus, if the
  //  primitive variables are modified as part of the error response, the
  //  conserved variables are *always* altered. The reasoning here is that
  //  flooring after a primitive solver indicates a physical state which is
  //  just slightly out of bounds. Depending on how the conserved variables
  //  are to be used afterward, the user may not find it necessary to rescale
  //  them. On the other hand, a failure mode generally indicates that the
  //  state itself is unphysical, so any modification to the primitives
  //  also requires a modification to the conserved variables.
  //
  //  \param[in,out] prim  The array of primitive variables
  //  \param[in,out] cons  The array of conserved variables
  //  \param[in,out] bu    The magnetic field
  //  \param[in]     g3d   The 3x3 spatial metric
  KOKKOS_INLINE_FUNCTION void HandleFailure(Real prim[NPRIM], Real cons[NCONS],
                     Real bu[NMAG], Real g3d[NSPMETRIC]) const {
    bool result = eos.DoFailureResponse(prim);
    if (result) {
      Real z_u[3] = {prim[PVX], prim[PVY], prim[PVZ]};
      Real zsq = SquareVector(z_u, g3d);
      Real W = sqrt(1 + zsq);

      Real Bsq = SquareVector(bu, g3d);
      Real z_d[3] = {0.0};
      LowerVector(z_d, z_u, g3d);
      Real Bdotv = Contract(bu, z_d) / W;

      Real b2_local = Bsq / (1 + zsq) + Bdotv * Bdotv;
      bool floor = eos.ReinforceSigmaLimit(prim[PRH], b2_local);

      PrimToCon(prim, cons, bu, g3d);
    }
  }
};

// CheckDensityValid {{{
template<typename EOSPolicy, typename ErrorPolicy>
KOKKOS_INLINE_FUNCTION
Error PrimitiveSolver<EOSPolicy, ErrorPolicy>::CheckDensityValid(Real& mul, Real& muh,
      Real D, Real bsq, Real rsq, Real rbsq, Real h_min) const {
  // There are a few things considered:
  // 1. If D > rho_max, we need to make sure that W isn't too large.
  //    W_max can be estimated by considering the zero-field limit
  //    of S^2/D^2 if h = h_min.
  //    - If W is larger than W_max, then rho is just too big.
  //    - Otherwise, we can bound mu by using W to do a root solve
  //      for mu.
  // 2. If D < W_max*rho_min, then we need to make sure W isn't less
  //    than 1.
  //    - If W is less than 1, it means rho is actually smaller than
  //      rho_min.
  //    - Otherwise, we can bound mu by using W to do a root solve
  //      for mu.
  Real W_max = sqrt(1.0 + rsq/(h_min*h_min));
  Real rho_max = eos.GetMaximumDensity()*eos.GetBaryonMass();
  Real rho_min = eos.GetMinimumDensity()*eos.GetBaryonMass();
  if (D > rho_max) {
    Real W = D/rho_max;
    Real f, df;
    MuFromW(f, df, muh, bsq, rsq, rbsq, W);
    if (f <= 0) {
      // W is not physical, so rho must be larger than rho_max.
      return Error::RHO_TOO_BIG;
    } else {
      MuFromW(f, df, mul, bsq, rsq, rbsq, W);
      if (f < 0) {
        Real mu;
        Real mulc = mul;
        Real muhc = muh;
        // We can tighten up the bounds for mul.
        // The derivative is zero at mu = 0, so we perturb it slightly.
        /*if (mu <= root.tol) {
          mu += root.tol;
        }*/
        bool result = root.NewtonSafe(MuFromW, mulc, muhc, mu, 1e-10, bsq, rsq, rbsq, W);
        if (!result) {
          return Error::BRACKETING_FAILED;
        }
        mul = (mu > mul) ? mu : mul;
      }
    }
  }
  if (D < W_max*rho_min) {
    Real W = D/rho_min;
    Real f, df;
    MuFromW(f, df, mul, bsq, rsq, rbsq, W);
    if (f >= 0) {
      // W is not physical, so rho must be smaller than rho_min.
      return Error::RHO_TOO_SMALL;
    } else {
      MuFromW(f, df, muh, bsq, rsq, rbsq, W);
      if (f > 0) {
        Real mu = muh;
        Real mulc = mul;
        Real muhc = muh;
        // We can tighten up the bounds for muh.
        bool result = root.NewtonSafe(MuFromW, mulc, muhc, mu, 1e-10, bsq, rsq, rbsq, W);
        if (!result) {
          return Error::BRACKETING_FAILED;
        }
        muh = (mu < muh) ? mu : muh;
      }
    }
  }
  return Error::SUCCESS;
}
// }}}

// ConToPrim {{{
template<typename EOSPolicy, typename ErrorPolicy>
KOKKOS_INLINE_FUNCTION
SolverResult PrimitiveSolver<EOSPolicy, ErrorPolicy>::ConToPrim(Real prim[NPRIM],
      Real cons[NCONS], Real b[NMAG], Real g3d[NSPMETRIC], Real g3u[NSPMETRIC],
      Real metric[NTPO]) const {
  SolverResult solver_result{Error::SUCCESS, 0, false, false, false};
  FloorFlag floor_flag{false, false, false, false};

  // Extract the undensitized conserved variables.
  Real D = cons[CDN];
  Real S_d[3] = {cons[CSX], cons[CSY], cons[CSZ]};
  Real tau = cons[CTA];
  Real B_u[3] = {b[IBX], b[IBY], b[IBZ]};
  // Extract the particle fractions.
  const int n_species = eos.GetNSpecies();
  Real Y[MAX_SPECIES] = {0.0};
  for (int s = 0; s < n_species; s++) {
    Y[s] = cons[CYD + s]/cons[CDN];
  }

  // Check the conserved variables for consistency and do whatever
  // the EOSPolicy wants us to.
  bool floored = eos.ApplyConservedFloor(D, S_d, tau, Y, SquareVector(B_u, g3d), floor_flag);
  solver_result.cons_floor = floored;
  if ((floored && eos.IsConservedFlooringFailure()) || floor_flag.d_floor) {
    HandleFailure(prim, cons, b, g3d);
    // solver_result.error = Error::CONS_FLOOR;
    return solver_result;
  }
  if (floor_flag.tau_floor) {
    cons[CTA] = tau; // Keep variables consistent.
  }

  // Calculate some utility quantities.
  Real sqrtD = sqrt(D);
  Real b_u[3] = {B_u[0]/sqrtD, B_u[1]/sqrtD, B_u[2]/sqrtD};
  Real r_d[3] = {S_d[0]/D, S_d[1]/D, S_d[2]/D};
  Real r_u[3];
  RaiseForm(r_u, r_d, g3u);
  Real rsqr   = Contract(r_u, r_d);
  Real rb     = Contract(b_u, r_d);
  Real rbsqr  = rb*rb;
  Real bsqr   = SquareVector(b_u, g3d);
  Real q      = tau/D;

  // Make sure there are no NaNs at this point.
  if (!isfinite(D) || !isfinite(rsqr) || !isfinite(q) ||
      !isfinite(rbsqr) || !isfinite(bsqr)) {
    HandleFailure(prim, cons, b, g3d);
    solver_result.error = Error::NANS_IN_CONS;
    solver_result.cons_adjusted = true;
    return solver_result;
  }
  // We have to check the particle fractions separately.
  for (int s = 0; s < n_species; s++) {
    if (!isfinite(Y[s])) {
      HandleFailure(prim, cons, b, g3d);
      solver_result.error = Error::NANS_IN_CONS;
      solver_result.cons_adjusted = true;
      return solver_result;
    }
  }


  // Bracket the root.
  Real min_h = eos.GetMinimumEnthalpy();
  Real mul = 0.0;
  Real muh = 1.0/min_h;
  // Check if a tighter upper bound exists.
  if (rsqr > min_h*min_h) {
    Real mu = 0.0;
    // We don't need the bound to be that tight, so we reduce
    // the accuracy of the root solve for speed reasons.
    Real mulc = mul;
    Real mulh = muh;
    bool result = root.NewtonSafe(UpperRoot, mulc, mulh, mu, 1e-10,
                                  bsqr, rsqr, rbsqr, min_h);
    // Scream if the bracketing failed.
    if (!result) {
      HandleFailure(prim, cons, b, g3d);
      solver_result.error = Error::BRACKETING_FAILED;
      solver_result.cons_adjusted = true;
      return solver_result;
    } else {
      // To avoid problems with the case where the root and the upper bound collide,
      // we will perturb the bound slightly upward.
      // TODO(JF): Is there a more rigorous way of treating this?
      muh = mu*(1. + 1e-10);
    }
  }

  // Check the corner case where the density is outside the permitted
  // bounds according to the ErrorPolicy.
  Error error = CheckDensityValid(mul, muh, D, bsqr, rsqr, rbsqr, min_h);
  // TODO(JF): This is probably something that should be handled by the ErrorPolicy.
  if (error != Error::SUCCESS) {
    HandleFailure(prim, cons, b, g3d);
    solver_result.error = error;
    solver_result.cons_adjusted = true;
    return solver_result;
  }


  // Do the root solve.
  Real n, P, T, mu;
  bool result = root.FalsePosition(RootFunction, mul, muh, mu, tol,
                                   D, q, bsqr, rsqr, rbsqr, Y, &eos, &n, &T, &P);
  // WARNING: the reported number of iterations is not thread-safe and should only be
  // trusted on single-thread benchmarks.
  solver_result.iterations = root.iterations;
  if (!result) {
    HandleFailure(prim, cons, b, g3d);
    solver_result.error = Error::NO_SOLUTION;
    solver_result.cons_adjusted = true;
    return solver_result;
  }
  // Lapse excision
  if (metric[LAPSE] < eos.GetLapseExcision()) {
    HandleFailure(prim, cons, b, g3d);
    solver_result.error = Error::LAPSE_TOO_SMALL;
    solver_result.cons_adjusted = true;
    return solver_result;
  }

  // Retrieve the primitive variables.
  Real rho = n*eos.GetBaryonMass();
  Real rbmu = rb*mu;
  Real W = D/rho;
  Real Wmux = W*mu/(1.0 + mu*bsqr);
  // Before we retrieve the velocity, we need to raise S.
  Real S_u[3] = {0.0};
  RaiseForm(S_u, S_d, g3u);
  // Now we can get Wv.
  Real Wv_u[3] = {0.0};
  Wv_u[0] = Wmux*(r_u[0] + rbmu*b_u[0]);
  Wv_u[1] = Wmux*(r_u[1] + rbmu*b_u[1]);
  Wv_u[2] = Wmux*(r_u[2] + rbmu*b_u[2]);
  // Check NaNs in prim
  if (!isfinite(rho) || !isfinite(W)) {
    HandleFailure(prim, cons, b, g3d);
    solver_result.error = Error::NANS_IN_PRIM;
    solver_result.prim_floor = true;
    solver_result.cons_adjusted = true;
    return solver_result;
  }

  // Apply the flooring policy to the primitive variables.
  solver_result.prim_floor = eos.ApplyPrimitiveFloor(n, Wv_u, P, T, Y);
  if (solver_result.prim_floor && eos.IsPrimitiveFlooringFailure()) {
    HandleFailure(prim, cons, b, g3d);
    solver_result.error = Error::PRIM_FLOOR;
    solver_result.cons_adjusted = true;
    return solver_result;
  }

  // Apply drift floor
  bool drift_floor = false;
  if (!solver_result.prim_floor) {
    Real Wvsq = SquareVector(Wv_u, g3d);
    Real W = sqrt(1 + Wvsq);
    Real Wv_d[3] = {0.0};
    LowerVector(Wv_d, Wv_u, g3d);
    Real Bdotv = Contract(B_u, Wv_d) / W;
    Real Bsq = SquareVector(B_u, g3d);
    Real b2_local = Bsq / (1 + Wvsq) + Bdotv * Bdotv;
    Real sigma = b2_local / (n * eos.GetBaryonMass());
    Real Bnorm = sqrt(Bsq);
    Real vpar = Bdotv / Bnorm;
    if (!isfinite(vpar)) {vpar = 0;}

    drift_floor = sigma > eos.GetSigmaLimit();
    if (drift_floor) {
      Real BdotS = Contract(B_u, S_d);
      Real Bconst = BdotS / Bnorm;
      Real rhoh = n * eos.GetBaryonMass() * eos.GetEnthalpy(n, T, Y);
      Real vdrift_u[3] = {0.0};
      vdrift_u[0] = Wv_u[0] / W - vpar * B_u[0] / Bnorm;
      vdrift_u[1] = Wv_u[1] / W - vpar * B_u[1] / Bnorm;
      vdrift_u[2] = Wv_u[2] / W - vpar * B_u[2] / Bnorm;
      Real vdrift_min = 0.0, vdrift_max = 0.9999999;
      Real vdriftsqr = fmax(vdrift_min, fmin(SquareVector(vdrift_u, g3d), vdrift_max));
      Real gamma_drift = sqrt(1. + vdriftsqr / fabs(1. - vdriftsqr));
      // Inject floors
      n = b2_local / (eos.GetSigmaLimit() * eos.GetBaryonMass());
      T = eos.GetTemperatureFloor();
      P = eos.GetPressure(n, T, Y);
      rhoh = n * eos.GetBaryonMass() * eos.GetEnthalpy(n, T, Y);
      Real Gamma_v2 = 2.0 * Bconst / (rhoh * gamma_drift);
      vpar = Gamma_v2 / ((1.0 + sqrt(fabs(1.0 + Gamma_v2))) * gamma_drift);
      Real Gamma = 1.0 / sqrt(fabs(1.0 - vdriftsqr - vpar * vpar));
      if (!isfinite(Gamma)) {Gamma = 1.0;}
      Wv_u[0] = (vdrift_u[0] + B_u[0] * vpar / Bnorm) * Gamma;
      Wv_u[1] = (vdrift_u[1] + B_u[1] * vpar / Bnorm) * Gamma;
      Wv_u[2] = (vdrift_u[2] + B_u[2] * vpar / Bnorm) * Gamma;

      Wvsq = SquareVector(Wv_u, g3d);
      Gamma = sqrt(1.0 + fabs(Wvsq));
      if ((!isfinite(Gamma)) || (!isfinite(vpar))) {
        Wv_u[0] = 0.0;
        Wv_u[1] = 0.0;
        Wv_u[2] = 0.0;
      }
    }
  }

  // Reinforce sigma limit regardless of primitive floor or drift floor
  Real Wvsq = SquareVector(Wv_u, g3d);
  W = sqrt(1 + Wvsq);
  Real Wv_d[3] = {0.0};
  LowerVector(Wv_d, Wv_u, g3d);
  Real Bdotv = Contract(B_u, Wv_d) / W;
  Real Bsq = SquareVector(B_u, g3d);
  Real b2_local = Bsq / (1 + Wvsq) + Bdotv * Bdotv;
  bool sigma_floor = eos.ReinforceSigmaLimit(n, b2_local);

  // Apply Lorentz capping
  Real W_max = eos.GetMaximumLorentz();
  bool W_cap = false;
  if (W > W_max) {
    Wv_u[0] *= sqrt((W_max * W_max - 1) / (W * W - 1));
    Wv_u[1] *= sqrt((W_max * W_max - 1) / (W * W - 1));
    Wv_u[2] *= sqrt((W_max * W_max - 1) / (W * W - 1));
    W_cap = true;
  }

  // Decide whether we need to update conservative variables
  solver_result.cons_adjusted = solver_result.prim_floor || drift_floor || sigma_floor || W_cap;
  // Update primitive variables first
  prim[PRH] = n;
  prim[PPR] = P;
  prim[PTM] = T;
  prim[PVX] = Wv_u[0];
  prim[PVY] = Wv_u[1];
  prim[PVZ] = Wv_u[2];
  for (int s = 0; s < n_species; s++) {
    prim[PYF + s] = Y[s];
  }

  // If we floored the primitive variables, we should check
  // if the EOS wants us to adjust the conserved variables back
  // in bounds. If that's the case, then we'll do it.
  if (solver_result.cons_adjusted && eos.KeepPrimAndConConsistent()) {
    PrimToCon(prim, cons, b, g3d);
  } else {
    solver_result.cons_adjusted = false;
  }

  return solver_result;
}
// }}}

// PrimToCon {{{
template<typename EOSPolicy, typename ErrorPolicy>
KOKKOS_INLINE_FUNCTION
Error PrimitiveSolver<EOSPolicy, ErrorPolicy>::PrimToCon(Real prim[NPRIM],
      Real cons[NCONS], Real bu[NMAG], Real g3d[NSPMETRIC]) const {
  // Extract the primitive variables
  const Real &n = prim[PRH]; // number density
  const Real Wv_u[3] = {prim[PVX], prim[PVY], prim[PVZ]};
  const Real &p   = prim[PPR]; // pressure
  const Real &t   = prim[PTM]; // temperature
  const Real B_u[3] = {bu[IBX], bu[IBY], bu[IBZ]};
  const int n_species = eos.GetNSpecies();
  Real Y[MAX_SPECIES] = {0.0};
  for (int s = 0; s < n_species; s++) {
    Y[s] = prim[PYF + s];
  }

  // Note that Athena passes in Wv, not v.
  // Lower u.
  Real Wv_d[3];
  LowerVector(Wv_d, Wv_u, g3d);
  Real Wvsq = Contract(Wv_u, Wv_d);
  Real Wsq = 1.0 + Wvsq;
  Real W = sqrt(Wsq);
  Real iW = 1.0/W;
  // Get the 3-velocity.
  Real v_d[3] = {Wv_d[0]*iW, Wv_d[1]*iW, Wv_d[2]*iW};

  // For the magnetic field contribution, we need to find
  // B_i, B^2, and B^i*v_i.
  Real B_d[3];
  LowerVector(B_d, B_u, g3d);
  Real Bsq = Contract(B_u, B_d);
  Real Bv = Contract(B_u, v_d);

  // Some utility quantities that will be helpful.
  const Real mb = eos.GetBaryonMass();

  // Extract the conserved variables
  Real &D = cons[CDN]; // Relativistic density
  Real &Sx = cons[CSX]; // Relativistic momentum density (x)
  Real &Sy = cons[CSY]; // Relativistic momentum density (y)
  Real &Sz = cons[CSZ]; // Relativistic momentum density (z)
  Real &tau = cons[CTA]; // Relativistic energy - D

  // Set the conserved quantities.
  // Total enthalpy density
  Real H = n*eos.GetEnthalpy(n, t, Y)*mb;
  Real HWsq = H*Wsq;
  D = n*mb*W;
  for (int s = 0; s < n_species; s++) {
    cons[CYD + s]= D*Y[s];
  }
  Real HWsqpb = HWsq + Bsq;
  Sx = (HWsqpb*v_d[0] - Bv*B_d[0]);
  Sy = (HWsqpb*v_d[1] - Bv*B_d[1]);
  Sz = (HWsqpb*v_d[2] - Bv*B_d[2]);
  tau = (HWsqpb - p - 0.5*(Bv*Bv + Bsq*(iW*iW))) - D;

  return Error::SUCCESS;
}
// }}}

} // namespace Primitive
#endif  // EOS_PRIMITIVE_SOLVER_PRIMITIVE_SOLVER_HPP_
