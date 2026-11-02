#ifndef PARTICLES_RK_MIDPOINT_HPP_
#define PARTICLES_RK_MIDPOINT_HPP_

#include <cmath>

namespace particles {

// Solve z = z_n + h G(U_B, (z_n+z)/2). F evaluates the right side with U_B fixed.
// The residual is checked at the returned iterate, never at an unverified next value.
// No Euler fallback: failed solves must not silently reduce the temporal order.
template <class F>
KOKKOS_INLINE_FUNCTION
bool RKMidpointSolve(const F &f, const Real x0[3], const Real u0[3], const Real dx[3],
                     Real x[3], Real u[3], Real tol, int max_iter,
                     int &iterations, Real &residual) {
  iterations = 0;
  residual = 0.0;
  for (int it = 0; it < max_iter; ++it) {
    Real next_x[3], next_u[3];
    iterations = it+1;
    // Bound the spatial midpoint by half a cell of the original owner. The existing
    // clamped Lagrange stencil then stays on its current ghost-supported interval.
    for (int a = 0; a < 3; ++a) {
      if (!std::isfinite(x[a]) || !std::isfinite(u[a]) ||
          fabs(x[a]-x0[a]) > dx[a]) {return false;}
    }
    if (!f(x, u, next_x, next_u, false)) {return false;}
    residual = 0.0;
    for (int a = 0; a < 3; ++a) {
      if (!std::isfinite(next_x[a]) || !std::isfinite(next_u[a])) {return false;}
      residual = fmax(residual, fabs(next_x[a]-x[a])/dx[a]);
      residual = fmax(residual, fabs(next_u[a]-u[a])/fmax(1.0, fabs(u0[a])));
    }
    if (residual <= tol) {return true;}
    for (int a = 0; a < 3; ++a) {x[a] = next_x[a]; u[a] = next_u[a];}
  }
  return false;
}

} // namespace particles
#endif  // PARTICLES_RK_MIDPOINT_HPP_
