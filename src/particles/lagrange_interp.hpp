#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace particles {

template <int ORDER>
KOKKOS_INLINE_FUNCTION
void CalcInterpWght(const Real *x0, const Real *grid, const int *ncell, const int *interp_indcs,
                    Real *Lx, Real *Ly, Real *Lz) {
  constexpr int N = 2 * ORDER;
  Real xmin = grid[0], xmax = grid[1];
  Real ymin = grid[3], ymax = grid[4];
  Real zmin = grid[6], zmax = grid[7];
  Real x[N] = {0.0}, y[N] = {0.0}, z[N] = {0.0};
  for (int i = 0; i < N; ++i) {
    x[i] = CellCenterX(interp_indcs[1] - ORDER + 1 + i, ncell[0], xmin, xmax);
    y[i] = CellCenterX(interp_indcs[2] - ORDER + 1 + i, ncell[1], ymin, ymax);
    z[i] = CellCenterX(interp_indcs[3] - ORDER + 1 + i, ncell[2], zmin, zmax);
  }

  for (int i = 0; i < N; ++i) {
    Lx[i] = 1.0, Ly[i] = 1.0, Lz[i] = 1.0;
    for (int j = 0; j < N; ++j) {
      if (i == j) continue;
      Lx[i] *= (x0[0] - x[j]) / (x[i] - x[j]);
      Ly[i] *= (x0[1] - y[j]) / (y[i] - y[j]);
      Lz[i] *= (x0[2] - z[j]) / (z[i] - z[j]);
    }
  }
}

template <int ORDER>
KOKKOS_INLINE_FUNCTION
void CalcInterpWghtAndDrv(const Real *x0, const Real *grid, const int * ncell, const int *interp_indcs,
                          Real *Lx, Real *Ly, Real *Lz, Real *dLx, Real *dLy, Real *dLz) {
  constexpr int N = 2 * ORDER;
  Real xmin = grid[0], xmax = grid[1];
  Real ymin = grid[3], ymax = grid[4];
  Real zmin = grid[6], zmax = grid[7];
  Real x[N] = {0.0}, y[N] = {0.0}, z[N] = {0.0};
  for (int i = 0; i < N; ++i) {
    x[i] = CellCenterX(interp_indcs[1] - ORDER + 1 + i, ncell[0], xmin, xmax);
    y[i] = CellCenterX(interp_indcs[2] - ORDER + 1 + i, ncell[1], ymin, ymax);
    z[i] = CellCenterX(interp_indcs[3] - ORDER + 1 + i, ncell[2], zmin, zmax);
  }

  for (int i = 0; i < N; ++i) {
    Lx[i] = 1.0, Ly[i] = 1.0, Lz[i] = 1.0;
    dLx[i] = 0.0, dLy[i] = 0.0, dLz[i] = 0.0;
    for (int j = 0; j < N; ++j) {
      if (j == i) continue;
      Lx[i] *= (x0[0] - x[j]) / (x[i] - x[j]);
      Ly[i] *= (x0[1] - y[j]) / (y[i] - y[j]);
      Lz[i] *= (x0[2] - z[j]) / (z[i] - z[j]);
      Real xterm = 1.0 / (x[i] - x[j]);
      Real yterm = 1.0 / (y[i] - y[j]);
      Real zterm = 1.0 / (z[i] - z[j]);
      for (int k = 0; k < N; ++k) {
        if (k == i || k == j) continue;
        xterm *= (x0[0] - x[k]) / (x[i] - x[k]);
        yterm *= (x0[1] - y[k]) / (y[i] - y[k]);
        zterm *= (x0[2] - z[k]) / (z[i] - z[k]);
      }
      dLx[i] += xterm;
      dLy[i] += yterm;
      dLz[i] += zterm;
    }
  }
}

template <int ORDER>
KOKKOS_INLINE_FUNCTION
Real LagrangeInterpolator(const DvceArray5D<Real> &u0, const int nvar, const int *interp_indcs,
                          const Real *Lx, const Real *Ly, const Real *Lz) {
  constexpr int N = 2 * ORDER;
  Real results = 0.0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        Real weight = Lx[i] * Ly[j] * Lz[k];
        // u0 has ghosts zones
        // i.e. indc[1] - (ORDER - 1) + i + NGHOST and NGHOST=ORDER
        results += weight * u0(interp_indcs[0], nvar, interp_indcs[3] + k + 1,
                               interp_indcs[2] + j + 1, interp_indcs[1] + i + 1);
      }
    }
  }
  return results;
}

template <int ORDER>
KOKKOS_INLINE_FUNCTION
void LagrangeInterpolator(const DvceArray5D<Real> &u0, const int nvar, const int *interp_indcs,
                          const Real *Lx, const Real *Ly, const Real *Lz, Real *results) {
  for (int i = 0; i < 6; ++i) {
    results[i] = 0.0;
  }
  constexpr int N = 2 * ORDER;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        Real weight = Lx[i] * Ly[j] * Lz[k];
        // u0 has ghosts zones
        // There are 6 variables for g3d
        Real g3d[6] = {0.0};
        for (int m = 0; m < 6; ++m) {
          g3d[m] = u0(interp_indcs[0], nvar+m, interp_indcs[3] + k + 1,
                      interp_indcs[2] + j + 1, interp_indcs[1] + i + 1);
        }
        Real g3u[6] = {0.0};
        Primitive::InvertMatrix(g3u, g3d, Primitive::GetDeterminant(g3d));
        for (int m = 0; m < 6; ++m) {
          results[m] += weight * g3u[m];
        }
      }
    }
  }
}
} // end namespace particles