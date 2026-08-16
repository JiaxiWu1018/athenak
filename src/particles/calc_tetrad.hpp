#ifndef PARTICLES_CALC_TETRAD_HPP_
#define PARTICLES_CALC_TETRAD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file calc_tetrad.hpp
//! \brief build the local orthonormal tetrad (and its inverse) of the normal observer
//! from the ADM lapse/shift/3-metric, and transform spatial 3-vectors between the
//! coordinate basis and the tetrad frame. The GR Boris pusher applies its electromagnetic
//! half-kicks in this locally-flat frame (FlatBorisPush) and converts back. Following
//! eqs.(A14-A21) of Bacchini/Boyeneni et al. (CalcFaraday, used only by geo_boris, is
//! intentionally omitted.)

#include <cmath>

#include "athena.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace particles {

//----------------------------------------------------------------------------------------
//! \fn void CalcTetrad
//! \brief tetrad A^{a}_{mu} (lower-triangular spatial block) and its inverse A^{mu}_{a}
//! from lapse alp, shift beta^i and the symmetric 3-metric
//! g3d[6] = {gxx,gxy,gxz,gyy,gyz,gzz}.

KOKKOS_INLINE_FUNCTION
void CalcTetrad(const Real alp, const Real beta[3], const Real g3d[6],
                Real tetrad[4][4], Real inv_tetrad[4][4]) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      tetrad[i][j] = 0.0;
      inv_tetrad[i][j] = 0.0;
    }
  }
  // tetrad A^{a}_{mu}
  tetrad[0][0] = alp;
  tetrad[1][1] = std::sqrt(g3d[0]);
  Real ig11 = 1. / std::sqrt(g3d[0]);
  tetrad[1][2] = g3d[1] * ig11;
  tetrad[1][3] = g3d[2] * ig11;
  tetrad[2][2] = std::sqrt((g3d[0] * g3d[3] - g3d[1] * g3d[1])) * ig11;
  Real inv_denom = ig11 / std::sqrt(g3d[0] * g3d[3] - g3d[1] * g3d[1]);
  tetrad[2][3] = (g3d[0] * g3d[4] - g3d[1] * g3d[2]) * inv_denom;
  tetrad[3][3] = std::sqrt((g3d[0] * g3d[5] - g3d[2] * g3d[2]) *
                           (g3d[0] * g3d[3] - g3d[1] * g3d[1]) -
                           (g3d[0] * g3d[4] - g3d[1] * g3d[2]) *
                           (g3d[0] * g3d[4] - g3d[1] * g3d[2])) * inv_denom;
  for (int i = 1; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      tetrad[i][0] += tetrad[i][j + 1] * beta[j];
    }
  }

  // inverse tetrad A^{mu}_{a}
  Real det = Primitive::GetDeterminant(g3d);
  Real g3u[6] = {0.0};
  Primitive::InvertMatrix(g3u, g3d, det);
  Real ialp = 1. / alp;
  inv_tetrad[0][0] = ialp;
  for (int i = 0; i < 3; ++i) {
    inv_tetrad[i + 1][0] = -ialp * beta[i];
  }
  inv_tetrad[1][1] = ig11;
  inv_tetrad[1][2] = -g3d[1] * inv_denom;
  inv_tetrad[2][2] = g3d[0] * inv_denom;
  Real ig33 = 1. / std::sqrt(g3u[5]);
  inv_tetrad[1][3] = g3u[2] * ig33;
  inv_tetrad[2][3] = g3u[4] * ig33;
  inv_tetrad[3][3] = std::sqrt(g3u[5]);
}

//----------------------------------------------------------------------------------------
//! \fn void TetradCvrtU
//! \brief contravariant transform of a spatial 3-vector: v_out^i = A^{i}_{j} v_in^j using
//! the spatial block of the (forward) tetrad.

KOKKOS_INLINE_FUNCTION
void TetradCvrtU(Real v_out[3], const Real v_in[3], const Real tetrad[4][4]) {
  for (int i = 0; i < 3; ++i) {
    v_out[i] = 0.0;
    for (int j = 0; j < 3; ++j) {
      v_out[i] += tetrad[i + 1][j + 1] * v_in[j];
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void TetradCvrtL
//! \brief covariant transform of a spatial 3-vector: v_out_i = A^{j}_{i} v_in_j using the
//! transpose of the spatial tetrad block (used for u_i, E_i, and to map back).

KOKKOS_INLINE_FUNCTION
void TetradCvrtL(Real v_out[3], const Real v_in[3], const Real tetrad[4][4]) {
  for (int i = 0; i < 3; ++i) {
    v_out[i] = 0.0;
    for (int j = 0; j < 3; ++j) {
      v_out[i] += tetrad[j + 1][i + 1] * v_in[j];
    }
  }
}

} // namespace particles

#endif // PARTICLES_CALC_TETRAD_HPP_
