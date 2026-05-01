#include "athena.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace particles {

KOKKOS_INLINE_FUNCTION
void CalcTetrad(const Real alp, const Real beta[3], const Real g3d[6],
                Real tetrad[4][4], Real inv_tetrad[4][4]) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      tetrad[i][j] = 0.0;
      inv_tetrad[i][j] = 0.0;
    }
  }
  // Calculate tetrad A^{\hat{a}}_{\mu} following eq.(A14-A21) of Boyeneni+25
  tetrad[0][0] = alp;
  tetrad[1][1] = std::sqrt(g3d[0]);
  Real ig11 = 1. / std::sqrt(g3d[0]);
  tetrad[1][2] = g3d[1] * ig11;
  tetrad[1][3] = g3d[2] * ig11;
  tetrad[2][2] = std::sqrt((g3d[0] * g3d[3] - g3d[1] * g3d[1])) * ig11;
  Real inv_denom = ig11 / std::sqrt(g3d[0] * g3d[3] - g3d[1] * g3d[1]);
  tetrad[2][3] = (g3d[0] * g3d[4] - g3d[1] * g3d[2]) * inv_denom;
  tetrad[3][3] = std::sqrt((g3d[0] * g3d[5] - g3d[2] * g3d[2]) * (g3d[0] * g3d[3] - g3d[1] * g3d[1]) -
                    (g3d[0] * g3d[4] - g3d[1] * g3d[2]) * (g3d[0] * g3d[4] - g3d[1] * g3d[2])) * inv_denom;
  for (int i = 1; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      tetrad[i][0] += tetrad[i][j + 1] * beta[j];
    }
  }

  // Calculate the inverse of tetrad A^{\mu}_{\hat{a}} following eq.(A3-A10)
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

KOKKOS_INLINE_FUNCTION
void CalcFaraday(const Real &alp, const Real beta[3], const Real g3d[6],
                 const Real dalp[4], const Real dbeta[4][3], const Real dg3d[4][6],
                 const Real tetrad[4][4], Real F[4][4][4]) {
  Real dtetrad[4][4][4] = {0.0};
  for (int i = 0; i < 4; ++i) {
    dtetrad[i][0][0] = dalp[i];
    Real ig11 = 1. / std::sqrt(g3d[0]);
    dtetrad[i][1][1] = 0.5 * dg3d[i][0] * ig11;
    Real ig11_3 = ig11 * ig11 * ig11;
    dtetrad[i][1][2] = dg3d[i][1] * ig11 - 0.5 * g3d[1] * ig11_3 * dg3d[i][0];
    dtetrad[i][1][3] = dg3d[i][2] * ig11 - 0.5 * g3d[2] * ig11_3 * dg3d[i][0];
    // auxiliary derivative of 1/sqrt(g11(g11g22-g12g12))
    Real det12 = g3d[0] * g3d[3] - g3d[1] * g3d[1];
    Real ddet12 = dg3d[i][0] * g3d[3] + g3d[0] * dg3d[i][3] - 2. * g3d[1] * dg3d[i][1];
    Real denom = std::sqrt(g3d[0] * det12);
    Real inv_denom = 1. / denom;
    Real inv_denom_3 = inv_denom * inv_denom * inv_denom;
    Real dinv_denom = -0.5 * inv_denom_3 * (det12 * dg3d[i][0] + g3d[0] * ddet12);
    dtetrad[i][2][2] = ddet12 * inv_denom + det12 * dinv_denom;
    Real det1123 = g3d[0] * g3d[4] - g3d[1] * g3d[2];
    Real ddet1123 = dg3d[i][0] * g3d[4] + g3d[0] * dg3d[i][4] - dg3d[i][1] * g3d[2] - g3d[1] * dg3d[i][2];
    dtetrad[i][2][3] = ddet1123 * inv_denom + det1123 * dinv_denom;
    // auxiliary derivative of sqrt((g11g33-g13g13)(g11g22-g12g12)-(g11g23-g12g13)^2)
    Real det13 = g3d[0] * g3d[5] - g3d[2] * g3d[2];
    Real ddet13 = dg3d[i][0] * g3d[5] + g3d[0] * dg3d[i][5] - 2. * g3d[2] * dg3d[i][2];
    Real nomi = std::sqrt(det13 * det12 - det1123 * det1123);
    Real dnomi = 0.5 / nomi * (ddet13 * det12 + det13 * ddet12 - 2. * det1123 * ddet1123);
    dtetrad[i][3][3] = dnomi * inv_denom + nomi * dinv_denom;
    for (int j = 1; j < 4; ++j) {
      dtetrad[i][j][0] = 0.0;
      for (int k = 0; k < 3; ++k) {
        dtetrad[i][j][0] += dtetrad[i][j][k + 1] * beta[k] + tetrad[j][k + 1] * dbeta[i][k];
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        F[a][mu][nu] = dtetrad[mu][a][nu] - dtetrad[nu][a][mu];
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void TetradCvrtU(Real v_out[3], const Real v_in[3], const Real tetrad[4][4]) {
  for (int i = 0; i < 3; ++i) {
    v_out[i] = 0.0;
    for (int j = 0; j < 3; ++j) {
      v_out[i] += tetrad[i + 1][j + 1] * v_in[j];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void TetradCvrtL(Real v_out[3], const Real v_in[3], const Real tetrad[4][4]) {
  for (int i = 0; i < 3; ++i) {
    v_out[i] = 0.0;
    for (int j = 0; j < 3; ++j) {
      v_out[i] += tetrad[j + 1][i + 1] * v_in[j];
    }
  }
}
} // end namespace particles