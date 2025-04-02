//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture_UIUC.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain using
//         UIUC initial data

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"


void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin);
void RefinementCondition(MeshBlockPack* pmbp);
void Lorentz_Transformation_3D(Real LT[3][3], Real iLT[3][3], const Real par_P[3], const Real M);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func  = RefinementCondition;

  if (restart)
    return;
  
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  ADMOnePuncture(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  std::cout<<"OnePuncture initialized."<<std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to single puncture (no spin)

void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  // For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;

  Real M = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.); // BH ADM mass
  Real chi = pin->GetOrAddReal("problem", "punc_spin", 0.); // Dimensionless spin J/M^2
  Real a = M * chi; // Spin per unit mass
  Real rm = M - sqrt(M * M - a * a); // Inner horizon
  Real rp = M + sqrt(M * M - a * a); // Outer horizon

  Real center_offset[3] = {0., 0., 0.}; // Location of the puncture
  center_offset[0] = pin->GetOrAddReal("problem", "punc_center_x1", 0.);
  center_offset[1] = pin->GetOrAddReal("problem", "punc_center_x2", 0.);
  center_offset[2] = pin->GetOrAddReal("problem", "punc_center_x3", 0.);

  Real par_P[3] = {0., 0., 0.}; // Momentum of the puncture
  par_P[0] = pin->GetOrAddReal("problem", "punc_p1", 0.);
  par_P[1] = pin->GetOrAddReal("problem", "punc_p2", 0.);
  par_P[2] = pin->GetOrAddReal("problem", "punc_p3", 0.);

  Real LT[3][3], iLT[3][3];
  Lorentz_Transformation_3D(LT, iLT, par_P, M);

  // Not evaluate on puncture
  bool avoid_puncture = pin->GetOrAddBoolean("problem", "avoid_punc", 1.);
  Real r_small = pin->GetOrAddReal("problem", "r_small", 0.001);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen one puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real xf[3]; // Field coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    xf[0] = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    xf[1] = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    xf[2] = CellCenterX(k-ks, nx3, x3min, x3max);

    xf[0] -= center_offset[0];
    xf[1] -= center_offset[1];
    xf[2] -= center_offset[2];

    Real xS[3]; // Stationary coordinates
    for (int ii = 0; ii < 3; ++ii) {
      xS[ii] = 0;
      for (int jj = 0; jj < 3; ++jj) {
        xS[ii] += iLT[ii][jj] * xf[jj];
      }
    }

    Real rUIUC = sqrt(xS[0] * xS[0] + xS[1] * xS[1] + xS[2] * xS[2]); // radial coord
    // If we are too close to the singularity, shift away
    if (rUIUC < r_small && avoid_puncture) {rUIUC = r_small;}

    Real th = acos(xS[2] / rUIUC); // UIUC latitude coord
    Real ph = atan2(xS[1], xS[0]); // UIUC longitude coord
    Real Sth = sin(th), Cth = cos(th), Tth = tan(th), Sph = sin(ph), Cph = cos(ph);
    Real rBL = rUIUC * pow(1 + 0.25 * rp / rUIUC, 2); // Boyer-Lindquist radial coord
    Real SIG = rBL * rBL + pow(a * Cth, 2); // Boyer-Lindquist Sigma
    Real DEL = rBL * rBL - 2 * M * rBL + a * a; // Boyer-Lindquist Delta
    Real A = pow(rBL * rBL + a * a, 2) - DEL * pow(a * Sth, 2);

    // Physical spatial metric and extrinsic curvature in spherical basis
    Real grr = SIG * pow(rUIUC + 0.25 * rp, 2) * pow(rUIUC, -3) / (rBL - rm);
    Real gthth = SIG;
    Real gphph = A * Sth * Sth / SIG;
    Real Krph = M * a * Sth * Sth * (3 * pow(rBL, 4) + 2 * pow(a * rBL, 2) - pow(a, 4)
                                     - (rBL * rBL - a * a) * pow(a * Sth, 2))
                * (1 + 0.25 * rp / rUIUC) / (SIG * sqrt(A * SIG * rUIUC * (rBL - rm)));
    Real Kthph = - 2 * pow(a, 3) * M * rBL * Cth * pow(Sth, 3) * (rUIUC - 0.25 * rp)
                 * sqrt(rBL - rm) / (SIG * sqrt(A * SIG * rUIUC));
    
    // Stationary spatial metric and extrinsic curvature
    Real g0[3][3], K0[3][3];
    if (rUIUC * Sth < 1.0e-10) {
      g0[0][0] = (a * a + pow(rp + 4 * fabs(xS[2]), 4) / (256 * xS[2] * xS[2])) / (xS[2] * xS[2]);
      g0[0][1] = 0;
      g0[0][2] = 0;
      g0[1][1] = g0[0][0];
      g0[1][2] = 0;
      g0[2][2] = -pow(rp + 4 * fabs(xS[2]), 2) * g0[0][0] / (a * a - 2 * (M * rp + 8 * xS[2] * xS[2]) + 8 * fabs(xS[2]) * (M - 3 * sqrt(M * M - a * a)));
      g0[1][0] = g0[0][1];
      g0[2][0] = g0[0][2];
      g0[2][1] = g0[1][2];

      for(int ii = 0; ii < 3; ii++)
      {
        for(int jj = 0; jj < 3; jj++)
        {
          K0[ii][jj] = 0;
        }
      }
    } else {
      // ADMBase physical spatial metric (Cartesian basis)
      g0[0][0] = (gthth * pow(Cph * Cth, 2)) / pow(rUIUC, 2) + (gphph * pow(Sph, 2)) / (pow(rUIUC, 2) * pow(Sth, 2)) + grr * pow(Cph, 2) * pow(Sth, 2);
      g0[0][1] = (gthth * Cph * pow(Cth, 2) * Sph) / pow(rUIUC, 2) - (gphph * Cph * Sph) / (pow(rUIUC, 2) * pow(Sth, 2)) + grr * Cph * Sph * pow(Sth, 2);
      g0[0][2] = grr * Cph * Cth * Sth - (gthth * Cph * Cth * Sth) / pow(rUIUC, 2);
      g0[1][1] = (gphph * pow(Cph, 2)) / (pow(rUIUC, 2) * pow(Sth, 2)) + (gthth * pow(Cth, 2) * pow(Sph, 2)) / pow(rUIUC, 2) + grr * pow(Sph, 2) * pow(Sth, 2);
      g0[1][2] = grr * Cth * Sph * Sth - (gthth * Cth * Sph * Sth) / pow(rUIUC, 2);
      g0[2][2] = grr * pow(Cth, 2) + (gthth * pow(Sth, 2)) / pow(rUIUC, 2);
      
      g0[1][0] = g0[0][1];
      g0[2][0] = g0[0][2];
      g0[2][1] = g0[1][2];

      // ADMBase physical extrinsic curvature (Cartesian basis)
      K0[0][0] = (-2 * Krph * Cph * Sph) / rUIUC - (2 * Kthph * Cph / Tth * Sph) / pow(rUIUC, 2);
      K0[0][1] = (Krph * pow(Cph, 2)) / rUIUC + (Kthph * pow(Cph, 2) / Tth) / pow(rUIUC, 2) - (Krph * pow(Sph, 2)) / rUIUC - (Kthph / Tth * pow(Sph, 2)) / pow(rUIUC, 2);
      K0[0][2] = (Kthph * Sph) / pow(rUIUC, 2) - (Krph / Tth * Sph) / rUIUC;
      K0[1][1] = (2 * Krph * Cph * Sph) / rUIUC + (2 * Kthph * Cph / Tth * Sph) / pow(rUIUC, 2);
      K0[1][2] = -((Kthph * Cph) / pow(rUIUC, 2)) + (Krph * Cph / Tth) / rUIUC;
      K0[2][2] = 0;

      K0[1][0] = K0[0][1];
      K0[2][0] = K0[0][2];
      K0[2][1] = K0[1][2];
    }

    // Lorentz transformation
    Real g_dd[3][3], vK_dd[3][3];
    for(int ii = 0; ii < 3; ii++) {
      for(int jj = ii; jj < 3; jj++) {
        g_dd[ii][jj] = 0;
        vK_dd[ii][jj] = 0;
        for(int kk = 0; kk < 3; kk++) {
          for(int ll = 0; ll < 3; ll++) {
            g_dd[ii][jj] += LT[ii][kk] * LT[jj][ll] * g0[kk][ll];
            vK_dd[ii][jj] += LT[ii][kk] * LT[jj][ll] * K0[kk][ll];
          }
        }
      }
    }

    // Determinant of metric and lapse
    Real detg = -g_dd[0][2] * g_dd[0][2] * g_dd[1][1] + 2 * g_dd[0][1] * g_dd[0][2] * g_dd[1][2] - g_dd[0][0] * g_dd[1][2] * g_dd[1][2] - g_dd[0][1] * g_dd[0][1] * g_dd[2][2] + g_dd[0][0] * g_dd[1][1] * g_dd[2][2];
    Real psi = pow(detg, 1.0/12.0);
    adm.alpha(m, k, j, i) = pow(psi, -2);
    adm.psi4(m, k, j, i) = pow(psi, 4);

    // Shift vector
    adm.beta_u(m, 0, k, j, i) = 0;
    adm.beta_u(m, 1, k, j, i) = 0;
    adm.beta_u(m, 2, k, j, i) = 0;
    
    // Metric and Curvature
    for(int ii = 0; ii < 3; ii++) {
      for(int jj = ii; jj < 3; jj++) {
        adm.g_dd(m, ii, jj, k, j, i) = g_dd[ii][jj];
        adm.vK_dd(m, ii, jj, k, j, i) = vK_dd[ii][jj];
      }
    }

    return;
  });
}

// how decide the refinement
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

void Lorentz_Transformation_3D(Real LT[3][3], Real iLT[3][3], const Real par_P[3], const Real M) {
  Real KD[3][3] = {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}; // Kronecker
  Real Pmag = sqrt(par_P[0] * par_P[0] + par_P[1] * par_P[1] + par_P[2] * par_P[2]); // Momentum magnitude
  Real v[3]; // Three-velocity

  for (int i = 0; i < 3; ++i) {
    v[i] = par_P[i] / sqrt(M * M + Pmag * Pmag);
  }
  Real W = pow(1. - v[0] * v[0] - v[1] * v[1] - v[2] * v[2], -0.5); // Lorentz factor

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      LT[i][j] = KD[i][j] + W * W / (1. + W) * v[i] * v[j]; // Lorentz
      iLT[i][j] = KD[i][j] - W / (1. + W) * v[i] * v[j]; // Inverse Lorentz
    }
  }
}