//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file Schwz_ptcl.cpp
//! \brief Problem generator for particle pusher tests in Schwarzschild spacetime.
//!
//! Particles are read from an HDF5 file via the standard `<particles> init = file`
//! path. Two coordinate charts are available, selected by `<problem> metric_type`:
//!   - "ks"        : Kerr-Schild (default ADM setter). Requires `<coord> a = 0.0`.
//!   - "isotropic" : Isotropic Schwarzschild, M = 1 (helper defined below).

#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "particles/particles.hpp"

void SetADMVariablesToIsotropic(MeshBlockPack *pmbp);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  std::string metric = pin->GetOrAddString("problem", "metric_type", "ks");
  if (metric == "isotropic") {
    pmbp->padm->SetADMVariables = &SetADMVariablesToIsotropic;
  } else if (metric != "ks") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown metric_type \"" << metric
              << "\" (expected \"ks\" or \"isotropic\")" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  pmbp->padm->SetADMVariables(pmbp);
  Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
}

void SetADMVariablesToIsotropic(MeshBlockPack *pmbp) {
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real r = std::sqrt(x1v * x1v + x2v * x2v + x3v * x3v);
    Real ir = 1. / r;

    adm.alpha(m, k, j, i) = (1. - 0.5 * ir) / (1. + 0.5 * ir);
    adm.beta_u(m, 0, k, j, i) = 0.;
    adm.beta_u(m, 1, k, j, i) = 0.;
    adm.beta_u(m, 2, k, j, i) = 0.;

    Real power4 = (1. + 0.5 * ir) * (1. + 0.5 * ir) * (1. + 0.5 * ir) * (1. + 0.5 * ir);
    adm.psi4(m, k, j, i) = power4;
    adm.g_dd(m, 0, 0, k, j, i) = power4;
    adm.g_dd(m, 1, 1, k, j, i) = power4;
    adm.g_dd(m, 2, 2, k, j, i) = power4;
    adm.g_dd(m, 0, 1, k, j, i) = 0.;
    adm.g_dd(m, 0, 2, k, j, i) = 0.;
    adm.g_dd(m, 1, 2, k, j, i) = 0.;

    adm.vK_dd(m, 0, 0, k, j, i) = 0.;
    adm.vK_dd(m, 0, 1, k, j, i) = 0.;
    adm.vK_dd(m, 0, 2, k, j, i) = 0.;
    adm.vK_dd(m, 1, 1, k, j, i) = 0.;
    adm.vK_dd(m, 1, 2, k, j, i) = 0.;
    adm.vK_dd(m, 2, 2, k, j, i) = 0.;
  });
}
