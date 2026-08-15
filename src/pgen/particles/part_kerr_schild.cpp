//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file part_kerr_schild.cpp
//! \brief NRPIC Stage-1 verification problem generator: load particles from an HDF5 file
//! (via <particles> init=file) into an analytic Cartesian Kerr-Schild ADM background, so the
//! conserved-energy calculation (calc_energy.cpp) can be checked. A static particle (u_i=0)
//! at areal radius r should get IPEN = lapse alpha = 1/sqrt(1+2M/r) with M=1 (=1 in the
//! Minkowski limit, <coord> minkowski=true). Requires <coord> and <adm> blocks. The
//! particles themselves are read by the Particles constructor; here we only populate the
//! metric. (The fuller Schwarzschild / Kerr-Wald particle pgens are Stage 2.)

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_kerr_schild requires an <adm> block (analytic ADM background)."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_kerr_schild requires a <particles> block (use init=file)."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // Populate the ADM variables. The ADM constructor defaults SetADMVariables to
  // Kerr-Schild, reading the spin from <coord> a and the Minkowski flag from
  // <coord> minkowski (mass M=1). Particles were already loaded by the Particles ctor.
  pmbp->padm->SetADMVariables(pmbp);
}
