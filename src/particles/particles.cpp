//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.cpp
//! \brief implementation of Particles class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack) {
  // check this is at least a 2D problem
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle module only works in 2D/3D" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // select particle type
  {
    std::string ptype = pin->GetString("particles","particle_type");
    if (ptype.compare("cosmic_ray") == 0) {
      particle_type = ParticleType::cosmic_ray;
      q_over_m = pin->GetReal("particles", "charge_mass_ratio");
    } else if (ptype.compare("dust") == 0) {
      particle_type = ParticleType::dust;
      mass = pin->GetOrAddReal("particles", "mass", 1.0);
      q_over_m = 0.0;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle type = '" << ptype << "' not recognized"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // select pusher algorithm
  {
    std::string ppush = pin->GetString("particles","pusher");
    if (ppush.compare("drift") == 0) {
      pusher = ParticlesPusher::drift;
    } else if (ppush.compare("boris") == 0) {
      pusher = ParticlesPusher::boris;
    } else if (ppush.compare("gr_boris") == 0) {
      if (pmy_pack->padm == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "GR Boris pusher needs ADM variables" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      pusher = ParticlesPusher::gr_boris;
    } else if (ppush.compare("geo_boris") == 0) {
      if (pmy_pack->padm == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Geodesic Boris pusher needs ADM variables" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      pusher = ParticlesPusher::geo_boris;
    } else if (ppush.compare("geo_boris_fw") == 0 ||
               ppush.compare("geo_boris_fw_boris") == 0) {
      if (pmy_pack->padm == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Transported-tetrad geo_boris_fw pushers need ADM variables"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (std::abs(q_over_m) > 0.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << ppush
                  << " is geodesic-only in this experimental version"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (pmy_pack->pz4c != nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << ppush
                  << " currently supports stationary ADM metrics only"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      pusher = (ppush.compare("geo_boris_fw") == 0) ?
               ParticlesPusher::geo_boris_fw :
               ParticlesPusher::geo_boris_fw_boris;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher must be specified in <particles> block"
                <<std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // set dimensions of particle arrays. Note particles only work in 2D/3D
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particles only work in 2D/3D, but 1D problem initialized" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // set number of particle properties
  int ndim=2;
  if (pmy_pack->pmesh->three_d) {ndim+=1;}
  nrdata = ndim * 2;
  if (pusher == ParticlesPusher::geo_boris) {
    nrdata = ndim * 3;
  }
  nrdata += 1; // conserved energy u_t
  if (pusher == ParticlesPusher::geo_boris_fw ||
      pusher == ParticlesPusher::geo_boris_fw_boris) {
    nrdata = IPFW + 16; // transported tetrad e_hat_a^mu lives in IPFW..IPFW+15
  }
  nidata = 2;
  // switch (particle_type) {
  //   case ParticleType::cosmic_ray:
  //     {}
  //   default:
  //     break;
  // }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  std::string init = pin->GetOrAddString("particles", "init", "ppc");
  if (init.compare("ppc") == 0){
    // read number of particles per cell, and calculate number of particles this pack
    Real ppc = pin->GetOrAddReal("particles", "ppc", 0.);
    // compute number of particles as real number, since ppc can be < 1
    Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
    // then cast to integer
    nprtcl_thispack = static_cast<int>(r_npart);
    Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
    Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);
  } else if (init.compare("file") == 0) {
    std::string fname_str = pin->GetString("particles", "prtcl_init_file");
    const char* fname = fname_str.c_str();
    // load ptcl_data
    read_prtcl_table(fname);
  }

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);

  // Allocate memory for field variables in previous step
  if (pusher == ParticlesPusher::gr_boris ||
      pusher == ParticlesPusher::geo_boris ||
      pusher == ParticlesPusher::geo_boris_fw ||
      pusher == ParticlesPusher::geo_boris_fw_boris) {
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));

    // allocate memory for mhd variables depending on whether mhd is enabled
    if (pmy_pack->pmhd != nullptr) {
      int nvar = pmy_pack->pmhd->nmhd + pmy_pack->pmhd->nscalars;
      Kokkos::realloc(w0_last, nmb, nvar, ncells3, ncells2, ncells1);
      Kokkos::realloc(bcc0_last, nmb, 3, ncells3, ncells2, ncells1);
    }

    // allocate memory for spacetime variables depending on dynamical evolution of z4c
    if (pmy_pack->pz4c == nullptr) {
      int nadm = pmy_pack->padm->nadm;
      Kokkos::realloc(adm_last, nmb, nadm, ncells3, ncells2, ncells1);
    } else {
      int nadm = pmy_pack->padm->nadm - 4;
      Kokkos::realloc(adm_last, nmb, nadm, ncells3, ncells2, ncells1);
      int nz4c = pmy_pack->pz4c->nz4c;
      Kokkos::realloc(z4c_last, nmb, nz4c, ncells3, ncells2, ncells1);
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
}

//----------------------------------------------------------------------------------------
// CreateParticleTags()
// Assigns tags to particles (unique integer).  Note that tracked particles are always
// those with tag numbers less than ntrack.

void Particles::CreateParticleTags(ParameterInput *pin) {
  std::string assign = pin->GetOrAddString("particles","assign_tag","index_order");

  // tags are assigned sequentially within this rank, starting at 0 with rank=0
  if (assign.compare("index_order") == 0) {
    int tagstart = 0;
    for (int n=1; n<=global_variable::my_rank; ++n) {
      tagstart += pmy_pack->pmesh->nprtcl_eachrank[n-1];
    }

    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = tagstart + p;
    });

  // tags are assigned sequentially across ranks
  } else if (assign.compare("rank_order") == 0) {
    int myrank = global_variable::my_rank;
    int nranks = global_variable::nranks;
    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = myrank + nranks*p;
    });

  // tag algorithm not recognized, so quit with error
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle tag assignment type = '" << assign << "' not recognized"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace particles
