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

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

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

  // select particle type and per-type attributes
  {
    std::string ptype = pin->GetString("particles","particle_type");
    if (ptype.compare("cosmic_ray") == 0) {
      particle_type = ParticleType::cosmic_ray;
      // charge-to-mass ratio is only used by the (Stage 2) EM pushers; default keeps
      // existing drift inputs working without the key.
      q_over_m = pin->GetOrAddReal("particles","charge_mass_ratio",1.0);
      mass = pin->GetOrAddReal("particles","mass",1.0);
    } else if (ptype.compare("dust") == 0) {
      particle_type = ParticleType::dust;
      mass = pin->GetOrAddReal("particles","mass",1.0);
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
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher must be specified in <particles> block"
                <<std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Particle real data uses the contiguous layout {IPM,IPEN,IPX,IPVX,IPY,IPVY,IPZ,IPVZ}:
  // dimension-independent scalars first, then position+velocity with the z-pair last so a
  // 2D problem can drop it (8 reals in 3D, 6 in 2D). int data = {PGID,PTAG}.
  nrdata = (pmy_pack->pmesh->three_d) ? (IPVZ+1) : (IPVY+1);
  nidata = 2;

  // populate particles: on restart the ProblemGenerator restart constructor restores them
  // (allocate empty here); otherwise load from an HDF5 file (init=file) or create ppc
  // particles-per-cell (init=ppc, filled by the problem generator).
  std::string init = pin->GetOrAddString("particles","init","ppc");
  if (pmy_pack->pmesh->is_restart) {
    nprtcl_thispack = 0;
    Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
    Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);
  } else if (init.compare("ppc") == 0) {
    // number of particles as a real (ppc can be < 1), then cast to integer
    Real ppc = pin->GetOrAddReal("particles","ppc",1.0);
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
    Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
    nprtcl_thispack = static_cast<int>(r_npart);
    Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
    Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);
  } else if (init.compare("file") == 0) {
    std::string fname = pin->GetString("particles","prtcl_init_file");
    read_prtcl_table(fname.c_str());   // sets nprtcl_thispack and reallocs the arrays
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<particles> init = '" << init << "' not recognized (use ppc|file)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
}

//----------------------------------------------------------------------------------------
//! \fn int Particles::FindContainingMeshBlock
//! \brief return the local MeshBlock index whose bounding box contains (x,y,z) using a
//! half-open [min,max) convention, or -1 if none on this rank. The z- (and in 1D, y-) test
//! is skipped when the problem is not three_d (multi_d), where xNmin==xNmax. Host-side;
//! used by the HDF5 reader and the restart reader to assign loaded particles to MeshBlocks.

int Particles::FindContainingMeshBlock(Real x, Real y, Real z) const {
  int nmb = pmy_pack->nmb_thispack;
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  for (int m=0; m<nmb; ++m) {
    if (x < size.h_view(m).x1min || x >= size.h_view(m).x1max) {continue;}
    if (multi_d && (y < size.h_view(m).x2min || y >= size.h_view(m).x2max)) {continue;}
    if (three_d && (z < size.h_view(m).x3min || z >= size.h_view(m).x3max)) {continue;}
    return m;
  }
  return -1;
}

//----------------------------------------------------------------------------------------
// CreateParticleTags()
// Assigns tags to particles (unique integer).  Note that tracked particles are always
// those with tag numbers less than ntrack.

void Particles::CreateParticleTags(ParameterInput *pin) {
  // file-loaded particles already carry globally-unique file-index tags (read_particle.cpp);
  // do not overwrite them with the decomposition-dependent index_order/rank_order tags
  // (a given physical particle must keep the same tag regardless of the MPI rank count).
  if (pin->GetOrAddString("particles","init","ppc").compare("file") == 0) {return;}

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
