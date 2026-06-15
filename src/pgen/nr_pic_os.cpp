//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_pic_os.cpp
//! \brief Oppenheimer-Snyder pressureless-dust collapse (NRPIC Stage 4b): a homogeneous
//! dust ball, momentarily at rest, collapsing to a black hole. The dust is represented by
//! self-gravitating particles whose stress-energy is deposited into Tmunu and fed to
//! the Z4c spacetime (requires <z4c> + <particles> feedback=true). The headline NRPIC
//! validation -- compared against the closed-form OS solution and the moving-puncture
//! literature (Staley, Baumgarte, Brown, Farris & Shapiro, CQG 29, 015003 (2012)).
//!
//! Initial data (conformally-flat, time-symmetric K_ij = 0; isotropic Cartesian coords;
//! M = ADM mass, R0 = areal surface radius = os_radius_over_mass * os_mass):
//!   isotropic surface radius  r0  = (R0/2)(1 - M/R0 + sqrt(1 - 2M/R0))
//!   interior (r <= r0)  psi^2 = (1+sqrt(1-2M/R0)) r0 R0^2 / (2 r0^3 + M r^2)
//!   exterior (r >  r0)  psi   = 1 + M/(2r)        [vacuum Schwarzschild-isotropic]
//!   gamma_ij = psi^4 delta_ij  (so sqrt(gamma) = psi^6); gauge alpha = 1, beta^i = 0
//!     (os_precollapsed_lapse=true instead seeds alpha = psi^-2). The 1+log slicing and
//!     Gamma-driver shift are set in <z4c>; the lapse then collapses under evolution.
//! The Hamiltonian constraint (time-symmetric) gives a SPATIALLY-UNIFORM normal-observer
//! energy density rho0 = 3 M r0 / (pi (1+sqrt(1-2M/R0))^2 R0^4) inside the ball; this is
//! the homogeneous OS dust. Particles must reproduce E(x) = rho0 there (u_i = 0, W = 1).
//!
//! Two particle-placement paths (per-particle mass either way -- NOT the legacy m_p=M/N,
//! which ignores both the binding energy and the sqrt(gamma) sampling measure):
//!   init=pgen (recommended): a Cartesian lattice (os_lattice_n cell centres across the
//!     diameter) of cells with r <= r0, each carrying m_p = rho0 psi^6(x_p) h^3 -- the
//!     metric AND the particle masses derive from the same <problem> keys, so they are
//!     consistent by construction (the t=0 Hamiltonian-constraint test then converges).
//!   init=file: particles (incl. per-particle mass) are loaded by the HDF5 reader from
//!     scripts/particles/gen_os_dust.py; os_mass/os_radius_over_mass here MUST match the
//!     generator's --mass/--radius (the constraint test fails loudly on a mismatch).
//!
//! On restart the z4c/ADM state and particles are restored by the restart reader; this
//! pgen then only re-seeds the GR-pusher previous-step snapshots (the z4c_one_puncture
//! pattern; Driver::Initialize refreshes them again after the ghost exchange).

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "particles/particles.hpp"
#include "pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

// conformal factor psi(r) for the interior r <= r0 (host); psi^2 = C / (2 r0^3 + M r^2)
KOKKOS_INLINE_FUNCTION
Real PsiInterior(Real r, Real M, Real r0, Real C) {
  return std::sqrt(C / (2.0 * r0 * r0 * r0 + M * r * r));
}

// staged particle data (host) before the device fill -- carries per-particle mass
struct PrtclStage {
  std::vector<Real> x, y, z, mass;
  std::vector<int> gid, tag;
  void Add(Real x_, Real y_, Real z_, int gid_, int tag_, Real m_) {
    x.push_back(x_); y.push_back(y_); z.push_back(z_);
    gid.push_back(gid_); tag.push_back(tag_); mass.push_back(m_);
  }
};

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "nr_pic_os (Oppenheimer-Snyder collapse) requires a <z4c> block."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "nr_pic_os requires a <particles> block (the collapsing dust)."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // OS parameters (host): M, R0, isotropic surface r0, interior psi^2 numerator C, plus
  // the uniform interior energy density rho0 (see file docstring).
  Real R0_over_M = pin->GetReal("problem", "os_radius_over_mass");
  Real M = pin->GetReal("problem", "os_mass");
  if (R0_over_M <= 2.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "os_radius_over_mass = " << R0_over_M << " must exceed 2 (dust must "
              << "start outside the horizon)." << std::endl;
    exit(EXIT_FAILURE);
  }
  Real R0 = R0_over_M * M;
  Real om2 = 1.0 - 2.0 / R0_over_M;           // 1 - 2M/R0
  Real sq = std::sqrt(om2);
  Real r0 = 0.5 * R0 * (1.0 - 1.0 / R0_over_M + sq);   // isotropic surface radius
  Real Cnum = (1.0 + sq) * r0 * R0 * R0;               // interior psi^2 numerator
  Real rho0 = 3.0 * M * r0 / (M_PI * (1.0 + sq) * (1.0 + sq) * R0 * R0 * R0 * R0);
  bool precollapsed = pin->GetOrAddBoolean("problem", "os_precollapsed_lapse", false);

  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;

  if (!restart) {
    // -------- conformally-flat, time-symmetric ADM initial data (incl. ghosts) --------
    auto &size = pmbp->pmb->mb_size;
    int isg = is - indcs.ng, ieg = indcs.ie + indcs.ng;
    int jsg = js - indcs.ng, jeg = indcs.je + indcs.ng;
    int ksg = ks - indcs.ng, keg = indcs.ke + indcs.ng;
    int nmb = pmbp->nmb_thispack;
    adm::ADM::ADM_vars &adm = pmbp->padm->adm;
    z4c::Z4c::Z4c_vars &z4c = pmbp->pz4c->z4c;
    par_for("pgen nr_pic_os ID", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real r = std::sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
      Real psi4;
      if (r <= r0) {
        Real psi2 = Cnum / (2.0*r0*r0*r0 + M*r*r);
        psi4 = psi2*psi2;
      } else {
        Real psi = 1.0 + 0.5*M/r;
        psi4 = psi*psi*psi*psi;
      }
      adm.psi4(m,k,j,i) = psi4;
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          adm.g_dd(m,a,b,k,j,i) = psi4 * ((a == b) ? 1.0 : 0.0);
          adm.vK_dd(m,a,b,k,j,i) = 0.0;     // time-symmetric
        }
      }
      // gauge: alpha = 1 (default) or pre-collapsed psi^-2; shift beta^i = 0
      z4c.alpha(m,k,j,i) = precollapsed ? 1.0/std::sqrt(psi4) : 1.0;
      for (int a = 0; a < 3; ++a) { z4c.beta_u(m,a,k,j,i) = 0.0; }
    });

    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
    }
    pmbp->pz4c->Z4cToADM(pmbp);
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMConstraints<2>(pmbp); break;
      case 3: pmbp->pz4c->ADMConstraints<3>(pmbp); break;
      case 4: pmbp->pz4c->ADMConstraints<4>(pmbp); break;
    }

    // -------- particle placement --------
    particles::Particles *ppart = pmbp->ppart;
    std::string init = pin->GetOrAddString("particles", "init", "ppc");
    if (init.compare("file") == 0) {
      // particles (incl. per-particle mass) already loaded by the HDF5 reader
    } else if (init.compare("pgen") == 0) {
      if (!pmy_mesh_->three_d) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "nr_pic_os init=pgen is 3D-only." << std::endl;
        exit(EXIT_FAILURE);
      }
      // Cartesian lattice of cell centres in [-r0, r0]^3, kept where r <= r0; each cell
      // carries the proper rest mass m_p = rho0 psi^6 h^3. Iteration order (z slow, x
      // fast) and the running tag match gen_os_dust.py --scheme lattice, so init=pgen
      // and init=file lattices are per-tag comparable.
      int nlat = pin->GetOrAddInteger("problem", "os_lattice_n", 48);
      Real h = 2.0*r0/static_cast<Real>(nlat);
      PrtclStage st;
      int tag = 0;
      for (int kz = 0; kz < nlat; ++kz) {
        Real zc = -r0 + (kz + 0.5)*h;
        for (int ky = 0; ky < nlat; ++ky) {
          Real yc = -r0 + (ky + 0.5)*h;
          for (int kx = 0; kx < nlat; ++kx) {
            Real xc = -r0 + (kx + 0.5)*h;
            Real r = std::sqrt(xc*xc + yc*yc + zc*zc);
            if (r > r0) { continue; }
            Real psi6 = std::pow(PsiInterior(r, M, r0, Cnum), 6);
            Real mp = rho0 * psi6 * h*h*h;
            int mb = ppart->FindContainingMeshBlock(xc, yc, zc);
            if (mb >= 0) { st.Add(xc, yc, zc, pmbp->gids + mb, tag, mp); }
            ++tag;   // global lattice (file-row) index over inside-ball points
          }
        }
      }
      int npart = static_cast<int>(st.x.size());
      Kokkos::realloc(ppart->prtcl_rdata, ppart->nrdata, npart);
      Kokkos::realloc(ppart->prtcl_idata, ppart->nidata, npart);
      auto hr = Kokkos::create_mirror_view(ppart->prtcl_rdata);
      auto hi = Kokkos::create_mirror_view(ppart->prtcl_idata);
      for (int p = 0; p < npart; ++p) {
        hi(PGID,p) = st.gid[p];
        hi(PTAG,p) = st.tag[p];
        hr(IPM,p)  = st.mass[p];     // per-particle proper rest mass (D6/D10 fix)
        hr(IPEN,p) = 0.0;
        hr(IPX,p)  = st.x[p];  hr(IPVX,p) = 0.0;   // u_i = 0 (momentarily at rest)
        hr(IPY,p)  = st.y[p];  hr(IPVY,p) = 0.0;
        hr(IPZ,p)  = st.z[p];  hr(IPVZ,p) = 0.0;
      }
      Kokkos::deep_copy(ppart->prtcl_rdata, hr);
      Kokkos::deep_copy(ppart->prtcl_idata, hi);
      ppart->nprtcl_thispack = npart;
      pmy_mesh_->nprtcl_thisrank = npart;
      pmy_mesh_->nprtcl_eachrank[global_variable::my_rank] = npart;
#if MPI_PARALLEL_ENABLED
      MPI_Allgather(&npart, 1, MPI_INT, pmy_mesh_->nprtcl_eachrank, 1, MPI_INT,
                    MPI_COMM_WORLD);
#endif
      pmy_mesh_->nprtcl_total = 0;
      for (int nr = 0; nr < global_variable::nranks; ++nr) {
        pmy_mesh_->nprtcl_total += pmy_mesh_->nprtcl_eachrank[nr];
      }
      if (global_variable::my_rank == 0) {
        Real Mp = 0.0;
        for (int p = 0; p < npart; ++p) { Mp += st.mass[p]; }
        std::cout << "nr_pic_os: placed " << pmy_mesh_->nprtcl_total << " dust particles "
                  << "(lattice n=" << nlat << ", r0_iso=" << r0 << ", rho0=" << rho0
                  << ", sum m_p=" << Mp << " = proper rest mass)" << std::endl;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "nr_pic_os requires <particles> init = pgen or file."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }  // if (!restart)

  // seed the GR-pusher previous-step snapshots (fresh start AND restart; not in the
  // restart file). On restart the copy carries unfilled ghosts -- Driver::Initialize
  // refreshes both after the ghost exchange (the authoritative restart-path seed).
  if (pmbp->ppart->pusher == ParticlesPusher::gr_boris) {
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);
  }

  if (global_variable::my_rank == 0 && !restart) {
    std::cout << "Oppenheimer-Snyder initialized (M=" << M << ", R0=" << R0 << " = "
              << R0_over_M << " M, alpha0=" << (precollapsed ? "psi^-2" : "1") << ")."
              << std::endl;
  }
  return;
}
