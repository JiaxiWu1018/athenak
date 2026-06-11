//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file part_crossing.cpp
//! \brief Problem generator for the particle MeshBlock-migration test (NRPIC Stage 3).
//! Requires <particles> init=pgen and the drift pusher; designed to run with
//! <particles> debug >= 1 so the post-migration validation is the pass/fail criterion.
//!
//! mode=targeted (default): for every LOCAL MeshBlock and every potential neighbor slot
//! (ix,iy,iz,f1,f2) -- 6 faces x 4 subslots + 12 edges x 2 + 8 corners = 56 in 3D (the
//! exact nghbr-array enumeration of mesh/nghbr_index.hpp) -- place one particle a small
//! distance delta inside that face/edge/corner, at the subslot quarter-points in the
//! transverse direction(s), moving outward along the offset direction fast enough to
//! cross in ONE cycle. The particle tag is
//!     tag = gid * 56 + NeighborIndex(ix,iy,iz,f1,f2)
//! which is globally unique, decomposition-invariant, and DECODABLE: a failing tag
//! identifies the source block, crossing direction, and subslot immediately. Running with
//! time/nlim=1 tests every crossing class in the mesh in a single step; with larger nlim
//! the particles keep streaming block-to-block (a natural soak test).
//!
//! mode=lattice: npx x npy x npz lattice of particles with velocities cycling all 26
//! neighbor directions (plus a rest particle every 27th), tag = lattice index. A
//! statistical multi-cycle soak complement to the targeted mode.
//!
//! <problem> options: mode, vmax (speed, default 1.0), select_gid, select_slot (targeted
//! mode filters: place only particles matching that gid and/or slot, for isolating a
//! single failing crossing), npx/npy/npz (lattice mode).

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "particles/particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

// staged particle data accumulated on the host before the device fill
struct PrtclStage {
  std::vector<Real> x, y, z, vx, vy, vz;
  std::vector<int> gid, tag;
  void Add(Real x_, Real y_, Real z_, Real vx_, Real vy_, Real vz_, int gid_, int tag_) {
    x.push_back(x_); y.push_back(y_); z.push_back(z_);
    vx.push_back(vx_); vy.push_back(vy_); vz.push_back(vz_);
    gid.push_back(gid_); tag.push_back(tag_);
  }
};

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief sets up the targeted (or lattice) particle crossing test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_crossing requires a <particles> block in the input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  // exhaustive destination-search audit (independent of the particle init mode)
  if (pin->GetOrAddBoolean("problem","audit",false)) {
    pmbp->ppart->AuditDestinationSearch();
  }

  std::string init = pin->GetOrAddString("particles","init","ppc");
  if (init.compare("file") == 0) {
    return;   // particles already loaded by the HDF5 reader (init=file cross-check path)
  }
  if (init.compare("pgen") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "part_crossing requires <particles> init = pgen (or file)" << std::endl;
    exit(EXIT_FAILURE);
  }

  particles::Particles *ppart = pmbp->ppart;
  Mesh *pm = pmy_mesh_;
  bool three_d = pm->three_d;
  auto &msize = pm->mesh_size;
  auto &mbsize = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  int gids = pmbp->gids;

  std::string mode = pin->GetOrAddString("problem","mode","targeted");
  Real vmax = pin->GetOrAddReal("problem","vmax",1.0);
  PrtclStage st;

  if (mode.compare("targeted") == 0) {
    int select_gid  = pin->GetOrAddInteger("problem","select_gid",-1);
    int select_slot = pin->GetOrAddInteger("problem","select_slot",-1);
    // inward starting offset as a fraction of the block's min cell; 0 places particles
    // EXACTLY on the boundary (degenerate ownership test: with the half-open [min,max)
    // convention a particle on the max edge already belongs to the neighbor, so the
    // first migration must relabel it before the first validation)
    Real delta_frac = pin->GetOrAddReal("problem","delta_frac",0.1);

    // crossing feasibility: the per-component step is vmax*cfl*dx_min/sqrt(3) (the particle
    // timestep is the light-crossing dt = cfl * smallest cell in the mesh) and must exceed
    // the largest inward offset delta = delta_frac * (min cell of the particle's block)
    Real cfl = pin->GetReal("time","cfl_number");
    Real dxmin = std::numeric_limits<Real>::max(), delta_max = 0.0;
    for (int m=0; m<nmb; ++m) {
      Real d = mbsize.h_view(m).dx1;
      d = std::fmin(d, mbsize.h_view(m).dx2);
      if (three_d) {d = std::fmin(d, mbsize.h_view(m).dx3);}
      dxmin = std::fmin(dxmin, d);
      delta_max = std::fmax(delta_max, delta_frac*d);
    }
    Real step_min = vmax*cfl*dxmin/sqrt(3.0);
    if (step_min <= delta_max) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "targeted crossing infeasible: per-component step " << step_min
                << " <= max inward offset " << delta_max
                << " (raise vmax/cfl_number or lower delta_frac)" << std::endl;
      exit(EXIT_FAILURE);
    }

    // one particle per (block, neighbor-slot): offsets (ix,iy,iz), subslots (f1,f2)
    for (int m=0; m<nmb; ++m) {
      int gid = gids + m;
      if (select_gid >= 0 && gid != select_gid) {continue;}
      Real bmin[3] = {mbsize.h_view(m).x1min, mbsize.h_view(m).x2min,
                      mbsize.h_view(m).x3min};
      Real bmax[3] = {mbsize.h_view(m).x1max, mbsize.h_view(m).x2max,
                      mbsize.h_view(m).x3max};
      Real d = mbsize.h_view(m).dx1;
      d = std::fmin(d, mbsize.h_view(m).dx2);
      if (three_d) {d = std::fmin(d, mbsize.h_view(m).dx3);}
      Real delta = delta_frac*d;

      for (int iz=-1; iz<=1; ++iz) {
        if (!three_d && iz != 0) {continue;}
        for (int iy=-1; iy<=1; ++iy) {
          for (int ix=-1; ix<=1; ++ix) {
            int off[3] = {ix, iy, iz};
            int ndir = abs(ix) + abs(iy) + abs(iz);
            if (ndir == 0) {continue;}
            // subslot ranges: one f-bit per zero-offset dimension, in ascending dimension
            // order (matching SetNeighbors/NeighborIndex); a z f-bit is fixed to 0 in 2D
            int fmaxs[2] = {1, 1};
            int nf = 0;
            for (int dim=0; dim<3; ++dim) {
              if (off[dim] == 0) {fmaxs[nf++] = (dim == 2 && !three_d) ? 1 : 2;}
            }
            for (int f2=0; f2<fmaxs[1]; ++f2) {
              for (int f1=0; f1<fmaxs[0]; ++f1) {
                int slot = NeighborIndex(ix,iy,iz,f1,f2);
                if (select_slot >= 0 && slot != select_slot) {continue;}
                // position: delta inside each crossed boundary; subslot quarter-points
                // in the uncrossed (transverse) dimensions
                int fbits[2] = {f1, f2};
                Real pos[3];
                int kf = 0;
                for (int dim=0; dim<3; ++dim) {
                  Real len = bmax[dim] - bmin[dim];
                  if (off[dim] == 0) {
                    pos[dim] = bmin[dim] + (0.25 + 0.5*fbits[kf++])*len;
                  } else {
                    pos[dim] = (off[dim] > 0) ? bmax[dim] - delta : bmin[dim] + delta;
                  }
                }
                Real vfac = vmax/sqrt(static_cast<Real>(ndir));
                st.Add(pos[0], pos[1], pos[2], vfac*ix, vfac*iy, vfac*iz,
                       gid, gid*56 + slot);
              }
            }
          }
        }
      }
    }
  } else if (mode.compare("lattice") == 0) {
    // lattice of particles with velocities cycling all neighbor directions (+ rest)
    int npx = pin->GetOrAddInteger("problem","npx",8);
    int npy = pin->GetOrAddInteger("problem","npy",8);
    int npz = three_d ? pin->GetOrAddInteger("problem","npz",8) : 1;
    // direction table: all (a,b,c) in {-1,0,1}^3 minus rest, lexicographic
    std::vector<std::array<int,3>> dirs;
    for (int c=(three_d ? -1 : 0); c<=(three_d ? 1 : 0); ++c) {
      for (int b=-1; b<=1; ++b) {
        for (int a=-1; a<=1; ++a) {
          if (a == 0 && b == 0 && c == 0) {continue;}
          dirs.push_back({a, b, c});
        }
      }
    }
    int ncyc = static_cast<int>(dirs.size()) + 1;   // +1 = a rest particle per cycle
    for (int k=0; k<npz; ++k) {
      for (int j=0; j<npy; ++j) {
        for (int i=0; i<npx; ++i) {
          Real x = msize.x1min + (i + 0.5)*(msize.x1max - msize.x1min)/npx;
          Real y = msize.x2min + (j + 0.5)*(msize.x2max - msize.x2min)/npy;
          Real z = three_d ? msize.x3min + (k + 0.5)*(msize.x3max - msize.x3min)/npz : 0.0;
          int m = ppart->FindContainingMeshBlock(x, y, z);
          if (m < 0) {continue;}   // belongs to another rank
          int tag = i + npx*(j + npy*k);
          Real vx = 0.0, vy = 0.0, vz = 0.0;
          int idir = tag % ncyc;
          if (idir < static_cast<int>(dirs.size())) {
            auto &dd = dirs[idir];
            int nd = abs(dd[0]) + abs(dd[1]) + abs(dd[2]);
            Real vfac = vmax/sqrt(static_cast<Real>(nd));
            vx = vfac*dd[0]; vy = vfac*dd[1]; vz = vfac*dd[2];
          }
          st.Add(x, y, z, vx, vy, vz, gids + m, tag);
        }
      }
    }
  } else if (mode.compare("random") == 0) {
    // fuzz mode: nrand particles per local block at random in-block positions, with a
    // finite probability per coordinate of snapping EXACTLY onto a block boundary or
    // midline (degenerate ownership/subslot positions), and isotropic random velocity
    // directions at speed vmax. Deterministic per seed; each block re-seeds with
    // (seed + gid) so the draw is independent of iteration order and (later) of the
    // MPI decomposition. tag = gid*nrand + i: globally unique and decodable.
    int nrand = pin->GetOrAddInteger("problem","nrand",64);
    int seed  = pin->GetOrAddInteger("problem","seed",1);
    for (int m=0; m<nmb; ++m) {
      int gid = gids + m;
      std::mt19937 gen(static_cast<unsigned int>(seed) + 7919u*gid);
      std::uniform_real_distribution<Real> u01(0.0, 1.0);
      std::normal_distribution<Real> gauss(0.0, 1.0);
      Real bmin[3] = {mbsize.h_view(m).x1min, mbsize.h_view(m).x2min,
                      mbsize.h_view(m).x3min};
      Real bmax[3] = {mbsize.h_view(m).x1max, mbsize.h_view(m).x2max,
                      mbsize.h_view(m).x3max};
      for (int i=0; i<nrand; ++i) {
        Real pos[3];
        for (int dim=0; dim<3; ++dim) {
          Real len = bmax[dim] - bmin[dim];
          pos[dim] = bmin[dim] + u01(gen)*len;
          Real s = u01(gen);
          if (s < 0.05)      {pos[dim] = bmin[dim];}             // exactly on min edge
          else if (s < 0.10) {pos[dim] = bmax[dim];}             // exactly on max edge
          else if (s < 0.15) {pos[dim] = bmin[dim] + 0.5*len;}   // exactly on midline
        }
        Real vx = gauss(gen), vy = gauss(gen), vz = three_d ? gauss(gen) : 0.0;
        Real vn = sqrt(vx*vx + vy*vy + vz*vz);
        if (vn < 1.0e-12) {vx = 1.0; vy = 0.0; vz = 0.0; vn = 1.0;}
        st.Add(pos[0], pos[1], pos[2], vmax*vx/vn, vmax*vy/vn, vmax*vz/vn,
               gid, gid*nrand + i);
      }
    }
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem> mode = '" << mode
              << "' not recognized (use targeted|lattice|random)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // fill the particle arrays from the staged host data
  int npart = static_cast<int>(st.x.size());
  Kokkos::realloc(ppart->prtcl_rdata, ppart->nrdata, npart);
  Kokkos::realloc(ppart->prtcl_idata, ppart->nidata, npart);
  auto hr = Kokkos::create_mirror_view(ppart->prtcl_rdata);
  auto hi = Kokkos::create_mirror_view(ppart->prtcl_idata);
  for (int p=0; p<npart; ++p) {
    hi(PGID,p) = st.gid[p];
    hi(PTAG,p) = st.tag[p];
    hr(IPM,p)  = ppart->mass;
    hr(IPEN,p) = 0.0;
    hr(IPX,p)  = st.x[p];
    hr(IPVX,p) = st.vx[p];
    hr(IPY,p)  = st.y[p];
    hr(IPVY,p) = st.vy[p];
    if (three_d) {
      hr(IPZ,p)  = st.z[p];
      hr(IPVZ,p) = st.vz[p];
    }
  }
  Kokkos::deep_copy(ppart->prtcl_rdata, hr);
  Kokkos::deep_copy(ppart->prtcl_idata, hi);
  ppart->nprtcl_thispack = npart;

  // refresh the Mesh particle counts (AddCoordinatesAndPhysics counted zero particles
  // before this pgen ran; mirror its logic, cf. mesh.cpp)
  pm->nprtcl_thisrank = npart;
  pm->nprtcl_eachrank[global_variable::my_rank] = npart;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&npart, 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
#endif
  pm->nprtcl_total = 0;
  for (int n=0; n<global_variable::nranks; ++n) {
    pm->nprtcl_total += pm->nprtcl_eachrank[n];
  }

  if (global_variable::my_rank == 0) {
    std::cout << "part_crossing: mode=" << mode << " placed " << pm->nprtcl_total
              << " particles (vmax=" << vmax << ")" << std::endl;
  }
  // block map for the analysis script: gid, owning rank, level, logical-location
  // parity, bbox (parity = subslot at which SetNeighbors stores this block's coarser
  // neighbors). Printed rank by rank behind barriers so the lines come out grouped --
  // best effort only, since mpirun merges the streams asynchronously: parsers should
  // also key on the `mpirun --tag-output` rank prefixes for hard attribution.
  auto &mblev = pmbp->pmb->mb_lev;
  for (int r=0; r<global_variable::nranks; ++r) {
#if MPI_PARALLEL_ENABLED
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (r != global_variable::my_rank) {continue;}
    for (int m=0; m<nmb; ++m) {
      int gid = gids + m;
      auto &lloc = pm->lloc_eachmb[gid];
      std::cout << "[part_crossing] block gid=" << gid << " rank=" << r
                << " level=" << mblev.h_view(m)
                << " parity=(" << (lloc.lx1 & 1) << "," << (lloc.lx2 & 1) << ","
                << (lloc.lx3 & 1) << ")"
                << " x1=[" << mbsize.h_view(m).x1min << "," << mbsize.h_view(m).x1max
                << ") x2=[" << mbsize.h_view(m).x2min << "," << mbsize.h_view(m).x2max
                << ") x3=[" << mbsize.h_view(m).x3min << "," << mbsize.h_view(m).x3max
                << ")" << std::endl << std::flush;
    }
  }
  return;
}
