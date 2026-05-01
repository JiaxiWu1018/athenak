//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "z4c/z4c_amr.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"

#define SQ(X) ((X)*(X))

namespace z4c {

// set some parameters
Z4c_AMR::Z4c_AMR(ParameterInput *pin) {
  std::string ref_method = pin->GetOrAddString("z4c_amr", "method", "trivial");
  if (ref_method == "trivial") {
    method = Trivial;
  } else if (ref_method == "tracker") {
    method = Tracker;
  } else if (ref_method == "chi") {
    method = Chi;
    chi_thresh = pin->GetOrAddReal("z4c_amr", "chi_min", 0.2);
  } else if (ref_method == "dchi") {
    method = dChi;
    dchi_thresh = pin->GetOrAddReal("z4c_amr", "dchi_max", 0.01);
  } else if (ref_method == "loehner") {
    method = Loehner;
    loehner_threshold = pin->GetOrAddReal("z4c_amr", "loehner_threshold", 0.2);
    std::string ref_var = pin->GetOrAddString("z4c_amr", "loehner_variable", "alp_psi7");
    if (ref_var == "alp_psi7") {
      loehner_var = alp_psi7;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl;
      std::cout << "Unknown Loehner refinement variable: " << ref_var << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl;
    std::cout << "Unknown refinement strategy: " << ref_method << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (int nr = 0; nr < 16; ++nr) {
    std::string name = "radius_" + std::to_string(nr) + "_rad";
    if (pin->DoesParameterExist("z4c_amr", name)) {
      radius.push_back(pin->GetReal("z4c_amr", name));
      reflevel.push_back(pin->GetOrAddInteger(
          "z4c_amr", "radius_" + std::to_string(nr) + "_reflevel", -1));
    } else {
      break;
    }
  }
}

// 1: refines, -1: de-refines, 0: does nothing
void Z4c_AMR::Refine(MeshBlockPack *pmy_pack) {
  if (method == Tracker) {
    RefineTracker(pmy_pack);
  } else if (method == Chi) {
    RefineChiMin(pmy_pack);
  } else if (method == dChi) {
    RefineDchiMax(pmy_pack);
  } else if (method == Loehner) {
    RefineLoehner(pmy_pack);
  }
  RefineRadii(pmy_pack);
}

// refine region within a certain distance from each compact object
// using exact minimum distance via AABB clamping, which correctly handles
// all cases: tracker nearest to a face, edge, or corner of the block.
void Z4c_AMR::RefineTracker(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  std::vector<int> flag;
  flag.reserve(pmbp->pz4c->ptracker.size());

  for (int m = 0; m < nmb; ++m) {
    // current refinement level
    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    flag.clear();
    for (auto &pt : pmbp->pz4c->ptracker) {
      // clamp tracker position to box bounds: closest point on the box
      Real cx = fmax(x1min, fmin(pt->GetPos(0), x1max));
      Real cy = fmax(x2min, fmin(pt->GetPos(1), x2max));
      Real cz = fmax(x3min, fmin(pt->GetPos(2), x3max));

      Real dmin2 = SQ(pt->GetPos(0) - cx) \
                   + SQ(pt->GetPos(1) - cy) \
                   + SQ(pt->GetPos(2) - cz);

      // safety net for radius = 0: dmin2 = 0 inside the block but 0 < SQ(0) is false
      bool iscontained =
        (pt->GetPos(0) >= x1min && pt->GetPos(0) <= x1max) &&
        (pt->GetPos(1) >= x2min && pt->GetPos(1) <= x2max) &&
        (pt->GetPos(2) >= x3min && pt->GetPos(2) <= x3max);

      if (dmin2 < SQ(pt->GetRadius()) || iscontained) {
        if (pt->GetReflevel() < 0 || level < pt->GetReflevel()) {
          flag.push_back(1);
        } else if (level == pt->GetReflevel()) {
          flag.push_back(0);
        } else {
          flag.push_back(-1);
        }
      } else {
        flag.push_back(-1);
      }
    }
    refine_flag.h_view(m + mbs) = *std::max_element(flag.begin(), flag.end());
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

// refine based on min{chi}
void Z4c_AMR::RefineChiMin(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &indcs       = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->pz4c->u0;
  int I_Z4C_CHI  = pmbp->pz4c->I_Z4C_CHI;
  // note: we need this to prevent capture by this in the lambda expr.
  auto chi_thresh = this->chi_thresh;

  par_for_outer(
    "Z4c_AMR::ChiMin", DevExeSpace(), 0, 0, 0, (nmb - 1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_dmin;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &dmin) {
          int k = (idx) / nji;
          int j = (idx - k * nji) / nx1;
          int i = (idx - k * nji - j * nx1) + is;
          j += js;
          k += ks;
          dmin = fmin(u0(m, I_Z4C_CHI, k, j, i), dmin);
        },
        Kokkos::Min<Real>(team_dmin));

      if (team_dmin < chi_thresh) {
        refine_flag.d_view(m + mbs) = 1;
      }
      if (team_dmin > 1.25 * chi_thresh) {
        refine_flag.d_view(m + mbs) = -1;
      }
    });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

// refine based on max{dchi}
void Z4c_AMR::RefineDchiMax(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &indcs       = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->pz4c->u0;
  int I_Z4C_CHI  = pmbp->pz4c->I_Z4C_CHI;
  // note: we need this to prevent capture by this in the lambda expr.
  auto dchi_thresh = this->dchi_thresh;

  par_for_outer(
    "Z4c_AMR::ChiMin", DevExeSpace(), 0, 0, 0, (nmb - 1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_dmax;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &dmax) {
          int k = (idx) / nji;
          int j = (idx - k * nji) / nx1;
          int i = (idx - k * nji - j * nx1) + is;
          j += js;
          k += ks;
          Real d2 = SQR(u0(m,I_Z4C_CHI,k,j,i+1) - u0(m,I_Z4C_CHI,k,j,i-1));
          d2 += SQR(u0(m,I_Z4C_CHI,k,j+1,i) - u0(m,I_Z4C_CHI,k,j-1,i));
          d2 += SQR(u0(m,I_Z4C_CHI,k+1,j,i) - u0(m,I_Z4C_CHI,k-1,j,i));
          dmax = fmax((sqrt(d2)), dmax);
        },
        Kokkos::Max<Real>(team_dmax));

      if (team_dmax > dchi_thresh) {
        refine_flag.d_view(m + mbs) = 1;
      }
      if (team_dmax < 0.5 * dchi_thresh) {
        refine_flag.d_view(m + mbs) = -1;
      }
    });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

void Z4c_AMR::RefineLoehner(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &indcs       = pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1; int nx2 = indcs.nx2; int nx3 = indcs.nx3;
  int ng = indcs.ng;
  int ncell1 = nx1 + 2 * ng; int ncell2 = nx2 + 2 * ng; int ncell3 = nx3 + 2 * ng;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->pz4c->u0;

  auto &loehner_threshold = this->loehner_threshold;
  Real eps = 0.01;

  // compute refinement variable
  DvceArray4D<Real> var;
  Kokkos::realloc(var, nmb, ncell3, ncell2, ncell1);
  if (loehner_var == alp_psi7) {
    par_for("Calc_refine_var", DevExeSpace(), 0,nmb-1,ks-2,ke+2,js-2,je+2,is-2,ie+2,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i){
      Real alp = u0(m, pmbp->pz4c->I_Z4C_ALPHA, k, j, i);
      Real chi = u0(m, pmbp->pz4c->I_Z4C_CHI, k, j, i);
      var(m,k,j,i) = alp * std::pow(chi, -1.75);
    });
  }

  // compute first derivatives
  DvceArray5D<Real> dvar, abs_dvar;
  Kokkos::realloc(dvar, nmb, 3, ncell3, ncell2, ncell1);
  Kokkos::realloc(abs_dvar, nmb, 3, ncell3, ncell2, ncell1);
  par_for("Calc_dvar", DevExeSpace(), 0,nmb-1,ks-1,ke+1,js-1,je+1,is-1,ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i){
    // dvar/dx1
    dvar(m,0,k,j,i) = var(m,k,j,i+1) - var(m,k,j,i-1);
    abs_dvar(m,0,k,j,i) = fabs(var(m,k,j,i+1)) + fabs(var(m,k,j,i-1));
    // dvar/dx2
    dvar(m,1,k,j,i) = var(m,k,j+1,i) - var(m,k,j-1,i);
    abs_dvar(m,1,k,j,i) = fabs(var(m,k,j+1,i)) + fabs(var(m,k,j-1,i));
    // dvar/dx3
    dvar(m,2,k,j,i) = var(m,k+1,j,i) - var(m,k-1,j,i);
    abs_dvar(m,2,k,j,i) = fabs(var(m,k+1,j,i)) + fabs(var(m,k-1,j,i));
  });

  par_for_outer(
  "Z4c_AMR::Loehner", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_dmax;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &dmax) {
        int k = (idx) / nji;
        int j = (idx - k * nji) / nx1;
        int i = (idx - k * nji - j * nx1) + is;
        j += js;
        k += ks;

        Real ddvar1[3][3], ddvar2[3][3], ddvar3[3][3];
        for (int d = 0; d < 3; ++d) {
          // second derivatives
          ddvar1[0][d] = dvar(m,d,k,j,i+1) - dvar(m,d,k,j,i-1);
          ddvar1[1][d] = dvar(m,d,k,j+1,i) - dvar(m,d,k,j-1,i);
          ddvar1[2][d] = dvar(m,d,k+1,j,i) - dvar(m,d,k-1,j,i);

          ddvar2[0][d] = fabs(dvar(m,d,k,j,i+1)) + fabs(dvar(m,d,k,j,i-1));
          ddvar2[1][d] = fabs(dvar(m,d,k,j+1,i)) + fabs(dvar(m,d,k,j-1,i));
          ddvar2[2][d] = fabs(dvar(m,d,k+1,j,i)) + fabs(dvar(m,d,k-1,j,i));

          ddvar3[0][d] = abs_dvar(m,d,k,j,i+1) + abs_dvar(m,d,k,j,i-1);
          ddvar3[1][d] = abs_dvar(m,d,k,j+1,i) + abs_dvar(m,d,k,j-1,i);
          ddvar3[2][d] = abs_dvar(m,d,k+1,j,i) + abs_dvar(m,d,k-1,j,i);
        }

        Real num = 0.0;
        Real den = 0.0;
        for (int d = 0; d < 3; ++d) {
          for (int dd = 0; dd < 3; ++dd) {
            num += ddvar1[dd][d] * ddvar1[dd][d];
            Real tmp = ddvar2[dd][d] + (eps * ddvar3[dd][d] + 1e-99);
            den += tmp * tmp;
          }
        }
        Real loehner_error = sqrt(num / den);
        dmax = fmax(loehner_error, dmax);
      },
      Kokkos::Max<Real>(team_dmax));

    if (team_dmax > loehner_threshold) {
      refine_flag.d_view(m + mbs) = 1;
    }
    // What is derefinement criterion?
    if (team_dmax < 0.1 * loehner_threshold) {
      refine_flag.d_view(m + mbs) = -1;
    }
  });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

// Enforce some minimum resolution within a certain spherical region
void Z4c_AMR::RefineRadii(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  for (int m = 0; m < nmb; ++m) {
    // current refinement level
    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    Real r2[8] = {
      SQ(x1min) + SQ(x2min) + SQ(x3min),
      SQ(x1max) + SQ(x2min) + SQ(x3min),
      SQ(x1min) + SQ(x2max) + SQ(x3min),
      SQ(x1max) + SQ(x2max) + SQ(x3min),
      SQ(x1min) + SQ(x2min) + SQ(x3max),
      SQ(x1max) + SQ(x2min) + SQ(x3max),
      SQ(x1min) + SQ(x2max) + SQ(x3max),
      SQ(x1max) + SQ(x2max) + SQ(x3max),
    };
    Real rmin2 = *std::min_element(&r2[0], &r2[8]);

    for (int ir = 0; ir < radius.size(); ++ir) {
      if (rmin2 < SQ(radius[ir])) {
        if (level < reflevel[ir]) {
          refine_flag.h_view(m + mbs) = 1;
        } else if (level == reflevel[ir] && refine_flag.h_view(m + mbs) == -1) {
          refine_flag.h_view(m + mbs) = 0;
        }
      }
    }
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

} // namespace z4c
