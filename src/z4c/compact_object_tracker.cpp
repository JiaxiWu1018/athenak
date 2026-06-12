//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <assert.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "compact_object_tracker.hpp"

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/lagrange_interpolator.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"

//----------------------------------------------------------------------------------------
CompactObjectTracker::CompactObjectTracker(Mesh *pmesh, ParameterInput *pin, int n):
              owns_compact_object{false}, pos{NAN, NAN, NAN}, vel{NAN, NAN, NAN},
              active_tracker{true}, pmesh{pmesh}, out_every{1} {
  std::string nstr = std::to_string(n);
  std::string ofname = pin->GetString("job", "basename") + ".";
  ofname += pin->GetOrAddString("z4c", "filename", "co_");
  ofname += nstr + ".txt";

  std::string cotype = pin->GetString("z4c", "co_" + nstr + "_type");
  if (cotype == "BH" || cotype == "BlackHole") {
    type = BlackHole;
  } else if (cotype == "NS" || cotype == "NeutronStar") {
    type = NeutronStar;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl;
    std::cout << "Unknown compact object type: " << cotype << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string trmode = pin->GetOrAddString("z4c", "tracker_mode", "ode");
  if (trmode == "ode") {
    mode = ODE;
  } else if (trmode == "walk") {
    mode = Walk;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl;
    std::cout << "Unknown tracker mode: " << trmode << std::endl;
    std::exit(EXIT_FAILURE);
  }

  pos[0] = pin->GetOrAddReal("z4c", "co_" + nstr + "_x", 0.0);
  pos[1] = pin->GetOrAddReal("z4c", "co_" + nstr + "_y", 0.0);
  pos[2] = pin->GetOrAddReal("z4c", "co_" + nstr + "_z", 0.0);
  mass = pin->GetOrAddReal("z4c", "co_" + nstr + "_M", 1.0);

  reflevel = pin->GetOrAddInteger("z4c", "co_" + nstr + "_reflevel", -1);
  radius = pin->GetOrAddReal("z4c", "co_" + nstr + "_radius", 0.0);

  out_every = pin->GetOrAddInteger("z4c", "co_" + nstr + "_out_every", 1);

  if (0 == global_variable::my_rank) {
    ofile.open(ofname.c_str());

    if (type == BlackHole) {
      ofile << "# Black Hole";
    } else {
      ofile << "# Neutron Star";
    }
    ofile << std::endl;

    ofile << "# 1:iter 2:time 3:x 4:y 5:z 6:vx 7:vy 8:vz\n";
    ofile << std::flush;
    ofile << std::setprecision(19);
  }
}

//----------------------------------------------------------------------------------------
CompactObjectTracker::~CompactObjectTracker() { }

//----------------------------------------------------------------------------------------
void CompactObjectTracker::InterpolateVelocity(MeshBlockPack *pmbp) {
  auto &padm = pmbp->padm;
  auto &pmhd = pmbp->pmhd;
  auto &pz4c = pmbp->pz4c;
  auto *S    = new LagrangeInterpolator(pmbp, pos);

  if (S->point_exist) {
    owns_compact_object = true;

    Real betax = S->Interpolate(pz4c->u0, pz4c->I_Z4C_BETAX);
    Real betay = S->Interpolate(pz4c->u0, pz4c->I_Z4C_BETAY);
    Real betaz = S->Interpolate(pz4c->u0, pz4c->I_Z4C_BETAZ);
    vel[0] = - betax;
    vel[1] = - betay;
    vel[2] = - betaz;
    if (type == NeutronStar) {
      Real alp = S->Interpolate(pz4c->u0, pz4c->I_Z4C_ALPHA);

      Real zx = S->Interpolate(pmhd->w0, IVX);
      Real zy = S->Interpolate(pmhd->w0, IVY);
      Real zz = S->Interpolate(pmhd->w0, IVZ);

      Real gxx = S->Interpolate(padm->u_adm, padm->I_ADM_GXX);
      Real gxy = S->Interpolate(padm->u_adm, padm->I_ADM_GXY);
      Real gxz = S->Interpolate(padm->u_adm, padm->I_ADM_GXZ);
      Real gyy = S->Interpolate(padm->u_adm, padm->I_ADM_GYY);
      Real gyz = S->Interpolate(padm->u_adm, padm->I_ADM_GYZ);
      Real gzz = S->Interpolate(padm->u_adm, padm->I_ADM_GZZ);

      Real z_x = gxx*zx + gxy*zy + gxz*zz;
      Real z_y = gxy*zx + gyy*zy + gyz*zz;
      Real z_z = gxz*zx + gyz*zy + gzz*zz;
      Real W = std::sqrt(z_x*zx + z_y*zy + z_z*zz + 1);

      vel[0] += alp*zx/W;
      vel[1] += alp*zy/W;
      vel[2] += alp*zz/W;
    }
  } else {
    owns_compact_object = false;
  }

  delete S;
}

//----------------------------------------------------------------------------------------
void CompactObjectTracker::EvolveTracker(MeshBlockPack *pmbp) {
  if (owns_compact_object) {
    if (mode == ODE) {
      // Debugging output only (per-step tracker pos/vel); commented out for production.
      // Kokkos::printf("co on rank %d has position %.5g %.5g %.5g and
      //                velocity %.5g %.5g %.5g\n", global_variable::my_rank,
      //                pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
      for (int a = 0; a < NDIM; ++a) {
        pos[a] += pmesh->dt * vel[a];
      }
    } else {
      auto &padm = pmbp->padm;
      auto &size = pmbp->pmb->mb_size;
      auto &indcs = pmbp->pmesh->mb_indcs;

      int nmb1 = pmbp->nmb_thispack;
      for (int m = 0; m < nmb1; ++m) {
        // extract MeshBlock bounds
        Real x1min = size.h_view(m).x1min;
        Real x1max = size.h_view(m).x1max;
        Real x2min = size.h_view(m).x2min;
        Real x2max = size.h_view(m).x2max;
        Real x3min = size.h_view(m).x3min;
        Real x3max = size.h_view(m).x3max;

        // extract MeshBlock grid cell spacings
        Real dx1 = size.h_view(m).dx1;
        Real dx2 = size.h_view(m).dx2;
        Real dx3 = size.h_view(m).dx3;

        // check if the compact object is in the current mesh block
        if ((pos[0] >= x1min && pos[0] < x1max) &&
            (pos[1] >= x2min && pos[1] < x2max) &&
            (pos[2] >= x3min && pos[2] < x3max)) {
          int ic = std::round((pos[0] - (x1min + 0.5 * dx1)) / dx1);
          int jc = std::round((pos[1] - (x2min + 0.5 * dx2)) / dx2);
          int kc = std::round((pos[2] - (x3min + 0.5 * dx3)) / dx3);

          DualArray3D<Real> alp("lapse", 3, 3, 3);
          auto& adm = padm->adm;
          par_for("Copy lapse neighborhood", DevExeSpace(), 0, 2, 0, 2, 0, 2,
          KOKKOS_LAMBDA(const int k, const int j, const int i){
            alp.d_view(k,j,i) = adm.alpha(m,kc + indcs.ks + k - 1,
                                            jc + indcs.js + j - 1,
                                            ic + indcs.is + i - 1);
          });

          alp.template modify<DevMemSpace>();
          alp.template sync<typename DualArray3D<Real>::host_mirror_space>();

          Real alp_min = std::numeric_limits<Real>::max();
          for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j) {
              for (int i = 0; i < 3; ++i) {
                if (alp.h_view(k, j, i) < alp_min) {
                  alp_min = alp.h_view(k, j, i);
                  pos[0] = CellCenterX(ic + (i - 1), indcs.nx1,
                                       x1min, x1max);
                  pos[1] = CellCenterX(jc + (j - 1), indcs.nx2,
                                       x2min, x2max);
                  pos[2] = CellCenterX(kc + (k - 1), indcs.nx3,
                                       x3min, x3max);
                }
              }
            }
          }

          break;
        }
      }
    }
#if !(MPI_PARALLEL_ENABLED)
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    std::cout << "couldn't find the compact object!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#else
  }
  Real buf[2 * NDIM + 1] = {0., 0., 0., 0., 0., 0., 0.};
  if (owns_compact_object) {
    buf[0] = pos[0];
    buf[1] = pos[1];
    buf[2] = pos[2];
    buf[3] = vel[0];
    buf[4] = vel[1];
    buf[5] = vel[2];
    buf[6] = 1.0;
  }
  MPI_Allreduce(
    MPI_IN_PLACE, buf, 2 * NDIM + 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (buf[6] < 0.5) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    std::cout << "The compact object has left the grid" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pos[0] = buf[0] / buf[6];
  pos[1] = buf[1] / buf[6];
  pos[2] = buf[2] / buf[6];
  vel[0] = buf[3] / buf[6];
  vel[1] = buf[4] / buf[6];
  vel[2] = buf[5] / buf[6];
#endif // MPI_PARALLEL_ENABLED

  // After the compact object has moved it might have changed ownership
  owns_compact_object = false;
}

//----------------------------------------------------------------------------------------
void CompactObjectTracker::WriteTracker() {
  if (0 == global_variable::my_rank && 0 == pmesh->ncycle % out_every) {
    ofile << pmesh->ncycle << " "
          << pmesh->time << " "
          << pos[0] << " "
          << pos[1] << " "
          << pos[2] << " "
          << vel[0] << " "
          << vel[1] << " "
          << vel[2] << std::endl << std::flush;
  }
}

namespace z4c {
//----------------------------------------------------------------------------------------
void Z4c::BHMergeDetector() {
  while (true) {
    // Get all activate black hole trackers
    std::vector<int> bh_ids;
    bh_ids.reserve(ptracker.size());
    for (int i = 0; i < static_cast<int>(ptracker.size()); ++i) {
      auto &pt = ptracker[i];
      if (! (pt->is_active() && pt->is_BH())) continue;
      bh_ids.push_back(i);
    }
    if (bh_ids.size() < 2) return;

    // Find the nearest pair of black holes that are within merger threshold
    bool found = false;
    int best_i = -1, best_j = -1;
    Real best_d2 = std::numeric_limits<Real>::max();
    const Real d2_merge = opt.d_merge * opt.d_merge;
    for (int a = 0; a < static_cast<int>(bh_ids.size()); ++a) {
      const int i = bh_ids[a];
      const Real xi = ptracker[i]->GetPos(0);
      const Real yi = ptracker[i]->GetPos(1);
      const Real zi = ptracker[i]->GetPos(2);
      for (int b = a + 1; b < static_cast<int>(bh_ids.size()); ++b) {
        const int j = bh_ids[b];
        const Real xj = ptracker[j]->GetPos(0);
        const Real yj = ptracker[j]->GetPos(1);
        const Real zj = ptracker[j]->GetPos(2);
        const Real dx = xi - xj;
        const Real dy = yi - yj;
        const Real dz = zi - zj;
        const Real d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < d2_merge) {
          if (d2 < best_d2) {
            best_d2 = d2;
            best_i = i;
            best_j = j;
            found = true;
          }
        }
      }
    }

    if (!found) return;

    // Deactivate one of them and modify the mass of the other
    const Real mi = ptracker[best_i]->GetMass();
    const Real mj = ptracker[best_j]->GetMass();
    int survivor = -1; int victim = -1;
    if (std::abs(mi - mj) < 1e-6) {
      survivor = (best_i < best_j) ? best_i : best_j;
      victim = (best_i < best_j) ? best_j : best_i;
    } else {
      survivor = (mi > mj) ? best_i : best_j;
      victim = (mi > mj) ? best_j : best_i;
    }
    // std::cout << "two masses " << mi << " " << mj << std::endl;
    // debug, off for production
    ptracker[survivor]->SetMass(mi + mj);
    // std::cout << "After set mass " << ptracker[survivor]->GetMass() << std::endl;
    // debug
    ptracker[victim]->SetActive(false);
    // Debugging output only (stdout); commented out for production. The merger is
    // still recorded to the tracker file via WriteMergedInto() below.
    /* if (global_variable::my_rank == 0) {
      std::cout << "BHMergeDetector: merged tracker " << victim
                << " into " << survivor
                << " at iter=" << pmy_pack->pmesh->ncycle
                << " time=" << pmy_pack->pmesh->time
                << " survivor location=" << ptracker[survivor]->GetPos(0)
                << " " << ptracker[survivor]->GetPos(1)
                << " " << ptracker[survivor]->GetPos(2)
                << " victim location=" << ptracker[victim]->GetPos(0)
                << " " << ptracker[victim]->GetPos(1)
                << " " << ptracker[victim]->GetPos(2)
                << std::endl;
    } */
    // Record merger info into tracker file
    const Real sep = std::sqrt(best_d2);
    ptracker[victim]->WriteMergedInto(survivor, pmy_pack->pmesh->ncycle,
                                      pmy_pack->pmesh->time, sep);
  }
}
} // end namespace z4c
