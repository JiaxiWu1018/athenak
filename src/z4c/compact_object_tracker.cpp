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

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "compact_object_tracker.hpp"

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "particles/particles.hpp"
#include "utils/lagrange_interpolator.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "z4c/fastflow.hpp"

//----------------------------------------------------------------------------------------
CompactObjectTracker::CompactObjectTracker(Mesh *pmesh, ParameterInput *pin, int n):
              owns_compact_object{false}, vel{NAN, NAN, NAN},
              pmesh{pmesh}, out_every{1}, walk_every{1}, pos{NAN, NAN, NAN} {
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
  } else if (trmode == "particle_lapse") {
    mode = ParticleLapse;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl;
    std::cout << "Unknown tracker mode: " << trmode << std::endl;
    std::exit(EXIT_FAILURE);
  }

  pos[0] = pin->GetOrAddReal("z4c", "co_" + nstr + "_x", 0.0);
  pos[1] = pin->GetOrAddReal("z4c", "co_" + nstr + "_y", 0.0);
  pos[2] = pin->GetOrAddReal("z4c", "co_" + nstr + "_z", 0.0);

  tracker_index = n;
  particle_tag_min = pin->GetOrAddInteger("z4c", "co_" + nstr + "_tag_min", -1);
  particle_tag_max = pin->GetOrAddInteger(
      "z4c", "co_" + nstr + "_tag_max", std::numeric_limits<int>::max());
  particle_core_radius = pin->GetOrAddReal(
      "z4c", "co_" + nstr + "_particle_core_radius", 0.0);
  walk_velocity[0] = pin->GetOrAddReal("z4c", "co_" + nstr + "_predict_vx", 0.0);
  walk_velocity[1] = pin->GetOrAddReal("z4c", "co_" + nstr + "_predict_vy", 0.0);
  walk_velocity[2] = pin->GetOrAddReal("z4c", "co_" + nstr + "_predict_vz", 0.0);
  for (int a = 0; a < NDIM; ++a) {motion_pos[a] = pos[a];}
  walk_last_time = pmesh->time;
  have_motion_pos = false;
  horizon_tracking = false;
  track_source = 0;
  core_count = 0.0;
  core_rest_mass = 0.0;
  lapse_min = NAN;
  // Particles are attached to MeshBlockPack after Z4c construction, so only validate
  // the input contract here; EvolveParticleLapse dereferences the live particle object.
  if (mode == ParticleLapse && (particle_tag_min < 0
      || particle_tag_max < particle_tag_min || particle_core_radius <= 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "particle_lapse tracker " << n
              << " requires particles, 0 <= tag_min <= tag_max, and "
              << "particle_core_radius > 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  mass = pin->GetOrAddReal("z4c", "co_" + nstr + "_mass", 0.0);

  reflevel = pin->GetOrAddInteger("z4c", "co_" + nstr + "_reflevel", -1);
  radius = pin->GetOrAddReal("z4c", "co_" + nstr + "_radius", 0.0);

  out_every = pin->GetOrAddInteger("z4c", "co_" + nstr + "_out_every", 1);

  walk_every = pin->GetOrAddInteger("z4c", "tracker_walk_every", 1);
  if (walk_every < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl;
    std::cout << "tracker_walk_every must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (0 == global_variable::my_rank) {
    ofile.open(ofname.c_str());

    if (type == BlackHole) {
      ofile << "# Black Hole";
    } else {
      ofile << "# Neutron Star";
    }
    ofile << std::endl;

    ofile << "# 1:iter 2:time 3:x 4:y 5:z 6:vx 7:vy 8:vz "
          << "9:amr_level 10:dx1 11:dx2 12:dx3 13:track_source "
          << "14:core_count 15:core_rest_mass 16:lapse_min "
          << "17:horizon_tracking\n";
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
    if (mode != ParticleLapse) {
      vel[0] = - betax;
      vel[1] = - betay;
      vel[2] = - betaz;
    }
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
//! \fn void CompactObjectTracker::EvolveParticleLapse(MeshBlockPack *pmbp)
//! \brief Track a tagged dense-particle core before AH formation, then continue from
//! a motion-predicted local lapse minimum independently of surviving particles.
//!
//! A usable prior FastFlow snapshot switches `horizon_tracking` on permanently for this
//! process. Before that switch, the rest-mass-weighted center of tagged particles inside
//! `particle_core_radius` of the predicted center seeds the lapse search. Each search then
//! selects the minimum lapse in the 3^3 cell neighborhood around the seed. The predictor
//! uses the measured displacement of successive core centers (pre-AH) or lapse minima
//! (post-AH). Failed core or lapse searches are explicit in track_source=3 and in stdout.
void CompactObjectTracker::EvolveParticleLapse(MeshBlockPack *pmbp) {
  if ((pmesh->ncycle % walk_every) != 0) return;

  Real dtwalk = pmesh->time - walk_last_time;
  if (!(dtwalk > 0.0)) {dtwalk = walk_every*pmesh->dt;}
  Real predicted[NDIM];
  for (int a = 0; a < NDIM; ++a) {
    predicted[a] = pos[a] + walk_velocity[a]*dtwalk;
  }

  if (tracker_index < static_cast<int>(pmbp->pz4c->pfastflow.size())
      && pmbp->pz4c->pfastflow[tracker_index]->ah_surf_valid) {
    horizon_tracking = true;
  }

  Real target[NDIM] = {predicted[0], predicted[1], predicted[2]};
  core_count = 0.0;
  core_rest_mass = 0.0;
  track_source = horizon_tracking ? 2 : 3;
  if (!horizon_tracking) {
    auto *ppart = pmbp->ppart;
    auto &pr = ppart->prtcl_rdata;
    auto &pi = ppart->prtcl_idata;
    int npart = ppart->nprtcl_thispack;
    int tag_min = particle_tag_min;
    int tag_max = particle_tag_max;
    Real px = predicted[0], py = predicted[1], pz = predicted[2];
    Real radius2 = particle_core_radius*particle_core_radius;
    Real core[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    Kokkos::parallel_reduce("tracker_tagged_core",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &msum, Real &mx, Real &my, Real &mz,
                    Real &count) {
        int tag = pi(PTAG,p);
        Real dx = pr(IPX,p) - px;
        Real dy = pr(IPY,p) - py;
        Real dz = pr(IPZ,p) - pz;
        if (tag >= tag_min && tag <= tag_max
            && dx*dx + dy*dy + dz*dz <= radius2) {
          Real mass = pr(IPM,p);
          msum += mass;
          mx += mass*pr(IPX,p);
          my += mass*pr(IPY,p);
          mz += mass*pr(IPZ,p);
          count += 1.0;
        }
      }, Kokkos::Sum<Real>(core[0]), Kokkos::Sum<Real>(core[1]),
         Kokkos::Sum<Real>(core[2]), Kokkos::Sum<Real>(core[3]),
         Kokkos::Sum<Real>(core[4]));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, core, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    core_rest_mass = core[0];
    core_count = core[4];
    if (core[0] > 0.0) {
      target[0] = core[1]/core[0];
      target[1] = core[2]/core[0];
      target[2] = core[3]/core[0];
      track_source = 1;
      if (have_motion_pos && dtwalk > 0.0) {
        for (int a = 0; a < NDIM; ++a) {
          Real observed = (target[a] - motion_pos[a])/dtwalk;
          walk_velocity[a] = 0.5*walk_velocity[a] + 0.5*observed;
        }
      }
      for (int a = 0; a < NDIM; ++a) {motion_pos[a] = target[a];}
      have_motion_pos = true;
    }
  }

  // Find the owner of the predicted/core seed and its local 3^3 lapse minimum. The
  // neighborhood may use ghost cells, so a seed beside a MeshBlock face still sees the
  // adjacent block. The owner count should be exactly one on the half-open leaf mesh.
  auto &padm = pmbp->padm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  Real found[5] = {0.0, 0.0, 0.0, 0.0, 0.0}; // x,y,z,alpha,owner count
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    Real x1min = size.h_view(m).x1min, x1max = size.h_view(m).x1max;
    Real x2min = size.h_view(m).x2min, x2max = size.h_view(m).x2max;
    Real x3min = size.h_view(m).x3min, x3max = size.h_view(m).x3max;
    if (!(target[0] >= x1min && target[0] < x1max
          && target[1] >= x2min && target[1] < x2max
          && target[2] >= x3min && target[2] < x3max)) continue;
    Real dx1 = size.h_view(m).dx1, dx2 = size.h_view(m).dx2,
         dx3 = size.h_view(m).dx3;
    int ic = std::round((target[0] - (x1min + 0.5*dx1))/dx1);
    int jc = std::round((target[1] - (x2min + 0.5*dx2))/dx2);
    int kc = std::round((target[2] - (x3min + 0.5*dx3))/dx3);
    DualArray3D<Real> alp("particle_lapse_tracker", 3, 3, 3);
    auto &adm = padm->adm;
    par_for("Copy particle-lapse neighborhood", DevExeSpace(), 0, 2, 0, 2, 0, 2,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        alp.d_view(k,j,i) = adm.alpha(m, kc + indcs.ks + k - 1,
                                        jc + indcs.js + j - 1,
                                        ic + indcs.is + i - 1);
      });
    alp.template modify<DevMemSpace>();
    alp.template sync<typename DualArray3D<Real>::host_mirror_space>();
    Real amin = std::numeric_limits<Real>::max();
    for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
          if (alp.h_view(k,j,i) < amin) {
            amin = alp.h_view(k,j,i);
            found[0] = CellCenterX(ic + i - 1, indcs.nx1, x1min, x1max);
            found[1] = CellCenterX(jc + j - 1, indcs.nx2, x2min, x2max);
            found[2] = CellCenterX(kc + k - 1, indcs.nx3, x3min, x3max);
            found[3] = amin;
            found[4] = 1.0;
          }
        }
      }
    }
    break;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, found, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  Real oldpos[NDIM] = {pos[0], pos[1], pos[2]};
  if (found[4] > 0.5) {
    for (int a = 0; a < NDIM; ++a) {pos[a] = found[a]/found[4];}
    lapse_min = found[3]/found[4];
  } else {
    for (int a = 0; a < NDIM; ++a) {pos[a] = target[a];}
    lapse_min = NAN;
    track_source = 3;
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING: particle_lapse tracker " << tracker_index
                << " could not reacquire a local lapse neighborhood at cycle="
                << pmesh->ncycle << " time=" << pmesh->time << std::endl;
    }
  }

  if (horizon_tracking && dtwalk > 0.0) {
    if (have_motion_pos) {
      for (int a = 0; a < NDIM; ++a) {
        Real observed = (pos[a] - motion_pos[a])/dtwalk;
        walk_velocity[a] = 0.5*walk_velocity[a] + 0.5*observed;
      }
    }
    for (int a = 0; a < NDIM; ++a) {motion_pos[a] = pos[a];}
    have_motion_pos = true;
  } else if (!have_motion_pos && dtwalk > 0.0) {
    for (int a = 0; a < NDIM; ++a) {
      walk_velocity[a] = (pos[a] - oldpos[a])/dtwalk;
      motion_pos[a] = pos[a];
    }
    have_motion_pos = true;
  }
  for (int a = 0; a < NDIM; ++a) {vel[a] = walk_velocity[a];}
  walk_last_time = pmesh->time;
}

//----------------------------------------------------------------------------------------
void CompactObjectTracker::EvolveTracker(MeshBlockPack *pmbp) {
  if (mode == ParticleLapse) {
    EvolveParticleLapse(pmbp);
    owns_compact_object = false;
    return;
  }
  if (owns_compact_object) {
    if (mode == ODE) {
      for (int a = 0; a < NDIM; ++a) {
        pos[a] += pmesh->dt * vel[a];
      }
    } else if ((pmesh->ncycle % walk_every) == 0) {
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
void CompactObjectTracker::WriteTracker(MeshBlockPack *pmbp) {
  if ((pmesh->ncycle % out_every) != 0) return;

  auto &size = pmbp->pmb->mb_size;
  auto &lev = pmbp->pmb->mb_lev;
  Real local[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    if ((pos[0] >= size.h_view(m).x1min && pos[0] < size.h_view(m).x1max) &&
        (pos[1] >= size.h_view(m).x2min && pos[1] < size.h_view(m).x2max) &&
        (pos[2] >= size.h_view(m).x3min && pos[2] < size.h_view(m).x3max)) {
      local[0] = static_cast<Real>(lev.h_view(m));
      local[1] = size.h_view(m).dx1;
      local[2] = size.h_view(m).dx2;
      local[3] = size.h_view(m).dx3;
      local[4] = 1.0;
      break;
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, local, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (0 == global_variable::my_rank) {
    Real inv_owners = (local[4] > 0.0 ? 1.0 / local[4] : 0.0);
    ofile << pmesh->ncycle << " "
          << pmesh->time << " "
          << pos[0] << " "
          << pos[1] << " "
          << pos[2] << " "
          << vel[0] << " "
          << vel[1] << " "
          << vel[2] << " "
          << (local[4] > 0.0 ? local[0] * inv_owners : -1.0) << " "
          << (local[4] > 0.0 ? local[1] * inv_owners : NAN) << " "
          << (local[4] > 0.0 ? local[2] * inv_owners : NAN) << " "
          << (local[4] > 0.0 ? local[3] * inv_owners : NAN) << " "
          << track_source << " "
          << core_count << " "
          << core_rest_mass << " "
          << lapse_min << " "
          << static_cast<int>(horizon_tracking)
          << std::endl << std::flush;
  }
}
