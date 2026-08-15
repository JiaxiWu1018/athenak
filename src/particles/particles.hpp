#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.hpp
//  \brief definitions for Particles class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations

// constants that enumerate ParticlesPusher options
// (boris = special-relativistic Boris; gr_boris = general-relativistic Boris whose q=0 limit
// is the geodesic integrator. geo_boris is intentionally NOT implemented for NRPIC.)
enum class ParticlesPusher {drift, leap_frog, lagrangian_tracer, lagrangian_mc,
                            boris, gr_boris};

// constants that enumerate ParticleTypes
enum class ParticleType {cosmic_ray, dust};

//----------------------------------------------------------------------------------------
//! \struct ParticlesTaskIDs
//  \brief container to hold TaskIDs of all particles tasks

struct ParticlesTaskIDs {
  TaskID push;
  TaskID newgid;
  TaskID count;
  TaskID irecv;
  TaskID sendp;
  TaskID recvp;
  TaskID newdt;
  TaskID energy;
  TaskID csend;
  TaskID crecv;
  TaskID check;
};

namespace particles {

//----------------------------------------------------------------------------------------
//! \class Particles

class Particles {
  friend class ParticlesBoundaryValues;
 public:
  Particles(MeshBlockPack *ppack, ParameterInput *pin);
  ~Particles();

  // data
  ParticleType particle_type;
  int nprtcl_thispack;             // number of particles this MeshBlockPack
  int nrdata, nidata;
  DvceArray2D<Real> prtcl_rdata;   // real number properties each particle (x,v,etc.)
  DvceArray2D<int>  prtcl_idata;   // integer properties each particle (gid, tag, etc.)
  Real dtnew;
  Real mass;                       // default/scalar rest mass (per-particle override in IPM)
  Real q_over_m;                   // charge-to-mass ratio (0 for dust)

  // migration debug instrumentation (<particles> debug = 0|1|2, default 0):
  //   1 = per-cycle migration summary + post-migration validation task (CheckMigration),
  //       FATAL on any violation (GID out of pack range, particle outside its MeshBlock
  //       bbox, destination-search failure, or particle-count change);
  //   2 = level 1 plus a per-event migration log (cycle, tag, old->new gid, offsets)
  int debug_lvl;
  // per-cycle counters written by SetNewPrtclGID when debug_lvl >= 1, classified by the
  // crossing offset |ix|+|iy|+|iz| = 1/2/3. nsearch_fail counts particles for which no
  // destination MeshBlock was found (always fatal; stays 0 until the Stage-3a(c) search
  // rewrite wires failure detection -- the legacy neighbor walk cannot detect failure).
  int nmigr_face, nmigr_edge, nmigr_corner, nsearch_fail;
  // particle count at the first CheckMigration call (-1 until then); without destruction
  // (not yet implemented) the count must stay exactly constant on a single rank
  int nprtcl_initial;

  // snapshots of the field/metric at the previous step, used by the GR pusher to evaluate
  // the implicit geodesic substep at the time midpoint. Allocated only for gr_boris. For a
  // static background (all Stage-2 tests) these equal the current arrays; they carry real
  // information once the metric is dynamical (Stage 4 live Z4c).
  DvceArray5D<Real> w0_last;       // MHD primitives at step n
  DvceArray5D<Real> bcc0_last;     // cell-centred B at step n
  DvceArray5D<Real> adm_last;      // ADM metric at step n
  DvceArray5D<Real> z4c_last;      // Z4c variables at step n

  ParticlesPusher pusher;

  // Boundary communication buffers and functions for particles
  ParticlesBoundaryValues *pbval_part;

  // container to hold names of TaskIDs
  ParticlesTaskIDs id;

  // functions...
  void CreateParticleTags(ParameterInput *pin);
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // pusher kernels (particles_pushers.cpp dispatches Push() to these)
  void BorisPush();      // special-relativistic Boris (boris_pusher.cpp)
  void GR_BorisPush();   // general-relativistic Boris / geodesic (gr_boris.cpp)
  TaskStatus Push(Driver *pdriver, int stage);
  TaskStatus NewGID(Driver *pdriver, int stage);
  TaskStatus SendCnt(Driver *pdriver, int stage);
  TaskStatus InitRecv(Driver *pdriver, int stage);
  TaskStatus SendP(Driver *pdriver, int stage);
  TaskStatus RecvP(Driver *pdriver, int stage);
  TaskStatus ClearSend(Driver *pdriver, int stage);
  TaskStatus ClearRecv(Driver *pdriver, int stage);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);
  TaskStatus EnergyCalculation(Driver *pdriver, int stage);
  // post-migration validation: containment/GID-range/count checks (particles_debug.cpp);
  // no-op unless <particles> debug >= 1, fatal (exit) on any violation
  TaskStatus CheckMigration(Driver *pdriver, int stage);
  // exhaustive host-side enumeration audit of the destination search against a
  // brute-force bbox oracle (particles_debug.cpp); fatal on any mismatch. Single-rank,
  // strictly-periodic meshes only (test utility, invoked by the part_crossing pgen).
  void AuditDestinationSearch();

  // load particle initial conditions from an HDF5 file (read_particle.cpp)
  void read_prtcl_table(const char *fname);
  // host helper: local MeshBlock index whose bbox contains (x,y,z), or -1 (read_particle/restart)
  int FindContainingMeshBlock(Real x, Real y, Real z) const;
  // compute conserved specific energy -u_t into IPEN (calc_energy.cpp)
  template <int NGHOST>
  void calc_prtcl_energy();

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Particles
};

} // namespace particles
#endif // PARTICLES_PARTICLES_HPP_
