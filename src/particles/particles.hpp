#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.hpp
//  \brief definitions for Particles class

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations

// constants that enumerate ParticlesPusher options
// (boris = special-relativistic Boris; gr_boris = general-relativistic Boris whose q=0
// limit is the geodesic integrator. geo_boris is intentionally NOT implemented for
// NRPIC.)
enum class ParticlesPusher {drift, leap_frog, lagrangian_tracer, lagrangian_mc,
                            boris, gr_boris};

// constants that enumerate ParticleTypes
enum class ParticleType {cosmic_ray, dust};

// Cross-level Tmunu deposition: conservative restricts fine cloud cells into covering
// coarse cells; native deposits independently at each target block's resolution.
enum class CrossLevelDeposit {conservative, native};

//----------------------------------------------------------------------------------------
//! \struct ParticlesTaskIDs
//  \brief container to hold TaskIDs of all particles tasks

struct ParticlesTaskIDs {
  TaskID push;
  TaskID excise;
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
  TaskID tmunu;
};

namespace particles {

//----------------------------------------------------------------------------------------
//! \struct TmunuImage
//  \brief local ghost-image record for cross-block stress-energy deposition.

struct TmunuImage {
  int target_m;     // local MeshBlock index (gid - gids) of the receiving block
  int tag;          // source particle tag (canonical-order key)
  int off_code;     // (bx+1) + 3*(by+1) + 9*(bz+1), in 0..26; 13 is self
  int lev;          // -1 for same-level routing; otherwise the target refinement level
  int slev;         // source stencil level; > lev selects conservative restrict
  int idx[3];       // same-level stencil, or fine stencil for conservative restriction
  Real delta[3];    // CIC offset matching idx, clamped to [0,1]
  Real x[3];        // absolute position used to rebuild a target-native CIC
  Real sxmin[3];    // origin of the fine stencil used for conservative restriction
  Real mass;        // particle rest mass (IPM)
  Real lorentz;     // normal-frame Lorentz factor W at the particle
  Real u_d[3];      // covariant velocity u_i
};

//----------------------------------------------------------------------------------------
//! \struct TmunuImageWire
//  \brief wire form of a Tmunu image destined for a MeshBlock on another rank.

struct TmunuImageWire {
  int target_gid;   // global id of the receiving block
  int tag;
  int off_code;
  int lev;
  int slev;
  int idx[3];
  Real delta[3];
  Real x[3];
  Real sxmin[3];
  Real mass;
  Real lorentz;
  Real u_d[3];
};

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
  Real mass;                       // default/scalar rest mass (per-particle override
                                   // in IPM)
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
  // migration conservation ledger (CheckMigration, debug >= 1): GLOBAL {particle count,
  // sum of tags, sum of tag^2} captured at the first check and recomputed every cycle.
  // Since destruction exists (Stage 3c) the ledger is TWO-SIDED: alive + destroyed must
  // equal the captured totals component-wise, where the destroyed-side checksums are
  // accumulated at marking time in ledger_dead (per-rank cumulative {sum tag, sum
  // tag^2}, Allreduced at check time) and the destroyed count comes from the global
  // census ledger on Mesh. The tag checksums catch identity corruption that count
  // conservation alone cannot (a lost particle replaced by a duplicate of another --
  // the compaction-bug signature) -- including across destruction events. Unsigned-64
  // sums are modular: wraparound is harmless for equality tests.
  bool ledger_init;
  uint64_t ledger0[3];
  uint64_t ledger_dead[2];
  // per-cycle destruction counters by reason {0=exit, 1=sphere, 2=lapse}, set by
  // SetNewPrtclGID's readback each cycle (this rank only; the global census lives in
  // ParticlesBoundaryValues::ndest_global)
  int ndestroy_thisrank[3];
  // death-record ledger: every destruction appends one row to <basename>.prtcl_destroy
  // .csv (exact cycle/time/position/velocity/reason at marking), flushed collectively
  // on every destroy-cycle; <particles> destroy_log = true|false (default true)
  bool destroy_log;
  std::string destroy_log_fname;

  // parameterized excision (Stage 3c(b); replaces the prototype's hardcoded
  // rexcise=2-iff-not-Minkowski, bug B1). Two independent criteria, both default OFF:
  //   excise_radius > 0: destroy at |x - excise_center| < excise_radius (pure geometry,
  //     works under any pusher; the sphere lives in coordinate space -- periodic images
  //     are not considered, so keep it away from periodic boundaries);
  //   excise_lapse > 0: destroy where alpha(x_p) < excise_lapse, with alpha Lagrange-
  //     interpolated from the live Z4c (I_Z4C_ALPHA) or ADM (I_ADM_ALPHA) arrays --
  //     the same source switch as the gr_boris pusher. NOTE the threshold is GAUGE-
  //     dependent: in Cartesian Kerr-Schild the Schwarzschild horizon sits at
  //     alpha = 1/sqrt(2) ~ 0.707 (sensible thresholds 0.5-0.6 excise INSIDE the
  //     horizon); values like 0.1 belong to moving-puncture/1+log collapsed lapses.
  Real excise_radius;
  Real excise_x1, excise_x2, excise_x3;
  Real excise_lapse;
  bool excise_any;
  // per-cycle marking written by MarkExcised, consumed by SetNewPrtclGID:
  // flag 0 = keep, 1 = sphere, 2 = lapse; crit = criterion value at marking (r or alpha)
  DvceArray1D<int>  excise_flag;
  DvceArray1D<Real> excise_crit;

  // Bounded gr_boris non-convergence diagnostic. The implicit geodesic substep falls
  // back to forward Euler when the fixed-point iteration does not converge; that is a
  // legitimate, documented outcome (it is expected at large CFL), but warning once per
  // particle per cycle can emit O(N_particle) lines per cycle and tens of GB of log.
  // Instead the device kernel COUNTS every failure in boris_nfail(0) and prints a
  // detailed line for at most kBorisDetail of them per cycle (boris_nfail(1) is the
  // detail budget claimed so far); the host then emits ONE summary line per rank per
  // cycle. boris_nfail_cum is the per-rank running total, reported in the summary so no
  // failure is ever hidden. The FIRST failure of a run prints full particle state.
  static constexpr int kBorisDetail = 3;
  // {0: non-convergence this cycle, 1: reserved detail budget, 2: REJECTED geodesic
  //  substeps this cycle, 3: non-finite write-backs refused this cycle}
  DvceArray1D<int> boris_nfail;
  std::int64_t boris_nfail_cum;
  bool boris_first_fail_seen;
  // A REJECTED update was NOT TAKEN: the particle keeps its step-n position and
  // momentum. Together, slots 2 and 3 make "a non-finite state is never written into
  // prtcl_rdata" an invariant of GR_BorisPush. A rejection costs that particle accuracy,
  // so it is reported always -- never only under <particles> debug.
  std::int64_t boris_nreject_cum;
  bool boris_first_reject_seen;

  // Stress-energy feedback supports a 3D Z4c consumer without dynamical GRMHD. Images
  // may cross ranks and coarse-fine interfaces, including through dynamic AMR regrids.
  bool feedback;
  CrossLevelDeposit xlevel_deposit;
  // Unified canonical queue: self records, same-rank images, then received images.
  DualArray1D<TmunuImage> tmunu_images;
  DvceArray1D<int> tmunu_nimg;  // {same-rank image count, cross-rank send count}
  int nimages_thispack;
  DualArray1D<TmunuImageWire> tmunu_img_send;
  int nimg_send_thispack;
  // Particle-side and cell-side sums of the ten conserved Tmunu combinations.
  DvceArray1D<Real> tmunu_psums;
  DvceArray1D<Real> tmunu_csums;

  // snapshots of the field/metric at the previous step, used by the GR pusher to evaluate
  // the implicit geodesic substep at the time midpoint. Allocated only for gr_boris. For
  // a static background (all Stage-2 tests) these equal the current arrays; they
  // carry real information once the metric is dynamical (Stage 4 live Z4c).
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
  // Dynamic-AMR redistribution. RelabelForAMR runs while the old block geometry is
  // available; ShipAfterAMR uses the regular migration chain after the new grid is live.
  TaskStatus RelabelForAMR(const DualArray1D<int> &oldtonew,
                           const DualArray1D<int> &newrank,
                           const DualArray1D<int> &refflag, int old_gids);
  void ShipAfterAMR();
  // death-record ledger + end-of-run accounting (particles_destroy.cpp): FlushDeathLog
  // gathers this cycle's death records to rank 0 and appends them to the CSV (collective
  // -- called on every rank whenever the global census is nonzero); PrintFinalSummary
  // prints the initial/final/destroyed-by-reason tally + conservation verdict (rank 0)
  void FlushDeathLog();
  void PrintFinalSummary();
  // excision marking task (particles_excise.cpp), scheduled between Push and NewGID
  // only when a criterion is enabled; mark_excised is its NGHOST-dispatched kernel
  TaskStatus MarkExcised(Driver *pdriver, int stage);
  template <int NGHOST> void mark_excised();
  // Stress-energy deposition, scheduled after EnergyCalculation when feedback is on.
  TaskStatus SetPrtclTmunu(Driver *pdriver, int stage);
  template <int NGHOST> void set_prtcl_tmunu();
  // exhaustive host-side enumeration audit of the destination search against a
  // brute-force bbox oracle (particles_debug.cpp); fatal on any mismatch. Single-rank,
  // strictly-periodic meshes only (test utility, invoked by the part_crossing pgen).
  void AuditDestinationSearch();

  // load particle initial conditions from an HDF5 file (read_particle.cpp)
  void read_prtcl_table(const char *fname);
  // host helper: local MeshBlock index whose bbox contains (x,y,z), or -1
  // (read_particle/restart)
  int FindContainingMeshBlock(Real x, Real y, Real z) const;
  // compute conserved specific energy -u_t into IPEN (calc_energy.cpp)
  template <int NGHOST>
  void calc_prtcl_energy();

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Particles
};

} // namespace particles
#endif // PARTICLES_PARTICLES_HPP_
