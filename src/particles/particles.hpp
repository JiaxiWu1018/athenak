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
#include "deposit_shape.hpp"
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
  int order;        // MOOD hierarchy level of the SOURCE PARTICLE (0 = highest order).
                    // Must be identical on every record of a particle -- the weights of
                    // one kernel sum to one over the WHOLE cloud, so mixing kernels
                    // within a particle would break the conservation identity. Decided on
                    // the source rank and shipped; refreshed each MOOD sweep.
  int src_p;        // source-rank particle index for records generated here, else -1
  int aux;          // receive-buffer index for records that arrived over MPI, else -1
                    // (survives the canonical sort, so the order refresh can find them)
  int idx[3];       // same-level stencil, or fine stencil for conservative restriction
  Real delta[3];    // shape-function offset matching idx, clamped to [0,1]
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
  int order;        // MOOD hierarchy level of the source particle (see TmunuImage)
  int src_p;        // source-rank particle index (local bookkeeping; not on the wire)
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
  DvceArray1D<int> boris_nfail;   // {failures this cycle, detail slots claimed}
  std::int64_t boris_nfail_cum;
  bool boris_first_fail_seen;

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

  // ---- higher-order particle->mesh deposition (deposit_shape.hpp, particles_mood.cpp)
  // --
  // <particles> deposit_shape = cic | m4 | lambda22 | lambda44 (default cic = the
  // historical kernel, reproduced bit for bit through its own untouched code path).
  // deposit_shape names the TOP of the MOOD hierarchy; the cascade below it is fixed
  // (lambda44 -> lambda22 -> m4), exactly as in Diener, Rosswog & Torsello (2022) Sec.
  // 2.1.3, with the positive-definite m4 as the untested "parachute".
  DepositShape deposit_shape;
  bool deposit_renorm;       // divide the W weights by their sum (partition of unity to
                             // 1 ulp); structural no-op for cic
  bool deposit_generic_cic;  // TEST KNOB: route cic through the generalised kernel so a
                             // single run can assert the two paths agree bitwise
  // MOOD ("repeat-until-valid") fallback. mood_nlevels == 1 disables the cascade.
  bool mood_on;
  bool mood_monitor;   // detect + report, never demote (pure higher-order runs)
  int  mood_nlevels;
  DepositShape mood_hier[3];
  int  mood_max_sweeps;
  int  mood_detector;        // 0 = off, 1 = E >= 0 only, 2 = order-1+2 principal minors
  Real mood_tol;             // relative slack, scaled by the global max E
  Real mood_neg_frac;        // extra bar, as a fraction of the peak E (0 = strict)
  int  mood_diag_cadence;
  // per-particle hierarchy level (0 = top). Sized to the local particle count each cycle.
  DvceArray1D<int> deposit_order_p;
  // one-component ghosted inadmissibility flag + its dedicated CC boundary machinery.
  // Ghost-filling the flag lets the SOURCE rank see a particle's entire stencil, which is
  // what makes the demotion decision identical on every record of that particle.
  DvceArray5D<Real> u_mood;
  DvceArray5D<Real> coarse_u_mood;
  MeshBoundaryValuesCC *pbval_mood;
  DvceArray1D<int>  mood_ncell;    // per-criterion inadmissible-cell census (5 entries)
  DvceArray1D<int>  mood_level_ct; // {n at level 0, 1, 2}
  DvceArray1D<int>  mood_radial;   // radial histogram of demoted particles
  int mood_nbin;
  Real mood_rmax;
  Real mood_center[3];
  std::string mood_log_fname;
  bool mood_log_open;
  // recv-buffer index -> canonical-queue slot, rebuilt after each cycle's sort
  DvceArray1D<int> tmunu_recv_slot;
  // per-particle scratch reused by the deferred identity bookkeeping
  DvceArray1D<Real> tmunu_lor;          // normal-frame Lorentz factor W
    DvceArray1D<int>  tmunu_finestencil;  // 1 iff the fine sublevel was used

  // snapshots of the field/metric at the previous step, used by the GR pusher to evaluate
  // the implicit geodesic substep at the time midpoint. Allocated only for gr_boris. For
  // a static background (all Stage-2 tests) these equal the current arrays; they
  // carry real information once the metric is dynamical (Stage 4 live Z4c).
  DvceArray5D<Real> w0_last;       // MHD primitives at step n
  DvceArray5D<Real> bcc0_last;     // cell-centred B at step n
  DvceArray5D<Real> adm_last;      // ADM metric at step n
  DvceArray5D<Real> z4c_last;      // Z4c variables at step n
  // Optional geodesic diagnostics, recomputed after particle migration so their
  // particle index matches prtcl_rdata at output time.  gr_boris_du_dt stores the
  // instantaneous covariant force du_i/dt and gr_boris_dL_dt stores
  // d(x cross u)/dt from the same interpolated metric derivatives as the pusher.
  DvceArray2D<Real> gr_boris_du_dt;
  DvceArray2D<Real> gr_boris_dL_dt;
  // In the live-monopole control, retain the unfiltered live-field rates beside the
  // rates actually used by the spherical pusher.  These are diagnostics only.
  DvceArray2D<Real> gr_boris_raw_du_dt;
  DvceArray2D<Real> gr_boris_raw_dL_dt;
  bool gr_boris_diagnostics;
  // When enabled, particles continue to move but always read the initial numerical
  // Z4c/ADM snapshots.  This isolates initial mesh/interpolation anisotropy from
  // time-dependent matter/metric feedback.
  bool gr_boris_freeze_metric;
  // Experimental live-monopole control.  Deposition and Z4c remain live, while the
  // particle geodesic pusher uses only a spherical angular average of the current
  // numerical metric.  All switches are opt-in and default false/off.
  bool gr_boris_live_monopole;
  bool gr_boris_monopole_profile_valid;
  int gr_boris_monopole_nr;
  int gr_boris_monopole_sample_stride;
  int gr_boris_monopole_profile_interval;
  Real gr_boris_monopole_rmax;
  Real gr_boris_monopole_dr;
  Real gr_boris_monopole_center[3];
  DvceArray2D<Real> gr_boris_monopole_profile_old;
  DvceArray2D<Real> gr_boris_monopole_profile_new;
  DvceArray2D<Real> gr_boris_monopole_accum;
  std::string gr_boris_monopole_profile_fname;

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
  void GRBorisDiagnostics();  // instantaneous du_i/dt and dL_i/dt (gr_boris.cpp)
  void BuildGRBorisMonopoleProfiles(
      const DvceArray5D<Real> &adm_old, const DvceArray5D<Real> &adm_new,
      bool use_z4c, const DvceArray5D<Real> &z4c_old,
      const DvceArray5D<Real> &z4c_new, bool equal_time);
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
  void DepositAllRecords();
  // ---- MOOD (particles_mood.cpp) ----
  void MoodAllocate(ParameterInput *pin);
  int  MoodDetect();                 // fill u_mood on physical cells; global bad count
  void MoodFillGhosts();             // synchronous CC ghost fill of u_mood
  int  MoodDemote();                 // bump deposit_order_p where a stencil is flagged
  void MoodStampRecords();           // copy per-particle orders onto the local records
  void MoodReport(int ncycle, Real time, int nsweep, int nbad0, int nbad1);
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
