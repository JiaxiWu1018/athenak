//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fastflow.hpp
//! \brief Basic functionality for the FastFlow class.

#ifndef Z4C_FASTFLOW_HPP_
#define Z4C_FASTFLOW_HPP_

#include <cstdio>

#include <string>
#include <vector>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "geodesic-grid/gauss_legendre.hpp"
#include "z4c_macros.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class MeshBlockPack;
class ParameterInput;

// Enum variables for extrinsic curvature and
// the metric derivatives.
enum ExtrinsicCurvatureIndex {K11=0, K12=1, K13=2, K22=3, K23=4, K33=5, NEXCURV=6};
enum SpatialMetricDrvsIndex {D1S11=0, D1S12=1, D1S13=2, D1S22=3, D1S23=4, D1S33=5,
                            D2S11=6, D2S12=7, D2S13=8, D2S22=9, D2S23=10, D2S33=11,
                            D3S11=12, D3S12=13, D3S13=14, D3S22=15, D3S23=16, D3S33=17,
                            NDRVSSPMETRIC=18};

//! \class FastFlow
//! \brief Apparent Horizon Finder class based on fast-flow algorithm
class FastFlow {
 public:
  // Constructor for FastFlow object
  FastFlow(MeshBlockPack *pmbp, ParameterInput *pin, int n);

  // Default Destructor for FastFlow object (closes output file)
  ~FastFlow();

  void Find(int iter, Real time); // main functionality for finding AH
  void Write(int iter, Real time); // function for result writing
  template <int NGHOST>
  void MetricDerivatives(Real time); // compute the metric derivatives
  template <int NGHOST>
  void MetricInterp();
  void ComputeSphericalHarmonics();
  void RadiiFromSphericalHarmonics();
  void UpdateFlowSpectralComponents();
  void SurfaceIntegrals();

  // ---- Snapshot of the last surface that converged IN THIS RUN -----------------------
  //! Consumers outside FastFlow (particle excision) must key off `ah_surf_valid`, not
  //! `ah_found`. `ah_found` is reset at the top of every FastFlowLoop, is not updated
  //! outside [start_time, stop_time], and is restored from the restart parameter dump
  //! while the l>0 shape coefficients and rr_min are NOT -- so after a restart a restored
  //! `ah_found == true` can coexist with rr_min == -1 and ac/as == 0. `ah_surf_valid` is
  //! sticky and is set only by SnapshotSurface(), after the consumer geometry and
  //! persistence gates pass and the surface is wholly on the mesh.
  bool ah_surf_valid;
  Real ah_surf_center[3];
  Real ah_surf_rmin, ah_surf_rmax;   // angular extrema, for the consumer's bracket
  Real ah_surf_rmin_limit, ah_surf_rmax_limit; // optional physical publication limits
  Real ah_surf_rmin_cells;            // optional minimum radius in finder-local cells
  Real ah_surf_local_dx;              // local cell width used by the last candidate gate
  Real ah_surf_hrel_limit;            // optional |<H>|/area publication limit
  int ah_surf_persist, ah_surf_candidate_streak;
  //! Capability mode deliberately leaves the resolution, expansion, and persistence
  //! tests above as diagnostics only. Its consumer gate is limited to finite positive,
  //! wholly on-grid geometry plus explicit catastrophic radius/centre-jump bounds.
  bool ah_surf_capability_mode, ah_surf_capability_have_center;
  Real ah_surf_capability_rmax, ah_surf_capability_center_jump;
  Real ah_surf_capability_last_center[3];
  bool ah_surf_warned;                // the off-grid warning is printed once
  bool ah_surf_geometry_warned;       // the geometry warning is printed once
  DualArray1D<Real> a0_surf, ac_surf, as_surf;   // packing as a0/ac/as
  bool SnapshotSurface();

  // ---- Weaker snapshot used only as a deep-lapse geometric permission mask ---------
  //! This snapshot never makes a horizon authoritative.  When explicitly enabled, it
  //! records the last raw FastFlow convergence in this run whose radii are finite and
  //! positive, whose entire collocation surface is on-grid, and whose maximum radius
  //! passes the same absurd-size guard as the trusted consumer path.  In particular it
  //! does NOT apply the minimum-cell, normalized-expansion, or persistence gates above.
  //! Particle excision may use it only in conjunction with a deep-lapse threshold; AH
  //! publication and ordinary AH excision continue to key exclusively on ah_surf_valid.
  bool ah_raw_surf_enabled, ah_raw_surf_valid;
  bool ah_raw_surf_reported, ah_raw_surf_warned;
  Real ah_raw_surf_center[3];
  Real ah_raw_surf_rmin, ah_raw_surf_rmax;
  Real ah_raw_surf_time, ah_raw_surf_area, ah_raw_surf_hrel;
  int ah_raw_surf_cycle;
  DualArray1D<Real> a0_raw_surf, ac_raw_surf, as_raw_surf;
  void SnapshotRawSurface(int iter, Real time, Real candidate_rmax);
  bool CurrentSurfaceOnGrid(int &noff);
  int GetLmax() const {return lmax;}
  int GetLmpoints() const {return lmpoints;}

  // Some of the main parameters in the fast-flow algorithm
  bool ah_found; // Horizon found
  Real time_first_found; // Time, when horizon first found
  Real initial_radius; // Initial guess for the radius of the horizon
  Real rr_min; // Minimum radius
  Real expand_guess; // Expand the initial guess by this factor
  Real center[3]; // Center around which the horizon is searched

  // Fast-Flow parameters
  Real hmean_tol; // for convergence
  Real mass_tol; // for convergence
  int flow_iterations; // number of flow iterations
  Real flow_alpha_beta_const; // alpha & beta constants in the iteration formula
                              // Eqs. (43) & (44) of https://arxiv.org/pdf/gr-qc/9707050
  bool verbose;
  bool output_ylm;
  bool output_grid;

  // Spherical harmonics & Legendre polynomials
  int lmax; // Multipoles
  int nangles; // Number of angles on Gauss-Legendre grid
  int ntheta; // Number of theta points

  // Compact Object Tracker variables
  int use_puncture; // n surface follows the puncture tracker if use_puncture[n] > 0
  Real merger_distance; // Distance in M at which BHs are considered as merged
  bool use_puncture_massweighted_center;

  // Start and Stop times for each surface
  Real start_time;
  Real stop_time;

 private:
  int npunct; // Number of punctures
  int lmax1; // lmax + 1
  int lmpoints; // lmax * lmax
  int nh; // Counter variable
  bool wait_until_punc_are_close;
  [[maybe_unused]] bool use_stored_metric_drvts;
  int nhorizon; // Number of horizons
  std::string flow_function;
  int flowflag = 0;
  int fastflow_iter = 0;

  // Pointer to Gauss-Legendre object
  GaussLegendreGrid *gl_grid;

  // Arrays of spherical harmonics and derivatives
  DualArray2D<Real> Y0, Yc, Ys;
  DualArray2D<Real> dY0dth, dYcdth, dYsdth, dYcdph, dYsdph;
  DualArray2D<Real> dY0dth2, dYcdth2, dYcdthdph, dYsdth2, dYsdthdph, dYcdph2, dYsdph2;

  // Arrays for spectral coefficients
  DualArray1D<Real> a0, ac, as;
  Real last_a0; // last coefficient a_00

  // Arrays used for the fields on the sphere
  DvceArray1D<Real> rr, rr_dth, rr_dph;

  // Array computed in Surface Integrals
  DualArray1D<Real> rho;

  // Indexes of surface integrals
  enum {
    iarea,
    icoarea,
    ihrms,
    ihmean,
    iSx, iSy, iSz,
    invar
  };
  static constexpr int kInvar = 7;
  Real integrals[kInvar]; // Array of surface integrals

  // Indexes of horizon quantities
  enum{
    harea,
    hcoarea,
    hhrms,
    hhmean,
    hSx, hSy, hSz, hS,
    hmass,
    hmeanradius,
    hminradius,
    hnvar
  };
  static constexpr int kHnvar = 11;
  Real ah_prop[kHnvar]; // Array of horizon quantities

  // 5D Device array for the metric derivatives
  DvceArray5D<Real> dg;

  // Vectors to hold the DvceArray1D interpolated values of GaussLegendreGrid
  DvceArray2D<Real> g_interp, K_interp, dg_interp;

  // Flag points
  DualArray1D<int> havepoint;

  // Functions used in the fast-flow algorithm
  void FastFlowLoop();
  void InitialGuess();

  // Pointers to MeshBlockPack and ParameterInput
  MeshBlockPack *pmbp;
  ParameterInput *pin;

  // Control parameters
  int root;
  int ioproc;
  std::string ofname_summary;
  std::string ofname_shape;
  std::string ofname_verbose;
  std::string ofname_ylm;
  std::string ofname_grid;
  std::string ofname_consumer;
  FILE *pofile_summary;
  FILE *pofile_shape;
  FILE *pofile_verbose;
  FILE *pofile_ylm;
  FILE *pofile_grid;
  FILE *pofile_consumer;

  // Functions to interface with puncture tracker
  Real PuncMaxDistance();
  Real PuncMaxDistance(const int pix);
  Real PuncSumMasses();
  void PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc);
  bool PuncAreClose();
  Real LocalCellWidthAtCenter();
};

#endif  // Z4C_FASTFLOW_HPP_
