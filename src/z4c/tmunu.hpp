#ifndef Z4C_TMUNU_HPP_
#define Z4C_TMUNU_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tmunu.hpp
//! \brief definitions for Tmunu class, which represents the stress-energy tensor
//!  decomposed into E, S_i, and S_{ij}, all undensitized.

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mesh/mesh.hpp"
#include "eos/primitive-solver/ps_types.hpp"

// forward declarations
class MeshBlockPack;
class MeshBoundaryValuesCC;

//! \class Tmunu
class Tmunu {
 public:
  Tmunu(MeshBlockPack *ppack, ParameterInput *pin);
  ~Tmunu();

  // Indices of Tmunu variables
  enum {
    I_Tmunu_Sxx, I_Tmunu_Sxy, I_Tmunu_Sxz, I_Tmunu_Syy, I_Tmunu_Syz, I_Tmunu_Szz,
    I_Tmunu_E, I_Tmunu_Sx, I_Tmunu_Sy, I_Tmunu_Sz,
    N_Tmunu
  };
  // Names of Tmunu variables
  static char const * const Tmunu_names[N_Tmunu];

  // Number of spatial dimensions (3+1 gravity)
  int const NDIM = 3;

  struct Tmunu_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> E;      // energy density
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> S_d;    // momentum density
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> S_dd;   // stress tensor
  };

  Tmunu_vars tmunu;

  DvceArray5D<Real> u_tmunu;                          // Tmunu

  // ---- Entity-style digital filter of the deposited sources (tmunu_filter.cpp) ----
  // <particles>/tmunu_filter_passes (default 0 = OFF: no allocation, no communication,
  // no kernel -- the pre-filter code path is untouched). Each pass applies the exact
  // separable [1/4,1/2,1/4]^3 kernel of Hakobyan et al. arXiv:2511.17710 Eq. (11) to all
  // ten RAW (undensitized) components, after a cell-centered ghost fill of u_tmunu.
  int nfilter_passes;
  int filter_diag_cadence;   // <time>/ndiag: cycle cadence of the conservation report
  DvceArray5D<Real> coarse_u_tmunu;    // coarse buffer for the SMR ghost fill
  DvceArray4D<Real> u_filt_scratch;    // ONE-component scratch (components filter
                                       // independently; avoids a 10-component copy)
  DvceArray1D<Real> filt_sums;         // 20 = {pre,post} x 10 proper-volume integrals
  MeshBoundaryValuesCC *pbval_tmunu;   // dedicated bvals object (own MPI communicator)

  // filter driver: ghost fill + n passes + conservation diagnostics. Called from
  // Particles::SetPrtclTmunu so it runs on EVERY deposit path (per-cycle task,
  // init/restart seed, post-regrid re-deposit). No-op when nfilter_passes == 0.
  void ApplyDigitalFilter(int ncycle, int debug_lvl);

 private:
  void FillTmunuGhosts();              // synchronous CC exchange (driver-init pattern)
  void FilterOnePass();                // Eq. (11) stencil, all 10 components
  void ComputeSourceIntegrals(int slot);  // sum q sqrt(gamma) dV -> filt_sums[slot*10+]
  void ReportFilterDiagnostics(int ncycle, bool full);

  MeshBlockPack* pmy_pack;
};

#endif  // Z4C_TMUNU_HPP_
