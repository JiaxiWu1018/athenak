//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_kretschmann.cpp
//! \brief curvature invariants from the 3+1 (ADM) variables and the matter sources.
//!
//! Computes on the MeshBlock interior
//!   u_kretsch(m,0,...) = I  = R_{abcd} R^{abcd}                 (Kretschmann scalar)
//!   u_kretsch(m,1,...) = C^2 = C_{abcd} C^{abcd} = I - 2 R_{ab}R^{ab} + R^2/3   (Weyl^2)
//! from the standard decomposition of the 4D Riemann tensor with respect to the unit
//! normal n^a (the same assembly as z4c_calculate_weyl_scalars.cpp):
//!   A_{abcd} = gamma-projection of (4)R_{abcd}
//!            = (3)R_{abcd} + K_{ac}K_{bd} - K_{ad}K_{bc}                        (Gauss)
//!   B_{abc}  = gamma-projection of n^d (4)R_{dabc} = D_b K_{ac} - D_c K_{ab}  (Codazzi)
//!   C_{ab}   = n^c n^d (4)R_{cadb} = (3)R_{ab} + K K_{ab} - K_{ac}K^c_b
//!              - 8 pi [S_{ab} - (1/2) gamma_{ab} (S - E)]                    (Ricci eq.)
//! so that, using g^{mu nu} = gamma^{mu nu} - n^mu n^nu on every index pair,
//!   I = A_{abcd}A^{abcd} - 4 B_{abc}B^{abc} + 4 C_{ab}C^{ab}.
//! The matter term in C_{ab} comes from the projected 4D Ricci tensor via the Einstein
//! equations (it is what z4c_calculate_weyl_scalars.cpp omits, being vacuum-only);
//! R_{ab}R^{ab} = 64 pi^2 T_{ab}T^{ab} and R = -8 pi T with
//!   T_{ab}T^{ab} = E^2 - 2 S_a S^a + S_{ab}S^{ab},   T = -E + S.
//! In vacuum I = C^2 = 8 (E_ij E^ij - B_ij B^ij) and both slots agree. The kernel is
//! meant for diagnostics at output/history cadence (it is as expensive as the Weyl
//! kernel), not for every cycle.

#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"

namespace z4c {

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CalcKretschmann(MeshBlockPack *pmbp)
//! \brief NGHOST dispatcher for Z4cKretschmann (fills u_kretsch on the interior)
void Z4c::CalcKretschmann(MeshBlockPack *pmbp) {
  switch (pmbp->pmesh->mb_indcs.ng) {
    case 2: Z4cKretschmann<2>(pmbp); break;
    case 3: Z4cKretschmann<3>(pmbp); break;
    case 4: Z4cKretschmann<4>(pmbp); break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::Z4cKretschmann(MeshBlockPack *pmbp)
//! \brief compute the Kretschmann and Weyl-squared invariants (interior points only)
template <int NGHOST>
void Z4c::Z4cKretschmann(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  auto &adm = pmbp->padm->adm;
  auto &u_k = pmbp->pz4c->u_kretsch;
  Kokkos::deep_copy(u_k, 0.);

  const bool is_vacuum = (pmbp->ptmunu == nullptr);
  Tmunu::Tmunu_vars tmunu;
  if (!is_vacuum) {
    tmunu = pmbp->ptmunu->tmunu;
  }

  par_for("z4c_kretschmann", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real detg = 0.0;
    Real R = 0.0;
    Real K = 0.0;

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> K_ud;
    AthenaPointTensor<Real, TensorSymm::SYM2,  3, 3> dg_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2,  3, 3> dK_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      g_uu(a,b) = 0.0;
      R_dd(a,b) = 0.0;
      for (int c = 0; c < 3; ++c) {
        dg_ddd(c,a,b) = 0.0;
        dK_ddd(c,a,b) = 0.0;
        Gamma_ddd(c,a,b) = 0.0;
        Gamma_udd(c,a,b) = 0.0;
        DK_ddd(c,a,b) = 0.0;
        for (int d = c; d < 3; ++d) {
          ddg_dddd(c,d,a,b) = 0.0;
        }
      }
    }
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> C_dd;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 3> B_ddd;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 4> A_dddd;
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      C_dd(a,b) = 0.0;
      for (int c = 0; c < 3; ++c) {
        B_ddd(a,b,c) = 0.0;
        for (int d = 0; d < 3; ++d) {
          A_dddd(a,b,c,d) = 0.0;
        }
      }
    }

    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    // derivatives of g and K (same stencils as the Weyl kernel)
    for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      dg_ddd(c,a,b) = Dx<NGHOST>(c, idx, adm.g_dd, m,a,b,k,j,i);
      dK_ddd(c,a,b) = Dx<NGHOST>(c, idx, adm.vK_dd, m,a,b,k,j,i);
    }
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    for (int d = c; d < 3; ++d) {
      if (a == b) {
        ddg_dddd(a,b,c,d) = Dxx<NGHOST>(a, idx, adm.g_dd, m,c,d,k,j,i);
      } else {
        ddg_dddd(a,b,c,d) = Dxy<NGHOST>(a, b, idx, adm.g_dd, m,c,d,k,j,i);
      }
    }

    // inverse metric
    detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1.0/detg,
                adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
                &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));

    // Christoffel symbols
    for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      Gamma_ddd(c,a,b) = 0.5*(dg_ddd(a,b,c) + dg_ddd(b,a,c) - dg_ddd(c,a,b));
    }
    for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
    for (int d = 0; d < 3; ++d) {
      Gamma_udd(c,a,b) += g_uu(c,d)*Gamma_ddd(d,a,b);
    }

    // Ricci tensor and scalar
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      for (int c = 0; c < 3; ++c)
      for (int d = 0; d < 3; ++d) {
        for (int e = 0; e < 3; ++e) {
          R_dd(a,b) += g_uu(c,d) * Gamma_udd(e,a,c) * Gamma_ddd(e,b,d);
          R_dd(a,b) -= g_uu(c,d) * Gamma_udd(e,a,b) * Gamma_ddd(e,c,d);
        }
        R_dd(a,b) += 0.5*g_uu(c,d)*(
            - ddg_dddd(c,d,a,b) - ddg_dddd(a,b,c,d) +
              ddg_dddd(a,c,b,d) + ddg_dddd(b,c,a,d));
      }
    }
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      R += g_uu(a,b) * R_dd(a,b);
    }

    // extrinsic curvature: mixed form, trace, covariant derivative
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        K_ud(a,b) = 0.0;
        for (int c = 0; c < 3; ++c) {
          K_ud(a,b) += g_uu(a,c) * adm.vK_dd(m,c,b,k,j,i);
        }
      }
      K += K_ud(a,a);
    }
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) {
      DK_ddd(a,b,c) = dK_ddd(a,b,c);
      for (int d = 0; d < 3; ++d) {
        DK_ddd(a,b,c) -= Gamma_udd(d,a,b) * adm.vK_dd(m,d,c,k,j,i);
        DK_ddd(a,b,c) -= Gamma_udd(d,a,c) * adm.vK_dd(m,b,d,k,j,i);
      }
    }

    // matter sources (normal-frame): E, S_a, S_ab, S = gamma^{ab} S_ab
    Real E = 0.0, S = 0.0, SS = 0.0, SaSa = 0.0;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> S_dd;
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      S_dd(a,b) = 0.0;
    }
    if (!is_vacuum) {
      E = tmunu.E(m,k,j,i);
      for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) {
        S_dd(a,b) = tmunu.S_dd(m,a,b,k,j,i);
      }
      for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        S += g_uu(a,b) * S_dd(a,b);
        SaSa += g_uu(a,b) * tmunu.S_d(m,a,k,j,i) * tmunu.S_d(m,b,k,j,i);
        for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d) {
          SS += g_uu(a,c) * g_uu(b,d) * S_dd(a,b) * S_dd(c,d);
        }
      }
    }

    // A_{abcd}: Gauss;  B_{abc}: Codazzi;  C_{ab}: Ricci equation with matter
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d) {
      Real riem3 = adm.g_dd(m,a,c,k,j,i)*R_dd(b,d)
                 + adm.g_dd(m,b,d,k,j,i)*R_dd(a,c)
                 - adm.g_dd(m,a,d,k,j,i)*R_dd(b,c)
                 - adm.g_dd(m,b,c,k,j,i)*R_dd(a,d)
                 - 0.5*R*adm.g_dd(m,a,c,k,j,i)*adm.g_dd(m,b,d,k,j,i)
                 + 0.5*R*adm.g_dd(m,a,d,k,j,i)*adm.g_dd(m,b,c,k,j,i);
      A_dddd(a,b,c,d) = riem3
                      + adm.vK_dd(m,a,c,k,j,i)*adm.vK_dd(m,b,d,k,j,i)
                      - adm.vK_dd(m,a,d,k,j,i)*adm.vK_dd(m,b,c,k,j,i);
    }
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) {
      B_ddd(a,b,c) = DK_ddd(b,a,c) - DK_ddd(c,a,b);
    }
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      C_dd(a,b) = R_dd(a,b) + K*adm.vK_dd(m,a,b,k,j,i);
      for (int c = 0; c < 3; ++c)
      for (int d = 0; d < 3; ++d) {
        C_dd(a,b) -= g_uu(c,d)*adm.vK_dd(m,a,c,k,j,i)*adm.vK_dd(m,d,b,k,j,i);
      }
      if (!is_vacuum) {
        C_dd(a,b) -= 8.0*M_PI*(S_dd(a,b) - 0.5*adm.g_dd(m,a,b,k,j,i)*(S - E));
      }
    }

    // full contractions with the inverse spatial metric
    Real CC = 0.0;
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d) {
      CC += g_uu(a,c) * g_uu(b,d) * C_dd(a,b) * C_dd(c,d);
    }
    Real BB = 0.0;
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) {
      // raise the first two indices, contract the third pairwise
      Real Bup_ab_c = 0.0;  // B^{ab}_c
      for (int e = 0; e < 3; ++e)
      for (int f = 0; f < 3; ++f) {
        Bup_ab_c += g_uu(a,e) * g_uu(b,f) * B_ddd(e,f,c);
      }
      for (int g = 0; g < 3; ++g) {
        BB += Bup_ab_c * g_uu(c,g) * B_ddd(a,b,g);
      }
    }
    Real AA = 0.0;
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d) {
      // A^{abcd} built by four successive raisings, then dotted with A_{abcd}
      Real Aup = 0.0;
      for (int e = 0; e < 3; ++e) {
        Real t1 = 0.0;
        for (int f = 0; f < 3; ++f) {
          Real t2 = 0.0;
          for (int g = 0; g < 3; ++g) {
            Real t3 = 0.0;
            for (int h = 0; h < 3; ++h) {
              t3 += g_uu(d,h) * A_dddd(e,f,g,h);
            }
            t2 += g_uu(c,g) * t3;
          }
          t1 += g_uu(b,f) * t2;
        }
        Aup += g_uu(a,e) * t1;
      }
      AA += Aup * A_dddd(a,b,c,d);
    }

    Real I = AA - 4.0*BB + 4.0*CC;
    // 4D Ricci contributions via the Einstein equations
    Real TT = E*E - 2.0*SaSa + SS;     // T_{ab}T^{ab}
    Real Ttr = -E + S;                 // T
    Real RicSq = 64.0*M_PI*M_PI*TT;    // R_{ab}R^{ab}
    Real R4 = -8.0*M_PI*Ttr;           // R
    u_k(m,0,k,j,i) = I;
    u_k(m,1,k,j,i) = I - 2.0*RicSq + R4*R4/3.0;
  });
}

template void Z4c::Z4cKretschmann<2>(MeshBlockPack *pmbp);
template void Z4c::Z4cKretschmann<3>(MeshBlockPack *pmbp);
template void Z4c::Z4cKretschmann<4>(MeshBlockPack *pmbp);

}  // namespace z4c
