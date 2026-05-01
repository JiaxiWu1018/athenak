//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file Kerr_Wald_ptcl.cpp
//! \brief Problem generator for particle pusher tests in the Kerr spactime with a Wald
//!     solution for the magnetic field.
//!
//! REFERENCE: Fabio Bacchini+2019

#include <math.h>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "particles/particles.hpp"
#include "particles/lagrange_interp.hpp"
#include "particles/calc_tetrad.hpp"

template<int NG>
KOKKOS_INLINE_FUNCTION
void GetADMVariables(Real &alp, Real *beta, Real *g3d, const Real *x_mid,
                     const int mb, const Real *mb_par, const int *ncell,
                     const DvceArray5D<Real> &adm_n, const DvceArray5D<Real> &adm_nm1);

KOKKOS_INLINE_FUNCTION
void GetBLCoord(Real spin, Real x1, Real x2, Real x3, Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
void CalcWaldVecPot(Real spin, Real Bz, Real r, Real theta, Real phi, Real *paphi);

KOKKOS_INLINE_FUNCTION
Real A1(Real spin, Real Bz, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(Real spin, Real Bz, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(Real spin, Real Bz, Real x1, Real x2, Real x3);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  Real Bz = pin->GetReal("problem", "Bz");

  auto &coord = pmbp->pcoord->coord_data;
  Real spin = coord.bh_spin;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  // Set primitive variables
  auto &w0_ = pmbp->pmhd->w0;
  Real dfloor = pmbp->pmhd->peos->eos_data.dfloor;
  Real pfloor = pmbp->pmhd->peos->eos_data.pfloor;
  par_for("pgen_w0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      w0_(m, IDN, k, j, i) = dfloor;
      w0_(m, IVX, k, j, i) = 0.0;
      w0_(m, IVY, k, j, i) = 0.0;
      w0_(m, IVZ, k, j, i) = 0.0;
      w0_(m, IEN, k, j, i) = pfloor;
    });
  Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->w0_last, w0_);

  // Set magnetic field
  // compute vector potential over all faces
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

  par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    a1(m,k,j,i) = A1(spin, Bz, x1v, x2f, x3f);
    a2(m,k,j,i) = A2(spin, Bz, x1f, x2v, x3f);
    a3(m,k,j,i) = A3(spin, Bz, x1f, x2f, x3v);
  });

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute face-centered fields from curl(A).
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                        (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                        (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                        (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                            (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                            (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                            (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
    }
  });

  // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
  par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc_(m,IBX,k,j,i);
    Real& w_by = bcc_(m,IBY,k,j,i);
    Real& w_bz = bcc_(m,IBZ,k,j,i);
    w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });
  Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->bcc0_last, bcc_);

  // Set spacetime and perform p2c
  pmbp->padm->SetADMVariables(pmbp);
  auto &adm = pmbp->padm->u_adm;
  Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, adm);
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);

  return;
}

KOKKOS_INLINE_FUNCTION
void GetBLCoord(Real spin, Real x1, Real x2, Real x3, Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = (sqrt( SQR(rad) - SQR(spin) + sqrt(SQR(SQR(rad)-SQR(spin))
            + 4.0*SQR(spin)*SQR(x3)) ) / sqrt(2.0));
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2-spin*x1, spin*x2+r*x1) -
          spin*r/(SQR(r)-2.0*r+SQR(spin));
  return;
}

KOKKOS_INLINE_FUNCTION
void CalcWaldVecPot(Real spin, Real Bz, Real r, Real theta, Real phi, Real *paphi) {
  Real sin_theta = sin(theta);
  Real cos_theta = cos(theta);
  Real Sigma = r * r + spin * spin * cos_theta * cos_theta;
  *paphi = Bz * sin_theta * sin_theta * ((r * r + spin * spin) * 0.5 -
                                         spin * spin * r * (1. + cos_theta * cos_theta) / Sigma);
  return;
}

KOKKOS_INLINE_FUNCTION
Real A1(Real spin, Real Bz, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBLCoord(spin, x1, x2, x3, &r, &theta, &phi);
  Real aphi;
  CalcWaldVecPot(spin, Bz, r, theta, phi, &aphi);
  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(spin);
  Real isin_term = sqrt((SQR(spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));
  return aphi*(-x2/(SQR(x1)+SQR(x2))+spin*x1*r/((SQR(spin)+SQR(r))*sqrt_term));
}

KOKKOS_INLINE_FUNCTION
Real A2(Real spin, Real Bz, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBLCoord(spin, x1, x2, x3, &r, &theta, &phi);
  Real aphi;
  CalcWaldVecPot(spin, Bz, r, theta, phi, &aphi);
  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(spin);
  Real isin_term = sqrt((SQR(spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));
  return aphi*(x1/(SQR(x1)+SQR(x2))+spin*x2*r/((SQR(spin)+SQR(r))*sqrt_term));
}

KOKKOS_INLINE_FUNCTION
Real A3(Real spin, Real Bz, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBLCoord(spin, x1, x2, x3, &r, &theta, &phi);
  Real aphi;
  CalcWaldVecPot(spin, Bz, r, theta, phi, &aphi);
  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(spin);
  Real isin_term = sqrt((SQR(spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));
  return aphi*(spin*x3/(r*sqrt_term));
}