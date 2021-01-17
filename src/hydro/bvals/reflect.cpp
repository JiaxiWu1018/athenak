//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reflect.cpp
//  \brief implementation of reflecting BCs for Hydro conserved vars in each dimension

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void Hydro::ReflectInnerX1(
//  \brief REFLECTING boundary conditions, inner x1 boundary

void Hydro::ReflectInnerX1()
{
  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
  int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;
  int &is = ncells.is;
  int nvar = nhydro + nscalars;
  int nmb = pmy_pack->nmb_thispack;
  auto &u0_ = u0;

  // copy hydro variables into ghost zones, reflecting v1
  par_for("reflect_ix1", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    {
      if (n == (hydro::IVX)) {  // reflect 1-velocity
        u0_(m,n,k,j,is-i-1) = -u0_(m,n,k,j,is+i);
      } else {
        u0_(m,n,k,j,is-i-1) =  u0_(m,n,k,j,is+i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::ReflectOuterX1(
//  \brief REFLECTING boundary conditions, outer x1 boundary

void Hydro::ReflectOuterX1()
{
  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
  int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;
  int &ie = ncells.ie;
  int nvar = nhydro + nscalars;
  int nmb = pmy_pack->nmb_thispack;
  auto &u0_ = u0;

  // copy hydro variables into ghost zones, reflecting v1
  par_for("reflect_ox1", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    {
      if (n == (hydro::IVX)) {  // reflect 1-velocity
        u0_(m,n,k,j,ie+i+1) = -u0_(m,n,k,j,ie-i);
      } else {
        u0_(m,n,k,j,ie+i+1) =  u0_(m,n,k,j,ie-i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::ReflectInnerX2(
//  \brief REFLECTING boundary conditions, inner x2 boundary

void Hydro::ReflectInnerX2()
{
  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n1 = ncells.nx1 + 2*ng;
  int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;
  int &js = ncells.js;
  int nvar = nhydro + nscalars;
  int nmb = pmy_pack->nmb_thispack;
  auto &u0_ = u0;

  // copy hydro variables into ghost zones, reflecting v2
  par_for("reflect_ix2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    { 
      if (n == (hydro::IVY)) {  // reflect 2-velocity
        u0_(m,n,k,js-j-1,i) = -u0_(m,n,k,js+j,i);
      } else {
        u0_(m,n,k,js-j-1,i) =  u0_(m,n,k,js+j,i);
      }
    }
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::ReflectOuterX2(
//  \brief REFLECTING boundary conditions, outer x2 boundary

void Hydro::ReflectOuterX2()
{
  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n1 = ncells.nx1 + 2*ng;
  int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;
  int &je = ncells.je;
  int nvar = nhydro + nscalars;
  int nmb = pmy_pack->nmb_thispack;
  auto &u0_ = u0;

  // copy hydro variables into ghost zones, reflecting v2
  par_for("reflect_ox2", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    { 
      if (n == (hydro::IVY)) {  // reflect 2-velocity
        u0_(m,n,k,je+j+1,i) = -u0_(m,n,k,je-j,i);
      } else {
        u0_(m,n,k,je+j+1,i) =  u0_(m,n,k,je-j,i);
      }   
    }   
  );

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Hydro::ReflectInnerX3(
//  \brief REFLECTING boundary conditions, inner x3 boundary

void Hydro::ReflectInnerX3()
{
  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n1 = ncells.nx1 + 2*ng;
  int n2 = ncells.nx2 + 2*ng;
  int &ks = ncells.ks;
  int nvar = nhydro + nscalars;
  int nmb = pmy_pack->nmb_thispack;
  auto &u0_ = u0;

  // copy hydro variables into ghost zones, reflecting v3
  par_for("reflect_ix3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    { 
      if (n == (hydro::IVZ)) {  // reflect 3-velocity
        u0_(m,n,ks-k-1,j,i) = -u0_(m,n,ks+k,j,i);
      } else {
        u0_(m,n,ks-k-1,j,i) =  u0_(m,n,ks+k,j,i);
      }   
    }   
  );

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::ReflectOuterX3(
//  \brief REFLECTING boundary conditions, outer x3 boundary

void Hydro::ReflectOuterX3()
{
  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n1 = ncells.nx1 + 2*ng;
  int n2 = ncells.nx2 + 2*ng;
  int &ke = ncells.ke;
  int nvar = nhydro + nscalars;
  int nmb = pmy_pack->nmb_thispack;
  auto &u0_ = u0;

  // copy hydro variables into ghost zones, reflecting v3
  par_for("reflect_ox3", DevExeSpace(),0,(nmb-1),0,(nvar-1),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i)
    {   
      if (n == (hydro::IVZ)) {  // reflect 3-velocity
        u0_(m,n,ke+k+1,j,i) = -u0_(m,n,ke-k,j,i);
      } else {
        u0_(m,n,ke+k+1,j,i) =  u0_(m,n,ke-k,j,i);
      }
    }
  );

  return;
}
} // namespace hydro
