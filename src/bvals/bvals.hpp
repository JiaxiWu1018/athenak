#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_cc.hpp
//! \brief defines classes for handling boundary values for all types of variables.
//! Currently methods for cell-centered and face-centered fields implemented.

// identifiers for all 6 faces of a MeshBlock
enum BoundaryFace {undef=-1, inner_x1, outer_x1, inner_x2, outer_x2, inner_x3, outer_x3};

// identifiers for boundary conditions
enum class BoundaryFlag {undef=-1,block, reflect, inflow, outflow, diode, user, periodic};

// identifiers for status of MPI boundary communications
enum class BoundaryCommStatus {undef=-1, waiting, sent, received};

// integer constants to specify variables communicated in MPI calls (maximum of 16 set by
// number of bits used to encode ID in CreateMPItag function in src/utils)
enum VariablesID {FluidCons_ID, BField_ID};

//----------------------------------------------------------------------------------------
//! \fn int CreateMPITag(int lid, int bufid)
//  \brief calculate an MPI tag for boundary buffer communications
//  MPI tag = lid (remaining bits) + bufid (6 bits)
//  Note the convention in Athena++ is lid and bufid are both for the *receiving* process
//
// WARNING (KGF): Generating unsigned integer bitfields from signed integer types and
// converting output to signed integer tags (required by MPI) may lead to unsafe
// conversions (and overflows from built-in types and MPI_TAG_UB).  Note, the MPI standard
// requires signed int tag, with MPI_TAG_UB>= 2^15-1 = 32,767 (inclusive)

static int CreateMPITag(int lid, int bufid)
{
  return (lid<<6) | bufid;
}

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "tasklist/task_list.hpp"

//----------------------------------------------------------------------------------------
//! \struct BufferIndcs
//! \brief indices for range of cells packed/unpacked into boundary buffers

struct BufferIndcs
{
  int bis,bie,bjs,bje,bks,bke;  // start/end buffer ("b") indices in each dir
  int ndat;                     // number of data elements
  BufferIndcs() :
   bis(0), bie(0), bjs(0), bje(0), bks(0), bke(0), ndat(1) {}
};

//----------------------------------------------------------------------------------------
//! \struct BoundaryBuffer
//! \brief container for index ranges, storage, and flags for boundary buffers

struct BoundaryBuffer
{
  // fixed-length-3 arrays used to store indices of each buffer for cell-centered vars, or
  // each component of a face-centered vector field ([0,1,2] --> [x1f, x2f, x3f]). For
  // cell-centered variables only first [0] component of index arrays are needed. 
  BufferIndcs isame[3];  // indices for pack/unpack when dest/src at same level
  BufferIndcs icoar[3];  // indices for pack/unpack when dest/src at coarser level
  BufferIndcs ifine[3];  // indices for pack/unpack when dest/src at finer level
  BufferIndcs iprol[3];  // indices for prolongation (only used for receives)
  BufferIndcs iflux[3];  // indices for pack/unpack for flux correction

  // 3D Views that store buffer data on device
  DvceArray3D<Real> vars, flux;

  // following two 1D arrays only accessed from host, so can use STL vector
  std::vector<BoundaryCommStatus> vars_stat, flux_stat;
#if MPI_PARALLEL_ENABLED
  std::vector<MPI_Request> vars_req, flux_req;
#endif

  // function to allocate memory for buffers for variables and their fluxes
  void AllocateBuffers(int nmb, int nvar) {
    int nmax = 0;
    for (int i=0; i<=2; ++i) {
      nmax = std::max(nmax, isame[i].ndat);
      nmax = std::max(nmax, icoar[i].ndat);
      nmax = std::max(nmax, ifine[i].ndat);
    }
    Kokkos::realloc(vars, nmb, nvar, nmax);
    nmax = 0;
    for (int i=0; i<=2; ++i) {
      nmax = std::max(nmax, iflux[i].ndat);
    }
    Kokkos::realloc(flux, nmb, nvar, nmax);
  }
};

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class BoundaryValues
//  \brief Abstract base class for boundary values for different kinds of variables

class BoundaryValues
{
public:
  BoundaryValues(MeshBlockPack *ppack, ParameterInput *pin);

  // data for all 56 buffers in most general 3D case. Not all elements used in most cases.
  // However each BoundaryBuffer is lightweight, so the convenience of fixed array
  // sizes and index values for array elements outweighs cost of extra memory. 
  BoundaryBuffer send_buf[56], recv_buf[56];

  // constant inflow states at each face
  DvceArray2D<Real> u_in, b_in;
  
  //functions
  virtual void InitSendIndices(BoundaryBuffer &buf, int x, int y, int z, int a, int b)=0;
  virtual void InitRecvIndices(BoundaryBuffer &buf, int x, int y, int z, int a, int b)=0;
  void InitializeBuffers(const int nvar);
  TaskStatus InitRecv(int nvar);
  TaskStatus ClearRecv();
  TaskStatus ClearSend();

  // BCs associated with various physics modules
  static void HydroBCs(MeshBlockPack *pp, DvceArray2D<Real> uin, DvceArray5D<Real> u0);
  static void BFieldBCs(MeshBlockPack *pp, DvceArray2D<Real> uin, DvceFaceFld4D<Real> b0);

protected:
  MeshBlockPack* pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class BoundaryValuesCC
//  \brief boundary values for cell-centered variables

class BoundaryValuesCC : public BoundaryValues
{
public:
  BoundaryValuesCC(MeshBlockPack *ppack, ParameterInput *pin);

  //functions
  void InitSendIndices(BoundaryBuffer &buf, int o1, int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(BoundaryBuffer &buf, int o1, int o2,int o3,int f1,int f2) override;
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a, DvceArray5D<Real> &c, int key);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, DvceArray5D<Real> &c);
  void ProlongCC(DvceArray5D<Real> &a, DvceArray5D<Real> &c);
  TaskStatus PackAndSendFluxCC(DvceFaceFld5D<Real> &flx);
  TaskStatus RecvAndUnpackFluxCC(DvceFaceFld5D<Real> &flx);
};

//----------------------------------------------------------------------------------------
//! \class BoundaryValuesFC
//  \brief boundary values for face-centered vector fields

class BoundaryValuesFC : public BoundaryValues
{
public:
  BoundaryValuesFC(MeshBlockPack *ppack, ParameterInput *pin);

  //functions
  void InitSendIndices(BoundaryBuffer &buf, int o1, int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(BoundaryBuffer &buf, int o1, int o2,int o3,int f1,int f2) override;
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &c, int key);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &c);
  void ProlongFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &c);
  TaskStatus PackAndSendFluxFC(DvceEdgeFld4D<Real> &flx);
  TaskStatus RecvAndUnpackFluxFC(DvceEdgeFld4D<Real> &flx);
};

#endif // BVALS_BVALS_HPP_
