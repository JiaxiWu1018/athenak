//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file track_prtcl.cpp
//! \brief writes data for tracked particles in unformatted binary

#include <sys/stat.h>  // mkdir
#include <vector>

#include <algorithm>
#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <array>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor

TrackedParticleOutput::TrackedParticleOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create new directory for this output. Comments in binary.cpp constructor explain why
  mkdir("trk",0775);
  // allocate arrays
  npout_eachrank.resize(global_variable::nranks);
  ntrack = pin->GetInteger(op.block_name,"nparticles");
  // TODO(@user) improve guess below?
  ntrack_thisrank = ntrack;
  format_version = pin->GetOrAddInteger(op.block_name, "format_version", 1);
  component_count = pin->GetOrAddInteger(op.block_name, "cohort_per_component", 0);
  if (format_version != 1 && format_version != 2)
    throw std::runtime_error("trk format_version must be 1 or 2");
  if (component_count < 0 || component_count > 1000000 ||
      (component_count && format_version != 2))
    throw std::runtime_error("invalid component cohort request");
}

//----------------------------------------------------------------------------------------
// TrackedParticleOutput::LoadOutputData()
// Copies data for tracked particles on this rank to host outpart array

void TrackedParticleOutput::LoadOutputData(Mesh *pm) {
  const int npart = pm->nprtcl_thisrank;
  auto pr = pm->pmb_pack->ppart->prtcl_rdata;
  auto pi = pm->pmb_pack->ppart->prtcl_idata;
  const int limit = ntrack;
  const int per_component = component_count;
  // Count and prefix-pack using a device scan; no host pointer enters a GPU kernel.
  auto selected = KOKKOS_LAMBDA(int tag) {
    if (!per_component) return tag >= 0 && tag < limit;
    if (tag < 0 || tag >= 5000000) return false;
    const int base = tag < 3000000 ? 0 : (tag < 4000000 ? 3000000 : 4000000);
    const int size = base == 0 ? 3000000 : 1000000;
    // 104729 is coprime to both component sizes: exactly per_component unique tags.
    return (static_cast<int64_t>(tag-base)*104729 + 4001)%size < per_component;
  };
  Kokkos::parallel_reduce("trk_count", Kokkos::RangePolicy<DevExeSpace>(0,npart),
      KOKKOS_LAMBDA(int p, int &sum) {if (selected(pi(PTAG,p))) ++sum;}, npout);
  DualArray1D<TrackedParticleData> packed("tracked_particles", npout);
  Kokkos::parallel_scan("trk_pack", Kokkos::RangePolicy<DevExeSpace>(0,npart),
      KOKKOS_LAMBDA(int p, int &offset, bool final) {
        if (selected(pi(PTAG,p))) {
          if (final) {
            auto &r = packed.d_view(offset);
            r.tag=pi(PTAG,p); r.x=pr(IPX,p); r.y=pr(IPY,p); r.z=pr(IPZ,p);
            r.vx=pr(IPVX,p); r.vy=pr(IPVY,p); r.vz=pr(IPVZ,p);
            r.mass=pr(IPM,p); r.energy=pr(IPEN,p);
          }
          ++offset;
        }
      });
  packed.template modify<DevExeSpace>(); packed.template sync<HostMemSpace>();
  Kokkos::realloc(outpart, npout);
  Kokkos::deep_copy(outpart, packed.h_view);
  npout_eachrank[global_variable::my_rank] = npout;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&npout,1,MPI_INT,npout_eachrank.data(),1,MPI_INT,MPI_COMM_WORLD);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void TrackedParticleOutput:::WriteOutputFile(Mesh *pm)
//! \brief Cycles over all tracked particles on this rank and writes ouput data
//! With MPI, all particles are written to the same file.

void TrackedParticleOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  if (format_version == 2) {
    // Versioned snapshots: 512-byte ASCII header; int64 tag and eight float64 values.
    // Packing fields explicitly avoids compiler struct padding and preserves IDs.
    std::ostringstream name;
    name << "trk/" << out_params.file_basename << ".c" << std::setw(10)
         << std::setfill('0') << pm->ncycle << ".trk2";
    IOWrapper file;
    file.Open(name.str().c_str(), IOWrapper::FileMode::write);
    int64_t total=0, prefix=0;
    for (int r=0; r<global_variable::nranks; ++r) {
      if (r<global_variable::my_rank) prefix+=npout_eachrank[r];
      total+=npout_eachrank[r];
    }
    std::array<char,512> header{};
    std::snprintf(header.data(),header.size(),
      "ATHENAK_TRK2 version=2 endian=native record_bytes=72 count=%lld cycle=%d time=%.17g\n"
      "int64:tag float64:mass,minus_ut,x,y,z,u1_cov,u2_cov,u3_cov\n",
      static_cast<long long>(total),pm->ncycle,pm->time);
    if (global_variable::my_rank==0 &&
        file.Write_any_type_at(header.data(),header.size(),0,"byte")!=header.size())
      throw std::runtime_error("trk2 header write failed");
    std::vector<unsigned char> records(static_cast<size_t>(npout)*72);
    for (int p=0; p<npout; ++p) {
      int64_t tag=outpart(p).tag;
      const double values[8]={outpart(p).mass,outpart(p).energy,outpart(p).x,
          outpart(p).y,outpart(p).z,outpart(p).vx,outpart(p).vy,outpart(p).vz};
      std::memcpy(records.data()+72*p,&tag,8);
      std::memcpy(records.data()+72*p+8,values,64);
    }
    if (file.Write_any_type_at_all(records.data(),records.size(),512+72*prefix,"byte")
        !=records.size()) throw std::runtime_error("trk2 payload write failed");
    file.Close();
    out_params.last_time = pm->time;
    pin->SetReal(out_params.block_name,"last_time",out_params.last_time);
    return;
  }
  // create filename: "trk/file_basename".trk
  std::string fname;
  fname.assign("trk/");
  fname.append(out_params.file_basename);
  fname.append(".trk");

  // Root process opens/creates file and appends string
  if (global_variable::my_rank == 0) {
    std::stringstream msg;
    msg << std::endl << "# AthenaK tracked particle data at time= " << pm->time
        << "  nranks= " << global_variable::nranks
        << "  cycle=" << pm->ncycle
        << "  ntracked_prtcls=" << ntrack << std::endl;
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
      exit(EXIT_FAILURE);
    }
    std::fprintf(pfile,"%s \n",msg.str().c_str());
    std::fclose(pfile);
  }

  // Now all ranks open file and append data
  IOWrapper partfile;
  partfile.Open(fname.c_str(), IOWrapper::FileMode::append);
  std::size_t header_offset = partfile.GetPosition();
//  std::size_t header_offset = 0;

  // allocate 1D vector of floats used to convert and output particle data
  float *data = new float[6*npout];
  // Loop over particles, load positions into data[]
  for (int p=0; p<npout; ++p) {
    data[ 6*p   ] = static_cast<float>(outpart(p).x);
    data[(6*p)+1] = static_cast<float>(outpart(p).y);
    data[(6*p)+2] = static_cast<float>(outpart(p).z);
    data[(6*p)+3] = static_cast<float>(outpart(p).vx);
    data[(6*p)+4] = static_cast<float>(outpart(p).vy);
    data[(6*p)+5] = static_cast<float>(outpart(p).vz);
  }
  // calculate local data offset
  std::vector<int> rank_offset(global_variable::nranks, 0);
  int npout_min = pm->nprtcl_eachrank[0];
  for (int n=1; n<global_variable::nranks; ++n) {
    rank_offset[n] = rank_offset[n-1] + npout_eachrank[n-1];
    npout_min = std::min(npout_min, npout_eachrank[n]);
  }

  // Write tracked particle data collectively over minimum shared number of prtcls
  for (int p=0; p<npout_min; ++p) {
    // offset computed assuming tags run 0...(ntrack-1) sequentially
    std::size_t myoffset = header_offset + 6*sizeof(float)*outpart(p).tag;
    // Write particle positions collectively for minimum number of particles across ranks
    if (partfile.Write_any_type_at_all(&(data[6*p]),6,myoffset,"float") != 6) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to tracked particle file"
          << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Write particle positions individually for remaining particles on each rank
  for (int p=npout_min; p<npout; ++p) {
    // offset computed assuming tags run 0...(ntrack-1) sequentially
    std::size_t myoffset = header_offset + 6*sizeof(float)*outpart(p).tag;
    // Write particle positions collectively for minimum number of particles across ranks
    if (partfile.Write_any_type_at(&(data[6*p]),6,myoffset,"float") != 6) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to tracked particle file"
          << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // close the output file and clean up
  partfile.Close();
  delete[] data;

  // increment counters
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
