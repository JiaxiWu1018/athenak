//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_pic_spheroid.cpp
//! \brief Shapiro-Teukolsky prolate collisionless spheroid (NRPIC): a homogeneous-family
//! prolate spheroid of collisionless particles, momentarily at rest on a conformally-flat
//! time-symmetric slice, evolved with the Z4c moving-puncture gauge. The classic naked-
//! singularity candidate (Shapiro & Teukolsky PRL 66, 994 (1991); East PRL 122, 231103
//! (2019)), M=1, e=0.9, b/M=10.
//!
//! Initial data follow naked_singularity_initial_data_moving_puncture.pdf:
//!   gamma_ij = Psi^4 delta_ij, K_ij = 0, u_i = 0 (W = 1), alpha = Psi^-2 (pre-collapsed
//!   lapse; sph_precollapsed_lapse=false gives alpha = 1), beta^i = 0, B^i = 0,
//!   and the Hamiltonian constraint  Lap Psi = -(2 pi/Psi) sum_p m_p W_p(x)
//! solved for the ACTUAL discrete CIC-deposited particle realization (not the continuum
//! Psi_bar) by scripts/particles/spheroid_id.py, which writes a single binary file
//! (<problem> sph_id_file) containing
//!   * N equal-mass particles (positions; symmetric realization: z->-z and antipodal
//!     (x,y)->(-x,-y) images, so the deposited centre of mass vanishes exactly),
//!   * Psi on the uniform fine table that coincides with the finest initial refinement
//!     level (cell centres xmin + (i+1/2) h),
//!   * the converged source list s_c dV (s = sigma/Psi) on the same fine cells.
//! Psi at every mesh cell (all levels, incl. ghosts) is then
//!   * the table value where the cell centre is a fine-table node (exact transfer),
//!   * otherwise 1 + (1/2) sum_c s_c dV K(|x - x_c|) with the SAME discrete kernel the
//!     solver used (K = 1/r, K(0) = 2.3800774/h: cube-averaged self term), evaluated with
//!     the fine sources within sph_near_margin of the table box and with coarse
//!     centre-of-mass aggregates (sph_coarse_dx) further out (aggregation error ~1e-8).
//! sph_psi_mode = analytic instead uses the continuum Psi_bar = 1 - Phi_N (interior
//! Eq. (13) of the PDF; exterior from the ellipsoidal-coordinate closed form) with the
//! same particles -- the "continuum + noisy particles" comparison baseline.
//!
//! A user history function (<problem> user_hist = true) records rho_max, alpha_min,
//! max |Kretschmann| with their locations, rho and alpha at the curvature maximum, the
//! alive particle count/mass, cumulative lapse/other destructions, the MeshBlock count
//! and the maximum physical refinement level. Adaptive refinement is delegated to
//! <z4c_amr> (Loehner) through user_ref_func.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "z4c/tmunu.hpp"
#include "particles/particles.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

void SpheroidRefinementCondition(MeshBlockPack *pmbp);
void SpheroidHistory(HistoryData *pdata, Mesh *pm);

namespace {

//----------------------------------------------------------------------------------------
// continuum reference (homogeneous prolate spheroid, G = 1, rho_N = 3M/(8 pi a^2 b))
struct SphParams {
  Real M, e, b, a, beta0, eps;   // eps = sqrt(b^2 - a^2) = b e
};

KOKKOS_INLINE_FUNCTION
Real PhiInterior(const SphParams &s, Real x, Real y, Real z) {
  Real w2 = x*x + y*y;
  Real b3e3 = s.b*s.b*s.b*s.e*s.e*s.e;
  return -3.0*s.M*s.beta0/(4.0*s.b*s.e)
         + 3.0*s.M/(8.0*b3e3)*(s.e/(1.0 - s.e*s.e) - s.beta0)*w2
         + 3.0*s.M/(4.0*b3e3)*(s.beta0 - s.e)*z*z;
}

KOKKOS_INLINE_FUNCTION
Real PhiExterior(const SphParams &s, Real x, Real y, Real z) {
  Real a2 = s.a*s.a, b2 = s.b*s.b;
  Real w2 = x*x + y*y, z2 = z*z;
  // largest root of w2/(a2+l) + z2/(b2+l) = 1
  Real B = a2 + b2 - w2 - z2;
  Real C = a2*b2 - w2*b2 - z2*a2;
  Real disc = sqrt(fmax(B*B - 4.0*C, 0.0));
  Real lam = (B > 0.0) ? (-2.0*C/(B + disc)) : 0.5*(-B + disc);
  lam = fmax(lam, 0.0);
  Real sq = sqrt(b2 + lam);
  Real at = atanh(s.eps/sq);
  Real e2 = s.eps*s.eps, e3 = e2*s.eps;
  Real I  = 2.0*at/s.eps;
  Real A1 = sq/(e2*(a2 + lam)) - at/e3;
  Real A3 = 2.0*at/e3 - 2.0/(e2*sq);
  return -(3.0*s.M/8.0)*(I - w2*A1 - z2*A3);
}

KOKKOS_INLINE_FUNCTION
Real PsiAnalytic(const SphParams &s, Real x, Real y, Real z) {
  bool inside = ((x*x + y*y)/(s.a*s.a) + z*z/(s.b*s.b) <= 1.0);
  return 1.0 - (inside ? PhiInterior(s, x, y, z) : PhiExterior(s, x, y, z));
}

//----------------------------------------------------------------------------------------
// discrete Newtonian-type convolution u(x) = (1/2) sum_c w_c K(|x - x_c|)
KOKKOS_INLINE_FUNCTION
Real DirectSum(Real x, Real y, Real z, const DvceArray1D<Real> &sx,
               const DvceArray1D<Real> &sy, const DvceArray1D<Real> &sz,
               const DvceArray1D<Real> &sw, int n, Real kself) {
  Real u = 0.0;
  for (int c = 0; c < n; ++c) {
    Real dx = x - sx(c), dy = y - sy(c), dz = z - sz(c);
    Real r2 = dx*dx + dy*dy + dz*dz;
    u += sw(c) * ((r2 > 0.0) ? 1.0/sqrt(r2) : kself);
  }
  return 0.5*u;
}

// host mirror of the same sum (verification prints)
Real DirectSumHost(Real x, Real y, Real z, const std::vector<Real> &sx,
                   const std::vector<Real> &sy, const std::vector<Real> &sz,
                   const std::vector<Real> &sw, Real kself) {
  Real u = 0.0;
  for (size_t c = 0; c < sw.size(); ++c) {
    Real dx = x - sx[c], dy = y - sy[c], dz = z - sz[c];
    Real r2 = dx*dx + dy*dy + dz*dz;
    u += sw[c] * ((r2 > 0.0) ? 1.0/std::sqrt(r2) : kself);
  }
  return 0.5*u;
}

//----------------------------------------------------------------------------------------
// initial-data file (written by spheroid_id.py; little-endian native doubles/int64)
struct IDFile {
  int64_t version, N, fold, nx, ny, nz, nsrc, seed;
  Real M, e, b, a, sum_m, mp, M_adm, psi_center, xmin, ymin, zmin, h, cube_self;
  std::vector<Real> px, py, pz, pm, tab, sx, sy, sz, sw;
};

bool ReadIDFile(const std::string &fname, IDFile &f) {
  FILE *fp = std::fopen(fname.c_str(), "rb");
  if (fp == nullptr) {
    return false;
  }
  char magic[8];
  if (std::fread(magic, 1, 8, fp) != 8 || std::memcmp(magic, "NRPICSPH", 8) != 0) {
    std::fclose(fp); return false;
  }
  int64_t ih[8];
  double dh[13];
  bool hdr_ok = (std::fread(ih, sizeof(int64_t), 8, fp) == 8)
             && (std::fread(dh, sizeof(double), 13, fp) == 13);
  if (!hdr_ok) {
    std::fclose(fp);
    return false;
  }
  f.version = ih[0]; f.N = ih[1]; f.fold = ih[2]; f.nx = ih[3]; f.ny = ih[4];
  f.nz = ih[5]; f.nsrc = ih[6]; f.seed = ih[7];
  f.M = dh[0]; f.e = dh[1]; f.b = dh[2]; f.a = dh[3]; f.sum_m = dh[4]; f.mp = dh[5];
  f.M_adm = dh[6]; f.psi_center = dh[7]; f.xmin = dh[8]; f.ymin = dh[9]; f.zmin = dh[10];
  f.h = dh[11]; f.cube_self = dh[12];
  if (f.version != 1 || f.N <= 0 || f.nx <= 0 || f.ny <= 0 || f.nz <= 0 || f.nsrc <= 0) {
    std::fclose(fp); return false;
  }
  auto rd = [&](std::vector<Real> &v, int64_t n) {
    std::vector<double> tmp(n);
    if (std::fread(tmp.data(), sizeof(double), n, fp) != static_cast<size_t>(n)) {
      return false;
    }
    v.assign(tmp.begin(), tmp.end());
    return true;
  };
  bool ok = rd(f.px, f.N) && rd(f.py, f.N) && rd(f.pz, f.N) && rd(f.pm, f.N)
         && rd(f.tab, f.nx*f.ny*f.nz)
         && rd(f.sx, f.nsrc) && rd(f.sy, f.nsrc) && rd(f.sz, f.nsrc) && rd(f.sw, f.nsrc);
  std::fclose(fp);
  return ok;
}

// centre-of-mass aggregation of the fine source list onto cubic cells of size hc
void Aggregate(const IDFile &f, Real hc, std::vector<Real> &cx, std::vector<Real> &cy,
               std::vector<Real> &cz, std::vector<Real> &cw) {
  Real x0 = f.xmin, y0 = f.ymin, z0 = f.zmin;
  int64_t nx = static_cast<int64_t>(std::ceil(f.nx*f.h/hc)) + 1;
  int64_t ny = static_cast<int64_t>(std::ceil(f.ny*f.h/hc)) + 1;
  int64_t nz = static_cast<int64_t>(std::ceil(f.nz*f.h/hc)) + 1;
  std::vector<Real> w(nx*ny*nz, 0.0), wx(nx*ny*nz, 0.0), wy(nx*ny*nz, 0.0),
                    wz(nx*ny*nz, 0.0);
  for (int64_t c = 0; c < f.nsrc; ++c) {
    int64_t i = static_cast<int64_t>(std::floor((f.sx[c] - x0)/hc));
    int64_t j = static_cast<int64_t>(std::floor((f.sy[c] - y0)/hc));
    int64_t k = static_cast<int64_t>(std::floor((f.sz[c] - z0)/hc));
    i = std::min(std::max<int64_t>(i, 0), nx-1);
    j = std::min(std::max<int64_t>(j, 0), ny-1);
    k = std::min(std::max<int64_t>(k, 0), nz-1);
    int64_t lin = (k*ny + j)*nx + i;
    w[lin] += f.sw[c]; wx[lin] += f.sw[c]*f.sx[c];
    wy[lin] += f.sw[c]*f.sy[c]; wz[lin] += f.sw[c]*f.sz[c];
  }
  cx.clear(); cy.clear(); cz.clear(); cw.clear();
  for (size_t l = 0; l < w.size(); ++l) {
    if (w[l] != 0.0) {
      cw.push_back(w[l]); cx.push_back(wx[l]/w[l]);
      cy.push_back(wy[l]/w[l]); cz.push_back(wz[l]/w[l]);
    }
  }
}

DvceArray1D<Real> ToDevice(const std::string &name, const std::vector<Real> &v) {
  DvceArray1D<Real> d(name, std::max<size_t>(v.size(), 1));
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < v.size(); ++i) {
    h(i) = v[i];
  }
  Kokkos::deep_copy(d, h);
  return d;
}

// staged particle data (host) before the device fill
struct PrtclStage {
  std::vector<Real> x, y, z, mass;
  std::vector<int> gid, tag;
};

// global pick of the rank holding the extreme value; broadcasts its payload
void GlobalPick(Real key, Real *payload, int npay) {
#if MPI_PARALLEL_ENABLED
  struct { double v; int r; } in, out;
  in.v = key; in.r = global_variable::my_rank;
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  std::vector<double> buf(npay);
  for (int n = 0; n < npay; ++n) {
    buf[n] = payload[n];
  }
  MPI_Bcast(buf.data(), npay, MPI_DOUBLE, out.r, MPI_COMM_WORLD);
  for (int n = 0; n < npay; ++n) {
    payload[n] = buf[n];
  }
#endif
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_ref_func = SpheroidRefinementCondition;
  user_hist_func = SpheroidHistory;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "nr_pic_spheroid requires a <z4c> block." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "nr_pic_spheroid requires a <particles> block." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmy_mesh_->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "nr_pic_spheroid is 3D-only." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string fname = pin->GetString("problem", "sph_id_file");
  std::string mode = pin->GetOrAddString("problem", "sph_psi_mode", "table");
  bool precollapsed = pin->GetOrAddBoolean("problem", "sph_precollapsed_lapse", true);
  Real near_margin = pin->GetOrAddReal("problem", "sph_near_margin", 1.0);
  Real coarse_dx = pin->GetOrAddReal("problem", "sph_coarse_dx", 0.5);
  bool do_checks = pin->GetOrAddBoolean("problem", "sph_checks", true);
  bool use_table = (mode.compare("table") == 0);
  if (!use_table && mode.compare("analytic") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem> sph_psi_mode must be table or analytic." << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;

  if (!restart) {
    IDFile f;
    if (!ReadIDFile(fname, f)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "could not read the spheroid initial-data file '" << fname
                << "'"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    SphParams sp;
    sp.M = f.M; sp.e = f.e; sp.b = f.b; sp.a = f.a;
    sp.beta0 = std::atanh(f.e); sp.eps = f.b*f.e;
    Real kself = f.cube_self/f.h;
    Real tx0 = f.xmin, ty0 = f.ymin, tz0 = f.zmin, th = f.h;
    Real tx1 = tx0 + f.nx*th, ty1 = ty0 + f.ny*th, tz1 = tz0 + f.nz*th;
    int tnx = static_cast<int>(f.nx), tny = static_cast<int>(f.ny);
    int tnz = static_cast<int>(f.nz);

    // coarse aggregates for the far zone
    std::vector<Real> cx, cy, cz, cw;
    Aggregate(f, coarse_dx, cx, cy, cz, cw);

    auto d_tab = ToDevice("sph_tab", f.tab);
    auto d_sx = ToDevice("sph_sx", f.sx); auto d_sy = ToDevice("sph_sy", f.sy);
    auto d_sz = ToDevice("sph_sz", f.sz); auto d_sw = ToDevice("sph_sw", f.sw);
    auto d_cx = ToDevice("sph_cx", cx); auto d_cy = ToDevice("sph_cy", cy);
    auto d_cz = ToDevice("sph_cz", cz); auto d_cw = ToDevice("sph_cw", cw);
    int nsrc = static_cast<int>(f.nsrc), ncoarse = static_cast<int>(cw.size());

    // -------- conformally-flat, time-symmetric ADM initial data (incl. ghosts) --------
    auto &size = pmbp->pmb->mb_size;
    int isg = is - indcs.ng, ieg = indcs.ie + indcs.ng;
    int jsg = js - indcs.ng, jeg = indcs.je + indcs.ng;
    int ksg = ks - indcs.ng, keg = indcs.ke + indcs.ng;
    int nmb = pmbp->nmb_thispack;
    adm::ADM::ADM_vars &adm = pmbp->padm->adm;
    z4c::Z4c::Z4c_vars &z4c = pmbp->pz4c->z4c;
    DvceArray1D<int> zone_count("sph_zone_count", 3);
    Kokkos::deep_copy(zone_count, 0);
    par_for("pgen nr_pic_spheroid ID", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg,
            isg, ieg,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real psi;
      if (use_table) {
        // distance from the point to the table box (0 inside)
        Real ddx = fmax(fmax(tx0 - x1v, x1v - tx1), 0.0);
        Real ddy = fmax(fmax(ty0 - x2v, x2v - ty1), 0.0);
        Real ddz = fmax(fmax(tz0 - x3v, x3v - tz1), 0.0);
        Real dist = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        bool done = false;
        if (dist == 0.0) {
          Real fx = (x1v - tx0)/th - 0.5;
          Real fy = (x2v - ty0)/th - 0.5;
          Real fz = (x3v - tz0)/th - 0.5;
          int ii = static_cast<int>(round(fx)), jj = static_cast<int>(round(fy));
          int kk = static_cast<int>(round(fz));
          bool aligned = (fabs(fx - ii) < 1.0e-6 && fabs(fy - jj) < 1.0e-6 &&
                          fabs(fz - kk) < 1.0e-6);
          bool inbox = (ii >= 0 && ii < tnx && jj >= 0 && jj < tny &&
                        kk >= 0 && kk < tnz);
          if (aligned && inbox) {
            psi = d_tab((static_cast<int64_t>(kk)*tny + jj)*tnx + ii);
            done = true;
            Kokkos::atomic_inc(&zone_count(0));
          }
        }
        if (!done) {
          if (dist < near_margin) {
            psi = 1.0 + DirectSum(x1v, x2v, x3v, d_sx, d_sy, d_sz, d_sw, nsrc, kself);
            Kokkos::atomic_inc(&zone_count(1));
          } else {
            psi = 1.0 + DirectSum(x1v, x2v, x3v, d_cx, d_cy, d_cz, d_cw, ncoarse, kself);
            Kokkos::atomic_inc(&zone_count(2));
          }
        }
      } else {
        psi = PsiAnalytic(sp, x1v, x2v, x3v);
      }
      Real psi4 = psi*psi*psi*psi;
      adm.psi4(m,k,j,i) = psi4;
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          adm.g_dd(m,a,b,k,j,i) = psi4 * ((a == b) ? 1.0 : 0.0);
          adm.vK_dd(m,a,b,k,j,i) = 0.0;     // time-symmetric
        }
      }
      z4c.alpha(m,k,j,i) = precollapsed ? 1.0/(psi*psi) : 1.0;
      for (int a = 0; a < 3; ++a) {
        z4c.beta_u(m,a,k,j,i) = 0.0;
        z4c.b_u(m,a,k,j,i) = 0.0;
      }
    });

    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
    }
    pmbp->pz4c->Z4cToADM(pmbp);
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMConstraints<2>(pmbp); break;
      case 3: pmbp->pz4c->ADMConstraints<3>(pmbp); break;
      case 4: pmbp->pz4c->ADMConstraints<4>(pmbp); break;
    }

    // -------- particles from the file (only those inside this rank's blocks) --------
    particles::Particles *ppart = pmbp->ppart;
    std::string init = pin->GetOrAddString("particles", "init", "pgen");
    if (init.compare("pgen") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "nr_pic_spheroid requires <particles> init = pgen "
                << "(particles come from sph_id_file)." << std::endl;
      exit(EXIT_FAILURE);
    }
    PrtclStage st;
    for (int64_t p = 0; p < f.N; ++p) {
      int mb = ppart->FindContainingMeshBlock(f.px[p], f.py[p], f.pz[p]);
      if (mb >= 0) {
        st.x.push_back(f.px[p]); st.y.push_back(f.py[p]); st.z.push_back(f.pz[p]);
        st.mass.push_back(f.pm[p]); st.gid.push_back(pmbp->gids + mb);
        st.tag.push_back(static_cast<int>(p));
      }
    }
    int npart = static_cast<int>(st.x.size());
    Kokkos::realloc(ppart->prtcl_rdata, ppart->nrdata, npart);
    Kokkos::realloc(ppart->prtcl_idata, ppart->nidata, npart);
    auto hr = Kokkos::create_mirror_view(ppart->prtcl_rdata);
    auto hi = Kokkos::create_mirror_view(ppart->prtcl_idata);
    for (int p = 0; p < npart; ++p) {
      hi(PGID,p) = st.gid[p];
      hi(PTAG,p) = st.tag[p];
      hr(IPM,p)  = st.mass[p];
      hr(IPEN,p) = 0.0;
      hr(IPX,p)  = st.x[p];  hr(IPVX,p) = 0.0;   // u_i = 0: at rest w.r.t. the normal
      hr(IPY,p)  = st.y[p];  hr(IPVY,p) = 0.0;
      hr(IPZ,p)  = st.z[p];  hr(IPVZ,p) = 0.0;
    }
    Kokkos::deep_copy(ppart->prtcl_rdata, hr);
    Kokkos::deep_copy(ppart->prtcl_idata, hi);
    ppart->nprtcl_thispack = npart;
    pmy_mesh_->nprtcl_thisrank = npart;
    pmy_mesh_->nprtcl_eachrank[global_variable::my_rank] = npart;
#if MPI_PARALLEL_ENABLED
    MPI_Allgather(&npart, 1, MPI_INT, pmy_mesh_->nprtcl_eachrank, 1, MPI_INT,
                  MPI_COMM_WORLD);
#endif
    pmy_mesh_->nprtcl_total = 0;
    for (int nr = 0; nr < global_variable::nranks; ++nr) {
      pmy_mesh_->nprtcl_total += pmy_mesh_->nprtcl_eachrank[nr];
    }
    if (pmy_mesh_->nprtcl_total != f.N) {
      if (global_variable::my_rank == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "placed " << pmy_mesh_->nprtcl_total << " of " << f.N
                  << " file particles: some lie outside the mesh." << std::endl;
      }
      exit(EXIT_FAILURE);
    }

    // Deposit the particle stress-energy now and recompute the ADM constraints, so the
    // t=0 'con' output and the hst norms describe the matter-sourced Hamiltonian
    // constraint of the discrete realization (not the vacuum residual R alone).
    if (ppart->feedback) {
      (void) ppart->SetPrtclTmunu(nullptr, 1);
      switch (indcs.ng) {
        case 2: pmbp->pz4c->ADMConstraints<2>(pmbp); break;
        case 3: pmbp->pz4c->ADMConstraints<3>(pmbp); break;
        case 4: pmbp->pz4c->ADMConstraints<4>(pmbp); break;
      }
    }

    // -------- banner and consistency checks (rank 0) --------
    auto h_zone = Kokkos::create_mirror_view(zone_count);
    Kokkos::deep_copy(h_zone, zone_count);
    int zc[3] = {h_zone(0), h_zone(1), h_zone(2)};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, zc, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      Real mloc = 0.0;
      for (int p = 0; p < npart; ++p) {
        mloc += st.mass[p];
      }
      std::cout << "nr_pic_spheroid: ID file '" << fname << "' N=" << f.N
                << " fold=" << f.fold
                << " seed=" << f.seed << " M=" << f.M << " e=" << f.e << " b=" << f.b
                << " a=" << f.a << " m_p=" << f.mp << " sum m_p=" << f.sum_m
                << " M_ADM(file)=" << f.M_adm << " Psi_c(table)=" << f.psi_center
                << std::endl;
      std::cout << "nr_pic_spheroid: table " << f.nx << "x" << f.ny << "x" << f.nz
                << " h=" << f.h << " box=[" << tx0 << "," << tx1 << "]x[" << ty0 << ","
                << ty1
                << "]x[" << tz0 << "," << tz1 << "] nsrc=" << f.nsrc << " ncoarse="
                << ncoarse << " (hc=" << coarse_dx << ", near_margin=" << near_margin
                << "); psi_mode=" << mode << " alpha0=" << (precollapsed ? "Psi^-2" : "1")
                << std::endl;
      std::cout << "nr_pic_spheroid: cells filled by table/fine-sum/coarse-sum = "
                << zc[0] << "/" << zc[1] << "/" << zc[2] << "; particles on rank 0: "
                << npart << " (mass " << mloc << "), total " << pmy_mesh_->nprtcl_total
                << std::endl;
      if (do_checks) {
        // (a) table vs fine direct sum at the table node nearest the origin
        int ic = tnx/2, jc = tny/2, kc = tnz/2;
        Real xc = tx0 + (ic + 0.5)*th;
        Real yc = ty0 + (jc + 0.5)*th;
        Real zc_ = tz0 + (kc + 0.5)*th;
        Real ptab = f.tab[(static_cast<int64_t>(kc)*tny + jc)*tnx + ic];
        Real pfine = 1.0 + DirectSumHost(xc, yc, zc_, f.sx, f.sy, f.sz, f.sw, kself);
        Real pana = PsiAnalytic(sp, xc, yc, zc_);
        // (b) fine vs coarse direct sums at points near the box boundary and far away
        Real xf[3] = {tx1 + near_margin, 0.0, 0.0};
        Real pf1 = 1.0 + DirectSumHost(xf[0], xf[1], xf[2], f.sx, f.sy, f.sz, f.sw,
                                       kself);
        Real pc1 = 1.0 + DirectSumHost(xf[0], xf[1], xf[2], cx, cy, cz, cw, kself);
        Real pa1 = PsiAnalytic(sp, xf[0], xf[1], xf[2]);
        Real zf[3] = {0.0, 0.0, tz1 + near_margin};
        Real pf2 = 1.0 + DirectSumHost(zf[0], zf[1], zf[2], f.sx, f.sy, f.sz, f.sw,
                                       kself);
        Real pc2 = 1.0 + DirectSumHost(zf[0], zf[1], zf[2], cx, cy, cz, cw, kself);
        Real pa2 = PsiAnalytic(sp, zf[0], zf[1], zf[2]);
        Real far[3] = {40.0, 30.0, 50.0};
        Real pf3 = 1.0 + DirectSumHost(far[0], far[1], far[2], f.sx, f.sy, f.sz, f.sw,
                                       kself);
        Real pc3 = 1.0 + DirectSumHost(far[0], far[1], far[2], cx, cy, cz, cw, kself);
        Real pa3 = PsiAnalytic(sp, far[0], far[1], far[2]);
        Real rr = std::sqrt(far[0]*far[0] + far[1]*far[1] + far[2]*far[2]);
        std::printf("nr_pic_spheroid check: node (%.4f,%.4f,%.4f): Psi table=%.10f"
                    " fine-sum=%.10f (diff %.2e) analytic=%.10f\n", xc, yc, zc_, ptab,
                    pfine, pfine - ptab, pana);
        std::printf("nr_pic_spheroid check: (%.3f,%.3f,%.3f): fine=%.10f coarse=%.10f"
                    " (diff %.2e) analytic=%.10f\n", xf[0], xf[1], xf[2], pf1, pc1,
                    pc1 - pf1, pa1);
        std::printf("nr_pic_spheroid check: (%.3f,%.3f,%.3f): fine=%.10f coarse=%.10f"
                    " (diff %.2e) analytic=%.10f\n", zf[0], zf[1], zf[2], pf2, pc2,
                    pc2 - pf2, pa2);
        std::printf("nr_pic_spheroid check: (%.1f,%.1f,%.1f) r=%.3f: fine=%.10f"
                    " coarse=%.10f (diff %.2e) analytic=%.10f 1+M/(2r)=%.10f\n", far[0],
                    far[1], far[2], rr, pf3, pc3, pc3 - pf3, pa3, 1.0 + 0.5*f.M_adm/rr);
        std::fflush(stdout);
      }
    }
  }  // if (!restart)

  // seed the GR-pusher previous-step snapshots (fresh start AND restart; not in the
  // restart file). On restart Driver::Initialize refreshes both after the ghost exchange.
  if (pmbp->ppart->pusher == ParticlesPusher::gr_boris) {
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->adm_last, pmbp->padm->u_adm);
    Kokkos::deep_copy(DevExeSpace(), pmbp->ppart->z4c_last, pmbp->pz4c->u0);
  }

  if (global_variable::my_rank == 0 && !restart) {
    std::cout << "Prolate spheroid initialized (moving-puncture gauge seed alpha="
              << (precollapsed ? "Psi^-2" : "1") << ", beta=0)." << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SpheroidRefinementCondition
//! \brief delegate to the <z4c_amr> refinement machinery (Loehner etc.)
void SpheroidRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

//----------------------------------------------------------------------------------------
//! \fn void SpheroidHistory
//! \brief user history: extrema (with locations) of rho, alpha and |Kretschmann|, the
//! matter and lapse at the curvature maximum, particle ledger, and AMR state. Since the
//! history writer MPI-sums hdata over ranks, the globally reduced values are placed on
//! rank 0 only.
void SpheroidHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 20;
  // labels are truncated to 10 characters by the history writer
  const char *labels[20] = {"rho_max", "rho_x", "rho_y", "rho_z",
                            "alpha_min", "alp_x", "alp_y", "alp_z",
                            "Imax", "I_x", "I_y", "I_z",
                            "rho_at_I", "alp_at_I",
                            "N_alive", "M_alive", "N_exc_lps", "N_exc_oth",
                            "nmb_total", "max_level"};
  for (int n = 0; n < 20; ++n) {
    pdata->label[n] = labels[n];
    pdata->hdata[n] = 0.0;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int64_t ntot = static_cast<int64_t>(nmb)*nkji;

  // curvature invariants on the interior (uses current ADM + Tmunu)
  pmbp->pz4c->CalcKretschmann(pmbp);
  auto &u_k = pmbp->pz4c->u_kretsch;
  auto &z4c = pmbp->pz4c->z4c;
  bool have_matter = (pmbp->ptmunu != nullptr);
  Tmunu::Tmunu_vars tmunu;
  if (have_matter) {
    tmunu = pmbp->ptmunu->tmunu;
  }

  auto decode = [&](int64_t idx, int &m, int &k, int &j, int &i) {
    m = static_cast<int>(idx / nkji);
    int rem = static_cast<int>(idx - static_cast<int64_t>(m)*nkji);
    k = rem / nji; j = (rem - k*nji) / nx1; i = rem - k*nji - j*nx1;
    k += ks; j += js; i += is;
  };

  // pass 1: extrema
  Real rho_max = -1.0e300, alp_min = 1.0e300, kr_max = -1.0e300;
  Kokkos::parallel_reduce("sph_hist_ext", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
  KOKKOS_LAMBDA(const int64_t &idx, Real &mx_rho, Real &mn_alp, Real &mx_kr) {
    int m = static_cast<int>(idx / nkji);
    int rem = static_cast<int>(idx - static_cast<int64_t>(m)*nkji);
    int k = rem / nji; int j = (rem - k*nji) / nx1; int i = rem - k*nji - j*nx1;
    k += ks; j += js; i += is;
    if (have_matter) {
      mx_rho = fmax(mx_rho, tmunu.E(m,k,j,i));
    }
    mn_alp = fmin(mn_alp, z4c.alpha(m,k,j,i));
    mx_kr = fmax(mx_kr, fabs(u_k(m,0,k,j,i)));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alp_min), Kokkos::Max<Real>(kr_max));

  // pass 2: first cell attaining each extremum
  int64_t i_rho = ntot, i_alp = ntot, i_kr = ntot;
  Kokkos::parallel_reduce("sph_hist_loc", Kokkos::RangePolicy<>(DevExeSpace(), 0, ntot),
  KOKKOS_LAMBDA(const int64_t &idx, int64_t &l_rho, int64_t &l_alp, int64_t &l_kr) {
    int m = static_cast<int>(idx / nkji);
    int rem = static_cast<int>(idx - static_cast<int64_t>(m)*nkji);
    int k = rem / nji; int j = (rem - k*nji) / nx1; int i = rem - k*nji - j*nx1;
    k += ks; j += js; i += is;
    if (have_matter && tmunu.E(m,k,j,i) == rho_max) {
      l_rho = (idx < l_rho) ? idx : l_rho;
    }
    if (z4c.alpha(m,k,j,i) == alp_min) {
      l_alp = (idx < l_alp) ? idx : l_alp;
    }
    if (fabs(u_k(m,0,k,j,i)) == kr_max) {
      l_kr = (idx < l_kr) ? idx : l_kr;
    }
  }, Kokkos::Min<int64_t>(i_rho), Kokkos::Min<int64_t>(i_alp),
     Kokkos::Min<int64_t>(i_kr));

  auto &size = pmbp->pmb->mb_size;
  auto loc = [&](int64_t idx, Real *xyz) {
    xyz[0] = xyz[1] = xyz[2] = 0.0;
    if (idx < 0 || idx >= ntot) {
      return;
    }
    int m, k, j, i; decode(idx, m, k, j, i);
    xyz[0] = CellCenterX(i-is, nx1, size.h_view(m).x1min, size.h_view(m).x1max);
    xyz[1] = CellCenterX(j-js, nx2, size.h_view(m).x2min, size.h_view(m).x2max);
    xyz[2] = CellCenterX(k-ks, nx3, size.h_view(m).x3min, size.h_view(m).x3max);
  };
  // rho and alpha at the curvature maximum (single-cell read-back)
  Real at_kr[2] = {0.0, 0.0};
  if (i_kr < ntot) {
    int m, k, j, i; decode(i_kr, m, k, j, i);
    DvceArray1D<Real> tmp("sph_tmp", 2);
    Kokkos::parallel_for("sph_hist_pick", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
    KOKKOS_LAMBDA(const int &) {
      tmp(0) = have_matter ? tmunu.E(m,k,j,i) : 0.0;
      tmp(1) = z4c.alpha(m,k,j,i);
    });
    auto htmp = Kokkos::create_mirror_view(tmp);
    Kokkos::deep_copy(htmp, tmp);
    at_kr[0] = htmp(0); at_kr[1] = htmp(1);
  }

  // global reductions: payload = {value, x, y, z, (extras)}
  Real prho[4], palp[4], pkr[6];
  prho[0] = rho_max; loc(i_rho, prho + 1);
  palp[0] = alp_min; loc(i_alp, palp + 1);
  pkr[0] = kr_max;   loc(i_kr, pkr + 1); pkr[4] = at_kr[0]; pkr[5] = at_kr[1];
  GlobalPick(rho_max, prho, 4);
  GlobalPick(-alp_min, palp, 4);
  GlobalPick(kr_max, pkr, 6);

  // particle ledger: alive count and mass on this rank
  Real m_alive = 0.0;
  auto &ppart = pmbp->ppart;
  int np = ppart->nprtcl_thispack;
  if (np > 0) {
    auto &pr = ppart->prtcl_rdata;
    Kokkos::parallel_reduce("sph_hist_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, np),
    KOKKOS_LAMBDA(const int &p, Real &msum) {
      msum += pr(IPM,p);
    }, Kokkos::Sum<Real>(m_alive));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &m_alive, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  int max_level = 0;
  for (int m = 0; m < pm->nmb_total; ++m) {
    max_level = std::max(max_level, pm->lloc_eachmb[m].level - pm->root_level);
  }

  if (global_variable::my_rank == 0) {
    for (int n = 0; n < 4; ++n) {
      pdata->hdata[n] = prho[n];
      pdata->hdata[4+n] = palp[n];
      pdata->hdata[8+n] = pkr[n];
    }
    pdata->hdata[12] = pkr[4];
    pdata->hdata[13] = pkr[5];
    pdata->hdata[14] = static_cast<Real>(pm->nprtcl_total);
    pdata->hdata[15] = m_alive;
    pdata->hdata[16] = static_cast<Real>(pm->nprtcl_destroyed_cum[PrtclDeathLapse]);
    pdata->hdata[17] = static_cast<Real>(pm->nprtcl_destroyed_cum[PrtclDeathExit]
                                         + pm->nprtcl_destroyed_cum[PrtclDeathSphere]
                                         + pm->nprtcl_destroyed_cum[PrtclDeathHorizon]);
    pdata->hdata[18] = static_cast<Real>(pm->nmb_total);
    pdata->hdata[19] = static_cast<Real>(max_level);
  }
}
