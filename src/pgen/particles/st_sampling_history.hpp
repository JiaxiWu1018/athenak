#ifndef PGEN_PARTICLES_ST_SAMPLING_HISTORY_HPP_
#define PGEN_PARTICLES_ST_SAMPLING_HISTORY_HPP_

#include "globals.hpp"
#include "outputs/outputs.hpp"
#include "z4c/tmunu.hpp"

// Coordinate-volume diagnostics within fixed origin-centred regions. Native global
// norms use their existing proper-volume convention and are written alongside these.
void STSamplingHistory(HistoryData *data, Mesh *mesh) {
  auto *pack = mesh->pmb_pack;
  auto &idx = mesh->mb_indcs;
  auto &size = pack->pmb->mb_size;
  auto con = pack->pz4c->u_con;
  auto adm = pack->padm->adm;
  auto mat = pack->ptmunu->tmunu;
  data->nhist = 17;
  const char *labels[17] = {"V10", "H2r10", "M2r10", "E1r10", "E2r10", "J2r10",
                           "V16", "H2r16", "M2r16", "E1r16", "E2r16", "J2r16",
                           "Vcore", "acore", "Vouter", "aouter", "dt_done"};
  for (int n=0; n<17; ++n) {
    data->label[n] = labels[n];
  }
  const int is=idx.is, js=idx.js, ks=idx.ks;
  const int n1=idx.nx1, n2=idx.nx2, n3=idx.nx3;
  const int nc=n1*n2*n3;
  array_sum::GlobalSum sum;
  Kokkos::parallel_reduce("STCoreHistory",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, pack->nmb_thispack*nc),
      KOKKOS_LAMBDA(int p, array_sum::GlobalSum &acc) {
    int m=p/nc, l=p-m*nc;
    int kk=l/(n1*n2), jj=(l-kk*n1*n2)/n1, ii=l-kk*n1*n2-jj*n1;
    int i=ii+is, j=jj+js, k=kk+ks;
    auto s=size.d_view(m);
    Real x=CellCenterX(ii,n1,s.x1min,s.x1max);
    Real y=CellCenterX(jj,n2,s.x2min,s.x2max);
    Real z=CellCenterX(kk,n3,s.x3min,s.x3max);
    Real r2=x*x+y*y+z*z, dv=s.dx1*s.dx2*s.dx3;
    array_sum::GlobalSum values;
    for (int n=0; n<NHISTORY_VARIABLES; ++n) {
      values.the_array[n]=0.0;
    }
    Real e=mat.E(m,k,j,i), j2=0.0;
    for (int a=0; a<3; ++a) {
      j2+=SQR(mat.S_d(m,a,k,j,i));
    }
    for (int b=0; b<2; ++b) {
      if (r2 < (b==0 ? 100.0 : 256.0)) {
        int o=6*b;
        values.the_array[o]=dv;
        values.the_array[o+1]=dv*SQR(con(m,z4c::Z4c::I_CON_H,k,j,i));
        values.the_array[o+2]=dv*con(m,z4c::Z4c::I_CON_M,k,j,i);
        values.the_array[o+3]=dv*e;
        values.the_array[o+4]=dv*e*e;
        values.the_array[o+5]=dv*j2;
      }
    }
    if (r2 < 0.75*0.75) {
      values.the_array[12]=dv;
      values.the_array[13]=dv*adm.alpha(m,k,j,i);
    }
    if (r2 >= 96.0*96.0 && r2 < 120.0*120.0) {
      values.the_array[14]=dv;
      values.the_array[15]=dv*adm.alpha(m,k,j,i);
    }
    acc+=values;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum));
  for (int n=0; n<16; ++n) {
    data->hdata[n]=sum.the_array[n];
  }
  // History's standard dt can be the NEXT timestep at final output. Retain the
  // completed step explicitly; user history is summed over ranks, so emit it once.
  data->hdata[16]=(global_variable::my_rank == 0 ? mesh->dt_last_completed : 0.0);
}

#endif  // PGEN_PARTICLES_ST_SAMPLING_HISTORY_HPP_
