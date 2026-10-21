#ifndef PARTICLES_GI_DIAGNOSTICS_HPP_
#define PARTICLES_GI_DIAGNOSTICS_HPP_

#include <fstream>
#include <string>
#include "athena.hpp"
#include "tasklist/task_list.hpp"
class Mesh;
class MeshBlockPack;
class ParameterInput;
class Outputs;

namespace particles {
// Passive, opt-in Session 011 accounting. All scratch storage is independent of
// integrator/particle state; no field or particle value is modified by these hooks.
class GiDiagnostics {
 public:
  GiDiagnostics(MeshBlockPack *, ParameterInput *);
  void BeforeStep();
  void AfterPush();
  void Sample(bool force = false);
  void Milestones(Outputs *);
 private:
  MeshBlockPack *pack;
  Mesh *mesh;
  ParameterInput *pin;
  int nh, lm, lmp;
  int old_valid=0, new_valid=0, old_ever=0, new_ever=0;
  bool common=false, first_pair=false, close_pair=false;
  Real next_sample=0, previous_separation=-1;
  DvceArray2D<Real> before, old_par, new_par, old_coef, new_coef;
  DvceArray3D<Real> shell_ledger;
  std::ofstream events, profiles, grid, status, transport, milestones;
  void Stage(DvceArray2D<Real> &, DvceArray2D<Real> &, int &, int &, int);
};
}  // namespace particles
#endif
