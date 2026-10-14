#ifndef DRIVER_CODE_EFFICIENCY_HPP_
#define DRIVER_CODE_EFFICIENCY_HPP_
#include <cstdint>
#include <map>
#include <string>
#include "athena.hpp"
class Mesh;
class ParameterInput;
namespace efficiency {
struct State {
  bool enabled=false, profile=false, mesh_only=false, measuring=false;
  double warmup=0.5, start_time=0.0, elapsed=0.0;
  int start_cycle=0;
  std::uint64_t initial_count=0, initial_sum=0, initial_square=0;
  std::uint64_t geometry[4]={0,0,0,0};
  std::uint64_t sent_particles=0, source_images=0;
  Kokkos::Timer timer;
  std::map<std::string,double> phases;
  DvceArray2D<unsigned long long> device_cycles;
  DvceArray1D<Real> source_mass, source_u;
};
State &Get();
void Configure(ParameterInput *pin);
void Audit(Mesh *mesh, bool initial);
void StartWindow(Mesh *mesh);
void Finish(Mesh *mesh);
class Phase {
 public:
  explicit Phase(const char *name);
  ~Phase();
  void Stop();
 private:
  std::string name_;
  bool active_;
  Kokkos::Timer timer_;
};
} // namespace efficiency
#endif
