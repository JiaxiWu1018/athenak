#ifndef PGEN_PARTICLES_ST_SAMPLE_INPUT_HPP_
#define PGEN_PARTICLES_ST_SAMPLE_INPUT_HPP_

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// Ordered, double-precision initial samples produced by the source-matched generator.
// Loading is rank-independent. Each rank retains particles through normal ownership.
class STSampleInput {
 public:
  STSampleInput(const std::string &path, int n, int seed, Real mass, Real yc,
                Real dx, const std::string &sampling, const std::string &radial,
                const std::string &velocity) : sum_(0.0L) {
    if (path.empty()) {
      return;
    }
    input_.open(path, std::ios::binary);
    std::string line, magic, s, r, v;
    int nn = 0, ss = 0;
    double mm = 0, yy = 0, dd = 0;
    if (!std::getline(input_, line)) {
      Fail("cannot read sample header");
    }
    std::istringstream head(line);
    head >> magic >> nn >> ss >> mm >> yy >> dd >> s >> r >> v;
    if (!head || magic != "STSAMP05" || nn != n || ss != seed || s != sampling ||
        r != radial || v != velocity || std::abs(mm-mass) > 1.0e-14 ||
        std::abs(yy-yc) > 1.0e-14 || std::abs(dd-dx) > 1.0e-14) {
      Fail("sample metadata differs from input");
    }
  }

  bool Enabled() const {return input_.is_open();}

  std::array<double, 8> Read(int tag) {
    std::array<double, 8> row;
    input_.read(reinterpret_cast<char *>(row.data()), sizeof(row));
    if (!input_ || row[0] != tag || !(row[7] > 0.0)) {
      Fail("invalid tag, mass, or truncated sample");
    }
    for (double value : row) {
      if (!std::isfinite(value)) {
        Fail("non-finite sample");
      }
    }
    sum_ += row[7];
    return row;
  }

  void Finish(Real expected_mass) {
    if (!Enabled()) {
      return;
    }
    if (input_.peek() != std::char_traits<char>::eof() ||
        std::abs(sum_/expected_mass-1.0L) > 1.0e-11L) {
      Fail("sample length or total rest mass differs");
    }
  }

 private:
  [[noreturn]] void Fail(const char *message) {
    std::cerr << "### FATAL ERROR ST sample: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::ifstream input_;
  long double sum_;
};

#endif  // PGEN_PARTICLES_ST_SAMPLE_INPUT_HPP_
